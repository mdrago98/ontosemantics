"""
Neuro-Symbolic LangGraph: Evidence + LLM Reasoning
- Evidence:
  * SPARQL (exact-by-URI, subproperty chains, two-hop, label fallbacks)
  * Ontology (domain/range/disjoint; lightweight fallback when silent)
- LLM Reasoner:
  * Synthesizes a short justification from evidence (Ollama optional)
- Control:
  * Planner accepts/rejects/swap/query_more using score+reason+LLM if present
"""

import json
import re
import logging
from typing import Dict, Tuple, Optional

import requests
from langgraph.graph import StateGraph, END
from langchain.tools import tool

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("neuro_symbolic_verifier")

# ---------- Optional LLM (Ollama via LangChain) ----------
try:
    from langchain_ollama import ChatOllama
    _HAS_OLLAMA = True
except Exception:
    _HAS_OLLAMA = False

# ---------- Config ----------
DEFAULT_SPARQL_ENDPOINT = "https://sparql.hegroup.org/sparql/"
# DEFAULT_SPARQL_ENDPOINT = "https://ubergraph.apps.renci.org/sparql"
VALIDATION_THRESHOLD = 0.70
MAX_PLANNER_ITERS_PER_NEED = 2
LLM_MODEL = "gemma3:1b"  # set your local model here

PREFIXES = """\
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
"""

_PFX = {
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "oboInOwl": "http://www.geneontology.org/formats/oboInOwl#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}

PREDICATE_URI = {
    "treats": "http://purl.obolibrary.org/obo/RO_0002606",
    "causes": "http://purl.obolibrary.org/obo/RO_0002410",
    "association": "http://purl.obolibrary.org/obo/RO_0002327",
    "bind": "http://purl.obolibrary.org/obo/RO_0002434",
    "comparison": "http://purl.obolibrary.org/obo/RO_0002502",
    "conversion": "http://purl.obolibrary.org/obo/RO_0002449",
    "cotreatment": "http://purl.obolibrary.org/obo/RO_0002460",
    "drug_interaction": "http://purl.obolibrary.org/obo/RO_0002436",
    "negative_correlation": "http://purl.obolibrary.org/obo/RO_0002608",
    "positive_correlation": "http://purl.obolibrary.org/obo/RO_0002607",
}

from pyoxigraph import *

store = Store()
store.load(path='../../notebooks/data/mondo.owl', format=RdfFormat.RDF_XML)
store.load(path='../../notebooks/data/go.owl', format=RdfFormat.RDF_XML)
store.load(path='../../notebooks/data/cl.owl', format=RdfFormat.RDF_XML)
store.load(path='../../notebooks/data/chebi.owl', format=RdfFormat.RDF_XML)
store.load(path='../../notebooks/data/ro.owl', format=RdfFormat.RDF_XML)


# n = store.query("SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }",
#                     use_default_graph_as_union=True)

HEGROUP_ENDPOINT = "https://sparql.hegroup.org/sparql"

PREDICATE_SCHEMA_FALLBACK = {
    # RO_0002606 = treats: drug/chemical → disease/phenotype
    "http://purl.obolibrary.org/obo/RO_0002606": {
        "domain_prefixes": ("CHEBI_", "DRUGBANK_", "PR_", "NCIT_C") ,  # chemical/protein-ish
        "range_prefixes": ("MONDO_", "DOID_", "HP_", "EFO_"),          # disease/phenotype-ish
    },
}


# ---------- SPARQL executor with cache ----------
_SPARQL_CACHE: Dict[int, dict] = {}

def _hashable(q: str) -> int:
    return hash(q)

import json
from pyoxigraph import Store, QueryResultsFormat

# _SPARQL_CACHE: dict[int, dict] = {}

def sparql_post(query: str) -> dict:
    """Run SPARQL query against local pyoxigraph store and normalize results."""
    key = hash(query)
    if key in _SPARQL_CACHE:
        out = dict(_SPARQL_CACHE[key])
        out["from_cache"] = True
        return out

    if store is None:
        return {"success": False, "error": "Store not initialized."}

    try:
        result = store.query(query, prefixes=_PFX, use_default_graph_as_union=True)

        # ASK queries → bool
        if isinstance(result, bool):
            res = {
                "success": True,
                "is_ask": True,
                "ask_result": result,
                "results": []
            }

        # SELECT/CONSTRUCT/DESCRIBE → serialize to JSON
        else:
            raw_json = result.serialize(format=QueryResultsFormat.JSON)
            parsed = json.loads(raw_json)

            if "boolean" in parsed:
                res = {
                    "success": True,
                    "is_ask": True,
                    "ask_result": bool(parsed["boolean"]),
                    "results": []
                }
            else:
                res = {
                    "success": True,
                    "is_ask": False,
                    "results": parsed.get("results", {}).get("bindings", [])
                }

        _SPARQL_CACHE[key] = res
        return res

    except Exception as e:
        return {"success": False, "error": str(e)}

print(sparql_post("""
ASK { <http://purl.obolibrary.org/obo/CHEBI_9947> <http://purl.obolibrary.org/obo/RO_0002606> <http://purl.obolibrary.org/obo/MONDO_0005148> }
"""))

@tool("sparql_executor", return_direct=True)
def sparql_executor(query: str) -> dict:
    """
    executes sparql
    :param query:
    :return:
    """
    return sparql_post(query)

# ---------- Helpers ----------
def sparql_escape(text: str) -> str:
    return re.sub(r'["\'\\\n\r\t]+', ' ', text or "").strip()

# ---------- Entity Resolver ----------
@tool("entity_resolver", return_direct=True)
def entity_resolver(term: str) -> dict:
    """
    Resolve biomedical entity to ontology IRI using HEGroup SPARQL.
    Includes labels and synonyms for more robust text matching.
    """
    t = sparql_escape(term)
    query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?entity ?label ?syn ?alt WHERE {{
  ?entity rdfs:label ?label .
  OPTIONAL {{ ?entity oboInOwl:hasExactSynonym ?syn }}
  OPTIONAL {{ ?entity skos:altLabel ?alt }}
  FILTER(
    CONTAINS(LCASE(STR(?label)), LCASE("{t}")) ||
    (BOUND(?syn) && CONTAINS(LCASE(STR(?syn)), LCASE("{t}"))) ||
    (BOUND(?alt) && CONTAINS(LCASE(STR(?alt)), LCASE("{t}")))
  )
}}
LIMIT 40
"""
    return sparql_post(query)

def _first_uri(res: dict) -> Optional[str]:
    if not res or res.get("is_ask"):
        return None
    for b in res.get("results", []):
        uri = b.get("entity", {}).get("value")
        if uri:
            return uri
    return None


class OntologyVerifier:
    def __init__(self, executor=None):
        self.executor = executor

    def _domain_range_via_sparql(self, predicate_uri: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.executor or not predicate_uri:
            return None, None
        q = f"""
{PREFIXES}
SELECT ?domain ?range WHERE {{
  OPTIONAL {{ <{predicate_uri}> rdfs:domain ?domain . }}
  OPTIONAL {{ <{predicate_uri}> rdfs:range ?range . }}
}}
"""
        res = self.executor(q)
        if res.get("success") and not res.get("is_ask"):
            bindings = res.get("results", [])
            if bindings:
                return (
                    bindings[0].get("domain", {}).get("value"),
                    bindings[0].get("range", {}).get("value"),
                )
        return None, None

    @staticmethod
    def _iri_has_prefix(iri: str, prefixes) -> bool:
        if not iri: return False
        return any(p in iri for p in prefixes)

    def verify(self, subject_uri: str, predicate_uri: str, object_uri: str) -> dict:
        domain, rng = self._domain_range_via_sparql(predicate_uri)

        subj_ok = obj_ok = None
        # If explicit domain/range IRIs are available, we cannot reliably ASK instance-of here
        # without a reasoner. As a pragmatic fallback, accept prefix compatibility.
        if domain:
            subj_ok = subject_uri == domain or self._iri_has_prefix(subject_uri, (domain,))
        if rng:
            obj_ok = object_uri == rng or self._iri_has_prefix(object_uri, (rng,))

        # If ontology is silent, use coarse schema fallback
        if domain is None and rng is None:
            fb = PREDICATE_SCHEMA_FALLBACK.get(predicate_uri)
            if fb:
                subj_ok = self._iri_has_prefix(subject_uri, fb["domain_prefixes"])
                obj_ok  = self._iri_has_prefix(object_uri,  fb["range_prefixes"])

        return {
            "domain": domain, "range": rng,
            "subject_in_domain": subj_ok, "object_in_range": obj_ok,
            "direction_ok": bool(subj_ok and obj_ok),
            "swapped": False, "disjoint": False,
            "reasoner_used": False
        }

global_verifier = OntologyVerifier(sparql_post)

@tool("ontology_verification", return_direct=True)
def ontology_verification(subject: str, predicate_uri: str, obj: str) -> dict:
    """
    Verifies the terms from ontologies
    :param subject:
    :param predicate_uri:
    :param obj:
    :return:
    """
    return global_verifier.verify(subject, predicate_uri, obj)

# ---------- SPARQL templates (URI-first with label fallbacks) ----------
def q_exact_uri(pred_uri, subj_uri, obj_uri):
    return f"""{PREFIXES} ASK {{ <{subj_uri}> <{pred_uri}> <{obj_uri}> . }}"""

def q_subprop_chain_uri(pred_uri, subj_uri, obj_uri):
    return f"""{PREFIXES} ASK {{ <{subj_uri}> ?p <{obj_uri}> . ?p rdfs:subPropertyOf* <{pred_uri}> . }}"""

def q_twohop_uri(subj_uri, obj_uri, limit=200):
    return f"""{PREFIXES} SELECT ?mid ?p1 ?p2 WHERE {{ <{subj_uri}> ?p1 ?mid . ?mid ?p2 <{obj_uri}> . }} LIMIT {limit}"""

def q_flexible(subj, obj, limit=120):
    return f"""{PREFIXES}
SELECT ?s ?p ?o WHERE {{
  ?s rdfs:label ?sl . FILTER(CONTAINS(LCASE(STR(?sl)), LCASE("{sparql_escape(subj)}")))
  ?o rdfs:label ?ol . FILTER(CONTAINS(LCASE(STR(?ol)), LCASE("{sparql_escape(obj)}")))
  ?s ?p ?o .
}} LIMIT {limit}"""

def q_twohop_label(subj, obj, limit=200):
    return f"""{PREFIXES}
SELECT ?mid ?p1 ?p2 WHERE {{
  ?S rdfs:label ?sl . FILTER(CONTAINS(LCASE(STR(?sl)), LCASE("{sparql_escape(subj)}")))
  ?O rdfs:label ?ol . FILTER(CONTAINS(LCASE(STR(?ol)), LCASE("{sparql_escape(obj)}")))
  ?S ?p1 ?mid .
  ?mid ?p2 ?O .
}} LIMIT {limit}"""

def generate_template_pack(subj, pred, obj, subj_uri=None, obj_uri=None) -> Dict[str, str]:
    pred_uri = PREDICATE_URI.get(pred.lower())
    pack: Dict[str, str] = {}
    # Prefer URI-based evidence if we resolved URIs
    if subj_uri and obj_uri:
        if pred_uri:
            pack["exact_uri"] = q_exact_uri(pred_uri, subj_uri, obj_uri)
            pack["subprop_chain_uri"] = q_subprop_chain_uri(pred_uri, subj_uri, obj_uri)
        pack["twohop_uri"] = q_twohop_uri(subj_uri, obj_uri)
    # Always include label fallbacks to harvest some signal
    pack["flexible"] = q_flexible(subj, obj)
    pack["twohop_label"] = q_twohop_label(subj, obj)
    return pack

# ---------- LLM Reasoner ----------
class LLMReasoner:
    def __init__(self, model: str = LLM_MODEL):
        self.available = _HAS_OLLAMA
        if self.available:
            try:
                self.llm = ChatOllama(model=model, temperature=0)
                log.info(f"LLMReasoner using Ollama model: {model}")
            except Exception as e:
                log.warning(f"LLM init failed ({e}); falling back to rule-based explanations.")
                self.available = False

    def explain(self, relation: str, evidence: dict) -> str:
        # Minimal fallback if no LLM
        if not self.available:
            if evidence.get("ontology", {}).get("direction_ok"):
                return "Ontology type-check passed; plausible subject→object typing for predicate."
            if "ASK confirmed" in str(evidence.get("sparql_summary", "")):
                return "Exact ASK confirmed in knowledge base."
            return "Insufficient evidence (no ontology typing and no strong SPARQL hits)."

        prompt = (
            "You are a biomedical reasoning agent. "
            "The candidate relation may be novel (not in existing knowledge bases). "
            "Your task: assess if it is ontologically valid (drug→disease, gene→phenotype, etc.), "
            "even if ASK/SPARQL evidence is missing.\n\n"
            f"Relation: {relation}\n\n"
            f"SPARQL summary: {evidence.get('sparql_summary')}\n"
            f"Ontology verdict: {json.dumps(evidence.get('ontology'), indent=2)}\n\n"
            "If SPARQL found no evidence but ontology typing is consistent, state that the relation is novel but plausible. "
            "If ontology typing contradicts (wrong domain/range), explain why it is invalid."
        )
        try:
            resp = self.llm.invoke(prompt)
            return (getattr(resp, "content", str(resp)) or "").strip()
        except Exception as e:
            return f"LLM failed to produce a justification: {e}"

reasoner = LLMReasoner()

# ---------- Heuristic verifier aggregator ----------
def agg_verifier(result: dict) -> Dict:
    if not result or not result.get("success"):
        err = result.get("error", "unknown error") if isinstance(result, dict) else "unknown error"
        return {"score": 0.0, "reason": f"execution failed: {err}"}
    if result.get("is_ask"):
        return {"score": 0.95 if result.get("ask_result") else 0.0,
                "reason": "ASK confirmed" if result.get("ask_result") else "ASK false"}
    hits = len(result.get("results", []))
    if hits > 0:
        return {"score": min(0.85, 0.4 + hits/20.0), "reason": f"{hits} hits"}
    return {"score": 0.0, "reason": "no evidence"}

# ---------- Graph Nodes ----------
def node_supervisor_plan(state: Dict) -> Dict:
    rels = state.get("extracted_relations", [])
    needs = [f"{r['subject']}:{r['predicate']}:{r['object']}" for r in rels]
    state["queue"] = needs
    state["done"] = {n: False for n in needs}
    state["attempts"] = {n: 0 for n in needs}
    state["validation_status"] = {}
    state["ontology_status"] = {}
    state["evidence"] = {}
    state["reasoning"] = {}
    return state

def node_supervisor_select(state: Dict) -> Dict:
    while state["queue"] and state["done"][state["queue"][0]]:
        state["queue"].pop(0)
    state["current_need"] = state["queue"][0] if state["queue"] else None
    return state

def node_resolve_entities(state: Dict) -> Dict:
    if not state.get("current_need"): return state
    subj, pred, obj = state["current_need"].split(":")
    em = state.setdefault("entity_mappings", {})
    if subj not in em:
        em[subj] = entity_resolver.invoke({"term": subj})
        em[subj]["uri"] = _first_uri(em[subj])
    if obj not in em:
        em[obj] = entity_resolver.invoke({"term": obj})
        em[obj]["uri"] = _first_uri(em[obj])
    log.info("Resolved %s → %s, %s → %s", subj, em.get(subj, {}).get("uri"), obj, em.get(obj, {}).get("uri"))
    return state

def node_generator(state: Dict) -> Dict:
    if not state.get("current_need"): return state
    subj, pred, obj = state["current_need"].split(":")
    em = state.get("entity_mappings", {})
    subj_uri = em.get(subj, {}).get("uri")
    obj_uri  = em.get(obj, {}).get("uri")
    state["current_queries"] = generate_template_pack(subj, pred, obj, subj_uri, obj_uri)
    return state

def node_executor(state: Dict) -> Dict:
    if not state.get("current_queries"): return state
    results = {}
    for name, q in state["current_queries"].items():
        results[name] = sparql_executor.invoke({"query": q})
    state["last_results"] = results
    return state

def node_verifier(state: Dict) -> Dict:
    need = state.get("current_need")
    results = state.get("last_results", {})
    scores, reasons = [], []
    # Strong signal: any exact ASK true?
    for strat, res in results.items():
        v = agg_verifier(res)
        scores.append(v["score"]); reasons.append(f"{strat}: {v['reason']}")
    state["validation_status"][need] = {
        "score": max(scores) if scores else 0.0,
        "reason": " | ".join(reasons) if reasons else "no queries"
    }
    return state

def node_ontology(state: Dict) -> Dict:
    need = state.get("current_need")
    if not need: return state
    subj_txt, pred, obj_txt = need.split(":")
    pred_uri = PREDICATE_URI.get(pred.lower(), "")
    subj_uri = state.get("entity_mappings", {}).get(subj_txt, {}).get("uri")
    obj_uri  = state.get("entity_mappings", {}).get(obj_txt, {}).get("uri")

    if pred_uri and subj_uri and obj_uri:
        verdict = ontology_verification.invoke({
            "subject": subj_uri, "predicate_uri": pred_uri, "obj": obj_uri
        })
        state.setdefault("ontology_status", {})[need] = verdict
        ctx = state["validation_status"].setdefault(need, {"score": 0.0, "reason": ""})
        if verdict.get("direction_ok"):
            ctx["score"] = max(ctx["score"], 0.9)
            ctx["reason"] += " | ontology consistent"
        else:
            ctx["reason"] += " | ontology no evidence"
        state["validation_status"][need] = ctx
    else:
        state.setdefault("ontology_status", {})[need] = {
            "direction_ok": False, "reason": "missing URIs or predicate URI"
        }
    return state

def node_collect_evidence(state: Dict) -> Dict:
    need = state.get("current_need")
    if not need: return state
    ev = {
        "sparql_summary": state["validation_status"].get(need, {}).get("reason", ""),
        "sparql_score": state["validation_status"].get(need, {}).get("score", 0.0),
        "ontology": state.get("ontology_status", {}).get(need, {}),
        "raw_results": state.get("last_results", {}),
    }
    state["evidence"][need] = ev
    return state

# ---------- LLM Reasoning node ----------
class LLMController:
    def __init__(self, model: str = LLM_MODEL):
        self.available = _HAS_OLLAMA
        if self.available:
            try:
                self.llm = ChatOllama(model=model, temperature=0)
                log.info(f"Planner using Ollama model: {model}")
            except Exception as e:
                log.warning(f"Planner LLM init failed ({e}); using rule planner.")
                self.available = False

    def plan(self, relation: str, score: float, reason: str, attempts: int, explanation: str) -> str:
        # Rule fallback is robust and fast
        if score >= VALIDATION_THRESHOLD: return "accept"
        if "contradiction" in reason or "disjoint" in reason: return "reject"
        if "swap" in reason: return "swap"
        if attempts < MAX_PLANNER_ITERS_PER_NEED: return "query_more"
        return "reject"

planner = LLMController()

def node_llm_reasoner(state: Dict) -> Dict:
    need = state.get("current_need")
    if not need: return state
    explanation = reasoner.explain(need, state["evidence"].get(need, {}))
    state["reasoning"][need] = explanation
    log.info(f"Reasoning for {need}: {explanation}")
    return state

def node_planner(state: Dict) -> Dict:
    need = state.get("current_need")
    if not need: return state
    attempts = state["attempts"].get(need, 0)
    val = state["validation_status"].get(need, {"score": 0.0, "reason": ""})
    explanation = state.get("reasoning", {}).get(need, "")
    action = planner.plan(need, val.get("score", 0.0), val.get("reason", ""), attempts, explanation)
    state["planner_action"] = action
    state["attempts"][need] = attempts + (1 if action == "query_more" else 0)
    log.info(f"Planner action for {need}: {action}")
    return state

def route_after_planner(state: Dict) -> str:
    act = state.get("planner_action", "reject")
    if act == "accept": return "accept"
    if act == "reject": return "reject"
    if act == "swap": return "swap"
    if act == "query_more": return "refine"
    return "reject"

def node_refine(state: Dict) -> Dict:
    """Simple refinement: re-run generator with only label fallbacks (already included).
    In a more advanced version you'd add synonym expansions or alter thresholds."""
    # No-op here: generator already includes flexible/twohop label queries.
    return state

def node_accept(state: Dict) -> Dict:
    need = state.get("current_need")
    if need: state["done"][need] = True
    return state

def node_reject(state: Dict) -> Dict:
    need = state.get("current_need")
    if need: state["done"][need] = True
    return state

def node_swap(state: Dict) -> Dict:
    need = state.get("current_need")
    if not need: return state
    s, p, o = need.split(":")
    swapped = f"{o}:{p}:{s}"
    state["done"][need] = True
    if swapped not in state["done"]:
        state["queue"].append(swapped)
        state["done"][swapped] = False
        state["attempts"][swapped] = 0
    return state

def route_after_decide(state: Dict) -> str:
    if all(state["done"].values()): return "end"
    return "continue"

# ---------- Graph ----------
graph = StateGraph(dict)
graph.add_node("plan", node_supervisor_plan)
graph.add_node("select", node_supervisor_select)
graph.add_node("resolve", node_resolve_entities)
graph.add_node("generate", node_generator)
graph.add_node("exec", node_executor)
graph.add_node("verify", node_verifier)
graph.add_node("ontology", node_ontology)
graph.add_node("collect_evidence", node_collect_evidence)
graph.add_node("llm_reasoner", node_llm_reasoner)
graph.add_node("planner", node_planner)
graph.add_node("refine", node_refine)
graph.add_node("accept", node_accept)
graph.add_node("reject", node_reject)
graph.add_node("swap", node_swap)

graph.add_edge("plan", "select")
graph.add_edge("select", "resolve")
graph.add_edge("resolve", "generate")
graph.add_edge("generate", "exec")
graph.add_edge("exec", "verify")
graph.add_edge("verify", "ontology")
graph.add_edge("ontology", "collect_evidence")
graph.add_edge("collect_evidence", "llm_reasoner")
graph.add_edge("llm_reasoner", "planner")

graph.add_conditional_edges("planner", route_after_planner,
                            {"accept": "accept", "reject": "reject", "swap": "swap", "refine": "refine"})
graph.add_edge("refine", "exec")
graph.add_conditional_edges("accept", route_after_decide, {"end": END, "continue": "select"})
graph.add_conditional_edges("reject", route_after_decide, {"end": END, "continue": "select"})
graph.add_conditional_edges("swap", route_after_decide, {"end": END, "continue": "select"})

graph.set_entry_point("plan")
app = graph.compile()

# ---------- Runner ----------
class HierarchicalVerifier:
    def run_case_stream(self, case: Dict):
        init = {
            "original_text": case["text"],
            "extracted_entities": case["entities"],
            "extracted_relations": case["relations"],
        }
        final = app.invoke(init, config={"recursion_limit": 200})
        # Attach compact outputs
        final_out = {
            "validation_status": final.get("validation_status", {}),
            "ontology_status": final.get("ontology_status", {}),
            "entity_mappings": {k: {"uri": v.get("uri")} for k, v in final.get("entity_mappings", {}).items()},
            "evidence": final.get("evidence", {}),
            "reasoning": final.get("reasoning", {}),
            "attempts": final.get("attempts", {}),
        }
        return final_out

# ---------- Example ----------
if __name__ == "__main__":
    hv = HierarchicalVerifier()
    cases = [
        {
            "text": "Metformin treats type 2 diabetes.",
            "entities": [{"text": "Metformin"}, {"text": "type 2 diabetes"}],
            "relations": [{"subject":"Metformin","predicate":"treats","object":"type 2 diabetes"}],
        },
        {
            "text": "Type 2 diabetes treats Metformin (nonsense).",
            "entities": [{"text": "type 2 diabetes"}, {"text": "Metformin"}],
            "relations": [{"subject":"type 2 diabetes","predicate":"treats","object":"Metformin"}],
        },
    ]
    for case in cases:
        out = hv.run_case_stream(case)
        print("\n=== CASE ===")
        print(case["text"])
        print("\nValidation:", json.dumps(out["validation_status"], indent=2))
        print("\nOntology:", json.dumps(out["ontology_status"], indent=2))
        print("\nReasoning:", json.dumps(out["reasoning"], indent=2))
        print("\nEntities:", json.dumps(out["entity_mappings"], indent=2))
