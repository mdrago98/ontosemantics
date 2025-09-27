"""
Neuro-Symbolic LangGraph: LLM Planner + SPARQL + Ontology + Inference
- Agents/Tools:
  * Template Bank + JSON Plan Compiler (SPARQL generation)
  * SPARQL Executor (cached)
  * Entity Resolver (to URIs)
  * Ontology Verifier (domain/range/disjoint via SPARQL)
- Flow per relation:
  plan -> select -> resolve -> generate -> exec -> verify -> infer -> ontology -> planner
  -> [ accept | reject | swap | query_more ]
  If query_more -> refine -> exec -> verify -> infer -> ontology -> planner ...
  Inferred relations are queued like originals.
"""

import json, re, requests, logging
from typing import Dict, Tuple, List, Optional
from langchain.tools import tool
from langgraph.graph import StateGraph, END

# Optional LLM (Ollama via LangChain)
try:
    from langchain_ollama import ChatOllama
    _HAS_OLLAMA = True
except Exception:
    _HAS_OLLAMA = False

# ---------------- Logging ----------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("neuro_symbolic_planner")

# ---------------- Config ----------------
DEFAULT_SPARQL_ENDPOINT = "https://sparql.hegroup.org/sparql/"
VALIDATION_THRESHOLD = 0.70
MAX_PLANNER_ITERS_PER_NEED = 3
LLM_MODEL = "gemma3:1b"

# ---------------- SPARQL Cache + Executor ----------------
_SPARQL_CACHE: Dict[int, dict] = {}

# ---------------- SPARQL Generator (Agent) ----------------
class SPARQLGenerator:
    predicate_map = {
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

    def generate_all(self, subj: str, pred: str, obj: str) -> Dict[str, str]:
        subj, obj = sparql_escape(subj), sparql_escape(obj)
        pred_uri = self.predicate_map.get(pred.lower())
        queries = {}

        if pred_uri:
            queries["standard"] = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
ASK {{
  ?s rdfs:label ?sl . FILTER(CONTAINS(LCASE(str(?sl)), LCASE("{subj}")))
  ?o rdfs:label ?ol . FILTER(CONTAINS(LCASE(str(?ol)), LCASE("{obj}")))
  ?s <{pred_uri}> ?o .
}}"""

        queries["flexible"] = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?s ?p ?o WHERE {{
  ?s rdfs:label ?sl . FILTER(CONTAINS(LCASE(str(?sl)), LCASE("{subj}")))
  ?o rdfs:label ?ol . FILTER(CONTAINS(LCASE(str(?ol)), LCASE("{obj}")))
  ?s ?p ?o .
}} LIMIT 50
"""

        queries["indirect"] = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?mid ?p1 ?p2 ?S ?O WHERE {{
  ?S rdfs:label ?sl . FILTER(CONTAINS(LCASE(str(?sl)), LCASE("{subj}")))
  ?O rdfs:label ?ol . FILTER(CONTAINS(LCASE(str(?ol)), LCASE("{obj}")))
  ?S ?p1 ?mid .
  ?mid ?p2 ?O .
}} LIMIT 200
"""
        return queries


def sparql_post(query: str) -> dict:
    key = hash(query)
    if key in _SPARQL_CACHE:
        out = _SPARQL_CACHE[key].copy()
        out["from_cache"] = True
        return out
    try:
        r = requests.post(
            DEFAULT_SPARQL_ENDPOINT,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
        if "boolean" in data:
            res = {"success": True, "is_ask": True, "ask_result": bool(data["boolean"])}
        else:
            res = {"success": True, "is_ask": False,
                   "results": data.get("results", {}).get("bindings", [])}
        _SPARQL_CACHE[key] = res
        return res
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool("sparql_executor", return_direct=True)
def sparql_executor(query: str) -> dict:
    """Execute a SPARQL query and return results (cached)."""
    return sparql_post(query)

# ---------------- Entity Resolver Tool ----------------
def sparql_escape(text: str) -> str:
    return re.sub(r'["\'\\\n\r\t]+', ' ', text or "").strip()

@tool("entity_resolver", return_direct=True)
def entity_resolver(term: str) -> dict:
    """
    Resolves entities from an ontology
    :param term:
    :return:
    """
    t = sparql_escape(term)
    query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
SELECT DISTINCT ?entity ?label ?syn WHERE {{
  ?entity rdfs:label ?label .
  OPTIONAL {{ ?entity oboInOwl:hasExactSynonym ?syn }}
  FILTER(
    CONTAINS(LCASE(STR(?label)), LCASE("{t}")) ||
    (BOUND(?syn) && CONTAINS(LCASE(STR(?syn)), LCASE("{t}")))
  )
}}
LIMIT 40
"""
    return sparql_executor.invoke({"query": query})


def _first_uri(res: dict) -> Optional[str]:
    if not res or res.get("is_ask"): return None
    for b in res.get("results", []):
        uri = b.get("entity", {}).get("value")
        if uri: return uri
    return None

# ---------------- Ontology Verifier ----------------
class OntologyVerifier:
    def __init__(self, executor):
        self.executor = executor
    def get_domain_range(self, predicate_uri: str) -> Tuple[Optional[str], Optional[str]]:
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?domain ?range WHERE {{
          OPTIONAL {{ <{predicate_uri}> rdfs:domain ?domain . }}
          OPTIONAL {{ <{predicate_uri}> rdfs:range ?range . }}
        }}
        """
        res = self.executor(query)
        if res.get("success") and not res.get("is_ask"):
            bindings = res.get("results", [])
            if bindings:
                return (
                    bindings[0].get("domain", {}).get("value"),
                    bindings[0].get("range", {}).get("value"),
                )
        return None, None
    def is_instance_of(self, entity_uri: str, class_uri: str) -> bool:
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        ASK {{ <{entity_uri}> a/rdfs:subClassOf* <{class_uri}> . }}
        """
        res = self.executor(query)
        return bool(res.get("ask_result", False))
    def check_disjoint(self, class1: str, class2: str) -> bool:
        query = f"""
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        ASK {{ <{class1}> owl:disjointWith <{class2}> . }}
        """
        res = self.executor(query)
        return bool(res.get("ask_result", False))
    def verify(self, subject_uri: str, predicate_uri: str, object_uri: str) -> dict:
        domain, range_ = self.get_domain_range(predicate_uri)

        subj_ok, obj_ok = None, None
        if domain:
            subj_ok = self.is_instance_of(subject_uri, domain)
        if range_:
            obj_ok = self.is_instance_of(object_uri, range_)

        disjoint = False
        if domain and range_:
            disjoint = self.check_disjoint(domain, range_)

        # Detect swap: subject fits range and object fits domain
        swapped = False
        if domain and range_:
            swapped = self.is_instance_of(subject_uri, range_) and self.is_instance_of(
                object_uri, domain
            )

        return {
            "subject_uri": subject_uri,
            "object_uri": object_uri,
            "domain": domain,
            "range": range_,
            "subject_in_domain": subj_ok,
            "object_in_range": obj_ok,
            "direction_ok": bool(subj_ok and obj_ok),
            "swapped": swapped,
            "disjoint": disjoint,
        }


global_verifier = OntologyVerifier(sparql_post)

@tool("ontology_verification", return_direct=True)
def ontology_verification(subject: str, predicate_uri: str, obj: str) -> dict:
    """Verify triple consistency against ontology domain/range/disjointness."""
    return global_verifier.verify(subject, predicate_uri, obj)

# ---------------- SPARQL Templates + Compiler ----------------
PREFIXES = """\
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
"""

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

def _esc(s: str) -> str:
    return re.sub(r'["\'\\\n\r\t]+', ' ', (s or "")).strip()

def q_exact(pred_uri, subj, obj):
    return f"""{PREFIXES}
ASK {{ ?s rdfs:label "{_esc(subj)}" .
      ?o rdfs:label "{_esc(obj)}" .
      ?s <{pred_uri}> ?o . }}"""

def q_flexible(subj, obj, limit=120):
    return f"""{PREFIXES}
SELECT ?s ?p ?o WHERE {{
  ?s rdfs:label ?sl .
  FILTER(CONTAINS(LCASE(STR(?sl)), LCASE("{_esc(subj)}")))
  ?o rdfs:label ?ol .
  FILTER(CONTAINS(LCASE(STR(?ol)), LCASE("{_esc(obj)}")))
  ?s ?p ?o .
}} LIMIT {limit}"""

def q_twohop(subj, obj, limit=200):
    return f"""{PREFIXES}
SELECT ?mid ?p1 ?p2 ?S ?O WHERE {{
  ?S rdfs:label ?sl . FILTER(CONTAINS(LCASE(STR(?sl)), LCASE("{_esc(subj)}")))
  ?O rdfs:label ?ol . FILTER(CONTAINS(LCASE(STR(?ol)), LCASE("{_esc(obj)}")))
  ?S ?p1 ?mid .
  ?mid ?p2 ?O .
}} LIMIT {limit}"""

def q_subprop_chain(pred_uri, subj, obj, limit=150):
    return f"""{PREFIXES}
ASK {{
  ?s rdfs:label ?sl .
  FILTER(CONTAINS(LCASE(STR(?sl)), LCASE("{_esc(subj)}")))
  ?o rdfs:label ?ol .
  FILTER(CONTAINS(LCASE(STR(?ol)), LCASE("{_esc(obj)}")))
  ?s ?p ?o .
  ?p rdfs:subPropertyOf* <{pred_uri}> .
}}"""

def generate_template_pack(subj: str, pred: str, obj: str) -> dict:
    pred_uri = PREDICATE_URI.get(pred.lower())
    pack = {
        "flexible": q_flexible(subj, obj),
        "twohop": q_twohop(subj, obj),
    }
    if pred_uri:
        pack["exact"] = q_exact(pred_uri, subj, obj)
        pack["subprop_chain"] = q_subprop_chain(pred_uri, subj, obj)
    if pred.lower() in ("bind","drug_interaction","cotreatment"):
        pack["flexible_swapped"] = q_flexible(obj, subj)
    return pack

def compile_plan(plan: dict, subj: str, pred: str, obj: str) -> dict:
    out = {}
    if not isinstance(plan, dict): return out
    steps = plan.get("steps", [])
    for i, step in enumerate(steps[:4]):
        t = (step.get("type") or "").lower()
        swap = bool(step.get("swap", False))
        s, o = (obj, subj) if swap else (subj, obj)
        if t == "flexible": out[f"llm_flex_{i}"] = q_flexible(s, o)
        elif t == "two_hop": out[f"llm_twohop_{i}"] = q_twohop(s, o)
        elif t == "subprop_chain":
            pu = PREDICATE_URI.get(pred.lower())
            if pu: out[f"llm_subprop_{i}"] = q_subprop_chain(pu, s, o)
        elif t == "exact":
            pu = PREDICATE_URI.get(pred.lower())
            if pu: out[f"llm_exact_{i}"] = q_exact(pu, s, o)
    return out

# ---------------- LLM Controller ----------------
class LLMController:
    def __init__(self, model: str = LLM_MODEL):
        self.available = _HAS_OLLAMA
        if self.available:
            try: self.llm = ChatOllama(model=model, temperature=0)
            except Exception as e:
                logger.warning(f"LLM init failed: {e}; fallback to rules.")
                self.available = False
    def plan(self, relation: str, score: float, reason: str, attempts: int) -> str:
        if not self.available:
            if score >= VALIDATION_THRESHOLD: return "accept"
            if "contradiction" in reason or "disjoint" in reason: return "reject"
            if "direction" in reason or "swap" in reason: return "swap"
            return "query_more" if attempts < MAX_PLANNER_ITERS_PER_NEED else "reject"
        prompt = (
            "Decide ONE action from [accept, reject, swap, query_more].\n"
            f"Relation: {relation}\nScore: {score:.3f}\nReason: {reason}\nAttempts: {attempts}\n"
        )
        try:
            resp = self.llm.invoke(prompt)
            action = (getattr(resp, "content", str(resp)) or "").strip().lower()
            for a in ("accept","reject","swap","query_more"):
                if a in action: return a
            return "query_more"
        except Exception as e:
            logger.warning(f"LLM planner failed: {e}")
            return "query_more"
    def refine(self, subj: str, pred: str, obj: str, last_reason: str) -> dict:
        if not self.available: return {}
        prompt = (
            "Return ONLY JSON: {\"steps\":[{\"type\":\"flexible|two_hop|subprop_chain|exact\",\"swap\":false}]}\n"
            f"Subject: {subj}\nPredicate: {pred}\nObject: {obj}\nReason: {last_reason}"
        )
        try:
            resp = self.llm.invoke(prompt)
            content = getattr(resp, "content", "{}").strip()
            try: plan = json.loads(content)
            except Exception:
                content = re.sub(r"^```(?:json)?|```$", "", content)
                plan = json.loads(content)
            return compile_plan(plan, subj, pred, obj)
        except Exception as e:
            logger.warning(f"LLM refine failed: {e}")
            return {}

controller = LLMController()

# ---------------- Verifier ----------------
def verifier(result: dict) -> Dict:
    if not result or not result.get("success"):
        return {"score": 0.0, "reason": "execution failed"}
    if result.get("is_ask") and result.get("ask_result"):
        return {"score": 0.95, "reason": "ASK confirmed"}
    hits = len(result.get("results", []))
    if hits > 0:
        return {"score": min(0.85, 0.4 + hits/20.0), "reason": f"{hits} hits"}
    return {"score": 0.0, "reason": "no evidence"}

# ---------------- Graph Nodes ----------------
def node_supervisor_plan(state: Dict) -> Dict:
    rels = state.get("extracted_relations", [])
    needs = [f"{r['subject']}:{r['predicate']}:{r['object']}" for r in rels]
    state["queue"] = needs
    state["done"] = {n: False for n in needs}
    state["attempts"] = {n: 0 for n in needs}
    state["validation_status"] = {}
    state["ontology_status"] = {}
    state["inferred_relations_all"] = set()
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
        em[subj] = entity_resolver.invoke({"term": subj}); em[subj]["uri"] = _first_uri(em[subj])
    if obj not in em:
        em[obj] = entity_resolver.invoke({"term": obj}); em[obj]["uri"] = _first_uri(em[obj])
    logger.info(
        "Resolved %s → %s, %s → %s",
        subj,
        em.get(subj, {}).get("uri"),
        obj,
        em.get(obj, {}).get("uri"),
    )

    return state

def node_generator(state: Dict) -> Dict:
    if not state.get("current_need"): return state
    subj, pred, obj = state["current_need"].split(":")
    state["current_queries"] = generate_template_pack(subj, pred, obj)
    return state

def node_query_refine(state: Dict) -> Dict:
    need = state.get("current_need")
    if not need: return state
    subj, pred, obj = need.split(":")
    last = state.get("validation_status", {}).get(need, {})
    extra = controller.refine(subj, pred, obj, last.get("reason", ""))
    if extra:
        cq = state.setdefault("current_queries", {})
        cq.update(extra)
        state["current_queries"] = cq
    return state

def _sanitize_query(q: str) -> Optional[str]:
    if not isinstance(q, str): return None
    if "PREFIX rdfs:" not in q: q = PREFIXES + "\n" + q
    if not ("select" in q.lower() or "ask" in q.lower()): return None
    return q

def node_executor(state: Dict) -> Dict:
    if not state.get("current_queries"): return state
    results = {}
    for name, q in state["current_queries"].items():
        sq = _sanitize_query(q)
        if not sq: continue
        results[name] = sparql_executor.invoke({"query": sq})
    state["last_results"] = results
    return state

def node_verifier(state: Dict) -> Dict:
    need = state.get("current_need")
    results = state.get("last_results", {})
    scores, reasons = [], []
    for strat, res in results.items():
        v = verifier(res)
        scores.append(v["score"]); reasons.append(f"{strat}: {v['reason']}")
    state["validation_status"][need] = {
        "score": max(scores) if scores else 0.0,
        "reason": " | ".join(reasons)
    }
    return state

def _triple_tuple(s,p,o): return (s,p,o)

def node_infer(state: Dict) -> Dict:
    known = {(r["subject"], r["predicate"], r["object"]) for r in state.get("extracted_relations", [])}
    inferred_all = state.setdefault("inferred_relations_all", set())
    known |= inferred_all
    triples = list(known)
    new = set()
    for (s1,p1,o1) in triples:
        lp1 = p1.lower()
        if lp1 in ("bind","drug_interaction","cotreatment"):
            new.add(_triple_tuple(o1,p1,s1))
        if lp1 == "positive_correlation":
            for (s2,p2,o2) in triples:
                if o1==s2 and p2.lower()=="positive_correlation":
                    new.add(_triple_tuple(s1,"positive_correlation",o2))
        if lp1 == "negative_correlation":
            for (s2,p2,o2) in triples:
                if o1==s2 and p2.lower()=="negative_correlation":
                    new.add(_triple_tuple(s1,"positive_correlation",o2))
        if lp1 in ("association","associated_with"):
            for (s2,p2,o2) in triples:
                if o1==s2 and p2.lower() in ("association","associated_with"):
                    new.add(_triple_tuple(s1,"association",o2))
    truly_new = [ {"subject":s,"predicate":p,"object":o} for (s,p,o) in new if (s,p,o) not in known ]
    for t in truly_new: inferred_all.add(_triple_tuple(t["subject"], t["predicate"], t["object"]))
    state["new_inferred_relations"] = truly_new
    return state

def node_ontology(state: Dict) -> Dict:
    need = state.get("current_need")
    if not need:
        return state
    subj_txt, pred, obj_txt = need.split(":")
    pred_uri = SPARQLGenerator.predicate_map.get(pred.lower(), "")
    subj_uri = state.get("entity_mappings", {}).get(subj_txt, {}).get("uri")
    obj_uri  = state.get("entity_mappings", {}).get(obj_txt, {}).get("uri")

    if pred_uri and subj_uri and obj_uri:
        res = ontology_verification.invoke({
            "subject": subj_uri,
            "predicate_uri": pred_uri,
            "obj": obj_uri
        })

        ctx = state["validation_status"].setdefault(
            need, {"score": 0.0, "reason": ""}
        )

        if res.get("disjoint"):
            ctx["score"] = 0.0
            ctx["reason"] += " | ontology contradiction"
        elif res.get("direction_ok"):
            ctx["score"] = max(ctx["score"], 0.9)  # bump high if ontology matches
            ctx["reason"] += " | ontology consistent"
        elif res.get("swapped"):
            ctx["score"] = 0.0
            ctx["reason"] += " | ontology swap detected"
        else:
            ctx["score"] = 0.0
            ctx["reason"] += " | ontology mismatch"

        state["validation_status"][need] = ctx

    return state


def node_planner(state: Dict) -> Dict:
    need = state.get("current_need")
    if not need: return state
    attempts = state["attempts"].get(need, 0)
    val = state["validation_status"].get(need, {"score":0.0,"reason":""})
    action = controller.plan(need, val.get("score", 0.0), val.get("reason", ""), attempts)
    state["planner_action"] = action
    state["attempts"][need] = attempts + (1 if action == "query_more" else 0)
    logger.info(f"Planner action for {need}: {action}")
    return state

def route_after_planner(state: Dict) -> str:
    act = state.get("planner_action", "reject")
    if act == "accept": return "accept"
    if act == "reject": return "reject"
    if act == "swap": return "swap"
    if act == "query_more":
        need = state.get("current_need")
        if need and state["attempts"].get(need, 0) > MAX_PLANNER_ITERS_PER_NEED:
            return "reject"
        return "refine"
    return "reject"

def node_accept(state: Dict) -> Dict:
    need = state.get("current_need")
    if need: state["done"][need] = True
    for rel in state.pop("new_inferred_relations", []):
        new_need = f"{rel['subject']}:{rel['predicate']}:{rel['object']}"
        if new_need not in state["done"]:
            state["queue"].append(new_need); state["done"][new_need] = False
    return state

def node_reject(state: Dict) -> Dict:
    need = state.get("current_need");
    if need: state["done"][need] = True
    for rel in state.pop("new_inferred_relations", []):
        new_need = f"{rel['subject']}:{rel['predicate']}:{rel['object']}"
        if new_need not in state["done"]:
            state["queue"].append(new_need); state["done"][new_need] = False
    return state

def node_swap(state: Dict) -> Dict:
    need = state.get("current_need")
    if need:
        s, p, o = need.split(":")
        swapped = f"{o}:{p}:{s}"
        state["done"][need] = True
        if swapped not in state["done"]:
            state["queue"].append(swapped); state["done"][swapped] = False
    for rel in state.pop("new_inferred_relations", []):
        new_need = f"{rel['subject']}:{rel['predicate']}:{rel['object']}"
        if new_need not in state["done"]:
            state["queue"].append(new_need); state["done"][new_need] = False
    return state

def route_after_decide(state: Dict) -> str:
    if all(state["done"].values()): return "end"
    return "continue"

# ---------------- Graph ----------------
graph = StateGraph(dict)
graph.add_node("plan", node_supervisor_plan)
graph.add_node("select", node_supervisor_select)
graph.add_node("entity_resolution", node_resolve_entities)
graph.add_node("generator", node_generator)
graph.add_node("executor", node_executor)
graph.add_node("verifier", node_verifier)
graph.add_node("infer", node_infer)
graph.add_node("ontology", node_ontology)
graph.add_node("planner", node_planner)
graph.add_node("accept", node_accept)
graph.add_node("reject", node_reject)
graph.add_node("swap", node_swap)
graph.add_node("refine", node_query_refine)

graph.add_edge("plan","select")
graph.add_edge("select","entity_resolution")
graph.add_edge("entity_resolution","generator")
graph.add_edge("generator","executor")
graph.add_edge("executor","verifier")
graph.add_edge("verifier","infer")
graph.add_edge("infer","ontology")
graph.add_edge("ontology","planner")

graph.add_conditional_edges("planner", route_after_planner,
    {"accept": "accept","reject":"reject","swap":"swap","refine":"refine"})
graph.add_conditional_edges("accept", route_after_decide, {"end": END, "continue":"select"})
graph.add_conditional_edges("reject", route_after_decide, {"end": END, "continue":"select"})
graph.add_conditional_edges("swap", route_after_decide, {"end": END, "continue":"select"})
graph.add_edge("refine","executor")

graph.set_entry_point("plan")
app = graph.compile()

# ---------------- Runner ----------------
class HierarchicalVerifier:
    def run_case_stream(self, case: Dict):
        init = {
            "original_text": case["text"],
            "extracted_entities": case["entities"],
            "extracted_relations": case["relations"],
        }
        final = app.invoke(init, config={"recursion_limit": 200})
        final["inferred_relations"] = [
            {"subject": s, "predicate": p, "object": o}
            for (s,p,o) in final.get("inferred_relations_all", set())
        ]
        return final

# ---------------- Example ----------------
if __name__=="__main__":
    hv = HierarchicalVerifier()
    cases = [
        {"text": "Metformin treats type 2 diabetes.",
         "entities": [{"text":"Metformin"},{"text":"type 2 diabetes"}],
         "relations":[{"subject":"Metformin","predicate":"treats","object":"type 2 diabetes"}]},
        {"text": "Type 2 diabetes treats Metformin (nonsense).",
         "entities": [{"text":"type 2 diabetes"},{"text":"Metformin"}],
         "relations":[{"subject":"type 2 diabetes","predicate":"treats","object":"Metformin"}]},
    ]
    for case in cases:
        out = hv.run_case_stream(case)
        print("\nCase:", case["text"])
        print("Validation:", json.dumps(out.get("validation_status", {}), indent=2))
        print("Planner attempts:", json.dumps(out.get("attempts", {}), indent=2))
        print("Inferred:", json.dumps(out.get("inferred_relations", []), indent=2))
