"""
Neuro-Symbolic LangGraph: Evidence + LLM Reasoning
Refactor with explicit dataclass state to make data flow robust.

- Evidence:
  * SPARQL (exact-by-URI, subproperty chains, two-hop, label fallbacks)
  * Ontology (domain/range/restrictions; robust fallback via prefix schema)
- LLM Reasoner:
  * Concise justification with skepticism & predicate semantics
- Control:
  * Planner accepts / rejects / swap / query_more using score + ontology + LLM
"""

import json
import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

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
LLM_MODEL = "gemma3:1b"              # local Ollama model name
VALIDATION_THRESHOLD = 0.70
MAX_PLANNER_ITERS_PER_NEED = 2

PREFIXES = """\
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
"""

_PFX = {
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "oboInOwl": "http://www.geneontology.org/formats/oboInOwl#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}

# ---------- Predicate URIs (double-check these for your dataset) ----------
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

# ---------- Pyoxigraph ----------
from pyoxigraph import Store, QueryResultsFormat, RdfFormat

store = Store()
for path in [
    "../../notebooks/data/mondo.owl",
    "../../notebooks/data/go.owl",
    "../../notebooks/data/cl.owl",
    "../../notebooks/data/chebi.owl",
    "../../notebooks/data/ro.owl",
    "../../notebooks/data/omim-ordo/omim.owl",
]:
    try:
        # Accepts either string mime or enum; using enum for clarity
        store.load(path=path, format=RdfFormat.RDF_XML)
        log.info(f"Loaded {path}")
    except Exception as e:
        log.warning(f"Failed to load {path}: {e}")

# ---------- Dataclasses ----------
@dataclass
class RelationVerdict:
    domain: str = "N/A"
    range: str = "N/A"
    domain_prefixes: List[str] = field(default_factory=list)
    range_prefixes: List[str] = field(default_factory=list)
    subject_in_domain: Optional[bool] = None
    object_in_range: Optional[bool] = None
    direction_ok: bool = False
    swapped: bool = False
    disjoint: bool = False
    reasoner_used: bool = False
    used_fallback: bool = False

@dataclass
class VerifierState:
    original_text: str = ""
    extracted_entities: List[Dict] = field(default_factory=list)
    extracted_relations: List[Dict] = field(default_factory=list)

    queue: List[str] = field(default_factory=list)
    done: Dict[str, bool] = field(default_factory=dict)
    attempts: Dict[str, int] = field(default_factory=dict)

    entity_mappings: Dict[str, Dict] = field(default_factory=dict)
    validation_status: Dict[str, Dict] = field(default_factory=dict)
    ontology_status: Dict[str, RelationVerdict] = field(default_factory=dict)
    evidence: Dict[str, Dict] = field(default_factory=dict)
    reasoning: Dict[str, str] = field(default_factory=dict)

    planner_action: Optional[str] = None
    current_need: Optional[str] = None
    current_queries: Optional[Dict[str, str]] = None
    last_results: Optional[Dict[str, dict]] = None
    inferred_relations: List[Dict] = field(default_factory=list)

# ---------- SPARQL executor with cache ----------
_SPARQL_CACHE: Dict[int, dict] = {}

def _hashable(q: str) -> int:
    return hash(q)

def sparql_post(query: str) -> dict:
    key = _hashable(query)
    if key in _SPARQL_CACHE:
        out = dict(_SPARQL_CACHE[key])
        out["from_cache"] = True
        return out
    try:
        result = store.query(query, prefixes=_PFX, use_default_graph_as_union=True)
        if isinstance(result, bool):
            res = {"success": True, "is_ask": True, "ask_result": result, "results": []}
        else:
            raw = result.serialize(format=QueryResultsFormat.JSON)
            parsed = json.loads(raw)
            if "boolean" in parsed:
                res = {"success": True, "is_ask": True, "ask_result": bool(parsed["boolean"]), "results": []}
            else:
                res = {"success": True, "is_ask": False, "results": parsed.get("results", {}).get("bindings", [])}
        _SPARQL_CACHE[key] = res
        return res
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool("sparql_executor", return_direct=True)
def sparql_executor(query: str) -> dict:
    """
    Executes sparql
    :param query:
    :return:
    """
    return sparql_post(query)

# ---------- Helpers ----------
def sparql_escape(text: str) -> str:
    return re.sub(r'["\'\\\n\r\t]+', ' ', text or "").strip()

def _normalize_term(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r'\s*-\s*', '-', t)            # collapse spaced dashes
    t = re.sub(r'[^a-z0-9\- ]+', ' ', t)      # strip non-alnums
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# ---------- Entity Resolver ----------
def _resolver_query_regex(t: str) -> str:
    return f"""
{PREFIXES}
SELECT DISTINCT ?entity ?label WHERE {{
  VALUES ?p {{ rdfs:label oboInOwl:hasExactSynonym skos:altLabel }}
  ?entity ?p ?label .
  FILTER regex(lcase(str(?label)), "{sparql_escape(t)}", "i")
}}
LIMIT 120
"""

def _resolver_query_tokens(t: str, tokens: List[str]) -> str:
    toks = [sparql_escape(x) for x in tokens if x]
    if not toks:
        toks = [sparql_escape(t)]
    filters = " && ".join([f'regex(lcase(str(?label)), "{tok}", "i")' for tok in toks])
    return f"""
{PREFIXES}
SELECT DISTINCT ?entity ?label WHERE {{
  VALUES ?p {{ rdfs:label oboInOwl:hasExactSynonym skos:altLabel }}
  ?entity ?p ?label .
  FILTER ( {filters} )
}}
LIMIT 120
"""

@tool("entity_resolver", return_direct=True)
def entity_resolver(term: str) -> dict:
    """
    Resolves entities
    :param term:
    :return:
    """
    t = _normalize_term(term)
    res = sparql_post(_resolver_query_regex(t))
    if res.get("success") and res.get("results"):
        return res
    tokens = t.split(" ")
    return sparql_post(_resolver_query_tokens(t, tokens))

def _first_uri(res: dict) -> Optional[str]:
    if not res or res.get("is_ask"):
        return None
    for b in res.get("results", []):
        uri = b.get("entity", {}).get("value")
        if uri:
            return uri
    return None

# ---------- Build predicate schema fallback (restrictions) ----------
def _restriction_query() -> str:
    return f"""
{PREFIXES}
SELECT DISTINCT ?prop ?onClass ?restrictionType
WHERE {{
  ?x rdfs:subClassOf ?restriction .
  ?restriction a owl:Restriction ;
               owl:onProperty ?prop ;
               ?restrictionType ?onClass .
  FILTER (?restrictionType IN (owl:someValuesFrom, owl:allValuesFrom))
}}
LIMIT 5000
"""

from typing import Dict, Tuple

PREDICATE_SCHEMA_HARDCODED = {
    "treats": {
        "domain_prefixes": ("CHEBI_", "DRUGBANK_", "ChEMBL", "NCIT_C", "PR_"),  # drugs, chemicals, proteins
        "range_prefixes": ("MONDO_", "DOID_", "HP_", "EFO_", "OMIM"),           # diseases, phenotypes
    },
    "causes": {
        "domain_prefixes": ("CHEBI_", "PR_", "MONDO_", "HP_"),  # drugs, proteins, diseases
        "range_prefixes": ("MONDO_", "HP_"),                    # disease/phenotype
    },
    "association": {
        "domain_prefixes": ("CHEBI_", "PR_", "MONDO_", "DOID_", "HP_", "EFO_", "OMIM"),
        "range_prefixes": ("CHEBI_", "PR_", "MONDO_", "DOID_", "HP_", "EFO_", "OMIM"),
    },
    "positive_correlation": {
        "domain_prefixes": ("CHEBI_", "PR_", "MONDO_", "DOID_", "HP_", "EFO_", "OMIM"),
        "range_prefixes": ("CHEBI_", "PR_", "MONDO_", "DOID_", "HP_", "EFO_", "OMIM"),
    },
    "negative_correlation": {
        "domain_prefixes": ("CHEBI_", "PR_", "MONDO_", "DOID_", "HP_", "EFO_", "OMIM"),
        "range_prefixes": ("CHEBI_", "PR_", "MONDO_", "DOID_", "HP_", "EFO_", "OMIM"),
    },
    "drug_interaction": {
        "domain_prefixes": ("CHEBI_", "DRUGBANK_", "ChEMBL"),
        "range_prefixes": ("CHEBI_", "DRUGBANK_", "ChEMBL"),
    },
    "cotreatment": {
        "domain_prefixes": ("CHEBI_", "DRUGBANK_", "ChEMBL"),
        "range_prefixes": ("CHEBI_", "DRUGBANK_", "ChEMBL"),
    },
    "bind": {
        "domain_prefixes": ("CHEBI_", "PR_", "NCIT_C"),
        "range_prefixes": ("CHEBI_", "PR_", "NCIT_C"),
    },
    "conversion": {
        "domain_prefixes": ("CHEBI_", "ChEMBL"),
        "range_prefixes": ("CHEBI_", "ChEMBL"),
    },
}



def build_schema_from_restrictions(res: dict) -> dict:
    """
    Build a heuristic mapping prop -> {domain_prefixes, range_prefixes}
    from OWL restrictions found in the loaded graphs.
    """
    schema: Dict[str, Dict[str, Tuple[str, ...]]] = {}

    for b in res.get("results", []):
        prop = b.get("prop", {}).get("value")
        onclass = b.get("onClass", {}).get("value")
        if not prop or not onclass:
            continue

        if prop not in schema:
            schema[prop] = {"domain_prefixes": set(), "range_prefixes": set()}

        tail = onclass.split("/")[-1]
        root = tail.split("_")[0] + "_" if "_" in tail else tail

        # --- Heuristic assignment rules ---
        # Disease/phenotype ontologies → usually range
        if any(
            k in onclass
            for k in [
                "MONDO_",
                "DOID_",
                "HP_",
                "EFO_",
                "OMIM",
                "Orphanet",
                "phenotypicSeries",
            ]
        ):
            schema[prop]["range_prefixes"].add(root)

        # Chemicals, proteins, drugs → usually domain (but can also be range!)
        if any(
            k in onclass
            for k in [
                "CHEBI_",
                "PR_",
                "NCIT_C",
                "DRUGBANK_",
                "UNII",
                "ChEMBL",
                "CHEMBL",
            ]
        ):
            schema[prop]["domain_prefixes"].add(root)
            schema[prop]["range_prefixes"].add(root)  # <-- allow in range too

        # --- Symmetric fix for correlation/association predicates ---
        if any(
            kw in prop.lower() for kw in ["correlation", "associated", "interaction"]
        ):
            schema[prop]["domain_prefixes"].update(
                {
                    "CHEBI_",
                    "PR_",
                    "NCIT_C",
                    "DRUGBANK_",
                    "ChEMBL",
                    "MONDO_",
                    "DOID_",
                    "HP_",
                    "EFO_",
                    "OMIM",
                }
            )
            schema[prop]["range_prefixes"].update(
                {
                    "CHEBI_",
                    "PR_",
                    "NCIT_C",
                    "DRUGBANK_",
                    "ChEMBL",
                    "MONDO_",
                    "DOID_",
                    "HP_",
                    "EFO_",
                    "OMIM",
                }
            )

    # Normalize to tuples
    normalized = {}
    for prop, buckets in schema.items():
        normalized[prop] = {
            "domain_prefixes": tuple(sorted(buckets["domain_prefixes"])),
            "range_prefixes": tuple(sorted(buckets["range_prefixes"])),
        }
    return normalized


# Seed fallback for some known relations
PREDICATE_SCHEMA_FALLBACK = {
    # treats: drug/chemical → disease/phenotype-ish
    PREDICATE_URI["treats"]: {
        "domain_prefixes": ("CHEBI_", "DRUGBANK_", "PR_", "NCIT_C", "ChEMBL"),
        "range_prefixes": ("MONDO_", "DOID_", "HP_", "EFO_", "OMIM"),
    },
}

# Merge in restriction-based schema (best-effort)
try:
    _res = sparql_post(_restriction_query())
    if _res.get("success"):
        # PREDICATE_SCHEMA_FALLBACK.update(build_schema_from_restrictions(_res))
        PREDICATE_SCHEMA_FALLBACK.update(PREDICATE_SCHEMA_HARDCODED)
        log.info("Built PREDICATE_SCHEMA_FALLBACK from restrictions with %d props",
                 len(PREDICATE_SCHEMA_FALLBACK))
except Exception as e:
    log.warning(f"Failed building restriction schema: {e}")

# ---------- Ontology Verifier ----------
class OntologyVerifier:
    def __init__(self, executor=None):
        self.executor = executor

    @staticmethod
    def _iri_has_prefix(iri: str, prefixes: Tuple[str, ...] | List[str]) -> bool:
        if not iri:
            return False
        return any(p in iri for p in prefixes)

    def _domain_range_via_sparql(self, predicate_uri: str) -> Tuple[Optional[str], Optional[str]]:
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

    def verify(self, subject_uri: str, predicate_uri: str, object_uri: str) -> RelationVerdict:
        verdict = RelationVerdict()
        if not self.executor or not predicate_uri:
            return verdict

        domain, rng = self._domain_range_via_sparql(predicate_uri)
        verdict.domain = domain if domain else "N/A"
        verdict.range = rng if rng else "N/A"

        subj_ok = obj_ok = None
        used_fallback = False
        domain_prefixes: List[str] = []
        range_prefixes: List[str] = []

        # If explicit domain/range present, do minimal compatibility check
        if domain:
            subj_ok = (subject_uri == domain) or self._iri_has_prefix(subject_uri, (domain,))
        if rng:
            obj_ok = (object_uri == rng) or self._iri_has_prefix(object_uri, (rng,))

        # If ontology silent, use schema fallback built from restrictions
        if domain is None and rng is None:
            fb = PREDICATE_SCHEMA_FALLBACK.get(predicate_uri)
            if fb:
                domain_prefixes = list(fb.get("domain_prefixes", []))
                range_prefixes = list(fb.get("range_prefixes", []))
                subj_ok = self._iri_has_prefix(subject_uri, domain_prefixes)
                obj_ok = self._iri_has_prefix(object_uri, range_prefixes)
                used_fallback = True

        verdict.domain_prefixes = domain_prefixes
        verdict.range_prefixes = range_prefixes
        verdict.subject_in_domain = subj_ok
        verdict.object_in_range = obj_ok
        verdict.direction_ok = bool(subj_ok and obj_ok)
        verdict.used_fallback = used_fallback
        return verdict

global_verifier = OntologyVerifier(sparql_post)

@tool("ontology_verification", return_direct=True)
def ontology_verification(subject: str, predicate_uri: str, obj: str) -> dict:
    """
    Verifies entitfies from ontology
    :param subject:
    :param predicate_uri:
    :param obj:
    :return:
    """
    return asdict(global_verifier.verify(subject, predicate_uri, obj))

# ---------- SPARQL templates ----------
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
    if subj_uri and obj_uri:
        if pred_uri:
            pack["exact_uri"] = q_exact_uri(pred_uri, subj_uri, obj_uri)
            pack["subprop_chain_uri"] = q_subprop_chain_uri(pred_uri, subj_uri, obj_uri)
        pack["twohop_uri"] = q_twohop_uri(subj_uri, obj_uri)
    pack["flexible"] = q_flexible(subj, obj)
    pack["twohop_label"] = q_twohop_label(subj, obj)
    return pack

# ---------- Evidence aggregator ----------
def agg_verifier(result: dict) -> Dict:
    if not result or not result.get("success"):
        err = result.get("error", "unknown error") if isinstance(result, dict) else "unknown error"
        return {"score": 0.0, "reason": f"execution failed: {err}"}
    if result.get("is_ask"):
        if result.get("ask_result"):
            return {"score": 0.95, "reason": "ASK confirmed"}
        return {"score": 0.2, "reason": "ASK false (no prior triple)"}  # not fatal for novelty
    hits = len(result.get("results", []))
    if hits > 0:
        return {"score": min(0.85, 0.4 + hits/20.0), "reason": f"{hits} hits"}
    return {"score": 0.15, "reason": "no evidence (candidate may be novel)"}

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
        # fallback if no LLM
        if not self.available:
            ont = evidence.get("ontology", {})
            if ont.get("direction_ok"):
                return "Ontology typing consistent; plausible subject→object."
            if "ASK confirmed" in str(evidence.get("sparql_summary", "")):
                return "Exact ASK confirmed in KB."
            return "Unsupported: no ASK and no ontology typing."

        # extract entities
        subj = evidence.get("entities", {}).get("subject", {})
        obj  = evidence.get("entities", {}).get("object", {})
        subj_str = f"{subj.get('text')} ({subj.get('uri')})" if subj.get("uri") else subj.get("text")
        obj_str  = f"{obj.get('text')} ({obj.get('uri')})" if obj.get("uri") else obj.get("text")

        # predicate semantics
        verdict = evidence.get("ontology", {})
        predicate_sem = (
            f"Explicit domain={verdict.get('domain','N/A')} "
            f"range={verdict.get('range','N/A')} "
            f"(expected subject types: {verdict.get('domain_prefixes')}, "
            f"expected object types: {verdict.get('range_prefixes')})\n"
            f"direction_ok={verdict.get('direction_ok')} "
            f"used_fallback={verdict.get('used_fallback')}"
        )

        # build prompt
        prompt = (
            "You are a biomedical reasoning agent. Default stance is skepticism.\n"
            "Assess whether the relation is ontologically valid given the evidence.\n\n"
            f"Relation: {relation}\n"
            f"Subject: {subj_str}\n"
            f"Object: {obj_str}\n\n"
            f"SPARQL evidence: {evidence.get('sparql_summary')}\n"
            f"Predicate semantics: {predicate_sem}\n\n"
            "Rules:\n"
            "- If ASK/SPARQL confirms, mark as supported.\n"
            "- If ontology typing contradicts, say invalid and why.\n"
            "- If fallback typing matches (direction_ok=True, used_fallback=True), "
            "mark as novel but plausible.\n"
            "- Do not assume plausibility just because no contradiction was found.\n"
            "Answer concisely."
        )

        try:
            resp = self.llm.invoke(prompt)
            return (getattr(resp, "content", str(resp)) or "").strip()
        except Exception as e:
            return f"LLM failed to produce a justification: {e}"
reasoner = LLMReasoner()

# ---------- Graph Nodes ----------
def node_supervisor_plan(state: VerifierState) -> VerifierState:
    rels = state.extracted_relations
    state.queue = [f"{r['subject']}:{r['predicate']}:{r['object']}" for r in rels]
    state.done = {n: False for n in state.queue}
    state.attempts = {n: 0 for n in state.queue}
    state.validation_status = {}
    state.ontology_status = {}
    state.evidence = {}
    state.reasoning = {}
    return state

def node_supervisor_select(state: VerifierState) -> VerifierState:
    while state.queue and state.done[state.queue[0]]:
        state.queue.pop(0)
    state.current_need = state.queue[0] if state.queue else None
    return state

def node_resolve_entities(state: VerifierState) -> VerifierState:
    if not state.current_need:
        return state
    subj, pred, obj = state.current_need.split(":")
    em = state.entity_mappings
    if subj not in em:
        res = entity_resolver.invoke({"term": subj})
        em[subj] = res
        em[subj]["uri"] = _first_uri(res)
    if obj not in em:
        res = entity_resolver.invoke({"term": obj})
        em[obj] = res
        em[obj]["uri"] = _first_uri(res)
    log.info("Resolved %s → %s, %s → %s", subj, em.get(subj, {}).get("uri"), obj, em.get(obj, {}).get("uri"))
    return state

def node_generator(state: VerifierState) -> VerifierState:
    if not state.current_need:
        return state
    subj, pred, obj = state.current_need.split(":")
    em = state.entity_mappings
    subj_uri = em.get(subj, {}).get("uri")
    obj_uri = em.get(obj, {}).get("uri")
    state.current_queries = generate_template_pack(subj, pred, obj, subj_uri, obj_uri)
    return state

def node_executor(state: VerifierState) -> VerifierState:
    if not state.current_queries:
        return state
    results = {}
    for name, q in state.current_queries.items():
        results[name] = sparql_executor.invoke({"query": q})
    state.last_results = results
    return state

def node_verifier(state: VerifierState) -> VerifierState:
    need = state.current_need
    if not need:
        return state
    results = state.last_results or {}
    scores, reasons = [], []
    for strat, res in results.items():
        v = agg_verifier(res)
        scores.append(v["score"])
        reasons.append(f"{strat}: {v['reason']}")
    state.validation_status[need] = {
        "score": max(scores) if scores else 0.0,
        "reason": " | ".join(reasons) if reasons else "no queries"
    }
    return state

def node_ontology(state: VerifierState) -> VerifierState:
    need = state.current_need
    if not need:
        return state
    subj_txt, pred, obj_txt = need.split(":")
    pred_uri = PREDICATE_URI.get(pred.lower(), "")
    subj_uri = state.entity_mappings.get(subj_txt, {}).get("uri")
    obj_uri  = state.entity_mappings.get(obj_txt, {}).get("uri")
    if pred_uri and subj_uri and obj_uri:
        verdict = global_verifier.verify(subj_uri, pred_uri, obj_uri)
        state.ontology_status[need] = verdict
        ctx = state.validation_status.setdefault(need, {"score": 0.0, "reason": ""})
        if verdict.direction_ok:
            ctx["score"] = max(ctx["score"], 0.9)
            ctx["reason"] += " | ontology consistent"
        else:
            ctx["reason"] += " | ontology no evidence"
        state.validation_status[need] = ctx
    else:
        state.ontology_status[need] = RelationVerdict(
            subject_in_domain=False, object_in_range=False, direction_ok=False
        )
    return state

def node_collect_evidence(state: VerifierState) -> VerifierState:
    need = state.current_need
    if not need:
        return state

    subj_txt, pred, obj_txt = need.split(":")
    subj_map = state.entity_mappings.get(subj_txt, {})
    obj_map  = state.entity_mappings.get(obj_txt, {})

    def _entity_context(txt, mapping):
        if not mapping:
            return {"text": txt, "uri": None, "candidates": []}
        candidates = [
            {"uri": b["entity"]["value"], "label": b["label"]["value"]}
            for b in mapping.get("results", [])
            if b.get("entity") and b.get("label")
        ]
        return {
            "text": txt,
            "uri": mapping.get("uri"),
            "candidates": candidates
        }

    state.evidence[need] = {
        "sparql_summary": state.validation_status.get(need, {}).get("reason", ""),
        "sparql_score": state.validation_status.get(need, {}).get("score", 0.0),
        "ontology": vars(state.ontology_status.get(need, RelationVerdict())),
        "raw_results": state.last_results or {},
        "entities": {
            "subject": _entity_context(subj_txt, subj_map),
            "object":  _entity_context(obj_txt, obj_map),
        }
    }
    return state


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
        # Reject if ontology direction clearly wrong
        if "ontology no evidence" in reason and "direction" in explanation.lower() and "false" in explanation.lower():
            return "reject"
        # Accept if ontology consistent
        if "ontology consistent" in reason:
            return "accept"
        # Accept if high score
        if score >= VALIDATION_THRESHOLD:
            return "accept"
        # Contradictions / disjoint (not implemented, placeholder)
        if "contradiction" in reason or "disjoint" in reason:
            return "reject"
        # Iterate if not enough attempts
        if attempts < MAX_PLANNER_ITERS_PER_NEED:
            return "query_more"
        return "reject"

planner = LLMController()

def node_llm_reasoner(state: VerifierState) -> VerifierState:
    need = state.current_need
    if not need:
        return state
    explanation = reasoner.explain(need, state.evidence.get(need, {}))
    state.reasoning[need] = explanation
    log.info(f"Reasoning for {need}: {explanation}")
    return state

def node_planner(state: VerifierState) -> VerifierState:
    need = state.current_need
    if not need:
        return state
    attempts = state.attempts.get(need, 0)
    val = state.validation_status.get(need, {"score": 0.0, "reason": ""})
    explanation = state.reasoning.get(need, "")
    action = planner.plan(need, val.get("score", 0.0), val.get("reason", ""), attempts, explanation)
    state.planner_action = action
    state.attempts[need] = attempts + (1 if action == "query_more" else 0)
    log.info(f"Planner action for {need}: {action}")
    return state

def route_after_planner(state: VerifierState) -> str:
    act = state.planner_action or "reject"
    if act == "accept": return "accept"
    if act == "reject": return "reject"
    if act == "swap": return "swap"
    if act == "query_more": return "refine"
    return "reject"

def node_refine(state: VerifierState) -> VerifierState:
    # Simple no-op: we already include flexible/twohop label queries.
    return state

def node_accept(state: VerifierState) -> VerifierState:
    need = state.current_need
    if need:
        state.done[need] = True
    return state

def node_reject(state: VerifierState) -> VerifierState:
    need = state.current_need
    if need:
        state.done[need] = True
    return state

def node_swap(state: VerifierState) -> VerifierState:
    need = state.current_need
    if not need:
        return state
    s, p, o = need.split(":")
    swapped = f"{o}:{p}:{s}"
    state.done[need] = True
    if swapped not in state.done:
        state.queue.append(swapped)
        state.done[swapped] = False
        state.attempts[swapped] = 0
    return state

def route_after_decide(state: VerifierState) -> str:
    if state.done and all(state.done.values()):
        return "end"
    return "continue"

# ---------- Inference (optional, simple) ----------
def node_infer(state: VerifierState) -> VerifierState:
    need = state.current_need
    if not need:
        return state
    # Example: if drug treats disease, link drug to downstream phenotypes caused by disease
    subj, pred, obj = need.split(":")
    if pred.lower() == "treats":
        q = f"""
{PREFIXES}
SELECT ?downstream WHERE {{
  <{obj}> <http://purl.obolibrary.org/obo/RO_0002410> ?downstream .
}}
"""
        res = sparql_post(q)
        new_relations = []
        for b in res.get("results", []):
            dwn = b.get("downstream", {}).get("value")
            if dwn:
                new_relations.append({"subject": subj, "predicate": "association", "object": dwn})
        if new_relations:
            state.inferred_relations.extend(new_relations)
            for r in new_relations:
                rel_str = f"{r['subject']}:{r['predicate']}:{r['object']}"
                if rel_str not in state.done:
                    state.queue.append(rel_str)
                    state.done[rel_str] = False
                    state.attempts[rel_str] = 0
    return state

# ---------- Graph ----------
graph = StateGraph(VerifierState)
graph.add_node("plan", node_supervisor_plan)
graph.add_node("select", node_supervisor_select)
graph.add_node("resolve", node_resolve_entities)
graph.add_node("generate", node_generator)
graph.add_node("exec", node_executor)
graph.add_node("verify", node_verifier)
graph.add_node("ontology", node_ontology)
graph.add_node("infer", node_infer)  # optional
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
graph.add_edge("ontology", "infer")  # run inference before collecting evidence
graph.add_edge("infer", "collect_evidence")
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
        init = VerifierState(
            original_text=case.get("text", ""),
            extracted_entities=case.get("entities", []),
            extracted_relations=case.get("relations", []),
        )
        final: VerifierState = app.invoke(init, config={"recursion_limit": 200})
        return {
            "validation_status": final['validation_status'],
            "ontology_status": {k: asdict(v) for k, v in final['ontology_status'].items()},
            "entity_mappings": {k: {"uri": v.get("uri")} for k, v in final['entity_mappings'].items()},
            "evidence": final['evidence'],
            "reasoning": final['reasoning'],
            "attempts": final['attempts'],
            "inferred_relations": final['inferred_relations'],
            'state': final
        }

# ---------- Example ----------
if __name__ == "__main__":
    hv = HierarchicalVerifier()
    cases = [
        {
            "text": "Metformin treats type 2 diabetes.",
            "entities": [{"text": "Metformin"}, {"text": "type 2 diabetes"}],
            "relations": [{"subject": "Metformin", "predicate": "treats", "object": "type 2 diabetes"}],
        },
        {
            "text": "Type 2 diabetes treats Metformin (nonsense).",
            "entities": [{"text": "type 2 diabetes"}, {"text": "Metformin"}],
            "relations": [{"subject": "type 2 diabetes", "predicate": "treats", "object": "Metformin"}],
        },
        {
            "text": "glucose positively correlates with insulin.",
            "entities": [{"text": "glucose"}, {"text": "insulin"}],
            "relations": [{"subject": "glucose", "predicate": "positive_correlation", "object": "insulin"}],
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
        if out["inferred_relations"]:
            print("\nInferred:", json.dumps(out["inferred_relations"], indent=2))
