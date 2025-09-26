"""
SPARQL Query Agent with Ontology-backed Hallucination Verification
- Iterative query expansion to gather deep ontological context
- Evidence scoring to decide if an LLM-asserted relation is supported or hallucinated
"""

import json
import time
import re
import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import requests
import numpy as np

# Optional: pyoxigraph Store (if installed and you want to load local ontologies)
try:
    from pyoxigraph import Store
    PYOX_AVAILABLE = True
except Exception:
    Store = None
    PYOX_AVAILABLE = False

# LLM client placeholders (keeps your previous ChatOllama usage)
# Keep same imports you used earlier - adapt to your environment:
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

# For state machine / workflow - keep using your langgraph
# (we will instantiate nodes like before)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ---------------------------------------------------------
# Config & helpers
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SPARQL_ENDPOINT = "https://sparql.hegroup.org/sparql/"  # your previous fallback

# protect labels for SPARQL string usage
def sparql_escape_label(label: str) -> str:
    if label is None:
        return ""
    # Escape backslashes and quotes, collapse newlines
    safe = label.replace("\\", "\\\\").replace('"', r'\"').replace("\n", " ")
    return safe

# detect if query is ASK
def is_ask_query(q: str) -> bool:
    return bool(re.search(r'\bASK\b', q, re.IGNORECASE))

# ---------------------------------------------------------
# SPARQL TEMPLATES (expanded)
# ---------------------------------------------------------
SPARQL_TEMPLATES = {
    "validate_relation": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
ASK WHERE {{
  ?subject rdfs:label "{subject_label}" .
  ?object rdfs:label "{object_label}" .
  ?subject {predicate} ?object .
}}
""",
    "get_entity_type": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?type ?typeLabel WHERE {{
  ?entity rdfs:label "{entity_label}" .
  ?entity rdf:type ?type .
  ?type rdfs:label ?typeLabel .
}}
LIMIT 10
""",
    "find_relations": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?predicate ?object ?objectLabel WHERE {{
  ?subject rdfs:label "{subject_label}" .
  ?subject ?predicate ?object .
  OPTIONAL {{ ?object rdfs:label ?objectLabel . }}
  FILTER(!isBlank(?object))
}}
LIMIT 50
""",
    "get_hierarchy": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdfs2: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?parent ?parentLabel WHERE {{
  ?entity rdfs:label "{entity_label}" .
  ?entity rdfs:subClassOf+ ?parent .
  OPTIONAL {{ ?parent rdfs:label ?parentLabel . }}
}}
LIMIT 50
""",
    "find_synonyms": """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?altLabel WHERE {{
  ?entity rdfs:label "{entity_label}" .
  {{
    ?entity skos:altLabel ?altLabel .
  }} UNION {{
    ?entity rdfs:label ?altLabel .
  }}
}}
LIMIT 50
""",
    "find_mechanism": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
SELECT ?mechanism ?mechanismLabel WHERE {{
  ?drug rdfs:label "{drug_label}" .
  ?drug ?p ?mechanism .
  OPTIONAL {{ ?mechanism rdfs:label ?mechanismLabel . }}
  FILTER(CONTAINS(LCASE(STR(?p)), "mechan") || CONTAINS(LCASE(STR(?p)), "role") || CONTAINS(LCASE(STR(?p)), "inheres"))
}}
LIMIT 20
"""
}

# ---------------------------------------------------------
# Template matcher and predicate mapping
# ---------------------------------------------------------
class TemplateMatcher:
    def __init__(self):
        # extended mapping including common RO relations and simpler verbs
        self.predicate_map = {
            "treats": "obo:RO_0002606",
            "causes": "obo:RO_0002410",
            "prevents": "obo:RO_0002559",
            "inhibits": "obo:RO_0002449",
            "activates": "obo:RO_0002448",
            # alternate names
            "reduces risk of": "obo:RO_0002606",
            "is associated with": "obo:RO_0002610"
        }

    def match(self, need: str, context: Dict) -> Optional[str]:
        parts = need.split(":")
        t = parts[0]
        if t == "identify_type" and len(parts) > 1:
            entity = sparql_escape_label(parts[1])
            return SPARQL_TEMPLATES["get_entity_type"].format(entity_label=entity)

        if t == "get_hierarchy" and len(parts) > 1:
            entity = sparql_escape_label(parts[1])
            return SPARQL_TEMPLATES["get_hierarchy"].format(entity_label=entity)

        if t == "validate_relation" and len(parts) > 3:
            subj, pred, obj = parts[1], parts[2], parts[3]
            predicate_uri = self._convert_predicate(pred)
            return SPARQL_TEMPLATES["validate_relation"].format(
                subject_label=sparql_escape_label(subj),
                predicate=predicate_uri,
                object_label=sparql_escape_label(obj)
            )

        if t == "find_relations" and len(parts) > 1:
            entity = sparql_escape_label(parts[1])
            return SPARQL_TEMPLATES["find_relations"].format(subject_label=entity)

        if t == "find_synonyms" and len(parts) > 1:
            entity = sparql_escape_label(parts[1])
            return SPARQL_TEMPLATES["find_synonyms"].format(entity_label=entity)

        if t == "find_mechanism" and len(parts) > 1:
            drug = sparql_escape_label(parts[1])
            return SPARQL_TEMPLATES["find_mechanism"].format(drug_label=drug)

        return None

    def _convert_predicate(self, predicate: str) -> str:
        return self.predicate_map.get(predicate.lower(), f"obo:{predicate}")

# ---------------------------------------------------------
# SPARQL Executor (with caching and retries)
# ---------------------------------------------------------
class SPARQLExecutor:
    def __init__(self, endpoint: str = DEFAULT_SPARQL_ENDPOINT, ontology_path: Optional[str] = None, max_retries: int = 3):
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.cache: Dict[str, Dict] = {}
        if PYOX_AVAILABLE and ontology_path:
            try:
                self.store = Store()
                self.store.load(ontology_path, format="application/rdf+xml")
                logger.info("Loaded ontology into local store.")
            except Exception as e:
                logger.warning(f"Could not load ontology locally: {e}")
                self.store = None
        else:
            self.store = None

    def execute(self, query: str) -> Dict[str, Any]:
        key = hash(query)
        if key in self.cache:
            return {"from_cache": True, **self.cache[key]}

        start = time.time()
        # If local store exists and supports querying, you could run it here.
        # For portability we use endpoint HTTP POST
        attempt = 0
        backoff = 0.5
        while attempt < self.max_retries:
            try:
                headers = {"Accept": "application/sparql-results+json"}
                resp = requests.post(self.endpoint, data={"query": query}, headers=headers, timeout=15)
                if resp.status_code != 200:
                    raise Exception(f"Bad status: {resp.status_code} {resp.text[:200]}")
                data = resp.json()

                # convert to canonical results
                bindings = data.get("results", {}).get("bindings", [])
                exec_time = time.time() - start
                result = {
                    "success": True,
                    "execution_time": exec_time,
                    "results": bindings,
                    "count": len(bindings),
                    "is_ask": is_ask_query(query),
                    "raw": data
                }
                self.cache[key] = result
                return result
            except Exception as e:
                attempt += 1
                logger.warning(f"SPARQL attempt {attempt} failed: {e}")
                time.sleep(backoff)
                backoff *= 2

        # final failure
        return {
            "success": False,
            "error": f"Failed after {self.max_retries} attempts"
        }

# ---------------------------------------------------------
# QueryPlanner: expand needs and orchestrate iterative exploration
# ---------------------------------------------------------
@dataclass
class QueryPlanner:
    matcher: TemplateMatcher
    max_depth: int = 3
    visited_needs: set = field(default_factory=set)

    def plan_initial(self, state: Dict) -> List[str]:
        # initial needs from extraction
        needs = []
        for ent in state["extracted_entities"]:
            text = ent.get("text")
            if not ent.get("ontology_id") or ent.get("type") == "UNKNOWN":
                needs.append(f"identify_type:{text}")
            # always consider synonym discovery to catch label variance
            needs.append(f"find_synonyms:{text}")

            if ent.get("type") in ["DISEASE", "DISORDER"]:
                needs.append(f"get_hierarchy:{text}")

        for rel in state["extracted_relations"]:
            subj = rel["subject"]
            pred = rel["predicate"]
            obj = rel["object"]
            needs.append(f"validate_relation:{subj}:{pred}:{obj}")
            # if drug-disease candidate, seek mechanism
            if ("DRUG" in rel.get("subject_type", "").upper() or "CHEMICAL" in rel.get("subject_type", "").upper()) and \
               ("DISEASE" in rel.get("object_type", "").upper() or "DISORDER" in rel.get("object_type", "").upper()):
                needs.append(f"find_mechanism:{subj}")

        # deduplicate while preserving priority
        result = []
        seen = set()
        for n in needs:
            if n not in seen:
                seen.add(n)
                result.append(n)
        return result

    def expand(self, need: str, response_results: Dict, depth: int) -> List[str]:
        """From a need and results produce follow-ups; depth-limited"""
        if depth >= self.max_depth:
            return []

        outs = []
        parts = need.split(":")
        t = parts[0]
        if t == "validate_relation":
            # if ASK failed, search for relations and synonyms of subject/object and mechanistic links
            if not response_results.get("success") or (response_results.get("is_ask") and response_results.get("count", 0) == 0):
                subj = parts[1]
                obj = parts[3] if len(parts) > 3 else None
                if subj:
                    outs.append(f"find_relations:{subj}")
                    outs.append(f"find_synonyms:{subj}")
                if obj:
                    outs.append(f"find_relations:{obj}")
                    outs.append(f"find_synonyms:{obj}")
                # try mechanism for drug-disease even if initial ASK fails
                outs.append(f"find_mechanism:{subj}")
        elif t == "find_synonyms":
            # if synonyms found, plan to re-run validate relations using synonyms (higher chance of match)
            if response_results.get("count", 0) > 0:
                for b in response_results.get("results", []):
                    alt = b.get("altLabel", {}).get("value")
                    if alt:
                        # create new needs to revalidate relations using alt label
                        # We don't know which relation yet; caller should trigger revalidation
                        outs.append(f"resolved_synonym:{alt}")
        elif t == "get_hierarchy":
            # if we get parents, check parent labels for relations (broaden)
            if response_results.get("count", 0) > 0:
                for b in response_results.get("results", []):
                    parent_label = b.get("parentLabel", {}).get("value")
                    if parent_label:
                        outs.append(f"find_relations:{parent_label}")
                        outs.append(f"find_synonyms:{parent_label}")
        # deduplicate
        return list(dict.fromkeys(outs))

# ---------------------------------------------------------
# SPARQL Generator with LLM fallback & context seeding
# ---------------------------------------------------------
class SPARQLGenerator:
    def __init__(self, model_name: str = "llama3.1:8b"):
        # deterministic
        self.llm = ChatOllama(model=model_name, temperature=0)
        self.matcher = TemplateMatcher()

    def generate(self, need: str, context: Dict) -> str:
        # try template first
        templ = self.matcher.match(need, context)
        if templ:
            return templ

        # build a richer prompt including ontology-derived context to encourage expansion
        prompt = self._create_prompt(need, context)
        sys = SystemMessage(content=prompt)
        try:
            resp = self.llm.invoke([sys])
            # `resp` might be structured or text — adapt extracting content
            content = getattr(resp, "content", resp)
            query = self._extract_sparql_from_response(content)
            return query
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            # fallback: return an empty safe ASK that will fail but is predictable
            return f"ASK WHERE {{ ?s ?p ?o . FILTER(false) }}"

    def _create_prompt(self, need: str, context: Dict) -> str:
        short_ctx = {
            "entities": [e.get("text") for e in context.get("entities", [])],
            "relations": context.get("relations", [])
        }
        return (
            "You are an expert SPARQL builder for biomedical ontologies (MONDO, CHEBI, GO, OBO). "
            "Use available labels, synonyms, and hierarchy to create SPARQL queries that help validate claims. "
            "Return ONLY the SPARQL query.\n\n"
            f"Information need: {need}\n"
            f"Context (entities/relations): {json.dumps(short_ctx, indent=2)}\n\n"
            "Prefer ASK for direct validation and SELECT to gather supporting evidence (labels, predicate URIs)."
        )

    def _extract_sparql_from_response(self, response: str) -> str:
        # remove code fences
        s = re.sub(r'```(?:sparql)?\n', '', response)
        s = s.replace('```', '')
        return s.strip()

# ---------------------------------------------------------
# Context Augmenter + Evidence scoring
# ---------------------------------------------------------
class ContextAugmenter:
    def augment(self, state: Dict) -> str:
        parts = []
        parts.append("Original text: " + state.get("original_text", ""))
        parts.append("\n--- Ontology Evidence ---\n")
        for r in state.get("query_results", []):
            if r.get("success"):
                parts.append(self._format_single_result(r))
        if state.get("validation_status"):
            parts.append("\n--- Validation Summary ---\n")
            parts.append(json.dumps(state["validation_status"], indent=2))
        return "\n".join(parts)

    def _format_single_result(self, r: Dict) -> str:
        q = r.get("query", "")[:200]
        cnt = r.get("count", 0)
        sample = ""
        if r.get("results"):
            sample_items = []
            for row in r["results"][:3]:
                row_str = {k: v.get("value") if isinstance(v, dict) else str(v) for k, v in row.items()}
                sample_items.append(row_str)
            sample = json.dumps(sample_items, indent=2)
        return f"Query: {q}\nCount: {cnt}\nSample: {sample}\n"

def compute_validation_score(results_for_relation: List[Dict]) -> float:
    """
    Combines signals:
     - direct ASK true -> high score
     - number of supporting bindings
     - mechanistic evidence presence
     - depth of ontology support (parents)
    Returns 0..1
    """
    if not results_for_relation:
        return 0.0

    score = 0.0
    # if any ASK returned true
    for r in results_for_relation:
        if r.get("is_ask") and r.get("success") and r.get("count", 0) > 0:
            score = max(score, 0.9)

    # count supporting triples
    total_bindings = sum(r.get("count", 0) for r in results_for_relation)
    # log-scale contribution (diminishing returns)
    if total_bindings > 0:
        score = max(score, min(0.7, 0.2 + math.log1p(total_bindings) / 5.0))

    # mechanistic boost
    mech_evidence = any("mechanism" in (json.dumps(r.get("query", "")).lower()) or
                        any("mechanism" in k.lower() or "role" in k.lower() for k in (",".join(r.get("query", "").lower().split()) ,))
                        for r in results_for_relation)
    if mech_evidence:
        score = min(1.0, score + 0.15)

    return float(min(1.0, score))

# ---------------------------------------------------------
# Workflow wiring similar to your original LangGraph nodes
# ---------------------------------------------------------
class SPARQLAgent:
    def __init__(self, endpoint: str = DEFAULT_SPARQL_ENDPOINT, model_name: str = "llama3.1:8b"):
        self.generator = SPARQLGenerator(model_name=model_name)
        self.executor = SPARQLExecutor(endpoint=endpoint)
        self.matcher = self.generator.matcher
        self.planner = QueryPlanner(self.matcher, max_depth=3)
        self.augmenter = ContextAugmenter()

    def run_for_case(self, case: Dict, max_iterations: int = 20) -> Dict:
        # initial state
        state = {
            "original_text": case.get("text", ""),
            "extracted_entities": case.get("entities", []),
            "extracted_relations": case.get("relations", []),
            "information_needs": [],
            "generated_queries": [],
            "query_results": [],
            "validation_status": {},
            "messages": [],
            "metrics": {}
        }

        # create initial needs
        queue = self.planner.plan_initial(state)
        iteration = 0
        depth_map = {n: 0 for n in queue}

        all_query_objs = []
        # We'll also keep mapping relation->list of results for scoring
        relation_evidence: Dict[str, List[Dict]] = {}

        while queue and iteration < max_iterations:
            need = queue.pop(0)
            d = depth_map.get(need, 0)
            iteration += 1
            logger.info(f"[iter {iteration}] Processing need='{need}' (depth {d})")

            query = self.generator.generate(need, {
                "entities": state["extracted_entities"],
                "relations": state["extracted_relations"]
            })

            # store generated query
            qobj = {"need": need, "query": query, "priority": iteration}
            all_query_objs.append(qobj)
            # execute
            result = self.executor.execute(query)
            result["query"] = query
            result["need"] = need
            state["query_results"].append(result)

            # route results for relation scoring
            if need.startswith("validate_relation"):
                # extract canonical relation key
                parts = need.split(":")
                if len(parts) > 3:
                    key = f"{parts[1]}-{parts[2]}-{parts[3]}"
                    relation_evidence.setdefault(key, []).append(result)

            # planner expansion
            new_needs = self.planner.expand(need, result, d)
            for n in new_needs:
                if n not in depth_map or depth_map[n] > d + 1:
                    # set depth and enqueue
                    depth_map[n] = d + 1
                    queue.append(n)

            # If synonyms resolved -> requeue validation of relations using synonyms
            if need.startswith("find_synonyms") and result.get("success"):
                for b in result.get("results", []):
                    alt = b.get("altLabel", {}).get("value") or b.get("altLabel")
                    if alt:
                        # For each existing relation, attempt revalidation with alt label substitution
                        for rel in state["extracted_relations"]:
                            new_need = f"validate_relation:{alt}:{rel['predicate']}:{rel['object']}"
                            if new_need not in depth_map:
                                depth_map[new_need] = d + 1
                                queue.append(new_need)

            # stop early if we've got strong evidence for all relations
            # compute preliminary validation scores and decide
            for rel in state["extracted_relations"]:
                key = f"{rel['subject']}-{rel['predicate']}-{rel['object']}"
                if key in relation_evidence:
                    score = compute_validation_score(relation_evidence[key])
                    state["validation_status"][key] = {"score": score, "evidence_count": len(relation_evidence[key])}
            # if all relations have score >= threshold (0.7) we can stop
            if state["validation_status"] and all(v["score"] >= 0.7 for v in state["validation_status"].values()):
                logger.info("All relations validated above threshold, stopping early.")
                break

        # finalize metrics
        state["generated_queries"] = all_query_objs
        successful = sum(1 for r in state["query_results"] if r.get("success"))
        avg_time = np.mean([r.get("execution_time", 0) for r in state["query_results"]]) if state["query_results"] else 0.0
        state["metrics"] = {
            "total_queries": len(state["query_results"]),
            "successful_queries": successful,
            "average_execution_time": float(avg_time)
        }
        state["augmented_context"] = self.augmenter.augment(state)

        return state

# ---------------------------------------------------------
# Evaluation harness similar to your evaluate_sparql_agent
# ---------------------------------------------------------
def evaluate_agent(agent: SPARQLAgent, test_cases: List[Dict]) -> Dict:
    results = []
    for tc in test_cases:
        t0 = time.time()
        final_state = agent.run_for_case(tc)
        total_time = time.time() - t0
        res = {
            "test_case": tc["id"],
            "total_time": total_time,
            "queries_generated": len(final_state["generated_queries"]),
            "queries_successful": final_state["metrics"]["successful_queries"],
            "avg_query_time": final_state["metrics"]["average_execution_time"],
            "context_length": len(final_state["augmented_context"]),
            "validations": final_state["validation_status"]
        }
        logger.info(f"Case {tc['id']}: {res['queries_successful']}/{res['queries_generated']} queries successful")
        results.append(res)

    aggregate = {
        "total_cases": len(results),
        "avg_queries_per_case": float(np.mean([r["queries_generated"] for r in results])) if results else 0.0,
        "avg_total_time": float(np.mean([r["total_time"] for r in results])) if results else 0.0
    }
    return {"individual_results": results, "aggregate_metrics": aggregate}

# ---------------------------------------------------------
# Example usage: keep similar test cases
# ---------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        {
            "id": "test_1",
            "text": "Metformin is used to treat type 2 diabetes by improving insulin sensitivity.",
            "entities": [
                {"text": "Metformin", "type": "CHEMICAL", "ontology_id": "CHEBI:6801"},
                {"text": "type 2 diabetes", "type": "DISEASE", "ontology_id": "MONDO:0005148"},
                {"text": "insulin sensitivity", "type": "PHENOTYPE"}
            ],
            "relations": [
                {
                    "subject": "Metformin",
                    "predicate": "treats",
                    "object": "type 2 diabetes",
                    "confidence": 0.9
                }
            ]
        },
        {
            "id": "test_2",
            "text": "Aspirin may reduce the risk of colorectal cancer through COX-2 inhibition.",
            "entities": [
                {"text": "Aspirin", "type": "CHEMICAL", "ontology_id": "CHEBI:15365"},
                {"text": "colorectal cancer", "type": "DISEASE", "ontology_id": "MONDO:0005575"},
                {"text": "COX-2", "type": "GENE"}
            ],
            "relations": [
                {
                    "subject": "Aspirin",
                    "predicate": "prevents",
                    "object": "colorectal cancer",
                    "confidence": 0.7
                },
                {
                    "subject": "Aspirin",
                    "predicate": "inhibits",
                    "object": "COX-2",
                    "confidence": 0.95
                }
            ]
        }
    ]

    agent = SPARQLAgent(endpoint=DEFAULT_SPARQL_ENDPOINT, model_name="llama3.1:8b")
    print("Starting Ontology-backed SPARQL Agent Evaluation...")
    summary = evaluate_agent(agent, test_cases)
    print(json.dumps(summary["aggregate_metrics"], indent=2))
    for r in summary["individual_results"]:
        print(json.dumps(r, indent=2))
