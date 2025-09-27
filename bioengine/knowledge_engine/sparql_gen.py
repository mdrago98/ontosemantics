"""
Hierarchical LangGraph with Agents + Tools + Streaming
- Agents: Generator, Verifier
- Tools: SPARQL Executor, Ontology Verification
- Supervisor orchestrates reasoning and calls tools
- Streaming shows when each node (agent/tool) is invoked and what it outputs
"""

import json
import re
import requests
import logging
from typing import Dict
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hierarchical_verifier")

# ---------------- Config ----------------
DEFAULT_SPARQL_ENDPOINT = "https://sparql.hegroup.org/sparql/"
VALIDATION_THRESHOLD = 0.7

def sparql_escape(text: str) -> str:
    return re.sub(r'["\'\\\n\r\t]+', ' ', text or "").strip()

# ---------------- SPARQL Generator (Agent) ----------------
class SPARQLGenerator:
    predicate_map = {
        "treats": "http://purl.obolibrary.org/obo/RO_0002606",
        "causes": "http://purl.obolibrary.org/obo/RO_0002410",
    }

    def generate_all(self, subj: str, pred: str, obj: str) -> Dict[str, str]:
        """Return a dict of strategies -> query string."""
        subj, obj = sparql_escape(subj), sparql_escape(obj)
        pred_uri = self.predicate_map.get(pred.lower())

        queries = {}

        # 1. Direct ASK with mapped predicate
        if pred_uri:
            queries["standard"] = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
ASK {{
  ?s rdfs:label ?sl . FILTER(CONTAINS(LCASE(str(?sl)), LCASE("{subj}")))
  ?o rdfs:label ?ol . FILTER(CONTAINS(LCASE(str(?ol)), LCASE("{obj}")))
  ?s <{pred_uri}> ?o .
}}"""

        # 2. Flexible SELECT
        queries["flexible"] = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?s ?p ?o WHERE {{
  ?s rdfs:label ?sl . FILTER(CONTAINS(LCASE(str(?sl)), LCASE("{subj}")))
  ?o rdfs:label ?ol . FILTER(CONTAINS(LCASE(str(?ol)), LCASE("{obj}")))
  ?s ?p ?o .
}} LIMIT 50
"""

        # 3. Synonyms
        queries["synonym_subj"] = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
SELECT ?entity ?label ?syn WHERE {{
  ?entity rdfs:label ?label ;
          oboInOwl:hasExactSynonym ?syn .
  FILTER(CONTAINS(LCASE(str(?syn)), LCASE("{subj}")))
}}
LIMIT 20
"""
        queries["synonym_obj"] = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
SELECT ?entity ?label ?syn WHERE {{
  ?entity rdfs:label ?label ;
          oboInOwl:hasExactSynonym ?syn .
  FILTER(CONTAINS(LCASE(str(?syn)), LCASE("{obj}")))
}}
LIMIT 20
"""

        # 4. Indirect / multi-hop traversal
        queries["indirect"] = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?mid ?p1 ?p2 WHERE {{
  ?s rdfs:label ?sl . FILTER(CONTAINS(LCASE(str(?sl)), LCASE("{subj}")))
  ?o rdfs:label ?ol . FILTER(CONTAINS(LCASE(str(?ol)), LCASE("{obj}")))
  ?s ?p1 ?mid .
  ?mid ?p2 ?o .
}}
LIMIT 50
"""
        return queries
# ---------------- SPARQL Executor (Tool) ----------------
@tool("sparql_executor", return_direct=True)
def sparql_executor(query: str) -> dict:
    """Execute a SPARQL query and return results."""
    try:
        r = requests.post(DEFAULT_SPARQL_ENDPOINT,
                          data={"query": query},
                          headers={"Accept": "application/sparql-results+json"},
                          timeout=30)
        r.raise_for_status()
        data = r.json()
        if "boolean" in data:
            return {"success": True, "is_ask": True, "ask_result": data["boolean"]}
        return {"success": True, "is_ask": False,
                "results": data.get("results", {}).get("bindings", [])}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool("entity_resolver", return_direct=True)
def entity_resolver(term: str) -> dict:
    """Resolve a text term to ontology URIs and labels."""
    query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
SELECT DISTINCT ?entity ?label ?syn WHERE {{
  ?entity rdfs:label ?label .
  OPTIONAL {{ ?entity oboInOwl:hasExactSynonym ?syn }}
  FILTER(CONTAINS(LCASE(str(?label)), LCASE("{sparql_escape(term)}")))
}} LIMIT 20
"""
    return sparql_executor.invoke({"query": query})

# ---------------- Verifier (Agent) ----------------
def verifier(result: dict) -> Dict:
    if not result or not result.get("success"):
        return {"score": 0.0, "reason": "execution failed"}
    if result.get("is_ask") and result.get("ask_result"):
        return {"score": 0.95, "reason": "ASK confirmed"}
    hits = len(result.get("results", []))
    if hits > 0:
        return {"score": min(0.85, 0.4 + hits/20.0), "reason": f"{hits} hits"}
    return {"score": 0.0, "reason": "no evidence"}

# ---------------- Ontology Verification (Tool) ----------------
@tool("ontology_verification", return_direct=True)
def ontology_verification(subject: str, predicate_uri: str, obj: str) -> dict:
    """Check ontology consistency (toy version)."""
    # A real version would fetch domain/range from ontology.
    if subject.lower().startswith("type 2") and predicate_uri.endswith("2606"):
        # Swapped direction case
        return {"direction_ok": False, "swapped": True, "disjoint": False}
    return {"direction_ok": True, "swapped": False, "disjoint": False}

# ---------------- Nodes ----------------
def node_supervisor_plan(state: Dict) -> Dict:
    rels = state.get("extracted_relations", [])
    needs = [f"{r['subject']}:{r['predicate']}:{r['object']}" for r in rels]
    state["queue"] = needs; state["done"] = {n: False for n in needs}
    state["validation_status"] = {}
    return state

def node_supervisor_select(state: Dict) -> Dict:
    while state["queue"] and state["done"][state["queue"][0]]:
        state["queue"].pop(0)
    state["current_need"] = state["queue"][0] if state["queue"] else None
    return state

def node_resolve_entities(state: Dict) -> Dict:
    if not state.get("current_need"): return state
    subj, pred, obj = state["current_need"].split(":")
    subj_res = entity_resolver.invoke({"term": subj})
    obj_res  = entity_resolver.invoke({"term": obj})
    state.setdefault("entity_mappings", {})[subj] = subj_res
    state["entity_mappings"][obj] = obj_res
    return state

def node_generator(state: Dict) -> Dict:
    if not state.get("current_need"): return state
    subj, pred, obj = state["current_need"].split(":")
    queries = SPARQLGenerator().generate_all(subj, pred, obj)
    state["current_queries"] = queries
    return state

def node_executor(state: Dict) -> Dict:
    if not state.get("current_queries"): return state
    results = {}
    for strat, q in state["current_queries"].items():
        res = sparql_executor.invoke({"query": q})
        results[strat] = res
    state["last_results"] = results
    return state

def node_verifier(state: Dict) -> Dict:
    need = state.get("current_need")
    results = state.get("last_results", {})
    scores, reasons = [], []
    inferred = []

    for strat, res in results.items():
        v = verifier(res)
        scores.append(v["score"])
        reasons.append(f"{strat}: {v['reason']}")

        # Collect inferred triples if indirect query had hits
        if strat == "indirect" and res.get("results"):
            subj, pred, obj = need.split(":")
            for b in res["results"]:
                mid = b.get("mid", {}).get("value")
                p1 = b.get("p1", {}).get("value")
                p2 = b.get("p2", {}).get("value")
                if mid and p1 and p2:
                    inferred.append({
                        "subject": subj,
                        "predicate": f"{p1}+{p2}",  # composite predicate
                        "object": obj,
                        "via": mid
                    })

    state["validation_status"][need] = {
        "score": max(scores) if scores else 0.0,
        "reason": " | ".join(reasons)
    }
    state.setdefault("inferred_relations", []).extend(inferred)
    return state

def node_ontology(state: Dict) -> Dict:
    need = state.get("current_need")
    if not need: return state
    subj, pred, obj = need.split(":")
    pred_uri = SPARQLGenerator.predicate_map.get(pred.lower(), "")
    if pred_uri:
        res = ontology_verification.invoke({"subject": subj,
                                            "predicate_uri": pred_uri,
                                            "obj": obj})
        ctx = state["validation_status"][need]
        if res.get("swapped") or res.get("disjoint"):
            ctx["score"] = 0.0; ctx["reason"] += " | ontology contradiction"
        elif res.get("direction_ok"):
            ctx["score"] = max(ctx["score"], 0.75); ctx["reason"] += " | ontology consistent"
        state["validation_status"][need] = ctx
    return state

def node_decide(state: Dict) -> Dict:
    need = state.get("current_need")
    if not need: return state

    state["done"][need] = True  # mark current need as processed

    # Add inferred relations to queue
    new_relations = state.pop("inferred_relations", [])
    for rel in new_relations:
        new_need = f"{rel['subject']}:{rel['predicate']}:{rel['object']}"
        if new_need not in state["done"]:  # prevent duplicates
            state["queue"].append(new_need)
            state["done"][new_need] = False
            logger.info(f"Supervisor inferred new need: {new_need}")

    return state


def route_after_decide(state: Dict) -> str:
    if all(state["done"].values()): return "end"
    return "continue"

memory = MemorySaver()
# ---------------- Graph ----------------
graph = StateGraph(dict)
graph.add_node("plan", node_supervisor_plan)
graph.add_node("select", node_supervisor_select)
graph.add_node('entity_resolution', node_resolve_entities)
graph.add_node("generator", node_generator)
graph.add_node("executor", node_executor)
graph.add_node("verifier", node_verifier)
graph.add_node("ontology", node_ontology)
graph.add_node("decide", node_decide)

graph.add_edge("plan","select")
graph.add_edge("select","entity_resolution")
graph.add_edge("entity_resolution","generator")
graph.add_edge("generator","executor")
graph.add_edge("executor","verifier")
graph.add_edge("verifier","ontology")
graph.add_edge("ontology","decide")
graph.add_conditional_edges("decide", route_after_decide,
                            {"end": END,"continue":"select"})
graph.set_entry_point("plan")
app = graph.compile()

# ---------------- Runner with Streaming ----------------
class HierarchicalVerifier:
    def run_case_stream(self, case: Dict):
        init = {
            "original_text": case["text"],
            "extracted_entities": case["entities"],
            "extracted_relations": case["relations"],
        }
        final = app.invoke(init, config={"recursion_limit": 100})
        return final

# ---------------- Example ----------------
if __name__=="__main__":
    app.get_graph().print_ascii()
    hv = HierarchicalVerifier()
    bad = {
        "text": "Type 2 diabetes treats Metformin (nonsense).",
        "entities": [{"text":"type 2 diabetes"},{"text":"Metformin"}],
        "relations":[{"subject":"type 2 diabetes","predicate":"treats","object":"Metformin"}]
    }
    good = {
        "id": "metformin_correct",
        "text": "Metformin treats type 2 diabetes.",
        "entities": [{"text": "Metformin"}, {"text": "type 2 diabetes"}],
        "relations": [
            {"subject": "Metformin", "predicate": "treats", "object": "type 2 diabetes"}
        ],
    }


    for case in [good, bad]:
        print(f"\n=== Running {case} ===")
        out = hv.run_case_stream(case)
        print("\nFinal validation:", json.dumps(out["validation_status"], indent=2))
        print("\nResolved Entities: ", json.dumps(out['entity_mappings']))
        print("\nNew inferred relations discovered:")
        for rel in out.get("inferred_relations", []):
            print(rel)
