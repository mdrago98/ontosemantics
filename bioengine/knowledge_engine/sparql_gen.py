"""
Hierarchical LangGraph: Supervisor + Leaf Agents
- Supervisor plans and controls retry/strategy logic
- Leaf agents: Generator, Executor, Verifier, Augmenter
- Benchmarked evaluation harness at the bottom
"""

import time
import json
import re
import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import requests
import numpy as np

# LLM + LangGraph imports (adapt names if your environment differs)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ------------ Logging ------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hierarchical_verifier")

# ------------ Config ------------
DEFAULT_SPARQL_ENDPOINTS = [
    "https://sparql.hegroup.org/sparql/"  # fallback; replace with your endpoints
    # "https://sparql.uniprot.org/sparql"  # fallback; replace with your endpoints
]
LLM_MODEL = "gemma3:1b"  # replace if needed
VALIDATION_THRESHOLD = 0.7
MAX_SUPERVISOR_ITER = 30
MAX_STRATEGY_ATTEMPTS = 3   # attempts per strategy before moving on
STRATEGIES = ["standard", "synonym", "broad", "indirect"]
TIMEOUT = 30

# ------------ Utilities ------------
def sparql_escape(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'["\'\\\n\r\t]+', ' ', text).strip()

def now_ts() -> float:
    return time.time()

# ------------ SPARQL Templates ------------
SPARQL_TEMPLATES = {
    "validate_relation_standard": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
ASK WHERE {{
  ?s rdfs:label ?sl .
  FILTER(CONTAINS(LCASE(str(?sl)), LCASE("{sub}")))
  ?o rdfs:label ?ol .
  FILTER(CONTAINS(LCASE(str(?ol)), LCASE("{obj}")))
  ?s {pred} ?o .
}}
""",
    "validate_relation_flexible": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?s ?p ?o ?sl ?ol WHERE {{
  ?s rdfs:label ?sl .
  FILTER(CONTAINS(LCASE(str(?sl)), LCASE("{sub}")))
  ?o rdfs:label ?ol .
  FILTER(CONTAINS(LCASE(str(?ol)), LCASE("{obj}")))
  ?s ?p ?o .
  OPTIONAL {{ ?p rdfs:label ?pl . }}
}}
LIMIT 50
""",
    "find_synonyms": """
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?entity ?label ?syn WHERE {{
  ?entity rdfs:label ?label .
  ?entity oboInOwl:hasExactSynonym ?syn .
  FILTER(CONTAINS(LCASE(str(?syn)), LCASE("{term}")))
}}
LIMIT 20
""",
    "find_indirect": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?p1 ?mid ?p2 ?midLabel WHERE {{
  ?s rdfs:label ?sl .
  FILTER(CONTAINS(LCASE(str(?sl)), LCASE("{sub}")))
  ?o rdfs:label ?ol .
  FILTER(CONTAINS(LCASE(str(?ol)), LCASE("{obj}")))
  ?s ?p1 ?mid .
  ?mid ?p2 ?o .
  OPTIONAL {{ ?mid rdfs:label ?midLabel }}
}}
LIMIT 20
"""
}

# ------------ SPARQL Executor (robust) ------------
class SPARQLExecutor:
    def __init__(self, endpoints: List[str] = None, max_retries: int = 2, timeout: int = TIMEOUT):
        self.endpoints = endpoints or DEFAULT_SPARQL_ENDPOINTS
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/sparql-results+json"})
        self.cache: Dict[int, Dict] = {}

    def execute(self, query: str) -> Dict[str, Any]:
        key = hash(query)
        if key in self.cache:
            out = self.cache[key].copy()
            out["from_cache"] = True
            return out

        start = now_ts()
        last_err = None
        # Try endpoints in order
        for endpoint in self.endpoints:
            attempt = 0
            while attempt <= self.max_retries:
                try:
                    r = self.session.post(endpoint, data={"query": query}, timeout=self.timeout)
                    if r.status_code != 200:
                        last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                        attempt += 1
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    data = r.json()
                    # ASK case
                    if "boolean" in data:
                        ask = bool(data["boolean"])
                        res = {
                            "success": True,
                            "endpoint": endpoint,
                            "execution_time": now_ts() - start,
                            "is_ask": True,
                            "ask_result": ask,
                            "count": 1 if ask else 0,
                            "results": [],
                            "query": query,
                            "has_evidence": ask,
                            "empty": not ask,
                            "raw": data
                        }
                        self.cache[key] = res
                        return res
                    # SELECT case
                    bindings = data.get("results", {}).get("bindings", [])
                    has_evidence = len(bindings) > 0
                    res = {
                        "success": True,
                        "endpoint": endpoint,
                        "execution_time": now_ts() - start,
                        "is_ask": False,
                        "results": bindings,
                        "count": len(bindings),
                        "query": query,
                        "has_evidence": has_evidence,
                        "empty": not has_evidence,
                        "raw": data
                    }
                    self.cache[key] = res
                    return res
                except requests.exceptions.Timeout:
                    last_err = "timeout"
                    attempt += 1
                    time.sleep(0.2 * (2 ** attempt))
                except Exception as e:
                    last_err = str(e)
                    attempt += 1
                    time.sleep(0.2 * (2 ** attempt))
        # If here, failed all endpoints/attempts
        return {
            "success": False,
            "error": last_err or "Unknown endpoint or network error",
            "execution_time": now_ts() - start,
            "query": query
        }

# ------------ Generator Agent ------------
class SPARQLGenerator:
    def __init__(self, llm_model: str = LLM_MODEL, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = None

        if use_llm:
            try:
                self.llm = ChatOllama(model=llm_model, temperature=0, timeout=30)
                logger.info("LLM initialized successfully")
            except Exception as e:
                logger.warning(
                    f"LLM init failed: {e}; generator will use templates only"
                )
                self.llm = None
                self.use_llm = False

        # Predicate mappings
        self.predicate_map = {
            "treats": "obo:RO_0002606",
            "causes": "obo:RO_0002410",
            "prevents": "obo:RO_0002559",
            "inhibits": "obo:RO_0002449",
            "activates": "obo:RO_0002448",
            "improves": "obo:RO_0002252",
            "associated with": "obo:RO_0002436",
            "interacts with": "obo:RO_0002434",
        }

    def generate(
        self,
        need: str,
        context: Dict,
        strategy: str = "standard",
        previous_failures: List[Dict] = None,
    ) -> str:
        """
        Generate SPARQL query based on need and strategy.

        Args:
            need: Information need (e.g., "validate_relation:SUB:pred:OBJ")
            context: Context with entities and relations
            strategy: Query strategy ('standard', 'synonym', 'broad', 'indirect')
            previous_failures: List of previous failed attempts

        Returns:
            SPARQL query string
        """
        # Prevent None previous_failures from causing issues
        if previous_failures is None:
            previous_failures = []

        # Parse the need
        parts = need.split(":")
        need_type = parts[0] if parts else ""

        logger.debug(f"Generating query for need: {need}, strategy: {strategy}")

        try:
            # Handle different need types
            if need_type == "validate_relation" and len(parts) >= 4:
                return self._generate_relation_query(
                    parts[1], parts[2], parts[3], strategy
                )

            elif need_type == "identify_entity" and len(parts) > 1:
                return self._generate_entity_query(parts[1], strategy)

            elif need_type == "explore" and len(parts) > 1:
                return SPARQL_TEMPLATES["explore_relations"].format(
                    term=sparql_escape(parts[1])
                )

            # If we have an unrecognized need type, try LLM or fallback
            else:
                return self._generate_with_llm_or_fallback(
                    need, context, strategy, previous_failures
                )

        except Exception as e:
            logger.error(f"Error generating query: {e}")
            # Return a safe fallback query
            return self._get_safe_fallback()

    def _generate_relation_query(
        self, subject: str, predicate: str, obj: str, strategy: str
    ) -> str:
        """Generate query for relation validation"""
        sub_escaped = sparql_escape(subject)
        obj_escaped = sparql_escape(obj)

        if strategy == "standard":
            # Try to map predicate to known URI
            pred_lower = predicate.lower()
            if pred_lower in self.predicate_map:
                pred_uri = self.predicate_map[pred_lower]
                # Fixed template format - using PREFIX properly
                query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
ASK WHERE {{
  ?subject rdfs:label ?sub_label .
  ?object rdfs:label ?obj_label .
  FILTER(CONTAINS(LCASE(str(?sub_label)), LCASE("{sub_escaped}")))
  FILTER(CONTAINS(LCASE(str(?obj_label)), LCASE("{obj_escaped}")))
  ?subject {pred_uri} ?object .
}}"""
                return query
            else:
                # Unknown predicate - use flexible query
                return SPARQL_TEMPLATES["validate_relation_flexible"].format(
                    sub=sub_escaped, obj=obj_escaped
                )

        elif strategy == "synonym":
            # Look for synonyms of subject
            return SPARQL_TEMPLATES["find_synonyms"].format(term=sub_escaped)

        elif strategy == "broad":
            # Broad flexible search
            return SPARQL_TEMPLATES["validate_relation_flexible"].format(
                sub=sub_escaped, obj=obj_escaped
            )

        elif strategy == "indirect":
            # Look for indirect relations
            return SPARQL_TEMPLATES["find_indirect"].format(
                sub=sub_escaped, obj=obj_escaped
            )

        else:
            # Unknown strategy - use flexible
            return SPARQL_TEMPLATES["validate_relation_flexible"].format(
                sub=sub_escaped, obj=obj_escaped
            )

    def _generate_entity_query(self, entity: str, strategy: str) -> str:
        """Generate query for entity identification"""
        entity_escaped = sparql_escape(entity)

        if strategy == "synonym":
            return SPARQL_TEMPLATES["find_synonyms"].format(term=entity_escaped)
        else:
            return SPARQL_TEMPLATES["identify_entity"].format(term=entity_escaped)

    def _generate_with_llm_or_fallback(
        self, need: str, context: Dict, strategy: str, previous_failures: List[Dict]
    ) -> str:
        """Try LLM generation or return fallback"""
        if self.use_llm and self.llm:
            try:
                prompt = self._build_prompt(need, context, previous_failures, strategy)
                response = self.llm.invoke([SystemMessage(content=prompt)])
                content = (
                    response.content if hasattr(response, "content") else str(response)
                )
                return self._sanitize_sparql_from_llm(content)
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")

        return self._get_safe_fallback()

    def _build_prompt(
        self, need: str, context: Dict, failures: List[Dict], strategy: str
    ) -> str:
        """Build LLM prompt"""
        entities_text = ", ".join(
            [
                e.get("text", "") if isinstance(e, dict) else str(e)
                for e in context.get("entities", [])[:5]
            ]
        )

        return f"""You are a SPARQL expert for biomedical ontologies.

Generate a SPARQL query for this need: {need}
Strategy: {strategy}
Context entities: {entities_text}
Previous failures: {len(failures)}

Requirements:
- Use FILTER(CONTAINS(LCASE(str(?label)), LCASE("term"))) for flexible matching
- Use proper PREFIX declarations
- Return ONLY the SPARQL query, no explanations

Query:"""

    def _sanitize_sparql_from_llm(self, content: str) -> str:
        """Clean and validate SPARQL from LLM"""
        # Remove markdown code blocks
        content = re.sub(r"```(?:sparql)?\n?", "", str(content))
        content = content.replace("```", "").strip()

        # Check if it looks like SPARQL
        if any(
            keyword in content.upper() for keyword in ["SELECT", "ASK", "CONSTRUCT"]
        ):
            return content

        # Fallback if not valid SPARQL
        return self._get_safe_fallback()

    def _get_safe_fallback(self) -> str:
        """Return a safe fallback query that won't cause errors"""
        return """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?s ?p ?o WHERE {
  ?s ?p ?o .
  ?s rdfs:label ?label .
}
LIMIT 10
"""
# ------------ Verifier Agent ------------
def compute_validation_score(results: List[Dict]) -> Tuple[float, str]:
    """
    Combine signals:
     - ASK True -> high score
     - count of bindings -> scaled score
     - indirect evidence -> moderate score
    """
    if not results:
        return 0.0, "no evidence"
    score = 0.0
    reasons = []
    total = sum(r.get("count", 0) for r in results if r.get("count"))
    for r in results:
        if r.get("is_ask") and r.get("ask_result"):
            score = max(score, 0.95)
            reasons.append("ASK confirmed")
        elif r.get("has_evidence"):
            cnt = r.get("count", 0)
            score = max(score, min(0.85, 0.4 + min(0.45, cnt / 10.0)))
            reasons.append(f"{cnt} hits")
        elif r.get("empty"):
            reasons.append("empty")
    reason_text = " | ".join(reasons[:3]) if reasons else "no evidence"
    return min(1.0, score), reason_text

# ------------ Augmenter Agent ------------
class ContextAugmenter:
    def augment(self, state: Dict) -> str:
        parts = []
        parts.append(f"Original: {state.get('original_text','')}")
        parts.append("\n--- Evidence ---")
        for q in state.get("all_queries", []):
            if q.get("has_evidence"):
                parts.append(f"Query: {q.get('query')[:200]} | count={q.get('count',0)} | endpoint={q.get('endpoint')}")
        parts.append("\n--- Validation ---")
        for k, v in state.get("validation_status", {}).items():
            parts.append(f"{k}: score={v.get('score'):.2f} | reason={v.get('reason')}")
        return "\n".join(parts)

# ------------ Supervisor logic & LangGraph nodes ------------
# State dict structure will be simple dict; LangGraph will operate on it.

def node_supervisor_plan(state: Dict) -> Dict:
    """
    Supervisor plans the information needs (sparse templates)
    and initializes bookkeeping.
    """
    # We expect state to contain:
    # - 'extracted_relations': list of {subject,predicate,object}
    # - 'extracted_entities'
    relations = state.get("extracted_relations", [])
    needs = []
    for rel in relations:
        subj = rel.get("subject")
        pred = rel.get("predicate", "related_to")
        obj = rel.get("object")
        needs.append(f"validate_relation:{subj}:{pred}:{obj}")

    # initialize supervisor meta
    state.setdefault("supervisor", {})
    sup = state["supervisor"]
    sup.setdefault("needs_queue", needs.copy())
    sup.setdefault("strategy_index", {n: 0 for n in needs})   # index into STRATEGIES
    sup.setdefault("attempts", {n: 0 for n in needs})         # total attempts
    sup.setdefault("failures", {n: [] for n in needs})       # store failed query dicts
    sup.setdefault("done", {n: False for n in needs})
    state["all_queries"] = []
    state["iteration"] = 0
    state["metrics"] = {"queries": 0, "successful": 0}
    state.setdefault("validation_status", {})
    state.setdefault("relation_evidence", {})
    state.setdefault("entity_mappings", {})
    state.setdefault("reasoning_trace", [])
    state.setdefault("messages", [])
    logger.info("Supervisor planned %d needs", len(needs))
    state["messages"].append(HumanMessage(content=f"Supervisor planned {len(needs)} needs"))
    return state

def node_supervisor_select_need(state: Dict) -> Dict:
    """Supervisor chooses the next need to delegate. Leaves it in supervisor.needs_queue[0]"""
    sup = state["supervisor"]
    queue = sup.get("needs_queue", [])
    # pop finished
    while queue and sup["done"].get(queue[0], False):
        queue.pop(0)
    if not queue:
        state["current_need"] = None
        return state
    state["current_need"] = queue[0]
    # set strategy for this need based on attempts so far
    idx = sup["strategy_index"].get(state["current_need"], 0)
    strategy = STRATEGIES[min(idx, len(STRATEGIES)-1)]
    state["current_strategy"] = strategy
    logger.debug("Supervisor selected need %s with strategy %s", state["current_need"], strategy)
    return state

def node_generator(state: Dict) -> Dict:
    """Leaf: generate SPARQL query for current_need and strategy"""
    need = state.get("current_need")
    if not need:
        state["current_query"] = None
        return state
    gen = SPARQLGenerator()
    sup = state["supervisor"]
    prev_failures = sup.get("failures", {}).get(need, [])
    strategy = state.get("current_strategy", "standard")
    ctx = {"entities": state.get("extracted_entities", []), "relations": state.get("extracted_relations", [])}
    query = gen.generate(need, ctx, strategy=strategy, previous_failures=prev_failures)
    # logger.debug('Generated Query: ', query)
    state["current_query"] = query
    state.setdefault("messages", []).append(AIMessage(content=f"Generated query for {need} with strategy {strategy}"))
    return state

def node_executor(state: Dict) -> Dict:
    """Leaf: execute the current_query and attach result to state['last_query_result']"""
    query = state.get("current_query")
    need = state.get("current_need")
    if not query or not need:
        state["last_query_result"] = None
        return state
    executor = SPARQLExecutor()
    result = executor.execute(query)
    result["query"] = query
    result["need"] = need
    result["executed_at"] = now_ts()
    state.setdefault("all_queries", []).append(result)
    state.setdefault("metrics", {}).setdefault("queries", 0)
    state["metrics"]["queries"] += 1
    state["last_query_result"] = result
    state.setdefault("messages", []).append(HumanMessage(content=f"Executed query for {need}: success={result.get('success')} has_evidence={result.get('has_evidence', False)}"))
    return state

def node_verifier(state: Dict) -> Dict:
    """Process last_query_result; handle indirect/multihop results."""
    res = state.get("last_query_result")
    if not res:
        return state
    need = res.get("need")
    sup = state["supervisor"]

    if not res.get("success"):
        sup["failures"].setdefault(need, []).append(res)
        sup["attempts"][need] = sup["attempts"].get(need, 0) + 1
        state.setdefault("messages", []).append(
            HumanMessage(content=f"Execution error for {need}: {res.get('error')}")
        )
        return state

    has_evidence = False
    # Detect indirect evidence
    if not res.get("has_evidence") and not res.get("is_ask"):
        # Check if bindings contain expected subject and object through intermediates
        bindings = res.get("results", [])
        if bindings:
            has_evidence = True
            for b in bindings:
                # optionally normalize and record intermediate nodes
                state.setdefault("relation_evidence", {}).setdefault(need, []).append(b)
    else:
        has_evidence = res.get("has_evidence")

    if has_evidence:
        key = need
        state.setdefault("relation_evidence", {}).setdefault(key, []).append(res)
        score, reason = compute_validation_score(state["relation_evidence"][key])
        state.setdefault("validation_status", {})[key] = {
            "score": score,
            "reason": reason,
            "evidence_count": len(state["relation_evidence"][key]),
        }
        state.setdefault("messages", []).append(
            HumanMessage(content=f"Verifier found evidence for {key}, score={score:.2f}")
        )
        state["metrics"]["successful"] = state["metrics"].get("successful", 0) + 1
    else:
        sup["failures"].setdefault(need, []).append(res)
        sup["attempts"][need] = sup["attempts"].get(need, 0) + 1
        state.setdefault("messages", []).append(
            HumanMessage(content=f"No evidence for {need} (empty result)")
        )
    return state

def node_augment(state: Dict) -> Dict:
    """Leaf: augment context with evidence summary; used by supervisor decision"""
    augmenter = ContextAugmenter()
    ctx = augmenter.augment(state)
    state["augmented_context"] = ctx
    state.setdefault("messages", []).append(AIMessage(content="Context augmented"))
    return state

def node_supervisor_decide(state: Dict) -> Dict:
    sup = state["supervisor"]
    need = state.get("current_need")

    if not need:
        return state

    val = state.get("validation_status", {}).get(need)

    if val and val.get("score", 0) >= VALIDATION_THRESHOLD:
        sup["done"][need] = True
        state.setdefault("messages", []).append(
            HumanMessage(content=f"Need {need} validated (score={val.get('score'):.2f})")
        )

    else:
        attempts = sup["attempts"].get(need, 0)
        if attempts >= MAX_STRATEGY_ATTEMPTS:
            idx = sup["strategy_index"].get(need, 0)
            if idx + 1 < len(STRATEGIES):
                sup["strategy_index"][need] = idx + 1
                sup["attempts"][need] = 0
                state.setdefault("messages", []).append(
                    HumanMessage(content=f"Supervisor: switching strategy for {need} to {STRATEGIES[idx+1]}")
                )
            else:
                sup["done"][need] = True
                state.setdefault("messages", []).append(
                    HumanMessage(content=f"Supervisor: exhausted strategies for {need}, marking done (failed)")
                )

    return state


# ------------ LangGraph wiring ------------
graph = StateGraph(dict)

# Supervisor plans
graph.add_node("supervisor_plan", node_supervisor_plan)
graph.add_node("supervisor_select", node_supervisor_select_need)

# Generators & leaves
graph.add_node("generator", node_generator)
graph.add_node("executor", node_executor)
graph.add_node("verifier", node_verifier)
graph.add_node("augmenter", node_augment)

# Supervisor decision
graph.add_node("supervisor_decide", node_supervisor_decide)

# Edges: plan -> select -> generator -> executor -> verifier -> augmenter -> decide -> select ...
graph.add_edge("supervisor_plan", "supervisor_select")
graph.add_edge("supervisor_select", "generator")
graph.add_edge("generator", "executor")
graph.add_edge("executor", "verifier")
graph.add_edge("verifier", "augmenter")
graph.add_edge("augmenter", "supervisor_decide")
# graph.add_edge("supervisor_decide", "supervisor_select")
# termination: when no needs left or all done, go to END: we'll let supervisor_select set current_need to None and finish
# graph.add_edge("supervisor_select", END)  # if current_need None -> END
# graph.add_conditional_edges(
#     "supervisor_decide",
#     lambda state: "end" if all(state["supervisor"]["done"].values()) else "continue",
#     {
#         "end": END,
#         "continue": "supervisor_select"
#     },
# )

def route_after_decide(state: Dict) -> str:
    sup = state["supervisor"]
    # If all needs marked done -> terminate
    if all(sup["done"].values()):
        return "end"
    return "continue"

graph.add_conditional_edges(
    "supervisor_decide",
    route_after_decide,
    {
        "end": END,
        "continue": "supervisor_select",
    },
)
graph.set_entry_point("supervisor_plan")
app = graph.compile()
print(app.get_graph().print_ascii())

# ------------ Wrapper Agent and benchmark harness ------------
class HierarchicalVerifier:
    def __init__(self, endpoints: List[str] = None):
        self.endpoints = endpoints or DEFAULT_SPARQL_ENDPOINTS

    def run_case(self, case: Dict, max_iterations: int = MAX_SUPERVISOR_ITER) -> Dict:
        initial_state = {
            "original_text": case.get("text", ""),
            "extracted_entities": case.get("entities", []),
            "extracted_relations": case.get("relations", []),
            # supervisor fields will be initialized by node
        }
        # attach endpoints to executor via global config? We'll rely on SPARQLExecutor default; you can inject if needed
        t0 = now_ts()
        final_state = app.invoke(initial_state, {"recursion_limit": 100})
        elapsed = now_ts() - t0

        # compute final stats
        queries = final_state.get("all_queries", [])
        validations = final_state.get("validation_status", {})
        succeeded = sum(1 for v in validations.values() if v.get("score",0) >= VALIDATION_THRESHOLD)
        total = len(final_state.get("extracted_relations", []))
        return {
            "final_state": final_state,
            "duration": elapsed,
            "queries_executed": len(queries),
            "relations_validated": succeeded,
            "relations_total": total,
            "validation_status": validations
        }

# ------------ Example test cases and run ------------
if __name__ == "__main__":
    verifier = HierarchicalVerifier()

    sample_cases = [
        {
            "id": "case_metformin",
            "text": "Metformin is used to treat type 2 diabetes by improving insulin sensitivity.",
            "entities": [{"text":"Metformin"},{"text":"type 2 diabetes"},{"text":"insulin sensitivity"}],
            "relations": [{"subject":"Metformin","predicate":"treats","object":"type 2 diabetes"}]
        },
        # {
        #     "id": "case_aspirin",
        #     "text": "Aspirin reduces colorectal cancer risk through COX-2 inhibition.",
        #     "entities": [{"text":"Aspirin"},{"text":"colorectal cancer"},{"text":"COX-2"}],
        #     "relations": [{"subject":"Aspirin","predicate":"prevents","object":"colorectal cancer"}]
        # }
    ]

    results = []
    for case in sample_cases:
        print(f"\n=== Running case {case['id']} ===")
        out = verifier.run_case(case)
        final = out["final_state"]
        print(f"Duration: {out['duration']:.2f}s | Queries: {out['queries_executed']} | Validated: {out['relations_validated']}/{out['relations_total']}")
        print("Validation status:")
        print(json.dumps(out["validation_status"], indent=2))
        print("\nReasoning trace (sample):")
        rt = final.get("reasoning_trace", [])[:20]
        print("\n".join(rt))
        # optionally print messages
        msgs = final.get("messages", [])
        for m in msgs[-6:]:
            try:
                content = m.content
            except Exception:
                content = str(m)
            print("  MSG:", content)
        results.append(out)

    # Basic benchmark summary
    total_queries = sum(r["queries_executed"] for r in results)
    total_time = sum(r["duration"] for r in results)
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Cases: {len(results)} | Total queries: {total_queries} | Total time: {total_time:.2f}s")
