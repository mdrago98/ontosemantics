"""
SPARQL Query Agent using LangGraph for OntoSemantics-LLM
Focused on SPARQL generation and ontology querying for preliminary evaluation
"""

import json
import time
import re
from typing import Dict, List, TypedDict, Annotated, Optional
import numpy as np

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pyoxigraph import Store
import requests

# ============= State Definition =============

class SPARQLAgentState(TypedDict):
    """State for SPARQL query generation and execution"""
    # Input
    extracted_entities: List[Dict]  # Entities from your extraction tool
    extracted_relations: List[Dict]  # Relations from your extraction tool
    original_text: str

    # Processing
    information_needs: List[str]
    generated_queries: List[Dict]  # {need: str, query: str, priority: int}
    query_results: List[Dict]  # {query: str, results: Any, execution_time: float}

    # Output
    augmented_context: str
    validation_status: Dict

    # Metrics
    messages: Annotated[List[BaseMessage], add_messages]
    metrics: Dict

# ============= SPARQL Templates =============

SPARQL_TEMPLATES = {
    "validate_relation": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX chebi: <http://purl.obolibrary.org/obo/CHEBI_>
PREFIX mondo: <http://purl.obolibrary.org/obo/MONDO_>

ASK WHERE {{
    ?subject rdfs:label "{subject_label}" .
    ?object rdfs:label "{object_label}" .
    ?subject {predicate} ?object .
}}
""",

    "get_entity_type": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX obo: <http://purl.obolibrary.org/obo/>

SELECT ?type ?typeLabel WHERE {{
    ?entity rdfs:label "{entity_label}" .
    ?entity rdf:type ?type .
    ?type rdfs:label ?typeLabel .
}}
LIMIT 5
""",

    "find_relations": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX obo: <http://purl.obolibrary.org/obo/>

SELECT ?predicate ?object ?objectLabel WHERE {{
    ?subject rdfs:label "{subject_label}" .
    ?subject ?predicate ?object .
    ?object rdfs:label ?objectLabel .
    FILTER(!isBlank(?object))
}}
LIMIT 20
""",

    "get_hierarchy": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX obo: <http://purl.obolibrary.org/obo/>

SELECT ?parent ?parentLabel WHERE {{
    ?entity rdfs:label "{entity_label}" .
    ?entity rdfs:subClassOf+ ?parent .
    ?parent rdfs:label ?parentLabel .
}}
LIMIT 10
""",

    "find_mechanism": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX go: <http://purl.obolibrary.org/obo/GO_>

SELECT ?mechanism ?mechanismLabel WHERE {{
    ?drug rdfs:label "{drug_label}" .
    ?drug obo:RO_0000087 ?role .  # has role
    ?role obo:RO_0000052 ?mechanism .  # inheres in
    ?mechanism rdfs:label ?mechanismLabel .
}}
LIMIT 5
"""
}

# ============= SPARQL Query Generator Agent =============

class SPARQLGenerator:
    """Generates SPARQL queries based on extraction context"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0,
            # other params...
        )
        self.template_matcher = TemplateMatcher()

    def analyze_information_needs(self, state: SPARQLAgentState) -> List[str]:
        """Determine what information is needed based on extracted content"""

        needs = []

        # Analyze extracted entities
        for entity in state["extracted_entities"]:
            # Need to determine entity type if not clear
            if entity.get("type") == "UNKNOWN" or not entity.get("ontology_id"):
                needs.append(f"identify_type:{entity['text']}")

            # Need to get hierarchy for diseases
            if entity.get("type") in ["DISEASE", "DISORDER"]:
                needs.append(f"get_hierarchy:{entity['text']}")

        # Analyze extracted relations
        for relation in state["extracted_relations"]:
            # Validate the relation exists in ontology
            needs.append(f"validate_relation:{relation['subject']}:{relation['predicate']}:{relation['object']}")

            # If drug-disease relation, get mechanism
            if self._is_drug_disease_relation(relation):
                needs.append(f"find_mechanism:{relation['subject']}")

        # Check for ambiguous relationships
        if self._has_ambiguous_relations(state["extracted_relations"]):
            needs.append("disambiguate_relations")

        return needs

    def generate_sparql(self, need: str, context: Dict) -> str:
        """Generate SPARQL query for a specific information need"""

        # Try template matching first
        template_query = self.template_matcher.match(need, context)
        if template_query:
            return template_query

        # Use LLM for complex queries
        prompt = self._create_generation_prompt(need, context)
        response = self.llm.invoke([SystemMessage(content=prompt)])

        # Extract SPARQL from response
        query = self._extract_sparql_from_response(response.content)
        return query

    def _create_generation_prompt(self, need: str, context: Dict) -> str:
        """Create prompt for SPARQL generation"""

        return f"""You are a SPARQL expert for biomedical ontologies.

Available ontologies:
- MONDO (diseases): PREFIX mondo: <http://purl.obolibrary.org/obo/MONDO_>
- CHEBI (chemicals): PREFIX chebi: <http://purl.obolibrary.org/obo/CHEBI_>
- GO (gene ontology): PREFIX go: <http://purl.obolibrary.org/obo/GO_>

Information need: {need}
Context: {json.dumps(context, indent=2)}

Generate a SPARQL query to address this information need.
Return ONLY the SPARQL query, no explanation.
"""

    def _extract_sparql_from_response(self, response: str) -> str:
        """Extract SPARQL query from LLM response"""
        # Remove markdown code blocks if present
        response = re.sub(r'```sparql?\n?', '', response)
        response = re.sub(r'```', '', response)
        return response.strip()

    def _is_drug_disease_relation(self, relation: Dict) -> bool:
        """Check if relation is between drug and disease"""
        subj_type = relation.get("subject_type", "")
        obj_type = relation.get("object_type", "")
        return ("DRUG" in subj_type or "CHEMICAL" in subj_type) and \
            ("DISEASE" in obj_type or "DISORDER" in obj_type)

    def _has_ambiguous_relations(self, relations: List[Dict]) -> bool:
        """Check if there are ambiguous relations needing clarification"""
        for rel in relations:
            if rel.get("confidence", 1.0) < 0.5 or rel.get("predicate") == "UNKNOWN":
                return True
        return False

# ============= Template Matcher =============

class TemplateMatcher:
    """Matches information needs to SPARQL templates"""

    def match(self, need: str, context: Dict) -> Optional[str]:
        """Try to match need to a template"""

        parts = need.split(":")
        need_type = parts[0] if parts else ""

        if need_type == "identify_type" and len(parts) > 1:
            entity = parts[1]
            return SPARQL_TEMPLATES["get_entity_type"].format(entity_label=entity)

        elif need_type == "get_hierarchy" and len(parts) > 1:
            entity = parts[1]
            return SPARQL_TEMPLATES["get_hierarchy"].format(entity_label=entity)

        elif need_type == "validate_relation" and len(parts) > 3:
            subject, predicate, obj = parts[1], parts[2], parts[3]
            # Convert predicate to ontology format
            predicate_uri = self._convert_predicate(predicate)
            return SPARQL_TEMPLATES["validate_relation"].format(
                subject_label=subject,
                predicate=predicate_uri,
                object_label=obj
            )

        elif need_type == "find_mechanism" and len(parts) > 1:
            drug = parts[1]
            return SPARQL_TEMPLATES["find_mechanism"].format(drug_label=drug)

        elif need_type == "find_relations" and len(parts) > 1:
            entity = parts[1]
            return SPARQL_TEMPLATES["find_relations"].format(subject_label=entity)

        return None

    def _convert_predicate(self, predicate: str) -> str:
        """Convert natural language predicate to ontology predicate"""
        predicate_map = {
            "treats": "obo:RO_0002606",
            "causes": "obo:RO_0002410",
            "prevents": "obo:RO_0002559",
            "inhibits": "obo:RO_0002449",
            "activates": "obo:RO_0002448"
        }
        return predicate_map.get(predicate.lower(), f"obo:{predicate}")

# ============= SPARQL Executor =============

class SPARQLExecutor:
    """Executes SPARQL queries against ontology store"""

    def __init__(self, ontology_path: Optional[str] = None):
        self.store = Store()
        if ontology_path:
            self.load_ontology(ontology_path)

        # For demo: use a public SPARQL endpoint as fallback
        # self.endpoint = "https://sparql.uniprot.org/sparql"  # Example endpoint
        self.endpoint = "https://sparql.hegroup.org/sparql/"

    def load_ontology(self, path: str):
        """Load ontology into store"""
        try:
            self.store.load(path, format="application/rdf+xml")
        except Exception as e:
            print(f"Error loading ontology: {e}")

    def execute(self, query: str) -> Dict:
        """Execute SPARQL query"""
        start_time = time.time()

        try:
            # Try local store first
            # results = list(self.store.query(query))
            results =  self._query_endpoint(query)
            execution_time = time.time() - start_time

            return {
                "success": True,
                "results": results,
                "execution_time": execution_time,
                "source": "local"
            }
        except Exception:
            # Fallback to endpoint
            try:
                results = self._query_endpoint(query)
                execution_time = time.time() - start_time

                return {
                    "success": True,
                    "results": results,
                    "execution_time": execution_time,
                    "source": "endpoint"
                }
            except Exception as e2:
                return {
                    "success": False,
                    "error": str(e2),
                    "execution_time": time.time() - start_time
                }

    def _serialize_results(self, results):
        """Convert pyoxigraph results to JSON-serializable format"""
        serialized = []
        for row in results:
            serialized_row = {}
            for key, value in row.items():
                serialized_row[str(key)] = str(value)
            serialized.append(serialized_row)
        return serialized

    def _query_endpoint(self, query: str) -> List[Dict]:
        """Query a SPARQL endpoint"""
        response = requests.post(
            self.endpoint,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"}
        )

        if response.status_code == 200:
            return response.json().get("results", {}).get("bindings", [])
        else:
            raise Exception(f"Endpoint query failed: {response.status_code}")

# ============= Context Augmenter =============

class ContextAugmenter:
    """Creates augmented context from SPARQL results"""

    def augment(self, state: SPARQLAgentState) -> str:
        """Create augmented context from query results"""

        context_parts = []

        # Add original text for reference
        context_parts.append(f"Original text: {state['original_text']}")
        context_parts.append("\n--- Ontological Context ---\n")

        # Process each query result
        for query_result in state["query_results"]:
            if query_result.get("success") and query_result.get("results"):
                context = self._format_results(
                    query_result["query"],
                    query_result["results"]
                )
                context_parts.append(context)

        # Add validation status
        if state.get("validation_status"):
            context_parts.append("\n--- Validation ---\n")
            for relation, status in state["validation_status"].items():
                context_parts.append(f"{relation}: {status}")

        return "\n".join(context_parts)

    def _format_results(self, query: str, results: List[Dict]) -> str:
        """Format query results for context"""

        # Identify query type from the query string
        if "ASK WHERE" in query:
            # Boolean result
            return f"Validation result: {bool(results)}"

        elif "SELECT" in query:
            # Table results
            if not results:
                return "No results found."

            formatted = []
            for row in results[:5]:  # Limit to top 5 for context
                row_str = ", ".join([f"{k}: {v.get('value', v)}"
                                     for k, v in row.items()])
                formatted.append(f"  - {row_str}")

            return "\n".join(formatted)

        return str(results)

# ============= LangGraph Workflow =============

def create_sparql_workflow() -> StateGraph:
    """Create the LangGraph workflow for SPARQL agent"""

    # Initialize components
    generator = SPARQLGenerator()
    executor = SPARQLExecutor()
    augmenter = ContextAugmenter()

    # Create workflow
    graph = StateGraph(SPARQLAgentState)

    # Node: Analyze information needs
    def analyze_needs(state: SPARQLAgentState) -> SPARQLAgentState:
        """Determine what information is needed"""
        needs = generator.analyze_information_needs(state)
        state["information_needs"] = needs
        state["messages"].append(
            HumanMessage(content=f"Identified {len(needs)} information needs")
        )
        return state

    # Node: Generate SPARQL queries
    def generate_queries(state: SPARQLAgentState) -> SPARQLAgentState:
        """Generate SPARQL queries for information needs"""
        queries = []

        for i, need in enumerate(state["information_needs"][:10]):  # Limit for demo
            context = {
                "entities": state["extracted_entities"],
                "relations": state["extracted_relations"]
            }

            query = generator.generate_sparql(need, context)
            queries.append({
                "need": need,
                "query": query,
                "priority": i
            })

        state["generated_queries"] = queries
        state["messages"].append(
            AIMessage(content=f"Generated {len(queries)} SPARQL queries")
        )
        return state

    # Node: Execute queries
    def execute_queries(state: SPARQLAgentState) -> SPARQLAgentState:
        """Execute SPARQL queries"""
        results = []
        validation_status = {}

        for query_obj in state["generated_queries"]:
            result = executor.execute(query_obj["query"])
            result["query"] = query_obj["query"]
            result["need"] = query_obj["need"]
            results.append(result)

            # Track validation results
            if "validate_relation" in query_obj["need"]:
                parts = query_obj["need"].split(":")
                if len(parts) > 3:
                    relation_key = f"{parts[1]}-{parts[2]}-{parts[3]}"
                    validation_status[relation_key] = result.get("success", False)

        state["query_results"] = results
        state["validation_status"] = validation_status

        # Calculate metrics
        successful = sum(1 for r in results if r.get("success"))
        avg_time = np.mean([r.get("execution_time", 0) for r in results])

        state["metrics"] = {
            "total_queries": len(results),
            "successful_queries": successful,
            "average_execution_time": avg_time
        }

        state["messages"].append(
            AIMessage(content=f"Executed {len(results)} queries, {successful} successful")
        )
        return state

    # Node: Augment context
    def augment_context(state: SPARQLAgentState) -> SPARQLAgentState:
        """Create augmented context from results"""
        augmented = augmenter.augment(state)
        state["augmented_context"] = augmented
        state["messages"].append(
            AIMessage(content=f"Created augmented context ({len(augmented)} chars)")
        )
        return state

    # Add nodes to workflow
    graph.add_node("analyze_needs", analyze_needs)
    graph.add_node("generate_queries", generate_queries)
    graph.add_node("execute_queries", execute_queries)
    graph.add_node("augment_context", augment_context)

    # Define edges
    graph.add_edge("analyze_needs", "generate_queries")
    graph.add_edge("generate_queries", "execute_queries")
    graph.add_edge("execute_queries", "augment_context")
    graph.add_edge("augment_context", END)

    # Set entry point
    graph.set_entry_point("analyze_needs")

    return graph.compile()

# ============= Evaluation Functions =============

def evaluate_sparql_agent(test_cases: List[Dict]) -> Dict:
    """Evaluate SPARQL agent on test cases"""

    app = create_sparql_workflow()
    print(app.get_graph().print_ascii())
    results = []

    for test_case in test_cases:
        # Prepare initial state
        initial_state = {
            "extracted_entities": test_case["entities"],
            "extracted_relations": test_case["relations"],
            "original_text": test_case["text"],
            "information_needs": [],
            "generated_queries": [],
            "query_results": [],
            "augmented_context": "",
            "validation_status": {},
            "messages": [],
            "metrics": {}
        }

        # Run workflow
        start_time = time.time()
        final_state = app.invoke(initial_state)
        total_time = time.time() - start_time

        # Collect metrics
        result = {
            "test_case": test_case["id"],
            "total_time": total_time,
            "queries_generated": len(final_state["generated_queries"]),
            "queries_successful": final_state["metrics"].get("successful_queries", 0),
            "avg_query_time": final_state["metrics"].get("average_execution_time", 0),
            "context_length": len(final_state["augmented_context"]),
            "validations": final_state["validation_status"]
        }

        results.append(result)

        # Print progress
        print(f"Test case {test_case['id']}: {result['queries_successful']}/{result['queries_generated']} queries successful")

    # Calculate aggregate metrics
    aggregate = {
        "total_cases": len(results),
        "avg_queries_per_case": np.mean([r["queries_generated"] for r in results]),
        "success_rate": np.mean([r["queries_successful"]/max(r["queries_generated"], 1) for r in results]),
        "avg_total_time": np.mean([r["total_time"] for r in results]),
        "avg_query_time": np.mean([r["avg_query_time"] for r in results])
    }

    return {
        "individual_results": results,
        "aggregate_metrics": aggregate
    }

# ============= Example Usage =============

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

    # Run evaluation
    print("Starting SPARQL Agent Evaluation...")
    results = evaluate_sparql_agent(test_cases)

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Total test cases: {results['aggregate_metrics']['total_cases']}")
    print(f"Average queries per case: {results['aggregate_metrics']['avg_queries_per_case']:.1f}")
    print(f"Query success rate: {results['aggregate_metrics']['success_rate']:.2%}")
    print(f"Average total time: {results['aggregate_metrics']['avg_total_time']:.2f}s")
    print(f"Average query execution time: {results['aggregate_metrics']['avg_query_time']:.3f}s")

    # These numbers can be used in your presentation!
    print("\n=== Numbers for Presentation ===")
    print(f"• Generates {results['aggregate_metrics']['avg_queries_per_case']:.0f} SPARQL queries per extraction")
    print(f"• {results['aggregate_metrics']['success_rate']:.0%} query success rate")
    print(f"• {results['aggregate_metrics']['avg_query_time']*1000:.0f}ms average query time")
    print(f"• Total pipeline time: {results['aggregate_metrics']['avg_total_time']:.1f}s per document")