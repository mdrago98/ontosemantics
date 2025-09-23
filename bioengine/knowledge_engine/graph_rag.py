"""A module containing useful functions for performing agentic graph rag"""
from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict, Optional, List, Any, Dict, Set, Tuple, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import StateGraph
from pydantic import BaseModel
from rdflib.plugins.sparql.parser import GraphNode
from sqlalchemy.testing.pickleable import Mixin


class TraversalStrategy(Enum):
    BFS = "breadth_first"
    DFS = "depth_first"
    TARGETED = "targeted"
    SEMANTIC = "semantic"


class AgentState(TypedDict):
    # Input
    initial_query: str
    research_goal: str
    focal_entities: List[str]

    # Query execution state
    current_query: str
    query_results: List[Dict[str, Any]]
    query_history: List[Dict[str, Any]]

    # Analysis state
    analysis_summary: str
    key_findings: List[str]
    knowledge_gaps: List[str]

    # Decision state
    next_action: Literal["query", "refine", "explore", "summarize", "end"]
    refinement_strategy: str

    # Final output
    comprehensive_findings: Dict[str, Any]
    iteration_count: int
    max_iterations: int


@dataclass
class GraphPath:
    nodes: List[GraphNode]
    relationships: List[Tuple[str, str, str]]  # (subject, predicate, object)
    path_score: float = 0.0
    reasoning: str = ""


@dataclass
class GraphContext:
    focal_entities: List[str]
    discovered_nodes: Dict[str, GraphNode] = field(default_factory=dict)
    paths: List[GraphPath] = field(default_factory=list)
    visited: Set[str] = field(default_factory=set)
    query_history: List[str] = field(default_factory=list)


class NodeType(Enum):
    ENTITY = "entity"
    CONCEPT = "concept"
    RELATIONSHIP = "relationship"
    EVIDENCE = "evidence"


@dataclass
class GraphNode:
    uri: str
    labels: List[str] = field(default_factory=list)
    node_type: NodeType = NodeType.ENTITY
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    depth: int = 0
    source_query: str = ""


class EntitySuggestion(BaseModel):
    entity_uri: str
    relationships: list[str]


class PyOxiQuerieable(Mixin):
    def query(self, sparql_query: str) -> List[Dict[str, Any]]:
        """
        Execute SPARQL query against pyoxigraph store.
        :param sparql_query: the sparql query
        :return: A list results in the form of dict[str, Any]
        """
        try:
            results = []
            for solution in self.store.query(sparql_query):
                result = {}
                for var_name in solution.variables:
                    value = solution[var_name]
                    if value:
                        result[str(var_name)] = str(value)
                results.append(result)
            return results
        except Exception as e:
            print(f"Query execution failed: {e}")
            return []

    def _get_prefix_string(self) -> str:
        """Generate PREFIX declarations for SPARQL queries."""
        return "\n".join([f"PREFIX {prefix}: <{uri}>" for prefix, uri in self.prefixes.items()])


class GraphReasoningAgent(PyOxiQuerieable):

    def __init__(self, database, prefixes, llm):
        self.database = database
        self.llm = llm
        self.prefixes = prefixes

        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        graph = StateGraph(AgentState)
        # TODO: build graph
        return graph

    def explore_neighborhood(
        self,
        entity_uri: str,
        depth: int = 1,
        strategy: TraversalStrategy = TraversalStrategy.BFS,
    ) -> GraphContext:
        """Explore the neighborhood around an entity using specified traversal strategy."""

        context = GraphContext(focal_entities=[entity_uri])

        print(
            f"Exploring neighborhood of {entity_uri} (depth={depth}, strategy={strategy.value})"
        )

        # Initialize with the focal entity
        focal_node = self._get_entity_details(entity_uri)
        context.discovered_nodes[entity_uri] = focal_node

        # Traverse based on strategy
        if strategy == TraversalStrategy.BFS:
            self._traverse_bfs(entity_uri, context, depth)
        elif strategy == TraversalStrategy.DFS:
            self._traverse_dfs(entity_uri, context, depth, 0)
        elif strategy == TraversalStrategy.TARGETED:
            self._traverse_targeted(entity_uri, context, depth)
        elif strategy == TraversalStrategy.SEMANTIC:
            self._traverse_semantic(entity_uri, context, depth)

        return context

    def find_connecting_paths(
        self, start_entity: str, end_entity: str, max_hops: int = 3
    ) -> List[GraphPath]:
        """Find all paths connecting two entities within max_hops."""

        print(
            f"Finding paths from {start_entity} to {end_entity} (max_hops={max_hops})"
        )

        paths = []

        # Use iterative deepening to find paths of increasing length
        for hop_count in range(1, max_hops + 1):
            hop_paths = self._find_paths_with_hops(
                start_entity, end_entity, hop_count
            )
            paths.extend(hop_paths)

            if len(paths) >= 10:  # Limit to avoid explosion
                break

        # Score and rank paths
        scored_paths = []
        for path in paths:
            score = self._score_path(path)
            path.path_score = score
            scored_paths.append(path)

        return sorted(scored_paths, key=lambda p: p.path_score, reverse=True)[:5]

    def agentic_query_expansion(
        self, initial_query: str, context: GraphContext
    ) -> List[str]:
        """Use LLM to generate follow-up queries based on discovered context."""

        agent_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert biomedical knowledge graph analyst with access to tools for querying SPARQL endpoints.

        Your task is to autonomously explore and analyze biomedical relationships through iterative querying and refinement.

        Initial Context:
        - Initial Query: {initial_query}
        - Focal Entities: {focal_entities}
        - Discovered Nodes: {key_nodes}
        - Available Prefixes: {prefixes}

        Your approach should be:
        1. Start with targeted queries based on the initial context
        2. Analyze results to identify gaps or interesting patterns
        3. Refine and generate follow-up queries
        4. Look for contradictions, supporting evidence, or missing mechanisms
        5. Build a comprehensive picture of the biomedical relationships

        Available SPARQL prefixes:
        {prefixes}

        Focus on biomedical relationships like:
        - Drug mechanisms and targets
        - Disease pathways and causes
        - Protein-protein interactions
        - Gene-disease associations
        - Metabolic pathways

        Use the tools iteratively to build knowledge. When you execute queries, always specify the purpose clearly.
        """,
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Summarize context for LLM
        key_nodes = list(context.discovered_nodes.keys())[:10]
        relationships = [
            (path.relationships[0] if path.relationships else ("", "", ""))
            for path in context.paths[:5]
        ]

        response = self.llm.invoke(
            expansion_prompt.format_messages(
                initial_query=initial_query,
                focal_entities=context.focal_entities,
                key_nodes=key_nodes,
                relationships=relationships,
                prefixes=self._get_prefix_string(),
            )
        )

        # Extract queries from response
        return self._extract_sparql_queries(response.content)

    def intelligent_subgraph_extraction(
        self, seed_entities: List[str], target_relationship_types: List[str] = None
    ) -> GraphContext:
        """Extract a relevant subgraph around seed entities using intelligent traversal."""

        print(
            f"🧠 Intelligent subgraph extraction for {len(seed_entities)} seed entities"
        )

        context = GraphContext(focal_entities=seed_entities)

        # Start with seed entities
        for entity in seed_entities:
            node = self._get_entity_details(entity)
            context.discovered_nodes[entity] = node

        # Iteratively expand using LLM guidance
        for iteration in range(3):  # Max 3 expansion iterations
            print(f"   Expansion iteration {iteration + 1}")

            # Get LLM suggestions for next expansion
            expansion_targets = self._get_expansion_suggestions(
                context, target_relationship_types
            )

            # Execute suggested expansions
            for target in expansion_targets[:5]:  # Limit expansions per iteration
                self._expand_around_target(target, context)

        return context

    # ==== TRAVERSAL STRATEGY IMPLEMENTATIONS ====

    def _traverse_bfs(self, start_uri: str, context: GraphContext, max_depth: int):
        """Breadth-first traversal implementation."""

        current_level = [start_uri]
        context.visited.add(start_uri)

        for depth in range(max_depth):
            print(f"   BFS Level {depth + 1}: {len(current_level)} nodes")
            next_level = []

            for node_uri in current_level:
                neighbors = self._get_direct_neighbors(node_uri)

                for neighbor_uri, relationship in neighbors[
                    : self.max_nodes_per_level
                ]:
                    if neighbor_uri not in context.visited:
                        neighbor_node = self._get_entity_details(neighbor_uri)
                        neighbor_node.depth = depth + 1
                        context.discovered_nodes[neighbor_uri] = neighbor_node
                        context.visited.add(neighbor_uri)
                        next_level.append(neighbor_uri)

                        # Record path
                        path = GraphPath(
                            nodes=[
                                context.discovered_nodes[node_uri],
                                neighbor_node,
                            ],
                            relationships=[(node_uri, relationship, neighbor_uri)],
                        )
                        context.paths.append(path)

            current_level = next_level
            if not current_level:
                break

    def _traverse_dfs(
        self,
        node_uri: str,
        context: GraphContext,
        max_depth: int,
        current_depth: int,
    ):
        """Depth-first traversal implementation."""

        if current_depth >= max_depth or node_uri in context.visited:
            return

        context.visited.add(node_uri)
        neighbors = self._get_direct_neighbors(node_uri)

        print(
            f"   DFS Depth {current_depth}: exploring {node_uri} ({len(neighbors)} neighbors)"
        )

        for neighbor_uri, relationship in neighbors[: self.max_nodes_per_level]:
            if neighbor_uri not in context.visited:
                neighbor_node = self._get_entity_details(neighbor_uri)
                neighbor_node.depth = current_depth + 1
                context.discovered_nodes[neighbor_uri] = neighbor_node

                # Record path
                path = GraphPath(
                    nodes=[context.discovered_nodes[node_uri], neighbor_node],
                    relationships=[(node_uri, relationship, neighbor_uri)],
                )
                context.paths.append(path)

                # Recursive DFS
                self._traverse_dfs(
                    neighbor_uri, context, max_depth, current_depth + 1
                )

    def _traverse_targeted(
        self, entity_uri: str, context: GraphContext, depth: int
    ):
        """Targeted traversal focusing on specific relationship types."""

        # High-value relationship types for biomedical data
        priority_relations = [
            "treats",
            "causes",
            "prevents",
            "interacts_with",
            "regulates",
            "part_of",
            "has_mechanism",
            "targets",
        ]

        current_nodes = [entity_uri]
        context.visited.add(entity_uri)

        for level in range(depth):
            next_nodes = []

            for node_uri in current_nodes:
                # Focus on priority relationships
                for relation in priority_relations:
                    related_nodes = self._get_nodes_by_relationship(
                        node_uri, relation
                    )

                    for related_uri in related_nodes[:5]:  # Limit per relation type
                        if related_uri not in context.visited:
                            related_node = self._get_entity_details(related_uri)
                            related_node.depth = level + 1
                            context.discovered_nodes[related_uri] = related_node
                            context.visited.add(related_uri)
                            next_nodes.append(related_uri)

                            path = GraphPath(
                                nodes=[
                                    context.discovered_nodes[node_uri],
                                    related_node,
                                ],
                                relationships=[(node_uri, relation, related_uri)],
                            )
                            context.paths.append(path)

            current_nodes = next_nodes
            if not current_nodes:
                break

    def _traverse_semantic(
        self, entity_uri: str, context: GraphContext, depth: int
    ):
        """Semantic traversal using LLM to guide exploration."""

        current_nodes = [entity_uri]
        context.visited.add(entity_uri)

        for level in range(depth):
            print(
                f"   Semantic Level {level + 1}: analyzing {len(current_nodes)} nodes"
            )

            # Use LLM to determine most interesting expansion directions
            expansion_plan = self._get_semantic_expansion_plan(
                current_nodes, context
            )

            next_nodes = []
            for node_uri in current_nodes:
                if node_uri in expansion_plan:
                    targets = expansion_plan[node_uri]
                    for target_relation in targets[:3]:  # Top 3 directions per node
                        related_nodes = self._get_nodes_by_relationship(
                            node_uri, target_relation
                        )

                        for related_uri in related_nodes[:5]:
                            if related_uri not in context.visited:
                                related_node = self._get_entity_details(related_uri)
                                related_node.depth = level + 1
                                context.discovered_nodes[related_uri] = related_node
                                context.visited.add(related_uri)
                                next_nodes.append(related_uri)

                                path = GraphPath(
                                    nodes=[
                                        context.discovered_nodes[node_uri],
                                        related_node,
                                    ],
                                    relationships=[
                                        (node_uri, target_relation, related_uri)
                                    ],
                                )
                                context.paths.append(path)

            current_nodes = next_nodes

    # Query Helper functions

    def _get_entity_details(self, entity_uri: str) -> GraphNode:
        """Get detailed information about an entity."""

        query = f"""
        {self._get_prefix_string()}

        SELECT ?label ?type ?definition ?synonym WHERE {{
            <{entity_uri}> rdfs:label ?label .
            OPTIONAL {{ <{entity_uri}> rdf:type ?type }}
            OPTIONAL {{ <{entity_uri}> rdfs:comment ?definition }}
            OPTIONAL {{ <{entity_uri}> obo:IAO_0000115 ?definition }}
            OPTIONAL {{ <{entity_uri}> obo:hasExactSynonym ?synonym }}
        }}
        LIMIT 10
        """

        results = self.query(query)

        labels = list(set([r.get("label", "") for r in results if r.get("label")]))
        types = list(set([r.get("type", "") for r in results if r.get("type")]))
        definitions = [
            r.get("definition", "") for r in results if r.get("definition")
        ]
        synonyms = [r.get("synonym", "") for r in results if r.get("synonym")]

        return GraphNode(
            uri=entity_uri,
            labels=labels,
            node_type=self._infer_node_type(types),
            properties={
                "types": types,
                "definitions": definitions,
                "synonyms": synonyms,
            },
        )

    def _get_direct_neighbors(self, entity_uri: str) -> List[Tuple[str, str]]:
        """Get direct neighbors (1-hop connections) of an entity."""

        query = f"""
        {self._get_prefix_string()}

        SELECT DISTINCT ?neighbor ?relation WHERE {{
            {{
                <{entity_uri}> ?relation ?neighbor .
                FILTER(!isBlank(?neighbor) && isIRI(?neighbor))
            }} UNION {{
                ?neighbor ?relation <{entity_uri}> .
                FILTER(!isBlank(?neighbor) && isIRI(?neighbor))
            }}
            FILTER(?relation != rdf:type)
        }}
        LIMIT 50
        """

        results = self.query(query)
        return [
            (r["neighbor"], r["relation"])
            for r in results
            if "neighbor" in r and "relation" in r
        ]

    def _get_nodes_by_relationship(
        self, entity_uri: str, relation_type: str
    ) -> List[str]:
        """Get nodes connected by a specific relationship type."""

        query = f"""
        {self._get_prefix_string()}

        SELECT DISTINCT ?connected WHERE {{
            {{
                <{entity_uri}> ?pred ?connected .
                FILTER(contains(lcase(str(?pred)), lcase("{relation_type}")))
            }} UNION {{
                ?connected ?pred <{entity_uri}> .
                FILTER(contains(lcase(str(?pred)), lcase("{relation_type}")))
            }}
            FILTER(!isBlank(?connected) && isIRI(?connected))
        }}
        LIMIT 20
        """

        results = self.query(query)
        return [r["connected"] for r in results if "connected" in r]

    def _find_paths_with_hops(
        self, start: str, end: str, hops: int
    ) -> List[GraphPath]:
        """Find paths of specific hop length between two entities."""

        if hops == 1:
            # Direct connection
            query = f"""
            {self._get_prefix_string()}

            SELECT ?relation WHERE {{
                <{start}> ?relation <{end}> .
            }}
            """
            results = self.query(query)

            paths = []
            for result in results:
                path = GraphPath(
                    nodes=[
                        self._get_entity_details(start),
                        self._get_entity_details(end),
                    ],
                    relationships=[(start, result["relation"], end)],
                )
                paths.append(path)
            return paths

        elif hops == 2:
            # Two-hop connection
            query = f"""
            {self._get_prefix_string()}

            SELECT ?intermediate ?rel1 ?rel2 WHERE {{
                <{start}> ?rel1 ?intermediate .
                ?intermediate ?rel2 <{end}> .
                FILTER(?intermediate != <{start}> && ?intermediate != <{end}>)
            }}
            LIMIT 10
            """
            results = self.query(query)

            paths = []
            for result in results:
                intermediate = result["intermediate"]
                path = GraphPath(
                    nodes=[
                        self._get_entity_details(start),
                        self._get_entity_details(intermediate),
                        self._get_entity_details(end),
                    ],
                    relationships=[
                        (start, result["rel1"], intermediate),
                        (intermediate, result["rel2"], end),
                    ],
                )
                paths.append(path)
            return paths

        # For longer paths, would need more complex recursive queries
        return []

    # LLM guided EXPANSION FUNCTIONS

    def _get_expansion_suggestions(
        self, context: GraphContext, target_relations: List[str] = None, prompt=None
    ) -> List[str]:
        """
        Get LLM planing suggestions for graph expansion.
        :param context: Graph context for the current entity
        :param target_relations: the target relationships to expand on
        :param prompt: the prompt
        :return:
        """
        parser = JsonOutputParser()
        if prompt is None:
            prompt = PromptTemplate(
                template="""
    Analyze this biomedical knowledge graph context and suggest the most important entities to explore next.
    
    Current Context:
    - Focal Entities: {focal_entities}
    - Discovered Entities: {discovered_entities}
    - Key Relationships: {key_relationships}
    
    Target Relationship Types: {target_relations}
    
    Based on this biomedical context, suggest 3-5 specific entities (by URI or name) that would be most valuable to explore next. Focus on:
    1. Entities that could bridge important gaps
    2. Entities that represent key mechanisms or pathways
    3. Entities that could provide contradictory or supporting evidence
    4. Entities that are central to the biological process being investigated
    
    Return entity suggestions as {format_instructions}.
    """,
                input_variables=[
                    "focal_entities",
                    "discovered_entities",
                    "key_relationships",
                    "target_relationships",
                ],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

        # Prepare context summary
        discovered_entities = list(context.discovered_nodes.keys())[:20]
        key_relationships = [
            path.relationships[0] if path.relationships else ("", "", "")
            for path in context.paths[:10]
        ]

        chain = prompt | self.llm | parser

        return chain.invoke(
            {
                "focal_entities": context.focal_entities,
                "discovered_entities": discovered_entities,
                "key_relationships": key_relationships,
                "target_relations": target_relations or ["all types"],
            }
        )

    def _get_semantic_expansion_plan(
        self, current_nodes: List[str], context: GraphContext, prompt: PromptTemplate = None
    ) -> Dict[str, List[str]]:
        """
        LLM-guided expansion plan for semantic traversal.
        :param current_nodes:
        :param context:
        :return:
        """
        parser = JsonOutputParser(pydantic_object=EntitySuggestion)
        prompt = PromptTemplate(template="""
For each of these biomedical entities, determine the most semantically important relationship types to explore:

Entities: {entities}
Current Context: {context_summary}

For each entity, suggest the top 2-3 relationship types that would be most valuable to explore in a biomedical context.

{format_instructions}

Focus on biomedical relationships like: treats, causes, regulates, interacts_with, part_of, has_mechanism, targets, etc.
""",input_variables=[
                    "entities",
                    "context_summary",
                ],
                partial_variables={"format_instructions": parser.get_format_instructions()})

        context_summary = {
            "discovered_count": len(context.discovered_nodes),
            "path_count": len(context.paths),
            "key_entities": list(context.discovered_nodes.keys())[:10],
        }

        chain = prompt | self.llm | parser


        return chain.invoke(
            {
                'entities':current_nodes, 'context_summary':context_summary
            }
        )

        # Fallback plan
        return {node: ["treats", "causes", "regulates"] for node in current_nodes}

    # Utils FUNCTIONS

    def _infer_node_type(self, types: List[str]) -> NodeType:
        """Infer node type from RDF types."""

        type_str = " ".join(types).lower()

        if any(term in type_str for term in ["drug", "compound", "chemical"]):
            return NodeType.ENTITY
        elif any(term in type_str for term in ["disease", "disorder", "phenotype"]):
            return NodeType.ENTITY
        elif any(term in type_str for term in ["protein", "gene"]):
            return NodeType.ENTITY
        elif any(term in type_str for term in ["pathway", "process"]):
            return NodeType.CONCEPT
        else:
            return NodeType.ENTITY

    def _score_path(self, path: GraphPath) -> float:
        """Score a path based on various factors."""

        score = 1.0

        # Shorter paths are generally better (but not too short)
        path_length = len(path.relationships)
        if path_length == 1:
            score *= 1.2  # Direct connections are valuable
        elif path_length == 2:
            score *= 1.0  # Two-hop paths are good
        elif path_length >= 3:
            score *= 0.8 ** (path_length - 2)  # Penalty for longer paths

        # Bonus for biomedically relevant relationships
        biomedical_relations = {
            "treats",
            "causes",
            "regulates",
            "interacts_with",
            "part_of",
        }
        for _, relation, _ in path.relationships:
            if any(term in relation.lower() for term in biomedical_relations):
                score *= 1.3

        return score

    def _extract_sparql_queries(self, response_text: str) -> List[str]:
        """Extract SPARQL queries from LLM response."""

        sparql_pattern = r"```sparql\n(.*?)\n```"
        matches = re.findall(sparql_pattern, response_text, re.DOTALL)

        if matches:
            return [query.strip() for query in matches]

        # Fallback
        code_pattern = r"```\n(.*?)\n```"
        matches = re.findall(code_pattern, response_text, re.DOTALL)

        sparql_queries = []
        for match in matches:
            if any(
                keyword in match.upper()
                for keyword in ["SELECT", "PREFIX", "WHERE"]
            ):
                sparql_queries.append(match.strip())

        return sparql_queries



