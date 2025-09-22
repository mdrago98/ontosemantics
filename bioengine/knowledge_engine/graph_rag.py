"""A module containing useful functions for performing agentic graph rag"""
from dataclasses import dataclass, field
from typing import TypedDict, Optional, List, Any, Dict, Set, Tuple

from langgraph.graph import StateGraph
from rdflib.plugins.sparql.parser import GraphNode


class AgentState(TypedDict):
    """TypedDict to maintain state across nodes in the LangGraph workflow."""
    user_query: str
    reasoning: Optional[str]
    generated_sql: Optional[str]
    is_valid: Optional[bool]
    results: Optional[List[Any]]
    error: Optional[str]
    visualization_type: Optional[str]
    visualization_data: Optional[Dict[str, Any]]


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


class GraphReasoningAgent:

    def __init__(self, database, prefixes, llm):
        self.database = database
        self.llm = llm
        self.prefixes = prefixes

        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        graph = StateGraph(AgentState)
        # TODO: build graph
        return graph



