import re
from typing import Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import tqdm


class OntologyEmbeddingMixin:
    """Mixin providing embedding and latent subgraph utilities for ontologies."""

    def __init__(self, *args, **kwargs):
        self.embedder: Optional[Callable[[List[str]], np.ndarray]] = None
        self._node_embeddings: Optional[np.ndarray] = None
        self._node_ids: List[str] = []
        self._node_texts: List[str] = []
        self._id_to_index: Dict[str, int] = {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def _as_numpy(array_like) -> np.ndarray:
        """Best-effort conversion of embedder outputs to numpy arrays."""
        if isinstance(array_like, np.ndarray):
            return array_like
        # Lazy import to avoid hard dependency on torch
        if hasattr(array_like, "detach"):
            try:
                return array_like.detach().cpu().numpy()
            except Exception:
                pass
        return np.asarray(array_like)

    def register_embedder(self, embedder: Callable[[List[str]], np.ndarray]):
        """Register an embedding function used for ontology term retrieval."""
        self.embedder = embedder
        self._node_embeddings = None

    def _compose_node_text(self, node_data: Dict) -> str:
        parts = []
        name = node_data.get('name') or ''
        if name:
            parts.append(name)
        synonyms = node_data.get('synonyms') or []
        if synonyms:
            parts.extend([syn for syn in synonyms if isinstance(syn, str)])
        definition = node_data.get('definition') or ''
        if definition:
            parts.append(definition)
        return ' . '.join([p for p in parts if p])

    def _build_text_corpus(self):
        """Cache node text representations for retrieval."""
        self._node_ids = []
        self._node_texts = []
        self._id_to_index = {}
        for index, (node_id, data) in enumerate(self.ontology_graph.nodes(data=True)):
            self._node_ids.append(node_id)
            self._node_texts.append(self._compose_node_text(data))
            self._id_to_index[node_id] = index

    def precompute_embeddings(self, batch_size: int = 1024):
        if self.embedder is None:
            raise ValueError("Call register_embedder(...) first.")
        if not self._node_texts:
            self._build_text_corpus()

        embs = []
        for i in range(0, len(self._node_texts), batch_size):
            chunk = self._node_texts[i:i + batch_size]
            E = self.embedder(chunk)  # MUST return np.ndarray [len(chunk), d] float32
            assert isinstance(E,
                              np.ndarray) and E.ndim == 2, f"embedder must return (B, d), got {type(E)}, {getattr(E, 'shape', None)}"
            embs.append(E)

        self._node_embeddings = np.vstack(embs).astype(np.float32)  # [N, d]

    def retrieve_candidates(self, text: str, top_k: int = 1000) -> List[Tuple[str, float]]:
        """Retrieve ontology nodes by text similarity."""
        if self.embedder is not None:
            if self._node_embeddings is None or not self._node_texts:
                self.precompute_embeddings()
            query = self._as_numpy(self.embedder([text])).astype(np.float32)
            if query.ndim == 2:
                query = query[0]
            ontology_matrix = self._node_embeddings
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            matrix_norm = ontology_matrix / (np.linalg.norm(ontology_matrix, axis=1, keepdims=True) + 1e-8)
            sims = matrix_norm @ query_norm
            top_idx = np.argpartition(sims, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            return [(self._node_ids[i], float(sims[i])) for i in top_idx]
        if not self._node_texts:
            self._build_text_corpus()
        query_tokens = set(re.findall(r'[A-Za-z0-9]+', text.lower()))
        scores: List[Tuple[str, float]] = []
        for idx, document in enumerate(self._node_texts):
            doc_tokens = set(re.findall(r'[A-Za-z0-9]+', document.lower()))
            if not doc_tokens:
                continue
            jaccard = len(query_tokens & doc_tokens) / (len(query_tokens | doc_tokens) + 1e-8)
            if jaccard > 0:
                scores.append((self._node_ids[idx], float(jaccard)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def _ppr_scores(self, seeds: List[str], alpha: float = 0.2) -> Dict[str, float]:
        """Compute personalized PageRank scores rooted at the provided seeds."""
        if not seeds:
            return {}
        graph = self.ontology_graph.to_undirected()
        personalization = {node: 0.0 for node in graph.nodes()}
        valid_seeds = [seed for seed in seeds if seed in graph]
        if not valid_seeds:
            return {}
        weight = 1.0 / len(valid_seeds)
        for seed in valid_seeds:
            personalization[seed] = weight
        pagerank = nx.pagerank(graph, alpha=(1.0 - alpha), personalization=personalization)
        total = sum(pagerank.values()) + 1e-12
        return {node: score / total for node, score in pagerank.items()}

    def build_latent_subgraph(
        self,
        text: str,
        mentions: Optional[List[str]] = None,
        top_k_retrieval: int = 1500,
        seed_top_k: int = 50,
        expand_hops: int = 2,
        ppr_alpha: float = 0.2,
        final_top_k: int = 200,
        allowed_biolink_types: Optional[Set[str]] = None,
        allowed_relations: Optional[Set[str]] = None,
        weights: Dict[str, float] = None,
    ) -> nx.DiGraph:
        """Build a compact, scored, typed subgraph suitable for downstream reasoning."""
        if weights is None:
            weights = dict(text_sim=0.6, ppr=0.4, seed_bonus=0.05)

        seed_ids: Set[str] = set()
        if mentions:
            for mention in mentions:
                matches = self.validate_and_enrich_entity(mention)
                if matches:
                    seed_ids.add(self._iri_to_curie(matches[0].ontology_id))

        retrieved = self.retrieve_candidates(text, top_k=top_k_retrieval)
        for node_id, _ in retrieved[:seed_top_k]:
            seed_ids.add(node_id)

        frontier = set(seed_ids)
        expanded = set(seed_ids)
        for _ in range(expand_hops):
            next_frontier: Set[str] = set()
            for node in list(frontier):
                if node not in self.ontology_graph:
                    continue
                neighbors = list(self.ontology_graph.predecessors(node)) + list(self.ontology_graph.successors(node))
                for neighbor in neighbors:
                    expanded.add(neighbor)
                    next_frontier.add(neighbor)
            frontier = next_frontier

        candidate_ids = list(expanded)
        text_sims = {node_id: 0.0 for node_id in candidate_ids}
        if retrieved:
            retrieved_dict = dict(retrieved)
            for node_id in candidate_ids:
                if node_id in retrieved_dict:
                    text_sims[node_id] = retrieved_dict[node_id]

        pprs = self._ppr_scores(list(seed_ids), alpha=ppr_alpha)
        scores: Dict[str, float] = {}
        for node_id in candidate_ids:
            score = weights['text_sim'] * text_sims.get(node_id, 0.0) + weights['ppr'] * pprs.get(node_id, 0.0)
            if node_id in seed_ids:
                score += weights.get('seed_bonus', 0.0)
            scores[node_id] = score

        if allowed_biolink_types is not None:
            filtered: List[str] = []
            for node_id in candidate_ids:
                ontology_name = self.term_to_ontology.get(node_id, '')
                node_data = self.ontology_graph.nodes[node_id]
                biolink_type = self._get_biolink_type(ontology_name, node_data)
                if biolink_type in allowed_biolink_types:
                    filtered.append(node_id)
            candidate_ids = filtered
            scores = {node_id: scores[node_id] for node_id in candidate_ids}

        if len(candidate_ids) > final_top_k:
            candidates = np.array([(node_id, scores[node_id]) for node_id in candidate_ids], dtype=object)
            indices = np.argpartition(candidates[:, 1].astype(np.float32), -final_top_k)[-final_top_k:]
            indices = indices[np.argsort(-candidates[indices, 1].astype(np.float32))]
            chosen = [candidates[index, 0] for index in indices]
        else:
            chosen = sorted(candidate_ids, key=lambda node_id: -scores[node_id])

        chosen_set = set(chosen)
        subgraph = self.ontology_graph.subgraph(chosen_set).copy()

        for node_id in subgraph.nodes():
            node_info = self.ontology_graph.nodes[node_id]
            ontology_name = node_info.get('ontology', '')
            biolink_type = self._get_biolink_type(ontology_name, node_info)
            source = 'expanded'
            if node_id in seed_ids:
                source = 'seed'
            elif node_id in dict(retrieved).keys():
                source = 'retrieved'
            subgraph.nodes[node_id].update({
                'score': float(scores.get(node_id, 0.0)),
                'biolink_type': biolink_type,
                'source': source,
            })

        if allowed_relations is not None:
            to_remove = []
            for u, v, edge_data in subgraph.edges(data=True):
                if edge_data.get('relation') not in allowed_relations:
                    to_remove.append((u, v))
            subgraph.remove_edges_from(to_remove)

        for u, v in list(subgraph.edges()):
            relation = self.ontology_graph.edges[u, v].get('relation', 'related_to')
            subgraph.edges[u, v].update({
                'relation': relation,
                'weight': float(0.5 * (subgraph.nodes[u]['score'] + subgraph.nodes[v]['score'])),
            })

        return subgraph
