import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional

def create_context_subgraph(entity_id: str, ontology_graph: nx.Graph, context_depth: int = 2) -> nx.Graph:
    """
    Create a subgraph containing the entity and all its context relationships.

    Args:
        entity_id: The central entity to build context around
        ontology_graph: Full ontology graph
        context_depth: How many hops to include (1=direct neighbors, 2=neighbors of neighbors)

    Returns:
        NetworkX subgraph with context relationships
    """
    if entity_id not in ontology_graph:
        return nx.Graph()

    # Collect all relevant nodes
    relevant_nodes = set([entity_id])

    # Add neighbors at each depth level
    current_level = set([entity_id])
    for depth in range(context_depth):
        next_level = set()
        for node in current_level:
            if node in ontology_graph:
                neighbors = set(ontology_graph.neighbors(node))
                next_level.update(neighbors)
                relevant_nodes.update(neighbors)
        current_level = next_level - relevant_nodes  # Only new nodes

    # Create subgraph with relevant nodes
    context_subgraph = ontology_graph.subgraph(relevant_nodes).copy()

    return context_subgraph

def display_context_subgraph(entity_id: str, ontology_graph: nx.Graph,
                             figsize: tuple = (12, 8), context_depth: int = 2,
                             save_path: Optional[str] = None):
    """
    Display a visual subgraph of entity context relationships.

    Args:
        entity_id: The central entity
        ontology_graph: Full ontology graph
        figsize: Figure size for matplotlib
        context_depth: How many relationship hops to show
        save_path: Optional path to save the figure
    """
    # Create context subgraph
    subgraph = create_context_subgraph(entity_id, ontology_graph, context_depth)

    if len(subgraph.nodes()) == 0:
        print(f"No context found for entity: {entity_id}")
        return

    # Set up the plot
    plt.figure(figsize=figsize)

    # Create layout (spring layout works well for small graphs)
    pos = nx.spring_layout(subgraph, k=2, iterations=50)

    # Get node information for styling
    node_colors = []
    node_sizes = []
    node_labels = {}

    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]

        # Color by ontology type
        ontology = node_data.get('ontology', 'unknown')
        color_map = {
            'mondo': '#ff6b6b',      # Red for diseases
            'chebi': '#4ecdc4',      # Teal for chemicals
            'hp': '#45b7d1',         # Blue for phenotypes
            'go': '#96ceb4',         # Green for processes
            'uberon': '#feca57',     # Yellow for anatomy
            'cl': '#ff9ff3',         # Pink for cells
            'unknown': '#95a5a6'     # Gray for unknown
        }
        node_colors.append(color_map.get(ontology, '#95a5a6'))

        # Size by centrality (central entity larger)
        if node == entity_id:
            node_sizes.append(1500)  # Central entity is largest
        else:
            node_sizes.append(800)

        # Labels (use name if available, otherwise ID)
        name = node_data.get('name', node.split(':')[-1])
        # Truncate long names
        if len(name) > 15:
            name = name[:12] + "..."
        node_labels[node] = name

    # Draw the network
    nx.draw_networkx_nodes(subgraph, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.8)

    nx.draw_networkx_edges(subgraph, pos,
                           edge_color='gray',
                           alpha=0.6,
                           arrows=True,
                           arrowsize=20,
                           arrowstyle='->')

    nx.draw_networkx_labels(subgraph, pos,
                            node_labels,
                            font_size=8,
                            font_weight='bold')

    # Add edge labels for relationship types
    edge_labels = {}
    for u, v in subgraph.edges():
        edge_data = subgraph.get_edge_data(u, v, {})
        relation = edge_data.get('relation', 'related_to')
        edge_labels[(u, v)] = relation

    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=6)

    # Title and legend
    central_name = ontology_graph.nodes.get(entity_id, {}).get('name', entity_id)
    plt.title(f"Ontological Context for: {central_name}", fontsize=14, fontweight='bold')

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b', markersize=10, label='Disease'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ecdc4', markersize=10, label='Chemical'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#45b7d1', markersize=10, label='Phenotype'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#96ceb4', markersize=10, label='Process'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#feca57', markersize=10, label='Anatomy')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    plt.axis('off')
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Subgraph saved to: {save_path}")

    plt.show()

    # Print context information
    print("\nContext Subgraph Statistics:")
    print(f"  Nodes: {len(subgraph.nodes())}")
    print(f"  Edges: {len(subgraph.edges())}")
    print(f"  Central entity: {central_name} ({entity_id})")

def get_context_as_text(entity_id: str, ontology_graph: nx.Graph, max_items: int = 8) -> List[str]:
    """
    Get the context relationships as text (same as your extract_entity_context function)
    """
    subgraph = create_context_subgraph(entity_id, ontology_graph, context_depth=1)

    context = []
    central_name = ontology_graph.nodes.get(entity_id, {}).get('name', entity_id.split(':')[-1])

    # Get relationships from the subgraph
    for neighbor in subgraph.neighbors(entity_id):
        neighbor_data = subgraph.nodes.get(neighbor, {})
        neighbor_name = neighbor_data.get('name', neighbor.split(':')[-1])
        neighbor_ontology = neighbor_data.get('ontology', '')
        central_ontology = ontology_graph.nodes.get(entity_id, {}).get('ontology', '')

        # Infer relationship type
        if central_ontology == 'mondo' and neighbor_ontology == 'chebi':
            context.append(f"{neighbor_name} treats {central_name}")
        elif central_ontology == 'chebi' and neighbor_ontology == 'mondo':
            context.append(f"{central_name} treats {neighbor_name}")
        elif central_ontology == 'mondo' and neighbor_ontology == 'hp':
            context.append(f"{central_name} causes {neighbor_name}")
        else:
            # Check if it's a parent relationship
            if neighbor in ontology_graph.predecessors(entity_id):
                context.append(f"{central_name} is_a {neighbor_name}")
            else:
                context.append(f"{central_name} related_to {neighbor_name}")

    return context[:max_items]