from streamlit_agraph import Config, Edge, Node, agraph

from src.models.concept import KnowledgeGraph

# Color palette by depth level
LEVEL_COLORS = [
    "#FF6B6B",  # level 0 — root (red)
    "#4ECDC4",  # level 1 (teal)
    "#45B7D1",  # level 2 (blue)
    "#96CEB4",  # level 3 (green)
    "#FFEAA7",  # level 4 (yellow)
]


def render_graph(kg: KnowledgeGraph) -> str | None:
    """Render the knowledge graph as an interactive mind-map. Returns clicked node id."""
    nodes = []
    edges = []

    for concept in kg.concepts.values():
        color = LEVEL_COLORS[min(concept.level, len(LEVEL_COLORS) - 1)]
        size = max(35 - concept.level * 5, 15)

        nodes.append(
            Node(
                id=concept.id,
                label=concept.name,
                size=size,
                color=color,
                font={"size": 14, "color": "#333333"},
                shape="dot",
            )
        )

    for edge in kg.edges:
        edges.append(
            Edge(
                source=edge.source,
                target=edge.target,
                color="#CCCCCC",
                width=1.5,
            )
        )

    config = Config(
        width=900,
        height=600,
        directed=True,
        physics=True,
        hierarchical=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": False},
    )

    clicked_node = agraph(nodes=nodes, edges=edges, config=config)
    return clicked_node
