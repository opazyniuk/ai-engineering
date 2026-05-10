import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

from src.agent.graph import expand_concept, run_decomposition
from src.models.concept import KnowledgeGraph
from src.ui.graph_view import render_graph
from src.ui.sidebar import render_sidebar

st.set_page_config(
    page_title="AI Learning Roadmap Assistant",
    page_icon="🧠",
    layout="wide",
)


def main():
    result = render_sidebar()

    if result is not None:
        topic, user_level, max_depth = result
        with st.spinner(f"Building learning roadmap for **{topic}**..."):
            state = run_decomposition(topic, user_level, max_depth)
            st.session_state["agent_state"] = state
            st.session_state["knowledge_graph"] = state["knowledge_graph"]

    # Show graph if we have one
    kg: KnowledgeGraph | None = st.session_state.get("knowledge_graph")

    if kg is None:
        st.markdown("## 🧠 AI Learning Roadmap Assistant")
        st.markdown(
            "Enter a complex topic in the sidebar and get an interactive mind-map "
            "showing the prerequisite concepts you need to learn, from advanced to basics."
        )
        st.markdown("---")
        st.markdown(
            "**Example topics:** Gradient Descent, Transformer Architecture, "
            "Docker Containers, RSA Encryption, Backpropagation"
        )
        return

    st.markdown(f"### Learning Roadmap: {st.session_state['agent_state']['topic']}")

    col_graph, col_detail = st.columns([3, 1])

    with col_graph:
        clicked_node_id = render_graph(kg)

    with col_detail:
        st.markdown("#### 📖 Concept Details")

        if clicked_node_id and clicked_node_id in kg.concepts:
            concept = kg.concepts[clicked_node_id]
            st.markdown(f"**{concept.name}**")
            st.markdown(f"*Level: {concept.level}*")
            st.markdown("---")
            st.markdown(concept.explanation or "_No explanation yet._")

            if not concept.is_expanded:
                if st.button("🔍 Expand deeper", key=f"expand_{concept.id}"):
                    with st.spinner(f"Expanding **{concept.name}**..."):
                        state = st.session_state["agent_state"]
                        new_state = expand_concept(state, concept.id)
                        st.session_state["agent_state"] = new_state
                        st.session_state["knowledge_graph"] = new_state["knowledge_graph"]
                        st.rerun()
        else:
            st.info("Click on a node in the graph to see its explanation.")

    # Stats
    st.markdown("---")
    cols = st.columns(3)
    cols[0].metric("Total Concepts", len(kg.concepts))
    cols[1].metric("Connections", len(kg.edges))
    cols[2].metric("Max Depth", max((c.level for c in kg.concepts.values()), default=0))


if __name__ == "__main__":
    main()
