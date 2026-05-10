import streamlit as st


def render_sidebar() -> tuple[str, str, int] | None:
    """Render sidebar controls. Returns (topic, level, max_depth) or None if not submitted."""
    with st.sidebar:
        st.header("🧠 AI Learning Roadmap")
        st.markdown("Enter a complex topic and get an interactive learning path.")

        topic = st.text_input(
            "Topic to learn",
            placeholder="e.g. Gradient Descent, Neural Networks, Docker...",
        )

        user_level = st.selectbox(
            "Your knowledge level",
            options=["beginner", "intermediate", "advanced"],
            index=0,
        )

        max_depth = st.slider(
            "Decomposition depth",
            min_value=1,
            max_value=4,
            value=2,
            help="How deep to decompose concepts (higher = more detail, slower)",
        )

        submitted = st.button("🚀 Build Roadmap", type="primary", use_container_width=True)

        if submitted and topic.strip():
            return topic.strip(), user_level, max_depth

        if submitted and not topic.strip():
            st.warning("Please enter a topic.")

    return None
