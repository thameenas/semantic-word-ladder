import streamlit as st
from src.astar import astar_search
from src.visualize_streamlit import plot_path
from src.visualize_plotly import plot_path_3d

st.set_page_config(
    page_title="Semantic Word Ladder",
    layout="centered"
)

st.title("🔗 Semantic Word Ladder")
st.caption("Embeddings + FAISS + A* Search")

# ---- Inputs ----
col1, col2 = st.columns(2)

with col1:
    start_word = st.text_input("Start word", value="king")

with col2:
    end_word = st.text_input("End word", value="queen")

run = st.button("Find semantic path")

st.divider()

if run:
    if not start_word or not end_word:
        st.error("Please enter both start and end words.")
    else:
        with st.spinner("Searching semantic space..."):
            path = astar_search(start_word, end_word)

        if path is None:
            st.warning("No semantic path found.")
        else:
            st.markdown("Semantic ladder found!")
            st.success(" → ".join(f"**{w}**" for w in path))
            fig = plot_path_3d(path, method="umap")
            st.plotly_chart(fig, use_container_width=True)

