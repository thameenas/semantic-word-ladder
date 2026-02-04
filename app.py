import streamlit as st

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

# ---- Output area ----
if run:
    st.info("Next step: run A* search here")
