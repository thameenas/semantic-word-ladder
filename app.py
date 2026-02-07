import streamlit as st
from src.astar import astar_search

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

st.subheader("Search controls")

alpha = st.slider(
    "Alpha (goal pull)",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Higher = greedier toward target, Lower = smoother ladders"
)

k = st.slider(
    "k (neighbors per step)",
    min_value=2,
    max_value=15,
    value=5,
    step=1,
    help="Higher = more exploration, Lower = tighter local steps"
)

run = st.button("Find semantic path")

st.divider()

if run:
    if not start_word or not end_word:
        st.error("Please enter both start and end words.")
    else:
        try:
            with st.spinner("Searching semantic space..."):
                path = astar_search(start_word, end_word, alpha=alpha, k=k)

            if path is None:
                st.warning("No semantic path found.")
            else:
                st.markdown("Semantic ladder found!")
                st.success(" → ".join(f"**{w}**" for w in path))
        except (ValueError, KeyError) as e:
            st.error(f"❌ Word {e} not in vocabulary. Please try another word.")
            st.info("The vocabulary contains the 10,000 most common English words.")
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred. Please try again.")

