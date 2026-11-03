"""Small Streamlit UI for the injectable financial-document pipeline."""

import uuid

import streamlit as st

from src.data_loader import load_saved_sample
from src.pipeline import PipelineStrategies, run_pipeline


@st.cache_resource
def load_app():
    return load_saved_sample(), PipelineStrategies()


def main():
    st.set_page_config(page_title="Financial Document Copilot", page_icon="📄")
    st.title("Financial Document Copilot")
    st.caption("Answers are retrieved from a 10-document sample; this is a project demo.")

    try:
        documents, strategies = load_app()
    except FileNotFoundError:
        st.error("Sample data is missing. Run `uv run python scripts/download_sample.py`.")
        return

    st.write(f"Loaded {len(documents)} sample documents.")
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    question = st.text_input("Ask a question about the documents")
    uploaded_image = st.file_uploader("Optional image", type=["png", "jpg", "jpeg"])
    if st.button("Ask") and question:
        image = uploaded_image.getvalue() if uploaded_image else None
        try:
            result = run_pipeline(
                question,
                documents,
                session_id=st.session_state.session_id,
                image=image,
                strategies=strategies,
            )
        except ValueError as exc:
            st.warning(str(exc))
            return

        st.subheader("Answer")
        st.write(result["answer"]["answer"])
        if result["answer"]["should_refuse"]:
            st.info(result["answer"]["reason"])
        st.caption(f"Citations: {result['answer']['citations'] or 'None'}")
        with st.expander("Pipeline details"):
            st.write("Conversation history", result["history"])
            st.write(" → ".join(result["steps"]))
            st.json(result["metrics"])
            st.write("Retrieved evidence")
            for item in result["evidence"]:
                st.markdown(f"**{item.doc_id}** — {item.question}")
                st.write(item.answer)


if __name__ == "__main__":
    main()
