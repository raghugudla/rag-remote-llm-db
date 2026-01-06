"""
Streamlit app for RAG.
"""

import os
import streamlit as st
import pandas as pd
import altair as alt
import logging
from pathlib import Path

from src.data_processor import generate_chunks_from_pdfs, chunks_to_dicts
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from src.config import config
from src.metrics.generate_eval_dataset import generate_dataset
from src.metrics.evaluate_rag import run_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Skåne Mobility Assistant",
    layout="wide",
)

# -----------------------------
# Session state initialization
# -----------------------------
@st.cache_resource
def init_pipeline():
    """Cached pipeline init."""
    vector_store = VectorStore(namespace="") 
    rag = RAGPipeline(vector_store)
    return vector_store, rag

if "vector_store" not in st.session_state:
    st.session_state.vector_store, st.session_state.rag = init_pipeline()
    
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar: Controls & Metrics
# -----------------------------
st.sidebar.title("Mobility RAG Assistant")
st.sidebar.markdown("**Models:**")
st.sidebar.markdown(f"- Embeddings: {config.embedding_model_name}")
st.sidebar.markdown(f"- LLM: {config.llm_model_name}")

# Rebuild index
if st.sidebar.button("🔄 Rebuild Index", type="primary"):
    col1, col2 = st.columns(2)
    max_pdfs = col1.number_input("Max PDFs", min_value=1, max_value=10, value=3)
    max_chunks = col2.number_input("Max chunks/PDF", min_value=10, max_value=200, value=50)
    
    with st.spinner(f"Processing {max_pdfs} PDFs..."):
        pdf_paths = sorted(Path(config.raw_docs_dir).glob("*.pdf"))[:max_pdfs]
        chunks = generate_chunks_from_pdfs(pdf_paths)
        chunk_dicts = chunks_to_dicts(chunks[:max_chunks * max_pdfs])  # LIMIT!
        st.session_state.vector_store.add_chunks(chunk_dicts)
        st.sidebar.success(f"✅ Indexed {len(chunk_dicts)} chunks from {len(pdf_paths)} PDFs")

if st.sidebar.button("Generate Dataset"):
    with st.spinner("Generating evaluation dataset with questions, answers and citations"):
        generate_dataset()  # Or direct call to script logic
        
    st.sidebar.success("eval_dataset.json generated successfully!")

if st.sidebar.button("Run Evaluation"):
    with st.spinner("Running full RAG evaluation (correctness, citations, consistency, variance)..."):
        results = run_evaluation()
        
        st.subheader(" RAG Evaluation Results")
        
        # Per-temperature summary (variance across temps)
        st.markdown("### Per-Temperature Metrics")
        if results["temp_summary"]:
            temp_df = pd.DataFrame(results["temp_summary"])
            if 'Temperature' in temp_df.columns:
                st.table(temp_df.set_index('Temperature'))
            else:
                st.table(temp_df)
        else:
            st.warning("No temp summary data!")
        
        csv_path = results.get("csv_path", "")
        df = pd.read_csv(csv_path)

        # Row 2: CONSOLIDATED (mean across bases) + extras
        col3, col4 = st.columns(2)
        with col3:
            # Consolidated Correctness (mean per temp)
            df_mean = df.groupby('temp')[['correctness', 'consistency']].mean().reset_index()
            corr_consol = alt.Chart(df_mean).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('temp:Q', title="Temperature"),
                y=alt.Y('correctness:Q', title="Mean Correctness", scale=alt.Scale(domain=[0,1])),
                tooltip=['temp', 'correctness']
            ).properties(
                title="Correctness vs Temp",
                width=400,
                height=250
            )
            st.altair_chart(corr_consol, use_container_width=True)
        
        with col4:
            # Consolidated Consistency (mean per temp)
            cons_consol = alt.Chart(df_mean).mark_line(point=True, strokeWidth=3, color="orange").encode(
                x=alt.X('temp:Q', title="Temperature"),
                y=alt.Y('consistency:Q', title="Mean Consistency", scale=alt.Scale(domain=[0,1])),
                tooltip=['temp', 'consistency']
            ).properties(
                title="Consistency vs Temp",
                width=400,
                height=250
            )
            st.altair_chart(cons_consol, use_container_width=True)

        # Full variance across temperatures
        st.markdown("### Variance Across Temperatures")
        vars = results['variances']
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Correctness Var", f"{vars['correctness_var']:.3f}")
        with col2: st.metric("Citation Var", f"{vars['citation_var']:.3f}")
        with col3: st.metric("Consistency Var", f"{vars['consistency_var']:.3f}")
        #st.info(results["message"])
        
        # CSV download (safe)
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                csv_data = f.read()
            st.download_button(
                label="📥 Download Full Metrics CSV",
                data=csv_data,
                file_name="rag_eval_metrics.csv",
                mime="text/csv"
            )
        else:
            st.warning("CSV not generated - check eval logs.")
        
        # Quick charts (with fallback)
        try:
            # Per-base query summary (paraphrase robustness)
            st.markdown("### Per-Base Query Metrics")
            if results["base_summary"]:
                base_df = pd.DataFrame(results["base_summary"])
                if 'Base Query' in base_df.columns:
                    st.table(base_df.set_index('Base Query'))
                else:
                    st.table(base_df)  # Fallback
            else:
                st.warning("No base summary data - run dataset generation first!")
        
            col1, col2 = st.columns(2)
            with col1:
                corr_chart = alt.Chart(df).mark_line().encode(
                    x="temp:Q", y="correctness:Q", color="base_query:N"
                ).properties(title="Correctness vs Temperature")
                st.altair_chart(corr_chart, use_container_width=True)
            with col2:
                cons_chart = alt.Chart(df).mark_line().encode(
                    x="temp:Q", y="consistency:Q", color="base_query:N"
                ).properties(title="Consistency vs Temperature")
                st.altair_chart(cons_chart, use_container_width=True)
        except:
            st.info("Charts unavailable - download CSV for analysis.")
    
    st.sidebar.success("Full evaluation complete!")
    #st.balloons()


# -----------------------------
# Main Chat Interface
# -----------------------------
st.title("Skåne Mobility Assistant")
st.markdown("Fråga om transport, infrastruktur och resbeteende.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Källor"):
                for source, page in msg["sources"]:
                    st.markdown(f"- **{source}**, sida {page}")

# Chat input
if prompt := st.chat_input("Fråga om mobilitet i Skåne..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Söker i dokument..."):
            result = st.session_state.rag.answer(prompt)
            st.markdown(result["answer"])
        
        # Sources
        if result["sources"]:
            with st.expander("📚 Källor"):
                for source, page in result["sources"]:
                    st.markdown(f"- **{source}**, sida {page}")
    
    # Save to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": result["answer"],
        "sources": result["sources"]
    })