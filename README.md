# Skåne RAG (Retrieval‑Augmented Generation)

RAG pipeline for answering questions about mobility in Skåne based on PDF reports.

## Structure

- `data/input/` – put your PDF reports here (Skåne mobility docs).
- `src/` – core Python modules:
  - `config.py` – global configuration and paths.
  - `data_processor.py` – PDF extraction and chunking utilities.
  - `vector_store.py` – ChromaDB integration and similarity search.
  - `rag_pipeline.py` – retrieval + LLM prompting + citations.
- `app.py` – Streamlit chat interface (entry point).

## Quickstart

1. **Create and activate a virtual environment (optional but recommended)**:

   ```bash
   cd rag-remote-llm-db
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Add PDFs**:

   - Place your Skåne mobility PDF reports into `data/input/`.
   - Implement the TODOs in `data_processor.py` for actual PDF text extraction.

4. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

The current code is scaffolded: core flows and interfaces are defined, but PDF extraction, embeddings, and LLM calls are left as clearly-marked `NotImplementedError` placeholders for you to fill in based on your environment and model choices.


## RAG Evaluation Scripts

1. **generate_eval_dataset.py**

   **Purpose**: Creates `eval_dataset.json` with paraphrased queries targeting factual claims from transport documents.

   #### Features
   - Generates 6+ queries (e.g., "Vad är tidsperioden för programmet?")
   - 3 temperatures × 9+ repeated runs per query
   - Captures answers + retrieved sources for evaluation

   ### Usage
      ```bash
      python src/metrics/generate_eval_dataset.py # → eval_dataset.json (162+ evaluations)
      ```

2. **evaluate_rag.py**

   **Purpose**: Computes RAG metrics using fuzzy matching and citation validation.

   ### Usage
      ```bash
      python src/metrics/evaluate_rag.py
      ```

