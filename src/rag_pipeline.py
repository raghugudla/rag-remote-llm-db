"""
RAG Pipeline using Groq + Pinecone.
Key goals: citation-accurate, context-only answers.
"""

from typing import Dict, Any, List, Tuple

import logging
import re
from groq import Groq
from src.config import config

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(
        self,
        vector_store: 'VectorStore',
        llm_model: str = None,
        top_k: int = None
    ):
        self.vector_store = vector_store
        self.llm_model = llm_model or config.llm_model_name
        self.top_k = top_k or config.top_k
        self.client = Groq(api_key=config.groq_api_key)
        print(f"[DEBUG RAGPipeline] Initialized with top_k={self.top_k}, llm={self.llm_model}")

    def format_context(self, results: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]]]:
        """Format retrieval results with citations."""
        print(f"[DEBUG format_context] Full results keys: {list(results.keys())}")
        print(f"[DEBUG format_context] results['documents'] type/len: {type(results.get('documents'))}, len={len(results.get('documents', []))}")
        print(f"[DEBUG format_context] results['metadatas'] type/len: {type(results.get('metadatas'))}, len={len(results.get('metadatas', []))}")
        print(f"[DEBUG format_context] results['distances'] type/len: {type(results.get('distances'))}, len={len(results.get('distances', []))}")

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        print(f"[DEBUG format_context FIXED] Using all: docs={len(documents)}, metas={len(metadatas)}")

        print(f"[DEBUG format_context] After slicing - docs len: {len(documents) if documents else 0}, metas len: {len(metadatas) if metadatas else 0}")

        blocks = []
        sources = []
        if documents and metadatas:
            for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
                if isinstance(meta, str):
                    source, page = meta, "N/A"
                else:
                    source = meta.get("source", "unknown")
                    page = meta.get("page", "N/A")
                sources.append((source, page))
                blocks.append(f"[Context {i} | {source}, Page {page}]\n{doc}")
            print(f"[DEBUG format_context] Built {len(blocks)} context blocks")
        else:
            print("[DEBUG format_context] No documents or metadatas - returning empty context")

        context = "\n\n".join(blocks)
        return context, sources

    @staticmethod
    def build_prompt(question: str, context: str) -> str:
        """Strict RAG prompt with citation rules."""
        return f"""Du är en akademisk assistent specialiserad på mobilitet och transport i Skåne.

    KONTEXT (endast officiella dokument):
    {context}

    FRÅGA: {question}

    SVARSREGLER:
    - Svara ENDAST med information från KONEXTEN. Ingen extern kunskap.
    - Svara på svenska om inte annat anges.
    - Var kort och faktabaserad.
    - Citera ALLA fakta med format: [Källa: KällaNamn, Sida: sidnummer]
    (Använd exakt format från kontext headers)
    - Om frågan inte kan besvaras från kontexten: "Information saknas i dokumenten."
    - För dokumentlista: namnge dem exakt.

    SVAR:"""

    @staticmethod
    def extract_year_range(text: str) -> str | None:
        """Extract year ranges like '2022-2026'."""
        match = re.search(r'\b(19|20)\d{2}-(19|20)\d{2}\b', text)
        return match.group(0) if match else None

    @staticmethod
    def is_document_list_question(question: str) -> bool:
        """Detect 'what documents' questions."""
        q_lower = question.lower()
        phrases = ["vilka dokument", "indexerats", "använder systemet", "documents"]
        return any(p in q_lower for p in phrases)

    def get_indexed_documents(self) -> List[str]:
        """List unique sources in index."""
        print("[DEBUG get_indexed_documents] Fetching index stats...")
        # Note: Full list requires query or metadata fetch
        docs = ["trafikforsorjningsprogrammet.pdf", "regional-utvecklingsstrategi.pdf"]  # Update from actual metadata
        print(f"[DEBUG get_indexed_documents] Returning docs: {docs}")
        return docs

    def answer(self, question: str, temperature: float = 0.0) -> Dict[str, Any]:
        """Full RAG: retrieve -> prompt -> generate."""
        print(f"[DEBUG answer] Question: '{question}' (temp={temperature})")

        if self.is_document_list_question(question):
            print("[DEBUG answer] Document list question detected")
            docs = self.get_indexed_documents()
            return {
                "answer": f"Indexerade dokument:\n" + "\n".join(f"- {d}" for d in docs),
                "sources": []
            }

        # Retrieve
        print(f"[DEBUG answer] Calling vector_store.query with top_k={self.top_k}")
        results = self.vector_store.query(question, self.top_k)
        print(f"[DEBUG answer] Query returned: keys={list(results.keys())}")

        context, sources = self.format_context(results)
        print(f"[DEBUG answer] Context len: {len(context)}, sources len: {len(sources)}")

        if not context.strip():
            print("[DEBUG answer] Empty context - returning no info")
            return {"answer": "Ingen relevant information hittades.", "sources": []}

        prompt = self.build_prompt(question, context)
        print(f"[DEBUG answer] Prompt built (len={len(prompt)} chars)")

        try:
            print(f"[DEBUG answer] Calling Groq with model={self.llm_model}")
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024
            )
            answer_text = completion.choices[0].message.content.strip()
            print("[DEBUG answer] Groq response received")
        except Exception as e:
            logger.error(f"Groq error: {e}")
            answer_text = "Kunde inte generera svar på grund av tekniskt fel."
            print(f"[DEBUG answer] Groq error: {e}")

        print("[DEBUG answer] Returning result")
        return {"answer": answer_text, "sources": sources}
