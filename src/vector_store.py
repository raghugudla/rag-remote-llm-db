"""
Pinecone vector store for RAG
Compatible: dense1, cosine, 384 dims.
"""

from typing import List, Dict, Any
import logging
from pathlib import Path
import pinecone
from sentence_transformers import SentenceTransformer
from src.config import config

logger = logging.getLogger(__name__)

class VectorStore:
    """Pinecone-backed vector store with full chunk debugging."""
    BATCH_SIZE = 100

    def __init__(self, namespace: str = ""):  # '' matches your 3118 vectors!
        self.namespace = namespace
        print(f"[DEBUG VectorStore] namespace='{self.namespace}' (vectors should be here)")
        self.embedding_model = SentenceTransformer(config.embedding_model_name)
        print(f"[DEBUG VectorStore] embedding_model='{config.embedding_model_name}', dim={config.embedding_dim}")
        self.pc = pinecone.Pinecone(api_key=config.pinecone_api_key)
        self.index_name = config.pinecone_index_name
        print(f"[DEBUG VectorStore] index_name='{self.index_name}'")
        print(f"[DEBUG VectorStore] available_indexes={[idx.name for idx in self.pc.list_indexes()]}")
        
        # Connect (no auto-create for custom index)
        self.index = self.pc.Index(self.index_name)
        stats = self.index.describe_index_stats()
        print(f"[DEBUG VectorStore] CONNECTED: {stats}")
        print(f"[DEBUG VectorStore] namespaces: {stats.namespaces}")
        if self.namespace in stats.namespaces:
            print(f"[DEBUG VectorStore] '{self.namespace}' has {stats.namespaces[self.namespace].vector_count} vectors")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Batch upsert with debug."""
        if not chunks:
            print("[DEBUG add_chunks] No chunks to add")
            return
        print(f"[DEBUG add_chunks] Adding {len(chunks)} chunks to '{self.namespace}'")
        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [{"source": c["metadata"]["source"],
                      "page": c["metadata"]["page"],
                      "text": c["text"]} for c in chunks]
        print(f"[DEBUG add_chunks] Sample: {texts[0][:100]}...")
        embeddings = self.embedding_model.encode(texts).tolist()
        vectors = list(zip(ids, embeddings, metadatas))
        for i in range(0, len(vectors), self.BATCH_SIZE):
            batch = vectors[i:i + self.BATCH_SIZE]
            self.index.upsert(vectors=batch, namespace=self.namespace)
        print(f"[DEBUG add_chunks] ✅ Upserted {len(chunks)} to '{self.namespace}'")
        logger.info(f"Upserted {len(chunks)} chunks")

    def query(self, query_text: str, n_results: int = None) -> Dict[str, Any]:
        """FULL DEBUG: Print every retrieved chunk + scores."""
        if n_results is None:
            n_results = config.top_k
        print(f"\n{'='*60}")
        print(f"[DEBUG query] QUERY: '{query_text}' | top_k={n_results} | namespace='{self.namespace}'")
        query_emb = self.embedding_model.encode([query_text]).tolist()[0]
        print(f"[DEBUG query] query_emb shape={len(query_emb)}, first5={query_emb[:5]}")
        
        results = self.index.query(
            vector=query_emb,
            top_k=n_results,
            include_metadata=True,
            include_values=False,  # Save bandwidth
            namespace=self.namespace
        )
        
        print(f"[DEBUG query] Pinecone RAW: matches={len(results.matches)}")
        if results.matches:
            print(f"[DEBUG query] TOP SCORES: {[f'{m.score:.3f}' for m in results.matches[:3]]}")
        
        documents = []
        metadatas = []
        distances = []
        for i, match in enumerate(results.matches):
            text = match.metadata.get("text", "")[:200] + "..." if len(match.metadata.get("text", "")) > 200 else match.metadata.get("text", "")
            source = match.metadata.get("source", "unknown")
            page = match.metadata.get("page", "N/A")
            print(f"[DEBUG CHUNK {i+1}] score={match.score:.3f} | source={source} | page={page}")
            print(f"           TEXT: {text}")
            print()
            documents.append(match.metadata.get("text", ""))
            metadatas.append(match.metadata)
            distances.append(match.score)
        
        avg_score = sum(distances)/len(distances) if distances else 0.0
        print(f"[DEBUG query] RETURNING: {len(documents)} docs | avg_score={avg_score:.3f}")
#        print(f"[DEBUG query] RETURNING: {len(documents)} docs | avg_score={sum(distances)/len(distances):.3f if distances else 0:.3f}")
        print(f"{'='*60}\n")
        logger.debug(f"Query '{query_text[:50]}...': {len(documents)} results")
        return {"documents": documents, "metadatas": metadatas, "distances": distances}
