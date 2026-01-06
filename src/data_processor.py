"""
PDF data processing utilities.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import hashlib
import pypdf
import psutil  
from src.config import config
import logging
import gc

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Lightweight container for chunk metadata."""
    text: str
    source: str  # filename
    page: int  # 1-indexed
    chunk_id: str  # stable hash for upsert

def list_pdf_files(directory: Path = None) -> List[Path]:
    """Recursive list of all PDFs under directory."""
    if directory is None:
        directory = config.raw_docs_dir
    pdfs = sorted(p for p in directory.rglob("*.pdf") if p.is_file())
    print(f"[DEBUG list_pdf_files] Found {len(pdfs)} PDFs: { [p.name for p in pdfs[:5]] }...")
    return pdfs

def extract_pages_from_pdf(pdf_path: Path) -> List[str]:
    """Extract text page-by-page preserving page numbers."""
    pages: List[str] = []
    print(f"[DEBUG extract_pages] Processing {pdf_path.name} ({pdf_path.stat().st_size/1e6:.1f}MB)")
    try:
        with pdf_path.open("rb") as f:
            reader = pypdf.PdfReader(f)
            print(f"[DEBUG extract_pages] {pdf_path.name}: {len(reader.pages)} pages")
            for page_num, page in enumerate(reader.pages[:20], 1):  # Limit 20 pages
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append(text)
        print(f"[DEBUG extract_pages] {pdf_path.name}: extracted {len(pages)} non-empty pages")
    except Exception as e:
        print(f"[DEBUG ERROR extract_pages] {pdf_path}: {e}")
        logger.warning(f"Failed to extract {pdf_path}: {e}")
    return pages

def simple_recursive_splitter(
    text: str,
    chunk_size: int = config.chunk_size,
    chunk_overlap: int = config.chunk_overlap,
    max_depth: int = 3  # ADD RECURSION LIMIT
) -> List[str]:
    """Safe splitter - FIXED recursion."""
    print(f"[DEBUG splitter] len={len(text)} chars, depth max={max_depth}")
    if len(text) <= chunk_size or max_depth <= 0:
        print(f"[DEBUG splitter] Base case: returning 1 chunk ({len(text)} chars)")
        return [text.strip()]
    
    separators = ["\n\n", "\n", ". ", "! ", "? ", " "]
    best_chunks = [text[:chunk_size].strip()]  # Fallback
    
    for sep in separators:
        if sep in text:
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                last_sep = text.rfind(sep, start, end)
                if last_sep == -1:
                    last_sep = end
                chunk = text[start:last_sep].strip()
                if len(chunk) > 50:  # Min chunk size
                    chunks.append(chunk)
                start = max(last_sep - chunk_overlap, 0)
                if len(chunk) < chunk_size // 2:  # Prevent tiny chunks
                    break
            if len(chunks) > 1 and sum(len(c) for c in chunks) < len(text) * 0.9:
                print(f"[DEBUG splitter] Good split: {len(chunks)} chunks, recurse depth={max_depth-1}")
                return simple_recursive_splitter("\n\n".join(chunks), chunk_size, chunk_overlap, max_depth-1)
            break  # Use first good split
    
    print(f"[DEBUG splitter] Using fallback: {len(best_chunks)} chunks")
    return best_chunks

def chunks_to_dicts(chunks: List[DocumentChunk]) -> List[dict]:
    """Convert DocumentChunk to VectorStore dict format."""
    print(f"[DEBUG chunks_to_dicts] Converting {len(chunks)} chunks")
    return [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "metadata": {"source": c.source, "page": c.page}
        }
        for c in chunks
    ]

def generate_chunks_from_pdfs(
    pdf_paths: Iterable[Path] = None,
    max_pages_per_pdf: int = 10,
    max_chunks_per_pdf: int = 50
) -> List[DocumentChunk]:
    """End-to-end: PDFs -> pages -> chunks with metadata. MEMORY-SAFE."""
    print(f"[DEBUG generate_chunks] RAM before: {psutil.virtual_memory().percent}% used")
    if pdf_paths is None:
        pdf_paths = list_pdf_files()
    
    all_chunks: List[DocumentChunk] = []
    total_processed = 0
    
    for i, pdf_path in enumerate(pdf_paths):
        print(f"[DEBUG generate_chunks] PDF {i+1}/{len(pdf_paths)}: {pdf_path.name}")
        source_name = pdf_path.name
        
        pages = extract_pages_from_pdf(pdf_path)
        pdf_chunks = []
        
        for page_num, page_text in enumerate(pages[:max_pages_per_pdf], 1):
            if not page_text.strip():
                continue
            print(f"[DEBUG] Page {page_num}: {len(page_text)} chars")
            page_chunks = simple_recursive_splitter(page_text)
            print(f"[DEBUG] Page {page_num}: {len(page_chunks)} chunks")
            
            for chunk_idx, chunk_text in enumerate(page_chunks[:max_chunks_per_pdf]):
                stable = f"{source_name}:{page_num}:{chunk_idx}:{chunk_text[:50]}"
                chunk_hash = hashlib.md5(stable.encode("utf-8")).hexdigest()
                if len(chunk_text.strip()) > 100:  # Min 100 chars
                    pdf_chunks.append(DocumentChunk(
                        text=chunk_text.strip(),
                        source=source_name,
                        page=page_num,
                        chunk_id=chunk_hash
                    ))
                else:
                    print(f"[DEBUG SKIPPED] Tiny chunk: {len(chunk_text)} chars")
                total_processed += 1
                if total_processed % 100 == 0:
                    print(f"[DEBUG] RAM check: {psutil.virtual_memory().percent}% used, {total_processed} chunks")
        
        all_chunks.extend(pdf_chunks[:max_chunks_per_pdf])
        print(f"[DEBUG] {source_name}: {len(pdf_chunks)} chunks added (total={len(all_chunks)})")
        gc.collect()  # Critical: free memory after each PDF
    
    print(f"[DEBUG generate_chunks] FINAL: {len(all_chunks)} chunks, RAM: {psutil.virtual_memory().percent}%")
    logger.info(f"Generated {len(all_chunks)} chunks from {len(pdf_paths)} PDFs")
    return all_chunks
