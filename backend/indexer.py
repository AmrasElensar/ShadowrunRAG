"""Incremental indexer for ChromaDB with embedding support and semantic chunking."""

import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import ollama
from tqdm import tqdm
import logging
import tiktoken
# -----------------------------
# Optional: For better chunking
try:
    from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OllamaEmbeddings  # Optional: if using langchain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not installed. Install with: pip install langchain langchain-community")

# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncrementalIndexer:
    """Manage document indexing with ChromaDB and Ollama embeddings."""
    
    def __init__(
        self,
        chroma_path: str = "data/chroma_db",
        collection_name: str = "shadowrun_docs",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 512,           # now in tokens
        chunk_overlap: int = 100,
        use_semantic_splitting: bool = True
    ):
        self.chroma_path = Path(chroma_path)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_splitting = use_semantic_splitting and LANGCHAIN_AVAILABLE
        
        # Track indexed files
        self.index_file = self.chroma_path / "indexed_files.json"
        self.indexed_files = self._load_index_metadata()
        
        # Token counter (simple approximation if tiktoken not available)
        try:
            import tiktoken
            self._tokenize = lambda text: tiktoken.get_encoding("cl100k_base").encode(text)
        except ImportError:
            logger.warning("tiktoken not installed. Falling back to rough token estimation.")
            self._tokenize = lambda text: text.split()  # rough estimate

    def _load_index_metadata(self) -> Dict:
        """Load metadata about previously indexed files."""
        if self.index_file.exists():
            return json.loads(self.index_file.read_text())
        return {}
    
    def _save_index_metadata(self):
        """Save metadata about indexed files."""
        self.index_file.write_text(json.dumps(self.indexed_files, indent=2))
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content for change detection."""
        return hashlib.md5(file_path.read_bytes()).hexdigest()

    def _chunk_text_semantic(self, text: str, source: str) -> List[Dict]:
        """
        Split text using Markdown headers and token-aware fallback.
        Ideal for unstructured-processed Markdown files.
        """
        headers_to_split_on = [
            ("#", "Section"),  # e.g., "Combat"
            ("##", "Subsection"),  # e.g., "Ranged Attacks"
            ("###", "Subsubsection"),  # e.g., "Recoil"
        ]

        # First: split by headers
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        try:
            header_chunks = markdown_splitter.split_text(text)
        except Exception as e:
            logger.warning(f"Header splitting failed: {e}. Falling back to recursive.")
            header_chunks = [type('Chunk', (), {'page_content': text, 'metadata': {}})]

        # Second: split large chunks by token count
        final_chunks = []
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name="cl100k_base"
            )
        except:
            logger.warning("tiktoken unavailable. Using fallback splitter.")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len
            )

        for chunk in header_chunks:
            content = chunk.page_content
            metadata = chunk.metadata

            if len(encoder.encode(content)) > self.chunk_size:
                split_docs = splitter.split_text(content)
                for i, doc in enumerate(split_docs):
                    final_chunks.append({
                        "text": doc,
                        "source": source,
                        "metadata": {
                            **metadata,
                            "source": source,
                            "chunk_part": i,
                            "token_count": len(encoder.encode(doc))
                        }
                    })
            else:
                final_chunks.append({
                    "text": content,
                    "source": source,
                    "metadata": {
                        **metadata,
                        "source": source,
                        "token_count": len(encoder.encode(content))
                    }
                })

        return final_chunks

    def _chunk_text_simple(self, text: str, source: str) -> List[Dict]:
        """Simple word-based chunking (fallback)."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "source": source,
                "metadata": {
                    "source": source,
                    "word_count": len(chunk_words)
                }
            })
        return chunks

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from Ollama."""
        embeddings = []
        for text in tqdm(texts, desc="Generating embeddings"):
            try:
                response = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=text
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                embeddings.append([0.0] * 768)  # fallback
        return embeddings

    def index_directory(self, directory: str, force_reindex: bool = False):
        """Index all markdown files in a directory."""
        directory = Path(directory)
        md_files = list(directory.rglob("*.md"))
        
        files_to_index = []
        for file_path in md_files:
            file_hash = self._get_file_hash(file_path)
            relative_path = str(file_path.relative_to(directory))
            
            if force_reindex or relative_path not in self.indexed_files or self.indexed_files[relative_path] != file_hash:
                files_to_index.append(file_path)
                self.indexed_files[relative_path] = file_hash
        
        if not files_to_index:
            logger.info("No new or changed files to index")
            return
        
        logger.info(f"Indexing {len(files_to_index)} files...")
        all_chunks = []

        for file_path in files_to_index:
            content = file_path.read_text(encoding='utf-8')
            chunks = self._chunk_text_semantic(content, str(file_path))
            all_chunks.extend(chunks)
        
        if all_chunks:
            texts = [chunk['text'] for chunk in all_chunks]
            embeddings = self._get_embeddings(texts)
            
            ids = [f"{hashlib.md5(chunk['text'].encode()).hexdigest()[:16]}" for chunk in all_chunks]
            metadatas = [chunk['metadata'] for chunk in all_chunks]
            
            # Avoid ID collisions
            seen_ids = set()
            for i, id_ in enumerate(ids):
                original = id_
                counter = 1
                while id_ in seen_ids:
                    id_ = f"{original}_{counter}"
                    counter += 1
                ids[i] = id_
                seen_ids.add(id_)

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(all_chunks)} chunks to index")
        
        self._save_index_metadata()
    
    def remove_document(self, file_path: str):
        """Remove a document from the index."""
        # Normalize path
        file_path = str(Path(file_path).resolve())
        results = self.collection.get(
            where={"source": file_path}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Removed {len(results['ids'])} chunks for {file_path}")
        
        # Update metadata
        try:
            relative_path = str(Path(file_path).relative_to("data/processed_markdown"))
            if relative_path in self.indexed_files:
                del self.indexed_files[relative_path]
                self._save_index_metadata()
        except ValueError:
            pass  # Not under processed_markdown