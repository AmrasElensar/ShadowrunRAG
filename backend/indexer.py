"""Incremental indexer for ChromaDB with embedding support and enhanced semantic chunking."""

import hashlib
import json
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
import ollama
from tqdm import tqdm
import logging
from tools.llm_classifier import create_two_tier_classifier  # Updated
from tools.improved_semantic_chunker import create_improved_semantic_chunker  # New
from tools.enhanced_query_processor import create_enhanced_query_processor  # New

# Check for optional dependencies
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available - using word-based chunking")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncrementalIndexer:
    """Manage document indexing with ChromaDB and Ollama embeddings."""

    def __init__(
        self,
        chroma_path: str = "data/chroma_db",
        collection_name: str = "shadowrun_docs",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 800,
        chunk_overlap: int = 150,         # Increased proportionally
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
        self.use_semantic_splitting = use_semantic_splitting

        # Track indexed files
        self.index_file = self.chroma_path / "indexed_files.json"
        self.indexed_files = self._load_index_metadata()

        # Set up tokenizer
        if TIKTOKEN_AVAILABLE:
            self.encoder = tiktoken.get_encoding("cl100k_base")
            self._count_tokens = lambda text: len(self.encoder.encode(text))
            logger.info(f"Using tiktoken for accurate token counting - chunk size: {self.chunk_size} tokens")
        else:
            self._count_tokens = lambda text: len(text.split())  # rough word count
            logger.info(f"Using word-based token approximation - chunk size: {self.chunk_size} words")

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
        """Updated to use clean semantic chunker with existing regex cleaner."""

        if not self.use_semantic_splitting:
            return self._chunk_text_simple(text, source)

        # Import the clean chunker
        if not hasattr(self, '_clean_chunker'):
            from tools.improved_classifier import create_semantic_chunker
            self._clean_chunker = create_semantic_chunker(chunk_size=800, overlap=150)

        return self._clean_chunker.chunk_document(text, source, self._count_tokens)

    def _chunk_text_simple(self, text: str, source: str) -> List[Dict]:
        """Simple word/token-based chunking fallback with enhanced metadata."""
        base_metadata = {
            "document_type": "rulebook",
            "edition": "SR5",
            "primary_section": "Unknown"
        }

        if TIKTOKEN_AVAILABLE:
            # Token-based chunking
            tokens = self.encoder.encode(text)
            chunks = []

            for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunk_text = self.encoder.decode(chunk_tokens)
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "metadata": {
                        **base_metadata,
                        "token_count": len(chunk_tokens),
                        "chunk_index": i // (self.chunk_size - self.chunk_overlap)
                    }
                })
        else:
            # Word-based chunking
            words = text.split()
            chunks = []

            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = " ".join(chunk_words)
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "metadata": {
                        **base_metadata,
                        "word_count": len(chunk_words),
                        "chunk_index": i // (self.chunk_size - self.chunk_overlap)
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
        """Index all markdown files with semantic chunk IDs and proper file tracking."""
        directory = Path(directory)
        md_files = list(directory.rglob("*.md"))

        files_to_index = []
        for file_path in md_files:
            file_hash = self._get_file_hash(file_path)
            relative_path = str(file_path.relative_to(directory))

            if force_reindex or relative_path not in self.indexed_files or self.indexed_files[
                relative_path] != file_hash:
                files_to_index.append(file_path)
                self.indexed_files[relative_path] = file_hash

        if not files_to_index:
            logger.info("No new or changed files to index")
            return

        logger.info(f"Indexing {len(files_to_index)} files with semantic chunk IDs...")
        all_chunks = []

        for file_path in files_to_index:
            content = file_path.read_text(encoding='utf-8')
            chunks = self._chunk_text_semantic(content, str(file_path))
            all_chunks.extend(chunks)

        if all_chunks:
            texts = [chunk['text'] for chunk in all_chunks]
            embeddings = self._get_embeddings(texts)

            # FIXED: Use semantic IDs from chunker instead of MD5 hashes
            ids = [chunk['id'] for chunk in all_chunks]
            metadatas = [chunk['metadata'] for chunk in all_chunks]

            # Handle ID collisions with semantic IDs
            seen_ids = set()
            for i, id_ in enumerate(ids):
                original = id_
                counter = 1
                while id_ in seen_ids:
                    id_ = f"{original}_{counter}"
                    counter += 1
                ids[i] = id_
                seen_ids.add(id_)

                # CRITICAL: Update linking metadata if ID was changed
                if id_ != original:
                    # Update this chunk's metadata links
                    for link_field in ['next_chunk_global', 'prev_chunk_global']:
                        if metadatas[i].get(link_field) == original:
                            metadatas[i][link_field] = id_

                    # Update other chunks that reference this chunk
                    for j, other_metadata in enumerate(metadatas):
                        for link_field in ['next_chunk_global', 'prev_chunk_global']:
                            if other_metadata.get(link_field) == original:
                                other_metadata[link_field] = id_

            cleaned_metadatas = self._clean_metadata_for_chromadb(metadatas)

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=cleaned_metadatas
            )

            logger.info(f"Added {len(all_chunks)} chunks with semantic IDs")

            # Log stats (existing logging code)
            doc_types = {}
            editions = {}
            sections = {}
            primary_sections = {}
            content_types = {}
            chunk_links = 0

            for meta in metadatas:
                doc_type = meta.get('document_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

                edition = meta.get('edition', 'unknown')
                editions[edition] = editions.get(edition, 0) + 1

                sections_list = meta.get('sections', [])
                if isinstance(sections_list, str):
                    sections_list = sections_list.split(',')

                for section in sections_list:
                    section = section.strip()
                    sections[section] = sections.get(section, 0) + 1

                primary = meta.get('primary_section', 'General')
                primary_sections[primary] = primary_sections.get(primary, 0) + 1

                content_type = meta.get('content_type', 'general')
                content_types[content_type] = content_types.get(content_type, 0) + 1

                if meta.get('next_chunk_global') or meta.get('prev_chunk_global'):
                    chunk_links += 1

            logger.info(f"Document types: {doc_types}")
            logger.info(f"Editions: {editions}")
            logger.info(f"Primary sections: {primary_sections}")
            logger.info(f"All sections (multi-label): {dict(list(sections.items())[:10])}")
            logger.info(f"Content types: {content_types}")
            logger.info(f"Chunks with sequential links: {chunk_links}/{len(metadatas)}")

            unique_sections = len(sections)
            classification_diversity = unique_sections / len(metadatas) * 100 if metadatas else 0

            logger.info(
                f"Classification diversity: {classification_diversity:.1f}% ({unique_sections} unique sections)")

            rules_count = content_types.get('explicit_rule', 0)
            examples_count = content_types.get('example', 0)

            logger.info(f"Content analysis: {rules_count} explicit rules, {examples_count} examples")

            if classification_diversity < 5:
                logger.warning("⚠️ Low classification diversity - check classification patterns")

            if primary_sections.get('Combat', 0) / len(metadatas) > 0.8:
                logger.warning("⚠️ Over 80% classified as Combat - classification may be broken")

        self._save_index_metadata()

    def _clean_metadata_for_chromadb(self, metadatas: List[Dict]) -> List[Dict]:
        """UPDATED to handle new metadata fields."""
        cleaned_metadatas = []

        for metadata in metadatas:
            cleaned = {}

            for key, value in metadata.items():
                # Skip None values entirely
                if value is None:
                    continue

                # Handle lists (convert to comma-separated strings)
                if isinstance(value, list):
                    if value:  # Only if list is not empty
                        cleaned[key] = ",".join(str(item) for item in value if item is not None)
                    continue

                # Handle dictionaries (convert to JSON strings)
                elif isinstance(value, dict):
                    if value:  # Only if dict is not empty
                        import json
                        cleaned[key] = json.dumps(value)
                    continue

                # Handle basic types
                elif isinstance(value, (str, int, float, bool)):
                    cleaned[key] = value
                else:
                    # Convert other types to string
                    cleaned[key] = str(value)

            cleaned_metadatas.append(cleaned)

        return cleaned_metadatas
    
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