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

# Check for optional dependencies
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available - using word-based chunking")

try:
    from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not installed. Install with: pip install langchain langchain-community")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncrementalIndexer:
    """Manage document indexing with ChromaDB and Ollama embeddings."""

    def __init__(
        self,
        chroma_path: str = "data/chroma_db",
        collection_name: str = "shadowrun_docs",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 2024,           # Increased from 512 for better performance
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
        self.use_semantic_splitting = use_semantic_splitting and LANGCHAIN_AVAILABLE

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

    def _extract_shadowrun_metadata(self, content: str, source: str) -> Dict:
        """Extract Shadowrun-specific metadata from content with enhanced fields."""
        metadata = {
            "source": source,
            "document_type": "unknown",
            "edition": "unknown",
            "main_section": "General",
            "subsection": "General",
            # NEW: Enhanced fields for chunking
            "section_id": None,
            "section_title": None,
            "chunk_index": None,
            "total_chunks_in_section": None,
            "section_complete": None,
            "prev_chunk_global": None,
            "next_chunk_global": None,
            "prev_chunk_section": None,
            "next_chunk_section": None,
            "chunk_id": None,
            "continuation_of": None,
            "token_count": None
        }

        # Existing detection logic stays the same...
        filename = Path(source).name.lower()
        content_lower = content[:2000].lower()

        # Document type detection (existing code)
        if any(term in filename for term in ["character", "sheet", "pc", "npc"]):
            metadata["document_type"] = "character_sheet"
        elif any(term in content_lower for term in
                 ["character creation", "priority system", "attribute", "skill points"]):
            metadata["document_type"] = "character_sheet"
        elif any(term in filename for term in ["core", "rulebook", "rules", "manual"]):
            metadata["document_type"] = "rulebook"
        elif any(term in content_lower for term in ["combat", "initiative", "damage", "armor", "weapon"]):
            metadata["document_type"] = "rulebook"
        else:
            metadata["document_type"] = "universe_info"

        # Edition detection (existing code)
        if "shadowrun 6" in content_lower or "6th edition" in content_lower or "sr6" in content_lower:
            metadata["edition"] = "SR6"
        elif "shadowrun 5" in content_lower or "5th edition" in content_lower or "sr5" in content_lower:
            metadata["edition"] = "SR5"
        elif "shadowrun 4" in content_lower or "4th edition" in content_lower or "sr4" in content_lower:
            metadata["edition"] = "SR4"
        elif "shadowrun 3" in content_lower or "3rd edition" in content_lower or "sr3" in content_lower:
            metadata["edition"] = "SR3"

        # Section detection (existing code - now serves as fallback)
        section_keywords = {
            "Combat": ["combat", "initiative", "damage", "armor", "weapon", "attack", "defense", "wound"],
            "Magic": ["magic", "spell", "astral", "summoning", "enchanting", "adept", "mage", "spirit"],
            "Matrix": ["matrix", "hacking", "cyberdeck", "programs", "ic", "host", "decker", "technomancer"],
            "Riggers": ["rigger", "drone", "vehicle", "pilot", "autosofts", "jumped in"],
            "Gear": ["gear", "equipment", "cyberware", "bioware", "weapons", "armor", "electronics"],
            "Character Creation": ["character creation", "priority", "attributes", "skills", "metatype",
                                   "build points"],
            "Gamemaster": ["gamemaster", "gm", "adventure", "npc", "campaign", "plot", "scenario"],
            "Setting": ["seattle", "sixth world", "corporations", "shadowrun", "awakening", "crash"]
        }

        for section, keywords in section_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                metadata["main_section"] = section
                break

        return metadata

    def _chunk_text_semantic(self, text: str, source: str) -> List[Dict]:
        """Simplified semantic chunking: Split by # boundaries, simple paragraph fallback."""
        if not self.use_semantic_splitting:
            return self._chunk_text_simple(text, source)

        # Extract base metadata for all chunks
        base_metadata = self._extract_shadowrun_metadata(text, source)

        # Split by single # headers (major sections)
        main_sections = self._split_by_main_headers(text)

        if not main_sections:
            # Fallback to simple chunking if no headers found
            logger.warning(f"No main headers found in {source}, using simple chunking")
            return self._chunk_text_simple(text, source)

        final_chunks = []

        for section_data in main_sections:
            section_title = section_data['title']
            section_content = section_data['content']
            section_id = self._generate_section_id(section_title)

            # Check if section fits in one chunk (2048 tokens)
            token_count = self._count_tokens(section_content)

            if token_count <= 2048:
                # Single chunk for this section
                chunk_metadata = {
                    **base_metadata,
                    "section_id": section_id,
                    "section_title": section_title,
                    "chunk_index": 0,
                    "total_chunks_in_section": 1,
                    "section_complete": True,
                    "token_count": token_count
                }

                final_chunks.append({
                    "text": section_content,
                    "source": source,
                    "metadata": chunk_metadata
                })

            else:
                # Split large section with simple paragraph splitting
                sub_chunks = self._split_large_section_simple(
                    section_content, section_title, section_id, base_metadata, source
                )
                final_chunks.extend(sub_chunks)

        # Add prev/next chunk linking across all chunks
        self._add_sequential_links(final_chunks)

        logger.info(f"Created {len(final_chunks)} chunks from {len(main_sections)} main sections")
        return final_chunks

    def _split_by_main_headers(self, text: str) -> List[Dict]:
        """Split text by main headers (# ), preserving all content."""
        lines = text.split('\n')
        sections = []
        current_content = []
        current_title = "Introduction"  # Default for content before first header

        for line in lines:
            # Check for main header (single # at start of line, not ## or ###)
            if line.strip().startswith('# ') and not line.strip().startswith('##'):
                # Save previous section if it has content
                if current_content:
                    sections.append({
                        'title': current_title,
                        'content': '\n'.join(current_content).strip()
                    })

                # Start new section
                current_title = line.strip()[2:].strip()  # Remove '# '
                current_content = [line]  # Include the header line itself

            else:
                # Add all lines to current section
                current_content.append(line)

        # Add the final section
        if current_content:
            sections.append({
                'title': current_title,
                'content': '\n'.join(current_content).strip()
            })

        # Filter out empty sections
        sections = [s for s in sections if s['content'].strip()]

        logger.info(f"Split document into {len(sections)} sections:")
        for i, section in enumerate(sections):
            token_count = self._count_tokens(section['content'])
            logger.info(f"  {i + 1}. '{section['title']}': {token_count} tokens")

        return sections

    def _split_large_section_simple(self, content: str, section_title: str, section_id: str,
                                    base_metadata: Dict, source: str) -> List[Dict]:
        """Split a large section into smaller chunks using simple paragraph splitting."""

        # Simple paragraph-based splitting
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        if not paragraphs:
            # Emergency: just split by sentences
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            paragraphs = sentences

        chunks = []
        current_chunk = []
        current_tokens = 0
        max_tokens = 2048

        for paragraph in paragraphs:
            para_tokens = self._count_tokens(paragraph)

            # If adding this paragraph would exceed limit, finalize current chunk
            if current_tokens + para_tokens > max_tokens and current_chunk:
                # Save current chunk
                chunk_content = '\n\n'.join(current_chunk)
                chunks.append(chunk_content)

                # Start new chunk
                current_chunk = [paragraph]
                current_tokens = para_tokens
            else:
                # Add to current chunk
                current_chunk.append(paragraph)
                current_tokens += para_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunks.append(chunk_content)

        # Convert to chunk objects with metadata
        chunk_objects = []
        total_parts = len(chunks)

        for i, chunk_content in enumerate(chunks):
            chunk_metadata = {
                **base_metadata,
                "section_id": section_id,
                "section_title": section_title,
                "chunk_index": i,
                "total_chunks_in_section": total_parts,
                "section_complete": False,  # Multi-part section
                "continuation_of": section_id if i > 0 else None,
                "token_count": self._count_tokens(chunk_content)
            }

            chunk_objects.append({
                "text": chunk_content,
                "source": source,
                "metadata": chunk_metadata
            })

        logger.info(f"Split large section '{section_title}' into {total_parts} chunks")
        return chunk_objects

    def _generate_section_id(self, section_title: str) -> str:
        """Generate clean section ID from title."""
        import re
        # Remove special characters, keep alphanumeric and spaces
        clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', section_title)
        # Replace spaces with underscores, limit length
        clean_title = re.sub(r'\s+', '_', clean_title.strip())
        return clean_title.lower()[:50]

    def _add_sequential_links(self, chunks: List[Dict]) -> None:
        """Add prev_chunk/next_chunk links to all chunks."""

        for i, chunk in enumerate(chunks):
            metadata = chunk['metadata']

            # Add global chunk linking (across all chunks in document)
            if i > 0:
                metadata['prev_chunk_global'] = f"chunk_{i - 1:03d}"
            if i < len(chunks) - 1:
                metadata['next_chunk_global'] = f"chunk_{i + 1:03d}"

            # Add section-specific linking (within same section)
            section_id = metadata.get('section_id')
            if section_id:
                # Find other chunks in same section
                section_chunks = [
                    (j, c) for j, c in enumerate(chunks)
                    if c['metadata'].get('section_id') == section_id
                ]

                # Find current chunk's position within its section
                section_position = None
                for pos, (chunk_idx, _) in enumerate(section_chunks):
                    if chunk_idx == i:
                        section_position = pos
                        break

                if section_position is not None:
                    # Add section-specific prev/next links
                    if section_position > 0:
                        prev_global_idx = section_chunks[section_position - 1][0]
                        metadata['prev_chunk_section'] = f"chunk_{prev_global_idx:03d}"

                    if section_position < len(section_chunks) - 1:
                        next_global_idx = section_chunks[section_position + 1][0]
                        metadata['next_chunk_section'] = f"chunk_{next_global_idx:03d}"

            # Add a unique chunk ID for reference
            metadata['chunk_id'] = f"chunk_{i:03d}"

        logger.info(f"Added sequential links to {len(chunks)} chunks")

    def _chunk_text_simple(self, text: str, source: str) -> List[Dict]:
        """Simple word/token-based chunking fallback with enhanced metadata."""
        base_metadata = self._extract_shadowrun_metadata(text, source)

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
        """Index all markdown files in a directory with enhanced metadata support."""
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

        logger.info(f"Indexing {len(files_to_index)} files with enhanced sequential metadata...")
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

            # Avoid ID collisions (existing code)
            seen_ids = set()
            for i, id_ in enumerate(ids):
                original = id_
                counter = 1
                while id_ in seen_ids:
                    id_ = f"{original}_{counter}"
                    counter += 1
                ids[i] = id_
                seen_ids.add(id_)

            cleaned_metadatas = self._clean_metadata_for_chromadb(metadatas)

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=cleaned_metadatas  # ✅ No None values
            )

            logger.info(f"Added {len(all_chunks)} chunks to index with enhanced sequential metadata")

            # Enhanced metadata distribution logging
            doc_types = {}
            editions = {}
            sections = {}
            chunk_links = 0
            section_distribution = {}

            for meta in metadatas:
                # Existing logging
                doc_types[meta.get('document_type', 'unknown')] = doc_types.get(meta.get('document_type', 'unknown'),
                                                                                0) + 1
                editions[meta.get('edition', 'unknown')] = editions.get(meta.get('edition', 'unknown'), 0) + 1
                sections[meta.get('main_section', 'unknown')] = sections.get(meta.get('main_section', 'unknown'), 0) + 1

                # NEW: Enhanced logging
                if meta.get('next_chunk_global') or meta.get('prev_chunk_global'):
                    chunk_links += 1

                section_id = meta.get('section_id')
                if section_id:
                    section_distribution[section_id] = section_distribution.get(section_id, 0) + 1

            logger.info(f"Document types: {doc_types}")
            logger.info(f"Editions: {editions}")
            logger.info(f"Sections: {sections}")
            logger.info(f"Chunks with sequential links: {chunk_links}/{len(metadatas)}")
            logger.info(f"Section distribution: {len(section_distribution)} unique sections")

            # Log example section for debugging
            if section_distribution:
                example_section = list(section_distribution.keys())[0]
                example_count = section_distribution[example_section]
                logger.info(f"Example: Section '{example_section}' has {example_count} chunks")

        self._save_index_metadata()

    def _clean_metadata_for_chromadb(self, metadatas: List[Dict]) -> List[Dict]:
        """Clean metadata to remove None values and ensure ChromaDB compatibility."""
        cleaned_metadatas = []

        for metadata in metadatas:
            cleaned = {}

            for key, value in metadata.items():
                # Skip None values entirely
                if value is None:
                    continue

                # Ensure all values are basic types (str, int, float, bool)
                if isinstance(value, (str, int, float, bool)):
                    cleaned[key] = value
                elif isinstance(value, list):
                    # Convert lists to comma-separated strings
                    cleaned[key] = ",".join(str(item) for item in value if item is not None)
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