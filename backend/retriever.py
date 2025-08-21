"""Enhanced retriever.py with improved filtering logic and character role precedence."""

import ollama
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import logging
import os
from pathlib import Path

# Import prompts at top with fallback
try:
    from .prompts import get_prompt
    PROMPT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not load prompts module: {e}")
    PROMPT_AVAILABLE = False

    def get_prompt(query_type="general", **kwargs) -> str:
        return (
            "You are a helpful assistant for Shadowrun.\n"
            "Answer based on the context below.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    """Enhanced retriever with improved metadata filtering and character role precedence."""

    def __init__(
        self,
        chroma_path: str = None,
        collection_name: str = None,
        embedding_model: str = None,
        llm_model: str = None
    ):
        self.chroma_path = chroma_path or os.getenv("CHROMA_DB_PATH", "data/chroma_db")
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "shadowrun_docs")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "llama3:8b-instruct-q4_K_M")

        self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(self.collection_name)

        self.base_model_options = {
            "num_gpu": int(os.getenv("MODEL_NUM_GPU", 99)),
            "num_thread": int(os.getenv("MODEL_NUM_THREAD", 12)),
            "repeat_penalty": float(os.getenv("MODEL_REPEAT_PENALTY", 1.05)),
            "num_ctx": int(os.getenv("MODEL_NUM_CTX", 4096)),
            "num_batch": int(os.getenv("MODEL_NUM_BATCH", 128)),
        }

        # Enhanced character role to section mapping
        self.role_to_section_map = {
            "decker": "Matrix",
            "hacker": "Matrix",
            "mage": "Magic",
            "adept": "Magic",
            "street_samurai": "Combat",
            "rigger": "Riggers",
            "technomancer": "Matrix",  # Technomancers use Matrix rules
            "face": "Social"  # Added Face role
        }

        logger.info(f"Enhanced Retriever initialized with model: {self.llm_model}")
        logger.info(f"Context window: {self.base_model_options['num_ctx']} tokens")
        logger.info(f"Character role mappings: {self.role_to_section_map}")

    def get_model_options(self, query_type: str) -> dict:
        """Enhanced model options with better parameter tuning."""
        options = self.base_model_options.copy()

        # Enhanced settings for different query types
        if query_type == "rules":
            options.update({"temperature": 0.1, "top_k": 20, "top_p": 0.8})  # Very focused for rules
        elif query_type == "character":
            options.update({"temperature": 0.2, "top_k": 25, "top_p": 0.85})  # Focused for character info
        elif query_type == "session":
            options.update({"temperature": 0.6, "top_k": 40, "top_p": 0.9})  # More creative for sessions
        else:
            options.update({"temperature": 0.3, "top_k": 30, "top_p": 0.85})  # Balanced default

        logger.debug(f"Using {query_type} settings: temp={options['temperature']}")
        return options

    def build_enhanced_filter(
        self,
        character_role: Optional[str] = None,
        where_filter: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Enhanced filter building with character role precedence."""

        # Start with provided filter or empty dict
        final_filter = where_filter.copy() if where_filter else {}

        # Character role takes absolute precedence over section filters
        if character_role and character_role.lower() in self.role_to_section_map:
            role_section = self.role_to_section_map[character_role.lower()]

            # Remove any existing section filters and replace with role-based section
            final_filter.pop("Section", None)
            final_filter.pop("main_section", None)
            final_filter["main_section"] = role_section

            logger.info(f"Character role '{character_role}' overrode section filter → '{role_section}'")

        # Log final filter for debugging
        if final_filter:
            logger.info(f"Final enhanced filter: {final_filter}")
            return final_filter
        else:
            logger.info("No filters applied")
            return None

    def search(
            self,
            question: str,
            n_results: int = 5,
            where_filter: Optional[Dict] = None,
            character_role: Optional[str] = None
    ) -> Dict:
        """Enhanced search with improved filter handling."""
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=question)
            query_embedding = response['embedding']

            # Build enhanced filter with character role precedence
            enhanced_filter = self.build_enhanced_filter(character_role, where_filter)

            # Build query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results
            }

            # Only add where filter if it has actual conditions
            if enhanced_filter and len(enhanced_filter) > 0:
                query_params["where"] = enhanced_filter

            results = self.collection.query(**query_params)

            # Log search results for debugging
            doc_count = len(results['documents'][0]) if results['documents'] else 0
            logger.info(f"Search returned {doc_count} documents")

            if doc_count == 0 and enhanced_filter:
                logger.warning(f"No results found with filter: {enhanced_filter}")
                # Try a fallback search without filters for debugging
                fallback_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                fallback_count = len(fallback_results['documents'][0]) if fallback_results['documents'] else 0
                logger.info(f"Fallback search without filters returned {fallback_count} documents")

            return {
                "documents": results['documents'][0] if results['documents'] else [],
                "metadatas": results['metadatas'][0] if results['metadatas'] else [],
                "distances": results['distances'][0] if results['distances'] else []
            }
        except Exception as e:
            logger.error(f"Enhanced search error: {e}")
            return {"documents": [], "metadatas": [], "distances": []}

    def generate_answer(
            self,
            prompt: str,
            query_type: str = "general",
            custom_options: dict = None,
            model: str = None
    ) -> str:
        """Generate complete answer as string (non-streaming)."""
        model_options = self.get_model_options(query_type)
        if custom_options:
            model_options.update(custom_options)

        # Use passed model or fall back to default
        model_to_use = model or self.llm_model

        try:
            response = ollama.generate(
                model=model_to_use,
                prompt=prompt,
                stream=False,
                options=model_options
            )
            return response['response']

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {str(e)}"

    def generate_answer_stream(
            self,
            prompt: str,
            query_type: str = "general",
            custom_options: dict = None,
            model: str = None
    ):
        """Generate answer as streaming generator with enhanced error handling."""
        model_options = self.get_model_options(query_type)
        if custom_options:
            model_options.update(custom_options)

        # Use passed model or fall back to default
        model_to_use = model or self.llm_model

        try:
            response = ollama.generate(
                model=model_to_use,
                prompt=prompt,
                stream=True,
                options=model_options
            )
            for chunk in response:
                yield chunk['response']

        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield f"Error generating response: {str(e)}"

    def build_enhanced_context(self, search_results: Dict) -> str:
        """Build context with enhanced metadata display."""
        context_parts = []

        for doc, meta in zip(search_results['documents'], search_results['metadatas']):
            if not meta:
                context_parts.append(f"[Source: Unknown]\n{doc}")
                continue

            # Enhanced metadata extraction
            source = "Unknown"
            if meta.get('source'):
                try:
                    source = Path(meta['source']).name
                except Exception:
                    source = str(meta['source'])

            # Get enhanced metadata fields
            doc_type = meta.get('document_type', 'unknown')
            edition = meta.get('edition', 'unknown')
            section = meta.get('main_section', meta.get('Section', 'General'))
            subsection = meta.get('subsection', meta.get('Subsection', 'General'))

            # Build enhanced context header
            context_header = f"[{doc_type.title()} | {edition} | {section} → {subsection} | Source: {source}]"
            context_parts.append(f"{context_header}\n{doc}")

        return "\n\n---\n\n".join(context_parts)

    def query(
        self,
        question: str,
        n_results: int = 5,
        query_type: str = "general",
        where_filter: Optional[Dict] = None,
        character_role: Optional[str] = None,
        character_stats: Optional[str] = None,
        edition: Optional[str] = "SR5",  # Default to SR5
        model: Optional[str] = None
    ) -> Dict:
        """Enhanced RAG pipeline with improved filtering and context building."""

        # Enhanced search with character role precedence
        search_results = self.search(
            question,
            n_results,
            where_filter,
            character_role
        )

        if not search_results['documents']:
            return {
                "answer": "No relevant information found in the indexed documents. Try adjusting your filters or search terms.",
                "sources": [],
                "chunks": [],
                "distances": [],
                "metadatas": []
            }

        # Build enhanced context with metadata
        context = self.build_enhanced_context(search_results)

        # Build enhanced prompt with character role info
        if PROMPT_AVAILABLE:
            prompt_template = get_prompt(
                query_type=query_type,
                character_role=character_role,
                character_stats=character_stats,
                edition=edition or "SR5"
            )
        else:
            prompt_template = get_prompt()

        try:
            prompt = prompt_template.format(context=context, question=question)
        except Exception as e:
            logger.warning(f"Prompt formatting failed: {e}")
            prompt = f"{prompt_template}\n\nContext:\n{context}\n\nQuestion:\n{question}"

        # Generate answer with enhanced options
        answer = self.generate_answer(
            prompt=prompt,
            query_type=query_type,
            custom_options={"temperature": self.get_model_options(query_type)["temperature"]},
            model=model
        )

        # Enhanced source formatting
        sources = []
        seen_sources = set()
        for meta in search_results['metadatas']:
            if meta and meta.get('source'):
                source_path = meta['source']
                if source_path not in seen_sources:
                    sources.append(source_path)
                    seen_sources.add(source_path)

        return {
            "answer": answer,
            "sources": sources,
            "chunks": search_results['documents'],
            "distances": search_results['distances'],
            "metadatas": search_results['metadatas']
        }

    def query_stream(
        self,
        question: str,
        n_results: int = 5,
        query_type: str = "general",
        where_filter: Optional[Dict] = None,
        character_role: Optional[str] = None,
        character_stats: Optional[str] = None,
        edition: Optional[str] = "SR5",  # Default to SR5
        model: Optional[str] = None
    ):
        """Enhanced streaming query with character role precedence."""

        # Enhanced search with character role precedence
        search_results = self.search(
            question,
            n_results,
            where_filter,
            character_role
        )

        if not search_results['documents']:
            yield "No relevant information found in the indexed documents. Try adjusting your filters or search terms."
            return

        # Build enhanced context
        context = self.build_enhanced_context(search_results)

        # Build enhanced prompt
        if PROMPT_AVAILABLE:
            prompt_template = get_prompt(
                query_type=query_type,
                character_role=character_role,
                character_stats=character_stats,
                edition=edition or "SR5"
            )
        else:
            prompt_template = get_prompt()

        try:
            prompt = prompt_template.format(context=context, question=question)
        except Exception as e:
            logger.warning(f"Prompt formatting failed: {e}")
            prompt = f"{prompt_template}\n\nContext:\n{context}\n\nQuestion:\n{question}"

        # Stream enhanced answer
        for token in self.generate_answer_stream(
            prompt=prompt,
            query_type=query_type,
            custom_options={"temperature": self.get_model_options(query_type)["temperature"]},
            model=model
        ):
            yield token

    def debug_collection_metadata(self) -> Dict:
        """Debug function to analyze collection metadata structure."""
        try:
            # Get a sample of documents to analyze metadata structure
            sample = self.collection.get(limit=10, include=['metadatas'])

            metadata_analysis = {
                "total_sampled": len(sample.get('metadatas', [])),
                "metadata_fields": set(),
                "field_value_examples": {},
                "missing_fields": []
            }

            for meta in sample.get('metadatas', []):
                if meta:
                    for key, value in meta.items():
                        metadata_analysis["metadata_fields"].add(key)
                        if key not in metadata_analysis["field_value_examples"]:
                            metadata_analysis["field_value_examples"][key] = []
                        if value not in metadata_analysis["field_value_examples"][key]:
                            metadata_analysis["field_value_examples"][key].append(value)

            # Convert set to list for JSON serialization
            metadata_analysis["metadata_fields"] = list(metadata_analysis["metadata_fields"])

            # Check for expected fields
            expected_fields = ["main_section", "document_type", "edition", "source"]
            for field in expected_fields:
                if field not in metadata_analysis["metadata_fields"]:
                    metadata_analysis["missing_fields"].append(field)

            logger.info(f"Metadata analysis: {metadata_analysis}")
            return metadata_analysis

        except Exception as e:
            logger.error(f"Metadata analysis failed: {e}")
            return {"error": str(e)}