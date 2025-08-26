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

MODEL_SETTINGS = {
    "llama3:8b-instruct-q4_K_M": {
        "general": {
            "temperature": 0.6,
            "top_p": 0.9
        }
    },
    "qwen2.5:14b-instruct-q6_K": {
        "general": {
            "temperature": 0.7,
            "top_p": 0.8,
            "repetition_penalty": 1.05
        }
    },
    "mistral-nemo:12b": {
        "general": {
            "temperature": 0.3
        }
    },
    "deepseek-r1:14b": {
        "general": {
            "temperature": 0.6,
            "top_p": 0.95
        }
    },
    "deepseek-r1:8b": {
        "general": {
            "temperature": 0.6,
            "top_p": 0.95
        }
    }
}

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
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "phi4-reasoning:plus")

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

    def get_model_options(self, query_type: str, model: str = None) -> dict:
        """Enhanced model options with per-model, per-query-type settings."""
        # Start with base defaults (only what you specify)
        options = self.base_model_options.copy()

        model_to_check = model or self.llm_model

        if model_to_check in MODEL_SETTINGS:
            model_config = MODEL_SETTINGS[model_to_check]

            # Try specific query type first, fall back to "general"
            if query_type in model_config:
                query_settings = model_config[query_type]
            elif "general" in model_config:
                query_settings = model_config["general"]
                logger.debug(f"Query type '{query_type}' not found for {model_to_check}, using 'general'")
            else:
                query_settings = {}
                logger.debug(f"No settings found for {model_to_check}, using defaults only")

            options.update(query_settings)
            logger.debug(f"Applied {query_type} settings for {model_to_check}: {query_settings}")
        else:
            logger.debug(f"Unknown model {model_to_check}, using defaults only")

        logger.debug(f"Final {query_type} settings for {model_to_check}: {dict(options)}")
        return options

    def build_enhanced_filter(
            self,
            character_role: Optional[str] = None,
            where_filter: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Enhanced filter building with character role precedence, converted to ChromaDB format."""

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

        # Convert to ChromaDB format
        if final_filter:
            # ChromaDB uses implicit AND for multiple conditions
            chroma_filter = final_filter  # Just use the dict directly
            logger.info(f"Final enhanced filter (ChromaDB format): {chroma_filter}")
            return chroma_filter
        else:
            logger.info("No filters applied")
            return None

    def format_filter_for_chromadb(self, filter_dict: Dict) -> Dict:
        """Convert simple dict to ChromaDB 1.0.16 filter format."""
        if not filter_dict:
            return {}

        if len(filter_dict) == 1:
            # Single condition
            key, value = next(iter(filter_dict.items()))
            return {key: {'$eq': value}}
        else:
            # Multiple conditions - use $and
            conditions = []
            for key, value in filter_dict.items():
                conditions.append({key: {'$eq': value}})
            return {'$and': conditions}

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

            if enhanced_filter:
                enhanced_filter = self.format_filter_for_chromadb(enhanced_filter)

            # Build query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results
            }

            # Only add where filter if it has actual conditions
            if enhanced_filter and len(enhanced_filter) > 0:
                query_params["where"] = enhanced_filter

            logger.info(f"Enhanced filter before query: {enhanced_filter}")
            logger.info(f"Enhanced filter type: {type(enhanced_filter)}")

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
        model_options = self.get_model_options(query_type, model)  # Pass model parameter
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
        model_options = self.get_model_options(query_type, model)  # Pass model parameter
        if custom_options:
            model_options.update(custom_options)

        # Use passed model or fall back to default
        model_to_use = model or self.llm_model

        logger.info(f"Model {model_to_use} used with options: {model_options}")

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

    def search_with_linked_chunks(
            self,
            question: str,
            n_results: int = 5,
            where_filter: Optional[Dict] = None,
            character_role: Optional[str] = None,
            fetch_linked: bool = True
    ) -> Dict:
        """Enhanced search that automatically fetches linked chunks."""
        # Enhance query based on classification
        enhanced_question = self.enhance_query_with_classification(question)

        # First, do the normal search with enhanced query
        primary_results = self.search(enhanced_question, n_results, where_filter, character_role)

        if not fetch_linked or not primary_results['documents']:
            return primary_results

        # Collect linked chunk IDs from primary results
        linked_chunk_ids = set()
        primary_metadatas = primary_results['metadatas']

        for metadata in primary_metadatas:
            if not metadata:
                continue

            # Collect all linked chunk references
            for link_field in ['next_chunk_global', 'prev_chunk_global',
                               'next_chunk_section', 'prev_chunk_section']:
                link_id = metadata.get(link_field)
                if link_id:
                    linked_chunk_ids.add(link_id)

        if not linked_chunk_ids:
            logger.info("No linked chunks found in primary results")
            return primary_results

        # Fetch linked chunks
        linked_results = self._fetch_chunks_by_ids(linked_chunk_ids)

        if linked_results['documents']:
            # Merge results intelligently
            merged_results = self._merge_search_results(primary_results, linked_results)
            logger.info(
                f"Enhanced search: {len(primary_results['documents'])} primary + {len(linked_results['documents'])} linked = {len(merged_results['documents'])} total chunks")
            return merged_results

        return primary_results

    def _fetch_chunks_by_ids(self, chunk_ids: set) -> Dict:
        """Fetch specific chunks by their chunk_id metadata."""
        try:
            # Use ChromaDB where filter to find chunks by chunk_id
            # ✅ FIXED: Remove 'distances' from include - .get() doesn't support it
            results = self.collection.get(
                where={
                    "chunk_id": {"$in": list(chunk_ids)}
                },
                include=['documents', 'metadatas']  # Only documents and metadatas
            )

            # Convert to search result format
            if results and results.get('documents'):
                return {
                    "documents": results['documents'],
                    "metadatas": results['metadatas'],
                    "distances": [0.0] * len(results['documents'])  # Linked chunks get perfect relevance
                }
            else:
                logger.warning(f"Could not fetch linked chunks: {chunk_ids}")
                return {"documents": [], "metadatas": [], "distances": []}

        except Exception as e:
            logger.error(f"Error fetching linked chunks: {e}")
            return {"documents": [], "metadatas": [], "distances": []}

    def _merge_search_results(self, primary: Dict, linked: Dict) -> Dict:
        """Intelligently merge primary and linked search results."""

        # Start with primary results
        merged_docs = primary['documents'][:]
        merged_metas = primary['metadatas'][:]
        merged_distances = primary['distances'][:]

        # Track which chunks we already have (avoid duplicates)
        existing_chunk_ids = set()
        for meta in merged_metas:
            if meta and meta.get('chunk_id'):
                existing_chunk_ids.add(meta['chunk_id'])

        # Add linked chunks that aren't already included
        for i, linked_doc in enumerate(linked['documents']):
            linked_meta = linked['metadatas'][i] if i < len(linked['metadatas']) else {}
            linked_chunk_id = linked_meta.get('chunk_id')

            if linked_chunk_id and linked_chunk_id not in existing_chunk_ids:
                merged_docs.append(linked_doc)
                merged_metas.append(linked_meta)
                merged_distances.append(linked['distances'][i] if i < len(linked['distances']) else 0.5)
                existing_chunk_ids.add(linked_chunk_id)

        # Sort by a combination of relevance and sequence
        merged_items = list(zip(merged_docs, merged_metas, merged_distances))

        # Sort: primary results first (by distance), then linked chunks (by sequence)
        def sort_key(item):
            doc, meta, distance = item
            is_linked = distance == 0.0  # Linked chunks have 0.0 distance

            if is_linked:
                # For linked chunks, sort by section and chunk index
                section_id = meta.get('section_id', 'zzz')
                chunk_index = meta.get('chunk_index', 999)
                return (1, section_id, chunk_index)  # Secondary priority
            else:
                # For primary results, sort by relevance
                return (0, distance, 0)  # Primary priority

        sorted_items = sorted(merged_items, key=sort_key)

        # Unpack sorted results
        merged_docs, merged_metas, merged_distances = zip(*sorted_items) if sorted_items else ([], [], [])

        return {
            "documents": list(merged_docs),
            "metadatas": list(merged_metas),
            "distances": list(merged_distances)
        }

    def build_enhanced_context_with_sequence(self, search_results: Dict) -> str:
        """Build context with enhanced sequence and continuation information."""
        context_parts = []

        # Group chunks by section for better organization
        sections = {}
        standalone_chunks = []

        for doc, meta in zip(search_results['documents'], search_results['metadatas']):
            if not meta:
                standalone_chunks.append((doc, meta))
                continue

            section_id = meta.get('section_id')
            if section_id:
                if section_id not in sections:
                    sections[section_id] = []
                sections[section_id].append((doc, meta))
            else:
                standalone_chunks.append((doc, meta))

        # Add sectioned content first
        for section_id, section_chunks in sections.items():
            # Sort chunks within section by chunk_index
            section_chunks.sort(key=lambda x: x[1].get('chunk_index', 0))

            section_title = section_chunks[0][1].get('section_title', section_id)
            context_parts.append(f"## Section: {section_title}")

            for i, (doc, meta) in enumerate(section_chunks):
                chunk_info = self._build_chunk_header(meta, i, len(section_chunks))
                context_parts.append(f"{chunk_info}\n{doc}")

            context_parts.append("---")

        # Add standalone chunks
        for doc, meta in standalone_chunks:
            chunk_info = self._build_chunk_header(meta)
            context_parts.append(f"{chunk_info}\n{doc}")

        return "\n\n".join(context_parts)

    def _build_chunk_header(self, meta: Dict, chunk_pos: int = None, total_chunks: int = None) -> str:
        """Build informative header for each chunk."""
        if not meta:
            return "[Source: Unknown]"

        # Basic source info
        source = "Unknown"
        if meta.get('source'):
            try:
                source = Path(meta['source']).name
            except Exception:
                source = str(meta['source'])

        # Enhanced metadata
        doc_type = meta.get('document_type', 'unknown')
        edition = meta.get('edition', 'unknown')

        # Section info
        section_title = meta.get('section_title', 'General')

        # Sequence info
        sequence_info = ""
        if meta.get('total_chunks_in_section', 1) > 1:
            chunk_idx = meta.get('chunk_index', 0)
            total_in_section = meta.get('total_chunks_in_section', 1)
            sequence_info = f" | Part {chunk_idx + 1}/{total_in_section}"

            if not meta.get('section_complete', True):
                sequence_info += " (continued)"

        return f"[{doc_type.title()} | {edition} | {section_title}{sequence_info} | Source: {source}]"

    def query(
        self,
        question: str,
        n_results: int = 5,
        query_type: str = "general",
        where_filter: Optional[Dict] = None,
        character_role: Optional[str] = None,
        character_stats: Optional[str] = None,
        edition: Optional[str] = "SR5",  # Default to SR5
        model: Optional[str] = None,
        fetch_linked_chunks: bool = True
    ) -> Dict:
        """Enhanced RAG pipeline with improved filtering and context building."""

        # Enhanced search with character role precedence
        search_results = self.search_with_linked_chunks(
            question,
            n_results,
            where_filter,
            character_role,
            fetch_linked_chunks
        )

        if not search_results['documents']:
            return {
                "answer": "No relevant information found in the indexed documents. Try adjusting your filters or search terms.",
                "sources": [],
                "chunks": [],
                "distances": [],
                "metadatas": [],
                "linked_chunks_fetched": 0
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
            model=model
        )

        # Enhanced source formatting
        sources = []
        seen_sources = set()
        linked_count = 0

        for meta in search_results['metadatas']:
            if meta and meta.get('source'):
                source_path = meta['source']
                if source_path not in seen_sources:
                    sources.append(source_path)
                    seen_sources.add(source_path)

                # Count linked chunks
                if meta and meta.get('next_chunk_global'):
                    linked_count += 1

        return {
            "answer": answer,
            "sources": sources,
            "chunks": search_results['documents'],
            "distances": search_results['distances'],
            "metadatas": search_results['metadatas'],
            "linked_chunks_fetched": linked_count
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
            model: Optional[str] = None,
            fetch_linked_chunks: bool = True
    ):
        """Enhanced streaming query with complete filter logic and linked chunk fetching."""

        # Build enhanced filter with character role precedence (same as query() method)
        enhanced_filter = self.build_enhanced_filter(character_role, where_filter)

        # Enhanced search with character role precedence and linked chunks
        search_results = self.search_with_linked_chunks(
            question,
            n_results,
            enhanced_filter,  # Use the enhanced filter instead of where_filter
            character_role,
            fetch_linked_chunks
        )

        if not search_results['documents']:
            yield "No relevant information found in the indexed documents. Try adjusting your filters or search terms."
            return

        # Build enhanced context with sequence information
        context = self.build_enhanced_context_with_sequence(search_results)

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

    def classify_shadowrun_query(self, query_text: str) -> str:
        """Classify Shadowrun queries based on actual game mechanics and terminology."""
        query_lower = query_text.lower()

        rule_classifications = {
            "damage_resistance": [
                "damage", "resistance", "resist", "soak", "armor", "condition monitor",
                "physical damage", "stun damage", "overflow", "wound modifier", "dice pool"
            ],
            "matrix": [
                "matrix", "hack", "cyberdeck", "decker", "firewall", "attack", "sleaze",
                "data processing", "ic", "host", "biofeedback", "dumpshock", "overwatch",
                "device rating", "matrix damage", "silent running", "marks", "black ic"
            ],
            "magic": [
                "spell", "magic", "drain", "force", "mana", "astral", "summoning",
                "spirit", "adept", "mage", "enchanting", "focus", "reagents"
            ],
            "combat": [
                "combat", "initiative", "action phase", "attack", "defense", "cover",
                "melee", "ranged", "full auto", "burst fire", "called shot", "reach"
            ],
            "dice_pools": [
                "dice pool", "dice", "roll", "attribute", "skill", "threshold",
                "modifier", "bonus", "penalty", "glitch", "test"
            ]
        }

        # Score each category
        category_scores = {}
        for category, keywords in rule_classifications.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                category_scores[category] = score

        query_category = max(category_scores, key=category_scores.get) if category_scores else "general"
        logger.info(f"Query classified as: {query_category}")

        return query_category

    def enhance_query_with_classification(self, original_query: str) -> str:
        """Enhance query with relevant terms based on classification to improve ChromaDB retrieval."""

        # Get the classification
        category = self.classify_shadowrun_query(original_query)

        # Enhancement terms for each category
        enhancement_terms = {
            "damage_resistance": "Device Rating Firewall damage resistance soak armor condition monitor",
            "matrix": "matrix cyberdeck firewall biofeedback decker attack sleaze data processing",
            "magic": "spell drain force mana magic astral summoning spirit",
            "combat": "initiative combat attack defense armor penetration melee ranged",
            "dice_pools": "attribute skill dice pool threshold modifier test roll",
            "rigging": "rigger drone vehicle pilot jumped control rig autosofts"
        }

        if category in enhancement_terms:
            enhanced_query = f"{original_query} {enhancement_terms[category]}"
            logger.info(f"Query enhanced: '{original_query}' -> category: {category}")
            return enhanced_query

        logger.info(f"Query classified as: {category} (no enhancement)")
        return original_query

    def boost_chunks_by_category(self, search_results: Dict, category: str) -> Dict:
        """Boost chunk rankings based on detected query category."""
        if category == "general":
            return search_results

        boost_terms = {
            "damage_resistance": ["Device Rating + Firewall", "Willpower + Firewall", "Body + Armor"],
            "matrix": ["Device Rating + Firewall", "Willpower + Firewall", "biofeedback", "matrix damage"],
            "dice_pools": ["Attribute + Skill", "dice pool", "threshold", "net hits"],
            "combat": ["Initiative", "Defense", "armor penetration", "damage code"],
            "magic": ["Drain Value", "Force", "Magic + Skill", "astral plane"]
        }

        if category not in boost_terms:
            return search_results

        # Boost chunks containing relevant terms
        terms_to_boost = boost_terms[category]
        for i, document in enumerate(search_results['documents']):
            boost_score = sum(1 for term in terms_to_boost if term.lower() in document.lower())
            if boost_score > 0:
                # Improve the distance (lower = better ranking)
                search_results['distances'][i] *= (1 - (boost_score * 0.1))  # Up to 50% boost

        return search_results