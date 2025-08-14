"""Retrieval and answer generation with Ollama and metadata-aware filtering."""

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
    """Handle document retrieval and answer generation with metadata filtering."""

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
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "mixtral:8x7b-instruct-v0.1-q4_K_M")

        self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(self.collection_name)

        self.base_model_options = {
            "num_gpu": int(os.getenv("MODEL_NUM_GPU", 99)),
            "num_thread": int(os.getenv("MODEL_NUM_THREAD", 12)),
            "repeat_penalty": float(os.getenv("MODEL_REPEAT_PENALTY", 1.05)),
            "num_ctx": int(os.getenv("MODEL_NUM_CTX", 16384)),
            "num_batch": int(os.getenv("MODEL_NUM_BATCH", 512)),
        }

        logger.info(f"Initialized Retriever with model: {self.llm_model}")
        logger.info(f"Context window: {self.base_model_options['num_ctx']} tokens")

    def get_model_options(self, query_type: str) -> dict:
        options = self.base_model_options.copy()
        temp = 0.5
        if query_type == "rules":
            options.update({"temperature": 0.2, "top_k": 15, "top_p": 0.8, "mirostat": 2})
            temp = 0.2
        elif query_type == "character":
            options.update({"temperature": 0.3, "top_k": 25, "top_p": 0.85})
            temp = 0.3
        elif query_type == "session":
            options.update({"temperature": 0.7, "top_k": 50, "top_p": 0.9})
            temp = 0.7
        else:
            options.update({"temperature": 0.5, "top_k": 40, "top_p": 0.9})
            temp = 0.5

        logger.info(f"Using {query_type} settings: temp={temp}")
        return options

    def search(
        self,
        question: str,
        n_results: int = 5,
        where_filter: Optional[Dict] = None
    ) -> Dict:
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=question)
            query_embedding = response['embedding']

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
            )

            return {
                "documents": results['documents'][0] if results['documents'] else [],
                "metadatas": results['metadatas'][0] if results['metadatas'] else [],
                "distances": results['distances'][0] if results['distances'] else []
            }
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"documents": [], "metadatas": [], "distances": []}

    def generate_answer(
        self,
        prompt: str,  # ← Now receives fully rendered prompt
        query_type: str = "general",
        stream: bool = False,
        custom_options: dict = None
    ) -> str:
        """Generate answer using a pre-rendered prompt."""
        model_options = self.get_model_options(query_type)
        if custom_options:
            model_options.update(custom_options)

        try:
            if stream:
                return ollama.generate(
                    model=self.llm_model,
                    prompt=prompt,
                    stream=True,
                    options=model_options
                )
            else:
                response = ollama.generate(
                    model=self.llm_model,
                    prompt=prompt,
                    options=model_options
                )
                return response['response']
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {str(e)}"

    def query(
        self,
        question: str,
        n_results: int = 5,
        query_type: str = "general",
        where_filter: Optional[Dict] = None,
        character_role: Optional[str] = None,
        character_stats: Optional[str] = None,
        edition: Optional[str] = None
    ) -> Dict:
        """Complete RAG pipeline: search + generate."""
        search_results = self.search(question, n_results, where_filter)

        if not search_results['documents']:
            return {
                "answer": "No relevant information found in the indexed documents.",
                "sources": [],
                "chunks": [],
                "distances": [],
                "metadatas": []
            }

        # Build provenance into context
        context_parts = []
        for doc, meta in zip(search_results['documents'], search_results['metadatas']):
            source = "Unknown"
            if meta and isinstance(meta, dict):
                src = meta.get('source')
                if src:
                    try:
                        source = Path(src).name
                    except Exception:
                        source = str(src)
            section = meta.get('Section', 'General')
            subsection = meta.get('Subsection', 'General')
            context_parts.append(f"[Section: {section} → {subsection} | Source: {source}]\n{doc}")

        context = "\n\n---\n\n".join(context_parts)

        # Build final prompt
        if PROMPT_AVAILABLE:
            prompt_template = get_prompt(
                query_type=query_type,
                character_role=character_role,
                character_stats=character_stats,
                edition=edition
            )
        else:
            prompt_template = get_prompt()

        try:
            prompt = prompt_template.format(context=context, question=question)
        except Exception as e:
            logger.warning(f"Prompt formatting failed: {e}")
            prompt = f"{prompt_template}\n\nContext:\n{context}\n\nQuestion:\n{question}"

        # Generate answer
        answer = self.generate_answer(
            prompt=prompt,
            query_type=query_type,
            custom_options={"temperature": self.get_model_options(query_type)["temperature"]}
        )

        # Format sources
        sources = list({meta.get('source', 'Unknown') for meta in search_results['metadatas']})

        return {
            "answer": answer,
            "sources": sources,
            "chunks": search_results['documents'],
            "distances": search_results['distances'],
            "metadatas": search_results['metadatas']
        }