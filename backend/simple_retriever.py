"""Simplified retriever for Shadowrun RAG - Core functionality only."""

import ollama
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRetriever:
    """Simplified retriever with basic section filtering only."""

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
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "qwen2.5:14b-instruct-q6_K")

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(self.collection_name)

        # Basic model options
        self.model_options = {
            "num_gpu": int(os.getenv("MODEL_NUM_GPU", 99)),
            "num_thread": int(os.getenv("MODEL_NUM_THREAD", 12)),
            "temperature": 0.7,
            "top_p": 0.8,
            "repeat_penalty": 1.05,
            "num_ctx": int(os.getenv("MODEL_NUM_CTX", 8192)),
        }

        logger.info(f"SimpleRetriever initialized")
        logger.info(f"Embedding model: {self.embedding_model}")
        logger.info(f"LLM model: {self.llm_model}")

    def search(
        self,
        question: str,
        n_results: int = 5,
        where_filter: Optional[Dict] = None
    ) -> Dict:
        """Basic search with simple filtering."""
        try:
            # Get query embedding
            response = ollama.embeddings(model=self.embedding_model, prompt=question)
            query_embedding = response['embedding']

            # Prepare ChromaDB filter
            chroma_filter = None
            if where_filter:
                # Simple filter conversion for ChromaDB
                chroma_filter = {}
                for key, value in where_filter.items():
                    chroma_filter[key] = {"$eq": value}
                logger.info(f"Applied filter: {chroma_filter}")

            # Search in ChromaDB
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=chroma_filter,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            return {
                'documents': search_results['documents'][0] if search_results['documents'] else [],
                'metadatas': search_results['metadatas'][0] if search_results['metadatas'] else [],
                'distances': search_results['distances'][0] if search_results['distances'] else []
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            return {'documents': [], 'metadatas': [], 'distances': []}

    def query(
        self,
        question: str,
        n_results: int = 5,
        where_filter: Optional[Dict] = None,
        model: str = None
    ) -> Dict:
        """Generate answer using search results."""
        try:
            # Get search results
            search_results = self.search(question, n_results, where_filter)
            
            if not search_results['documents']:
                return {
                    'answer': "No relevant information found in the indexed documents.",
                    'sources': [],
                    'chunks': [],
                    'distances': [],
                    'metadatas': []
                }

            # Prepare context
            context_parts = []
            sources = set()
            
            for i, (doc, metadata) in enumerate(zip(search_results['documents'], search_results['metadatas'])):
                # Add source info
                source = metadata.get('source', 'Unknown')
                sources.add(source)
                
                # Format context chunk
                section = metadata.get('primary_section', '')
                if section:
                    context_parts.append(f"[{section}] {doc}")
                else:
                    context_parts.append(doc)

            context = "\n\n".join(context_parts)

            # Simple prompt
            prompt = f"""You are a helpful assistant for Shadowrun 5th Edition.
Answer the question based on the provided context from the rulebooks.

Context:
{context}

Question: {question}

Answer:"""

            # Get model to use
            model_to_use = model or self.llm_model
            
            # Generate response
            response = ollama.generate(
                model=model_to_use,
                prompt=prompt,
                options=self.model_options
            )

            return {
                'answer': response['response'],
                'sources': list(sources),
                'chunks': search_results['documents'],
                'distances': search_results['distances'],
                'metadatas': search_results['metadatas']
            }

        except Exception as e:
            logger.error(f"Query error: {e}")
            return {
                'answer': f"Error generating response: {str(e)}",
                'sources': [],
                'chunks': [],
                'distances': [],
                'metadatas': []
            }

    def query_stream(
        self,
        question: str,
        n_results: int = 5,
        where_filter: Optional[Dict] = None,
        model: str = None
    ):
        """Generate streaming answer using search results."""
        try:
            # Get search results
            search_results = self.search(question, n_results, where_filter)
            
            if not search_results['documents']:
                yield "No relevant information found in the indexed documents."
                return

            # Prepare context
            context_parts = []
            for i, (doc, metadata) in enumerate(zip(search_results['documents'], search_results['metadatas'])):
                section = metadata.get('primary_section', '')
                if section:
                    context_parts.append(f"[{section}] {doc}")
                else:
                    context_parts.append(doc)

            context = "\n\n".join(context_parts)

            # Simple prompt
            prompt = f"""You are a helpful assistant for Shadowrun 5th Edition.
Answer the question based on the provided context from the rulebooks.

Context:
{context}

Question: {question}

Answer:"""

            # Get model to use
            model_to_use = model or self.llm_model
            
            # Stream response
            response = ollama.generate(
                model=model_to_use,
                prompt=prompt,
                options=self.model_options,
                stream=True
            )

            for chunk in response:
                if 'response' in chunk:
                    yield chunk['response']

        except Exception as e:
            logger.error(f"Stream query error: {e}")
            yield f"Error generating response: {str(e)}"

    def get_available_sections(self) -> List[str]:
        """Get list of available sections from indexed data."""
        try:
            results = self.collection.get()
            sections = set()
            
            for metadata in results.get('metadatas', []):
                if metadata and 'primary_section' in metadata:
                    sections.add(metadata['primary_section'])
            
            return sorted(list(sections))
        except Exception as e:
            logger.error(f"Error getting sections: {e}")
            return ["Matrix", "Combat", "Magic", "Riggers", "Social"]  # Default sections
