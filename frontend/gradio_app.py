"""
Enhanced Gradio Frontend for Shadowrun RAG System
Includes document type selection, <think> tag support, and improved filtering
"""

import gradio as gr
import requests
import json

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

class RAGClient:
    """Enhanced client for interacting with the FastAPI backend."""

    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.active_jobs = {}
        self.lock = threading.Lock()

    def upload_pdf(self, file_path: str, document_type: str = "rulebook") -> Dict:
        """Upload a PDF with document type specification."""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f, 'application/pdf')}
                data = {'document_type': document_type}
                response = requests.post(
                    f"{self.api_url}/upload",
                    files=files,
                    data=data,
                    timeout=60
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return {"error": str(e)}

    def get_job_status(self, job_id: str) -> Dict:
        """Poll job status."""
        try:
            response = requests.get(f"{self.api_url}/job/{job_id}", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"error": str(e)}

    def query_stream(self, question: str, **params) -> tuple:
        """Enhanced stream query response with <think> tag detection."""
        try:
            response = requests.post(
                f"{self.api_url}/query_stream",
                json={"question": question, **params},
                stream=True,
                timeout=30
            )
            response.raise_for_status()

            full_response = ""
            thinking_content = ""
            metadata = None
            metadata_buffer = ""
            collecting_metadata = False
            in_thinking = False

            for chunk in response.iter_content(chunk_size=32, decode_unicode=True):
                if chunk:
                    # Handle thinking tags
                    if "__THINKING_START__" in chunk:
                        in_thinking = True
                        parts = chunk.split("__THINKING_START__")
                        if parts[0]:
                            full_response += parts[0]
                            yield full_response, thinking_content, None, "generating"
                        if len(parts) > 1:
                            thinking_content += parts[1]
                            yield full_response, thinking_content, None, "thinking"
                        continue

                    if "__THINKING_END__" in chunk and in_thinking:
                        in_thinking = False
                        parts = chunk.split("__THINKING_END__")
                        if parts[0]:
                            thinking_content += parts[0]
                            yield full_response, thinking_content, None, "thinking"
                        if len(parts) > 1:
                            full_response += parts[1]
                            yield full_response, thinking_content, None, "generating"
                        continue

                    # Handle metadata
                    if "__METADATA_START__" in chunk:
                        parts = chunk.split("__METADATA_START__")
                        if parts[0]:
                            if in_thinking:
                                thinking_content += parts[0]
                            else:
                                full_response += parts[0]
                            yield full_response, thinking_content, None, "generating"

                        collecting_metadata = True
                        if len(parts) > 1:
                            metadata_buffer = parts[1]
                        continue

                    if collecting_metadata:
                        metadata_buffer += chunk
                        if "__METADATA_END__" in metadata_buffer:
                            json_part = metadata_buffer.split("__METADATA_END__")[0].strip()
                            try:
                                metadata = json.loads(json_part)
                            except json.JSONDecodeError as e:
                                logger.error(f"Metadata parse failed: {e}")
                            break
                    else:
                        # Regular content
                        if in_thinking:
                            thinking_content += chunk
                            yield full_response, thinking_content, None, "thinking"
                        else:
                            full_response += chunk
                            yield full_response, thinking_content, None, "generating"

            # Final yield with metadata
            yield full_response, thinking_content, metadata, "complete"

        except Exception as e:
            logger.error(f"Query failed: {e}")
            yield f"Error: {str(e)}", "", None, "error"

    def get_models(self) -> List[str]:
        """Get available models."""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=5)
            response.raise_for_status()
            return response.json().get("models", ["llama3"])
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return ["llama3"]

    def get_documents(self) -> List[str]:
        """Get indexed documents."""
        try:
            response = requests.get(f"{self.api_url}/documents", timeout=10)
            response.raise_for_status()
            return response.json().get("documents", [])
        except Exception as e:
            logger.error(f"Failed to fetch documents: {e}")
            return []

    def get_document_stats(self) -> Dict:
        """Get enhanced document statistics."""
        try:
            response = requests.get(f"{self.api_url}/document_stats", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch document stats: {e}")
            return {"error": str(e)}

    def get_document_content(self, file_path: str) -> Dict:
        """Get content of a specific document."""
        try:
            response = requests.get(f"{self.api_url}/document/{file_path}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch document content: {e}")
            return {"error": str(e)}

    def search_documents(self, query: str, group: str = None, document_type: str = None,
                        edition: str = None, page: int = 1, page_size: int = 20) -> Dict:
        """Enhanced document search with metadata filtering."""
        try:
            search_params = {
                "query": query,
                "page": page,
                "page_size": page_size
            }
            if group and group != "All Groups":
                search_params["group"] = group
            if document_type and document_type != "All Types":
                search_params["document_type"] = document_type
            if edition and edition != "All Editions":
                search_params["edition"] = edition

            response = requests.post(
                f"{self.api_url}/search_documents",
                json=search_params,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": str(e)}

    def get_status(self) -> Dict:
        """Get system status."""
        try:
            response = requests.get(f"{self.api_url}/status", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch status: {e}")
            return {"status": "offline", "error": str(e)}

    def reindex(self, force: bool = False) -> Dict:
        """Trigger reindexing."""
        try:
            response = requests.post(
                f"{self.api_url}/index",
                json={"force_reindex": force},
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Reindex failed: {e}")
            return {"error": str(e)}


class CharacterAPIClient:
    """API client for character management operations."""

    def __init__(self, api_url: str):
        self.api_url = api_url

    # Character management
    def list_characters(self) -> List[Dict]:
        try:
            response = requests.get(f"{self.api_url}/characters", timeout=5)
            response.raise_for_status()
            return response.json().get("characters", [])
        except Exception as e:
            logger.error(f"Failed to list characters: {e}")
            return []

    def create_character(self, name: str, metatype: str = "Human", archetype: str = "") -> Dict:
        try:
            response = requests.post(
                f"{self.api_url}/characters",
                json={"name": name, "metatype": metatype, "archetype": archetype},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to create character: {e}")
            return {"error": str(e)}

    def get_character(self, character_id: int) -> Dict:
        try:
            response = requests.get(f"{self.api_url}/characters/{character_id}", timeout=10)
            response.raise_for_status()
            return response.json().get("character", {})
        except Exception as e:
            logger.error(f"Failed to get character: {e}")
            return {"error": str(e)}

    def delete_character(self, character_id: int) -> Dict:
        try:
            response = requests.delete(f"{self.api_url}/characters/{character_id}", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to delete character: {e}")
            return {"error": str(e)}

    def get_active_character(self) -> Optional[Dict]:
        try:
            response = requests.get(f"{self.api_url}/characters/active", timeout=5)
            response.raise_for_status()
            return response.json().get("active_character")
        except Exception as e:
            logger.error(f"Failed to get active character: {e}")
            return None

    def set_active_character(self, character_id: int) -> Dict:
        try:
            response = requests.post(f"{self.api_url}/characters/{character_id}/activate", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to set active character: {e}")
            return {"error": str(e)}

    # Character data updates
    def update_character_stats(self, character_id: int, stats: Dict) -> Dict:
        try:
            response = requests.put(
                f"{self.api_url}/characters/{character_id}/stats",
                json=stats,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to update character stats: {e}")
            return {"error": str(e)}

    def update_character_resources(self, character_id: int, resources: Dict) -> Dict:
        try:
            response = requests.put(
                f"{self.api_url}/characters/{character_id}/resources",
                json=resources,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to update character resources: {e}")
            return {"error": str(e)}

    # Skills management
    def add_character_skill(self, character_id: int, skill_data: Dict) -> Dict:
        try:
            response = requests.post(
                f"{self.api_url}/characters/{character_id}/skills",
                json=skill_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to add skill: {e}")
            return {"error": str(e)}

    def remove_character_skill(self, character_id: int, skill_name: str) -> Dict:
        try:
            response = requests.delete(
                f"{self.api_url}/characters/{character_id}/skills/{skill_name}",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to remove skill: {e}")
            return {"error": str(e)}

    # Qualities management
    def add_character_quality(self, character_id: int, quality_data: Dict) -> Dict:
        try:
            response = requests.post(
                f"{self.api_url}/characters/{character_id}/qualities",
                json=quality_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to add quality: {e}")
            return {"error": str(e)}

    def remove_character_quality(self, character_id: int, quality_name: str) -> Dict:
        try:
            response = requests.delete(
                f"{self.api_url}/characters/{character_id}/qualities/{quality_name}",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to remove quality: {e}")
            return {"error": str(e)}

    # Gear management
    def add_character_gear(self, character_id: int, gear_data: Dict) -> Dict:
        try:
            response = requests.post(
                f"{self.api_url}/characters/{character_id}/gear",
                json=gear_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to add gear: {e}")
            return {"error": str(e)}

    def remove_character_gear(self, character_id: int, gear_id: int) -> Dict:
        try:
            response = requests.delete(
                f"{self.api_url}/characters/{character_id}/gear/{gear_id}",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to remove gear: {e}")
            return {"error": str(e)}

    # Reference data
    def get_skills_library(self, skill_type: str = None) -> List[Dict]:
        try:
            url = f"{self.api_url}/reference/skills"
            if skill_type:
                url += f"?skill_type={skill_type}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json().get("skills", [])
        except Exception as e:
            logger.error(f"Failed to get skills library: {e}")
            return []

    def get_qualities_library(self, quality_type: str = None) -> List[Dict]:
        try:
            url = f"{self.api_url}/reference/qualities"
            if quality_type:
                url += f"?quality_type={quality_type}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json().get("qualities", [])
        except Exception as e:
            logger.error(f"Failed to get qualities library: {e}")
            return []

    def get_gear_library(self, category: str = None) -> List[Dict]:
        try:
            url = f"{self.api_url}/reference/gear"
            if category:
                url += f"?category={category}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json().get("gear", [])
        except Exception as e:
            logger.error(f"Failed to get gear library: {e}")
            return []

    def get_gear_categories(self) -> List[str]:
        try:
            response = requests.get(f"{self.api_url}/reference/gear/categories", timeout=5)
            response.raise_for_status()
            return response.json().get("categories", [])
        except Exception as e:
            logger.error(f"Failed to get gear categories: {e}")
            return []

    def populate_reference_data(self) -> Dict:
        try:
            response = requests.post(f"{self.api_url}/reference/populate", timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to populate reference data: {e}")
            return {"error": str(e)}

    # Export
    def export_character_json(self, character_id: int) -> bytes:
        try:
            response = requests.get(f"{self.api_url}/characters/{character_id}/export/json", timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to export character JSON: {e}")
            return b""

    def get_dice_pool(self, character_id: int, skill_name: str) -> Dict:
        try:
            response = requests.get(f"{self.api_url}/characters/{character_id}/dice_pool/{skill_name}", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get dice pool: {e}")
            return {"error": str(e)}

    def add_character_vehicle(self, character_id: int, vehicle_data: Dict) -> Dict:
        try:
            response = requests.post(
                f"{self.api_url}/characters/{character_id}/vehicles",
                json=vehicle_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to add vehicle: {e}")
            return {"error": str(e)}

    def add_character_weapon(self, character_id: int, weapon_data: Dict) -> Dict:
        try:
            response = requests.post(
                f"{self.api_url}/characters/{character_id}/weapons",
                json=weapon_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to add weapon: {e}")
            return {"error": str(e)}

    def update_character_cyberdeck(self, character_id: int, cyberdeck_data: Dict) -> Dict:
        try:
            response = requests.put(
                f"{self.api_url}/characters/{character_id}/cyberdeck",
                json=cyberdeck_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to update cyberdeck: {e}")
            return {"error": str(e)}

    def get_character_query_context(self, character_id: int) -> Dict:
        try:
            response = requests.get(f"{self.api_url}/characters/{character_id}/context", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get character context: {e}")
            return {"error": str(e)}


# Initialize client
client = RAGClient()

# Initialize character API client
char_api = CharacterAPIClient(API_URL)

# Global variables for pagination and state
search_file_paths = {}
current_search_params = {}
current_page = 1

# Global state for character management
selected_character_id = None
character_data_cache = {}

# ===== UPLOAD TAB FUNCTIONS =====

def process_uploads(files, document_type: str) -> tuple:
    """Process uploaded PDFs with document type specification."""
    if not files:
        return "No files selected", gr.update(value=[])

    if not document_type:
        document_type = "rulebook"  # Default fallback

    results = []
    for file in files:
        file_name = Path(file.name).name

        # Upload file with document type
        result = client.upload_pdf(file.name, document_type)
        if "error" in result:
            results.append(f"❌ {file_name}: {result['error']}")
            continue

        job_id = result.get("job_id")
        if not job_id:
            results.append(f"❌ {file_name}: No job ID returned")
            continue

        results.append(f"✅ {file_name}: Processing started as {document_type} (Job: {job_id[-8:]})")

    return "\n".join(results), gr.update(value=[])

def poll_progress():
    """Enhanced progress polling with document type information."""
    try:
        response = requests.get(f"{client.api_url}/jobs", timeout=5)
        response.raise_for_status()
        jobs = response.json().get("active_jobs", {})

        if not jobs:
            return "No active processing jobs", gr.update(visible=False), 0, "✅ All processing complete!"

        # Format progress information
        progress_lines = []
        progress_values = []

        for job_id, job_info in jobs.items():
            stage = job_info.get("stage", "unknown")
            progress = job_info.get("progress", 0)
            details = job_info.get("details", "")

            # Extract document type and filename from job_id
            parts = job_id.split("_")
            if len(parts) >= 2:
                doc_type = parts[0]
                filename = "_".join(parts[1:-1])  # Exclude timestamp
            else:
                doc_type = "unknown"
                filename = job_id

            # Enhanced progress display with document type
            progress_lines.append(
                f"📄 **{filename}** ({doc_type})\n"
                f"   Stage: {stage} | Progress: {progress:.0f}%\n"
                f"   {details}"
            )

            progress_values.append(progress)

        progress_text = "\n\n".join(progress_lines)
        avg_progress = sum(progress_values) / len(progress_values) if progress_values else 0

        # Create upload status summary
        upload_summary = f"📄 Processing {len(jobs)} file(s):\n" + "\n".join([
            f"• {parts[1] if len(parts) >= 2 else job_id}: {stage} ({progress:.0f}%)"
            for job_id, job_info in jobs.items()
            for stage, progress in [(job_info.get('stage', 'unknown'), job_info.get('progress', 0))]
            for parts in [job_id.split('_')]
        ])

        return progress_text, gr.update(visible=True, value=avg_progress), avg_progress, upload_summary

    except Exception as e:
        return f"Error checking progress: {str(e)}", gr.update(visible=False), 0

# ===== QUERY TAB FUNCTIONS =====

def submit_query(
        question: str,
        model: str,
        n_results: int,
        query_type: str,
        character_role: str,
        character_stats: str,
        edition: str,
        filter_section: str,
        filter_subsection: str,
        filter_document_type: str,
        filter_edition: str,
        character_selector: str = "None"  # NEW: Add this parameter
):
    """Enhanced query submission with character context integration."""
    if not question:
        yield "Please enter a question", "", "", [], gr.update(visible=False)
        return

    # NEW: Check for active character and dice pool queries
    try:
        active_char = char_api.get_active_character()

        # Check if this is a dice pool question first
        if active_char and any(keyword in question.lower() for keyword in
                               ['dice', 'roll', 'pool', 'test', 'check']):
            # Try to resolve as dice pool query
            dice_result = char_api.get_dice_pool(active_char['id'], question)

            if not dice_result.get('error') and dice_result.get('dice_pool', 0) > 0:
                # This was successfully resolved as a dice pool query
                explanation = dice_result.get('explanation', '')
                dice_pool = dice_result.get('dice_pool', 0)

                answer = f"🎲 **Dice Pool Calculation:**\n\n{explanation}\n\n**Total: {dice_pool} dice**"

                yield answer, "", f"**Character:** {active_char['name']}", [], gr.update(visible=False)
                return

    except Exception as e:
        logger.warning(f"Character context check failed: {e}")
        # Continue with normal query if character check fails

    # Prepare enhanced parameters (existing code)
    params = {
        "n_results": n_results,
        "query_type": query_type.lower(),
        "model": model,
        "edition": edition if edition != "None" else "SR5"
    }

    # Add optional parameters (existing code)
    if character_role != "None":
        params["character_role"] = character_role.lower().replace(" ", "_")
    if character_stats:
        params["character_stats"] = character_stats
    if filter_section != "All":
        params["filter_section"] = filter_section
    if filter_subsection:
        params["filter_subsection"] = filter_subsection
    if filter_document_type != "All Types":
        params["filter_document_type"] = filter_document_type
    if filter_edition != "All Editions":
        params["filter_edition"] = filter_edition

    # NEW: Add character context if available
    try:
        if active_char:
            # Get character context for the query
            context_response = char_api.get_character_query_context(active_char['id'])
            if not context_response.get('error'):
                character_context = context_response.get('context', '')
                if character_context:
                    params["character_context"] = character_context
                    logger.info(f"Added character context: {character_context}")
    except Exception as e:
        logger.warning(f"Failed to get character context: {e}")
        # Continue without character context

    # Stream response with thinking support (existing code continues unchanged...)
    for response, thinking, metadata, status in client.query_stream(question, **params):
        if status == "error":
            yield response, "", "", [], gr.update(visible=False)
        elif status == "complete" and metadata:
            # Format sources (existing code)
            sources_text = ""
            if metadata.get("sources"):
                sources_list = [Path(s).name for s in metadata["sources"]]
                sources_text = "**Sources:**\n" + "\n".join([f"📄 {s}" for s in sources_list])

            # NEW: Add character info to sources if used
            try:
                if active_char and params.get("character_context"):
                    sources_text += f"\n\n**Character:** {active_char['name']} ({active_char['metatype']})"
            except:
                pass

            # Show applied filters for debugging (existing code)
            if metadata.get("applied_filters"):
                filters_text = f"\n\n**Applied Filters:** {metadata['applied_filters']}"
                sources_text += filters_text

            # Create chunks dataframe (existing code)
            chunks_data = []
            if metadata.get("chunks"):
                for i, (chunk, dist) in enumerate(zip(
                        metadata.get("chunks", []),
                        metadata.get("distances", [])
                )):
                    relevance = f"{(1 - dist):.2%}" if dist else "N/A"
                    content = chunk[:200] + "..." if len(chunk) > 200 else chunk
                    chunks_data.append([relevance, content])

            # Show thinking accordion if there's thinking content (existing code)
            thinking_visible = bool(thinking and thinking.strip())

            yield response, thinking, sources_text, chunks_data, gr.update(visible=thinking_visible)
        else:
            # Still generating (existing code)
            thinking_visible = bool(thinking and thinking.strip())
            cursor = "▌" if status == "generating" else "🤔" if status == "thinking" else ""
            yield response + cursor, thinking, "", [], gr.update(visible=thinking_visible)


# ===== DOCUMENT TAB FUNCTIONS =====

def refresh_documents():
    """Enhanced document library refresh with metadata statistics."""
    docs = client.get_documents()
    stats = client.get_document_stats()

    if "error" in stats:
        return f"Error loading stats: {stats['error']}", {}, pd.DataFrame(), "", gr.update(choices=[]), gr.update(
            choices=[]), gr.update(choices=[]), gr.update(choices=[])

    if not docs:
        return "No documents indexed yet", {}, pd.DataFrame(), "", gr.update(choices=[]), gr.update(
            choices=[]), gr.update(choices=[]), gr.update(choices=[])

    # Group documents by parent directory
    doc_groups = {}
    for doc_path in docs:
        try:
            parent_name = Path(doc_path).parent.name
            relative_path = str(Path(doc_path).relative_to(Path("data/processed_markdown")))

            if parent_name not in doc_groups:
                doc_groups[parent_name] = []
            doc_groups[parent_name].append({
                "filename": Path(doc_path).name,
                "path": relative_path
            })
        except:
            if "other" not in doc_groups:
                doc_groups["other"] = []
            doc_groups["other"].append({
                "filename": str(doc_path),
                "path": str(doc_path)
            })

    # Enhanced statistics with metadata breakdown
    total_chunks = stats.get('total_chunks', 0)
    unique_docs = stats.get('unique_documents', 0)
    active_jobs = 0  # Will be updated from status

    try:
        status = client.get_status()
        active_jobs = status.get('active_jobs', 0)
    except:
        pass

    stats_text = f"""
**Enhanced System Statistics:**
- Unique Documents: {unique_docs}
- Total Text Chunks: {total_chunks}
- Active Jobs: {active_jobs}
- Document Types: {len(stats.get('document_types', {}))}
- Editions Found: {len(stats.get('editions', {}))}
- Sections Detected: {len(stats.get('sections', {}))}
"""

    # Create detailed stats dataframe
    stats_rows = [
        {"Category": "Documents", "Count": unique_docs},
        {"Category": "Text Chunks", "Count": total_chunks},
        {"Category": "Active Jobs", "Count": active_jobs},
    ]

    # Add document type breakdown
    doc_types = stats.get('document_types', {})
    for doc_type, count in doc_types.items():
        stats_rows.append({"Category": f"Type: {doc_type}", "Count": count})

    # Add edition breakdown
    editions = stats.get('editions', {})
    for edition, count in editions.items():
        stats_rows.append({"Category": f"Edition: {edition}", "Count": count})

    stats_df = pd.DataFrame(stats_rows)

    # Update dropdown choices
    group_choices = ["All Groups"] + list(doc_groups.keys())

    # Document type choices from stats
    type_choices = ["All Types"] + list(doc_types.keys())

    # Edition choices from stats
    edition_choices = ["All Editions"] + list(editions.keys())

    print(f"DEBUG: doc_groups keys: {list(doc_groups.keys()) if doc_groups else 'None'}")
    print(f"DEBUG: group_choices: {group_choices}")

    return (
        f"Total: {len(docs)} documents in {len(doc_groups)} groups",
        doc_groups,
        stats_df,
        stats_text,
        gr.update(choices=group_choices),
        gr.update(choices=type_choices),
        gr.update(choices=edition_choices),
        gr.update(choices=group_choices)
    )


def load_document_group(group_name, doc_groups):
    """Load all files for a specific document group."""
    if not group_name or group_name not in doc_groups:
        return gr.update(choices=[], visible=False)

    files = doc_groups[group_name]
    choices = []

    for file_info in files:
        display_name = f"📄 {file_info['filename']}"
        choices.append(display_name)

    return gr.update(choices=choices, visible=True, value=None)


def handle_library_file_selection(selected_file, group_name, doc_groups):
    """Handle file selection from document library."""
    if not selected_file or not group_name or group_name not in doc_groups:
        return "No file selected"

    # Find the file path
    filename = selected_file.replace("📄 ", "")
    for file_info in doc_groups[group_name]:
        if file_info['filename'] == filename:
            return load_document_content(file_info['path'])

    return "File not found"


def search_docs_fn(query: str, selected_group: str, selected_type: str, selected_edition: str, page: int = 1):
    """Enhanced document search with metadata filtering and pagination."""
    global search_file_paths, current_search_params, current_page

    # Store search parameters for pagination
    current_search_params = {
        "query": query,
        "selected_group": selected_group,
        "selected_type": selected_type,
        "selected_edition": selected_edition
    }
    current_page = page

    search_file_paths = {}

    if not query.strip():
        return (
            "Enter a search query to find documents",
            gr.update(choices=[], visible=False),
            "",
            gr.update(visible=False),
            gr.update(value="Page 0 of 0")
        )

    group_filter = None if selected_group == "All Groups" else selected_group
    type_filter = None if selected_type == "All Types" else selected_type
    edition_filter = None if selected_edition == "All Editions" else selected_edition

    results = client.search_documents(
        query, group_filter, type_filter, edition_filter, page, 20
    )

    if "error" in results:
        return (
            f"Search error: {results['error']}",
            gr.update(choices=[], visible=False),
            "",
            gr.update(visible=False),
            gr.update(value="Page 0 of 0")
        )

    files = results.get("results", [])
    total = results.get("total", 0)
    current_page_num = results.get("page", 1)
    total_pages = results.get("total_pages", 1)

    if not files:
        return (
            "No files found matching your search",
            gr.update(choices=[], visible=False),
            "",
            gr.update(visible=False),
            gr.update(value="Page 0 of 0")
        )

    # Create choices for radio buttons and store file paths
    choices = []
    for file_info in files:
        match_type = file_info.get("match_type", "")
        match_indicator = {"filename": "📝", "content": "🔍", "all": "📄"}.get(match_type, "📄")

        display_name = f"{match_indicator} {file_info['filename']} | 📂 {file_info['group']}"
        choices.append(display_name)
        search_file_paths[display_name] = file_info['file_path']

    # Enhanced summary with filter info
    filter_info = []
    if group_filter:
        filter_info.append(f"Group: {group_filter}")
    if type_filter:
        filter_info.append(f"Type: {type_filter}")
    if edition_filter:
        filter_info.append(f"Edition: {edition_filter}")

    filter_text = f" | Filters: {', '.join(filter_info)}" if filter_info else ""
    summary_text = f"**Found {total} files**{filter_text} - Click to view:"

    # Auto-load first result
    first_content = load_document_content(files[0]['file_path'])

    # Update pagination info
    page_text = f"Page {current_page_num} of {total_pages}"
    pagination_visible = total_pages > 1

    return (
        summary_text,
        gr.update(choices=choices, visible=True, value=choices[0]),
        first_content,
        gr.update(visible=pagination_visible),
        gr.update(value=page_text)
    )


def navigate_search_results(direction: int):
    """Navigate search results pagination."""
    global current_search_params, current_page

    if not current_search_params:
        return search_docs_fn("", "All Groups", "All Types", "All Editions", 1)

    new_page = max(1, current_page + direction)

    return search_docs_fn(
        current_search_params["query"],
        current_search_params["selected_group"],
        current_search_params["selected_type"],
        current_search_params["selected_edition"],
        new_page
    )


def handle_file_selection(selected_choice):
    """Handle radio button selection."""
    global search_file_paths

    if not selected_choice or selected_choice not in search_file_paths:
        return "No file selected or file not found"

    file_path = search_file_paths[selected_choice]
    return load_document_content(file_path)


def load_document_content(file_path: str):
    """Load and display document content with enhanced metadata display."""
    if not file_path:
        return "No file selected"

    result = client.get_document_content(file_path)

    if "error" in result:
        return f"❌ **Error loading file**\n\n{result['error']}"

    content = result.get("content", "")
    filename = Path(file_path).name

    # Extract and format YAML metadata if present
    metadata_display = ""
    if content.startswith('---'):
        yaml_end = content.find('---', 3)
        if yaml_end > 0:
            yaml_content = content[3:yaml_end].strip()
            main_content = content[yaml_end + 3:].strip()

            # Parse key metadata for display
            metadata_lines = []
            for line in yaml_content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    if key in ['document_type', 'edition', 'primary_focus']:
                        metadata_lines.append(f"**{key.title()}:** {value}")

            if metadata_lines:
                metadata_display = f"\n\n📊 **Metadata:**  \n" + "  \n".join(metadata_lines) + "\n\n---\n\n"
        else:
            main_content = content
    else:
        main_content = content

    # Format with enhanced header
    formatted_content = f"""# 📄 {filename}

**Path:** `{file_path}`{metadata_display}

{main_content}"""

    return formatted_content


def reindex_documents(force_reindex: bool):
    """Trigger enhanced document reindexing."""
    result = client.reindex(force=force_reindex)
    if "error" in result:
        return f"❌ Reindexing failed: {result['error']}"
    return "✅ Enhanced reindexing complete with metadata extraction!"


def refresh_character_list():
    """Refresh the character list dropdown."""
    characters = char_api.list_characters()

    if not characters:
        return gr.update(choices=["No characters"], value=None), "No characters found. Create one below!"

    # Format choices
    choices = []
    active_char = char_api.get_active_character()
    active_id = active_char.get("id") if active_char else None

    for char in characters:
        label = f"👤 {char['name']} ({char['metatype']})"
        if char['id'] == active_id:
            label += " ⭐"
        choices.append((label, char['id']))

    return gr.update(choices=choices, value=active_id), f"Found {len(characters)} characters"


def create_new_character(name: str, metatype: str, archetype: str):
    """Create a new character."""
    if not name.strip():
        return "❌ Character name is required", gr.update(), ""

    result = char_api.create_character(name.strip(), metatype, archetype.strip())

    if "error" in result:
        return f"❌ Failed to create character: {result['error']}", gr.update(), ""

    # Refresh character list
    char_dropdown, status = refresh_character_list()

    return f"✅ Character '{name}' created successfully!", char_dropdown, ""


def select_character(character_id):
    """Load character data when selected."""
    global selected_character_id, character_data_cache

    if not character_id:
        return "No character selected", {}, {}, [], [], [], [], [], [], {}

    selected_character_id = character_id

    # Get character data
    char_data = char_api.get_character(character_id)
    if "error" in char_data:
        return f"❌ Error loading character: {char_data['error']}", {}, {}, [], [], [], [], [], [], {}

    character_data_cache[character_id] = char_data

    # Extract data for forms
    stats = char_data.get('stats', {})
    resources = char_data.get('resources', {})

    # Format skills for display
    skills_display = []
    for skill_type, skills in char_data.get('skills', {}).items():
        for skill in skills:
            dice_pool = 0
            if skill.get('attribute') and stats:
                attr_val = stats.get(skill['attribute'].lower(), 0)
                dice_pool = attr_val + skill.get('rating', 0)

            skills_display.append([
                skill['name'], skill.get('rating', 0),
                skill.get('specialization', ''), skill_type, dice_pool
            ])

    # Format other data for display
    qualities_display = [[q['name'], q.get('rating', ''), q.get('quality_type', '')]
                         for q in char_data.get('qualities', [])]

    gear_display = [[g['name'], g.get('quantity', 1), g.get('category', ''), g.get('armor_value', 0)]
                    for g in char_data.get('gear', [])]

    weapons_display = [[w['name'], w.get('weapon_type', ''), w.get('damage_code', ''), w.get('armor_penetration', 0)]
                       for w in char_data.get('weapons', [])]

    vehicles_display = [[v['name'], v.get('vehicle_type', ''), v.get('handling', 0), v.get('speed', 0)]
                        for v in char_data.get('vehicles', [])]

    programs_display = [[p['name'], p.get('rating', 1), p.get('program_type', '')]
                        for p in char_data.get('programs', [])]

    cyberdeck = char_data.get('cyberdeck', {})

    return (
        f"✅ Loaded character: {char_data['name']}",
        stats, resources, skills_display, qualities_display,
        gear_display, weapons_display, vehicles_display, programs_display, cyberdeck
    )


def set_active_character(character_id):
    """Set the active character for queries."""
    if not character_id:
        return "No character selected"

    result = char_api.set_active_character(character_id)
    if "error" in result:
        return f"❌ Failed to set active character: {result['error']}"

    return f"✅ {result['message']}"


def delete_selected_character(character_id):
    """Delete the selected character."""
    if not character_id:
        return "No character selected", gr.update()

    # Get character name first
    char_data = character_data_cache.get(character_id, {})
    char_name = char_data.get('name', f'Character {character_id}')

    result = char_api.delete_character(character_id)
    if "error" in result:
        return f"❌ Failed to delete character: {result['error']}", gr.update()

    # Refresh character list
    char_dropdown, _ = refresh_character_list()

    return f"✅ Character '{char_name}' deleted successfully!", char_dropdown


# ===== STATS & RESOURCES FUNCTIONS =====

def update_character_stats(character_id, body, agility, reaction, strength, charisma, logic,
                           intuition, willpower, edge, essence, physical_limit, mental_limit,
                           social_limit, initiative, hot_sim_vr):
    """Update character statistics."""
    if not character_id:
        return "No character selected"

    stats_data = {
        "body": body, "agility": agility, "reaction": reaction, "strength": strength,
        "charisma": charisma, "logic": logic, "intuition": intuition, "willpower": willpower,
        "edge": edge, "essence": essence, "physical_limit": physical_limit,
        "mental_limit": mental_limit, "social_limit": social_limit,
        "initiative": initiative, "hot_sim_vr": hot_sim_vr
    }

    result = char_api.update_character_stats(character_id, stats_data)
    if "error" in result:
        return f"❌ Failed to update stats: {result['error']}"

    return "✅ Character stats updated successfully!"


def update_character_resources(character_id, nuyen, street_cred, notoriety, public_aware,
                               total_karma, available_karma, edge_pool):
    """Update character resources."""
    if not character_id:
        return "No character selected"

    resources_data = {
        "nuyen": nuyen, "street_cred": street_cred, "notoriety": notoriety,
        "public_aware": public_aware, "total_karma": total_karma,
        "available_karma": available_karma, "edge_pool": edge_pool
    }

    result = char_api.update_character_resources(character_id, resources_data)
    if "error" in result:
        return f"❌ Failed to update resources: {result['error']}"

    return "✅ Character resources updated successfully!"


# ===== SKILLS FUNCTIONS =====

def get_skills_for_dropdown(skill_type: str):
    """Get skills from library for dropdown - IMPROVED VERSION."""
    skills = char_api.get_skills_library(skill_type)

    if not skills:
        return gr.update(choices=[("No skills found - run 'Populate Reference Data'", None)], value=None)

    choices = []
    for skill in skills:
        # Create descriptive choice text
        attr_text = f" ({skill.get('linked_attribute', 'Unknown')})" if skill.get('linked_attribute') else ""
        choice_text = f"{skill['name']}{attr_text}"
        choices.append((choice_text, skill['name']))

    return gr.update(choices=choices, value=None)


def add_character_skill(character_id, skill_name, rating, specialization, skill_type, current_skills):
    """Add a skill to the character."""
    if not character_id or not skill_name:
        return "No character or skill selected", current_skills

    # Get skill details from library
    skills_library = char_api.get_skills_library(skill_type)
    skill_info = next((s for s in skills_library if s['name'] == skill_name), {})

    skill_data = {
        "name": skill_name,
        "rating": rating,
        "specialization": specialization,
        "skill_type": skill_type,
        "skill_group": skill_info.get('skill_group', ''),
        "attribute": skill_info.get('linked_attribute', '')
    }

    result = char_api.add_character_skill(character_id, skill_data)
    if "error" in result:
        return f"❌ Failed to add skill: {result['error']}", current_skills

    # Add to current skills display
    new_skills = current_skills + [[skill_name, rating, specialization, skill_type, rating]]

    return f"✅ Skill '{skill_name}' added successfully!", new_skills


def remove_character_skill(character_id, selected_skill_index, current_skills):
    """Remove a skill from the character."""
    if not character_id or selected_skill_index is None or not current_skills:
        return "No character or skill selected", current_skills

    try:
        skill_name = current_skills[selected_skill_index][0]

        result = char_api.remove_character_skill(character_id, skill_name)
        if "error" in result:
            return f"❌ Failed to remove skill: {result['error']}", current_skills

        # Remove from display
        updated_skills = [skill for i, skill in enumerate(current_skills) if i != selected_skill_index]

        return f"✅ Skill '{skill_name}' removed successfully!", updated_skills
    except (IndexError, KeyError) as e:
        return f"❌ Error removing skill: {str(e)}", current_skills


# ===== QUALITIES FUNCTIONS =====

def get_qualities_for_dropdown(quality_type: str):
    """Get qualities from library for dropdown - IMPROVED VERSION."""
    qualities = char_api.get_qualities_library(quality_type)

    if not qualities:
        return gr.update(choices=[("No qualities found - run 'Populate Reference Data'", None)], value=None)

    choices = []
    for quality in qualities:
        # Create descriptive choice text with karma cost
        karma_cost = quality.get('karma_cost', 0)
        karma_text = f" ({karma_cost} karma)" if karma_cost != 0 else ""
        choice_text = f"{quality['name']}{karma_text}"
        choices.append((choice_text, quality['name']))

    return gr.update(choices=choices, value=None)


def add_character_quality(character_id, quality_name, rating, quality_type, current_qualities):
    """Add a quality to the character."""
    if not character_id or not quality_name:
        return "No character or quality selected", current_qualities

    # Get quality details from library
    qualities_library = char_api.get_qualities_library(quality_type)
    quality_info = next((q for q in qualities_library if q['name'] == quality_name), {})

    quality_data = {
        "name": quality_name,
        "rating": rating,
        "quality_type": quality_type,
        "karma_cost": quality_info.get('karma_cost', 0),
        "description": quality_info.get('description', '')
    }

    result = char_api.add_character_quality(character_id, quality_data)
    if "error" in result:
        return f"❌ Failed to add quality: {result['error']}", current_qualities

    # Add to current qualities display
    new_qualities = current_qualities + [[quality_name, rating, quality_type]]

    return f"✅ Quality '{quality_name}' added successfully!", new_qualities


def remove_character_quality(character_id, selected_quality_index, current_qualities):
    """Remove a quality from the character."""
    if not character_id or selected_quality_index is None or not current_qualities:
        return "No character or quality selected", current_qualities

    try:
        quality_name = current_qualities[selected_quality_index][0]

        result = char_api.remove_character_quality(character_id, quality_name)
        if "error" in result:
            return f"❌ Failed to remove quality: {result['error']}", current_qualities

        # Remove from display
        updated_qualities = [q for i, q in enumerate(current_qualities) if i != selected_quality_index]

        return f"✅ Quality '{quality_name}' removed successfully!", updated_qualities
    except (IndexError, KeyError) as e:
        return f"❌ Error removing quality: {str(e)}", current_qualities


# ===== GEAR FUNCTIONS =====

def get_gear_for_dropdown(category: str):
    """Get gear from library for dropdown - IMPROVED VERSION."""
    if not category:
        return gr.update(choices=[], value=None)

    gear_items = char_api.get_gear_library(category)

    if not gear_items:
        return gr.update(choices=[("No gear found in this category", None)], value=None)

    choices = []
    for gear in gear_items:
        # Create descriptive choice text with cost
        cost = gear.get('base_cost', 0)
        cost_text = f" ({cost}¥)" if cost > 0 else ""
        choice_text = f"{gear['name']}{cost_text}"
        choices.append((choice_text, gear['name']))

    return gr.update(choices=choices, value=None)


def add_character_gear(character_id, gear_name, category, quantity, rating, current_gear):
    """Add gear to the character."""
    if not character_id or not gear_name:
        return "No character or gear selected", current_gear

    # Get gear details from library
    gear_library = char_api.get_gear_library(category)
    gear_info = next((g for g in gear_library if g['name'] == gear_name), {})

    gear_data = {
        "name": gear_name,
        "category": category,
        "subcategory": gear_info.get('subcategory', ''),
        "quantity": quantity,
        "rating": rating,
        "armor_value": gear_info.get('armor_value', 0),
        "cost": gear_info.get('base_cost', 0),
        "availability": gear_info.get('availability', ''),
        "description": gear_info.get('description', '')
    }

    result = char_api.add_character_gear(character_id, gear_data)
    if "error" in result:
        return f"❌ Failed to add gear: {result['error']}", current_gear

    # Add to current gear display
    new_gear = current_gear + [[gear_name, quantity, category, gear_info.get('armor_value', 0)]]

    return f"✅ Gear '{gear_name}' added successfully!", new_gear


# ===== WEAPONS FUNCTIONS =====

def add_character_weapon(character_id, weapon_name, weapon_type, mode_ammo, accuracy,
                         damage_code, armor_penetration, recoil_compensation, current_weapons):
    """Add a weapon to the character."""
    if not character_id or not weapon_name:
        return "No character or weapon name provided", current_weapons

    weapon_data = {
        "name": weapon_name,
        "weapon_type": weapon_type,
        "mode_ammo": mode_ammo,
        "accuracy": accuracy,
        "damage_code": damage_code,
        "armor_penetration": armor_penetration,
        "recoil_compensation": recoil_compensation,
        "cost": 0,  # Could add cost input if needed
        "availability": "",
        "description": ""
    }

    result = char_api.add_character_weapon(character_id, weapon_data)
    if "error" in result:
        return f"❌ Failed to add weapon: {result['error']}", current_weapons

    # Add to current weapons display
    new_weapons = current_weapons + [[weapon_name, weapon_type, damage_code, armor_penetration]]

    return f"✅ Weapon '{weapon_name}' added successfully!", new_weapons


def remove_character_weapon(character_id, selected_weapon_index, current_weapons):
    """Remove a weapon from the character."""
    if not character_id or selected_weapon_index is None or not current_weapons:
        return "No character or weapon selected", current_weapons

    try:
        # Note: This is a limitation - we need the actual weapon ID from the database
        # For now, we'll use the weapon name and let the backend handle it
        weapon_name = current_weapons[selected_weapon_index][0]

        # This would need to be modified to use actual weapon IDs
        # For now, this is a placeholder that shows the pattern
        result = {"message": f"Weapon removal not fully implemented yet"}

        # Remove from display (optimistic update)
        updated_weapons = [w for i, w in enumerate(current_weapons) if i != selected_weapon_index]

        return f"⚠️ Weapon '{weapon_name}' removed from display (backend removal needs weapon ID)", updated_weapons
    except (IndexError, KeyError) as e:
        return f"❌ Error removing weapon: {str(e)}", current_weapons


# ===== VEHICLES FUNCTIONS =====

def add_character_vehicle(character_id, vehicle_name, vehicle_type, handling, speed,
                          acceleration, body, armor, pilot, sensor, seats, current_vehicles):
    """Add a vehicle/drone to the character."""
    if not character_id or not vehicle_name:
        return "No character or vehicle name provided", current_vehicles

    vehicle_data = {
        "name": vehicle_name,
        "vehicle_type": vehicle_type,
        "handling": handling,
        "speed": speed,
        "acceleration": acceleration,
        "body": body,
        "armor": armor,
        "pilot": pilot,
        "sensor": sensor,
        "seats": seats,
        "cost": 0,
        "availability": "",
        "description": ""
    }

    result = char_api.add_character_vehicle(character_id, vehicle_data)
    if "error" in result:
        return f"❌ Failed to add vehicle: {result['error']}", current_vehicles

    # Add to current vehicles display
    new_vehicles = current_vehicles + [[vehicle_name, vehicle_type, handling, speed]]

    return f"✅ Vehicle '{vehicle_name}' added successfully!", new_vehicles


def remove_character_vehicle(character_id, selected_vehicle_index, current_vehicles):
    """Remove a vehicle from the character."""
    if not character_id or selected_vehicle_index is None or not current_vehicles:
        return "No character or vehicle selected", current_vehicles

    try:
        vehicle_name = current_vehicles[selected_vehicle_index][0]

        # Similar limitation as weapons - needs actual vehicle ID
        result = {"message": f"Vehicle removal not fully implemented yet"}

        # Remove from display (optimistic update)
        updated_vehicles = [v for i, v in enumerate(current_vehicles) if i != selected_vehicle_index]

        return f"⚠️ Vehicle '{vehicle_name}' removed from display (backend removal needs vehicle ID)", updated_vehicles
    except (IndexError, KeyError) as e:
        return f"❌ Error removing vehicle: {str(e)}", current_vehicles


# ===== PROGRAMS FUNCTIONS =====

def add_character_program(character_id, program_name, program_rating, program_type, current_programs):
    """Add a program to character's cyberdeck."""
    if not character_id or not program_name:
        return "No character or program name provided", current_programs

    program_data = {
        "name": program_name,
        "rating": program_rating,
        "program_type": program_type,
        "description": ""
    }

    result = char_api.add_character_program(character_id, program_data)
    if "error" in result:
        return f"❌ Failed to add program: {result['error']}", current_programs

    # Add to current programs display
    new_programs = current_programs + [[program_name, program_rating, program_type]]

    return f"✅ Program '{program_name}' added successfully!", new_programs


def remove_character_program(character_id, selected_program_index, current_programs):
    """Remove a program from the character."""
    if not character_id or selected_program_index is None or not current_programs:
        return "No character or program selected", current_programs

    try:
        program_name = current_programs[selected_program_index][0]

        # Similar limitation - needs actual program ID
        result = {"message": f"Program removal not fully implemented yet"}

        # Remove from display (optimistic update)
        updated_programs = [p for i, p in enumerate(current_programs) if i != selected_program_index]

        return f"⚠️ Program '{program_name}' removed from display (backend removal needs program ID)", updated_programs
    except (IndexError, KeyError) as e:
        return f"❌ Error removing program: {str(e)}", current_programs


# ===== CYBERDECK FUNCTIONS =====

def update_character_cyberdeck(character_id, deck_name, device_rating, attack, sleaze,
                               firewall, data_processing, matrix_damage):
    """Update character's cyberdeck."""
    if not character_id:
        return "No character selected"

    cyberdeck_data = {
        "name": deck_name,
        "device_rating": device_rating,
        "attack": attack,
        "sleaze": sleaze,
        "firewall": firewall,
        "data_processing": data_processing,
        "matrix_damage": matrix_damage,
        "cost": 0,
        "availability": "",
        "description": ""
    }

    result = char_api.update_character_cyberdeck(character_id, cyberdeck_data)
    if "error" in result:
        return f"❌ Failed to update cyberdeck: {result['error']}"

    return "✅ Cyberdeck updated successfully!"


# ===== EXPORT FUNCTIONS =====

def export_character_for_gm(character_id):
    """Export character data for GM."""
    if not character_id:
        return "No character selected"

    try:
        json_data = char_api.export_character_json(character_id)
        if json_data:
            # Save to temporary file for download
            import tempfile
            import os

            char_data = character_data_cache.get(character_id, {})
            char_name = char_data.get('name', 'character').replace(' ', '_')

            temp_file = tempfile.NamedTemporaryFile(
                mode='wb',
                suffix=f'_{char_name}.json',
                delete=False
            )
            temp_file.write(json_data)
            temp_file.close()

            return f"✅ Character exported successfully! Download: {temp_file.name}"
        else:
            return "❌ Failed to export character data"
    except Exception as e:
        return f"❌ Export error: {str(e)}"


def populate_reference_tables():
    """Populate reference tables from rulebooks."""
    result = char_api.populate_reference_data()
    if "error" in result:
        return f"❌ Failed to populate reference data: {result['error']}"

    return "✅ Reference tables populated from rulebooks successfully!"


# ===== CHARACTER SELECTOR FOR QUERY TAB =====

def get_character_selector_choices():
    """Get character choices for the query tab dropdown."""
    characters = char_api.list_characters()
    if not characters:
        return gr.update(choices=["None"], value="None")

    choices = ["None"] + [f"{char['name']} ({char['metatype']})" for char in characters]

    # Set active character as default
    active_char = char_api.get_active_character()
    default_value = "None"
    if active_char:
        default_value = f"{active_char['name']} ({active_char['metatype']})"

    return gr.update(choices=choices, value=default_value)


def handle_character_selection_for_query(character_choice):
    """Handle character selection in query tab."""
    if character_choice == "None":
        return "No character selected for queries"

    # Find character by name
    characters = char_api.list_characters()
    selected_char = None
    for char in characters:
        if f"{char['name']} ({char['metatype']})" == character_choice:
            selected_char = char
            break

    if not selected_char:
        return "Character not found"

    # Set as active
    result = char_api.set_active_character(selected_char['id'])
    if "error" in result:
        return f"Failed to set active character: {result['error']}"

    return f"Active character: {selected_char['name']} - queries will include character context"


# ===== CHARACTER MANAGEMENT TAB BUILDER =====

def build_character_management_tab():
    """Build the complete character management tab with all sub-sections."""

    with gr.Tab("👤 Characters"):
        # Character selection and management header
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎭 Character Management")

                # Character selection dropdown
                character_selector = gr.Dropdown(
                    label="Select Character",
                    choices=[("No characters", None)],
                    value=None,
                    interactive=True
                )

                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    set_active_btn = gr.Button("⭐ Set Active", variant="primary", size="sm")
                    delete_btn = gr.Button("🗑️ Delete", variant="stop", size="sm")

                character_status = gr.Markdown("Ready to manage characters")

                # Character creation section
                with gr.Accordion("➕ Create New Character", open=False):
                    new_char_name = gr.Textbox(label="Character Name", placeholder="Enter character name")

                    with gr.Row():
                        new_char_metatype = gr.Dropdown(
                            label="Metatype",
                            choices=["Human", "Elf", "Dwarf", "Ork", "Troll"],
                            value="Human"
                        )
                        new_char_archetype = gr.Textbox(
                            label="Archetype",
                            placeholder="e.g., Street Samurai, Decker"
                        )

                    create_btn = gr.Button("➕ Create Character", variant="primary")
                    create_status = gr.Textbox(label="Creation Status", interactive=False, lines=2)

        # Character details in tabbed interface
        with gr.Column(scale=3):
            with gr.Tabs():
                # ===== STATS & RESOURCES TAB =====
                with gr.Tab("📊 Stats & Resources"):
                    with gr.Row():
                        # Attributes
                        with gr.Column():
                            gr.Markdown("#### 🏋️ Attributes")

                            with gr.Row():
                                body_input = gr.Number(label="Body", value=1, minimum=1, maximum=12)
                                agility_input = gr.Number(label="Agility", value=1, minimum=1, maximum=12)
                                reaction_input = gr.Number(label="Reaction", value=1, minimum=1, maximum=12)

                            with gr.Row():
                                strength_input = gr.Number(label="Strength", value=1, minimum=1, maximum=12)
                                charisma_input = gr.Number(label="Charisma", value=1, minimum=1, maximum=12)
                                logic_input = gr.Number(label="Logic", value=1, minimum=1, maximum=12)

                            with gr.Row():
                                intuition_input = gr.Number(label="Intuition", value=1, minimum=1, maximum=12)
                                willpower_input = gr.Number(label="Willpower", value=1, minimum=1, maximum=12)
                                edge_input = gr.Number(label="Edge", value=1, minimum=1, maximum=7)

                            essence_input = gr.Number(label="Essence", value=6.0, minimum=0, maximum=6, step=0.01)

                        # Limits and Matrix
                        with gr.Column():
                            gr.Markdown("#### 🎯 Limits & Initiative")

                            physical_limit_input = gr.Number(label="Physical Limit", value=1, minimum=1)
                            mental_limit_input = gr.Number(label="Mental Limit", value=1, minimum=1)
                            social_limit_input = gr.Number(label="Social Limit", value=1, minimum=1)
                            initiative_input = gr.Number(label="Initiative", value=1, minimum=1)
                            hot_sim_vr_input = gr.Number(label="Hot Sim VR", value=0, minimum=0)

                    with gr.Row():
                        # Resources
                        with gr.Column():
                            gr.Markdown("#### 💰 Resources")

                            with gr.Row():
                                nuyen_input = gr.Number(label="Nuyen", value=0, minimum=0)
                                street_cred_input = gr.Number(label="Street Cred", value=0, minimum=0)

                            with gr.Row():
                                notoriety_input = gr.Number(label="Notoriety", value=0, minimum=0)
                                public_aware_input = gr.Number(label="Public Aware", value=0, minimum=0)

                            with gr.Row():
                                total_karma_input = gr.Number(label="Total Karma", value=0, minimum=0)
                                available_karma_input = gr.Number(label="Available Karma", value=0, minimum=0)
                                edge_pool_input = gr.Number(label="Edge Pool", value=1, minimum=1)

                    # Update buttons
                    with gr.Row():
                        update_stats_btn = gr.Button("💾 Save Stats", variant="primary")
                        update_resources_btn = gr.Button("💾 Save Resources", variant="primary")

                    stats_update_status = gr.Textbox(label="Update Status", interactive=False)

                # ===== SKILLS TAB =====
                with gr.Tab("🎯 Skills"):
                    with gr.Row():
                        # Current skills display
                        with gr.Column(scale=2):
                            gr.Markdown("#### 📋 Current Skills")

                            skills_table = gr.Dataframe(
                                headers=["Skill", "Rating", "Specialization", "Type", "Dice Pool"],
                                datatype=["str", "number", "str", "str", "number"],
                                value=[],
                                interactive=False,
                                label="Character Skills"
                            )

                            with gr.Row():
                                remove_skill_btn = gr.Button("🗑️ Remove Selected", variant="secondary")
                                selected_skill_index = gr.State()

                        # Add skills interface
                        with gr.Column(scale=1):
                            gr.Markdown("#### ➕ Add Skills")

                            skill_type_selector = gr.Radio(
                                label="Skill Type",
                                choices=["active", "knowledge", "language"],
                                value="active"
                            )

                            skill_dropdown = gr.Dropdown(
                                label="Select Skill",
                                choices=[],
                                interactive=True
                            )

                            skill_rating_input = gr.Number(
                                label="Rating",
                                value=1,
                                minimum=1,
                                maximum=12
                            )

                            skill_specialization_input = gr.Textbox(
                                label="Specialization",
                                placeholder="Optional specialization"
                            )

                            add_skill_btn = gr.Button("➕ Add Skill", variant="primary")

                            skills_status = gr.Textbox(label="Skills Status", interactive=False, lines=3)

                # ===== QUALITIES TAB =====
                with gr.Tab("⭐ Qualities"):
                    with gr.Row():
                        # Current qualities display
                        with gr.Column(scale=2):
                            gr.Markdown("#### 📋 Current Qualities")

                            qualities_table = gr.Dataframe(
                                headers=["Quality", "Rating", "Type"],
                                datatype=["str", "str", "str"],
                                value=[],
                                interactive=False,
                                label="Character Qualities"
                            )

                            with gr.Row():
                                remove_quality_btn = gr.Button("🗑️ Remove Selected", variant="secondary")
                                selected_quality_index = gr.State()

                        # Add qualities interface
                        with gr.Column(scale=1):
                            gr.Markdown("#### ➕ Add Qualities")

                            quality_type_selector = gr.Radio(
                                label="Quality Type",
                                choices=["positive", "negative"],
                                value="positive"
                            )

                            quality_dropdown = gr.Dropdown(
                                label="Select Quality",
                                choices=[],
                                interactive=True
                            )

                            quality_rating_input = gr.Number(
                                label="Rating",
                                value=0,
                                minimum=0,
                                maximum=6
                            )

                            add_quality_btn = gr.Button("➕ Add Quality", variant="primary")

                            qualities_status = gr.Textbox(label="Qualities Status", interactive=False, lines=3)

                # ===== GEAR TAB =====
                with gr.Tab("🎒 Gear"):
                    with gr.Row():
                        # Current gear display
                        with gr.Column(scale=2):
                            gr.Markdown("#### 📋 Current Gear")

                            gear_table = gr.Dataframe(
                                headers=["Item", "Quantity", "Category", "Armor"],
                                datatype=["str", "number", "str", "number"],
                                value=[],
                                interactive=False,
                                label="Character Gear"
                            )

                            with gr.Row():
                                remove_gear_btn = gr.Button("🗑️ Remove Selected", variant="secondary")
                                total_armor_display = gr.Number(
                                    label="Total Armor",
                                    value=0,
                                    interactive=False
                                )

                        # Add gear interface
                        with gr.Column(scale=1):
                            gr.Markdown("#### ➕ Add Gear")

                            gear_category_selector = gr.Dropdown(
                                label="Category",
                                choices=[]
                            )

                            gear_dropdown = gr.Dropdown(
                                label="Select Gear",
                                choices=[],
                                interactive=True
                            )

                            gear_quantity_input = gr.Number(
                                label="Quantity",
                                value=1,
                                minimum=1
                            )

                            gear_rating_input = gr.Number(
                                label="Rating",
                                value=0,
                                minimum=0,
                                maximum=6
                            )

                            add_gear_btn = gr.Button("➕ Add Gear", variant="primary")

                            gear_status = gr.Textbox(label="Gear Status", interactive=False, lines=3)

                # ===== ENHANCED WEAPONS TAB =====
                def create_enhanced_weapons_tab():
                    """Enhanced weapons tab with add/remove functionality."""
                    with gr.Tab("⚔️ Weapons"):
                        with gr.Row():
                            # Current weapons display
                            with gr.Column(scale=2):
                                gr.Markdown("#### 🗡️ Current Weapons")

                                weapons_table = gr.Dataframe(
                                    headers=["Weapon", "Type", "Damage", "AP"],
                                    datatype=["str", "str", "str", "number"],
                                    value=[],
                                    interactive=False,
                                    label="Character Weapons"
                                )

                                with gr.Row():
                                    remove_weapon_btn = gr.Button("🗑️ Remove Selected", variant="secondary")

                            # Add weapons interface
                            with gr.Column(scale=1):
                                gr.Markdown("#### ➕ Add Weapon")

                                weapon_name_input = gr.Textbox(
                                    label="Weapon Name",
                                    placeholder="Enter weapon name"
                                )

                                weapon_type_selector = gr.Radio(
                                    label="Weapon Type",
                                    choices=["melee", "ranged"],
                                    value="ranged"
                                )

                                with gr.Row():
                                    weapon_accuracy_input = gr.Number(
                                        label="Accuracy",
                                        value=0,
                                        minimum=0,
                                        maximum=10
                                    )
                                    weapon_ap_input = gr.Number(
                                        label="AP",
                                        value=0,
                                        minimum=-10,
                                        maximum=0
                                    )

                                weapon_damage_input = gr.Textbox(
                                    label="Damage Code",
                                    placeholder="e.g., 8P, 6S+1"
                                )

                                weapon_mode_input = gr.Textbox(
                                    label="Mode/Ammo",
                                    placeholder="e.g., SA, BF/FA, 30(c)"
                                )

                                weapon_rc_input = gr.Number(
                                    label="Recoil Comp",
                                    value=0,
                                    minimum=0,
                                    maximum=10
                                )

                                add_weapon_btn = gr.Button("➕ Add Weapon", variant="primary")

                        weapons_status = gr.Textbox(label="Weapons Status", interactive=False, lines=2)

                    return {
                        "weapons_table": weapons_table,
                        "weapon_name_input": weapon_name_input,
                        "weapon_type_selector": weapon_type_selector,
                        "weapon_accuracy_input": weapon_accuracy_input,
                        "weapon_ap_input": weapon_ap_input,
                        "weapon_damage_input": weapon_damage_input,
                        "weapon_mode_input": weapon_mode_input,
                        "weapon_rc_input": weapon_rc_input,
                        "add_weapon_btn": add_weapon_btn,
                        "remove_weapon_btn": remove_weapon_btn,
                        "weapons_status": weapons_status
                    }

                # ===== ENHANCED VEHICLES TAB =====
                def create_enhanced_vehicles_tab():
                    """Enhanced vehicles tab with add/remove functionality."""
                    with gr.Tab("🚗 Vehicles"):
                        with gr.Row():
                            # Current vehicles display
                            with gr.Column(scale=2):
                                gr.Markdown("#### 🚗 Vehicles & Drones")

                                vehicles_table = gr.Dataframe(
                                    headers=["Vehicle", "Type", "Handling", "Speed"],
                                    datatype=["str", "str", "number", "number"],
                                    value=[],
                                    interactive=False,
                                    label="Character Vehicles"
                                )

                                with gr.Row():
                                    remove_vehicle_btn = gr.Button("🗑️ Remove Selected", variant="secondary")

                            # Add vehicles interface
                            with gr.Column(scale=1):
                                gr.Markdown("#### ➕ Add Vehicle")

                                vehicle_name_input = gr.Textbox(
                                    label="Vehicle Name",
                                    placeholder="Enter vehicle/drone name"
                                )

                                vehicle_type_selector = gr.Radio(
                                    label="Type",
                                    choices=["vehicle", "drone"],
                                    value="vehicle"
                                )

                                with gr.Row():
                                    vehicle_handling_input = gr.Number(
                                        label="Handling",
                                        value=0,
                                        minimum=-5,
                                        maximum=5
                                    )
                                    vehicle_speed_input = gr.Number(
                                        label="Speed",
                                        value=0,
                                        minimum=0,
                                        maximum=300
                                    )

                                with gr.Row():
                                    vehicle_accel_input = gr.Number(
                                        label="Acceleration",
                                        value=0,
                                        minimum=0,
                                        maximum=50
                                    )
                                    vehicle_body_input = gr.Number(
                                        label="Body",
                                        value=0,
                                        minimum=0,
                                        maximum=20
                                    )

                                with gr.Row():
                                    vehicle_armor_input = gr.Number(
                                        label="Armor",
                                        value=0,
                                        minimum=0,
                                        maximum=20
                                    )
                                    vehicle_pilot_input = gr.Number(
                                        label="Pilot",
                                        value=0,
                                        minimum=0,
                                        maximum=6
                                    )

                                with gr.Row():
                                    vehicle_sensor_input = gr.Number(
                                        label="Sensor",
                                        value=0,
                                        minimum=0,
                                        maximum=8
                                    )
                                    vehicle_seats_input = gr.Number(
                                        label="Seats",
                                        value=0,
                                        minimum=0,
                                        maximum=20
                                    )

                                add_vehicle_btn = gr.Button("➕ Add Vehicle", variant="primary")

                        vehicles_status = gr.Textbox(label="Vehicles Status", interactive=False, lines=2)

                    return {
                        "vehicles_table": vehicles_table,
                        "vehicle_name_input": vehicle_name_input,
                        "vehicle_type_selector": vehicle_type_selector,
                        "vehicle_handling_input": vehicle_handling_input,
                        "vehicle_speed_input": vehicle_speed_input,
                        "vehicle_accel_input": vehicle_accel_input,
                        "vehicle_body_input": vehicle_body_input,
                        "vehicle_armor_input": vehicle_armor_input,
                        "vehicle_pilot_input": vehicle_pilot_input,
                        "vehicle_sensor_input": vehicle_sensor_input,
                        "vehicle_seats_input": vehicle_seats_input,
                        "add_vehicle_btn": add_vehicle_btn,
                        "remove_vehicle_btn": remove_vehicle_btn,
                        "vehicles_status": vehicles_status
                    }

                def create_enhanced_matrix_tab():
                    """Enhanced matrix tab with cyberdeck and programs."""
                    with gr.Tab("🖥️ Matrix"):
                        with gr.Row():
                            # Cyberdeck
                            with gr.Column():
                                gr.Markdown("#### 🖥️ Cyberdeck")

                                cyberdeck_name = gr.Textbox(label="Cyberdeck Name", placeholder="Enter cyberdeck name")
                                cyberdeck_device_rating = gr.Number(label="Device Rating", value=1, minimum=1,
                                                                    maximum=6)

                                with gr.Row():
                                    cyberdeck_attack = gr.Number(label="Attack", value=0, minimum=0, maximum=6)
                                    cyberdeck_sleaze = gr.Number(label="Sleaze", value=0, minimum=0, maximum=6)

                                with gr.Row():
                                    cyberdeck_firewall = gr.Number(label="Firewall", value=0, minimum=0, maximum=6)
                                    cyberdeck_data_proc = gr.Number(label="Data Processing", value=0, minimum=0,
                                                                    maximum=6)

                                cyberdeck_matrix_damage = gr.Number(label="Matrix Damage", value=0, minimum=0)

                                update_cyberdeck_btn = gr.Button("💾 Save Cyberdeck", variant="primary")

                            # Programs
                            with gr.Column():
                                gr.Markdown("#### 📱 Programs")

                                programs_table = gr.Dataframe(
                                    headers=["Program", "Rating", "Type"],
                                    datatype=["str", "number", "str"],
                                    value=[],
                                    interactive=False,
                                    label="Installed Programs"
                                )

                                with gr.Accordion("➕ Add Program", open=False):
                                    program_name_input = gr.Textbox(
                                        label="Program Name",
                                        placeholder="Enter program name"
                                    )

                                    program_rating_input = gr.Number(
                                        label="Rating",
                                        value=1,
                                        minimum=1,
                                        maximum=6
                                    )

                                    program_type_selector = gr.Dropdown(
                                        label="Program Type",
                                        choices=["common", "hacking", "cybercombat", "data"],
                                        value="common"
                                    )

                                    add_program_btn = gr.Button("➕ Add Program", variant="primary")

                                with gr.Row():
                                    remove_program_btn = gr.Button("🗑️ Remove Selected", variant="secondary")

                        matrix_status = gr.Textbox(label="Matrix Status", interactive=False, lines=2)

                    return {
                        "cyberdeck_name": cyberdeck_name,
                        "cyberdeck_device_rating": cyberdeck_device_rating,
                        "cyberdeck_attack": cyberdeck_attack,
                        "cyberdeck_sleaze": cyberdeck_sleaze,
                        "cyberdeck_firewall": cyberdeck_firewall,
                        "cyberdeck_data_proc": cyberdeck_data_proc,
                        "cyberdeck_matrix_damage": cyberdeck_matrix_damage,
                        "update_cyberdeck_btn": update_cyberdeck_btn,
                        "programs_table": programs_table,
                        "program_name_input": program_name_input,
                        "program_rating_input": program_rating_input,
                        "program_type_selector": program_type_selector,
                        "add_program_btn": add_program_btn,
                        "remove_program_btn": remove_program_btn,
                        "matrix_status": matrix_status
                    }

        # Bottom section with utilities
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔧 Utilities")

                with gr.Row():
                    populate_ref_btn = gr.Button("📚 Populate Reference Data", variant="secondary")
                    export_btn = gr.Button("📤 Export for GM", variant="primary")
                    dice_pool_btn = gr.Button("🎲 Quick Dice Pool", variant="secondary")

                utility_status = gr.Textbox(label="Utility Status", interactive=False, lines=2)

    # Return all the components we need to wire up
    return {
        # Character management
        "character_selector": character_selector,
        "refresh_btn": refresh_btn,
        "set_active_btn": set_active_btn,
        "delete_btn": delete_btn,
        "character_status": character_status,

        # Character creation
        "new_char_name": new_char_name,
        "new_char_metatype": new_char_metatype,
        "new_char_archetype": new_char_archetype,
        "create_btn": create_btn,
        "create_status": create_status,

        # Stats inputs
        "body_input": body_input,
        "agility_input": agility_input,
        "reaction_input": reaction_input,
        "strength_input": strength_input,
        "charisma_input": charisma_input,
        "logic_input": logic_input,
        "intuition_input": intuition_input,
        "willpower_input": willpower_input,
        "edge_input": edge_input,
        "essence_input": essence_input,
        "physical_limit_input": physical_limit_input,
        "mental_limit_input": mental_limit_input,
        "social_limit_input": social_limit_input,
        "initiative_input": initiative_input,
        "hot_sim_vr_input": hot_sim_vr_input,

        # Resources inputs
        "nuyen_input": nuyen_input,
        "street_cred_input": street_cred_input,
        "notoriety_input": notoriety_input,
        "public_aware_input": public_aware_input,
        "total_karma_input": total_karma_input,
        "available_karma_input": available_karma_input,
        "edge_pool_input": edge_pool_input,

        # Update buttons and status
        "update_stats_btn": update_stats_btn,
        "update_resources_btn": update_resources_btn,
        "stats_update_status": stats_update_status,

        # Skills components
        "skills_table": skills_table,
        "skill_type_selector": skill_type_selector,
        "skill_dropdown": skill_dropdown,
        "skill_rating_input": skill_rating_input,
        "skill_specialization_input": skill_specialization_input,
        "add_skill_btn": add_skill_btn,
        "remove_skill_btn": remove_skill_btn,
        "skills_status": skills_status,

        # Qualities components
        "qualities_table": qualities_table,
        "quality_type_selector": quality_type_selector,
        "quality_dropdown": quality_dropdown,
        "quality_rating_input": quality_rating_input,
        "add_quality_btn": add_quality_btn,
        "remove_quality_btn": remove_quality_btn,
        "qualities_status": qualities_status,

        # Gear components
        "gear_table": gear_table,
        "gear_category_selector": gear_category_selector,
        "gear_dropdown": gear_dropdown,
        "gear_quantity_input": gear_quantity_input,
        "gear_rating_input": gear_rating_input,
        "add_gear_btn": add_gear_btn,
        "remove_gear_btn": remove_gear_btn,
        "gear_status": gear_status,
        "total_armor_display": total_armor_display,

        # Other tables
        "weapons_table": weapons_table,
        "vehicles_table": vehicles_table,
        "programs_table": programs_table,

        # Cyberdeck
        "cyberdeck_name": cyberdeck_name,
        "cyberdeck_device_rating": cyberdeck_device_rating,
        "cyberdeck_attack": cyberdeck_attack,
        "cyberdeck_sleaze": cyberdeck_sleaze,
        "cyberdeck_firewall": cyberdeck_firewall,
        "cyberdeck_data_proc": cyberdeck_data_proc,
        "cyberdeck_matrix_damage": cyberdeck_matrix_damage,
        "update_cyberdeck_btn": update_cyberdeck_btn,
        "matrix_status": matrix_status,

        # Utilities
        "populate_ref_btn": populate_ref_btn,
        "export_btn": export_btn,
        "dice_pool_btn": dice_pool_btn,
        "utility_status": utility_status,
    }


# ===== WIRE UP ALL EVENT HANDLERS =====

def wire_character_management_events(components):
    """Wire up all the event handlers for character management."""

    # Character selection and management
    components["refresh_btn"].click(
        fn=refresh_character_list,
        outputs=[components["character_selector"], components["character_status"]]
    )

    components["character_selector"].change(
        fn=select_character,
        inputs=[components["character_selector"]],
        outputs=[
            components["character_status"],
            gr.State(),  # stats
            gr.State(),  # resources
            components["skills_table"],
            components["qualities_table"],
            components["gear_table"],
            components["weapons_table"],
            components["vehicles_table"],
            components["programs_table"],
            gr.State()  # cyberdeck
        ]
    )

    components["set_active_btn"].click(
        fn=set_active_character,
        inputs=[components["character_selector"]],
        outputs=[components["character_status"]]
    )

    components["delete_btn"].click(
        fn=delete_selected_character,
        inputs=[components["character_selector"]],
        outputs=[components["character_status"], components["character_selector"]]
    )

    # Character creation
    components["create_btn"].click(
        fn=create_new_character,
        inputs=[
            components["new_char_name"],
            components["new_char_metatype"],
            components["new_char_archetype"]
        ],
        outputs=[
            components["create_status"],
            components["character_selector"],
            components["new_char_name"]
        ]
    )

    # Stats and resources updates
    components["update_stats_btn"].click(
        fn=update_character_stats,
        inputs=[
            components["character_selector"],
            components["body_input"], components["agility_input"], components["reaction_input"],
            components["strength_input"], components["charisma_input"], components["logic_input"],
            components["intuition_input"], components["willpower_input"], components["edge_input"],
            components["essence_input"], components["physical_limit_input"],
            components["mental_limit_input"], components["social_limit_input"],
            components["initiative_input"], components["hot_sim_vr_input"]
        ],
        outputs=[components["stats_update_status"]]
    )

    components["update_resources_btn"].click(
        fn=update_character_resources,
        inputs=[
            components["character_selector"],
            components["nuyen_input"], components["street_cred_input"],
            components["notoriety_input"], components["public_aware_input"],
            components["total_karma_input"], components["available_karma_input"],
            components["edge_pool_input"]
        ],
        outputs=[components["stats_update_status"]]
    )

    # Skills management
    components["skill_type_selector"].change(
        fn=get_skills_for_dropdown,
        inputs=[components["skill_type_selector"]],
        outputs=[components["skill_dropdown"]]
    )

    components["add_skill_btn"].click(
        fn=add_character_skill,
        inputs=[
            components["character_selector"],
            components["skill_dropdown"],
            components["skill_rating_input"],
            components["skill_specialization_input"],
            components["skill_type_selector"],
            components["skills_table"]
        ],
        outputs=[components["skills_status"], components["skills_table"]]
    )

    # Qualities management
    components["quality_type_selector"].change(
        fn=get_qualities_for_dropdown,
        inputs=[components["quality_type_selector"]],
        outputs=[components["quality_dropdown"]]
    )

    components["add_quality_btn"].click(
        fn=add_character_quality,
        inputs=[
            components["character_selector"],
            components["quality_dropdown"],
            components["quality_rating_input"],
            components["quality_type_selector"],
            components["qualities_table"]
        ],
        outputs=[components["qualities_status"], components["qualities_table"]]
    )

    # Gear management
    components["gear_category_selector"].change(
        fn=get_gear_for_dropdown,
        inputs=[components["gear_category_selector"]],
        outputs=[components["gear_dropdown"]]
    )

    components["add_gear_btn"].click(
        fn=add_character_gear,
        inputs=[
            components["character_selector"],
            components["gear_dropdown"],
            components["gear_category_selector"],
            components["gear_quantity_input"],
            components["gear_rating_input"],
            components["gear_table"]
        ],
        outputs=[components["gear_status"], components["gear_table"]]
    )

    # Utilities
    components["populate_ref_btn"].click(
        fn=populate_reference_tables,
        outputs=[components["utility_status"]]
    )

    components["export_btn"].click(
        fn=export_character_for_gm,
        inputs=[components["character_selector"]],
        outputs=[components["utility_status"]]
    )

    # Auto-load reference data on tab load
    components["skill_type_selector"].select(
        fn=get_skills_for_dropdown,
        inputs=[components["skill_type_selector"]],
        outputs=[components["skill_dropdown"]]
    )

    components["quality_type_selector"].select(
        fn=get_qualities_for_dropdown,
        inputs=[components["quality_type_selector"]],
        outputs=[components["quality_dropdown"]]
    )


# ===== EVENT WIRING FOR NEW COMPONENTS =====
def wire_enhanced_component_events(components, character_selector):
    """Wire up events for weapons, vehicles, and programs."""

    # Weapons events
    if "add_weapon_btn" in components:
        components["add_weapon_btn"].click(
            fn=add_character_weapon,
            inputs=[
                character_selector,
                components["weapon_name_input"],
                components["weapon_type_selector"],
                components["weapon_mode_input"],
                components["weapon_accuracy_input"],
                components["weapon_damage_input"],
                components["weapon_ap_input"],
                components["weapon_rc_input"],
                components["weapons_table"]
            ],
            outputs=[components["weapons_status"], components["weapons_table"]]
        )

        components["remove_weapon_btn"].click(
            fn=remove_character_weapon,
            inputs=[
                character_selector,
                gr.State(),  # selected index - would need to implement selection
                components["weapons_table"]
            ],
            outputs=[components["weapons_status"], components["weapons_table"]]
        )

    # Vehicles events
    if "add_vehicle_btn" in components:
        components["add_vehicle_btn"].click(
            fn=add_character_vehicle,
            inputs=[
                character_selector,
                components["vehicle_name_input"],
                components["vehicle_type_selector"],
                components["vehicle_handling_input"],
                components["vehicle_speed_input"],
                components["vehicle_accel_input"],
                components["vehicle_body_input"],
                components["vehicle_armor_input"],
                components["vehicle_pilot_input"],
                components["vehicle_sensor_input"],
                components["vehicle_seats_input"],
                components["vehicles_table"]
            ],
            outputs=[components["vehicles_status"], components["vehicles_table"]]
        )

    # Programs events
    if "add_program_btn" in components:
        components["add_program_btn"].click(
            fn=add_character_program,
            inputs=[
                character_selector,
                components["program_name_input"],
                components["program_rating_input"],
                components["program_type_selector"],
                components["programs_table"]
            ],
            outputs=[components["matrix_status"], components["programs_table"]]
        )

    # Cyberdeck events
    if "update_cyberdeck_btn" in components:
        components["update_cyberdeck_btn"].click(
            fn=update_character_cyberdeck,
            inputs=[
                character_selector,
                components["cyberdeck_name"],
                components["cyberdeck_device_rating"],
                components["cyberdeck_attack"],
                components["cyberdeck_sleaze"],
                components["cyberdeck_firewall"],
                components["cyberdeck_data_proc"],
                components["cyberdeck_matrix_damage"]
            ],
            outputs=[components["matrix_status"]]
        )

# ===== BUILD ENHANCED GRADIO INTERFACE =====

def build_interface():
    """Build the enhanced Gradio interface with document types and thinking support."""

    with gr.Blocks(title="🎲 Shadowrun RAG Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎲 Shadowrun RAG Assistant")
        gr.Markdown("*Your Enhanced AI-powered Guide to the Sixth World*")

        with gr.Tabs():
            # ===== ENHANCED QUERY TAB =====
            with gr.Tab("💬 Query"):
                with gr.Row():
                    with gr.Column(scale=3):
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., 'How do recoil penalties work in Shadowrun 5e?'",
                            lines=3
                        )

                        with gr.Row():
                            submit_btn = gr.Button("🔍 Search", variant="primary")
                            clear_btn = gr.ClearButton(components=[question_input], value="Clear")

                        answer_output = gr.Markdown(label="Answer")

                        # Enhanced thinking accordion with visual indicator
                        with gr.Accordion("🤔 Model Thinking Process", open=False, visible=False) as thinking_accordion:
                            thinking_output = gr.Markdown(
                                label="AI Reasoning",
                                value="*The model's reasoning process will appear here...*"
                            )

                        sources_output = gr.Markdown(label="Sources & Filters")

                        with gr.Accordion("📊 Retrieved Chunks", open=False):
                            chunks_output = gr.Dataframe(
                                headers=["Relevance", "Content"],
                                label="Context Chunks"
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Enhanced Configuration")

                        model_select = gr.Dropdown(
                            choices=client.get_models() or ["llama3:8b-instruct-q4_K_M"],
                            value="llama3:8b-instruct-q4_K_M",
                            label="LLM Model",
                            allow_custom_value=True
                        )

                        n_results_slider = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Number of Sources"
                        )

                        query_type_select = gr.Dropdown(
                            choices=["General", "Rules", "Session"],
                            value="General",
                            label="Query Type"
                        )

                        with gr.Accordion("👤 Character Context", open=True):
                            character_query_selector = gr.Dropdown(
                                label="Active Character",
                                choices=["None"],
                                value="None",
                                info="Select character for context-aware queries"
                            )

                            character_context_display = gr.Textbox(
                                label="Character Status",
                                value="No character selected",
                                interactive=False,
                                lines=2
                            )

                            refresh_char_btn = gr.Button("🔄 Refresh Characters", size="sm")

                        with gr.Accordion("👤 Character Context", open=False):
                            character_role_select = gr.Dropdown(
                                choices=["None", "Decker", "Mage", "Street Samurai",
                                         "Rigger", "Adept", "Technomancer", "Face"],
                                value="None",
                                label="Character Role (overrides section filter)"
                            )

                            character_stats_input = gr.Textbox(
                                label="Character Stats",
                                placeholder="e.g., Logic 6, Hacking 5"
                            )

                            edition_select = gr.Dropdown(
                                choices=["SR5", "SR6", "SR4", "SR3", "None"],  # SR5 first as default
                                value="SR5",  # Default to SR5
                                label="Preferred Edition"
                            )

                        with gr.Accordion("🔍 Enhanced Filters", open=False):
                            gr.Markdown("*Character role selection overrides section filter*")

                            section_filter = gr.Dropdown(
                                choices=["All", "Combat", "Matrix", "Magic", "Gear",
                                         "Character Creation", "Riggers", "Technomancy", "Social"],
                                value="All",
                                label="Filter by Section"
                            )

                            subsection_filter = gr.Textbox(
                                label="Filter by Subsection",
                                placeholder="e.g., Hacking, Spellcasting"
                            )

                            document_type_filter = gr.Dropdown(
                                choices=["All Types", "rulebook", "character_sheet", "universe_info", "adventure"],
                                value="All Types",
                                label="Filter by Document Type"
                            )

                            edition_filter = gr.Dropdown(
                                choices=["All Editions", "SR5", "SR6", "SR4", "SR3"],
                                value="All Editions",
                                label="Filter by Edition"
                            )

                # Wire up enhanced query submission
                submit_btn.click(
                    fn=submit_query,
                    inputs=[
                        question_input, model_select, n_results_slider,
                        query_type_select, character_role_select, character_stats_input,
                        edition_select, section_filter, subsection_filter,
                        document_type_filter, edition_filter, character_query_selector
                    ],
                    outputs=[answer_output, thinking_output, sources_output, chunks_output, thinking_accordion]
                )

                # Character selector refresh
                refresh_char_btn.click(
                    fn=get_character_selector_choices,
                    outputs=[character_query_selector]
                )

                # Handle character selection for queries
                character_query_selector.change(
                    fn=handle_character_selection_for_query,
                    inputs=[character_query_selector],
                    outputs=[character_context_display]
                )

                app.load(
                    fn=get_character_selector_choices,
                    outputs=[character_query_selector]
                )

            # ===== ENHANCED UPLOAD TAB =====
            with gr.Tab("📤 Upload"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📄 Enhanced Document Upload")

                        document_type_select = gr.Dropdown(
                            choices=["rulebook", "character_sheet", "universe_info", "adventure"],
                            value="rulebook",
                            label="Document Type",
                            info="Affects processing strategy and metadata"
                        )

                        file_upload = gr.File(
                            label="Upload PDFs",
                            file_types=[".pdf"],
                            file_count="multiple"
                        )

                        upload_btn = gr.Button("📤 Process Files", variant="primary")

                        upload_status = gr.Textbox(
                            label="Upload Status",
                            lines=5,
                            interactive=False
                        )

                        progress_display = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            label="Overall Progress (%)",
                            interactive=False,
                            visible=False
                        )

                        with gr.Row():
                            check_progress_btn = gr.Button("🔄 Check Progress")
                            auto_refresh = gr.Checkbox(label="Auto-refresh progress", value=False)

                        progress_status = gr.Markdown(label="Enhanced Processing Status")

                    with gr.Column():
                        gr.Markdown("### 📋 Document Type Guide")

                        gr.Markdown("""
                        **📚 Rulebook**: Core rules, supplements, source books
                        - Optimized for rule extraction and semantic chunking
                        - Detects sections: Combat, Magic, Matrix, etc.

                        **👤 Character Sheet**: PC/NPC sheets, character data
                        - Faster processing with table detection
                        - Smaller chunks for precise character info

                        **🌍 Universe Info**: Lore, setting, background material
                        - Focuses on narrative content and world-building
                        - Detects corporations, locations, timeline events

                        **🎯 Adventure**: Scenarios, missions, campaigns
                        - Optimized for plot hooks and encounter data
                        - Extracts NPCs, locations, and story elements
                        """)

                        gr.Markdown("### 🔧 Manual Operations")

                        reindex_btn = gr.Button("🔄 Re-index All Documents")
                        index_new_btn = gr.Button("📊 Index New Documents Only")

                        reindex_output = gr.Textbox(label="Operation Result", lines=2)

                        # Wire up enhanced upload functionality
                        upload_btn.click(
                            fn=process_uploads,
                            inputs=[file_upload, document_type_select],
                            outputs=[upload_status, file_upload]
                        )

                        check_progress_btn.click(
                            fn=poll_progress,
                            outputs=[progress_status, progress_display, gr.State(), upload_status]
                        )

                        reindex_btn.click(
                            fn=lambda: reindex_documents(True),
                            outputs=[reindex_output]
                        )

                        index_new_btn.click(
                            fn=lambda: reindex_documents(False),
                            outputs=[reindex_output]
                        )

                        # Enhanced auto-refresh timer
                def auto_refresh_progress(enable):
                    if enable:
                        return poll_progress()
                    return "", gr.update(visible=False), 0

                timer = gr.Timer(5.0, active=False)
                auto_refresh.change(
                    fn=lambda x: gr.update(active=x),
                    inputs=[auto_refresh],
                    outputs=[timer]
                )
                timer.tick(
                    fn=poll_progress,
                    outputs=[progress_status, progress_display, gr.State(), upload_status]
                )

            # ===== ENHANCED DOCUMENTS TAB =====
            with gr.Tab("📚 Documents"):
                with gr.Row():
                    # Left Panel - Enhanced Search & Library (40% width)
                    with gr.Column(scale=40):
                        refresh_docs_btn = gr.Button("🔄 Refresh Document Library")

                        # Enhanced Search Results Accordion
                        with gr.Accordion("🔍 Enhanced Document Search", open=True):
                            search_query = gr.Textbox(
                                label="Search Query",
                                placeholder="Search filenames and content...",
                                lines=1
                            )

                            with gr.Row():
                                group_filter = gr.Dropdown(
                                    label="Group",
                                    choices=["All Groups"],
                                    value="All Groups",
                                    scale=1
                                )
                                type_filter = gr.Dropdown(
                                    label="Type",
                                    choices=["All Types"],
                                    value="All Types",
                                    scale=1
                                )

                            edition_filter = gr.Dropdown(
                                label="Edition",
                                choices=["All Editions"],
                                value="All Editions"
                            )

                            search_btn = gr.Button("🔍 Search", variant="primary")

                            # Search results with pagination
                            search_summary = gr.Markdown(
                                value="Enter a search query to find documents"
                            )

                            # Pagination controls
                            with gr.Row(visible=False) as pagination_row:
                                prev_btn = gr.Button("◀ Previous", size="sm", scale=1)
                                page_info = gr.Markdown("Page 1 of 1")
                                next_btn = gr.Button("Next ▶", size="sm", scale=1)

                            file_selector = gr.Radio(
                                label="Select File",
                                choices=[],
                                visible=False,
                                interactive=True
                            )

                        # Document Library Accordion
                        with gr.Accordion("📚 Document Library", open=False):
                            library_summary = gr.Markdown(value="Loading...")
                            doc_groups_state = gr.State({})

                            # Group selector
                            group_selector = gr.Radio(
                                label="Select Document Group",
                                choices=[],
                                interactive=True
                            )

                            # File selector for selected group
                            library_file_selector = gr.Radio(
                                label="Select File",
                                choices=[],
                                visible=False,
                                interactive=True
                            )

                        # Enhanced Statistics Accordion
                        with gr.Accordion("📊 Enhanced Statistics", open=False):
                            stats_display = gr.Markdown(label="System Stats")
                            stats_table = gr.Dataframe(
                                label="Detailed Metrics",
                                headers=["Category", "Count"],
                                datatype=["str", "number"]
                            )

                    # Right Panel - Enhanced Document Viewer (60% width)
                    with gr.Column(scale=60):
                        with gr.Group():
                            gr.Markdown("### 📖 Enhanced Document Viewer")
                            document_content = gr.Markdown(
                                value="🔍 Search for documents or browse the library to view content with metadata",
                                label="Content",
                                show_label=False
                            )

                # Wire up enhanced functionality
                refresh_docs_btn.click(
                    fn=refresh_documents,
                    outputs=[
                        library_summary, doc_groups_state, stats_table, stats_display,
                        group_filter, type_filter, edition_filter, group_selector
                    ]
                )

                search_btn.click(
                    fn=search_docs_fn,
                    inputs=[search_query, group_filter, type_filter, edition_filter],
                    outputs=[search_summary, file_selector, document_content, pagination_row, page_info]
                )

                # Pagination navigation
                prev_btn.click(
                    fn=lambda: navigate_search_results(-1),
                    outputs=[search_summary, file_selector, document_content, pagination_row, page_info]
                )

                next_btn.click(
                    fn=lambda: navigate_search_results(1),
                    outputs=[search_summary, file_selector, document_content, pagination_row, page_info]
                )

                file_selector.change(
                    fn=handle_file_selection,
                    inputs=[file_selector],
                    outputs=[document_content]
                )

                # Library group selection
                group_selector.change(
                    fn=load_document_group,
                    inputs=[group_selector, doc_groups_state],
                    outputs=[library_file_selector]
                )

                # Library file selection
                library_file_selector.change(
                    fn=handle_library_file_selection,
                    inputs=[library_file_selector, group_selector, doc_groups_state],
                    outputs=[document_content]
                )

                # Enhanced auto-load on startup
                app.load(
                    fn=refresh_documents,
                    outputs=[
                        library_summary, doc_groups_state, stats_table, stats_display,
                        group_filter, type_filter, edition_filter, group_selector
                    ]
                )

            # ===== CHARACTER MANAGEMENT TAB =====
            char_components = build_character_management_tab()
            wire_character_management_events(char_components)

            # ===== SESSION NOTES TAB =====
            with gr.Tab("📝 Session Notes"):
                gr.Markdown("""
                ### 🔧 Session Notes - Enhanced Features Coming Soon!

                **Enhanced features will include:**
                - **Campaign Management**: Upload session notes as "adventure" type documents
                - **NPC Tracking**: Automatic extraction of character names and relationships  
                - **Plot Thread Analysis**: AI-powered detection of ongoing story elements
                - **Timeline Integration**: Chronological organization of campaign events
                - **Cross-Reference Search**: Find connections between sessions and rulebook content

                **For now:**
                - Upload session notes as PDFs using **"adventure"** document type in Upload tab
                - Search and query them through the main Query interface
                - They'll be processed with adventure-optimized extraction
                """)

        # Enhanced Footer
        gr.Markdown("---")
        gr.Markdown(
            """
            <center>
            <small>🎲 Shadowrun RAG Assistant v4.0 | Enhanced with Document Types, Think Tags & SR5 Defaults</small><br>
            <small>Powered by Gradio, Ollama & ChromaDB | Character Role Precedence Active</small>
            </center>
            """,
            elem_classes=["footer"]
        )

        # Enhanced Custom CSS with better document viewer
        gr.HTML("""
        <style>
        .prose {
            padding-left: 1rem !important;
        }
        .label {
            padding-left: 1rem !important;
        }
        .group { 
            padding: 1rem !important; 
            margin: 0.5rem 0 !important; 
            border-radius: 8px !important;
        }
        
        /* Enhanced document viewer - remove nested scrolling */
        .gradio-markdown { 
            padding: 0.75rem !important;
            overflow: visible !important;
            height: auto !important;
            max-height: none !important;
        }
        
        /* Improve container heights for document viewer */
        .gradio-column:has(.gradio-markdown) {
            min-height: 70vh !important;
        }
        
        .gradio-dataframe { 
            margin: 0.5rem 0 !important; 
        }
        .gradio-textbox, .gradio-dropdown { 
            margin-bottom: 0.5rem !important; 
        }

        /* Enhanced thinking indicator */
        .thinking-active {
            border-left: 4px solid #ff6b6b !important;
            background: rgba(255, 107, 107, 0.1) !important;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { border-left-color: #ff6b6b; }
            50% { border-left-color: #ffd93d; }
            100% { border-left-color: #ff6b6b; }
        }

        /* Document type indicators */
        .doc-type-rulebook { border-left: 3px solid #4CAF50; }
        .doc-type-character { border-left: 3px solid #2196F3; }
        .doc-type-universe { border-left: 3px solid #FF9800; }
        .doc-type-adventure { border-left: 3px solid #9C27B0; }
        
        /* Pagination styling */
        .pagination-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """)
    return app


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    print("🚀 Starting Enhanced Gradio frontend for Shadowrun RAG...")
    print(f"📡 Connecting to API at: {API_URL}")

    # Check API connection with enhanced info
    try:
        status = client.get_status()
        stats = client.get_document_stats()

        print(f"✅ API Status: {status.get('status', 'unknown')}")
        print(f"📚 Indexed documents: {status.get('indexed_documents', 0)}")
        print(f"📊 Indexed chunks: {status.get('indexed_chunks', 0)}")

        if not isinstance(stats, dict) or "error" not in stats:
            doc_types = stats.get('document_types', {})
            editions = stats.get('editions', {})
            print(f"📋 Document types: {list(doc_types.keys())}")
            print(f"🎲 Editions found: {list(editions.keys())}")

    except Exception as e:
        print(f"⚠️ Warning: Could not connect to API: {e}")
        print("Make sure the enhanced FastAPI backend is running!")

    # Build and launch enhanced interface
    app = build_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )