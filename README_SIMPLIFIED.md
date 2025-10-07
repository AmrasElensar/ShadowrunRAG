# 🎲 Shadowrun RAG System - Simplified

A streamlined Retrieval-Augmented Generation (RAG) system for querying **Shadowrun 5th Edition** rulebooks using natural language.

## ✨ What's Different (Simplified Version)

This is a **cleaned-up, simplified version** of the original ShadowrunRAG system with:

### ✅ **Kept (Core Features)**
- **PDF Upload & Processing** - TOC-guided extraction only
- **Text Cleaning Tools** - All regex cleaners and header fixers
- **Manual Indexing** - Two-step process for better control
- **Basic Section Filtering** - Matrix, Combat, Magic, Riggers, etc.
- **Statistics Screen** - View what's in your database

### ❌ **Removed (Complexity)**
- Character management system
- Character-based filtering
- Edition filtering (hardcoded to SR5)
- Subsection filtering  
- Document type filtering
- Document browsing interface
- Session notes
- Multiple query types (only "rules" type)
- Vision/LLM extraction methods (TOC-only)

## 🚀 Quick Start (Docker)

### 1. Start the System
```bash
# Start all services (Ollama + Backend + Frontend)
docker-compose up -d

# Or rebuild if you made changes
docker-compose up --build -d
```

### 2. Pull Required Models
```bash
# Pull embedding model
docker exec shadowrun-ollama ollama pull nomic-embed-text

# Pull your preferred LLM (choose one)
docker exec shadowrun-ollama ollama pull qwen2.5:14b-instruct-q6_K
# OR
docker exec shadowrun-ollama ollama pull llama3:8b-instruct-q4_K_M
```

### 3. Access the System
- **Frontend**: http://localhost:8501
- **API**: http://localhost:8000
- **Ollama**: http://localhost:11434

### 4. Stop the System
```bash
docker-compose down
```

## 📝 Usage Workflow

1. **Upload PDFs** - Use the "Upload & Process" tab
2. **Wait for Processing** - Check status until complete  
3. **Index Content** - Use "Indexing" tab to add to database
4. **Query** - Ask questions in the "Query" tab
5. **View Stats** - Check "Statistics" tab for database info

## 🔧 System Architecture

```
Frontend (Gradio) ←→ Backend (FastAPI) ←→ ChromaDB
                                    ↓
                            TOC-guided PDF Processing
                                    ↓
                            Text Cleaning & Chunking
```

## 📊 Current State

- **PDF Processing**: TOC-guided extraction only
- **Indexing**: Manual trigger after processing
- **Retrieval**: Basic semantic search with section filtering
- **Models**: Ollama integration (nomic-embed-text + your choice of LLM)

## 🛠️ Files Changed

### New Files:
- `backend/simple_api.py` - Streamlined API
- `backend/simple_models.py` - Basic data models  
- `backend/simple_retriever.py` - Core retrieval only
- `frontend/simple_app.py` - Clean Gradio interface
- `tools/simple_pdf_processor.py` - TOC-only processor

### Modified Files:
- `docker-compose.yml` - Updated to use simplified components
- `Dockerfile.backend` - Points to simple_api
- `Dockerfile.frontend` - Points to simple_app
- `tools/pdf_processor.py` - Redirects to simplified version

### Updated Docker Services:
- **Backend**: `shadowrun-backend-simple` (port 8000)
- **Frontend**: `shadowrun-frontend-simple` (port 8501)
- **Ollama**: `shadowrun-ollama` (port 11434)

## 🐳 Docker Commands

```bash
# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f ollama

# Restart specific service
docker-compose restart backend
docker-compose restart frontend

# Check service status
docker-compose ps

# Access container shell
docker exec -it shadowrun-backend-simple bash
docker exec -it shadowrun-frontend-simple bash
docker exec -it shadowrun-ollama bash
```

## 🔍 Debugging Retrieval Issues

The main issue you mentioned is **retrieval returning irrelevant data**. With this simplified system, we can now focus on:

1. **Query Processing** - How questions are parsed and embedded
2. **Chunk Quality** - How documents are split and classified  
3. **Search Relevance** - How semantic similarity is calculated
4. **Filter Logic** - How section filtering affects results

The simplified system makes it much easier to debug these core issues without the complexity of character systems, multiple extraction methods, and advanced filtering.

## 🎯 Next Steps (After Cleanup)

1. **Debug Retrieval** - Fix query parsing and relevance issues
2. **Optimize Indexing** - Improve chunking and classification
3. **Enhance Search** - Better semantic matching
4. **Performance** - Speed up processing pipeline

---

**Ready to test and debug the core retrieval functionality in Docker!** 🎲
