#!/bin/bash

# Shadowrun RAG Docker Setup Script
# Works on Linux, macOS, and Windows (Git Bash / WSL)

set -euo pipefail  # Exit on error, undefined vars, pipe failure

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎲 Setting up Shadowrun RAG with Docker...${NC}"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    echo -e "${RED}❌ Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

echo -e "${GREEN}Detected OS: $OS${NC}"

# 1. Pull and prepare Ollama models
echo -e "${BLUE}📦 Preparing Ollama models...${NC}"

echo "Starting Ollama container..."
docker-compose up -d ollama
sleep 5

echo "Pulling models..."
docker exec shadowrun-ollama ollama pull mixtral:8x7b-instruct-v0.1-q4_K_M || {
    echo -e "${RED}❌ Failed to pull Mixtral model. Is Ollama running?${NC}"
    exit 1
}

docker exec shadowrun-ollama ollama pull nomic-embed-text || {
    echo -e "${YELLOW}⚠️  nomic-embed-text pull failed — might be built-in${NC}"
}

# 2. Optional: Add local DNS
HOSTS_LINE="127.0.0.1 shadowrun.local"
HOSTS_FILE=""

if [[ "$OS" == "linux" || "$OS" == "macos" ]]; then
    HOSTS_FILE="/etc/hosts"
elif [[ "$OS" == "windows" ]]; then
    HOSTS_FILE="C:/Windows/System32/drivers/etc/hosts"
fi

if [[ -n "$HOSTS_FILE" ]]; then
    echo -e "${BLUE}🌐 Setting up local DNS (shadowrun.local)...${NC}"
    
    if grep -q "shadowrun.local" "$HOSTS_FILE" 2>/dev/null; then
        echo "✅ shadowrun.local already in hosts file"
    else
        if [[ "$OS" == "linux" || "$OS" == "macos" ]]; then
            if sudo sh -c "echo '$HOSTS_LINE' >> $HOSTS_FILE"; then
                echo "✅ Added shadowrun.local to /etc/hosts"
            else
                echo -e "${YELLOW}⚠️  Could not modify /etc/hosts. Add this line manually as admin:${NC}"
                echo "    $HOSTS_LINE"
            fi
        elif [[ "$OS" == "windows" ]]; then
            echo -e "${YELLOW}⚠️  To enable shadowrun.local, add this line to C:\\Windows\\System32\\drivers\\etc\\hosts as Administrator:${NC}"
            echo "    $HOSTS_LINE"
            echo "   You can use Notepad as Admin to edit it."
            echo "   Or just use http://localhost:8501"
        fi
    fi
fi

# 3. Create necessary directories
echo -e "${BLUE}📁 Creating data directories...${NC}"

mkdir -p data/raw_pdfs
mkdir -p data/processed_markdown
mkdir -p data/chroma_db
mkdir -p ssl

# 4. Start all services
echo -e "${BLUE}🚀 Starting all services...${NC}"
docker-compose up -d

# Final message
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo -e "${BLUE}Access your Shadowrun RAG system at:${NC}"
echo "  - Web UI: http://shadowrun.local or http://localhost:8501"
echo "  - API: http://shadowrun.local/api or http://localhost:8000"
echo ""
echo "💡 Place your PDF rulebooks in: ./data/raw_pdfs/"
echo ""
echo -e "${YELLOW}Note: On Windows, you may need to run Git Bash as Administrator to edit hosts file.${NC}"
echo "   If you don’t, just use localhost instead of shadowrun.local"