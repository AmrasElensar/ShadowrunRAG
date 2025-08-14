# Shadowrun RAG Docker Setup Script (PowerShell 7)
# For Windows 10/11 with Docker Desktop
# Requires PowerShell 7+ and Administrator privileges

# Auto-elevate to Administrator using PowerShell 7 (pwsh.exe)
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(`
    [Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "⚠️  This script requires Administrator privileges to modify the hosts file."
    Write-Host "🔄 Restarting as Administrator using PowerShell 7..."

    # Ensure we use pwsh.exe (PowerShell 7), not powershell.exe (PS 5.1)
    if (-not (Get-Command pwsh -ErrorAction SilentlyContinue)) {
        Write-Host "❌ PowerShell 7 is not installed. Please install it from: https://aka.ms/powershell"
        Write-Host "👉 This script requires PowerShell 7 to avoid parsing errors during elevation."
        Write-Host "   Press Enter to exit..." ; Read-Host
        exit 1
    }

    # Build arguments: use -NoExit to keep window open after script finishes
    $Arguments = "-NoExit -ExecutionPolicy Bypass -File `"$PSCommandPath`""

    # Start elevated PowerShell 7 session
    try {
        Start-Process pwsh.exe -ArgumentList $Arguments -Verb RunAs
    } catch {
        Write-Host "❌ Failed to elevate: $($_.Exception.Message)"
        Write-Host "   Please run this script manually as Administrator in PowerShell 7."
        Write-Host "   Press Enter to exit..." ; Read-Host
        exit 1
    }

    # Exit current non-elevated session
    exit
}

# Now running as Admin in PowerShell 7
Write-Host "🔐 Running as Administrator in PowerShell 7"

# Set working directory to script location
$ScriptDir = Split-Path $PSCommandPath -Parent
Set-Location $ScriptDir
Write-Host "📁 Script running in: $ScriptDir"

$ErrorActionPreference = "Stop"

# Colors (ANSI escape codes - work in modern terminals)
$Green  = "$([char]27)[32m"
$Yellow = "$([char]27)[33m"
$Blue   = "$([char]27)[34m"
$Red    = "$([char]27)[31m"
$NC     = "$([char]27)[0m"  # No Color

Write-Host "${Blue}🎲 Setting up Shadowrun RAG with Docker...${NC}"

# Check Docker
Write-Host "${Blue}🔍 Checking Docker...${NC}"
try {
    docker --version | Out-Null
    Write-Host "${Green}✅ Docker is installed${NC}"
} catch {
    Write-Host "${Red}❌ Docker not found. Please install Docker Desktop: https://www.docker.com/products/docker-desktop${NC}"
    exit 1
}

try {
    docker-compose --version | Out-Null
    Write-Host "${Green}✅ Docker Compose is installed${NC}"
} catch {
    Write-Host "${Red}❌ Docker Compose not found. Make sure Docker Desktop includes Compose.${NC}"
    exit 1
}

# 1. Start Ollama and pull models
Write-Host "${Blue}📦 Preparing Ollama models...${NC}"

Write-Host "Starting Ollama container..."
docker-compose up -d ollama

# Wait for container to be running
Write-Host "${Blue}⏳ Waiting for Ollama container to start...${NC}"
while ($true) {
    $ContainerStatus = docker inspect shadowrun-ollama --format '{{.State.Running}}' 2>$null
    if ($ContainerStatus -eq 'true') { break }
    Start-Sleep -Seconds 2
}
Write-Host "${Green}✅ Ollama container is running${NC}"

Write-Host "Pulling Mixtral 8x7B model (this may take a few minutes)..."
try {
    docker exec shadowrun-ollama ollama pull mixtral:8x7b-instruct-v0.1-q4_K_M
    Write-Host "${Green}✅ Mixtral model downloaded${NC}"
} catch {
    Write-Host "${Red}❌ Failed to pull Mixtral model. Is Ollama running?${NC}"
    exit 1
}

Write-Host "Pulling nomic-embed-text embedding model..."
try {
    docker exec shadowrun-ollama ollama pull nomic-embed-text
    Write-Host "${Green}✅ nomic-embed-text downloaded${NC}"
} catch {
    Write-Host "${Yellow}⚠️  nomic-embed-text pull failed — might already be available${NC}"
}

# 2. Add shadowrun.local to hosts file
$HostsPath = "C:\Windows\System32\drivers\etc\hosts"
$HostsEntry = "127.0.0.1 shadowrun.local"

Write-Host "${Blue}🌐 Setting up local DNS (shadowrun.local)...${NC}"

if (Get-Content $HostsPath | Select-String -Pattern "shadowrun.local" -Quiet) {
    Write-Host "${Green}✅ shadowrun.local already exists in hosts file${NC}"
} else {
    try {
        Add-Content -Path $HostsPath -Value $HostsEntry -Encoding ASCII
        Write-Host "${Green}✅ Added shadowrun.local to hosts file${NC}"
    } catch {
        Write-Host "${Red}❌ Failed to write to hosts file. This script must run as Administrator.${NC}"
        Write-Host "${Yellow}💡 You can manually add this line to C:\Windows\System32\drivers\etc\hosts as Admin:${NC}"
        Write-Host "   $HostsEntry"
        Write-Host "   Or just use http://localhost:8501"
    }
}

# 3. Create data directories
Write-Host "${Blue}📁 Creating data directories...${NC}"

$Dirs = @(
    "data/raw_pdfs",
    "data/processed_markdown",
    "data/chroma_db",
    "ssl"
)

foreach ($Dir in $Dirs) {
    if (-not (Test-Path $Dir)) {
        New-Item -ItemType Directory -Path $Dir | Out-Null
        Write-Host "${Green}✅ Created: $Dir${NC}"
    } else {
        Write-Host "${Green}✅ Already exists: $Dir${NC}"
    }
}

# 4. Start all services
Write-Host "${Blue}🚀 Starting all services...${NC}"
docker-compose up -d

# Final message
Write-Host ""
Write-Host "${Green}✅ Setup complete!${NC}"
Write-Host ""
Write-Host "${Blue}🎯 Access your Shadowrun RAG system at:${NC}"
Write-Host "  - Web UI: http://shadowrun.local or http://localhost:8501"
Write-Host "  - API: http://shadowrun.local/api or http://localhost:8000"
Write-Host ""
Write-Host '💡 Place your PDF rulebooks in: .\data\raw_pdfs\'
Write-Host ""
Write-Host "${Yellow}ℹ️  If shadowrun.local doesn't work, your browser may block it. Just use localhost.${NC}"
Write-Host ""
Write-Host "🎉 Press Enter to close..." ; Read-Host