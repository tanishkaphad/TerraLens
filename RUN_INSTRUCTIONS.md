# GeoVerify-X — Run Instructions

## Prerequisites

- **Python 3.12+** ([download](https://www.python.org/downloads/))
- **Git** (already installed)
- **Groq API Key** (get free tier at [console.groq.com](https://console.groq.com))
- **4GB+ RAM** (for ML models)
- **~2GB disk space** (for ML checkpoints)

---

## Quick Start (5 minutes)

### 1. Clone & Navigate
```bash
cd C:\Users\tanee\TerraLens
```

### 2. Create Environment & Install
```bash
python -m venv venv
venv\Scripts\activate
cd pipeline
pip install -r requirements.txt
```

### 3. Configure Groq API
```bash
# Copy example config
copy .env.example .env

# Edit .env in your editor and add:
# LLM_PROVIDER=groq
# GROQ_API_KEY=gsk_your_key_here
# GROQ_MODEL=llama-3.3-70b-versatile
```

### 4. Run Dev Server
```bash
python -m uvicorn terralens.api:app --reload --host 127.0.0.1 --port 8000
```

### 5. Open Browser
```
http://127.0.0.1:8000
```

---

## Detailed Setup & Commands

### Windows Setup

#### Using PowerShell
```powershell
# Navigate to project
cd C:\Users\tanee\TerraLens

# Create virtual environment
python -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
cd pipeline
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Create .env file
Copy-Item .env.example .env

# Edit .env with your Groq API key
notepad .env

# Run server
python -m uvicorn terralens.api:app --reload --host 127.0.0.1 --port 8000
```

#### Using Command Prompt
```cmd
cd C:\Users\tanee\TerraLens

python -m venv venv

venv\Scripts\activate.bat

cd pipeline

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

copy .env.example .env

REM Edit .env with your Groq API key
start .env

python -m uvicorn terralens.api:app --reload --host 127.0.0.1 --port 8000
```

### Linux/Mac Setup

```bash
cd ~/TerraLens

python3 -m venv venv
source venv/bin/activate

cd pipeline
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

cp .env.example .env

# Edit .env with your favorite editor
nano .env  # or: vim .env, code .env

# Run server
python -m uvicorn terralens.api:app --reload --host 127.0.0.1 --port 8000
```

---

## Environment Configuration (.env)

Create `pipeline/.env` with these variables:

```env
# LLM Configuration
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Alternative providers (optional, uncomment to use)
# GEMINI_API_KEY=
# GEMINI_MODEL=gemini-2.0-flash
# OPENAI_API_KEY=
# OPENAI_MODEL=gpt-4o-mini
# XAI_API_KEY=
# XAI_MODEL=grok-3

# Conversation/Chat Settings
JOBS_DB=data/conversations.sqlite3
CHAT_REQUESTS_PER_HOUR=20
CHAT_REQUESTS_PER_DAY=100

# Place & Vector Data Providers
PHOTON_URL=https://photon.komoot.io/api/
OVERPASS_URL=https://overpass-api.de/api/interpreter
WORLD_CACHE_DIR=data/world-cache

# Durable Search Backend
SEARCH_DB=data/search.sqlite3
SEARCH_WORKERS=4
SEARCH_MAX_PENDING=32
SEARCH_VECTOR_CHUNK_SIZE=1000
SEARCH_PARSER_SAMPLE_VALUES=20

# Enable natural-language search plan parsing
SEARCH_PARSER_ENABLED=0
# SEARCH_PARSER_MODEL=llama-3.1-8b-instant
# SEARCH_PARSER_BASE_URL=https://api.groq.com/openai/v1
# SEARCH_PARSER_API_KEY=
```

**Get Groq API Key**:
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free tier available)
3. Create API key
4. Copy into `GROQ_API_KEY` in `.env`

---

## Running the Application

### Development Server (with auto-reload)

```bash
cd pipeline
python -m uvicorn terralens.api:app --reload --host 127.0.0.1 --port 8000
```

**Options**:
- `--reload`: Auto-restart on file changes
- `--host 127.0.0.1`: Listen on localhost only
- `--port 8000`: Use port 8000 (change if needed)

**Output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Press CTRL+C to quit
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Production Server (no auto-reload)

```bash
cd pipeline
python -m uvicorn terralens.api:app --host 0.0.0.0 --port 8000 --workers 4
```

**Options**:
- `--workers 4`: Use 4 worker processes
- `--host 0.0.0.0`: Listen on all interfaces

### Using Claude Code CLI

```bash
# From project root directory
python -m preview_start "GeoVerify-X Dev Server"

# Or manually start with the launch config
claude --help  # See available preview commands
```

---

## Accessing the Application

### Web UI
- **URL**: `http://127.0.0.1:8000`
- **Search Mode**: Default geographic search with map
- **Conversation Mode**: Toggle button to chat with Groq LLM

### API Endpoints

#### Check Status
```bash
curl http://127.0.0.1:8000/api/assistant/status
```

Response:
```json
{
  "configured": true,
  "provider": "groq",
  "model": "llama-3.3-70b-versatile",
  "tools": [...],
  "privacy_note": "Groq hosts open-source models...",
  "reason": null
}
```

#### Health Check
```bash
curl http://127.0.0.1:8000/health
```

#### Get Datasets
```bash
curl http://127.0.0.1:8000/api/datasets
```

#### Get Capabilities
```bash
curl "http://127.0.0.1:8000/api/capabilities?dataset_id=spacenet-s2-multispectral"
```

#### Search (Legacy)
```bash
curl -X POST http://127.0.0.1:8000/api/discover \
  -H "Content-Type: application/json" \
  -d '{"query": "Find buildings within 300 metres of a river in Pune", "dataset_id": "spacenet-s2-multispectral"}'
```

#### Chat (Conversation)
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What imagery is available in Pune?", "dataset_id": "spacenet-s2-multispectral"}'
```

---

## Troubleshooting

### Port Already in Use
```bash
# Change port
python -m uvicorn terralens.api:app --reload --port 8001

# Or kill existing process (Windows PowerShell)
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
```

### Module Not Found
```bash
# Verify virtual environment is activated
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Groq API Key Issues
```bash
# Test Groq key
python -c "
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv('GROQ_API_KEY'), base_url='https://api.groq.com/openai/v1')
print('✓ Groq API key is valid')
"
```

### Memory Issues (ML Models)
- Reduce workers: `--workers 1`
- Allocate more RAM to Python
- Disable certain features in `.env`

### File Not Found Errors
```bash
# Ensure you're in the pipeline directory
cd C:\Users\tanee\TerraLens\pipeline

# Or use absolute paths
python -m uvicorn terralens.api:app --app-dir "C:\Users\tanee\TerraLens\pipeline"
```

---

## Docker Deployment

### Build Image
```bash
docker build -t geoverify-x:latest .
```

### Run Container
```bash
docker run -p 8000:10000 \
  -e GROQ_API_KEY=gsk_your_key_here \
  -e LLM_PROVIDER=groq \
  -e GROQ_MODEL=llama-3.3-70b-versatile \
  -v geoverify-evidence:/app/pipeline/data \
  geoverify-x:latest
```

### With Compose
```bash
# Create docker-compose.yml
docker compose up -d

# View logs
docker compose logs -f
```

---

## Running Tests

### Unit Tests
```bash
cd pipeline
python -m pytest tests/ -v
```

### Specific Test File
```bash
python -m pytest tests/test_planner.py -v
```

### With Coverage
```bash
python -m pytest tests/ --cov=terralens --cov-report=html
```

---

## Development Workflow

### 1. Activate Environment
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Make Changes
- Edit files in `pipeline/terralens/`
- UI files in `pipeline/terralens/static/`

### 3. Server Auto-Reloads
Dev server automatically reloads on file changes

### 4. Test Changes
- Open http://127.0.0.1:8000 in browser
- Try searches and conversation mode
- Check browser console for errors

### 5. Commit Changes
```bash
git add .
git commit -m "Your description"
git push origin main
```

---

## Performance Tuning

### For Faster Response Times
```bash
# Use more workers
python -m uvicorn terralens.api:app --workers 4

# Increase cache
export TRANSFORMERS_CACHE=/path/to/cache
export HF_HOME=/path/to/cache
```

### For Lower Memory Usage
```bash
# Single worker
python -m uvicorn terralens.api:app --workers 1

# Set environment
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
```

### For GPU Acceleration (if available)
```bash
# Update .env
GROUNDING_DEVICE=cuda
ENCODER_DEVICE=cuda

# Reinstall torch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Monitoring & Debugging

### Enable Debug Logging
```bash
# Set environment variable
export LOGLEVEL=DEBUG
python -m uvicorn terralens.api:app --log-level debug
```

### Monitor API Calls
```bash
# Watch requests in real-time
curl -v http://127.0.0.1:8000/health
```

### Check Database
```bash
# View conversation history
sqlite3 pipeline/data/conversations.sqlite3 "SELECT * FROM conversations LIMIT 5;"

# View search database
sqlite3 pipeline/data/search.sqlite3 "SELECT * FROM search_runs LIMIT 5;"
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start dev server | `cd pipeline && python -m uvicorn terralens.api:app --reload --host 127.0.0.1 --port 8000` |
| Activate venv | `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac) |
| Install deps | `pip install -r requirements.txt` |
| Run tests | `python -m pytest tests/ -v` |
| Check Groq status | `curl http://127.0.0.1:8000/api/assistant/status` |
| Build Docker | `docker build -t geoverify-x:latest .` |
| View logs | `docker compose logs -f` |

---

## Getting Help

- **Groq API Issues**: Visit [console.groq.com](https://console.groq.com)
- **FastAPI Docs**: http://127.0.0.1:8000/docs (interactive API explorer)
- **Code Issues**: Check git logs with `git log --oneline -10`

---

**Last Updated**: 2026-10-09  
**Version**: GeoVerify-X / 02
