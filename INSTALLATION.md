# 🚀 Bignalytics Enhanced RAG System - Installation Guide

A comprehensive guide to set up the enhanced RAG system with ChromaDB, reranking, and FastAPI.

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- 8GB+ RAM (for embedding models)
- Internet connection (for downloading models)

## 🔧 Installation Steps

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd bignalytics-agentic-rag-8
```

### 2. Create Virtual Environment

#### Using `venv` (Recommended)
```bash
# Create virtual environment
python3.11 -m venv bign_env

# Activate virtual environment
# On macOS/Linux:
source bign_env/bin/activate

# On Windows:
bign_env\Scripts\activate
```

#### Using `conda` (Alternative)
```bash
# Create conda environment
conda create -n bign_env python=3.11

# Activate environment
conda activate bign_env
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 4. Environment Configuration

#### Copy Environment Template
```bash
cp .env.example .env
```

#### Configure API Keys in `.env`

Open `.env` file and add your API keys:

```properties
# Google Gemini API (Required)
GOOGLE_API_KEY=your_google_api_key_here

# Cloudflare Workers API (Optional - for cost optimization)
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id_here
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token_here

# LangSmith Tracing (Optional - for debugging)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=your_project_name_here
LANGCHAIN_TRACING_V2=true
```

#### How to Get API Keys:

1. **Google Gemini API**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key to `GOOGLE_API_KEY`

2. **Cloudflare Workers API** (Optional):
   - Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
   - Navigate to "AI" section
   - Get Account ID and create API token


### 5. Initialize Vector Database

The system will automatically create ChromaDB vectors on first run. You can also manually initialize:

```bash
# Test vector creation
python enhanced_vector_creator.py
```

### 6. Run the Application

#### Start FastAPI Server
```bash
python main.py
```

The server will start on `http://localhost:8000`

#### Test the API

```bash
# Test basic endpoint
curl http://localhost:8000/

# Test chat endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Bignalytics?", "conversation_id": null}'
```
