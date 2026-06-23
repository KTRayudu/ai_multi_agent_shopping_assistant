# End-to-End AI Engineering Bootcamp — Complete Code Explanation

> **Confirmation**: YES, the code in `ai-product-assistant-main` is DIRECTLY related to the bootcamp readme. This repository IS the capstone project built step-by-step across all 8 weeks of the bootcamp. Every video in the readme maps to a part of this codebase. The code evolves from a simple chatbot → RAG pipeline → Agentic RAG → Multi-Agent system → production-ready AI product.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Project Architecture — The Big Picture](#2-project-architecture--the-big-picture)
3. [How Every File Is Structured](#3-how-every-file-is-structured)
4. [Part 1 — Dev Setup, Chatbot UI, FastAPI Backend (Videos 1–4)](#part-1--dev-setup-chatbot-ui-fastapi-backend-videos-14)
5. [Part 2 — RAG Pipeline, Qdrant Vector DB (Videos 8–13)](#part-2--rag-pipeline-qdrant-vector-db-videos-813)
6. [Part 3 — Advanced RAG: Structured Outputs, Hybrid Search, Reranking (Videos 17–23)](#part-3--advanced-rag-structured-outputs-hybrid-search-reranking-videos-1723)
7. [Part 4 — LangGraph Agents: Nodes, Edges, ReAct Pattern (Videos 26–31)](#part-4--langgraph-agents-nodes-edges-react-pattern-videos-2631)
8. [Part 5 — Multi-Turn, Tools, MCP, Streaming (Videos 35–40)](#part-5--multi-turn-tools-mcp-streaming-videos-3540)
9. [Part 6 — Multi-Agent System: Coordinator + 3 Workers (Videos 45–52)](#part-6--multi-agent-system-coordinator--3-workers-videos-4552)
10. [Part 7 — LiteLLM, ADK, A2A Protocol, Prompt Caching (Videos 55–62)](#part-7--litellm-adk-a2a-protocol-prompt-caching-videos-5562)
11. [Part 8 — HITL, CI Eval Pipeline, Cloud Deployment (Videos 66–73)](#part-8--hitl-ci-eval-pipeline-cloud-deployment-videos-6673)
12. [Complete Code File-by-File Explanation](#complete-code-file-by-file-explanation)
13. [How to Run the Project End-to-End](#how-to-run-the-project-end-to-end)
14. [What Happens When You Run It — Expected Results](#what-happens-when-you-run-it--expected-results)
15. [Different Ways of Doing Each Process](#different-ways-of-doing-each-process)
16. [Differences Between Each Part / Step](#differences-between-each-part--step)
17. [Comparison Table — Evolution of the System](#comparison-table--evolution-of-the-system)
18. [Cheat Sheet — Quick Reference](#cheat-sheet--quick-reference)
19. [Summary](#summary)
20. [Conclusion](#conclusion)

---

## 1. What Is This Project?

This is an **AI-powered Amazon product assistant** — an e-commerce chatbot that helps users:
- Search and ask questions about products (electronics data from Amazon dataset)
- View product images and prices in a sidebar
- Add and remove items from a shopping cart
- Check warehouse inventory availability
- Reserve items from warehouses

It is built in an **8-week bootcamp** and evolves through multiple stages:

```
Week 1: Simple chatbot (OpenAI/Groq/Gemini) with Streamlit UI
Week 2: RAG pipeline on Qdrant vector DB
Week 3: Advanced RAG (structured outputs, hybrid search, reranking)
Week 4: LangGraph agents (nodes, edges, ReAct pattern)
Week 5: Multi-turn, MCP servers, streaming
Week 6: Multi-agent system (3 worker agents + 1 coordinator)
Week 7: LiteLLM, ADK, A2A, prompt caching
Week 8: HITL, CI pipeline, cloud deployment
```

**Technology stack:**
| Category | Tool |
|---|---|
| LLM Providers | OpenAI, Groq (Llama), Google Gemini |
| Vector DB | Qdrant |
| Relational DB | PostgreSQL |
| Agent Orchestration | LangGraph |
| Model Routing | LiteLLM |
| Observability | LangSmith |
| RAG Evals | Ragas framework |
| Structured Outputs | Instructor library |
| Alt. Orchestration | Google ADK |
| Agent Protocol | A2A (Agent-to-Agent) |
| Tool Protocol | MCP (Model Context Protocol) |
| Frontend | Streamlit |
| Backend | FastAPI |
| Containerization | Docker Compose |

---

## 2. Project Architecture — The Big Picture

```
User Browser
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Streamlit Chatbot UI  (port 8501)                          │
│  apps/chatbot_ui/src/chatbot_ui/app.py                      │
│                                                             │
│  Sidebar: Product Suggestions | Shopping Cart               │
│  Chat: User messages → streaming response                   │
│  Feedback: Thumbs up/down → LangSmith                       │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP POST /agent (SSE stream)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI Backend  (port 8000)                               │
│  apps/api/src/api/app.py                                    │
│                                                             │
│  Endpoints:                                                 │
│    POST /agent          → StreamingResponse (SSE)           │
│    POST /submit_feedback → LangSmith feedback               │
└──────────────────────────┬──────────────────────────────────┘
                           │ calls
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  LangGraph Multi-Agent Graph                                │
│  apps/api/src/api/agents/graph.py                           │
│                                                             │
│  ┌────────────────┐                                         │
│  │  Coordinator   │  Plans, delegates, synthesizes answer   │
│  │  Agent         │                                         │
│  └───────┬────────┘                                         │
│          │ delegates to one of three workers                │
│    ┌─────┴─────┬──────────────────────────────┐            │
│    ▼           ▼                              ▼            │
│  Product QA  Shopping Cart   Warehouse Manager             │
│  Agent       Agent           Agent                         │
│    │           │                              │            │
│    │           │                              │            │
│    ▼           ▼                              ▼            │
│  Qdrant     PostgreSQL       PostgreSQL                     │
│  (vector)   (shopping_carts) (warehouses)                  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────┐   ┌───────────────────────────────┐
│  Qdrant Vector DB     │   │  MCP Servers (optional)        │
│  port 6333            │   │  items_mcp_server  port 8001   │
│                       │   │  reviews_mcp_server port 8002  │
│  Collections:         │   └───────────────────────────────┘
│  - hybrid-search      │
│  - reviews            │   ┌───────────────────────────────┐
└───────────────────────┘   │  ADK Agent (optional)          │
                             │  adk_warehouse_manager_agent   │
┌───────────────────────┐   └───────────────────────────────┘
│  PostgreSQL DB        │
│  port 5433 (external) │   ┌───────────────────────────────┐
│  port 5432 (internal) │   │  A2A Server (optional)         │
│                       │   │  a2a_warehouse_manager_agent   │
│  Schemas:             │   └───────────────────────────────┘
│  - shopping_carts     │
│  - warehouses         │
│  - langgraph_db       │
└───────────────────────┘
```

**Data flow for a user question:**
```
User types: "Show me Bluetooth headphones with good battery life and add the best one to cart"
    │
    ▼ (SSE stream to frontend)
Coordinator Agent analyzes → creates plan:
    Plan: [
        {agent: "product_qa_agent", task: "Find bluetooth headphones with good battery"},
        {agent: "shopping_cart_agent", task: "Add best result to cart"}
    ]
    next_agent: "product_qa_agent"
    │
    ▼
Product QA Agent → calls tool: get_formatted_items_context("bluetooth headphones battery")
    │
    ▼
Tool → Qdrant hybrid search (BM25 + vector + RRF fusion) → top 5 products
    │
    ▼
Product QA Agent generates answer with product details + references
    │
    ▼
Returns to Coordinator → next_agent: "shopping_cart_agent"
    │
    ▼
Shopping Cart Agent → calls tool: add_to_shopping_cart([{product_id, quantity}])
    │
    ▼
Tool → PostgreSQL INSERT into shopping_carts.shopping_cart_items
    │
    ▼
Shopping Cart Agent confirms action
    │
    ▼
Coordinator → final_answer=True, synthesizes complete response
    │
    ▼
Frontend: Shows answer + product images in sidebar + cart updated
```

---

## 3. How Every File Is Structured

```
ai-product-assistant-main/
│
├── docker-compose.yml              # runs all services together
├── Makefile                        # shortcuts for common commands
├── pyproject.toml                  # root UV workspace config
├── uv.lock                         # exact package versions (lockfile)
├── .env.example                    # template for environment variables
│
├── apps/
│   ├── chatbot_ui/                 # Streamlit frontend
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── src/chatbot_ui/
│   │       ├── app.py              # THE frontend app (entire UI logic)
│   │       └── core/config.py     # reads API_URL from .env
│   │
│   ├── api/                        # FastAPI backend (core of the system)
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── evals/
│   │   │   └── eval_retriever.py  # RAG evaluation script
│   │   └── src/api/
│   │       ├── app.py             # FastAPI app setup + middleware
│   │       ├── api/
│   │       │   ├── endpoints.py   # /agent and /submit_feedback routes
│   │       │   ├── middleware.py  # adds request ID to every request
│   │       │   ├── models.py      # Pydantic request/response schemas
│   │       │   └── processors/
│   │       │       └── submit_feedback.py # posts feedback to LangSmith
│   │       ├── agents/
│   │       │   ├── agents.py      # 4 agent functions (coordinator + 3 workers)
│   │       │   ├── graph.py       # LangGraph state machine definition
│   │       │   ├── retrieval_generation.py # original RAG pipeline
│   │       │   ├── tools.py       # all tool functions (Qdrant + PostgreSQL)
│   │       │   ├── prompts/
│   │       │   │   ├── coordinator_agent.yaml
│   │       │   │   ├── product_qa_agent.yaml
│   │       │   │   ├── shopping_cart_agent.yaml
│   │       │   │   ├── warehouse_manager_agent.yaml
│   │       │   │   ├── retrieval_generation.yaml
│   │       │   │   ├── qa_agent.yaml
│   │       │   │   └── intent_router_agent.yaml
│   │       │   └── utils/
│   │       │       ├── prompt_management.py  # YAML+Jinja2 prompt loading
│   │       │       └── utils.py              # tool description parser
│   │       └── core/
│   │           └── config.py      # Pydantic settings for API keys
│   │
│   ├── items_mcp_server/           # MCP server for product search
│   │   ├── Dockerfile
│   │   └── src/items_mcp_server/
│   │       ├── main.py            # FastMCP tool definition
│   │       └── utils.py           # retrieval helper functions
│   │
│   ├── reviews_mcp_server/         # MCP server for product reviews
│   │   └── src/reviews_mcp_server/
│   │       ├── main.py            # FastMCP tool definition
│   │       └── utils.py
│   │
│   ├── adk_warehouse_manager_agent/ # Google ADK alternative implementation
│   │   └── warehouse_manager_agent/
│   │       ├── agent.py           # Google ADK Agent definition
│   │       └── tools.py           # same warehouse tools
│   │
│   └── a2a_warehouse_manager_agent/ # A2A protocol implementation
│       └── warehouse_manager_agent/
│           ├── agent.py           # LangGraph agent for A2A
│           ├── agent_executor.py  # A2A executor wrapping the agent
│           ├── app.py             # FastAPI app exposing A2A endpoints
│           └── tools.py
│
├── notebooks/
│   ├── prerequisites/             # Video 1: first LLM calls
│   ├── week_1/                    # Videos 8–13: RAG + evals
│   ├── week_2/                    # Videos 17–23: Advanced RAG
│   ├── week_3/                    # Videos 26–31: LangGraph intro
│   ├── week_4/                    # Videos 35–40: Multi-turn, MCP, streaming
│   ├── week_5/                    # Videos 45–52: Multi-agent
│   └── week_6/                    # Videos 55–62: LiteLLM, ADK, A2A, caching
│
└── scripts/
    └── sql/
        ├── shopping_cart_table.sql  # DB schema for shopping cart
        └── warehouse_management.sql # DB schema for warehouse inventory
```

---

## Part 1 — Dev Setup, Chatbot UI, FastAPI Backend (Videos 1–4)

### What was built

A minimal chatbot that routes to 3 LLM providers (OpenAI, Groq, Google), deployed with Docker Compose. This establishes the monorepo pattern used for the rest of the bootcamp.

### Video 1 — Dev Setup

**Goal:** Install UV, set up virtual environment, make first LLM API calls.

**Why UV over pip?**
- Written in Rust — 10–100x faster than pip
- Single lockfile for entire workspace (`uv.lock`)
- Built-in virtual environment management

```bash
# Installation
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up project
uv init --python 3.12          # creates pyproject.toml and .venv
uv add openai groq google-generativeai python-dotenv
```

**Environment variables** — never hardcode secrets:
```bash
# .env file (never commit this)
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ai-product-assistant
```

**Why three LLM providers?**
| Provider | Best For | Speed |
|---|---|---|
| OpenAI | Embeddings (RAG), best structured outputs | Medium |
| Groq | Open-source models (Llama), lightning fast | Very fast (0.6s for 20 words) |
| Google Gemini | PDF/document extraction | Medium |

### Video 2 — Streamlit Frontend

**Goal:** Build a chat UI with provider/model selection. Deploy in Docker.

**`apps/chatbot_ui/src/chatbot_ui/app.py`** — What each line does:

```python
import streamlit as st           # web UI framework (Python-only, no HTML/CSS needed)
import requests                  # HTTP client to call FastAPI backend
from chatbot_ui.core.config import config  # reads API_URL from .env
import uuid                      # generates unique session IDs
import json                      # parses SSE stream data

# Sets browser tab title, wide layout, sidebar always open
st.set_page_config(
    page_title="Ecommerce Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_session_id():
    # Creates a UUID once per browser session and saves it in st.session_state
    # This UUID is the thread_id passed to the backend → enables conversation memory
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id
```

**Streamlit session state** — how memory works in Streamlit:
- Streamlit re-runs the ENTIRE Python script top-to-bottom on every user event
- `st.session_state` is a dict that PERSISTS between reruns
- Without session_state, every click would reset the conversation to empty

**Sidebar with two tabs:**
```python
with st.sidebar:
    # Creates two tabs: "Suggestions" shows product images, "Shopping Cart" shows cart
    suggestions_tab, shopping_cart_tab = st.tabs(["🔍 Suggestions", "🛒 Shopping Cart"])
    
    with suggestions_tab:
        # Loops through items returned by backend (product images + prices)
        for item in st.session_state.used_context:
            st.caption(item.get('description'))  # product name
            st.image(item["image_url"], width=250)  # product image from Amazon
            st.caption(f"Price: {item['price']} USD")
```

**Streaming response handling:**
```python
# api_call_stream returns an iterator of lines from SSE (Server-Sent Events)
for line in api_call_stream("post", f"{config.API_URL}/agent", ...):
    line_text = line.decode("utf-8")  # bytes → string
    
    if line_text.startswith("data: "):  # SSE format: every line starts with "data: "
        data = line_text[6:]  # strip "data: " prefix
        
        try:
            output = json.loads(data)  # try to parse as JSON
            
            if output["type"] == "final_result":
                # This is the last SSE message — contains the complete answer
                answer = output["data"]["answer"]
                used_context = output["data"]["used_context"]   # product images
                shopping_cart = output["data"]["shopping_cart"] # cart contents
                
                # Update session state → triggers sidebar to update
                st.session_state.used_context = used_context
                st.session_state.shopping_cart = shopping_cart
                break  # stop reading the stream
        
        except json.JSONDecodeError:
            # Non-JSON lines are status updates like "Planning..." or "Looking for items..."
            status_placeholder.markdown(f"*{data}*")  # shown as italic text while processing
```

**Feedback buttons:**
```python
# st.feedback("thumbs") shows a thumbs-up/thumbs-down widget
feedback_result = st.feedback("thumbs", key=feedback_key)

if feedback_result is not None:
    # feedback_result: 1 = thumbs up, 0 = thumbs down
    feedback_type = "positive" if feedback_result == 1 else "negative"
    status, response = submit_feedback(feedback_type=feedback_type)
    # This POSTs to /submit_feedback → saves to LangSmith for future analysis
```

### Video 3 — FastAPI Backend

**Goal:** Move LLM logic out of frontend into a dedicated API service.

**`apps/api/src/api/app.py`** — FastAPI application setup:

```python
from fastapi import FastAPI
from api.api.middleware import RequestIDMiddleware  # adds X-Request-ID header
from fastapi.middleware.cors import CORSMiddleware  # allows frontend to call backend
from api.api.endpoints import api_router            # routes: /agent, /submit_feedback

app = FastAPI()

# Middleware 1: RequestIDMiddleware — adds unique ID to every request for tracing
app.add_middleware(RequestIDMiddleware)

# Middleware 2: CORSMiddleware — allows any origin to call this API
# In production: replace "*" with specific frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (dev only)
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],   # allow all headers
)

# Register all routes (prefixed: /agent, /submit_feedback)
app.include_router(api_router)
```

**`apps/api/src/api/api/middleware.py`** — What RequestIDMiddleware does:

```python
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())  # generate unique ID for this request
        request.state.request_id = request_id  # store on request object
        
        # Log request start with ID
        logger.info(f"Request started: {request.method} {request.url.path} (request_id: {request_id})")
        
        response = await call_next(request)  # process the actual request
        
        # Add ID to response header so frontend can reference it
        response.headers["X-Request-ID"] = request_id
        logger.info(f"Request completed: ...")
        
        return response
```

**`apps/api/src/api/api/endpoints.py`** — Route definitions:

```python
@rag_router.post("/")  # mapped to /agent via include_router(prefix="/agent")
def rag(request: Request, payload: RAGRequest) -> StreamingResponse:
    # Calls the LangGraph agent and returns an SSE stream
    return StreamingResponse(
        rag_agent_stream_wrapper(payload.query, payload.thread_id),
        media_type="text/event-stream"   # tells browser this is an SSE stream
    )

@feedback_router.post("/")  # mapped to /submit_feedback
def send_feedback(request: Request, payload: FeedbackRequest) -> FeedbackResponse:
    submit_feedback(...)   # saves to LangSmith
    return FeedbackResponse(request_id=request.state.request_id, status="success")
```

**`apps/api/src/api/api/models.py`** — Request/response contracts:

```python
class RAGRequest(BaseModel):
    query: str          # the user's question, e.g. "Find me bluetooth headphones"
    thread_id: str      # UUID from Streamlit session → enables conversation memory

class FeedbackRequest(BaseModel):
    feedback_score: Union[int, None]  # 1 = positive, 0 = negative, None = text only
    feedback_text: str                # optional detailed feedback text
    trace_id: str                     # LangSmith trace ID from previous response
    thread_id: str
    feedback_source_type: str         # "api" or "human"

class FeedbackResponse(BaseModel):
    request_id: str   # X-Request-ID from middleware
    status: str       # "success"
```

### Video 4 — Full Docker Stack

**`docker-compose.yml`** — All services together:

```yaml
services:
  streamlit-app:       # Port 8501 — what users see
    build: apps/chatbot_ui/Dockerfile
    volumes:
      - ./apps/chatbot_ui/src:/app/apps/chatbot_ui/src  # hot reload: change code → instant update

  api:                 # Port 8000 — FastAPI backend
    build: apps/api/Dockerfile
    volumes:
      - ./apps/api/src:/app/apps/api/src                # hot reload

  qdrant:              # Port 6333 — vector database (no custom build, uses official image)
    image: qdrant/qdrant
    volumes:
      - ./qdrant_storage:/qdrant/storage                # data persists between restarts

  postgres:            # Port 5433 (external) — relational DB for cart + warehouse
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: langgraph_db
      POSTGRES_USER: langgraph_user
      POSTGRES_PASSWORD: langgraph_password

  items_mcp_server:    # Port 8001 — MCP server for item search
    build: apps/items_mcp_server/Dockerfile

  reviews_mcp_server:  # Port 8002 — MCP server for reviews
    build: apps/reviews_mcp_server/Dockerfile
```

**What "hot reload" means:** When you edit a source file while Docker is running, the volume mount immediately reflects the change in the container. Streamlit or uvicorn detects the file change and restarts automatically — no need to run `docker compose up --build` again.

---

## Part 2 — RAG Pipeline, Qdrant Vector DB (Videos 8–13)

### What is RAG?

RAG = Retrieval Augmented Generation. Instead of asking an LLM to answer from memory (which leads to hallucinations), we:
1. Store product descriptions as vectors in Qdrant
2. When user asks a question, convert it to a vector too
3. Find the most similar product descriptions
4. Include those descriptions in the prompt
5. LLM generates an answer GROUNDED in real data

```
Without RAG:
User: "What headphones do you have?"
LLM: "We have Sony, Bose, and Apple AirPods..." (hallucinated — may not be true)

With RAG:
User: "What headphones do you have?"
→ Retrieve top 5 matching products from Qdrant
→ Prompt: "Based on these 5 products: [actual data]..."
LLM: "We have Nike Kids Headphones (4.5★, $29.99), Sony MDR..." (grounded in real data)
```

### `apps/api/src/api/agents/retrieval_generation.py` — Full RAG pipeline:

```python
# STEP 1: Convert text to vector (embedding)
@traceable(name="embed_query", run_type="embedding",
           metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"})
def get_embedding(text, model="text-embedding-3-small"):
    # text-embedding-3-small creates a 1536-dimensional vector
    # Every word/phrase is mapped to a point in 1536-dimensional space
    # Similar words → nearby points → similar vectors
    response = openai.embeddings.create(input=text, model=model)
    
    # Log token usage to LangSmith for cost tracking
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    
    return response.data[0].embedding  # list of 1536 floats e.g. [0.023, -0.041, ...]
```

```python
# STEP 2: Hybrid search — combines semantic + keyword search
@traceable(name="retrieve_data", run_type="retriever")
def retrieve_data(query, qdrant_client, k=5):
    query_embedding = get_embedding(query)  # convert query to vector
    
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid-search",
        
        # HYBRID SEARCH: Two different retrieval strategies run in parallel
        prefetch=[
            # Strategy 1: Semantic (vector) search
            # Finds products semantically similar to the query
            # Good for: "comfortable headphones" (understands meaning)
            Prefetch(
                query=query_embedding,        # our 1536-dim query vector
                using="text-embedding-3-small",  # use the embedding index
                limit=20                      # get top 20 semantic matches
            ),
            # Strategy 2: BM25 (keyword) search
            # Finds products with matching keywords
            # Good for: exact brand names, model numbers
            Prefetch(
                query=Document(text=query, model="qdrant/bm25"),
                using="bm25",                 # use the BM25 keyword index
                limit=20                      # get top 20 keyword matches
            )
        ],
        
        # FUSION: RRF (Reciprocal Rank Fusion) combines both result lists
        # A product ranked high in BOTH lists gets a very high combined score
        # This is better than either method alone
        query=FusionQuery(fusion="rrf"),
        
        limit=k,  # final output: top k results after fusion
    )
    
    # Extract data from each result
    retrieved_context_ids = []    # product IDs (Amazon ASIN codes)
    retrieved_context = []        # text descriptions
    similarity_scores = []        # RRF scores (for monitoring)
    retrieved_context_ratings = []  # average star ratings
    
    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])   # e.g. "B07PFFMP9P"
        retrieved_context.append(result.payload["description"])        # product description text
        retrieved_context_ratings.append(result.payload["average_rating"])  # e.g. 4.3
        similarity_scores.append(result.score)                         # e.g. 0.024
    
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }
```

```python
# STEP 3: Format context for LLM
@traceable(name="format_retrieved_context", run_type="prompt")
def process_context(context):
    formatted_context = ""
    
    # Build a formatted string with ID, rating, and description for each product
    # Why include ID? → Agent needs it to cite which product it used
    # Why include rating? → Allows questions like "show items rated above 4.5"
    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"]
    ):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"
    
    return formatted_context
    # Example output:
    # - ID: B07PFFMP9P, rating: 4.5, description: Sony MDR-ZX110 Stereo Headphones...
    # - ID: B08X2WMKHZ, rating: 4.2, description: JBL Tune 510BT Wireless...
```

```python
# STEP 4: Build the prompt
@traceable(name="build_prompt", run_type="prompt")
def build_prompt(preprocessed_context, question):
    # Load prompt from YAML file using Jinja2 templating
    # This allows changing prompts without code changes
    template = prompt_template_config("api/agents/prompts/retrieval_generation.yaml", "retrieval_generation")
    prompt = template.render(preprocessed_context=preprocessed_context, question=question)
    return prompt
```

```python
# STEP 5: Generate structured answer
@traceable(name="generate_answer", run_type="llm",
           metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"})
def generate_answer(prompt):
    # instructor.from_openai wraps OpenAI client to enforce structured output
    # WITHOUT instructor: response is a plain string
    # WITH instructor: response is validated against RAGGenerationResponse Pydantic model
    client = instructor.from_openai(openai.OpenAI())
    
    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
        response_model=RAGGenerationResponse  # MUST match this schema
    )
    # response.answer → the text answer
    # response.references → list of {id, description} for each cited product
    
    return response
```

```python
# Pydantic models that define the structured output schema
class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(description="Short description of the item")

class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    references: list[RAGUsedContext] = Field(description="List of items used")
    # instructor forces the LLM to ALWAYS return exactly this structure
    # If LLM tries to return something else → instructor retries with error feedback
```

### What @traceable Does

```python
@traceable(name="embed_query", run_type="embedding", metadata={...})
def get_embedding(text):
    ...
```

The `@traceable` decorator from LangSmith:
- Wraps the function so every call is automatically logged to LangSmith
- `name`: how the span appears in LangSmith UI
- `run_type`: determines the icon (embedding, retriever, llm, prompt)
- `metadata`: static info attached to every call of this function
- Creates a parent-child tree: `rag_pipeline` is parent → all sub-functions are children

**Without observability:** You have no idea what's happening inside your pipeline.
**With LangSmith:** You see every step, its input/output, token counts, costs, and latency.

### Evaluation with Ragas (Videos 12–13)

```
4 Ragas Metrics:

1. Context Precision  = (retrieved ∩ expected) / retrieved
   Measures: "Are all items we retrieved actually relevant?"
   Low score = retrieving too much noise

2. Context Recall     = (retrieved ∩ expected) / expected
   Measures: "Did we find all the relevant items?"
   Low score = missing important products

3. Response Relevancy = embedding similarity between answer and question
   Measures: "Does the answer actually address the question?"
   High score = answer is on-topic

4. Faithfulness       = LLM judge checks: are answer claims grounded in context?
   Measures: "Is the LLM hallucinating?"
   Low score = LLM is making things up not in the retrieved data
```

**Typical baseline scores (first experiment):**
```
Context Precision:  0.31  ← too low, retrieving irrelevant items
Context Recall:     0.72  ← decent, finding most relevant items
Faithfulness:       0.76  ← good, mostly grounded in context
Response Relevancy: 0.94  ← excellent, answers are on-topic
```

**The improvement loop:**
1. Run evals → find low scores
2. Identify the problem (wrong chunk size? bad embedding? wrong k?)
3. Fix it → run evals again → compare scores

---

## Part 3 — Advanced RAG: Structured Outputs, Hybrid Search, Reranking (Videos 17–23)

### Structured Outputs with Instructor

**Before structured outputs:** LLM returns plain text → manual parsing → fragile
**After structured outputs:** LLM always returns valid Pydantic object → type-safe

```python
# instructor forces structured output
client = instructor.from_openai(openai.OpenAI())

# response_model=RAGGenerationResponse → LLM MUST return this schema
response, raw_response = client.chat.completions.create_with_completion(
    model="gpt-4.1-mini",
    response_model=RAGGenerationResponse,
    messages=[{"role": "system", "content": prompt}],
    temperature=0,
)
# response.answer: str
# response.references: list[RAGUsedContext]
# Guaranteed structure — never a parsing error
```

### Hybrid Search Explained

**Pure vector search problem:** "bluetooth" might not appear in the description text, but a product about "wireless audio" is semantically similar.
**Pure keyword search problem:** Can't understand synonyms or context.
**Hybrid search:** Best of both worlds.

```
Query: "wireless earbuds good for gym"

Vector search finds:          BM25 finds:
1. Sports earphones           1. Gym earbuds (exact keyword)
2. Fitness audio gear         2. Wireless headphones (keyword)
3. Workout headphones         3. Running earphones (keyword)
4. Active noise cancelling    4. Bluetooth gym (keyword)
5. IPX4 waterproof earbuds    5. Exercise earphones (keyword)

RRF Fusion result (best of both):
1. Sports earphones       ← high in both → top score
2. IPX4 waterproof        ← good in vector, decent in BM25
3. Gym earbuds            ← perfect keyword match
4. Fitness audio gear     ← strong semantic match
5. Wireless headphones    ← relevant in both
```

**RRF formula:** Score(d) = Σ 1/(k + rank_in_list_i), where k=60 (constant)
Items ranked #1 in both lists get the highest combined score.

### YAML + Jinja2 Prompt Versioning

**`apps/api/src/api/agents/utils/prompt_management.py`:**

```python
def prompt_template_config(yaml_file, prompt_key):
    # Load the YAML file
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract the template for this specific key
    # e.g. config['prompts']['gpt-4.1'] = "You are a shopping assistant..."
    template_content = config['prompts'][prompt_key]
    
    # Wrap in Jinja2 Template for variable substitution
    template = Template(template_content)
    
    return template

# Usage:
template = prompt_template_config("api/agents/prompts/product_qa_agent.yaml", "gpt-4.1")
prompt = template.render(available_tools=state.product_qa_agent.available_tools)
# Jinja2 replaces {{ available_tools | tojson }} with the actual JSON
```

**Why YAML prompts?**
- Prompts live in version-controlled files (not hardcoded strings)
- Easy A/B testing: change YAML → redeploy → run evals → compare
- Per-model prompts: GPT-4.1 and Llama may need different prompt styles
- Non-engineers can edit prompts without touching Python code

---

## Part 4 — LangGraph Agents: Nodes, Edges, ReAct Pattern (Videos 26–31)

### What is LangGraph?

LangGraph is a framework for building **stateful, cyclical agent graphs**. Unlike a simple pipeline (A → B → C), agents can loop, branch, and call tools multiple times.

```
Simple pipeline (no LangGraph):
Question → Retrieve → Generate → Answer
(Can only go forward, one pass)

LangGraph agent (ReAct loop):
Question → Think → Call Tool → Observe Result → Think → Call Tool → ... → Answer
(Can loop as many times as needed, can branch)
```

### ReAct Pattern (Reasoning + Acting)

```
ReAct = Reason + Act cycle

Iteration 1:
  Reason: "User wants headphones. I should search for them."
  Act: call get_formatted_items_context(query="headphones")
  Observe: "Found 5 products: Sony MDR-ZX110, JBL Tune 510BT..."

Iteration 2:
  Reason: "User also wants reviews. I'll get them for these products."
  Act: call get_formatted_reviews_context(query="headphones", item_list=["B07P...", "B08X..."])
  Observe: "Reviews: Sony MDR has great comfort, JBL has long battery..."

Iteration 3:
  Reason: "I have enough information to answer."
  Act: Set final_answer=True, write answer
  (loop ends)
```

### `apps/api/src/api/agents/graph.py` — LangGraph State Machine

**State class — shared memory of the entire graph:**

```python
class AgentProperties(BaseModel):
    iteration: int = 0             # how many times this agent has run (prevents infinite loops)
    final_answer: bool = False     # has this agent finished?
    available_tools: List[Dict[str, Any]] = []  # what tools can this agent use?
    tool_calls: List[ToolCall] = []  # what tools did the agent just decide to call?

class CoordinatorAgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    plan: List[Delegation] = []   # list of {agent, task} delegations
    next_agent: str = ""          # which agent to call next

class State(BaseModel):
    # Annotated[List[Any], add] means: when merging states, ADD (append) to messages list
    messages: Annotated[List[Any], add] = []       # full conversation history
    user_intent: str = ""                          # classified intent (from earlier version)
    product_qa_agent: AgentProperties              # state of product QA agent
    shopping_cart_agent: AgentProperties           # state of shopping cart agent
    warehouse_manager_agent: AgentProperties       # state of warehouse agent
    coordinator_agent: CoordinatorAgentProperties  # state of coordinator
    answer: str = ""                               # final answer to return
    references: Annotated[List[RAGUsedContext], add] = []  # product references
    user_id: str = ""                              # user identifier
    cart_id: str = ""                              # shopping cart identifier
```

**Conditional edges — the routing logic:**

```python
def product_qa_agent_tool_edge(state) -> str:
    """After product_qa_agent runs, decide what to do next."""
    
    if state.product_qa_agent.final_answer:
        return "end"                    # agent is done → go back to coordinator
    elif state.product_qa_agent.iteration > 4:
        return "end"                    # safety limit: 4 iterations max → prevent infinite loops
    elif len(state.product_qa_agent.tool_calls) > 0:
        return "tools"                  # agent wants to use a tool → call tool node
    else:
        return "end"                    # no tools, no final answer → unexpected → end

def coordinator_agent_edge(state):
    """After coordinator runs, decide which agent to call next."""
    
    if state.coordinator_agent.iteration > 3:
        return "end"                    # max 3 coordination rounds
    elif state.coordinator_agent.final_answer and len(state.coordinator_agent.plan) == 0:
        return "end"                    # coordinator is done with all tasks
    elif state.coordinator_agent.next_agent == "product_qa_agent":
        return "product_qa_agent"       # delegate to product QA agent
    elif state.coordinator_agent.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"    # delegate to shopping cart agent
    elif state.coordinator_agent.next_agent == "warehouse_manager_agent":
        return "warehouse_manager_agent"
    else:
        return "end"
```

**Building the workflow graph:**

```python
workflow = StateGraph(State)   # create graph with our State as the shared data

# Register tool sets for each worker agent
product_qa_agent_tools = [get_formatted_items_context, get_formatted_reviews_context]
shopping_cart_agent_tools = [add_to_shopping_cart, remove_from_cart, get_shopping_cart]
warehouse_manager_agent_tools = [check_warehouse_availability, reserve_warehouse_items]

# ToolNode automatically executes the tool function and returns results to state
product_qa_agent_tool_node = ToolNode(product_qa_agent_tools)

# Add nodes to graph (each "node" is a function that takes State and returns dict)
workflow.add_node("coordinator_agent", coordinator_agent)
workflow.add_node("product_qa_agent", product_qa_agent)
workflow.add_node("shopping_cart_agent", shopping_cart_agent)
workflow.add_node("warehouse_manager_agent", warehouse_manager_agent)
workflow.add_node("product_qa_agent_tool_node", product_qa_agent_tool_node)
workflow.add_node("shopping_cart_agent_tool_node", shopping_cart_agent_tool_node)
workflow.add_node("warehouse_manager_agent_tool_node", warehouse_manager_agent_tool_node)

# Every graph starts at START
workflow.add_edge(START, "coordinator_agent")

# Coordinator decides which worker to call (conditional edge)
workflow.add_conditional_edges(
    "coordinator_agent",
    coordinator_agent_edge,   # this function returns the name of the next node
    {
        "product_qa_agent": "product_qa_agent",
        "shopping_cart_agent": "shopping_cart_agent",
        "warehouse_manager_agent": "warehouse_manager_agent",
        "end": END
    }
)

# Product QA agent loops with its tools, then returns to coordinator
workflow.add_conditional_edges(
    "product_qa_agent",
    product_qa_agent_tool_edge,
    {
        "tools": "product_qa_agent_tool_node",  # call tool → results come back to agent
        "end": "coordinator_agent"              # done → back to coordinator
    }
)
workflow.add_edge("product_qa_agent_tool_node", "product_qa_agent")  # tool results go back to agent

# Compile the graph (validates it, adds checkpointing support)
# PostgresSaver enables MULTI-TURN: saves state between requests using thread_id
graph = workflow.compile(checkpointer=PostgresSaver.from_conn_string("postgresql://..."))
```

---

## Part 5 — Multi-Turn, Tools, MCP, Streaming (Videos 35–40)

### Multi-Turn Conversation with PostgreSQL State

**Problem:** HTTP is stateless. Each API call starts fresh — the agent has no memory of previous messages.
**Solution:** LangGraph's checkpointing saves the graph state to PostgreSQL between requests, keyed by `thread_id`.

```python
with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:
    graph = workflow.compile(checkpointer=checkpointer)
    
    # First message (thread_id = "user-session-abc123")
    graph.invoke({"messages": [{"role": "user", "content": "Show me Sony headphones"}]},
                 config={"configurable": {"thread_id": "user-session-abc123"}})
    # State is saved to PostgreSQL
    
    # Second message (same thread_id → state is LOADED from PostgreSQL)
    graph.invoke({"messages": [{"role": "user", "content": "Add the first one to my cart"}]},
                 config={"configurable": {"thread_id": "user-session-abc123"}})
    # Graph has full history of "Sony headphones" conversation
    # Agent knows exactly which product "the first one" refers to
```

**Why PostgreSQL for state (not in-memory)?**
- Survives server restarts
- Multiple server instances can share state (horizontal scaling)
- State persists forever → users can return tomorrow with same conversation

### SSE Streaming

**SSE = Server-Sent Events** — a one-way channel where server pushes data to browser.

```python
def rag_agent_stream_wrapper(question: str, thread_id: str):
    # This function is a GENERATOR (uses yield instead of return)
    # FastAPI's StreamingResponse calls next() on this repeatedly
    
    def _string_for_sse(message: str):
        return f"data: {message}\n\n"  # SSE format: "data: " + content + two newlines
    
    # Start the LangGraph graph with streaming mode
    for chunk in graph.stream(
        initial_state, 
        config=config,
        stream_mode=["debug", "values"]
        # "debug": emits events for each node start/end
        # "values": emits full state after each node
    ):
        processed_chunk = _process_graph_event(chunk)
        
        if processed_chunk:
            # Yield status messages: "Planning...", "Looking for items..."
            yield _string_for_sse(processed_chunk)
        
        if chunk[0] == "values":
            result = chunk[1]  # save the latest state
    
    # After graph finishes, send the final result as JSON
    yield _string_for_sse(json.dumps({
        "type": "final_result",
        "data": {
            "answer": result.get("answer", ""),
            "used_context": used_context,   # product images + prices
            "trace_id": result.get("trace_id", ""),
            "shopping_cart": shopping_cart_items
        }
    }))
```

**What status messages look like on the frontend:**
```
User sends: "Find me Bluetooth headphones"
Frontend shows:
  ⏳ "Planning..."                    ← coordinator_agent started
  ⏳ "Looking for items: bluetooth headphones."  ← tool called
  → [Final answer appears]
```

### MCP Servers (Model Context Protocol)

**MCP** is a standard protocol for tools. Instead of defining tools as Python functions inside the agent, you expose them as remote HTTP services that any LLM client can discover and call.

**`apps/items_mcp_server/src/items_mcp_server/main.py`:**

```python
from fastmcp import FastMCP  # FastMCP = fast MCP server framework (like FastAPI for MCP)

mcp = FastMCP("items_mcp_server")

# @mcp.tool() makes this function available as an MCP tool
# The docstring becomes the tool description that the LLM reads
@mcp.tool()
def get_formatted_items_context(query: str, top_k: int = 5) -> str:
    """Get the top k context, each representing an inventory item for a given query.
    
    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more
    
    Returns:
        A string of the top k context chunks with IDs and average ratings...
    """
    context = retrieve_items_data(query, top_k)
    formatted_context = process_items_context(context)
    return formatted_context

# Run as HTTP server — accessible from any other service in Docker network
if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
```

**MCP vs direct Python tools:**
| Aspect | Python Tool | MCP Tool |
|---|---|---|
| Location | Same process as agent | Separate HTTP service |
| Language | Must be Python | Any language |
| Discovery | Hardcoded in code | Dynamic discovery |
| Scaling | Scales with agent | Independent scaling |
| Reuse | One agent | Any MCP-compatible client |

---

## Part 6 — Multi-Agent System: Coordinator + 3 Workers (Videos 45–52)

### Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    COORDINATOR AGENT                         │
│  Role: Plan, delegate, synthesize                           │
│  LLM: gpt-4.1 (primary) or llama-3.3-70b (fallback)        │
│  Input: User question + conversation history                │
│  Output: {next_agent, plan, final_answer, answer}           │
└─────────┬─────────────────┬──────────────────┬─────────────┘
          │                 │                  │
          ▼                 ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐
│ PRODUCT QA   │   │ SHOPPING     │   │ WAREHOUSE MANAGER    │
│ AGENT        │   │ CART AGENT   │   │ AGENT                │
│              │   │              │   │                      │
│ Tools:       │   │ Tools:       │   │ Tools:               │
│ - items      │   │ - add_to_cart│   │ - check_availability │
│ - reviews    │   │ - remove     │   │ - reserve_items      │
│              │   │ - get_cart   │   │                      │
│ DB: Qdrant   │   │ DB: Postgres │   │ DB: Postgres         │
└──────────────┘   └──────────────┘   └──────────────────────┘
```

### `apps/api/src/api/agents/agents.py` — Each Agent Function

**How every agent works (same pattern for all 3 workers):**

```python
@traceable(name="product_qa_agent", run_type="llm", ...)
def product_qa_agent(state, models=["gpt-4.1", "groq/llama-3.3-70b-versatile"]) -> dict:
    
    # STEP 1: Load prompts — one per model (GPT and Llama may need different formats)
    prompts = {}
    for model in models:
        prompts[model] = prompt_template_config(
            "api/agents/prompts/product_qa_agent.yaml",
            model  # key: "gpt-4.1" or "groq/llama-3.3-70b-versatile"
        ).render(
            available_tools=state.product_qa_agent.available_tools
            # Jinja2 injects the tool descriptions into the prompt
        )
    
    # STEP 2: Prepare conversation history
    # LangGraph stores messages as LangChain objects → convert to OpenAI format
    messages = state.messages
    conversation = []
    for message in messages:
        conversation.append(convert_to_openai_messages(message))
    # Result: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    # STEP 3: Call LLM with fallback (LiteLLM routes to correct provider)
    client = instructor.from_litellm(completion)  # instructor + LiteLLM combo
    
    for model in models:  # try primary model first, fallback to next if error
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,                          # "gpt-4.1" or "groq/..."
                response_model=ProductQAAgentResponse,  # MUST return this structure
                messages=[
                    {"role": "system", "content": prompts[model]},  # model-specific prompt
                    *conversation                                   # conversation history
                ],
                temperature=0.5,
            )
            break  # success! exit fallback loop
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue  # try next model
    
    # STEP 4: Log token usage to LangSmith
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
    
    # STEP 5: Format response as LangGraph-compatible AI message
    ai_message = format_ai_message(response)
    # If response.tool_calls is non-empty → creates AIMessage with tool_calls
    # This is how LangGraph's ToolNode knows which tool to execute
    
    # STEP 6: Return updated state (merged with existing state by LangGraph)
    return {
        "messages": [ai_message],                     # append to messages list
        "product_qa_agent": {
            "tool_calls": [tc.model_dump() for tc in response.tool_calls],
            "iteration": state.product_qa_agent.iteration + 1,  # increment counter
            "final_answer": response.final_answer,
            "available_tools": state.product_qa_agent.available_tools
        },
        "answer": response.answer,        # the agent's current best answer
        "references": response.references  # products cited in this answer
    }
```

**`ProductQAAgentResponse` schema:**

```python
class ToolCall(BaseModel):
    name: str       # e.g. "get_formatted_items_context"
    arguments: dict # e.g. {"query": "bluetooth headphones", "top_k": 5}

class RAGUsedContext(BaseModel):
    id: str          # product ID (Amazon ASIN), e.g. "B07PFFMP9P"
    description: str # short product description for sidebar

class ProductQAAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext]   # which products to show in sidebar
    final_answer: bool = False         # True = done, no more tools needed
    tool_calls: List[ToolCall] = []    # which tools to call next (if any)
    # Rules: final_answer=True → tool_calls must be []
    #        tool_calls non-empty → final_answer must be False
```

**Coordinator agent — the brain:**

```python
@traceable(name="coordinator_agent", run_type="llm", ...)
def coordinator_agent(state, models=["gpt-4.1", "groq/llama-3.3-70b-versatile"]):
    # Same pattern as worker agents, but with CoordinatorAgentResponse

    # CoordinatorAgentResponse fields:
    # next_agent: "product_qa_agent" | "shopping_cart_agent" | "warehouse_manager_agent" | ""
    # plan: [{agent: "...", task: "..."}]  ← coordinator's work plan
    # final_answer: bool  ← True when coordinator has enough info to answer user
    # answer: str         ← the final answer to return to user
    
    if response.final_answer:
        # Coordinator is done → create final AI message for conversation history
        ai_message = [AIMessage(content=response.answer)]
    else:
        ai_message = []  # coordinator is still planning → no message yet
    
    return {
        "messages": ai_message,
        "answer": response.answer,
        "coordinator_agent": {
            "iteration": state.coordinator_agent.iteration + 1,
            "final_answer": response.final_answer,
            "next_agent": response.next_agent,  # which worker to call
            "plan": [d.model_dump() for d in response.plan]
        },
        "trace_id": trace_id  # LangSmith trace ID for feedback correlation
    }
```

### Coordinator Prompt Logic

From `apps/api/src/api/agents/prompts/coordinator_agent.yaml`:

```yaml
# The prompt tells the coordinator its CRITICAL RULES:
CRITICAL RULES:
- If next_agent is "", final_answer MUST be false
  (You cannot delegate AND return to user in the same response)
- If final_answer is true, next_agent MUST be ""
  (You must wait for all agent results before answering)

# Example flow for: "Find me headphones and add the best to cart"
# Round 1 (coordinator):
  next_agent: "product_qa_agent"
  plan: [{agent: product_qa_agent, task: find bluetooth headphones}]
  final_answer: false

# Round 2 (after product_qa_agent returns):
  next_agent: "shopping_cart_agent"
  plan: [{agent: shopping_cart_agent, task: add best product B07PFFMP9P to cart}]
  final_answer: false

# Round 3 (after shopping_cart_agent returns):
  next_agent: ""
  final_answer: true
  answer: "I found Sony MDR-ZX110 headphones (4.5★, $24.99) and added them to your cart!"
```

### `apps/api/src/api/agents/tools.py` — All 6 Tools

**Tool 1 & 2: Product Search + Reviews**

```python
def get_formatted_items_context(query: str, top_k: int = 5) -> str:
    """Retrieves top products from Qdrant using hybrid search."""
    context = retrieve_items_data(query, top_k)   # Qdrant query
    formatted_context = process_items_context(context)  # format as string
    return formatted_context
    # Returns: "- ID: B07P..., rating: 4.5, description: Sony MDR..."

def get_formatted_reviews_context(query: str, item_list: list, top_k: int = 15) -> str:
    """Retrieves top reviews for a pre-filtered list of products."""
    context = retrieve_reviews_data(query, item_list, top_k)  # filter by item_list first
    formatted_context = process_reviews_context(context)
    return formatted_context
    # Returns: "- ID: B07P..., review: Great sound quality, comfortable..."
```

**Tool 3: Add to Shopping Cart**

```python
def add_to_shopping_cart(items: list[dict], user_id: str, cart_id: str) -> str:
    conn = psycopg2.connect(host="postgres", port=5432, database="tools_database", ...)
    conn.autocommit = True  # each operation is its own transaction
    
    for item in items:
        product_id = item['product_id']  # Amazon ASIN
        quantity = item['quantity']
        
        # Look up price and image from Qdrant (products data)
        payload = qdrant_client.query_points(...filter by product_id...).points[0].payload
        price = payload.get("price")
        product_image_url = payload.get("image")
        
        # Check if product already in cart (UPSERT pattern)
        cursor.execute("SELECT id, quantity FROM shopping_carts.shopping_cart_items WHERE user_id=%s AND product_id=%s", ...)
        existing_item = cursor.fetchone()
        
        if existing_item:
            # Product already in cart → increase quantity
            new_quantity = existing_item['quantity'] + quantity
            cursor.execute("UPDATE shopping_carts.shopping_cart_items SET quantity=%s WHERE ...", ...)
        else:
            # New product → insert row
            cursor.execute("INSERT INTO shopping_carts.shopping_cart_items (...) VALUES (...)", ...)
    
    return f"Added {items} to the shopping cart."
```

**Tool 4: Get Shopping Cart**

```python
def get_shopping_cart(user_id: str, cart_id: str) -> list[dict]:
    # SELECT from PostgreSQL — returns all items in this user's cart
    cursor.execute("""
        SELECT product_id, price, quantity, currency, product_image_url,
               (price * quantity) as total_price
        FROM shopping_carts.shopping_cart_items 
        WHERE user_id = %s AND shopping_cart_id = %s
        ORDER BY added_at DESC
    """, (user_id, cart_id))
    
    return [dict(row) for row in cursor.fetchall()]
    # Returns: [{"product_id": "B07P...", "price": 24.99, "quantity": 2, "total_price": 49.98, ...}]
```

**Tool 5: Check Warehouse Availability**

```python
def check_warehouse_availability(items: list[dict]) -> dict:
    # items = [{"product_id": "B07P...", "quantity": 2}, ...]
    
    result = {
        "can_fulfill_completely": False,
        "warehouses_full_fulfillment": [],    # warehouses that have ALL items
        "warehouses_partial_fulfillment": [], # warehouses with some items
        "unavailable_items": [],              # items out of stock everywhere
        "details": []                         # warehouse-by-warehouse breakdown
    }
    
    # Get all warehouses
    cursor.execute("SELECT DISTINCT warehouse_id, warehouse_name, warehouse_location FROM warehouses.inventory")
    warehouses = cursor.fetchall()
    
    for warehouse in warehouses:
        warehouse_can_fulfill_all = True
        has_any_availability = False
        
        for item in items:
            # Check available_quantity = total_quantity - reserved_quantity (computed column)
            cursor.execute("""
                SELECT available_quantity FROM warehouses.inventory
                WHERE warehouse_id = %s AND product_id = %s
            """, (warehouse['warehouse_id'], item['product_id']))
            inventory = cursor.fetchone()
            
            available_qty = inventory['available_quantity'] if inventory else 0
            
            if available_qty < item['quantity']:
                warehouse_can_fulfill_all = False
            if available_qty > 0:
                has_any_availability = True
        
        # Categorize this warehouse
        if warehouse_can_fulfill_all:
            result["warehouses_full_fulfillment"].append(warehouse)
        elif has_any_availability:
            result["warehouses_partial_fulfillment"].append(warehouse)
    
    result["can_fulfill_completely"] = len(result["warehouses_full_fulfillment"]) > 0
    return result
```

**Tool 6: Reserve Warehouse Items (Atomic Transaction)**

```python
def reserve_warehouse_items(reservations: list[dict]) -> dict:
    conn.autocommit = False  # START TRANSACTION — all-or-nothing
    
    for reservation in reservations:
        # FOR UPDATE = row-level lock (prevents double-booking)
        # If two users try to reserve the same item simultaneously,
        # one gets the lock, the other waits → correct sequential processing
        cursor.execute("""
            SELECT available_quantity FROM warehouses.inventory
            WHERE warehouse_id = %s AND product_id = %s
            FOR UPDATE
        """, (warehouse_id, product_id))
        
        if inventory and inventory['available_quantity'] >= quantity:
            cursor.execute("""
                UPDATE warehouses.inventory
                SET reserved_quantity = reserved_quantity + %s
                WHERE warehouse_id = %s AND product_id = %s
            """, (quantity, warehouse_id, product_id))
            result["reserved_items"].append(...)
        else:
            result["failed_items"].append(...)
    
    if len(result["failed_items"]) == 0:
        conn.commit()          # ALL succeeded → commit transaction
        result["success"] = True
    else:
        conn.rollback()        # ANY failed → rollback everything (atomic!)
        result["success"] = False
    
    return result
```

**Why `FOR UPDATE` and atomic transactions?**
- Race condition: User A and User B both want to buy the last unit
- Without `FOR UPDATE`: both read available_qty=1, both proceed, inventory goes negative
- With `FOR UPDATE`: User A gets the lock, User B waits → User A commits → User B sees qty=0 → correct

### Database Schema

**Shopping Cart (`scripts/sql/shopping_cart_table.sql`):**

```sql
CREATE TABLE shopping_carts.shopping_cart_items (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,        -- which user's cart
    shopping_cart_id VARCHAR(255),        -- allows multiple carts per user
    product_id VARCHAR(255) NOT NULL,     -- Amazon ASIN
    price DECIMAL(10, 2),
    quantity INTEGER NOT NULL DEFAULT 1,
    currency VARCHAR(3) DEFAULT 'USD',
    product_image_url VARCHAR(1000),
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Prevent negative prices or quantities
    CONSTRAINT positive_price CHECK (price >= 0),
    CONSTRAINT positive_quantity CHECK (quantity > 0),
    
    -- Prevent adding same product twice (use UPDATE instead)
    CONSTRAINT unique_user_cart_product UNIQUE (user_id, shopping_cart_id, product_id)
);
```

**Warehouse Inventory (`scripts/sql/warehouse_management.sql`):**

```sql
CREATE TABLE warehouses.inventory (
    warehouse_id VARCHAR(255) NOT NULL,   -- e.g. "WH-EAST-001"
    product_id VARCHAR(255) NOT NULL,     -- Amazon ASIN
    total_quantity INTEGER NOT NULL,      -- physical units on shelf
    reserved_quantity INTEGER NOT NULL DEFAULT 0,  -- units reserved, not yet shipped
    
    -- COMPUTED COLUMN: auto-calculated as total - reserved
    -- Cannot go negative due to constraint
    available_quantity INTEGER GENERATED ALWAYS AS (total_quantity - reserved_quantity) STORED,
    
    warehouse_name VARCHAR(255),
    warehouse_location VARCHAR(255),
    
    -- Prevent reserving more than exists
    CONSTRAINT valid_reservation CHECK (reserved_quantity <= total_quantity),
    
    -- Unique item per warehouse
    CONSTRAINT unique_warehouse_product UNIQUE (warehouse_id, product_id)
);
```

---

## Part 7 — LiteLLM, ADK, A2A Protocol, Prompt Caching (Videos 55–62)

### LiteLLM — Model Routing with Fallback

**Problem:** Your code is tightly coupled to one LLM provider. If OpenAI has an outage, your system fails.
**Solution:** LiteLLM provides a unified interface to 100+ LLM providers. Switch providers by changing the model string.

```python
from litellm import completion

# These ALL use the same function call — LiteLLM routes to correct provider
completion(model="gpt-4.1", ...)             # → OpenAI API
completion(model="groq/llama-3.3-70b-versatile", ...)  # → Groq API
completion(model="gemini/gemini-2.5-flash", ...)        # → Google API

# With instructor for structured output:
client = instructor.from_litellm(completion)

# Primary → fallback pattern:
for model in ["gpt-4.1", "groq/llama-3.3-70b-versatile"]:
    try:
        response = client.chat.completions.create_with_completion(model=model, ...)
        break  # success!
    except Exception as e:
        continue  # try next model
```

**Per-model prompts:** GPT-4.1 and Llama 3.3 respond better to different prompt formats.
```yaml
# coordinator_agent.yaml
prompts:
  gpt-4.1: |
    You are a Coordinator Agent...
    [GPT-optimized prompt with structured format]
  
  groq/llama-3.3-70b-versatile: |
    You are a Coordinator Agent...
    [Llama-optimized prompt — same logic, different phrasing]
```

### Google ADK (Agent Development Kit)

**`apps/adk_warehouse_manager_agent/warehouse_manager_agent/agent.py`:**

```python
from google.adk.agents import Agent       # Google's official agent framework
from google.adk.models.lite_llm import LiteLlm  # LiteLLM model wrapper for ADK

# Use OpenAI model through LiteLLM (not Google's own model)
model = LiteLlm(
    model="openai/gpt-4.1-mini",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ADK Agent is much simpler than LangGraph
# No state management, no graph, no edges
# Just: define tools + instruction → ADK handles the ReAct loop automatically
root_agent = Agent(
    name="warehouse_manager_agent",
    model=model,
    tools=[check_warehouse_availability, reserve_warehouse_items],
    description="A agent that can check warehouse availability and reserve items.",
    instruction="""
You are a part of the shopping assistant that can manage warehouse inventory.
Instructions:
- Always check availability BEFORE reserving
- Only reserve if entire order can be filled (or user confirmed partial)
- Try closest warehouse first if user location is provided
"""
)
# Run with: adk web (starts ADK web UI for testing)
```

**LangGraph vs Google ADK:**
| Feature | LangGraph | Google ADK |
|---|---|---|
| Control | Full (define every edge/node) | Automatic |
| Complexity | Higher | Lower |
| Flexibility | Maximum | More opinionated |
| Multi-agent | Native support | A2A protocol |
| UI for testing | LangSmith | adk web |
| Best for | Complex custom workflows | Quick agent prototypes |

### A2A Protocol (Agent-to-Agent)

**Problem:** Each agent is its own service. How do agents in different services communicate?
**Solution:** A2A is a standardized HTTP protocol for agent-to-agent communication.

```
┌─────────────────────────┐         ┌────────────────────────────────┐
│  LangGraph Coordinator  │         │  A2A Warehouse Manager Server  │
│  (inside FastAPI)       │──HTTP──▶│  (separate FastAPI service)     │
│                         │  A2A    │                                 │
│  agent.send_task(...)   │         │  POST /.well-known/agent.json   │
│                         │◀────────│  POST /tasks/send               │
└─────────────────────────┘         └────────────────────────────────┘
```

**A2A warehouse agent server (`apps/a2a_warehouse_manager_agent/warehouse_manager_agent/app.py`):**
- Exposes standard A2A endpoints
- LangGraph coordinator discovers it and sends tasks to it
- The warehouse agent runs its own LangGraph graph internally
- Returns results back to coordinator via A2A

### Prompt Caching (Video 61–62)

**Problem:** Long system prompts are sent with EVERY request → high costs, added latency.
**Solution:** Provider caches the prompt — only the changing part (user message) is billed.

```python
# OpenAI prompt caching: prompts > 1024 tokens are automatically cached
# Anthropic requires explicit cache_control markers

# Savings:
# Without caching: 2000 token prompt × 1000 requests = 2M tokens billed
# With caching:    2000 token prompt cached + 100 user tokens × 1000 = 100K tokens billed
# Savings: ~95% reduction in input token cost

# In the agents code, long system prompts (coordinator prompt, product_qa_agent prompt)
# are automatically eligible for caching with GPT-4.1 (OpenAI caches prompts > 1024 tokens)
```

---

## Part 8 — HITL, CI Eval Pipeline, Cloud Deployment (Videos 66–73)

### HITL (Human-in-the-Loop)

**Concept:** Before the agent commits to an action (like reserving expensive items from a warehouse), pause and ask the human for confirmation.

```
Without HITL:
User: "Buy 50 units of Sony headphones from warehouse"
Agent: [immediately reserves 50 units] "Done!"
Problem: What if the user typed 50 instead of 5? No undo!

With HITL:
User: "Buy 50 units of Sony headphones from warehouse"
Agent: "I'm about to reserve 50 × Sony MDR-ZX110 ($1,249.50 total) from Warehouse East.
        Confirm? [Yes] [No]"
User: "No, I meant 5 units"
Agent: "Understood. Reserving 5 units... Done!"
```

### LangSmith Feedback Integration

```python
# apps/api/src/api/api/processors/submit_feedback.py
def submit_feedback(trace_id, feedback_score, feedback_text, feedback_source_type):
    ls_client = Client()
    ls_client.create_feedback(
        run_id=trace_id,              # links feedback to the specific LangSmith trace
        key="user_feedback",
        score=feedback_score,         # 1 = positive, 0 = negative
        comment=feedback_text,        # optional explanation
        source_type=feedback_source_type,  # "api" or "human"
    )
# This creates a feedback entry in LangSmith UI
# You can see: "30% of users gave thumbs down on warehouse questions"
# → investigate those traces → improve those specific prompts
```

### CI Quality Gate (Video 69)

**Concept:** Run evals automatically on every code change (like unit tests in CI).

```yaml
# .github/workflows/eval.yml
name: RAG Eval Pipeline
on: [push, pull_request]

jobs:
  run-evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run evals
        run: |
          make run-evals-retriever
      - name: Check eval scores
        run: |
          python scripts/check_eval_scores.py --min-faithfulness 0.7 --min-precision 0.3
          # Fails CI if scores drop below thresholds
          # This prevents regressions: if a code change breaks retrieval, CI catches it
```

### Qdrant Cloud Migration (Video 70)

```python
# Local (dev): connecting to Docker Qdrant
qdrant_client = QdrantClient(url="http://qdrant:6333")

# Cloud (production): connecting to Qdrant Cloud
qdrant_client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY")
)
# Only change is the connection URL + API key
# All collection names, query patterns stay the same
```

---

## Complete Code File-by-File Explanation

### `apps/chatbot_ui/src/chatbot_ui/core/config.py`

```python
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    API_URL: str = "http://api:8000"  # Docker service name "api" resolves within Docker network
    # On your laptop (outside Docker): would be http://localhost:8000
    # Inside Docker: "api" is the service name from docker-compose.yml
    
    class Config:
        env_file = ".env"  # reads from .env file

config = Config()  # singleton — import this everywhere
```

### `apps/api/src/api/core/config.py`

```python
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    openai_api_key: str       # OPENAI_API_KEY env var
    langchain_api_key: str    # LANGCHAIN_API_KEY env var
    groq_api_key: str         # GROQ_API_KEY env var
    google_api_key: str       # GOOGLE_API_KEY env var
    # If any are missing → ValidationError at startup (fail-fast, not at runtime)

config = Config()
```

### `apps/api/src/api/agents/utils/utils.py` — Tool Description Parser

```python
def get_tool_descriptions(function_list):
    """Extract tool descriptions from Python functions using AST (Abstract Syntax Tree)."""
    descriptions = []
    
    for function in function_list:
        # inspect.getsource() gets the raw source code of the function as a string
        function_string = inspect.getsource(function)
        
        # parse_function_definition uses Python's ast module to parse the source code
        # and extract: function name, parameter names, types, docstring, return type
        result = parse_function_definition(function_string)
        descriptions.append(result)
    
    return descriptions
    # Output example:
    # [{
    #   "name": "get_formatted_items_context",
    #   "description": "Get the top k context...",
    #   "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}},
    #   "required": ["query"],
    #   "returns": {"type": "string"}
    # }]
```

**Why parse tool descriptions this way?**
- The agent's prompt includes descriptions of available tools
- Instead of hardcoding tool descriptions, we parse them from docstrings
- When you change a docstring, the agent's tool description updates automatically
- Single source of truth: the code IS the documentation

---

## How to Run the Project End-to-End

### Prerequisites

```bash
# 1. Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Docker Desktop
# Download from docker.com

# 3. Clone the repository
git clone https://github.com/thepembeweb/ai-product-assistant.git
cd ai-product-assistant
```

### Step 1: Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your actual API keys:
```

```bash
# .env
OPENAI_API_KEY=sk-...                  # from platform.openai.com
GROQ_API_KEY=gsk_...                   # from console.groq.com
GOOGLE_API_KEY=AIza...                 # from aistudio.google.com
LANGCHAIN_API_KEY=ls__...             # from smith.langchain.com
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ai-product-assistant
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
QDRANT_API_KEY=                        # only needed for Qdrant Cloud
```

### Step 2: Set Up Qdrant Vector Database

```bash
# Start Qdrant only first (you need it to load data)
docker compose up qdrant -d

# Run the data loading notebooks:
# notebooks/week_1/01-explore-amazon-dataset.ipynb  → explore the Amazon dataset
# notebooks/week_1/02-RAG-preprocessing-Amazon.ipynb → embed and index products into Qdrant
```

**What the notebooks do:**
1. Load Amazon electronics data (products with ASIN codes, descriptions, ratings, images)
2. Split into chunks
3. Create embeddings with `text-embedding-3-small`
4. Create Qdrant collection with two indexes: vector (`text-embedding-3-small`) + BM25
5. Upload all products to Qdrant

### Step 3: Set Up PostgreSQL Database

```bash
# Start PostgreSQL
docker compose up postgres -d

# Wait for postgres to start, then create the tables
docker compose exec postgres psql -U langgraph_user -d tools_database \
  -f /scripts/sql/shopping_cart_table.sql

docker compose exec postgres psql -U langgraph_user -d tools_database \
  -f /scripts/sql/warehouse_management.sql

# Optionally: run the warehouse data generation notebook
# notebooks/week_5/04-Warehouse-Agent-Database.ipynb
```

### Step 4: Start All Services

```bash
# Start everything
docker compose up --build

# Or start in background
docker compose up --build -d

# View logs
docker compose logs -f api
docker compose logs -f streamlit-app
```

### Step 5: Access the Application

```
Streamlit UI:     http://localhost:8501
FastAPI Swagger:  http://localhost:8000/docs
Qdrant UI:        http://localhost:6333/dashboard
```

### Step 6: Test via Swagger UI

```
Open http://localhost:8000/docs
→ POST /agent
→ "Try it out"
→ Body:
{
  "query": "Show me Bluetooth headphones under $50",
  "thread_id": "test-thread-001"
}
→ Execute
→ See streaming response
```

### Step 7: Run Evaluations

```bash
# Run RAG evaluation against LangSmith dataset
make run-evals-retriever
# Runs 31 reference questions × 4 metrics
# Results visible in LangSmith → Datasets & Experiments
```

---

## What Happens When You Run It — Expected Results

### User asks: "Show me Bluetooth headphones with good battery"

**Step 1 — Coordinator starts:**
```json
SSE message 1: "data: Planning..."
→ Coordinator creates plan:
  {
    "next_agent": "product_qa_agent",
    "plan": [{"agent": "product_qa_agent", "task": "Find bluetooth headphones with good battery life"}],
    "final_answer": false
  }
```

**Step 2 — Product QA Agent starts:**
```json
SSE message 2: "data: Looking for items: bluetooth headphones battery life."
→ Product QA Agent calls: get_formatted_items_context(query="bluetooth headphones battery", top_k=5)
```

**Step 3 — Qdrant returns 5 products:**
```
- ID: B07PFFMP9P, rating: 4.5, description: Sony WH-1000XM4 Wireless Noise Canceling Headphones, 30hr battery...
- ID: B08XB3HRBT, rating: 4.3, description: JBL Tune 760NC Wireless Headphones, 35 hours battery...
- ID: B09DRTC58P, rating: 4.1, description: Anker Soundcore Life Q20 Hybrid Headphones, 40 hrs...
- ID: B09KSMJ67J, rating: 4.4, description: Jabra Evolve2 55 Headset, 34 hours...
- ID: B07YZL4MQL, rating: 4.6, description: Bose QuietComfort 35 II, 20hr battery...
```

**Step 4 — Product QA Agent generates answer:**
```json
SSE message 3: (status update if reviews are also fetched)
→ Product QA Agent produces:
  {
    "answer": "Here are Bluetooth headphones with excellent battery life:
               1. **Anker Soundcore Life Q20** - Up to 40 hours battery, hybrid ANC...
               2. **JBL Tune 760NC** - 35 hours of battery life...
               3. **Sony WH-1000XM4** - 30 hours, industry-leading noise cancellation...",
    "references": [
      {"id": "B09DRTC58P", "description": "Anker Soundcore Life Q20"},
      {"id": "B08XB3HRBT", "description": "JBL Tune 760NC"},
      {"id": "B07PFFMP9P", "description": "Sony WH-1000XM4"}
    ],
    "final_answer": true,
    "tool_calls": []
  }
```

**Step 5 — Coordinator synthesizes:**
```json
→ Coordinator sees product_qa_agent is done
→ final_answer: true
→ answer: the product QA agent's answer
```

**Step 6 — Final SSE message:**
```json
SSE message (final): data: {
  "type": "final_result",
  "data": {
    "answer": "Here are Bluetooth headphones with excellent battery life...",
    "used_context": [
      {"image_url": "https://amazon.com/...jpg", "price": 59.99, "description": "Anker Soundcore Life Q20"},
      {"image_url": "https://...", "price": 79.99, "description": "JBL Tune 760NC"},
      {"image_url": "https://...", "price": 279.00, "description": "Sony WH-1000XM4"}
    ],
    "trace_id": "abc123-langsmith-trace-id",
    "shopping_cart": []  # cart is empty until user adds something
  }
}
```

**Frontend renders:**
- Chat: Full product comparison text
- Sidebar (Suggestions tab): 3 product images with prices
- LangSmith: Full trace visible with all spans, token counts, costs

---

## Different Ways of Doing Each Process

### 1. Vector Retrieval Options

| Method | How | When to Use |
|---|---|---|
| Pure Vector Search | `qdrant_client.search(query_vector=embedding)` | Semantic questions ("comfortable headphones") |
| Pure BM25 Search | BM25 keyword index | Exact product names, model numbers |
| Hybrid (RRF) | Both + RRF fusion | Default — best of both worlds |
| Filtered Search | Add `query_filter` parameter | "headphones under $50" (filter by price field) |
| Reranking | Cohere Rerank API after retrieval | When you need top precision (adds 1 extra API call) |

```python
# Pure vector (simplest)
results = qdrant_client.search(collection_name=..., query_vector=embedding, limit=5)

# Hybrid with RRF (best quality — used in this project)
results = qdrant_client.query_points(
    prefetch=[Prefetch(query=embedding, using="text-embedding-3-small"), 
              Prefetch(query=Document(text=query, model="qdrant/bm25"), using="bm25")],
    query=FusionQuery(fusion="rrf"), limit=5
)

# With reranking (highest quality, extra cost + latency)
# Step 1: retrieve 20 candidates
results = qdrant_client.query_points(..., limit=20)
# Step 2: rerank with Cohere (picks best 5 from 20)
reranked = cohere_client.rerank(query=query, documents=results, top_n=5)
```

### 2. Agent Orchestration Options

| Framework | Code Pattern | Trade-offs |
|---|---|---|
| LangGraph (this project) | Graph with nodes + conditional edges | Full control, complex setup |
| Google ADK | `Agent(tools=[...], instruction="...")` | Simple, less control |
| A2A Protocol | HTTP calls between agent services | Distributed, language-agnostic |
| Simple Chain | Sequential function calls | No loops, no tools |
| LangChain ReAct | `AgentExecutor` | Less flexible than LangGraph |

### 3. LLM Call Options

| Pattern | Code | When to Use |
|---|---|---|
| Direct OpenAI | `openai.ChatCompletion.create(...)` | Single provider, simple |
| LiteLLM | `litellm.completion(model="gpt-4.1", ...)` | Multi-provider, routing |
| Instructor | `instructor.from_openai/litellm(...)` | Need structured output |
| LangChain | `ChatOpenAI(model="gpt-4.1")` | LangChain ecosystem |
| With fallback | `for model in models: try...except` | High availability |

### 4. State Persistence Options

| Method | Trade-offs |
|---|---|
| PostgresSaver (this project) | Durable, scalable, survives restarts |
| SqliteSaver | Simple, file-based, not for production |
| MemorySaver | In-memory only, lost on restart |
| RedisSaver | Fast, distributed, TTL support |
| No persistence | Each request is independent (no multi-turn) |

### 5. Prompt Management Options

| Method | Trade-offs |
|---|---|
| YAML + Jinja2 (this project) | Version-controlled, no code changes needed |
| LangSmith Hub | Remote prompt registry, A/B testing |
| Hardcoded strings | Simple but not maintainable |
| Python f-strings | Flexible but not versionable |
| Langfuse | Open-source alternative to LangSmith |

---

## Differences Between Each Part / Step

### RAG Evolution: Simple → Advanced

| Step | Method | Precision | Speed | Complexity |
|---|---|---|---|---|
| Naive RAG | Single vector search | Low | Fast | Simple |
| + Hybrid Search | Vector + BM25 + RRF | Medium | Medium | Medium |
| + Reranking | Hybrid + Cohere rerank | High | Slower | Medium |
| + Agentic RAG | Agent calls retrieval tools | Highest | Slowest | Complex |

**What changes at each step:**
1. **Naive RAG:** One embedding lookup → top 5 → prompt → answer
2. **+ Structured Output:** Answer includes product IDs (not just text) → sidebar can show images
3. **+ Hybrid Search:** Two indexes (vector + BM25) → better recall for keyword queries
4. **+ Reranking:** 20 candidates → Cohere picks best 5 → better precision
5. **+ Prompt Versioning:** YAML prompts → can iterate without code changes
6. **+ Agentic RAG:** Agent DECIDES when to call retrieval, can call multiple times

### Agent Evolution: Single → Multi-Agent

| Step | Architecture | What's New |
|---|---|---|
| Simple RAG | No agent, just pipeline | - |
| ReAct Agent | Single LangGraph node with tools | Can loop, can use tools |
| Intent Router | Conditional edge based on intent | Routes to different handlers |
| Multi-Turn | PostgreSQL checkpointing | Remembers previous messages |
| + Tools | Multiple tool functions | Can modify data (cart, warehouse) |
| + MCP | Tools as remote HTTP services | Language-agnostic, discoverable |
| + Streaming | SSE from backend | Real-time status updates |
| Multi-Agent | Coordinator + 3 workers | Specialization, parallel work |
| + LiteLLM | Model fallback | High availability |
| + ADK | Google ADK agent | Alternative orchestration |
| + A2A | Remote agent communication | Distributed agents |

### Key Differences Between Coordinator and Intent Router

| Aspect | Intent Router (Video 29) | Coordinator Agent (Video 47+) |
|---|---|---|
| Type | Deterministic rules | LLM-powered |
| Routing | Based on classified intent | Based on understanding + planning |
| Multi-step | No | Yes (creates multi-step plans) |
| Fallback | Manual rules | LLM decides |
| Complex queries | Limited | Handles "find + add to cart" |

---

## Comparison Table — Evolution of the System

| Video | What Was Built | Key Addition | Files Changed |
|---|---|---|---|
| 1-4 | Basic chatbot | UV, Streamlit, FastAPI, Docker | app.py, Dockerfile, docker-compose.yml |
| 8 | Naive RAG | Qdrant, embeddings, retrieval | retrieval_generation.py |
| 10 | Observability | @traceable, LangSmith tracing | retrieval_generation.py + @traceable |
| 11 | Eval dataset | Synthetic Q&A generation | LangSmith dataset |
| 12-13 | Evals | Ragas metrics, eval script | evals/eval_retriever.py |
| 17-18 | Structured outputs | Instructor, RAGGenerationResponse | retrieval_generation.py |
| 19-20 | Backend integration | API serves structured output | endpoints.py, models.py |
| 21 | Hybrid search | BM25 + RRF in Qdrant | tools.py |
| 22 | Reranking | Cohere Rerank API | tools.py |
| 23 | Prompt versioning | YAML + Jinja2 prompts | prompts/*.yaml, prompt_management.py |
| 26-27 | LangGraph intro | StateGraph, nodes, edges | graph.py (initial) |
| 28 | Query decomposition | Multiple tool calls | agents.py |
| 29 | Intent router | Conditional routing | graph.py |
| 30 | ReAct agent | Tool-using loop | graph.py |
| 31 | Backend integration | Agent endpoint | endpoints.py |
| 35-36 | Multi-turn | PostgreSQL checkpointing | graph.py |
| 37 | Reviews tool | Second retrieval tool | tools.py |
| 38 | Feedback | LangSmith feedback | submit_feedback.py, app.py |
| 39 | MCP servers | FastMCP, items/reviews MCP | items_mcp_server/, reviews_mcp_server/ |
| 40 | Streaming | SSE, rag_agent_stream_wrapper | graph.py, endpoints.py, chatbot app.py |
| 45-46 | Shopping cart | Cart agent + tools | agents.py, tools.py |
| 47-48 | Coordinator | Replaces intent router | graph.py, agents.py |
| 49-50 | Warehouse | Warehouse DB + tools | tools.py, warehouse_management.sql |
| 51-52 | 3-agent system | Full multi-agent graph | graph.py |
| 55-56 | LiteLLM | Model routing + fallback | agents.py |
| 57-58 | Google ADK | ADK agent | adk_warehouse_manager_agent/ |
| 59-60 | A2A protocol | Agent-to-agent HTTP | a2a_warehouse_manager_agent/ |
| 61-62 | Prompt caching | Cache long prompts | agents.py |
| 66-67 | HITL | Human confirmation step | graph.py |
| 68-69 | CI evals | GitHub Actions eval pipeline | .github/workflows/ |
| 70 | Cloud Qdrant | Qdrant Cloud migration | config.py |

---

## Cheat Sheet — Quick Reference

### Start the System

```bash
# First time setup
cp .env.example .env && nano .env   # fill in API keys
docker compose up --build           # start all services

# Development (hot reload)
docker compose up -d                # start in background
# Edit files → changes apply automatically via volume mounts

# View logs
docker compose logs -f api          # watch API logs
docker compose logs -f streamlit-app

# Stop
docker compose down                 # stop all services
docker compose down -v              # stop + delete volumes (resets DB)
```

### Services and Ports

```
localhost:8501  → Streamlit chat UI
localhost:8000  → FastAPI backend
localhost:8000/docs → Swagger API docs
localhost:6333  → Qdrant vector DB
localhost:6333/dashboard → Qdrant UI
localhost:5433  → PostgreSQL (external port, internal is 5432)
localhost:8001  → Items MCP server
localhost:8002  → Reviews MCP server
```

### Key Environment Variables

```bash
OPENAI_API_KEY=sk-...           # required: embeddings + GPT-4.1
GROQ_API_KEY=gsk_...            # required: Llama fallback
GOOGLE_API_KEY=AIza...          # required (optional for basic setup)
LANGCHAIN_API_KEY=ls__...       # required: LangSmith tracing + evals
LANGCHAIN_TRACING_V2=true       # required: enable LangSmith
LANGCHAIN_PROJECT=...           # project name in LangSmith
QDRANT_API_KEY=...              # only for Qdrant Cloud
```

### Key API Calls

```bash
# POST a question (manual test)
curl -N -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query": "Show me Bluetooth headphones", "thread_id": "test-001"}'

# Submit feedback
curl -X POST http://localhost:8000/submit_feedback \
  -H "Content-Type: application/json" \
  -d '{"feedback_score": 1, "feedback_text": "", "trace_id": "...", "thread_id": "test-001", "feedback_source_type": "api"}'

# Run evals
make run-evals-retriever
```

### Agent Response Schemas

```python
# Coordinator
CoordinatorAgentResponse:
  next_agent: str    # which worker to call (or "" if done)
  plan: list[{agent, task}]
  final_answer: bool
  answer: str

# Workers (Product QA / Shopping Cart / Warehouse)
WorkerAgentResponse:
  answer: str
  final_answer: bool
  tool_calls: list[{name, arguments}]
  references: list[{id, description}]  # product QA only
```

### LangGraph State Fields

```python
State:
  messages        → full conversation history (appended with add)
  product_qa_agent.iteration    → how many times agent ran (max 4)
  product_qa_agent.final_answer → True when agent is done
  product_qa_agent.tool_calls   → tools agent wants to call
  coordinator_agent.next_agent  → which worker to route to
  coordinator_agent.plan        → multi-step work plan
  answer          → latest answer from any agent
  references      → products cited (appended with add)
  user_id         → user identifier (= thread_id)
  cart_id         → cart identifier (= thread_id)
```

### Prompt YAML Format

```yaml
# apps/api/src/api/agents/prompts/some_agent.yaml
metadata:
  name: Agent Name
  version: 1.0.0
  description: What this prompt does
  author: Aurimas Griciunas

prompts:
  gpt-4.1: |
    You are a shopping assistant...
    {{ variable_name }}              # Jinja2 substitution
    {{ list | tojson }}              # converts Python list to JSON string
  
  groq/llama-3.3-70b-versatile: |
    Same content, different phrasing...
```

### Ragas Eval Metrics Quick Reference

```
Context Precision  = signal / noise in retrieval       → want HIGH
Context Recall     = coverage of relevant items        → want HIGH
Faithfulness       = grounded in context (no hallucin) → want HIGH
Response Relevancy = answer addresses the question     → want HIGH

Run with: make run-evals-retriever
See in:   LangSmith → Datasets & Experiments → your project
```

---

## Summary

This project is a **complete, production-grade AI engineering reference implementation** built across 8 weeks. It demonstrates the full lifecycle of an AI system:

**Week 1-2 (Foundation):** Set up UV monorepo, Streamlit chatbot, FastAPI backend, Docker Compose. Implemented naive RAG with Qdrant vector DB. Added LangSmith observability and Ragas evaluations.

**Week 3-4 (Advanced RAG):** Structured outputs with Instructor, hybrid search (BM25 + RRF), Cohere reranking, YAML+Jinja2 prompt versioning. First LangGraph agents with ReAct pattern.

**Week 5-6 (Agentic System):** Multi-turn conversations with PostgreSQL state, second retrieval tool (reviews), LangSmith feedback, MCP servers with FastMCP, SSE streaming.

**Week 7-8 (Multi-Agent):** 3-worker multi-agent system (Product QA + Shopping Cart + Warehouse Manager) coordinated by a Coordinator Agent. Shopping cart (PostgreSQL), warehouse inventory with atomic reservations, LiteLLM model routing with fallback, Google ADK alternative, A2A protocol for remote agents, prompt caching.

**Production Readiness:** HITL (human-in-the-loop), CI eval pipeline, Qdrant Cloud migration, complete observability.

**Core insight:** The system evolves from `question → LLM → answer` to `question → plan → parallel agents → tools → databases → synthesize → stream answer`. Each step adds a specific capability that addresses a real limitation of the previous approach.

---

## Conclusion

### What Makes This System Production-Ready

1. **Observability:** Every function is @traceable → LangSmith shows every token, cost, latency
2. **Evaluations:** Ragas metrics run against LangSmith dataset → quantitative quality measurement
3. **Multi-model fallback:** If OpenAI fails → Groq kicks in → system stays up
4. **Atomic transactions:** Cart and warehouse use PostgreSQL transactions → no race conditions
5. **State persistence:** PostgreSQL checkpointing → conversation memory survives restarts
6. **Streaming:** SSE keeps users informed while long agent chains run
7. **Human feedback loop:** Thumbs up/down → LangSmith → identifies failure modes
8. **HITL:** Human confirmation for high-stakes actions (large reservations)
9. **CI eval gate:** Eval scores checked on every PR → prevents regressions
10. **Container-based:** Docker Compose → reproducible across dev/staging/prod

### What This Project Teaches You

| Concept | Where in Code |
|---|---|
| RAG pipeline | `retrieval_generation.py` |
| Hybrid search | `tools.py:retrieve_items_data()` |
| Structured outputs | `agents.py:ProductQAAgentResponse` |
| LangGraph state machine | `graph.py:StateGraph` |
| ReAct pattern | `graph.py:product_qa_agent_tool_edge` |
| Multi-agent coordination | `graph.py + agents.py:coordinator_agent` |
| PostgreSQL tools | `tools.py:add_to_shopping_cart()` |
| Atomic transactions | `tools.py:reserve_warehouse_items()` |
| Multi-turn memory | `graph.py:PostgresSaver` |
| SSE streaming | `graph.py:rag_agent_stream_wrapper()` |
| MCP protocol | `items_mcp_server/main.py` |
| LiteLLM routing | `agents.py:instructor.from_litellm(completion)` |
| Google ADK | `adk_warehouse_manager_agent/agent.py` |
| A2A protocol | `a2a_warehouse_manager_agent/` |
| Prompt versioning | `prompts/*.yaml + prompt_management.py` |
| Observability | `@traceable` decorators throughout |
| Evaluations | `evals/eval_retriever.py` |
| Feedback loop | `api/processors/submit_feedback.py` |

### The Main Takeaway

The progression from simple chatbot to multi-agent system is not just about adding features — it's about solving real problems:
- **Hallucination** → solved by RAG
- **Poor retrieval** → solved by hybrid search + reranking
- **One-turn only** → solved by PostgreSQL state checkpointing
- **Single LLM dependency** → solved by LiteLLM fallback
- **Complex multi-step tasks** → solved by multi-agent coordination
- **Black-box behavior** → solved by LangSmith observability
- **No quality guarantee** → solved by Ragas eval pipeline
- **Destructive actions without confirmation** → solved by HITL

Each solution introduces a new tool or pattern from the modern AI engineering toolkit. Together, they form a reference implementation that can be adapted to any domain — replace Amazon electronics with healthcare records, legal documents, or financial data, and the same architecture applies.
