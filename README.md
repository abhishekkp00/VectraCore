# VectraCore

A lightweight, high-performance, and zero-dependency **Vector Database Engine** built completely from scratch in Java, accompanied by a dynamic, interactive web visualizer. 

VectraCore implements production-grade index structures (HNSW) and space-partitioning search models (KD-Tree), enabling real-time semantic query processing and local RAG (Retrieval-Augmented Generation) pipelines.

---

## Why This Project?

Most modern search applications and RAG systems rely on black-box vector databases (such as Pinecone, Weaviate, or Chroma). This project was created to:
- **Demystify Indexing Algorithms**: Implement and visualize complex spatial search indexes side-by-side to understand execution latency and traversal differences.
- **Showcase a Zero-Dependency Stack**: Leverage native Java standard library APIs (`com.sun.net.httpserver`, `java.net.http.HttpClient`) without external libraries (like Jackson, Spring, or gRPC) to demonstrate core database engineering principles.
- **Bridge Backend Logic & Frontend Visualization**: Present vector spaces, HNSW layer transitions, and KD-Tree query boundaries visually in real-time.

---

## What VectraCore Does

### 1. Multi-Algorithm Vector Indexing
Run and compare three core search paradigms in parallel:
- **HNSW (Hierarchical Navigable Small World)**: A multi-layer probabilistic graph index yielding $O(\log N)$ average query complexity for approximate nearest neighbor (ANN) search.
- **KD-Tree (K-Dimensional Tree)**: An exact space-partitioning binary tree index optimized for low-to-medium dimensional queries.
- **Brute Force (Linear Scan)**: A baseline $O(N)$ exact search structure to calculate true nearest neighbors and measure index recall accuracy.

### 2. Distance Metrics
Computes proximity across three math operations:
- **Cosine Similarity**: Measures directional alignment, ideal for high-dimensional text embeddings.
- **Euclidean Distance**: Computes straight-line distance in Euclidean space.
- **Manhattan Distance**: Computes grid-based coordinates traversal.

### 3. Local RAG Pipeline & Document Ingestion
- Chunk and embed large documents locally via **Ollama** using the `nomic-embed-text` model.
- Store document chunks in the engine, search context queries using HNSW, and pass retrieved contexts to a local LLM (`llama3.2`) to generate answers.

---

## How VectraCore is Built

### Tech Stack
- **Backend**: Native Java SE (compiled class execution, asynchronous thread pools).
- **Frontend**: Vanilla HTML5, Canvas, and CSS3 with custom CSS variables and custom animations.
- **LLM/Embeddings**: Local Ollama runtime service via standard HTTP APIs.

### Project Directory Layout
```
VectraCore/
├── src/                      ← Java Database Engine (Source Root)
│   ├── Main.java             ← HTTP Router & Server Initialization
│   ├── VectorDB.java         ← Orchestrator for standard vector indexing
│   ├── DocumentDB.java       ← Orchestrator for chunked document indexing
│   ├── HNSW.java             ← Hierarchical Navigable Small World logic
│   ├── KDTree.java           ← K-Dimensional Tree traversal
│   ├── BruteForce.java       ← Linear scan nearest neighbors search
│   ├── DistanceMetrics.java  ← Cosine, Euclidean, and Manhattan metrics
│   ├── DistFn.java           ← Distance metric functional interface
│   ├── Pair.java             ← Search result container (distance, ID)
│   ├── VectorItem.java       ← In-memory model for standard vectors
│   ├── DocItem.java          ← In-memory model for document chunks
│   ├── HNSWNode.java         ← HNSW graph node adjacency representation
│   ├── KDNode.java           ← Binary KD-Tree node representation
│   ├── GraphInfo.java        ← Metadata container for HNSW visual representation
│   ├── NV.java & EV.java     ← Node/Edge representations for visualizers
│   ├── OllamaClient.java     ← HttpClient wrapper for Ollama API
│   └── JsonUtils.java        ← Lightweight custom parser for JSON responses
├── web/                      
│   └── index.html            ← Frontend dashboard & Canvas visualizer
└── README.md                 ← Project documentation
```

### Compiling and Running

1. **Start Ollama** (Ensure you have `nomic-embed-text` and `llama3.2` models pulled):
   ```bash
   ollama run nomic-embed-text
   ollama run llama3.2
   ```

2. **Compile VectraCore Backend**:
   ```bash
   javac -d bin src/*.java
   ```

3. **Start the Database Server**:
   ```bash
   java -cp bin Main
   ```

4. **Access the Dashboard**:
   Open a web browser and navigate to the printed server URL (default: `http://localhost:8080` or `http://localhost:8081`).
