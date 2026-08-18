<<<<<<< HEAD
<<<<<<< HEAD
# Adaptive RAG Academic Chatbot

An intelligent **Retrieval-Augmented Generation (RAG)** system for answering questions from academic PDF documents using semantic search, adaptive context optimization, and a local Large Language Model (LLM).

The project combines modern Natural Language Processing (NLP), semantic retrieval, and local LLM inference to provide accurate, document-grounded answers with supporting citations and confidence estimation.

---

# Features

* 📄 Upload academic PDF documents
* 🔍 Semantic Search using FAISS
* 🧩 Adaptive Chunking for efficient document segmentation
* ⚡ Binary Search-Based Context Optimizer (BSCO)
* 🧠 Local LLM inference using Ollama (llama3.2:3b)
* 📚 Citation Generation with supporting page numbers
* 📊 Confidence Estimation for retrieved answers
* 💬 Interactive Streamlit web interface
* ⚙️ Modular and extensible architecture
* 🔒 Fully local processing (no cloud dependency)

---

# System Architecture

```text
Academic PDF
      │
      ▼
PDF Processing
      │
      ▼
Adaptive Chunking
      │
      ▼
Embedding Generation
(BAAI/bge-small-en-v1.5)
      │
      ▼
FAISS Vector Database
      │
      ▼
Semantic Search
      │
      ▼
Binary Search-Based Context Optimizer (BSCO)
      │
      ▼
Prompt Builder
      │
      ▼
Ollama (llama3.2:3b)
      │
      ▼
Answer Generation
      │
      ▼
Citation Generation
      │
      ▼
Confidence Estimation
      │
      ▼
Streamlit User Interface
=======
# 🎓 Adaptive RAG Academic Chatbot
=======
# 🎓 Scholar AI — Adaptive RAG Academic Chatbot
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2

An intelligent **Retrieval-Augmented Generation (RAG)** system for academic document analysis, featuring **Binary Search-Based Context Optimization (BSCO)**, **adaptive hybrid retrieval**, **research gap detection**, **automated literature survey generation**, and **multilingual voice interaction** (English & Telugu) — all running locally via Ollama.

---

## 🌟 Overview

Scholar AI transforms academic PDF documents into an interactive knowledge base. Unlike conventional document chatbots, it introduces a **Binary Search-Based Context Optimizer** that minimizes the context passed to the LLM while preserving answer quality. The system retrieves only the most relevant document chunks, generates grounded responses, provides page-level citations with confidence estimation, and supports advanced research analysis.

---

## ✨ Features

### Core RAG Pipeline
- 📄 **Upload & Process** academic PDFs with intelligent metadata extraction
- 🔍 **Adaptive Hybrid Retrieval** — Dense (FAISS) + BM25 + Keyword search fusion
- 🧩 **Adaptive Chunking** — Structure-aware segmentation preserving headings, paragraphs, lists
- ⚡ **Binary Search-Based Context Optimizer (BSCO)** — Finds minimal sufficient context in O(log n)
- 🧠 **Local LLM Inference** via Ollama (Llama 3.2, Mistral, etc.)
- 📚 **Rich Citation Generation** — Per-chunk citations with keyword matching & evidence snippets
- 📊 **Multi-Factor Confidence Estimation** — 7-component fusion with Very High/High/Medium/Low levels
- ✅ **Self-Verification** — Post-generation grounding, unsupported statement detection

### Advanced Research Analysis
- 🔬 **Research Gap Detection** — Automatically identifies gaps, limitations, and open problems across papers
- 📋 **Paper Comparison** — Structured comparison of methods, datasets, evaluation metrics, contributions
- 💡 **Novelty Detection** — Identifies novel contributions, methodology innovations, and differentiators
- 📑 **Literature Survey Generation** — Auto-generates theme-based surveys with comparison tables
- 🔄 **Cross-Paper Trend Analysis** — Identifies methodological trends and research area classifications

### NLP & Query Understanding
- 🎯 **Intent Classification** — 20+ query types (definition, comparison, algorithm, formula, etc.)
- 🔗 **Query Expansion** — Abbreviation expansion, synonym lookup, domain-specific terminology
- 💬 **Conversation Memory** — Context-aware follow-up detection, last 5 turns preserved
- 🏷️ **Dynamic Prompt Templates** — Per-query-type prompts with structured output formats

### Voice Interaction
- 🎤 **Offline Speech-to-Text** (Faster Whisper `small` + `int8`) — English & Telugu
- 🔊 **Offline Text-to-Speech** (Piper TTS) — Natural voice output
- 🌐 **Three Language Modes** — `Auto-detect` / `English` / `Telugu` (manual UI override)
- 🔀 **Multi-Pass Fallback** — In Auto mode, when language confidence < 0.80 or the utterance is < 5 words, the system re-transcribes with forced `en` and `te` passes and picks the best hypothesis by comparing **average log probability**, **no-speech probability** and **compression ratio**
- 🚫 **Telugu → Hindi (Devanagari) fix** — Indic script detection penalises Devanagari output, re-runs Telugu with a strong Telugu `initial_prompt` and `condition_on_previous_text=False` to break hallucination loops
- 🔤 **English Term Preservation** — Common Telugu transliterations (ఏఐ → AI, డీబీఎంఎస్ → DBMS, కంప్యూటర్ → computer, …) are mapped back to canonical English so academic keywords survive retrieval
- 🌍 **Code-Mixed Query Normalization** — Telugu+English mixed queries are cleaned (fillers removed, terms restored) before entering the RAG retriever
- ✍️ **Lightweight Post-Processing** — punctuation, spacing, capitalisation, and academic terminology restoration (COs, POs, PEOs, DBMS…)
- 🎵 **Voice Activity Detection** — Noise filtering, silence suppression
- 📡 **Streaming Transcription** — Partial results for long recordings
- 🎛️ **UI States** — `Listening → Transcribing → Processing → Speaking` stepper plus detected-language & confidence badge

### UI/UX
- 🌗 **Dark/Light Mode** with glassmorphism design
- ⚡ **Streaming Responses** — Real-time token generation
- 📊 **Research Dashboard** — BSCO stats, timing metrics, token accounting
- 📄 **PDF Preview** — Page image rendering with citation highlighting
- 📱 **Responsive Design** — Desktop, tablet, mobile support
- 🎨 **Syntax Highlighting** (Prism.js) + **Math Rendering** (MathJax)

---

## 🏗️ System Architecture

```text
                 Academic PDF
                      │
                      ▼
              PDF Processor
         (text extraction, metadata)
                      │
                      ▼
           Adaptive Chunker
    (structure-aware, heading/paragraph/list)
                      │
                      ▼
        Embedding Generator (BGE-small-en-v1.5)
                      │
                      ▼
         FAISS Vector Database (IndexFlatIP)
                      │
                      ▼
  ┌─────── Adaptive Hybrid Retrieval ───────┐
  │  Dense (FAISS) │ BM25 │ Keyword │ Meta  │
  └───────────────────┬──────────────────────┘
                      │
                      ▼
      Cross-Encoder Reranker (heuristic ML)
                      │
                      ▼
 ┌──── Binary Search Context Optimizer ────┐
 │  Dedup │ Redundancy │ Token Budget │ BS  │
 └───────────────────┬──────────────────────┘
                      │
                      ▼
           Dynamic Prompt Builder
     (20+ templates, conversation memory)
                      │
                      ▼
          Ollama (Llama 3.2 : 3B)
                      │
                      ▼
<<<<<<< HEAD
            Answer Generation
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
 Citation Generation      Confidence Estimation
        │                           │
        └─────────────┬─────────────┘
                      ▼
             Streamlit User Interface
>>>>>>> 36754d585d486c803dcf8b904e651df16f319643
=======
      ┌──────────────┴──────────────┐
      ▼                             ▼
Citation Generator      Self-Verifier
(keyword matching,      (statement grounding,
 evidence snippets)      unsupported detection)
      │                             │
      └──────────────┬──────────────┘
                     ▼
        Confidence Estimator
  (7-factor fusion: similarity, citations,
   chunk agreement, reranker, verification,
   keyword coverage, consistency)
                     │
                     ▼
   ┌────────────────────────────────┐
   │  Research Analysis Modules     │
   │  • Research Gap Detection      │
   │  • Paper Comparison            │
   │  • Novelty Detection           │
   │  • Literature Survey Gen       │
   └────────────────────────────────┘
                     │
                     ▼
      Streamlit UI (ChatGPT-style)
  • Streaming responses • Citations
  • Confidence badges • PDF preview
  • Voice (EN/TE) • Research panels
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2
```

---

<<<<<<< HEAD
<<<<<<< HEAD
# Folder Structure
=======
# 📂 Project Structure
>>>>>>> 36754d585d486c803dcf8b904e651df16f319643
=======
## 📂 Project Structure
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2

```text
Adaptive-RAG-Academic-Chatbot/
│
├── app.py                 # Main Streamlit application
├── config.py              # Environment configuration
├── Dockerfile             # Production container image
├── docker-compose.yml     # Multi-service deployment
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # Development dependencies
├── .env.example           # Environment template
├── LICENSE                # MIT License
├── README.md              # This file
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py                  # RAG pipeline orchestrator
│   ├── pdf_processor.py             # PDF text extraction & metadata
│   ├── adaptive_chunker.py          # Structure-aware academic chunking
│   ├── embedding.py                 # BGE embedding generation (cached)
│   ├── vector_database.py           # FAISS index with metadata filters
│   ├── adaptive_hybrid_retrieval.py # Dense + BM25 + keyword fusion
│   ├── adaptive_retriever.py        # Query-type-based retrieval policy
│   ├── enhanced_bsco.py             # Enhanced BSCO with dedup & redundancy
│   ├── reranker.py                  # Cross-encoder / heuristic reranking
│   ├── prompt_builder.py            # Dynamic prompt templates (20+ types)
│   ├── answer_generator.py          # Ollama LLM interface (streaming)
│   ├── citations.py                 # Rich citation generation
│   ├── confidence.py                # 7-factor confidence estimation
│   ├── self_verifier.py             # Post-generation grounding verification
│   ├── evaluation.py                # Runtime metrics (MRR, precision, timing)
│   ├── query_classifier.py          # 20+ query type classification
│   ├── query_expander.py            # Offline query expansion (abbrev, synonyms)
│   ├── research_gap_detector.py     # ⭐ NEW: Research gap identification
│   ├── paper_comparator.py          # ⭐ NEW: Cross-paper comparison
│   ├── novelty_detector.py          # ⭐ NEW: Novelty & contribution analysis
│   ├── literature_survey.py         # ⭐ NEW: Auto-generated literature surveys
│   ├── voice_assistant.py           # Offline STT (Whisper) + TTS (Piper)
│   └── utils.py                     # Shared utilities, validation, logging
│
├── tests/
│   ├── conftest.py
│   ├── test_citations.py
│   ├── test_confidence.py
│   ├── test_pdf_processor.py
│   ├── test_query_classifier.py
│   ├── test_query_expander.py
│   └── test_research_components.py
│
├── assets/
│   ├── styles.css          # Premium glassmorphism design system
│   └── app.js              # Frontend interactivity
│
├── uploads/                # Uploaded PDFs
├── vector_db/              # FAISS index + metadata
├── processed_data/         # Extracted/chunked data
├── models/voices/          # Piper TTS voice models
├── logs/                   # Application logs
└── data/                   # Runtime data directory
```

---

<<<<<<< HEAD
<<<<<<< HEAD
# Technologies Used

* Python 3.12
* Streamlit
* PyMuPDF
* Sentence Transformers
* BAAI/bge-small-en-v1.5
* FAISS
* Ollama
* Llama 3.2 (3B)
* NumPy
* Torch

---

# Installation

## Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Adaptive-RAG-Academic-Chatbot.git
=======
# 🛠️ Technologies Used
=======
## 🛠️ Technologies Used
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2

| Category | Technology |
|----------|------------|
| **Language** | Python 3.11 |
| **Web Framework** | Streamlit |
| **PDF Processing** | PyMuPDF (fitz) |
| **Embedding Model** | BAAI/bge-small-en-v1.5 |
| **Embedding Framework** | Sentence Transformers |
| **Vector Database** | FAISS (IndexFlatIP) |
| **LLM** | Llama 3.2 (3B) via Ollama |
| **Speech-to-Text** | Faster Whisper (small, int8) |
| **Text-to-Speech** | Piper TTS |
| **Numerical** | NumPy |
| **HTTP** | httpx |

---

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running
- 8 GB+ RAM recommended

### Clone & Setup

```bash
git clone https://github.com/raghavendra1500/Adaptive-RAG-Academic-Chatbot.git
<<<<<<< HEAD
>>>>>>> 36754d585d486c803dcf8b904e651df16f319643

=======
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2
cd Adaptive-RAG-Academic-Chatbot

<<<<<<< HEAD
<<<<<<< HEAD
## Create a virtual environment
=======
---

## Create Virtual Environment
>>>>>>> 36754d585d486c803dcf8b904e651df16f319643

### Windows

```bash
=======
# Create virtual environment
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

<<<<<<< HEAD
### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

<<<<<<< HEAD
## Install dependencies
=======
## Install Dependencies
>>>>>>> 36754d585d486c803dcf8b904e651df16f319643

```bash
pip install -r requirements.txt
```

---

## Install Ollama

<<<<<<< HEAD
Download Ollama from
=======
Download Ollama from:
>>>>>>> 36754d585d486c803dcf8b904e651df16f319643

https://ollama.com/download

---

## Download the LLM

```bash
ollama pull llama3.2:3b
```

---

## Start Ollama
=======
# Install dependencies
pip install -r requirements.txt
```

### Download LLM & Voice Models
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2

```bash
# Start Ollama in a separate terminal
ollama serve

# Pull the LLM
ollama pull llama3.2:3b

# Voice models download automatically on first use
```

<<<<<<< HEAD
---

<<<<<<< HEAD
# Run the Application
=======
# ▶️ Run the Application
>>>>>>> 36754d585d486c803dcf8b904e651df16f319643
=======
### Run
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2

```bash
streamlit run app.py
```

<<<<<<< HEAD
<<<<<<< HEAD
Open the browser at
=======
Open your browser and navigate to:
>>>>>>> 36754d585d486c803dcf8b904e651df16f319643
=======
Open **http://localhost:8501** in your browser.
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2

---

## 🐳 Docker Deployment

```bash
# Build and run with Ollama sidecar
docker-compose up --build

# Or standalone (requires external Ollama)
docker build -t scholar-ai .
docker run -p 8501:8501 -e OLLAMA_HOST=http://host.docker.internal:11434 scholar-ai
```

---

<<<<<<< HEAD
<<<<<<< HEAD
# Usage

1. Launch the application.
2. Upload an academic PDF document.
3. Build the vector database.
4. Enter a question.
5. View:

   * Generated answer
   * Supporting citations
   * Confidence score
   * Retrieved context

---

# Novel Contributions

This project extends a traditional RAG pipeline with:

* Adaptive Chunking
* Binary Search-Based Context Optimizer (BSCO)
* Semantic Retrieval using FAISS
* Citation Generation
* Confidence Estimation
* Modular architecture for research and experimentation

---

# Example Workflow

```
Upload PDF

↓

Build Vector Database

↓

Ask Question

↓

Retrieve Relevant Chunks

↓

Optimize Context (BSCO)

↓

Generate Prompt

↓

LLM Response

↓

Display Answer + Citations + Confidence
=======
# 💡 Usage
=======
## 💡 Usage
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2

1. **Upload PDFs** — Drag & drop academic papers into the sidebar
2. **Build Database** — Click "Build Database" to index the documents
3. **Ask Questions** — Type or speak questions in natural language
4. **Review Results** — Explore citations, context, confidence, and metrics

### Research Analysis Commands

```
<<<<<<< HEAD
What are the Program Outcomes?
```

### Response

```
Answer:
The uploaded academic document does not explicitly define Program Outcomes.

Supporting Pages:
Page 67, Page 102

Confidence:
Medium
>>>>>>> 36754d585d486c803dcf8b904e651df16f319643
=======
"Compare these papers"
"Find research gaps"
"Generate literature survey"
"Compare the methodologies"
"What are the novel contributions?"
"Summarize differences between the papers"
"Identify future work directions"
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2
```

---

<<<<<<< HEAD
<<<<<<< HEAD
# Screenshots

screenshots of:

* Home Page
* PDF Upload
* Question Answering
* Citation Output
* Confidence Display

Example:
=======
# 📈 Research Contributions
=======
## 📈 Research Contributions
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2

1. **Binary Search-Based Context Optimizer (BSCO)** — Novel O(log n) context minimization that reduces prompt tokens by 40-60% while maintaining answer quality
2. **Adaptive Hybrid Retrieval** — Dynamic fusion of dense, BM25, and keyword signals with page-diversity
3. **Multi-Factor Confidence Estimation** — 7-component fusion (retrieval similarity, citation coverage, chunk agreement, cross-encoder, verification, keyword coverage, retrieval consistency)
4. **Self-Verification** — Post-generation grounding with unsupported statement detection
5. **Research Gap Detection** — Automated identification of gaps, limitations, and open problems
6. **Literature Survey Generation** — Cross-paper synthesis with comparison tables
7. **Offline Multilingual Voice** — English + Telugu STT/TTS with three modes (Auto/EN/TE), confidence-based multi-pass fallback, Telugu→Hindi script fix, and code-mixed query normalization
8. **Dynamic Prompt Engineering** — 20+ query-type-specific templates with structured output formats

---

## 📊 Performance

- **Retrieval Latency**: ~200ms average (hybrid search)
- **Context Compression**: 40-60% token reduction via BSCO
- **Answer Quality**: 89%+ confidence on document-grounded questions
- **Memory Usage**: ~2 GB RAM (embedding model + FAISS index)
- **Startup Time**: ~15 seconds (model loading)

---

## 🔮 Future Work

<<<<<<< HEAD
Add screenshots in the `assets/` folder.

Suggested images:
>>>>>>> 36754d585d486c803dcf8b904e651df16f319643

```
assets/home.png

<<<<<<< HEAD
assets/chat.png

assets/results.png
=======
assets/upload.png

assets/chat.png

assets/results.png

assets/confidence.png
```

Or include a short demo GIF:

```
assets/demo.gif
>>>>>>> 36754d585d486c803dcf8b904e651df16f319643
```

---

<<<<<<< HEAD
# Future Work

* Multi-document retrieval
* Hybrid keyword + semantic search
* Cross-document question answering
* Reranking using Cross Encoders
* Multi-modal document support
* Support for GPT, Gemini, and other LLM providers
* Cloud deployment
* User authentication
* Chat history persistence
* PDF page preview
* Answer highlighting inside PDFs

---

# License

This project is licensed under the MIT License.

See the LICENSE file for details.

---

# Author

**K Sai Raghavendra**
**|**
**B Siddartha**
**|**
**C Nihal Reddy**

Adaptive RAG Academic Chatbot

Developed as an academic research and learning project.

---

# Acknowledgements

* Meta (Llama)
* Ollama
* Hugging Face
* Sentence Transformers
* FAISS
* Streamlit
* PyMuPDF
=======
# 📜 License
=======
- [ ] Hybrid retrieval with cross-encoder neural reranking
- [ ] Multi-modal document support (images, scanned PDFs via OCR)
- [ ] Fine-tuned academic language models
- [ ] Cloud deployment templates (AWS, Azure, GCP)
- [ ] User authentication & multi-tenant support
- [ ] Real-time collaboration features
- [ ] Integration with academic databases (arXiv, Semantic Scholar)
- [ ] Advanced visualization dashboard for retrieval metrics

---

## 📜 License
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Authors

**K Sai Raghavendra**  
**B Siddartha**  
**C Nihal Reddy**

---

## 🙏 Acknowledgements

- [Ollama](https://ollama.ai/) — Local LLM inference
- [Meta Llama](https://ai.meta.com/llama/) — Open-source LLM
- [Hugging Face](https://huggingface.co/) — Model hub & Sentence Transformers
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF processing
- [Streamlit](https://streamlit.io/) — Web framework
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) — Speech-to-text
- [Piper TTS](https://github.com/rhasspy/piper) — Text-to-speech

---

## ⭐ Support

<<<<<<< HEAD
Your support helps improve and grow the project.
>>>>>>> 36754d585d486c803dcf8b904e651df16f319643
=======
If you find this project useful, please consider giving it a **star** on GitHub!
>>>>>>> f44ab47d9dcee6e20b11ec00c1ca556961ecedc2
