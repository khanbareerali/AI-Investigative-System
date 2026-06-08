# Document Matching App

Streamlit application for matching PDF crash reports against MCMIS CSV carrier data using fuzzy address scoring, with a RAG layer for natural-language Q&A over the evidence.

---

## What the app does

1. **Ingest** — load PDFs, CSVs, Word docs, and Excel files from Google Drive or local upload
2. **Extract** — pull carrier/owner fields from crash-report PDFs using regex patterns
3. **Match** — score every PDF carrier address against every CSV carrier address (deterministic fuzzy scoring, no AI required)
4. **Cluster** — group CSV records by normalized address and flag addresses with multiple carriers or USDOT numbers
5. **Ask** — use the RAG layer to ask natural-language questions about the evidence

---

## What the RAG layer does

The RAG (Retrieval-Augmented Generation) layer sits *on top of* the existing matching engine. It does **not** replace address matching.

It lets you ask questions like:

- Why was this match flagged?
- Which carriers share the same address?
- Which matches have different carrier names and different DOT numbers?
- Summarize the strongest high-confidence leads.
- Which matches need manual review?
- Show evidence for a specific carrier or VIN.

RAG retrieves relevant evidence chunks from a local ChromaDB vector index, then asks Claude to generate a grounded answer.

**RAG does NOT:**
- Replace the deterministic address-matching engine
- Prove chameleon-carrier behavior or insurance fraud
- Make legal or regulatory conclusions

All answers are investigative leads. Manual verification is always required.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create a `.env` file

Create a file named `.env` in the project root (next to `app.py`):

```
ANTHROPIC_API_KEY=your_key_here
```

Get your API key from [console.anthropic.com](https://console.anthropic.com).

The `.env` file is listed in `.gitignore` and will not be committed to version control.

### 3. (Optional) Configure Google Drive

To load files directly from a Google Drive folder:

1. Create a GCP service account and download the JSON key
2. Share your Drive folder with the service account email
3. Add to `.streamlit/secrets.toml`:
   ```toml
   GOOGLE_SERVICE_ACCOUNT_JSON = ".streamlit/your_key.json"
   ```

---

## Run the app

```bash
python -m streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

---

## Using the RAG tab (Tab 7 · Ask / RAG Evidence)

1. **Load data** — upload files on Tab 1 and run matching on Tab 4
2. **Build index** — go to Tab 7, click **Build / Refresh RAG Index**
   - The app chunks all loaded DataFrames (PDF records, CSV rows, matches, clusters) and stores them in a local ChromaDB database (`chroma_db/`)
3. **Ask a question** — type a question or click a suggested question button
4. **Review answer** — Claude's answer and the retrieved evidence chunks are displayed
5. **Explain a match** — select a match ID and click **Explain Selected Match** for a targeted explanation

### Example questions

- Why was the top match flagged?
- Which addresses have multiple carriers?
- Which matches have different USDOT numbers?
- Summarize evidence for 5433 S Clovis Ave.
- Which matches are high confidence?
- Show all CSV rows tied to a matched address.
- Summarize the strongest chameleon-carrier leads.

---

## Project structure

```
document_matching_app/
├── app.py                         # Streamlit entrypoint
├── requirements.txt
├── .env                           # ANTHROPIC_API_KEY (not committed)
├── .streamlit/
│   └── secrets.toml               # Google Drive service account path (not committed)
├── chroma_db/                     # ChromaDB local vector store (not committed)
└── src/
    ├── ingestion/
    │   ├── file_loader.py         # CSV / Excel loading
    │   ├── pdf_extractor.py       # PDF / DOCX / plain-text extraction
    │   └── drive_loader.py        # Google Drive API
    ├── matching/
    │   ├── normalizers.py         # Address / carrier name normalization
    │   ├── scoring.py             # Fuzzy address scoring
    │   ├── address_matcher.py     # Matching engine
    │   └── clustering.py         # Address cluster analysis
    ├── export/
    │   └── export_results.py      # CSV / Excel export
    └── rag/
        ├── chunk_builder.py       # DataFrame → text chunks
        ├── vector_store.py        # ChromaDB + local embeddings
        ├── retriever.py           # Semantic search helpers
        ├── claude_answerer.py     # Claude API integration
        └── rag_pipeline.py        # Orchestration: index + Q&A
```

---

## Notes

- Embeddings are generated **locally** using `sentence-transformers/all-MiniLM-L6-v2`. No embedding API key is needed.
- The Claude API key (`ANTHROPIC_API_KEY`) is only used for answer generation.
- The ChromaDB index is stored in `chroma_db/` inside the project directory. Delete this folder to reset all indexes.
- RAG answers cite chunk types, match IDs, CSV row IDs, and page numbers where available.
