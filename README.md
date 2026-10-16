# RAG Knowledge Assistant — Local AI Edition

A portfolio RAG application for uploading text-based PDFs and asking source-grounded questions. It uses local Sentence Transformers embeddings, PostgreSQL/pgvector semantic retrieval, BM25 keyword retrieval, reciprocal-rank fusion, an Ollama-hosted local chat model, and filename/page citations. No OpenAI API key or paid API calls are needed.

## Architecture

```text
PDF upload
  -> PyPDF page extraction
  -> page-aware text chunks
  -> Sentence Transformers model (local in API container)
  -> PostgreSQL + pgvector (384-dimensional vectors)

Question
  -> local query embedding -> pgvector semantic search ----+
  -> BM25 keyword retrieval --------------------------------+-> reciprocal-rank fusion
                                                            -> optional local Ollama reranker
                                                            -> Ollama answer grounded in context
                                                            -> source/page citations
```

The API and PostgreSQL remain in Docker Compose. Ollama runs as a native Mac application so it can use your Mac's resources; Docker reaches it at `host.docker.internal:11434`. The embedding model downloads from Hugging Face the first time it is needed, then the Compose volume caches it. After initial downloads, inference runs locally. You still need internet for installing packages/downloading models, but no paid inference service is called.

## Requirements

- macOS and VS Code
- Docker Desktop
- Ollama for macOS
- Internet during the initial package/model downloads

## 1. Set up Ollama on the Mac

1. Download and install Ollama from <https://ollama.com/download> and open the Ollama app.
2. Because the API is inside Docker while Ollama runs natively on macOS, Ollama must listen on an interface the container can reach. First **quit Ollama** from its menu-bar icon, then run this in Terminal:

   ```bash
   launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
   open -a Ollama
   ```

   This binding can expose Ollama's unauthenticated local API to other devices on your network. Use it only on a trusted/private network and keep the macOS firewall enabled. When you no longer need Docker-to-Ollama access, quit Ollama, run `launchctl unsetenv OLLAMA_HOST`, and reopen the app to return to the default local-only binding.
3. In a VS Code terminal, pull a small local chat model:

   ```bash
   ollama pull qwen2.5:3b
   ```

4. Test the model:

   ```bash
   ollama run qwen2.5:3b
   ```

   Type `Say hello in one sentence.` Wait for a reply, then type `/bye` to exit.
5. Confirm Ollama is listening locally:

   ```bash
   curl http://localhost:11434/api/tags
   ```

The chat model runs on the host Mac; do not add Ollama to Docker Compose for this setup. The host command `curl http://localhost:11434/api/tags` only verifies the Mac-side listener; later, use the container-side test in section 4 to verify Docker connectivity too.

## 2. Configure the project

Create your local environment file if you do not already have one:

```bash
cp .env.example .env
```

If you already have `.env`, edit it instead of overwriting it. Set these values:

```dotenv
OLLAMA_BASE_URL=http://host.docker.internal:11434
CHAT_MODEL=qwen2.5:3b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_DIMENSIONS=384
```

You can remove `OPENAI_API_KEY` from `.env`; this version does not read it. Never commit `.env`.

## 3. Important: change the existing vector dimension

The previous OpenAI embedding model used 1536 dimensions. `sentence-transformers/all-MiniLM-L6-v2` returns 384 dimensions, so the old `document_chunks.embedding` column must be rebuilt before the app starts indexing with the new model.

Check how many chunks are stored:

```bash
docker compose exec db psql -U rag_user -d rag_assistant -c "SELECT COUNT(*) AS stored_chunks FROM document_chunks;"
```

If you have no successfully indexed documents yet (as expected if every upload failed before committing), run:

```bash
docker compose exec db psql -U rag_user -d rag_assistant -c "DROP TABLE IF EXISTS document_chunks;"
```

This removes only the derived chunks/embeddings table, not the database or its volume. Any previously indexed PDFs must be uploaded again after the migration, because embeddings from different models cannot be reused. Keep your original PDF files so you can re-upload them. Do **not** run `docker compose down -v` for this migration; that would delete the entire database volume.

If `stored_chunks` is greater than zero and you need to retain that content, make sure the source PDFs are available to re-upload before dropping the table.

## 4. Rebuild and start the API/database

Make sure Docker Desktop and the Ollama app are running, then run from the project root:

```bash
docker compose up -d --build
```

Check containers:

```bash
docker compose ps
```

Check the API logs:

```bash
docker compose logs --tail=100 api
```

Test that Docker can reach Ollama:

```bash
docker compose exec api python -c "import os, requests; u=os.getenv('OLLAMA_BASE_URL','http://host.docker.internal:11434'); r=requests.get(u + '/api/tags', timeout=10); print('Ollama HTTP status:', r.status_code); print(r.text[:500])"
```

The request should return HTTP 200 and include the `qwen2.5:3b` model in the tags list. If connection fails, confirm Ollama is open and `curl http://localhost:11434/api/tags` works on the Mac host.

Open:
- Health: <http://localhost:8000/health>
- API docs: <http://localhost:8000/docs>

The first document upload may take longer because Sentence Transformers will download its embedding model into the API container's persistent `huggingface_cache` volume. The first Ollama response may also need to load the chat model into memory.

## 5. Upload a PDF and ask a question

1. In <http://localhost:8000/docs>, expand `POST /documents/upload`.
2. Click **Try it out**, choose a text-based PDF, and click **Execute**.
3. Expect a successful response reporting the filename, pages read, chunks added, and a success message.
4. Expand `POST /ask` and try a question answered by the PDF, for example:

   ```json
   {"question": "What are the main topics covered in this document?"}
   ```

5. Inspect the `answer` and `citations`. Each citation includes source filename, page, snippet, and a reciprocal-rank-fusion score.

For a visual interface, open a second terminal and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install streamlit requests
streamlit run frontend/app.py
```

Use the URL printed by Streamlit, usually <http://localhost:8501>.

## Run tests and evaluation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

Add PDF-specific questions to `evals/questions.json`, then run the evaluation script while the API is running:

```bash
python scripts/evaluate.py
```

The current harness reports expected-source hit rate, basic keyword coverage, citation count, and latency. Keyword coverage is a smoke test, not proof of factual accuracy.

## Environment settings

| Variable | Purpose | Default |
|---|---|---|
| `OLLAMA_BASE_URL` | Ollama service as reached from the API container | `http://host.docker.internal:11434` |
| `CHAT_MODEL` | Local Ollama chat model | `qwen2.5:3b` |
| `EMBEDDING_MODEL` | Local Sentence Transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_DEVICE` | Device used inside API container | `cpu` |
| `EMBEDDING_DIMENSIONS` | Embedding vector dimensions | `384` |
| `CHUNK_SIZE` | Chunk size in characters | `900` |
| `CHUNK_OVERLAP` | Overlap between chunks | `150` |
| `SEMANTIC_TOP_K` | Semantic candidates | `8` |
| `BM25_TOP_K` | Keyword candidates | `8` |
| `FINAL_TOP_K` | Chunks sent to the chat model | `5` |
| `MAX_COSINE_DISTANCE` | Semantic abstention threshold | `0.70` |
| `ENABLE_LLM_RERANKER` | Use another local Ollama call for reranking | `false` |

## Common issues

- **`Connection refused` to Ollama:** launch the Ollama app, then verify `curl http://localhost:11434/api/tags` on the Mac.
- **Ollama model not found:** run `ollama pull qwen2.5:3b` and make sure `CHAT_MODEL` matches the exact tag.
- **Embedding model download errors:** check internet access and API logs; Hugging Face access is needed for the first model download.
- **`expected 384 dimensions, not 1536` or similar:** verify `.env` contains `EMBEDDING_DIMENSIONS=384`, rebuild the old chunks table as described above, and restart the API.
- **Scanned PDF has no text:** this starter does not include OCR; use a text-based PDF or add OCR as a later feature.
- **Incorrect or missing answers:** inspect cited passages, improve chunking/retrieval, and add test questions before claiming quality or latency metrics.

## Safety and production notes

This is a local portfolio starter, not a publicly hardened service. Before exposing it, add authentication, authorization, rate limiting, upload validation and quotas, structured observability, migrations, and stronger evaluation. Keep `.env` and private documents out of GitHub. Model downloads are third-party artifacts; inspect their licences before distributing models with your application.
