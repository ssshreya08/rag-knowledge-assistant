from contextlib import asynccontextmanager
import logging
from io import BytesIO
from pathlib import Path
import time

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, engine, get_db, initialize_database
from app.models import Chunk
from app.rag import embeddings, generate_answer, retrieve
from app.schemas import AskRequest, AskResponse, UploadResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title=settings.app_name,
    description="A document-grounded RAG assistant with hybrid retrieval and source citations.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    return {"name": settings.app_name, "status": "running", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database is unavailable") from exc


@app.get("/documents")
def list_documents(db: Session = Depends(get_db)) -> dict:
    rows = db.execute(
        select(Chunk.source, func.count(Chunk.id).label("chunks"), func.min(Chunk.page).label("first_page"), func.max(Chunk.page).label("last_page"))
        .group_by(Chunk.source)
        .order_by(Chunk.source.asc())
    ).all()
    return {"documents": [
        {"filename": row.source, "chunks": row.chunks, "first_page": row.first_page, "last_page": row.last_page}
        for row in rows
    ]}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)) -> UploadResponse:
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(file_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="PDF is too large. Maximum supported size is 20 MB.")

    try:
        reader = PdfReader(BytesIO(file_bytes))
        if reader.is_encrypted:
            raise HTTPException(status_code=400, detail="Password-protected PDFs are not supported yet.")
        page_documents = [
            Document(page_content=(page.extract_text() or "").strip(), metadata={"source": filename, "page": page_number})
            for page_number, page in enumerate(reader.pages, start=1)
        ]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read this PDF. Try another PDF file.") from exc

    page_documents = [document for document in page_documents if document.page_content]
    if not page_documents:
        raise HTTPException(status_code=400, detail="No extractable text was found. Scanned PDFs need OCR, which is not enabled in this starter version.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(page_documents)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks could be created from this PDF.")

    try:
        vectors = embeddings.embed_documents([document.page_content for document in chunks])
        # Re-uploading a file with the same filename replaces its existing chunks.
        db.execute(delete(Chunk).where(Chunk.source == filename))
        db.add_all([
            Chunk(
                content=document.page_content,
                source=filename,
                page=int(document.metadata.get("page", 1)),
                chunk_index=index,
                embedding=vector,
            )
            for index, (document, vector) in enumerate(zip(chunks, vectors))
        ])
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("PDF upload failed")
        raise HTTPException(
            status_code=502,
            detail="Local embedding or database operation failed. Check API logs and verify the local model can be loaded.",
        ) from exc

    return UploadResponse(
        filename=filename,
        pages_read=len(reader.pages),
        chunks_added=len(chunks),
        message="PDF processed and indexed successfully.",
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest, db: Session = Depends(get_db)) -> AskResponse:
    started = time.perf_counter()
    if not db.scalar(select(func.count(Chunk.id))):
        return AskResponse(
            answer="There are no documents in the knowledge base yet. Upload a text-based PDF first.",
            citations=[],
            latency_ms=round((time.perf_counter() - started) * 1000),
            retrieval_count=0,
        )

    try:
        candidates, _best_distance = retrieve(request.question.strip(), db)
        answer, citations = generate_answer(request.question.strip(), candidates)
    except Exception as exc:
        logger.exception("RAG request failed")
        raise HTTPException(
            status_code=502,
            detail="The RAG request failed. Check API logs, Ollama status, and local model availability.",
        ) from exc

    return AskResponse(
        answer=answer,
        citations=citations,
        latency_ms=round((time.perf_counter() - started) * 1000),
        retrieval_count=len(citations),
    )
