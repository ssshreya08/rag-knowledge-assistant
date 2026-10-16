from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, description="Question to answer from uploaded documents")


class Citation(BaseModel):
    citation_id: str
    source: str
    page: int
    snippet: str
    retrieval_score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: int
    retrieval_count: int


class UploadResponse(BaseModel):
    filename: str
    pages_read: int
    chunks_added: int
    message: str
