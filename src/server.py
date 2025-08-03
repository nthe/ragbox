import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile
from fastapi.requests import Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from models import Prompt
from rag import RagEngine, Documents
from settings import settings
from utils import generate_answer, create_splitter


logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app):
    rag = RagEngine(uri=settings.ragbox_db_path.as_posix())
    await rag.connect(name=settings.ragbox_db_table, schema=Documents)
    app.state.rag = rag
    yield


app = FastAPI(
    root_path="/v1",
    lifespan=lifespan,
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: Request, prompt: Prompt) -> StreamingResponse:
    rag: RagEngine = request.app.state.rag
    chat_history = [
        {"role": msg.role, "content": msg.content} for msg in prompt.messages
    ]
    last_message = chat_history.pop()
    chat_history = chat_history[-settings.ragbox_chat_history_limit:]
    query = last_message["content"]
    response_generator = rag.generate(
        query=query,
        chat_history=chat_history,
        use_knowledge_base=prompt.use_knowledge_base,
        temperature=prompt.temperature,
        top_p=prompt.top_p,
        max_completion_tokens=prompt.max_tokens,
        query_type=prompt.query_type,
    )
    stream = generate_answer(response_generator, model=settings.ragbox_chat_model)
    return StreamingResponse(stream, media_type="text/event-stream")


@app.post("/chat/completions")
async def chat_completion(request: Request, prompt: Prompt) -> StreamingResponse:
    return await generate(request, prompt)


@app.post("/ingest")
async def ingest_data(
    request: Request,
    chunk_size: int = 200,
    chunk_overlap: int = 0,
    files: list[UploadFile] | None = None,
):
    texts = []
    for f in files or []:
        binary = await f.read()
        text = binary.decode()
        texts.append(text)

    if not texts:
        logger.error("[server] no texts or files provided")
        return JSONResponse(
            {"detail": "no files or texts provided"},
            status_code=400,
        )

    logger.info(f"[server] about to parse {len(texts)} texts")

    text_splitter = create_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = list(map(text_splitter.split_text, texts))
    chunks = [t for txt in texts for t in txt]

    logger.info(f"[server] about to index {len(chunks)} chunks")
    rag: RagEngine = request.app.state.rag
    await rag.ingest(data=[{"text": text} for text in chunks])

    logger.info("[server] ingestion completed")
    table_stats: dict = await rag.table.stats()  # type: ignore
    return {
        "added": {
            "num_texts": len(texts),
            "num_chunks": len(chunks),
        },
        "stats": {
            "rows": table_stats["num_rows"],
            "bytes": table_stats["total_bytes"],
        },
    }


@app.get("/stats")
async def stats(request: Request):
    rag: RagEngine = request.app.state.rag
    table_stats: dict = await rag.table.stats()  # type: ignore
    return {
        "rows": table_stats["num_rows"],
        "bytes": table_stats["total_bytes"],
    }


@app.delete("/db")
async def delete(request: Request):
    rag: RagEngine = request.app.state.rag
    await rag.table.delete("1=1")
    table_stats: dict = await rag.table.stats()  # type: ignore
    return {
        "rows": table_stats["num_rows"],
        "bytes": table_stats["total_bytes"],
    }
