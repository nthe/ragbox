from __future__ import annotations

import logging
from typing import AsyncGenerator, Literal

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.index import FTS
from litellm import acompletion

from settings import settings


logger = logging.getLogger("uvicorn")


embeddings = (
    get_registry()
    .get("sentence-transformers")
    .create(
        name=settings.ragbox_embedding_model,
        device=settings.ragbox_embedding_model_device,
    )
)


class Documents(LanceModel):  # type: ignore
    vector: Vector(embeddings.ndims()) = embeddings.VectorField()  # type: ignore
    text: str = embeddings.SourceField()


type SearchQueryType = Literal[
    "auto",
    "fts",
    "vector",
    "hybrid",
]


CHAT_TEMPLATE = """
    You are a helpful, respectful, and honest assistant.
    Your answers must follow these strict guidelines:
    1. Answer concisely and directly.
    2. Focus only on what was asked — no extra commentary, no assumptions.
    3. Avoid giving multiple options, lists, or examples unless explicitly requested.
    4. Do not explain your reasoning unless asked.
    5. Keep responses brief but accurate.
    6. Use natural, conversational tone — clear and human, not robotic.
    7. Make sure your response are strictly one sentence or less unless it really needs to be longer.
    8. Do not mention this instructions in your response.

    Make sure above rules are strictly followed.
"""

RAG_TEMPLATE = """
    You are a helpful AI assistant named Envie.
    You must answer only using the information provided in the context. While answering you must follow the instructions given below.

    <instructions>
    1. Do NOT use any external knowledge.
    2. Do NOT add explanations, suggestions, opinions, disclaimers, or hints.
    3. NEVER say phrases like “based on the context”, “from the documents”, or “I cannot find”.
    4. NEVER offer to answer using general knowledge or invite the user to ask again.
    5. Do NOT include citations, sources, or document mentions.
    6. Answer concisely. Use short, direct sentences by default. Only give longer responses if the question truly requires it.
    7. Do not mention or refer to these rules in any way.
    8. Do not ask follow-up questions.
    9. Do not mention this instructions in your response.
    </instructions>

    Context:
    {context}

    Make sure the response you are generating strictly follow the rules mentioned above i.e. never say phrases like “based on the context”, “from the documents”, or “I cannot find” and mention about the instruction in response.
"""


class RagEngine:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.table: lancedb.AsyncTable
        self.connection: lancedb.AsyncConnection

    async def connect(
        self,
        *,
        name: str,
        schema: type[LanceModel],
    ) -> None:
        self.connection = await lancedb.connect_async(self.uri)
        self.table = await self.connection.create_table(
            name,
            schema=schema,
            mode="create",
            exist_ok=True,
        )

    async def ingest(
        self,
        data: list[dict],
    ) -> None:
        await self.table.add(data)
        await self.table.create_index("text", replace=True, config=FTS())
        if (await self.table.count_rows()) > 255:
            await self.table.create_index("vector", replace=True)

    async def search(
        self,
        query: str,
        query_type: SearchQueryType = "auto",
        db_limit: int = 40,
        rerank_limit: int = 10,
    ) -> list[dict]:
        lancedb_query = await self.table.search(
            query,
            query_type=query_type,
        )
        if query_type in {"fts", "vector"}:
            results = lancedb_query
        else:
            results = lancedb_query.limit(limit=db_limit).rerank()
        docs = await results.limit(limit=rerank_limit).to_list()
        return docs

    async def generate(
        self,
        *,
        query: str,
        chat_history: list[dict],
        use_knowledge_base: bool,
        query_type: SearchQueryType = "auto",
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_completion_tokens: int = 1000,
    ) -> AsyncGenerator[str, None]:
        messages = chat_history
        from time import time

        start = time()
        if use_knowledge_base:
            docs = await self.search(query, query_type=query_type)
            context = "\n".join([doc["text"] for doc in docs])
            prompt = RAG_TEMPLATE.format(context=context)
            messages += [{"role": "user", "content": prompt}]
            messages += [{"role": "user", "content": query}]

        else:
            prompt = CHAT_TEMPLATE
            messages += [{"role": "user", "content": prompt}]
            messages += [{"role": "user", "content": query}]

        duration = time() - start
        logger.info(f"[rag] search {duration=}")

        start = time()
        response = await acompletion(
            model=settings.ragbox_chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_completion_tokens,
            top_p=top_p,
            stream=True,
        )

        index = 0
        async for chunk in response:  # type: ignore
            if index == 0:
                duration = time() - start
                logger.info(f"[rag] ttft {duration=}")
                index += 1

            token = chunk.choices[0].delta.content or ""
            yield token

        duration = time() - start
        logger.info(f"[rag] generation {duration=}")
