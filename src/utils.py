import logging
import time
from uuid import uuid4
from typing import AsyncGenerator, AsyncIterator

from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from models import ChainResponse, ChainResponseChoices, Message


logger = logging.getLogger(__name__)


FALLBACK_EXCEPTION_MSG = (
    "Error from rag-server. Please check rag-server logs for more details."
)


def create_splitter(chunk_size: int, chunk_overlap: int) -> TextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )


async def generate_answer(
    generator: AsyncGenerator[str, None],
    model: str = "",
) -> AsyncIterator[str]:
    """Generate and stream the response to the provided prompt.

    Args:
        generator: Generator that yields response chunks
        contexts: List of context documents used for generation
        model: Name of the model used for generation
        collection_name: Name of the collection used for retrieval
        enable_citations: Whether to enable citations in the response
    """

    try:
        # unique response id for every query
        resp_id = str(uuid4())
        if generator:
            logger.debug("Generated response chunks\n")
            # Create ChainResponse object for every token generated
            async for chunk in generator:
                chain_response = ChainResponse()
                response_choice = ChainResponseChoices(
                    index=0,
                    message=Message(role="assistant", content=chunk),
                    delta=Message(role=None, content=chunk),
                    finish_reason=None,
                )
                chain_response.id = resp_id
                chain_response.choices.append(response_choice)  # pylint: disable=E1101
                chain_response.model = model
                chain_response.object = "chat.completion.chunk"
                chain_response.created = int(time.time())
                logger.debug(response_choice)
                yield "data: " + str(chain_response.model_dump_json()) + "\n\n"

            chain_response = ChainResponse()
            response_choice = ChainResponseChoices(
                finish_reason="stop",
            )
            chain_response.id = resp_id
            chain_response.choices.append(response_choice)  # pylint: disable=E1101
            chain_response.model = model
            chain_response.object = "chat.completion.chunk"
            chain_response.created = int(time.time())
            logger.debug(response_choice)
            yield "data: " + str(chain_response.model_dump_json()) + "\n\n"
        else:
            chain_response = ChainResponse()
            yield "data: " + str(chain_response.model_dump_json()) + "\n\n"

    except Exception as e:
        logger.error(
            "Error from generate endpoint. Error details: %s",
            e,
            exc_info=logger.getEffectiveLevel() <= logging.DEBUG,
        )
        for err in error_response_generator(FALLBACK_EXCEPTION_MSG):
            yield err


def error_response_generator(exception_msg: str):
    """
    Generate a stream of data for the error response
    """

    def get_chain_response(
        content: str = "",
        finish_reason: str | None = None,
    ) -> ChainResponse:
        """
        Get a chain response for an exception
        Args:
            exception_msg: str - Exception message
        Returns:
            chain_response: ChainResponse - Chain response for an exception
        """
        chain_response = ChainResponse()
        chain_response.id = str(uuid4())
        response_choice = ChainResponseChoices(
            index=0,
            message=Message(role="assistant", content=content),
            delta=Message(role=None, content=content),
            finish_reason=finish_reason,
        )
        chain_response.choices.append(response_choice)  # pylint: disable=E1101
        chain_response.object = "chat.completion.chunk"
        chain_response.created = int(time.time())
        return chain_response

    for i in range(0, len(exception_msg), 5):
        exception_msg_content = exception_msg[i : i + 5]
        chain_response = get_chain_response(content=exception_msg_content)
        yield "data: " + str(chain_response.model_dump_json()) + "\n\n"
    chain_response = get_chain_response(finish_reason="stop")
    yield "data: " + str(chain_response.model_dump_json()) + "\n\n"


# Helper function to escape JSON-like structures in content
def escape_json_content(content: str) -> str:
    """Escape curly braces in content to avoid JSON parsing issues"""
    return content.replace("{", "{{").replace("}", "}}")
