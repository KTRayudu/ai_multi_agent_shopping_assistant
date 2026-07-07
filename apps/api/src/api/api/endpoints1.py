# from fastapi import APIRouter, Request, HTTPException
# from server.api.models1 import RAGRequest, RAGResponse
# import logging
# from server.agents.retrieval_generation1 import integrated_rag_pipeline

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# router = APIRouter()

# @router.post("/")
# async def amazon_product_assistant(request: Request, payload: RAGRequest) -> RAGResponse:
#     response = integrated_rag_pipeline(payload.query)
#     return RAGResponse(request_id=request.state.request_id, answer=response)

# api_router = APIRouter()
# api_router.include_router(router, prefix="/product_assistant", tags=["rag"])


from fastapi import Request, APIRouter
from api.api.models1 import RAGRequest, RAGResponse

from qdrant_client import QdrantClient
from api.agents.retrieval_generation1 import rag_pipeline

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(url="http://qdrant:6333")

rag_router = APIRouter()

@rag_router.post("/")
def rag(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:

    answer = rag_pipeline(payload.query, qdrant_client)

    return RAGResponse(
        request_id=request.state.request_id,
        answer=answer["answer"]
    )


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])