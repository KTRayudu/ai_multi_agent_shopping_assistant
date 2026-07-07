from fastapi import APIRouter, Request, HTTPException
from server.api.models1 import RAGRequest, RAGResponse
import logging
from server.agents.retrieval_generation1 import integrated_rag_pipeline

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/")
async def amazon_product_assistant(request: Request, payload: RAGRequest) -> RAGResponse:
    response = integrated_rag_pipeline(payload.query)
    return RAGResponse(request_id=request.state.request_id, answer=response)

api_router = APIRouter()
api_router.include_router(router, prefix="/product_assistant", tags=["rag"])
