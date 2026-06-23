from fastapi import Request, APIRouter
from fastapi.responses import StreamingResponse
from api.api.models import RAGRequest, RAGResponse, RAGUsedContext, FeedbackRequest, FeedbackResponse

from api.agents.graph import rag_agent_stream_wrapper
from api.api.processors.submit_feedback import submit_feedback
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()
feedback_router = APIRouter()

@rag_router.post("/")
def rag(
    request: Request,
    payload: RAGRequest
) -> StreamingResponse:

    return StreamingResponse(
        rag_agent_stream_wrapper(payload.query, payload.thread_id),
        media_type="text/event-stream"
    )


@feedback_router.post("/")
def send_feedback(
    request: Request,
    payload: FeedbackRequest
) -> FeedbackResponse:

    submit_feedback(payload.trace_id, payload.feedback_score, payload.feedback_text, payload.feedback_source_type)

    return FeedbackResponse(
        request_id=request.state.request_id,
        status="success"
    )


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/agent", tags=["agent"])
api_router.include_router(feedback_router, prefix="/submit_feedback", tags=["feedback"])