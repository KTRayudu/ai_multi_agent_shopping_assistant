from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RequestIDMiddleware(BaseHTTPMiddleware):
    """ Middleware that adds a unique request ID to each request """
    async def dispatch(self, request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        logger.info(f"Request ID: {request.state.request_id} received at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(f"Request ID: {request.state.request_id} , {request.url.path} completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return response
    
