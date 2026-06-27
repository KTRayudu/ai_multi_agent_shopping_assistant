from fastapi import FastAPI
import logging
from server.api.middleware1 import RequestIDMiddleware
from starlette.middleware.cors import CORSMiddleware
from server.api.endpoints1 import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(middleware_class=RequestIDMiddleware)

app.add_middleware(middleware_class=CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

app.include_router(api_router)
