# from pydantic import BaseModel, Field

# class ChatRequest(BaseModel):
#     provider: str
#     model_name: str
#     messages: list[dict]

# class ChatResponse(BaseModel):
#     message: str
    
# class RAGRequest(BaseModel):
#     query: str = Field(..., description="The query to search the RAG database on amazon products")

# class RAGResponse(BaseModel):
#     request_id: str = Field(..., description="The request id")
#     answer: str = Field(..., description="The answer to the query")
    
from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The answer to the query")   