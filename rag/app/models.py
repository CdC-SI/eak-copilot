from pydantic import BaseModel

class ResponseBody(BaseModel):
    content: str

class RAGRequest(BaseModel):
    query: str