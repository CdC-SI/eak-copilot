from pydantic import BaseModel


class FaqItem(BaseModel):
    id: int = None
    question: str
    answer: str
    url: str
    language: str
