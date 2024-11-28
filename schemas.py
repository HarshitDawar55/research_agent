from typing import List

from pydantic import BaseModel


class PaperInput(BaseModel):
    title: str
    abstract: str


class ListOfPapers(BaseModel):
    data: List[PaperInput]
    query: str


class AgentQuery(BaseModel):
    query: str
