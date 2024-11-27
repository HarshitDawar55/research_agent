from pydantic import BaseModel


class PaperInput(BaseModel):
    title: str
    abstract: str
    query: str


class ListOfPapers(BaseModel):
    data: PaperInput
    query: str


class AgentQuery(BaseModel):
    query: str
