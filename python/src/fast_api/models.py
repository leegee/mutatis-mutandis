from pydantic import BaseModel

class RunJobRequest(BaseModel):
    concept: str

class CreateConceptRequest(BaseModel):
    name: str
    forms: list[str]
    false_positives: list[str] = []

class CreateConceptAndRunRequest(BaseModel):
    name: str
    forms: list[str]
    false_positives: list[str] = []
