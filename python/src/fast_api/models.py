from pydantic import BaseModel

class RunJobRequest(BaseModel):
    concept: str

class CreateConceptRequest(BaseModel):
    concept: str
    forms: list[str]
    false_positives: list[str] = []

class CreateConceptAndRunRequest(BaseModel):
    concept: str
    forms: list[str]
    false_positives: list[str] = []
