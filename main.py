from fastapi import FastAPI, Depends
from pydantic import BaseModel
from auth import get_current_user
from team import run
import logging

app = FastAPI(title="Medical Records AI API", version="1.0")
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    response: str
    role_used: str
    patient_id: str | None

@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    user: dict = Depends(get_current_user)
):
    role       = user["role"]
    patient_id = user.get("patient_id")
    user_id    = user["user_id"]

    enriched_message = (
        f"[SYSTEM: Verified user. Role={role}. PatientID={patient_id or 'N/A'}]\n"
        f"User query: {request.message}"
    )

    logger.info(f"QUERY user={user_id} role={role} patient={patient_id} msg={request.message[:80]}")

    response_text = await run(enriched_message)

    return QueryResponse(
        response=response_text,
        role_used=role,
        patient_id=patient_id
    )

@app.get("/health")
async def health():
    return {"status": "ok"}