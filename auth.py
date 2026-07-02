from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Simplified role mapping — API key → role/patient_id
# In real production this would be Azure AD; here it simulates verified identity
API_KEY_MAP = {
    "key-doctor-001": {"role": "doctor", "patient_id": None},
    "key-patient-p002": {"role": "patient", "patient_id": "P002"},
    "key-pharmacist-001": {"role": "pharmacist", "patient_id": None},
}

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    user = API_KEY_MAP.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or unrecognized API key")
    return {"user_id": token, "role": user["role"], "patient_id": user["patient_id"]}