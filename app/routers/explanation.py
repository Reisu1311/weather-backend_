from fastapi import APIRouter

router = APIRouter()

@router.get("/explain")
async def explanation():
    return {
        "message": "Gunakan endpoint /predict "
                   "untuk mendapatkan LIME explanation."
    }