from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel
from judgement import JudgementStorage # Assuming judgement.py is in the parent directory or PYTHONPATH

router = APIRouter()
judgement_storage = JudgementStorage()

class JudgementText(BaseModel):
    text: str

class JudgementResponse(BaseModel):
    id: str
    text: str

@router.post("/judgements", response_model=JudgementResponse)
async def add_judgement(judgement_input: JudgementText):
    try:
        return judgement_storage.add_judgement(judgement_input.text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding judgement: {str(e)}"
        )

@router.get("/judgements", response_model=List[JudgementResponse])
async def get_all_judgements():
    return judgement_storage.get_all_judgements()

@router.get("/judgements/{judgement_id}", response_model=JudgementResponse)
async def get_judgement(judgement_id: str):
    judgement = judgement_storage.get_judgement_by_id(judgement_id)
    if judgement:
        return judgement
    raise HTTPException(status_code=404, detail="Judgement not found")

@router.put("/judgements/{judgement_id}", response_model=JudgementResponse)
async def update_judgement(judgement_id: str, judgement_input: JudgementText):
    try:
        updated_judgement = judgement_storage.update_judgement(judgement_id, judgement_input.text)
        if updated_judgement:
            return updated_judgement
        raise HTTPException(status_code=404, detail="Judgement not found")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating judgement: {str(e)}"
        )

@router.delete("/judgements/{judgement_id}")
async def delete_judgement(judgement_id: str):
    try:
        if judgement_storage.delete_judgement(judgement_id):
            return {"message": "Judgement deleted successfully"}
        raise HTTPException(status_code=404, detail="Judgement not found")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting judgement: {str(e)}"
        )
