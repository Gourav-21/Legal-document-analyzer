from fastapi import APIRouter, HTTPException
from typing import List, Dict
from labour_law import LaborLawStorage
from pydantic import BaseModel

router = APIRouter()
law_storage = LaborLawStorage()

class LawText(BaseModel):
    text: str

class LawResponse(BaseModel):
    id: str
    text: str

@router.post("/laws")
async def add_labor_law(law_input: LawText):
    try:
        return law_storage.add_law(law_input.text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding labor law: {str(e)}"
        )

@router.get("/laws", response_model=List[LawResponse])
async def get_all_laws():
    return law_storage.get_all_laws()

@router.delete("/laws/{law_id}")
async def delete_law(law_id: str):
    try:
        if law_storage.delete_law(law_id):
            return {"message": "Law deleted successfully"}
        raise HTTPException(
            status_code=404,
            detail="Law not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting labor law: {str(e)}"
        )

@router.put("/laws/{law_id}")
async def update_law(law_id: str, law_input: LawText):
    try:
        updated_law = law_storage.update_law(law_id, law_input.text)
        if updated_law:
            return updated_law
        raise HTTPException(
            status_code=404,
            detail="Law not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating labor law: {str(e)}"
        )