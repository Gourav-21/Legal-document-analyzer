from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel
from rag_storage import RAGLegalStorage

router = APIRouter()
rag_storage = RAGLegalStorage()

class JudgementText(BaseModel):
    text: str
    metadata: Optional[Dict] = None

class JudgementResponse(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict] = None
    distance: Optional[float] = None

@router.post("/judgements", response_model=JudgementResponse)
async def add_judgement(judgement_input: JudgementText):
    try:
        judgement_id = rag_storage.add_judgement(judgement_input.text, judgement_input.metadata)
        return {"id": judgement_id, "text": judgement_input.text, "metadata": judgement_input.metadata}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding judgement: {str(e)}"
        )

@router.get("/judgements", response_model=List[JudgementResponse])
async def get_all_judgements():
    try:
        judgements = rag_storage.get_all_judgements()
        return [JudgementResponse(
            id=judgement["id"], 
            text=judgement["text"], 
            created_at=judgement['created_at'],
            metadata=judgement.get("metadata")
        ) for judgement in judgements]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving judgements: {str(e)}"
        )

@router.get("/judgements/{judgement_id}", response_model=JudgementResponse)
async def get_judgement(judgement_id: str):
    try:
        judgements = rag_storage.get_all_judgements()
        judgement = next((j for j in judgements if j["id"] == judgement_id), None)
        if judgement:
            return JudgementResponse(
                id=judgement["id"], 
                text=judgement["text"], 
                created_at=judgement['created_at'],
                metadata=judgement.get("metadata")
            )
        raise HTTPException(status_code=404, detail="Judgement not found")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving judgement: {str(e)}"
        )

@router.put("/judgements/{judgement_id}", response_model=JudgementResponse)
async def update_judgement(judgement_id: str, judgement_input: JudgementText):
    try:
        if rag_storage.update_judgement(judgement_id, judgement_input.text, judgement_input.metadata):
            return {"id": judgement_id, "text": judgement_input.text, "metadata": judgement_input.metadata}
        raise HTTPException(status_code=404, detail="Judgement not found")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating judgement: {str(e)}"
        )

@router.delete("/judgements/{judgement_id}")
async def delete_judgement(judgement_id: str):
    try:
        if rag_storage.delete_judgement(judgement_id):
            return {"message": "Judgement deleted successfully"}
        raise HTTPException(status_code=404, detail="Judgement not found")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting judgement: {str(e)}"
        )
