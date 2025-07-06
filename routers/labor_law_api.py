from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
from rag_storage import RAGLegalStorage
from pydantic import BaseModel

router = APIRouter()
rag_storage = RAGLegalStorage()

class LawText(BaseModel):
    text: str
    metadata: Optional[Dict] = None

class LawUpdateRequest(BaseModel):
    text: str
    metadata: Optional[Dict] = None

class LawResponse(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict] = None
    distance: Optional[float] = None
    summary: Optional[str] = None
    created_at: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5

@router.post("/laws")
async def add_labor_law(law_input: LawText):
    try:
        law_id  = rag_storage.add_law(law_input.text, law_input.metadata)
        return {"id": law_id, "text": law_input.text, "metadata": law_input.metadata}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding labor law: {str(e)}"
        )

@router.get("/laws", response_model=List[LawResponse])
async def get_all_laws():
    try:
        laws = rag_storage.get_all_laws()
        return [LawResponse(
            id=law["id"], 
            text=law["text"], 
            summary=law['summary'],
            created_at=law['created_at'],
            metadata=law.get("metadata")
        ) for law in laws]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving laws: {str(e)}"
        )

# API to delete a law by ID
@router.delete("/laws/{law_id}")
async def delete_law(law_id: str):
    try:
        success = rag_storage.delete_law(law_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Law with ID {law_id} not found")
        return {"success": True, "id": law_id}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting law: {str(e)}"
        )

@router.put("/laws/{law_id}")
async def update_law(law_id: str, law_update: LawUpdateRequest):
    try:
        new_summary = rag_storage.update_law(law_id, law_update.text, law_update.metadata)
        if new_summary is None:
            raise HTTPException(status_code=404, detail=f"Law with ID {law_id} not found")
        return {"success": True, "id": law_id, "text": law_update.text, "metadata": law_update.metadata, "summary": new_summary}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating law: {str(e)}"
        )