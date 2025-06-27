from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
from rag_storage import RAGLegalStorage
from pydantic import BaseModel

router = APIRouter()
rag_storage = RAGLegalStorage()

class LawText(BaseModel):
    text: str
    metadata: Optional[Dict] = None

class LawResponse(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict] = None
    distance: Optional[float] = None
    summary: Optional[str] = None
    similarity_score: Optional[float] = None

class LawWithSummaryResponse(BaseModel):
    id: str
    text: str
    summary: str
    metadata: Optional[Dict] = None

class BulkSummaryResponse(BaseModel):
    updated_count: int
    message: str

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

@router.post("/laws/with-summary")
async def add_labor_law_with_summary(law_input: LawText):
    """Add a labor law with AI-generated summary"""
    try:
        law_id = rag_storage.add_law_with_summary(law_input.text, law_input.metadata)
        # Get the law back with summary
        laws_with_summaries = rag_storage.get_laws_with_summaries()
        added_law = next((law for law in laws_with_summaries if law['id'] == law_id), None)
        
        if added_law:
            return {
                "id": law_id, 
                "text": law_input.text, 
                "metadata": law_input.metadata,
                "summary": added_law['summary'],
                "ai_enabled": rag_storage.ai_enabled
            }
        else:
            return {"id": law_id, "text": law_input.text, "metadata": law_input.metadata}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding labor law with summary: {str(e)}"
        )

@router.get("/laws", response_model=List[LawResponse])
async def get_all_laws():
    try:
        laws = rag_storage.get_all_laws()
        return [LawResponse(
            id=law["id"], 
            text=law["text"], 
            metadata=law.get("metadata")
        ) for law in laws]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving laws: {str(e)}"
        )

@router.post("/laws/search", response_model=List[LawResponse])
async def search_laws(search_request: SearchRequest):
    try:
        results = rag_storage.search_laws(search_request.query, search_request.n_results)
        return [LawResponse(
            id=result["id"], 
            text=result["text"], 
            metadata=result.get("metadata"),
            distance=result.get("distance")
        ) for result in results]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching laws: {str(e)}"
        )

@router.post("/laws/search-by-summary", response_model=List[LawResponse])
async def search_laws_by_summary(search_request: SearchRequest):
    """Search laws using AI-generated summaries for faster, more targeted results"""
    try:
        results = rag_storage.search_laws_by_summary(search_request.query, search_request.n_results)
        return [LawResponse(
            id=result["id"], 
            text=result["text"], 
            metadata=result.get("metadata"),
            summary=result.get("summary"),
            similarity_score=result.get("similarity_score")
        ) for result in results]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching laws by summary: {str(e)}"
        )

@router.get("/laws/with-summaries", response_model=List[LawWithSummaryResponse])
async def get_laws_with_summaries():
    """Get all laws that have AI-generated summaries"""
    try:
        laws = rag_storage.get_laws_with_summaries()
        return [LawWithSummaryResponse(
            id=law["id"], 
            text=law["text"], 
            summary=law["summary"],
            metadata=law.get("metadata")
        ) for law in laws]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving laws with summaries: {str(e)}"
        )

@router.post("/laws/bulk-add-summaries", response_model=BulkSummaryResponse)
async def bulk_add_summaries():
    """Add AI-generated summaries to all laws that don't have them"""
    try:
        if not rag_storage.ai_enabled:
            raise HTTPException(
                status_code=400,
                detail="AI summarization is not enabled. Please configure GEMINI_API_KEY."
            )
        
        updated_count = rag_storage.bulk_add_summaries_to_existing_laws()
        return BulkSummaryResponse(
            updated_count=updated_count,
            message=f"Successfully added AI summaries to {updated_count} laws"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding bulk summaries: {str(e)}"
        )

@router.put("/laws/{law_id}/add-summary")
async def add_summary_to_law(law_id: str):
    """Add AI-generated summary to a specific law"""
    try:
        if not rag_storage.ai_enabled:
            raise HTTPException(
                status_code=400,
                detail="AI summarization is not enabled. Please configure GEMINI_API_KEY."
            )
        
        success = rag_storage.add_summary_to_existing_law(law_id)
        if success:
            return {"message": "Summary added successfully", "law_id": law_id}
        else:
            raise HTTPException(
                status_code=404,
                detail="Law not found or summary could not be generated"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding summary to law: {str(e)}"
        )

@router.get("/laws/ai-status")
async def get_ai_status():
    """Get the status of AI summarization capabilities"""
    return {
        "ai_enabled": rag_storage.ai_enabled,
        "total_laws": len(rag_storage.get_all_laws()),
        "laws_with_summaries": len(rag_storage.get_laws_with_summaries()),
        "message": "AI summarization is enabled" if rag_storage.ai_enabled else "AI summarization is disabled. Configure GEMINI_API_KEY to enable."
    }