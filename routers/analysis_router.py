from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from database import get_db, AnalysisHistory
from auth import get_current_user

router = APIRouter()

@router.get("/history", response_model=List[dict])
async def history(
    analysis_type: Optional[str] = Query(None),
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == current_user.id)
    
    if analysis_type:
        query = query.filter(AnalysisHistory.analysis_type == analysis_type)
    if from_date:
        query = query.filter(AnalysisHistory.created_at >= from_date)
    if to_date:
        query = query.filter(AnalysisHistory.created_at <= to_date)
        
    history = query.order_by(AnalysisHistory.created_at.desc()).all()
    
    return [{
        "id": record.id,
        "analysis_type": record.analysis_type,
        "analysis_result": record.analysis_result,
        "created_at": record.created_at
    } for record in history]