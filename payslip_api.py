from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List, Dict
import asyncio
from document_processor_pydantic import PydanticAIDocumentProcessor

app = FastAPI(title="PydanticAI + RAG Payslip Analyzer", version="1.0.0")

# Initialize the processor
processor = PydanticAIDocumentProcessor()

# Pydantic models for API
class LawRequest(BaseModel):
    title: str
    content: str
    year: Optional[str] = None
    category: Optional[str] = None

class JudgementRequest(BaseModel):
    case_name: str
    content: str
    year: Optional[str] = None
    court: Optional[str] = None

class PayslipAnalysisRequest(BaseModel):
    payslip_text: str
    analysis_type: str = "easy"  # easy, professional, report, table, claim
    context: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "PydanticAI + RAG Payslip Analysis System",
        "status": "operational",
        "model": processor.model_type
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    laws = processor.get_all_laws()
    judgements = processor.get_all_judgements()
    
    return {
        "total_laws": len(laws),
        "total_judgements": len(judgements),
        "model_type": processor.model_type,
        "rag_status": "operational"
    }

@app.post("/laws")
async def add_law(law: LawRequest):
    """Add a labor law to the system"""
    try:
        metadata = {"title": law.title}
        if law.year:
            metadata["year"] = law.year
        if law.category:
            metadata["category"] = law.category
            
        law_id = processor.add_law(law.content, metadata)
        
        return {
            "message": "Law added successfully",
            "law_id": law_id,
            "title": law.title
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding law: {str(e)}")

@app.post("/judgements")
async def add_judgement(judgement: JudgementRequest):
    """Add a legal judgement to the system"""
    try:
        metadata = {"case_name": judgement.case_name}
        if judgement.year:
            metadata["year"] = judgement.year
        if judgement.court:
            metadata["court"] = judgement.court
            
        judgement_id = processor.add_judgement(judgement.content, metadata)
        
        return {
            "message": "Judgement added successfully",
            "judgement_id": judgement_id,
            "case_name": judgement.case_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding judgement: {str(e)}")

@app.post("/analyze")
async def analyze_payslip(request: PayslipAnalysisRequest):
    """Analyze a payslip using PydanticAI with RAG context"""
    try:
        # Validate analysis type
        valid_types = ["easy", "professional", "report", "table", "claim"]
        if request.analysis_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid analysis type. Must be one of: {valid_types}"
            )
        
        # Perform analysis
        result = await processor.create_report(
            payslip_text=request.payslip_text,
            analysis_type=request.analysis_type,
            context=request.context
        )
        
        return {
            "analysis": result.legal_analysis,
            "relevant_laws_count": len(result.relevant_laws),
            "relevant_judgements_count": len(result.relevant_judgements),
            "relevant_laws": result.relevant_laws,
            "relevant_judgements": result.relevant_judgements,
            "status": result.status,
            "analysis_type": result.analysis_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing payslip: {str(e)}")

@app.post("/analyze/file")
async def analyze_payslip_file(
    file: UploadFile = File(...),
    analysis_type: str = "easy",
    context: Optional[str] = None
):
    """Analyze a payslip from uploaded file"""
    try:
        # Process the uploaded file
        file_content = await file.read()
        
        # Extract text from file
        extracted_text = processor._extract_text(file_content, file.filename)
        
        # Perform analysis
        result = await processor.create_report(
            payslip_text=extracted_text,
            analysis_type=analysis_type,
            context=context
        )
        
        return {
            "filename": file.filename,
            "extracted_text": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
            "analysis": result.legal_analysis,
            "relevant_laws_count": len(result.relevant_laws),
            "relevant_judgements_count": len(result.relevant_judgements),
            "status": result.status,
            "analysis_type": result.analysis_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")

@app.post("/search/laws")
async def search_laws(request: SearchRequest):
    """Search for relevant laws"""
    try:
        results = processor.search_laws(request.query, request.limit)
        return {
            "query": request.query,
            "results_count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching laws: {str(e)}")

@app.post("/search/judgements")
async def search_judgements(request: SearchRequest):
    """Search for relevant judgements"""
    try:
        results = processor.search_judgements(request.query, request.limit)
        return {
            "query": request.query,
            "results_count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching judgements: {str(e)}")

@app.get("/laws")
async def get_all_laws():
    """Get all laws in the system"""
    try:
        laws = processor.get_all_laws()
        return {
            "total_count": len(laws),
            "laws": laws
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving laws: {str(e)}")

@app.get("/judgements")
async def get_all_judgements():
    """Get all judgements in the system"""
    try:
        judgements = processor.get_all_judgements()
        return {
            "total_count": len(judgements),
            "judgements": judgements
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving judgements: {str(e)}")

# Example usage endpoint
@app.get("/examples")
async def get_examples():
    """Get example requests for testing the API"""
    return {
        "add_law_example": {
            "title": "חוק שעות עבודה ומנוחה",
            "content": "העובד זכאי לתשלום עבור שעות נוספות: 125% עבור השעתיים הראשונות ו-150% עבור השעות הנוספות.",
            "year": "1951",
            "category": "שעות עבודה"
        },
        "add_judgement_example": {
            "case_name": "פלוני נ. חברת XYZ - שעות נוספות",
            "content": "בית הדין קבע כי המעסיק חייב לשלם עבור שעות נוספות. העובד זכה לפיצוי של 15,000 ש״ח.",
            "year": "2023",
            "court": "בית דין אזורי תל אביב"
        },
        "analyze_payslip_example": {
            "payslip_text": "תלוש שכר: שכר יסוד 6000 ש״ח, שעות נוספות 12 שעות: 0 ש״ח, פנסיה: 0 ש״ח",
            "analysis_type": "easy",
            "context": "בדיקת התאמה לחוקי עבודה"
        },
        "search_example": {
            "query": "שעות נוספות",
            "limit": 5
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting PydanticAI + RAG Payslip Analysis API...")
    print("📖 API Documentation available at: http://localhost:8080/docs")
    print("🔍 Example endpoints: http://localhost:8080/examples")
    uvicorn.run(app, host="0.0.0.0", port=8080)
