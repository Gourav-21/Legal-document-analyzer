from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from document_processor import DocumentProcessor
from typing import Literal, List

router = APIRouter()
doc_processor = DocumentProcessor()

@router.post("/analyze")
async def analyze_documents(
    files: List[UploadFile] = File(...),
    doc_types: List[Literal["payslip", "contract"]] = Form(...)
):
    try:
        result = doc_processor.process_document(files, doc_types)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing documents: {str(e)}"
        )