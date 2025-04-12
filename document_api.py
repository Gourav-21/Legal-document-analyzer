from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from document_processor import DocumentProcessor
from typing import Literal, List, Dict, Optional

router = APIRouter()
doc_processor = DocumentProcessor()

@router.post("/process")
async def process_documents(
    files: List[UploadFile] = File(...),
    doc_types: List[Literal["payslip", "contract"]] = Form(...)
) -> Dict:
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

@router.post("/report")
async def create_report(
    payslip_text: Optional[str] = Body(None),
    contract_text: Optional[str] = Body(None)
) -> Dict:
    try:
        if not payslip_text and not contract_text:
            raise HTTPException(
                status_code=400,
                detail="At least one document text must be provided"
            )
        
        result = doc_processor.create_report(
            payslip_text=payslip_text,
            contract_text=contract_text
        )
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing documents: {str(e)}"
        )
