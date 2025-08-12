from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body, Depends
import io
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from document_processor_pydantic_ques import DocumentProcessor
from typing import Literal, List, Dict, Optional
from sqlalchemy.orm import Session
from database import get_db, AnalysisHistory, User
from auth import get_current_user

router = APIRouter()
doc_processor = DocumentProcessor()


class SummarizeRequest(BaseModel):
    ai_content: str

class QnARequest(BaseModel):
    report: str
    question: str

class ExportExcelRequest(BaseModel):
    processed_result: dict

@router.post("/export_excel")
async def export_excel_endpoint(
    request_body: ExportExcelRequest = Body(...)
):
    try:
        excel_bytes = doc_processor.export_to_excel(request_body.processed_result)
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": "attachment; filename=employee_data.xlsx"
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting to Excel: {str(e)}")

@router.post("/process")
async def process_documents(
    files: List[UploadFile] = File(...),
    doc_types: List[Literal["payslip", "contract", "attendance"]] = Form(...)
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
    contract_text: Optional[str] = Body(None),
    attendance_text: Optional[str] = Body(None),
    type: Optional[str] = Body(None),
    context: Optional[str] = Body(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict:
    try:
        if not payslip_text and not contract_text and not attendance_text:
            raise HTTPException(
                status_code=400,
                detail="At least one document text must be provided"
            )


        result = await doc_processor.create_report(
            payslip_text=payslip_text,
            contract_text=contract_text,
            attendance_text=attendance_text,
            analysis_type=type,
            context=context
        )

        # If result is a Pydantic model, convert to dict for FastAPI response validation
        if hasattr(result, 'dict') and callable(result.dict):
            result_dict = result.dict()
        else:
            result_dict = result

        # Save to AnalysisHistory
        analysis_type_value = type if type else "report"
        # Extract only the 'legal_analysis' part for storing
        analysis_content_to_store = ""
        if isinstance(result_dict, dict) and "legal_analysis" in result_dict:
            analysis_content_to_store = result_dict["legal_analysis"]
        elif isinstance(result_dict, str):  # Fallback if result is already just the analysis string
            analysis_content_to_store = result_dict
        # If result is a dict but doesn't have 'legal_analysis', it will store an empty string.
        # Consider if error handling or logging is needed here if the structure is unexpected.

        new_analysis = AnalysisHistory(
            analysis_type=analysis_type_value,
            analysis_result=analysis_content_to_store,  # Store only the analysis string
            user_id=current_user.id
        )
        db.add(new_analysis)
        db.commit()

        return result_dict
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error creating report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing documents: {str(e)}"
        )

@router.post("/summarise_analysis")
async def summarise_analysis_endpoint(
    request_body: SummarizeRequest = Body(...),
):
    try:
        summary = await doc_processor.summarise(ai_content_text=request_body.ai_content)
        return {"summary": summary}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# New QnA endpoint
@router.post("/qna")
async def qna_endpoint(
    request_body: QnARequest = Body(...)
):
    try:
        answer = await doc_processor.qna(
            report=request_body.report,
            questions=request_body.question
        )
        return {"answer": answer}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error in QnA: {str(e)}")

