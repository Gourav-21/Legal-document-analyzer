from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body, Depends
import io
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from document_processor_pydantic_ques import DocumentProcessor
from typing import Literal, List, Dict, Optional
from sqlalchemy.orm import Session
from database import get_db, AnalysisHistory, User
from auth import get_current_user
import json

router = APIRouter()
doc_processor = DocumentProcessor()


class SummarizeRequest(BaseModel):
    ai_content: str

class QnARequest(BaseModel):
    report: str
    question: str

class ExportExcelRequest(BaseModel):
    processed_result: dict

class FixOCRRequest(BaseModel):
    ocr_content: str

class GenerateRuleRequest(BaseModel):
    rule_description: str

class SuggestParamsFormulasRequest(BaseModel):
    law_description: str


@router.post("/export_excel")
async def export_excel_endpoint(
    request_body: ExportExcelRequest = Body(...)
):
    try:
        excel_bytes = await doc_processor.export_to_excel(request_body.processed_result)
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
        result = await doc_processor.process_document(files, doc_types)
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
    payslip_text: List[dict] = Body(None),
    contract_text: dict = Body(None),
    attendance_text: List[dict] = Body(None),
    employee_text: dict = Body(None),
    type: Optional[str] = Body(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict:
    try:
        if not payslip_text and not contract_text and not attendance_text and not employee_text:
            raise HTTPException(
                status_code=400,
                detail="At least one document text must be provided"
            )
        print("Payslip Text:", payslip_text)
        print("Attendance Text:", attendance_text)
        print("Employee Text:", employee_text)
        print("Contract Text:", contract_text)
        result = await doc_processor.create_report_with_rule_engine(
            payslip_data=payslip_text,
            attendance_data=attendance_text,
            contract_data=contract_text,
            employee_data=employee_text,
            analysis_type=type
        )

        # Save to AnalysisHistory
        analysis_type_value = type if type else "rule_based"
        analysis_content_to_store = result.get("legal_analysis", "")

        new_analysis = AnalysisHistory(
            analysis_type=analysis_type_value,
            analysis_result=analysis_content_to_store,
            user_id=current_user.id
        )
        db.add(new_analysis)
        db.commit()

        return result
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


@router.post("/generate-rule")
async def generate_rule_endpoint(
    request_body: GenerateRuleRequest = Body(...),
) -> Dict:
    """Generate a machine-readable rule JSON from a natural language description.

    Requires authentication (current_user) and saves nothing to the DB — just returns the generated checks.
    """
    try:
        # Generate the rule using the document processor (loads dynamic params internally)
        generated_checks = await doc_processor.generate_ai_rule_checks(request_body.rule_description)
        return {"generated_checks": generated_checks}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error generating rule: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating rule: {str(e)}")


@router.post("/suggest-params-formulas")
async def suggest_params_formulas_endpoint(
    request_body: SuggestParamsFormulasRequest = Body(...),
):
    try:
        # Delegate to document processor for AI-powered suggestions
        suggestions = await doc_processor.suggest_params_formulas(request_body.law_description)
        return {"suggestions": suggestions}

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in parameter/formula suggestions endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating AI suggestions: {str(e)}"
        )

