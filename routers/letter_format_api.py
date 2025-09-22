from fastapi import APIRouter, HTTPException
from typing import Dict
from letter_format import LetterFormatStorage
from pydantic import BaseModel

router = APIRouter()
format_storage = LetterFormatStorage()

class LetterFormatInput(BaseModel):
    content: str

class LetterFormatResponse(BaseModel):
    content: str

@router.post("/format", response_model=LetterFormatResponse)
async def add_letter_format(format_input: LetterFormatInput):
    try:
        # Trim input content
        content = format_input.content.strip() if format_input.content else format_input.content
        return format_storage.add_format(content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding letter format: {str(e)}"
        )

@router.get("/format", response_model=LetterFormatResponse)
async def get_format():
    try:
        format = format_storage.get_format()
        if not format:
            raise HTTPException(
                status_code=404,
                detail="Letter format not found"
            )
        return format
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting letter format: {str(e)}"
        )

@router.put("/format", response_model=LetterFormatResponse)
async def update_format(format_input: LetterFormatInput):
    try:
        # Trim input content
        content = format_input.content.strip() if format_input.content else format_input.content
        return format_storage.update_format(content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating letter format: {str(e)}"
        )

@router.delete("/format")
async def delete_format():
    try:
        if format_storage.delete_format():
            return {"message": "Letter format deleted successfully"}
        raise HTTPException(
            status_code=404,
            detail="Letter format not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting letter format: {str(e)}"
        )