from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
from rag_storage import RAGLegalStorage
from pydantic import BaseModel
import json
import os
import uuid

router = APIRouter()
rag_storage = RAGLegalStorage()

# --- CRUD for labor_law_rules.json ---
RULES_PATH = os.path.join(os.path.dirname(__file__), '../rules/labor_law_rules.json')

def load_rules():
    with open(RULES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_rules(data):
    with open(RULES_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@router.get('/labor-law-rules', response_model=dict)
def get_all_rules():
    return load_rules()

@router.get('/labor-law-rules/{rule_id}', response_model=dict)
def get_rule(rule_id: str):
    data = load_rules()
    rule = next((r for r in data['rules'] if r['rule_id'] == rule_id), None)
    if not rule:
        raise HTTPException(status_code=404, detail='Rule not found')
    return rule

@router.post('/labor-law-rules', response_model=dict)
def create_rule(rule: dict):
    data = load_rules()
    # Generate a random unique rule_id
    rule_id = str(uuid.uuid4())
    rule['rule_id'] = rule_id
    data['rules'].append(rule)
    save_rules(data)
    return rule

@router.put('/labor-law-rules/{rule_id}', response_model=dict)
def update_rule(rule_id: str, rule: dict):
    data = load_rules()
    for i, r in enumerate(data['rules']):
        if r['rule_id'] == rule_id:
            data['rules'][i] = rule
            save_rules(data)
            return rule
    raise HTTPException(status_code=404, detail='Rule not found')

@router.delete('/labor-law-rules/{rule_id}', response_model=dict)
def delete_rule(rule_id: str):
    data = load_rules()
    for i, r in enumerate(data['rules']):
        if r['rule_id'] == rule_id:
            removed = data['rules'].pop(i)
            save_rules(data)
            return removed
    raise HTTPException(status_code=404, detail='Rule not found')

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