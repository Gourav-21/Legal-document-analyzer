from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import json
import os
import uuid
import sys

# Add engine directory to path
sys.path.append('engine')

from engine.dynamic_params import DynamicParams
from engine.evaluator import RuleEvaluator
from engine.main import build_context

router = APIRouter()

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


# Expression Testing API Models
class ExpressionTestRequest(BaseModel):
    expression: str
    expression_type: str  # 'condition' or 'calculation'
    payslip: Optional[Dict[str, Any]] = {}
    attendance: Optional[Dict[str, Any]] = {}
    contract: Optional[Dict[str, Any]] = {}

class ExpressionTestResponse(BaseModel):
    success: bool
    result: Any
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    missing_variables: Optional[List[str]] = None

# Rule Testing API Models
class RuleCheck(BaseModel):
    id: Optional[str] = None
    condition: str
    amount_owed: str
    violation_message: Optional[str] = None

class RuleTestRequest(BaseModel):
    rule_id: str
    name: str
    description: Optional[str] = None
    law_reference: Optional[str] = None
    checks: List[RuleCheck]
    payslip: Optional[Dict[str, Any]] = {}
    attendance: Optional[Dict[str, Any]] = {}
    contract: Optional[Dict[str, Any]] = {}

class RuleCheckResult(BaseModel):
    check_id: Optional[str]
    condition_result: bool
    amount_owed: float
    violation_message: Optional[str]
    missing_fields: List[str]
    condition_error: Optional[str] = None
    amount_error: Optional[str] = None

class RuleTestResponse(BaseModel):
    success: bool
    rule_id: str
    rule_name: str
    total_violations: int
    total_amount_owed: float
    check_results: List[RuleCheckResult]
    error: Optional[str] = None
    context_used: Optional[Dict[str, Any]] = None


# Expression Testing API
@router.post("/test-expression", response_model=ExpressionTestResponse)
async def test_expression(request: ExpressionTestRequest):
    """Test an expression (condition or calculation) with provided data"""
    try:
        # Validate expression type
        if request.expression_type not in ['condition', 'calculation']:
            raise HTTPException(
                status_code=400,
                detail="expression_type must be either 'condition' or 'calculation'"
            )

        # Build context from the provided data
        context = build_context(
            request.payslip or {},
            request.attendance or {},
            request.contract or {}
        )

        # Find missing variables in the expression
        missing_vars = RuleEvaluator.find_missing_variables(request.expression, context)

        # Try to evaluate the expression
        try:
            # Use simple_eval for both conditions and calculations
            from simpleeval import simple_eval
            allowed_functions = {"min": min, "max": max, "abs": abs, "round": round}

            result = simple_eval(request.expression, names=context, functions=allowed_functions)

            return ExpressionTestResponse(
                success=True,
                result=result,
                details={
                    "expression_type": request.expression_type,
                    "context_used": context
                },
                missing_variables=missing_vars if missing_vars else None
            )

        except Exception as eval_error:
            return ExpressionTestResponse(
                success=False,
                result=None,
                error=f"Expression evaluation failed: {str(eval_error)}",
                details={
                    "expression_type": request.expression_type,
                    "context_used": context
                },
                missing_variables=missing_vars if missing_vars else None
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error testing expression: {str(e)}"
        )

# Rule Testing API
@router.post("/test-rule", response_model=RuleTestResponse)
async def test_rule(request: RuleTestRequest):
    """Test a complete rule with all its checks against provided data"""
    try:
        # Import simple_eval once at the top
        from simpleeval import simple_eval

        # Build context from the provided data
        context = build_context(
            request.payslip or {},
            request.attendance or {},
            request.contract or {}
        )

        # Define allowed functions once
        allowed_functions = {"min": min, "max": max, "abs": abs, "round": round}

        # Evaluate all checks in the rule
        check_results = []
        total_violations = 0
        total_amount_owed = 0.0

        for check in request.checks:
            # Find missing variables in both condition and amount_owed
            condition_missing = RuleEvaluator.find_missing_variables(check.condition, context)
            amount_missing = RuleEvaluator.find_missing_variables(check.amount_owed, context)
            all_missing = sorted(set(condition_missing + amount_missing))

            # Initialize result structure
            result = {
                "check_id": check.id,
                "condition_result": False,
                "amount_owed": 0.0,
                "violation_message": check.violation_message,
                "missing_fields": all_missing,
                "condition_error": None,
                "amount_error": None
            }

            # Try to evaluate condition
            try:
                cond = simple_eval(check.condition, names=context, functions=allowed_functions)
                result["condition_result"] = bool(cond)
            except Exception as e:
                result["condition_error"] = str(e)
                result["condition_result"] = False

            # If condition is true, try to evaluate amount
            if result["condition_result"]:
                try:
                    amount = simple_eval(check.amount_owed, names=context, functions=allowed_functions)
                    result["amount_owed"] = float(amount)
                    total_violations += 1
                    total_amount_owed += result["amount_owed"]
                except Exception as e:
                    result["amount_error"] = str(e)
                    result["amount_owed"] = 0.0

            check_results.append(result)

        return RuleTestResponse(
            success=True,
            rule_id=request.rule_id,
            rule_name=request.name,
            total_violations=total_violations,
            total_amount_owed=total_amount_owed,
            check_results=check_results,
            context_used=context
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error testing rule: {str(e)}"
        )