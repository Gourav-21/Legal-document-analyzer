import pytest
from loader import RuleLoader
from evaluator import RuleEvaluator
from penalty_calculator import PenaltyCalculator
import datetime


# Helper to build context (copied from main.py)
def build_context(payslip, attendance, contract):
    context = {
        'employee_id': payslip['employee_id'],
        'month': payslip['month'],
        'payslip': payslip,
        'attendance': attendance,
        'contract': contract
    }
    context.update(payslip)
    context.update(attendance)
    context.update(contract)
    return context

# Sample rules (minimal, for test isolation)
RULES = {
    "rules": [
        {
            "rule_id": "overtime_first_2h",
            "name": "Overtime First 2 Hours",
            "law_reference": "Section 16A",
            "description": "Pay 125% for first 2 hours of overtime",
            "effective_from": "2023-01-01",
            "effective_to": None,
            "checks": [
                {
                    "condition": "attendance.overtime_hours > 0",
                    "underpaid_amount": "(contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)",
                    "violation_message": "First 2 hours of overtime must be paid at 125%"
                }
            ],
            "penalty": [
                "total_underpaid_amount = check_results[0]",
                "penalty_amount = total_underpaid_amount * 0.05"
            ],
            "created_date": "2024-01-01T00:00:00Z",
            "updated_date": "2025-08-27T00:00:00Z"
        },
        {
            "rule_id": "minimum_wage",
            "name": "Minimum Wage",
            "law_reference": "Section 1",
            "description": "Base hourly rate must be >= 32.7 NIS",
            "effective_from": "2024-01-01",
            "effective_to": None,
            "checks": [
                {
                    "condition": "contract.hourly_rate < 32.7",
                    "underpaid_amount": "(32.7 - contract.hourly_rate) * attendance.total_hours",
                    "violation_message": "Hourly rate below minimum wage"
                }
            ],
            "penalty": [
                "total_underpaid_amount = check_results[0]",
                "penalty_amount = total_underpaid_amount * 0.10"
            ],
            "created_date": "2024-01-01T00:00:00Z",
            "updated_date": "2025-08-27T00:00:00Z"
        }
    ]
}

# Test 1: No violations
@pytest.mark.parametrize("payslip,attendance,contract", [
    ({"employee_id": "1", "month": "2024-07", "overtime_rate": 41.25},
     {"employee_id": "1", "month": "2024-07", "overtime_hours": 2, "total_hours": 160},
     {"employee_id": "1", "hourly_rate": 33.0})
])
def test_no_violations(payslip, attendance, contract):
    context = build_context(payslip, attendance, contract)
    for rule in RULES["rules"]:
        assert RuleEvaluator.is_rule_applicable(rule, payslip["month"])
        check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
        penalty = PenaltyCalculator.calculate_penalty(rule["penalty"], check_results, named_results)
        assert all(cr["amount"] == 0 for cr in check_results)
        assert penalty["total_underpaid_amount"] == 0
        assert penalty["penalty_amount"] == 0

# Test 2: Only first check triggered
@pytest.mark.parametrize("payslip,attendance,contract", [
    ({"employee_id": "2", "month": "2024-07", "overtime_rate": 30.0},
     {"employee_id": "2", "month": "2024-07", "overtime_hours": 2, "total_hours": 160},
     {"employee_id": "2", "hourly_rate": 32.0})
])
def test_first_check_triggered(payslip, attendance, contract):
    context = build_context(payslip, attendance, contract)
    rule = RULES["rules"][0]  # overtime_first_2h
    assert RuleEvaluator.is_rule_applicable(rule, payslip["month"])
    check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
    penalty = PenaltyCalculator.calculate_penalty(rule["penalty"], check_results, named_results)
    assert check_results[0]["amount"] > 0
    assert penalty["total_underpaid_amount"] == check_results[0]["amount"]
    assert penalty["penalty_amount"] == pytest.approx(penalty["total_underpaid_amount"] * 0.05)

# Test 3: Minimum wage violation
@pytest.mark.parametrize("payslip,attendance,contract", [
    ({"employee_id": "3", "month": "2024-07", "overtime_rate": 40.0},
     {"employee_id": "3", "month": "2024-07", "overtime_hours": 1, "total_hours": 180},
     {"employee_id": "3", "hourly_rate": 30.0})
])
def test_minimum_wage_violation(payslip, attendance, contract):
    context = build_context(payslip, attendance, contract)
    rule = RULES["rules"][1]  # minimum_wage
    assert RuleEvaluator.is_rule_applicable(rule, payslip["month"])
    check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
    penalty = PenaltyCalculator.calculate_penalty(rule["penalty"], check_results, named_results)
    assert check_results[0]["amount"] == pytest.approx((32.7 - 30.0) * 180)
    assert penalty["penalty_amount"] == pytest.approx(check_results[0]["amount"] * 0.10)

# More tests can be added for expired rules, missing fields, and multiple employees
