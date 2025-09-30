import pytest
from evaluator import RuleEvaluator
 # Removed PenaltyCalculator import
import datetime


# Helper to build context (copied from main.py)
from dynamic_params import DynamicParams

def build_context(payslip, attendance, contract, employee=None):
    params = DynamicParams.load()
    context = {
        'payslip': payslip,
        'attendance': attendance,
        'contract': contract,
        'employee': employee
    }
    # Flatten all dynamic params for direct access
    for section in ['payslip', 'attendance', 'contract', 'employee']:
        section_data = locals()[section] if section in locals() and locals()[section] is not None else {}
        for p in params.get(section, []):
            context[p['param']] = section_data.get(p['param'], None)
    # Add employee_id and month for legacy compatibility
    context['employee_id'] = context.get('employee_id', payslip.get('employee_id', None))
    context['month'] = context.get('month', payslip.get('month', None))
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
                    "amount_owed": "(contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)",
                    "violation_message": "First 2 hours of overtime must be paid at 125%"
                }
            ],
            # Removed penalty field
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
                    "amount_owed": "(32.7 - contract.hourly_rate) * attendance.total_hours",
                    "violation_message": "Hourly rate below minimum wage"
                }
            ],
            # Removed penalty field
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
        assert all(cr["amount"] == 0 for cr in check_results)
        total_amount_owed = sum(cr["amount"] for cr in check_results if cr["amount"] > 0)
        assert total_amount_owed == 0

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
    total_amount_owed = sum(cr["amount"] for cr in check_results if cr["amount"] > 0)
    assert check_results[0]["amount"] > 0
    assert total_amount_owed == check_results[0]["amount"]

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
    assert check_results[0]["amount"] == pytest.approx((32.7 - 30.0) * 180)

# More tests can be added for expired rules, missing fields, and multiple employees

def test_hebrew_identifiers_in_expressions():
    # Hebrew variable names should be supported in expressions and context
    context = {
        'משכורת_בסיס': 5000,
        'שעות_נוספות': 10,
        'שיעור_נוספות_125': 50,
    }
    condition_expr = 'משכורת_בסיס > 4000 and שעות_נוספות >= 10'
    amount_expr = 'שעות_נוספות * שיעור_נוספות_125'

    missing_cond = RuleEvaluator.find_missing_variables(condition_expr, context)
    missing_amount = RuleEvaluator.find_missing_variables(amount_expr, context)
    assert missing_cond == []
    assert missing_amount == []

    checks = [{
        'id': 'hebrew_check',
        'condition': condition_expr,
        'amount_owed': amount_expr,
        'violation_message': 'עברית תקינה'
    }]
    results, named = RuleEvaluator.evaluate_checks(checks, context)
    assert results[0]['amount'] == 500
    assert results[0]['message'] == 'עברית תקינה'

def test_hebrew_dot_notation_identifiers():
    # Support Hebrew identifiers with dot-notation
    context = {
        'תלוש': {
            'שעות_נוספות': 3,
            'שיעור_שעתי': 60,
        }
    }
    condition_expr = 'תלוש.שעות_נוספות > 2'
    amount_expr = 'תלוש.שעות_נוספות * תלוש.שיעור_שעתי'

    assert RuleEvaluator.find_missing_variables(condition_expr, context) == []
    assert RuleEvaluator.find_missing_variables(amount_expr, context) == []

    checks = [{
        'id': 'hebrew_dot',
        'condition': condition_expr,
        'amount_owed': amount_expr,
        'violation_message': 'עברית עם נקודה'
    }]
    results, _ = RuleEvaluator.evaluate_checks(checks, context)
    assert results[0]['amount'] == 180
    assert results[0]['message'] == 'עברית עם נקודה'

def test_hebrew_nested_attributes_three_levels():
    # Verify deeper nesting with Hebrew keys
    context = {
        'עובד': {
            'פרטים': {
                'שם_העובד': 'ישראל ישראלי',
                'גיל': 5,
            }
        }
    }
    condition_expr = 'עובד.פרטים.גיל >= 5'
    amount_expr = 'עובד.פרטים.גיל * 100'

    assert RuleEvaluator.find_missing_variables(condition_expr, context) == []
    assert RuleEvaluator.find_missing_variables(amount_expr, context) == []

    checks = [{
        'id': 'hebrew_nested',
        'condition': condition_expr,
        'amount_owed': amount_expr,
        'violation_message': 'עומק בעברית'
    }]
    results, _ = RuleEvaluator.evaluate_checks(checks, context)
    assert results[0]['amount'] == 500

def test_hebrew_missing_variables_detection():
    # Ensure missing-variable detection works with Hebrew
    context = {'שכר': 1000}
    condition_expr = 'שכר > 0 and שעות_נוספות > 0'
    amount_expr = 'שעות_נוספות * תעריף_נוספות'

    missing_cond = RuleEvaluator.find_missing_variables(condition_expr, context)
    missing_amount = RuleEvaluator.find_missing_variables(amount_expr, context)
    assert set(missing_cond) == {'שעות_נוספות'}
    assert set(missing_amount) == {'תעריף_נוספות', 'שעות_נוספות'}

def test_mixed_hebrew_latin_identifiers():
    # Expressions can mix Hebrew and Latin identifiers
    context = {
        'base_salary': 4000,
        'תוספת': 250,
    }
    condition_expr = 'base_salary >= 4000 and תוספת >= 200'
    amount_expr = 'base_salary + תוספת'

    assert RuleEvaluator.find_missing_variables(condition_expr, context) == []
    assert RuleEvaluator.find_missing_variables(amount_expr, context) == []

    checks = [{
        'id': 'mixed_identifiers',
        'condition': condition_expr,
        'amount_owed': amount_expr,
        'violation_message': 'מעורב'
    }]
    results, _ = RuleEvaluator.evaluate_checks(checks, context)
    assert results[0]['amount'] == 4250
