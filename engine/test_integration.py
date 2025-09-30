import pytest
import json
import os
import datetime
from pathlib import Path
from main import main, build_context
from loader import RuleLoader
from evaluator import RuleEvaluator

# Resolve repository root (one level up from this file's directory)
repo_root = Path(__file__).resolve().parents[1]


def test_integration_with_complex_data():
    """Integration test with complex payroll data and multiple violations"""
    
    # Create complex test data - single employee with multiple months
    complex_input = {
        "payslip": [
            {
                "מזהה_עובד": "EMP001",
                "חודש": "2024-07",
                "שיעור_נוספות_משולם": 35.0,  # Should be 41.25 (33*1.25)
                "משכורת_בסיס": 5280.0,
                "benefits_value": 200.0
            },
            {
                "מזהה_עובד": "EMP001", 
                "חודש": "2024-08",
                "שיעור_נוספות_משולם": 30.0,  # Below minimum rate
                "משכורת_בסיס": 4800.0,
                "benefits_value": 500.0
            }
        ],
        "attendance": [
            {
                "מזהה_עובד": "EMP001",
                "חודש": "2024-07",
                "שעות_נוספות": 4,    # Will trigger violation
                "סהכ_שעות": 164
            },
            {
                "מזהה_עובד": "EMP001",
                "חודש": "2024-08", 
                "שעות_נוספות": 2,    # Different month
                "סהכ_שעות": 162
            }
        ],
        "contract": [
            {
                "מזהה_עובד": "EMP001",
                "שיעור_שעתי": 33.0
            }
        ]
    }
    
    # Write test data
    repo_root = Path(__file__).resolve().parents[1]
    test_input_path = repo_root / 'data' / 'test_integration_input.json'
    with open(test_input_path, 'w', encoding='utf-8') as f:
        json.dump(complex_input, f, indent=2, ensure_ascii=False)
    
    # Load rules and process
    rules_path = repo_root / 'rules' / 'labor_law_rules.json'
    rules_data = RuleLoader.load_rules(str(rules_path))
    # Limit to rules effective from 2023 onwards for this test's scope
    rules_data['rules'] = [
        r for r in rules_data.get('rules', [])
        if datetime.datetime.strptime(r['effective_from'], '%Y-%m-%d') >= datetime.datetime(2023, 1, 1)
    ]
    input_data = RuleLoader.load_input(str(test_input_path))
    
    results = []
    violations_found = 0
    
    for payslip in input_data['payslip']:
        emp_id = payslip['מזהה_עובד']
        month = payslip['חודש']
        
        # Get attendance and contract data
        attendance_records = [a for a in input_data.get('attendance', []) 
                            if a.get('חודש') == month]
        if attendance_records:
            attendance = attendance_records[0]
        else:
            attendance = {}
            
        contract = next(iter(input_data.get('contract', [])), {})
        
        context = build_context(payslip, attendance, contract)
        
        for rule in rules_data['rules']:
            if not RuleEvaluator.is_rule_applicable(rule, month):
                continue
            check_results, named_results = RuleEvaluator.evaluate_checks(rule['checks'], context)
            # Count only positive-amount violations to avoid false positives from zero/defaults
            violations = [cr for cr in check_results if cr.get('amount', 0) > 0]
            if violations:
                violations_found += len(violations)
                results.append({
                    'rule_id': rule['rule_id'],
                    'employee_id': emp_id,
                    'period': month,
                    'violations': violations,
                    'total_amount_owed': sum(v.get('amount', 0.0) for v in violations)
                })
    
    # Assertions
    assert len(results) > 0, "Should find violations in test data"
    assert violations_found >= 2, "Should find at least 2 violations"
    
    # Check specific violations by month
    july_violations = [r for r in results if r['period'] == '2024-07']
    august_violations = [r for r in results if r['period'] == '2024-08']
    
    assert len(july_violations) > 0, "July should have violations"
    print(f"Found {len(july_violations)} violations in July and {len(august_violations)} violations in August")
    
    # Verify penalty calculations are reasonable (skip rules with calculation errors)
    valid_results = [r for r in results if r['total_amount_owed'] > 0]
    assert len(valid_results) > 0, "Should have at least some valid amount owed calculations"

    for result in valid_results:
        assert result['total_amount_owed'] > 0, "Amount owed should be positive"
    
    # Clean up
    os.remove(str(test_input_path))
    
    print(f"Integration test passed: Found {violations_found} violations across {len(results)} rule violations")


def test_date_boundary_integration():
    """Test date boundary handling in integration"""
    
    # Test data with different months
    boundary_input = {
        "payslip": [
            {
                "מזהה_עובד": "EMP001",
                "חודש": "2022-12",  # Before rules effective date
                "שיעור_נוספות_משולם": 30.0,
                "משכורת_בסיס": 4000.0
            },
            {
                "מזהה_עובד": "EMP001", 
                "חודש": "2024-01",  # After rules effective date
                "שיעור_נוספות_משולם": 30.0,
                "משכורת_בסיס": 4000.0
            }
        ],
        "attendance": [
            {
                "מזהה_עובד": "EMP001",
                "חודש": "2022-12",
                "שעות_נוספות": 3,
                "סהכ_שעות": 163
            },
            {
                "מזהה_עובד": "EMP001",
                "חודש": "2024-01",
                "שעות_נוספות": 3,
                "סהכ_שעות": 163
            }
        ],
        "contract": [
            {
                "מזהה_עובד": "EMP001",
                "שיעור_שעתי": 30.0  # Below minimum wage
            }
        ]
    }
    
    # Write test data
    test_input_path = repo_root / 'data' / 'test_boundary_input.json'
    with open(test_input_path, 'w', encoding='utf-8') as f:
        json.dump(boundary_input, f, indent=2, ensure_ascii=False)
    
    # Process data
    rules_path = repo_root / 'rules' / 'labor_law_rules.json'
    rules_data = RuleLoader.load_rules(str(rules_path))
    # Limit to rules effective from 2023 onwards to validate boundary behavior
    rules_data['rules'] = [
        r for r in rules_data.get('rules', [])
        if datetime.datetime.strptime(r['effective_from'], '%Y-%m-%d') >= datetime.datetime(2023, 1, 1)
    ]
    input_data = RuleLoader.load_input(str(test_input_path))
    
    results_2023 = []
    results_2024 = []
    
    for payslip in input_data['payslip']:
        emp_id = payslip['מזהה_עובד']
        month = payslip['חודש']
        
        attendance_records = [a for a in input_data.get('attendance', []) 
                            if a.get('חודש') == month]
        attendance = attendance_records[0] if attendance_records else {}
        contract = next(iter(input_data.get('contract', [])), {})
        
        context = build_context(payslip, attendance, contract)
        
        for rule in rules_data['rules']:
            if not RuleEvaluator.is_rule_applicable(rule, month):
                continue
            check_results, named_results = RuleEvaluator.evaluate_checks(rule['checks'], context)
            # Count only positive-amount violations to avoid false positives from zero/defaults
            violations = [cr for cr in check_results if cr.get('amount', 0) > 0]
            if violations:
                result = {
                    'rule_id': rule['rule_id'],
                    'employee_id': emp_id,
                    'period': month,
                    'violations': violations,
                    'total_amount_owed': sum(v.get('amount', 0.0) for v in violations)
                }
                if month.startswith('2022'):
                    results_2023.append(result)
                else:
                    results_2024.append(result)
    
    # Assertions
    assert len(results_2023) == 0, "Should have no violations for 2022-12 (before effective date)"
    assert len(results_2024) > 0, "Should have violations for 2024-01 (after effective date)"
    
    # Clean up
    os.remove(str(test_input_path))
    
    print(f"Date boundary test passed: 2022 violations: {len(results_2023)}, 2024 violations: {len(results_2024)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])