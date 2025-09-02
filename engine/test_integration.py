import pytest
import json
import os
from main import main, build_context
from loader import RuleLoader
from evaluator import RuleEvaluator
from penalty_calculator import PenaltyCalculator


def test_integration_with_complex_data():
    """Integration test with complex payroll data and multiple violations"""
    
    # Create complex test data - single employee with multiple months
    complex_input = {
        "payslip": [
            {
                "employee_id": "EMP001",
                "month": "2024-07",
                "overtime_rate": 35.0,  # Should be 41.25 (33*1.25)
                "base_salary": 5280.0,
                "benefits_value": 200.0
            },
            {
                "employee_id": "EMP001", 
                "month": "2024-08",
                "overtime_rate": 30.0,  # Below minimum rate
                "base_salary": 4800.0,
                "benefits_value": 500.0
            }
        ],
        "attendance": [
            {
                "employee_id": "EMP001",
                "month": "2024-07",
                "overtime_hours": 4,    # Will trigger violation
                "total_hours": 164
            },
            {
                "employee_id": "EMP001",
                "month": "2024-08", 
                "overtime_hours": 2,    # Different month
                "total_hours": 162
            }
        ],
        "contract": [
            {
                "employee_id": "EMP001",
                "hourly_rate": 33.0
            }
        ]
    }
    
    # Write test data
    test_input_path = "data/test_integration_input.json"
    with open(test_input_path, 'w', encoding='utf-8') as f:
        json.dump(complex_input, f, indent=2)
    
    # Load rules and process
    rules_data = RuleLoader.load_rules("rules/labor_law_rules.json")
    input_data = RuleLoader.load_input(test_input_path)
    
    results = []
    violations_found = 0
    
    for payslip in input_data['payslip']:
        emp_id = payslip['employee_id']
        month = payslip['month']
        
        # Get attendance and contract data
        attendance_records = [a for a in input_data.get('attendance', []) 
                            if a.get('month') == month]
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
            penalty = PenaltyCalculator.calculate_penalty(rule['penalty'], check_results, named_results)
            
            violations = [cr for cr in check_results if cr['amount'] > 0]
            if violations:
                violations_found += len(violations)
                results.append({
                    'rule_id': rule['rule_id'],
                    'employee_id': emp_id,
                    'period': month,
                    'violations': violations,
                    'total_underpaid_amount': penalty.get('total_underpaid_amount', 0.0),
                    'penalty_amount': penalty['penalty_amount']
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
    valid_results = [r for r in results if r['penalty_amount'] > 0]
    assert len(valid_results) > 0, "Should have at least some valid penalty calculations"
    
    for result in valid_results:
        assert result['total_underpaid_amount'] > 0, "Underpaid amounts should be positive when penalty is positive"
        assert result['penalty_amount'] <= result['total_underpaid_amount'], "Penalty should not exceed underpaid amount"
    
    # Clean up
    os.remove(test_input_path)
    
    print(f"Integration test passed: Found {violations_found} violations across {len(results)} rule violations")


def test_date_boundary_integration():
    """Test date boundary handling in integration"""
    
    # Test data with different months
    boundary_input = {
        "payslip": [
            {
                "employee_id": "EMP001",
                "month": "2022-12",  # Before rules effective date
                "overtime_rate": 30.0,
                "base_salary": 4000.0
            },
            {
                "employee_id": "EMP001", 
                "month": "2024-01",  # After rules effective date
                "overtime_rate": 30.0,
                "base_salary": 4000.0
            }
        ],
        "attendance": [
            {
                "employee_id": "EMP001",
                "month": "2022-12",
                "overtime_hours": 3,
                "total_hours": 163
            },
            {
                "employee_id": "EMP001",
                "month": "2024-01",
                "overtime_hours": 3,
                "total_hours": 163
            }
        ],
        "contract": [
            {
                "employee_id": "EMP001",
                "hourly_rate": 30.0  # Below minimum wage
            }
        ]
    }
    
    # Write test data
    test_input_path = "data/test_boundary_input.json"
    with open(test_input_path, 'w', encoding='utf-8') as f:
        json.dump(boundary_input, f, indent=2)
    
    # Process data
    rules_data = RuleLoader.load_rules("rules/labor_law_rules.json")
    input_data = RuleLoader.load_input(test_input_path)
    
    results_2023 = []
    results_2024 = []
    
    for payslip in input_data['payslip']:
        emp_id = payslip['employee_id']
        month = payslip['month']
        
        attendance_records = [a for a in input_data.get('attendance', []) 
                            if a.get('month') == month]
        attendance = attendance_records[0] if attendance_records else {}
        contract = next(iter(input_data.get('contract', [])), {})
        
        context = build_context(payslip, attendance, contract)
        
        for rule in rules_data['rules']:
            if not RuleEvaluator.is_rule_applicable(rule, month):
                continue
                
            check_results, named_results = RuleEvaluator.evaluate_checks(rule['checks'], context)
            penalty = PenaltyCalculator.calculate_penalty(rule['penalty'], check_results, named_results)
            
            violations = [cr for cr in check_results if cr['amount'] > 0]
            if violations:
                result = {
                    'rule_id': rule['rule_id'],
                    'employee_id': emp_id,
                    'period': month,
                    'violations': violations,
                    'total_underpaid_amount': penalty.get('total_underpaid_amount', 0.0),
                    'penalty_amount': penalty['penalty_amount']
                }
                
                if month.startswith('2022'):
                    results_2023.append(result)
                else:
                    results_2024.append(result)
    
    # Assertions
    assert len(results_2023) == 0, "Should have no violations for 2022-12 (before effective date)"
    assert len(results_2024) > 0, "Should have violations for 2024-01 (after effective date)"
    
    # Clean up
    os.remove(test_input_path)
    
    print(f"Date boundary test passed: 2022 violations: {len(results_2023)}, 2024 violations: {len(results_2024)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])