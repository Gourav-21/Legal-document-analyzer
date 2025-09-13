import pytest
from loader import RuleLoader
from evaluator import RuleEvaluator
# Removed PenaltyCalculator import - penalties no longer used
import datetime


def build_context(payslip, attendance, contract):
    """Helper to build context"""
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


# Edge case rules
EDGE_CASE_RULES = {
    "rules": [
        {
            "rule_id": "zero_division_protection",
            "name": "Zero Division Protection",
            "law_reference": "Edge Case 1",
            "description": "Test division by zero protection",
            "effective_from": "2024-01-01",
            "effective_to": None,
            "checks": [
                {
                    "condition": "attendance.total_hours > 0",
                    "amount_owed": "payslip.base_salary / max(attendance.total_hours, 1) - contract.hourly_rate",
                    "violation_message": "Effective hourly rate calculation error"
                }
            ],
            "penalty": [
                "total_amount_owed = max(0, check_results[0])",
                "penalty_amount = total_amount_owed * 0.10"
            ]
        },
        {
            "rule_id": "negative_values_handling",
            "name": "Negative Values Handling",
            "law_reference": "Edge Case 2", 
            "description": "Test handling of negative values",
            "effective_from": "2024-01-01",
            "effective_to": None,
            "checks": [
                {
                    "condition": "payslip.overtime_pay < 0 or attendance.overtime_hours < 0",
                    "amount_owed": "abs(min(payslip.overtime_pay, 0)) + abs(min(attendance.overtime_hours, 0)) * contract.hourly_rate",
                    "violation_message": "Negative overtime values detected"
                }
            ],
            "penalty": [
                "total_amount_owed = check_results[0]",
                "penalty_amount = total_amount_owed * 0.25"
            ]
        },
        {
            "rule_id": "boundary_date_rule",
            "name": "Boundary Date Rule",
            "law_reference": "Edge Case 3",
            "description": "Rule effective only for specific period",
            "effective_from": "2024-07-01",
            "effective_to": "2024-07-31",
            "checks": [
                {
                    "condition": "contract.hourly_rate < 35.0",
                    "amount_owed": "(35.0 - contract.hourly_rate) * attendance.total_hours",
                    "violation_message": "Special July 2024 minimum wage requirement"
                }
            ],
            "penalty": [
                "total_amount_owed = check_results[0]",
                "penalty_amount = total_amount_owed * 0.05"
            ]
        },
        {
            "rule_id": "missing_field_handling",
            "name": "Missing Field Handling",
            "law_reference": "Edge Case 4",
            "description": "Test handling of missing optional fields",
            "effective_from": "2024-01-01",
            "effective_to": None,
            "checks": [
                {
                    "condition": "payslip.get('bonus', 0) > 0",
                    "amount_owed": "payslip.get('bonus', 0) * 0.1 - payslip.get('bonus_tax', 0)",
                    "violation_message": "Bonus tax not properly calculated"
                }
            ],
            "penalty": [
                "total_amount_owed = max(0, check_results[0])",
                "penalty_amount = total_amount_owed * 0.15"
            ]
        },
        {
            "rule_id": "complex_conditional",
            "name": "Complex Conditional Logic",
            "law_reference": "Edge Case 5",
            "description": "Test complex conditional expressions",
            "effective_from": "2024-01-01",
            "effective_to": None,
            "checks": [
                {
                    "condition": "(attendance.overtime_hours > 10 and contract.hourly_rate < 40) or (attendance.night_hours > 20 and payslip.get('night_premium', 0) == 0)",
                    "amount_owed": "max((attendance.overtime_hours - 10) * contract.hourly_rate * 0.5, attendance.night_hours * contract.hourly_rate * 0.25)",
                    "violation_message": "Complex overtime or night shift violation"
                }
            ],
            "penalty": [
                "total_amount_owed = check_results[0]",
                "penalty_amount = total_amount_owed * 0.20"
            ]
        }
    ]
}


class TestEdgeCases:
    
    def test_zero_hours_protection(self):
        """Test protection against division by zero with zero hours"""
        payslip = {
            "employee_id": "EDGE001",
            "month": "2024-07",
            "base_salary": 5000.0
        }
        
        attendance = {
            "employee_id": "EDGE001", 
            "month": "2024-07",
            "total_hours": 0,  # Zero hours - potential division by zero
            "overtime_hours": 0
        }
        
        contract = {
            "employee_id": "EDGE001",
            "hourly_rate": 35.0
        }
        
        context = build_context(payslip, attendance, contract)
        rule = EDGE_CASE_RULES["rules"][0]  # zero_division_protection
        
        # Should not crash due to division by zero
        check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
        
        # Should handle gracefully
        assert isinstance(check_results[0]["amount"], (int, float))

    def test_negative_values_handling(self):
        """Test handling of negative overtime values"""
        payslip = {
            "employee_id": "EDGE002",
            "month": "2024-07",
            "overtime_pay": -100.0,  # Negative overtime pay
            "base_salary": 5000.0
        }
        
        attendance = {
            "employee_id": "EDGE002",
            "month": "2024-07", 
            "total_hours": 160,
            "overtime_hours": -5  # Negative overtime hours
        }
        
        contract = {
            "employee_id": "EDGE002",
            "hourly_rate": 30.0
        }
        
        context = build_context(payslip, attendance, contract)
        rule = EDGE_CASE_RULES["rules"][1]  # negative_values_handling
        
        check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
        
        # Should detect negative values and calculate violation
        assert check_results[0]["amount"] > 0
        # Expected: abs(-100) + abs(-5) * 30 = 100 + 150 = 250
        assert check_results[0]["amount"] == pytest.approx(250.0)

    def test_boundary_date_rule_applicable(self):
        """Test rule that's only applicable for July 2024"""
        payslip = {
            "employee_id": "EDGE003",
            "month": "2024-07",  # Within boundary
            "base_salary": 4800.0
        }
        
        attendance = {
            "employee_id": "EDGE003",
            "month": "2024-07",
            "total_hours": 160,
            "overtime_hours": 0
        }
        
        contract = {
            "employee_id": "EDGE003",
            "hourly_rate": 32.0  # Below 35.0 threshold
        }
        
        context = build_context(payslip, attendance, contract)
        rule = EDGE_CASE_RULES["rules"][2]  # boundary_date_rule
        
        # Rule should be applicable for July 2024
        assert RuleEvaluator.is_rule_applicable(rule, "2024-07")
        
        check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
        
        # Should have violation: (35.0 - 32.0) * 160 = 480
        assert check_results[0]["amount"] == pytest.approx(480.0)

    def test_boundary_date_rule_not_applicable(self):
        """Test same rule for August 2024 (outside boundary)"""
        payslip = {
            "employee_id": "EDGE003",
            "month": "2024-08",  # Outside boundary
            "base_salary": 4800.0
        }
        
        attendance = {
            "employee_id": "EDGE003",
            "month": "2024-08",
            "total_hours": 160,
            "overtime_hours": 0
        }
        
        contract = {
            "employee_id": "EDGE003",
            "hourly_rate": 32.0
        }
        
        rule = EDGE_CASE_RULES["rules"][2]  # boundary_date_rule
        
        # Rule should NOT be applicable for August 2024
        assert not RuleEvaluator.is_rule_applicable(rule, "2024-08")

    def test_missing_optional_fields(self):
        """Test handling of missing optional fields using .get()"""
        payslip = {
            "employee_id": "EDGE004",
            "month": "2024-07",
            "base_salary": 5000.0,
            "bonus": 1000.0,
            # bonus_tax is missing - should default to 0
        }
        
        attendance = {
            "employee_id": "EDGE004",
            "month": "2024-07",
            "total_hours": 160,
            "overtime_hours": 0
        }
        
        contract = {
            "employee_id": "EDGE004",
            "hourly_rate": 35.0
        }
        
        context = build_context(payslip, attendance, contract)
        rule = EDGE_CASE_RULES["rules"][3]  # missing_field_handling
        
        check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
        
        # Should handle missing bonus_tax field gracefully
        # Expected: 1000 * 0.1 - 0 = 100
        assert check_results[0]["amount"] == pytest.approx(100.0)

    def test_complex_conditional_logic_first_condition(self):
        """Test complex conditional - first condition triggers"""
        payslip = {
            "employee_id": "EDGE005",
            "month": "2024-07",
            "base_salary": 5000.0
        }
        
        attendance = {
            "employee_id": "EDGE005",
            "month": "2024-07",
            "total_hours": 175,
            "overtime_hours": 15,  # > 10
            "night_hours": 5
        }
        
        contract = {
            "employee_id": "EDGE005",
            "hourly_rate": 35.0  # < 40, so first condition should trigger
        }
        
        context = build_context(payslip, attendance, contract)
        rule = EDGE_CASE_RULES["rules"][4]  # complex_conditional
        
        check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
        
        # First condition: overtime_hours > 10 and hourly_rate < 40
        # Expected: max((15-10) * 35 * 0.5, 5 * 35 * 0.25) = max(87.5, 43.75) = 87.5
        assert check_results[0]["amount"] == pytest.approx(87.5)

    def test_complex_conditional_logic_second_condition(self):
        """Test complex conditional - second condition triggers"""
        payslip = {
            "employee_id": "EDGE006",
            "month": "2024-07",
            "base_salary": 5000.0,
            # night_premium is missing, defaults to 0
        }
        
        attendance = {
            "employee_id": "EDGE006",
            "month": "2024-07",
            "total_hours": 185,
            "overtime_hours": 5,   # <= 10
            "night_hours": 25      # > 20
        }
        
        contract = {
            "employee_id": "EDGE006",
            "hourly_rate": 45.0  # >= 40, so first condition false
        }
        
        context = build_context(payslip, attendance, contract)
        rule = EDGE_CASE_RULES["rules"][4]  # complex_conditional
        
        check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
        
        # Second condition: night_hours > 20 and night_premium == 0
        # Expected: max(5 * 45 * 0.5, 25 * 45 * 0.25) = max(112.5, 281.25) = 281.25
        assert check_results[0]["amount"] == pytest.approx(281.25)

    def test_invalid_expression_handling(self):
        """Test handling of invalid expressions in rules"""
        invalid_rule = {
            "rule_id": "invalid_expr",
            "name": "Invalid Expression Test",
            "law_reference": "Edge Case 6",
            "description": "Test invalid expression handling",
            "effective_from": "2024-01-01",
            "effective_to": None,
            "checks": [
                {
                    "condition": "undefined_variable > 0",  # This will fail
                    "amount_owed": "another_undefined_var * 100",  # This will also fail
                    "violation_message": "Invalid expression test"
                }
            ],
            "penalty": [
                "total_amount_owed = check_results[0]",
                "penalty_amount = total_amount_owed * 0.10"
            ]
        }
        
        payslip = {"employee_id": "EDGE007", "month": "2024-07"}
        attendance = {"employee_id": "EDGE007", "month": "2024-07"}
        contract = {"employee_id": "EDGE007", "hourly_rate": 30.0}
        
        context = build_context(payslip, attendance, contract)
        
        # Should handle invalid expressions gracefully without crashing
        check_results, named_results = RuleEvaluator.evaluate_checks(invalid_rule["checks"], context)
        
        # Should return safe defaults
        assert check_results[0]["amount"] == 0.0
        # Should identify missing variables in the message
        assert "missing fields" in check_results[0]["message"]
        assert "undefined_variable" in check_results[0]["message"]
        assert "another_undefined_var" in check_results[0]["message"]

    def test_extreme_values(self):
        """Test handling of extreme numerical values"""
        payslip = {
            "employee_id": "EDGE008",
            "month": "2024-07",
            "base_salary": 999999999.99,  # Very large salary
            "overtime_pay": 0.01          # Very small overtime
        }
        
        attendance = {
            "employee_id": "EDGE008",
            "month": "2024-07",
            "total_hours": 1,             # Very few hours
            "overtime_hours": 1000        # Extreme overtime
        }
        
        contract = {
            "employee_id": "EDGE008",
            "hourly_rate": 0.01           # Very low rate
        }
        
        context = build_context(payslip, attendance, contract)
        rule = EDGE_CASE_RULES["rules"][0]  # zero_division_protection
        
        # Should handle extreme values without overflow/underflow issues
        check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
        
        # Should complete without errors
        assert isinstance(check_results[0]["amount"], (int, float))
        assert not (check_results[0]["amount"] == float('inf') or check_results[0]["amount"] == float('-inf'))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])