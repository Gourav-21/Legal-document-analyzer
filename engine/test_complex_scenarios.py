import pytest
from loader import RuleLoader
from evaluator import RuleEvaluator
from penalty_calculator import PenaltyCalculator
import datetime


def build_context(payslip, attendance, contract):
    """Helper to build context (copied from main.py)"""
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


# Complex rules for comprehensive testing
COMPLEX_RULES = {
    "rules": [
        {
            "rule_id": "overtime_tiered",
            "name": "Tiered Overtime Rates",
            "law_reference": "Section 16",
            "description": "125% for first 2h, 150% for next 3h, 200% for beyond 5h",
            "effective_from": "2024-01-01",
            "effective_to": None,
            "checks": [
                {
                    "id": "first_2h_violation",
                    "condition": "attendance.overtime_hours > 0",
                    "underpaid_amount": "max(0, (contract.hourly_rate * 1.25 - payslip.overtime_rate_tier1) * min(attendance.overtime_hours, 2))",
                    "violation_message": "First 2 hours of overtime must be paid at 125%"
                },
                {
                    "id": "next_3h_violation", 
                    "condition": "attendance.overtime_hours > 2",
                    "underpaid_amount": "max(0, (contract.hourly_rate * 1.5 - payslip.overtime_rate_tier2) * min(max(attendance.overtime_hours - 2, 0), 3))",
                    "violation_message": "Hours 3-5 of overtime must be paid at 150%"
                },
                {
                    "id": "beyond_5h_violation",
                    "condition": "attendance.overtime_hours > 5",
                    "underpaid_amount": "max(0, (contract.hourly_rate * 2.0 - payslip.overtime_rate_tier3) * max(attendance.overtime_hours - 5, 0))",
                    "violation_message": "Overtime beyond 5 hours must be paid at 200%"
                }
            ],
            "penalty": [
                "total_underpaid_amount = first_2h_violation + next_3h_violation + beyond_5h_violation",
                "penalty_amount = total_underpaid_amount * 0.15"
            ]
        },
        {
            "rule_id": "night_shift_premium",
            "name": "Night Shift Premium",
            "law_reference": "Section 22",
            "description": "25% premium for hours worked between 22:00-06:00",
            "effective_from": "2024-01-01",
            "effective_to": None,
            "checks": [
                {
                    "condition": "attendance.night_hours > 0",
                    "underpaid_amount": "max(0, (contract.hourly_rate * 0.25) * attendance.night_hours - payslip.night_premium_paid)",
                    "violation_message": "Night shift hours (22:00-06:00) must include 25% premium"
                }
            ],
            "penalty": [
                "total_underpaid_amount = check_results[0]",
                "penalty_amount = total_underpaid_amount * 0.20"
            ]
        },
        {
            "rule_id": "weekly_rest",
            "name": "Weekly Rest Violation",
            "law_reference": "Section 8",
            "description": "Employee must have 36 consecutive hours of rest per week",
            "effective_from": "2024-01-01",
            "effective_to": None,
            "checks": [
                {
                    "condition": "attendance.max_consecutive_work_hours > 144",  # 6 days * 24 hours
                    "underpaid_amount": "contract.hourly_rate * 8 * attendance.weekly_rest_violations",
                    "violation_message": "Employee worked more than 6 consecutive days without 36h rest"
                }
            ],
            "penalty": [
                "total_underpaid_amount = check_results[0]",
                "penalty_amount = total_underpaid_amount * 0.25"
            ]
        },
        {
            "rule_id": "vacation_pay",
            "name": "Vacation Pay Calculation",
            "law_reference": "Section 12",
            "description": "Vacation pay must include average overtime from last 3 months",
            "effective_from": "2024-01-01",
            "effective_to": None,
            "checks": [
                {
                    "condition": "payslip.vacation_days_taken > 0",
                    "underpaid_amount": "max(0, (payslip.vacation_days_taken * (contract.daily_rate + attendance.avg_daily_overtime_last_3m)) - payslip.vacation_pay_given)",
                    "violation_message": "Vacation pay must include average overtime compensation"
                }
            ],
            "penalty": [
                "total_underpaid_amount = check_results[0]", 
                "penalty_amount = total_underpaid_amount * 0.12"
            ]
        },
        {
            "rule_id": "minimum_wage_with_benefits",
            "name": "Minimum Wage Including Benefits",
            "law_reference": "Section 1A",
            "description": "Total compensation (salary + benefits) must meet minimum wage",
            "effective_from": "2024-01-01",
            "effective_to": None,
            "checks": [
                {
                    "condition": "(payslip.base_salary + payslip.benefits_value) / attendance.total_hours < 32.7",
                    "underpaid_amount": "(32.7 * attendance.total_hours) - (payslip.base_salary + payslip.benefits_value)",
                    "violation_message": "Total compensation (salary + benefits) below minimum wage"
                }
            ],
            "penalty": [
                "total_underpaid_amount = check_results[0]",
                "penalty_amount = total_underpaid_amount * 0.18"
            ]
        }
    ]
}


class TestComplexScenarios:
    
    def test_perfect_compliance_no_violations(self):
        """Test case where all labor laws are perfectly followed"""
        payslip = {
            "employee_id": "EMP001",
            "month": "2024-07",
            "overtime_rate_tier1": 41.25,  # 33 * 1.25
            "overtime_rate_tier2": 49.5,   # 33 * 1.5  
            "overtime_rate_tier3": 66.0,   # 33 * 2.0
            "night_premium_paid": 33.0,    # 33 * 0.25 * 4 hours
            "vacation_days_taken": 0,
            "vacation_pay_given": 0,
            "base_salary": 5280.0,         # 33 * 160
            "benefits_value": 500.0
        }
        
        attendance = {
            "employee_id": "EMP001",
            "month": "2024-07", 
            "overtime_hours": 3,
            "night_hours": 4,
            "total_hours": 163,
            "max_consecutive_work_hours": 120,  # 5 days
            "weekly_rest_violations": 0,
            "avg_daily_overtime_last_3m": 0
        }
        
        contract = {
            "employee_id": "EMP001",
            "hourly_rate": 33.0,
            "daily_rate": 264.0  # 33 * 8
        }
        
        context = build_context(payslip, attendance, contract)
        
        for rule in COMPLEX_RULES["rules"]:
            check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
            penalty = PenaltyCalculator.calculate_penalty(rule["penalty"], check_results, named_results)
            
            # Should have no violations
            assert all(cr["amount"] == 0 for cr in check_results), f"Rule {rule['rule_id']} should have no violations"
            assert penalty["total_underpaid_amount"] == 0
            assert penalty["penalty_amount"] == 0

    def test_multiple_overtime_violations(self):
        """Test complex overtime scenario with violations in all tiers"""
        payslip = {
            "employee_id": "EMP002", 
            "month": "2024-07",
            "overtime_rate_tier1": 35.0,   # Should be 37.5 (30 * 1.25)
            "overtime_rate_tier2": 40.0,   # Should be 45.0 (30 * 1.5)
            "overtime_rate_tier3": 50.0,   # Should be 60.0 (30 * 2.0)
            "night_premium_paid": 0,
            "vacation_days_taken": 0,
            "vacation_pay_given": 0,
            "base_salary": 4800.0,
            "benefits_value": 200.0
        }
        
        attendance = {
            "employee_id": "EMP002",
            "month": "2024-07",
            "overtime_hours": 8,  # 2h at 125%, 3h at 150%, 3h at 200%
            "night_hours": 0,
            "total_hours": 168,
            "max_consecutive_work_hours": 120,
            "weekly_rest_violations": 0,
            "avg_daily_overtime_last_3m": 0
        }
        
        contract = {
            "employee_id": "EMP002", 
            "hourly_rate": 30.0,
            "daily_rate": 240.0
        }
        
        context = build_context(payslip, attendance, contract)
        rule = COMPLEX_RULES["rules"][0]  # overtime_tiered
        
        check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
        penalty = PenaltyCalculator.calculate_penalty(rule["penalty"], check_results, named_results)
        
        # Expected violations:
        # Tier 1: (37.5 - 35.0) * 2 = 5.0
        # Tier 2: (45.0 - 40.0) * 3 = 15.0  
        # Tier 3: (60.0 - 50.0) * 3 = 30.0
        # Total: 50.0
        
        assert check_results[0]["amount"] == pytest.approx(5.0)
        assert check_results[1]["amount"] == pytest.approx(15.0)
        assert check_results[2]["amount"] == pytest.approx(30.0)
        assert penalty["total_underpaid_amount"] == pytest.approx(50.0)
        assert penalty["penalty_amount"] == pytest.approx(7.5)  # 50.0 * 0.15

    def test_night_shift_violation(self):
        """Test night shift premium violation"""
        payslip = {
            "employee_id": "EMP003",
            "month": "2024-07",
            "overtime_rate_tier1": 40.0,
            "overtime_rate_tier2": 48.0,
            "overtime_rate_tier3": 64.0,
            "night_premium_paid": 50.0,  # Should be 80.0 (32 * 0.25 * 10)
            "vacation_days_taken": 0,
            "vacation_pay_given": 0,
            "base_salary": 5120.0,
            "benefits_value": 300.0
        }
        
        attendance = {
            "employee_id": "EMP003",
            "month": "2024-07",
            "overtime_hours": 2,
            "night_hours": 10,  # 10 hours of night work
            "total_hours": 162,
            "max_consecutive_work_hours": 96,
            "weekly_rest_violations": 0,
            "avg_daily_overtime_last_3m": 0
        }
        
        contract = {
            "employee_id": "EMP003",
            "hourly_rate": 32.0,
            "daily_rate": 256.0
        }
        
        context = build_context(payslip, attendance, contract)
        rule = COMPLEX_RULES["rules"][1]  # night_shift_premium
        
        check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
        penalty = PenaltyCalculator.calculate_penalty(rule["penalty"], check_results, named_results)
        
        # Expected: (32 * 0.25 * 10) - 50 = 80 - 50 = 30
        assert check_results[0]["amount"] == pytest.approx(30.0)
        assert penalty["penalty_amount"] == pytest.approx(6.0)  # 30.0 * 0.20

    def test_weekly_rest_violation(self):
        """Test weekly rest period violation"""
        payslip = {
            "employee_id": "EMP004",
            "month": "2024-07",
            "overtime_rate_tier1": 37.5,
            "overtime_rate_tier2": 45.0,
            "overtime_rate_tier3": 60.0,
            "night_premium_paid": 0,
            "vacation_days_taken": 0,
            "vacation_pay_given": 0,
            "base_salary": 4800.0,
            "benefits_value": 400.0
        }
        
        attendance = {
            "employee_id": "EMP004",
            "month": "2024-07",
            "overtime_hours": 0,
            "night_hours": 0,
            "total_hours": 160,
            "max_consecutive_work_hours": 168,  # 7 days straight
            "weekly_rest_violations": 2,  # 2 weeks with violations
            "avg_daily_overtime_last_3m": 0
        }
        
        contract = {
            "employee_id": "EMP004",
            "hourly_rate": 30.0,
            "daily_rate": 240.0
        }
        
        context = build_context(payslip, attendance, contract)
        rule = COMPLEX_RULES["rules"][2]  # weekly_rest
        
        check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
        penalty = PenaltyCalculator.calculate_penalty(rule["penalty"], check_results, named_results)
        
        # Expected: 30 * 8 * 2 = 480
        assert check_results[0]["amount"] == pytest.approx(480.0)
        assert penalty["penalty_amount"] == pytest.approx(120.0)  # 480.0 * 0.25

    def test_vacation_pay_violation(self):
        """Test vacation pay calculation violation"""
        payslip = {
            "employee_id": "EMP005",
            "month": "2024-07",
            "overtime_rate_tier1": 41.25,
            "overtime_rate_tier2": 49.5,
            "overtime_rate_tier3": 66.0,
            "night_premium_paid": 0,
            "vacation_days_taken": 5,
            "vacation_pay_given": 1200.0,  # Should be 1400.0 (5 * (240 + 40))
            "base_salary": 5280.0,
            "benefits_value": 500.0
        }
        
        attendance = {
            "employee_id": "EMP005",
            "month": "2024-07",
            "overtime_hours": 0,
            "night_hours": 0,
            "total_hours": 155,  # 5 days vacation
            "max_consecutive_work_hours": 120,
            "weekly_rest_violations": 0,
            "avg_daily_overtime_last_3m": 40.0  # Average daily overtime
        }
        
        contract = {
            "employee_id": "EMP005",
            "hourly_rate": 33.0,
            "daily_rate": 264.0
        }
        
        context = build_context(payslip, attendance, contract)
        rule = COMPLEX_RULES["rules"][3]  # vacation_pay
        
        check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
        penalty = PenaltyCalculator.calculate_penalty(rule["penalty"], check_results, named_results)
        
        # Expected: 5 * (264 + 40) - 1200 = 1520 - 1200 = 320
        assert check_results[0]["amount"] == pytest.approx(320.0)
        assert penalty["penalty_amount"] == pytest.approx(38.4)  # 320.0 * 0.12

    def test_minimum_wage_with_benefits_violation(self):
        """Test minimum wage violation considering benefits"""
        payslip = {
            "employee_id": "EMP006",
            "month": "2024-07",
            "overtime_rate_tier1": 37.5,
            "overtime_rate_tier2": 45.0,
            "overtime_rate_tier3": 60.0,
            "night_premium_paid": 0,
            "vacation_days_taken": 0,
            "vacation_pay_given": 0,
            "base_salary": 4800.0,  # 30 * 160
            "benefits_value": 200.0   # Total: 5000, per hour: 31.25 < 32.7
        }
        
        attendance = {
            "employee_id": "EMP006",
            "month": "2024-07",
            "overtime_hours": 0,
            "night_hours": 0,
            "total_hours": 160,
            "max_consecutive_work_hours": 120,
            "weekly_rest_violations": 0,
            "avg_daily_overtime_last_3m": 0
        }
        
        contract = {
            "employee_id": "EMP006",
            "hourly_rate": 30.0,
            "daily_rate": 240.0
        }
        
        context = build_context(payslip, attendance, contract)
        rule = COMPLEX_RULES["rules"][4]  # minimum_wage_with_benefits
        
        check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
        penalty = PenaltyCalculator.calculate_penalty(rule["penalty"], check_results, named_results)
        
        # Expected: (32.7 * 160) - (4800 + 200) = 5232 - 5000 = 232
        assert check_results[0]["amount"] == pytest.approx(232.0)
        assert penalty["penalty_amount"] == pytest.approx(41.76)  # 232.0 * 0.18

    def test_multiple_rules_multiple_violations(self):
        """Test scenario where multiple rules are violated simultaneously"""
        payslip = {
            "employee_id": "EMP007",
            "month": "2024-07",
            "overtime_rate_tier1": 35.0,   # Violation: should be 37.5
            "overtime_rate_tier2": 40.0,   # Violation: should be 45.0
            "overtime_rate_tier3": 50.0,   # Violation: should be 60.0
            "night_premium_paid": 30.0,    # Violation: should be 37.5
            "vacation_days_taken": 0,
            "vacation_pay_given": 0,
            "base_salary": 4500.0,          # Violation: too low
            "benefits_value": 100.0
        }
        
        attendance = {
            "employee_id": "EMP007",
            "month": "2024-07",
            "overtime_hours": 6,  # Triggers overtime violations
            "night_hours": 5,     # Triggers night premium violation
            "total_hours": 166,
            "max_consecutive_work_hours": 120,
            "weekly_rest_violations": 0,
            "avg_daily_overtime_last_3m": 0
        }
        
        contract = {
            "employee_id": "EMP007",
            "hourly_rate": 30.0,
            "daily_rate": 240.0
        }
        
        context = build_context(payslip, attendance, contract)
        
        total_violations = 0
        total_penalties = 0
        
        for rule in COMPLEX_RULES["rules"]:
            if not RuleEvaluator.is_rule_applicable(rule, payslip["month"]):
                continue
                
            check_results, named_results = RuleEvaluator.evaluate_checks(rule["checks"], context)
            penalty = PenaltyCalculator.calculate_penalty(rule["penalty"], check_results, named_results)
            
            violations = [cr for cr in check_results if cr["amount"] > 0]
            if violations:
                total_violations += penalty["total_underpaid_amount"]
                total_penalties += penalty["penalty_amount"]
                print(f"Rule {rule['rule_id']}: Underpaid={penalty['total_underpaid_amount']:.2f}, Penalty={penalty['penalty_amount']:.2f}")
        
        # Should have multiple violations across different rules
        assert total_violations > 0
        assert total_penalties > 0
        print(f"Total violations: {total_violations:.2f}, Total penalties: {total_penalties:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])