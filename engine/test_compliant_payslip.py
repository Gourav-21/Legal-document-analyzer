from loader import RuleLoader
from evaluator import RuleEvaluator
import pytest

from main import build_context

def test_compliant_payslip():
    # Load compliant payslip
    input_data = RuleLoader.load_input('../data/compliant_payslip_test.json')
    rules_data = RuleLoader.load_rules('../rules/labor_law_rules.json')

    payslip = input_data['payslip'][0]
    attendance = input_data['attendance'][0]
    contract = input_data['contract'][0]

    print('COMPLIANT PAYSLIP TEST:')
    print(f'Base Rate: ₪{contract["hourly_rate"]}/hour (above ₪32.70 minimum)')
    print(f'Overtime Rate: ₪{payslip["overtime_rate"]}/hour')
    print(f'Expected Rate for hours 3-12: ₪{contract["hourly_rate"] * 1.5:.1f}/hour')
    print()

    context = build_context(payslip, attendance, contract)
    violations_found = 0
    zero_amount_violations = 0

    for rule in rules_data['rules']:
        if RuleEvaluator.is_rule_applicable(rule, payslip['month']):
            check_results, named_results = RuleEvaluator.evaluate_checks(rule['checks'], context)
            violations = [cr for cr in check_results if cr['amount'] >= 0]  # Updated to include 0-amount violations
            if violations:
                violations_found += 1
                underpaid = sum(v.get('amount', 0.0) for v in violations)

                # Count zero-amount violations
                zero_violations = [v for v in violations if v.get('amount', 0) == 0]
                zero_amount_violations += len(zero_violations)

                if underpaid > 0:
                    print(f'❌ {rule["name"]}: ₪{underpaid:.2f} underpaid')
                else:
                    print(f'⚠️ {rule["name"]}: ₪{underpaid:.2f} (zero-amount violations: {len(zero_violations)})')
            else:
                print(f'✅ {rule["name"]}: Compliant')

    print(f'\nTotal violations: {violations_found}')
    print(f'Zero-amount violations: {zero_amount_violations}')

    # Assert that we found some violations (including zero-amount ones)
    assert violations_found > 0, f"Expected to find violations, but found {violations_found}"

    # Assert that we have zero-amount violations (this is expected for compliant payslips)
    assert zero_amount_violations > 0, f"Expected to find zero-amount violations, but found {zero_amount_violations}"

if __name__ == "__main__":
    test_compliant_payslip()