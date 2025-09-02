from loader import RuleLoader
from evaluator import RuleEvaluator
from penalty_calculator import PenaltyCalculator
from main import build_context

def test_compliant_payslip():
    # Load compliant payslip
    input_data = RuleLoader.load_input('data/compliant_payslip_test.json')
    rules_data = RuleLoader.load_rules('rules/labor_law_rules.json')

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

    for rule in rules_data['rules']:
        if RuleEvaluator.is_rule_applicable(rule, payslip['month']):
            check_results, named_results = RuleEvaluator.evaluate_checks(rule['checks'], context)
            penalty = PenaltyCalculator.calculate_penalty(rule['penalty'], check_results, named_results)
            
            violations = [cr for cr in check_results if cr['amount'] > 0]
            if violations:
                violations_found += 1
                print(f'❌ {rule["name"]}: ₪{penalty["total_underpaid_amount"]:.2f} underpaid')
            else:
                print(f'✅ {rule["name"]}: Compliant')

    print(f'\nTotal violations: {violations_found}')

if __name__ == "__main__":
    test_compliant_payslip()