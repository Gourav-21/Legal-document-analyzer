import os
import json
from loader import RuleLoader
from evaluator import RuleEvaluator
from penalty_calculator import PenaltyCalculator

RULES_PATH = os.path.join(os.path.dirname(__file__), '../rules/labor_law_rules.json')
INPUT_PATH = os.path.join(os.path.dirname(__file__), '../data/sample_input.json')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../output/report_2024_07.json')

def build_context(payslip, attendance, contract):
    context = {
        'employee_id': payslip['employee_id'],
        'month': payslip['month'],
        'payslip': payslip,
        'attendance': attendance,
        'contract': contract
    }
    # Flatten top-level fields for direct access in expressions
    context.update(payslip)
    context.update(attendance)
    context.update(contract)
    return context

def main():
    rules_data = RuleLoader.load_rules(RULES_PATH)
    input_data = RuleLoader.load_input(INPUT_PATH)
    results = []
    for payslip in input_data['payslip']:
        emp_id = payslip['employee_id']
        month = payslip['month']
        # Aggregate all attendance records for this month
        attendance_records = [a for a in input_data.get('attendance', []) if a.get('month') == month]
        if attendance_records:
            # Sum numeric fields, keep others from the first record
            aggregated = dict(attendance_records[0])
            for key in aggregated:
                if isinstance(aggregated[key], (int, float)):
                    aggregated[key] = sum(a.get(key, 0) for a in attendance_records if isinstance(a.get(key, 0), (int, float)))
            attendance = aggregated
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
            
            # Check if any checks have missing fields
            missing_fields_results = [cr for cr in check_results if cr.get('missing_fields', [])]
            has_missing_fields = len(missing_fields_results) > 0
            
            # Collect all missing fields for this rule
            all_missing_fields = set()
            for cr in check_results:
                if cr.get('missing_fields'):
                    all_missing_fields.update(cr['missing_fields'])
            
            # Include rule in results if there are violations OR missing fields
            if violations or has_missing_fields:
                result = {
                    'rule_id': rule['rule_id'],
                    'employee_id': emp_id,
                    'period': month,
                    'violations': violations,
                    'total_underpaid_amount': penalty.get('total_underpaid_amount', 0.0),
                    'penalty_amount': penalty['penalty_amount'],
                    'check_results': check_results,  # Include all check results
                    'has_missing_fields': has_missing_fields,
                    'missing_fields': sorted(list(all_missing_fields))  # Add missing fields to result
                }
                
                # Add compliance status
                if has_missing_fields:
                    result['compliance_status'] = 'inconclusive'
                elif len(violations) == 0:
                    result['compliance_status'] = 'compliant'
                else:
                    result['compliance_status'] = 'violation'
                    
                results.append(result)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Report generated: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
