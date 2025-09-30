import os
import json
from loader import RuleLoader
from evaluator import RuleEvaluator
 # Removed PenaltyCalculator import

RULES_PATH = os.path.join(os.path.dirname(__file__), '../rules/labor_law_rules.json')
INPUT_PATH = os.path.join(os.path.dirname(__file__), '../data/sample_input.json')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../output/report_2024_07.json')

from dynamic_params import DynamicParams

def build_context(payslip, attendance, contract, employee=None):
    # Use dynamic param config to build context
    params = DynamicParams.load()
    context = {
        'payslip': payslip,
        'attendance': attendance,
        'contract': contract,
        'employee': employee
    }

    # Helper to coerce values based on type hints in dynamic params
    def _coerce_value(raw_value, param_type):
        if raw_value is None:
            return None  # Keep None as None for missing field detection
        if param_type == 'number':
            if isinstance(raw_value, (int, float)):
                return raw_value
            if isinstance(raw_value, str):
                v = raw_value.strip().replace(',', '')
                try:
                    return int(v) if '.' not in v else float(v)
                except Exception:
                    return raw_value
            return raw_value
        return raw_value
    # Hebrew mapping: create mirrored Hebrew-accessible section aliases
    hebrew_section_names = {
        'payslip': 'תלוש',
        'attendance': 'נוכחות',
        'contract': 'חוזה',
        'employee': 'עובד',
    }

    # Initialize Hebrew section dicts as direct aliases to underlying dicts
    for eng_section, heb_section in hebrew_section_names.items():
        section_data = locals()[eng_section] if eng_section in locals() else None
        context[heb_section] = section_data if isinstance(section_data, dict) else {}

    # Populate section dictionaries with coerced param values under PARAM names (no label-based keys)
    for section in ['payslip', 'attendance', 'contract', 'employee']:
        section_data = locals()[section] if section in locals() and locals()[section] is not None else {}
        heb_section = hebrew_section_names[section]
        for p in params.get(section, []):
            param_name = p['param']
            param_type = p.get('type', 'number')
            raw_val = section_data.get(param_name, None)
            coerced_val = _coerce_value(raw_val, param_type)
            if isinstance(context.get(section), dict):
                context[section][param_name] = coerced_val
            if isinstance(context.get(heb_section), dict):
                context[heb_section][param_name] = coerced_val

    # Add employee_id and month for legacy compatibility
    context['employee_id'] = context.get('employee_id', payslip.get('מזהה_עובד', None))
    context['month'] = context.get('month', payslip.get('חודש', None))
    return context

def main():
    rules_data = RuleLoader.load_rules(RULES_PATH)
    input_data = RuleLoader.load_input(INPUT_PATH)
    results = []
    for payslip in input_data['payslip']:
        emp_id = payslip['מזהה_עובד']
        month = payslip['חודש']
        # Aggregate all attendance records for this month
        attendance_records = [a for a in input_data.get('attendance', []) if a.get('חודש') == month]
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
        employee = next(iter(input_data.get('employee', [])), {})
        context = build_context(payslip, attendance, contract, employee)
        for rule in rules_data['rules']:
            if not RuleEvaluator.is_rule_applicable(rule, month):
                continue
            check_results, named_results = RuleEvaluator.evaluate_checks(rule['checks'], context)
            violations = [cr for cr in check_results if cr['amount'] >= 0]

            # Dynamically calculate total_amount_owed
            total_amount_owed = sum(cr['amount'] for cr in check_results if cr['amount'] >= 0)

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
                    'total_amount_owed': total_amount_owed,
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
