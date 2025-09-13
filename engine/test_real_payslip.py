import json
from loader import RuleLoader
from evaluator import RuleEvaluator
from main import build_context


def analyze_real_payslip():
    """Analyze a real payslip and explain all violations found"""
    
    print("=" * 80)
    print("REAL PAYSLIP ANALYSIS")
    print("=" * 80)
    
    # Load the real payslip data
    input_data = RuleLoader.load_input("data/real_payslip_test.json")
    rules_data = RuleLoader.load_rules("rules/labor_law_rules.json")

    payslip = input_data['payslip'][0]
    attendance = input_data['attendance'][0]
    contract = input_data['contract'][0]
    context = build_context(payslip, attendance, contract)

    total_violations = 0
    violation_details = []

    for rule in rules_data['rules']:
        if not RuleEvaluator.is_rule_applicable(rule, payslip['month']):
            print(f"Rule {rule['name']} not applicable for this period")
            continue

        check_results, named_results = RuleEvaluator.evaluate_checks(rule['checks'], context)
        total_amount_owed = sum(cr.get('amount', 0) for cr in check_results if cr.get('amount', 0) >= 0)

        has_violations = False
        for check_idx, (check, result) in enumerate(zip(rule['checks'], check_results)):
            print(f"\n  📝 Check {check_idx + 1}:")
            print(f"     Condition: {check['condition']}")

            # Evaluate condition for display
            try:
                from simpleeval import simple_eval
                from evaluator import DotDict
                context_dict = DotDict(context)
                allowed_functions = {"min": min, "max": max, "abs": abs, "round": round}
                condition_result = simple_eval(check['condition'], names=context_dict, functions=allowed_functions)
                print(f"     Condition Result: {condition_result}")
            except Exception as e:
                print(f"     Condition Result: Error - {e}")

            if result.get('amount', 0) >= 0:
                has_violations = True
                print(f"     ⚠️  VIOLATION FOUND!")
                print(f"     Underpaid Amount: ₪{result.get('amount',0):,.2f}")
                print(f"     Message: {result.get('message')}")

                # Show calculation details
                print(f"     Calculation: {check.get('amount_owed', 'N/A')}")

                # Try to show step-by-step calculation
                try:
                    violation_expr = check.get('amount_owed', '0')
                    print(f"     Step-by-step:")
                    # For overtime rules, show the detailed calculation
                    if 'overtime' in rule['rule_id']:
                        expected_rate = None
                        if 'first_2h' in rule['rule_id']:
                            expected_rate = contract['hourly_rate'] * 1.25
                            print(f"       Expected Rate: ₪{contract['hourly_rate']} × 1.25 = ₪{expected_rate}")
                        elif 'after_2h' in rule['rule_id']:
                            expected_rate = contract['hourly_rate'] * 1.5
                            print(f"       Expected Rate: ₪{contract['hourly_rate']} × 1.5 = ₪{expected_rate}")

                        if expected_rate:
                            print(f"       Actual Rate Paid: ₪{payslip['overtime_rate']}")
                            print(f"       Rate Difference: ₪{expected_rate - payslip['overtime_rate']:.2f} per hour")

                    elif 'minimum_wage' in rule['rule_id']:
                        effective_rate = payslip['base_salary'] / attendance['total_hours']
                        print(f"       Effective Hourly Rate: ₪{payslip['base_salary']} ÷ {attendance['total_hours']} = ₪{effective_rate:.2f}")
                        print(f"       Minimum Required: ₪32.70")
                        print(f"       Shortfall per hour: ₪{32.7 - effective_rate:.2f}")

                except Exception as e:
                    print(f"     Calculation details: {e}")

            else:
                print(f"     ✅ No violation")

        if has_violations:
            total_violations += total_amount_owed
            print(f"\n  💸 AMOUNT OWED CALCULATION:")
            print(f"     Total Amount Owed: ₪{total_amount_owed:,.2f}")
            violation_details.append({
                'rule': rule['name'],
                'rule_id': rule['rule_id'],
                'amount_owed': total_amount_owed,
                'checks': [cr for cr in check_results if cr.get('amount', 0) >= 0]
            })
        else:
            print(f"\n  ✅ No violations found for this rule")
        for check_idx, (check, result) in enumerate(zip(rule['checks'], check_results)):
            print(f"\n  📝 Check {check_idx + 1}:")
            print(f"     Condition: {check['condition']}")
            
            # Evaluate condition for display
            try:
                from simpleeval import simple_eval
                from evaluator import DotDict
                context_dict = DotDict(context)
                allowed_functions = {"min": min, "max": max, "abs": abs, "round": round}
                condition_result = simple_eval(check['condition'], names=context_dict, functions=allowed_functions)
                print(f"     Condition Result: {condition_result}")
            except Exception as e:
                print(f"     Condition Result: Error - {e}")
            
            if result.get('amount', 0) >= 0:
                has_violations = True
                print(f"     ⚠️  VIOLATION FOUND!")
                print(f"     Underpaid Amount: ₪{result['amount']:,.2f}")
                print(f"     Message: {result['message']}")
                
                # Show calculation details
                print(f"     Calculation: {check.get('amount_owed', 'N/A')}")
                
                # Try to show step-by-step calculation
                try:
                    violation_expr = check.get('amount_owed', '0')
                    print(f"     Step-by-step:")
                    
                    # For overtime rules, show the detailed calculation
                    if 'overtime' in rule['rule_id']:
                        expected_rate = None
                        if 'first_2h' in rule['rule_id']:
                            expected_rate = contract['hourly_rate'] * 1.25
                            print(f"       Expected Rate: ₪{contract['hourly_rate']} × 1.25 = ₪{expected_rate}")
                        elif 'after_2h' in rule['rule_id']:
                            expected_rate = contract['hourly_rate'] * 1.5
                            print(f"       Expected Rate: ₪{contract['hourly_rate']} × 1.5 = ₪{expected_rate}")
                        
                        if expected_rate:
                            print(f"       Actual Rate Paid: ₪{payslip['overtime_rate']}")
                            print(f"       Rate Difference: ₪{expected_rate - payslip['overtime_rate']:.2f} per hour")
                            
                    elif 'minimum_wage' in rule['rule_id']:
                        effective_rate = payslip['base_salary'] / attendance['total_hours']
                        print(f"       Effective Hourly Rate: ₪{payslip['base_salary']} ÷ {attendance['total_hours']} = ₪{effective_rate:.2f}")
                        print(f"       Minimum Required: ₪32.70")
                        print(f"       Shortfall per hour: ₪{32.7 - effective_rate:.2f}")
                        
                except Exception as e:
                    print(f"     Calculation details: {e}")
                    
            else:
                print(f"     ✅ No violation")
        
        if has_violations:
            total_violations += total_amount_owed
            print(f"\n  💸 AMOUNT OWED CALCULATION:")
            print(f"     Total Amount Owed: ₪{total_amount_owed:,.2f}")
            violation_details.append({
                'rule': rule['name'],
                'rule_id': rule['rule_id'],
                'amount_owed': total_amount_owed,
                'checks': [cr for cr in check_results if cr.get('amount', 0) >= 0]
            })
        else:
            print(f"\n  ✅ No violations found for this rule")
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPLIANCE SUMMARY")
    print("=" * 80)
    
    if violation_details:
        print(f"\n❌ VIOLATIONS FOUND: {len(violation_details)} rule(s) violated")
        print(f"💰 Total Amount Owed: ₪{total_violations:,.2f}")
        print(f"\n📊 BREAKDOWN BY RULE:")
        for violation in violation_details:
            print(f"  • {violation['rule']}")
            print(f"    Amount Owed: ₪{violation['amount_owed']:,.2f}")
            for check in violation['checks']:
                print(f"    - {check['message']}")
        print(f"\n💡 RECOMMENDATIONS:")
        for violation in violation_details:
            if 'overtime' in violation['rule_id']:
                if 'first_2h' in violation['rule_id']:
                    correct_rate = contract['hourly_rate'] * 1.25
                    print(f"  • Pay first 2 overtime hours at ₪{correct_rate:.2f}/hour (125% rate)")
                elif 'after_2h' in violation['rule_id']:
                    correct_rate = contract['hourly_rate'] * 1.5
                    print(f"  • Pay overtime beyond 2 hours at ₪{correct_rate:.2f}/hour (150% rate)")
            elif 'minimum_wage' in violation['rule_id']:
                print(f"  • Increase hourly rate to at least ₪32.70 or reduce working hours")
    else:
        print("\n✅ FULLY COMPLIANT")
        print("No labor law violations found in this payslip.")
    
    return violation_details


if __name__ == "__main__":
    analyze_real_payslip()