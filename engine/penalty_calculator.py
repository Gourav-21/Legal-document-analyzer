from simpleeval import simple_eval
from typing import List, Dict, Any

class PenaltyCalculator:
    @staticmethod
    def calculate_penalty(penalty_exprs: List[str], check_results: List[Dict[str, Any]], named_results: dict = None) -> Dict[str, float]:
        if named_results is None:
            named_results = {}
        
        allowed_functions = {"min": min, "max": max, "abs": abs, "round": round}
        
        local_vars = {
            'check_results': [r['amount'] for r in check_results],
            'total_amount_owed': 0.0,
            'penalty_amount': 0.0,
            **named_results
        }
        for expr in penalty_exprs:
            if '=' in expr:
                var, formula = expr.split('=', 1)
                var = var.strip()
                formula = formula.strip()
                try:
                    local_vars[var] = simple_eval(formula, names=local_vars, functions=allowed_functions)
                except Exception as e:
                    print(f"[Penalty Eval Error] {expr}: {e}")
                    local_vars[var] = 0.0
        return {
            'total_amount_owed': local_vars.get('total_amount_owed', 0.0),
            'penalty_amount': local_vars.get('penalty_amount', 0.0)
        }
