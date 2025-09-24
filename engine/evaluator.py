from typing import Dict, Any, List, Set
from simpleeval import simple_eval
import datetime
import ast
import re

class DotDict(dict):
    def __getattr__(self, item):
        value = self.get(item)
        if isinstance(value, dict):
            return DotDict(value)
        return value

class VariableExtractor(ast.NodeVisitor):
    """Extract variable names from Python expressions using AST parsing"""
    def __init__(self):
        self.variables = set()
    
    def visit_Name(self, node):
        # Add variable names (excluding built-in functions)
        if node.id not in ('min', 'max', 'abs', 'round', 'len'):
            self.variables.add(node.id)
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        # Handle dot notation like payslip.overtime_rate, contract.hourly_rate
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            # Add the full dot notation path
            if len(parts) >= 2:
                self.variables.add('.'.join(reversed(parts)))
            # Also add the base variable for fallback
            self.variables.add(parts[-1])
        self.generic_visit(node)

class RuleEvaluator:
    @staticmethod
    def extract_variables_from_expression(expression: str) -> Set[str]:
        """Extract variable names from a Python expression using AST parsing"""
        try:
            # Strip whitespace to handle leading/trailing spaces that cause syntax errors
            expression = expression.strip()
            tree = ast.parse(expression, mode='eval')
            extractor = VariableExtractor()
            extractor.visit(tree)
            return extractor.variables
        except SyntaxError as e:
            print(f"[Variable Extract Error] Invalid expression: {expression}: {e}")
            return set()
    
    @staticmethod
    def find_missing_variables(expression: str, context: Dict[str, Any]) -> List[str]:
        """Find variables from expression that are missing in the context"""
        variables = RuleEvaluator.extract_variables_from_expression(expression)
        missing = []
        
        for var in variables:
            # Check if variable exists in context (including nested dot notation)
            if '.' in var:
                # Handle dot notation like 'payslip.overtime_rate'
                parts = var.split('.')
                current = context
                found = True
                try:
                    for part in parts:
                        if isinstance(current, dict) and part in current:
                            current = current[part]
                        else:
                            found = False
                            break
                    if not found:
                        missing.append(var)
                except (KeyError, TypeError, AttributeError):
                    missing.append(var)
            else:
                # Handle simple variable names
                if var not in context:
                    missing.append(var)
        
        return sorted(missing)
    
    @staticmethod
    def is_rule_applicable(rule: Dict[str, Any], date_str: str) -> bool:
        date = datetime.datetime.strptime(date_str, "%Y-%m")
        eff_from = datetime.datetime.strptime(rule["effective_from"], "%Y-%m-%d")
        eff_to = None
        if rule.get("effective_to"):
            eff_to = datetime.datetime.strptime(rule["effective_to"], "%Y-%m-%d")
        if eff_to:
            return eff_from <= date <= eff_to
        return eff_from <= date

    @staticmethod
    def evaluate_checks(checks: List[Dict[str, Any]], context: Dict[str, Any]) -> tuple:
        # Normalize context values (coerce numeric-like strings to numbers)
        def _coerce_value(v):
            # Convert numeric strings to int/float, leave others unchanged
            if isinstance(v, str):
                v_str = v.strip()
                # Try int
                try:
                    if v_str.isdigit() or (v_str.startswith('-') and v_str[1:].isdigit()):
                        return int(v_str)
                except Exception:
                    pass
                # Try float
                try:
                    return float(v_str)
                except Exception:
                    return v
            elif isinstance(v, dict):
                return {k: _coerce_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [_coerce_value(item) for item in v]
            else:
                return v

        normalized_context = {k: _coerce_value(v) for k, v in context.items()}

        # Wrap nested dicts for dot-access
        context = DotDict(normalized_context)
        allowed_functions = {"min": min, "max": max, "abs": abs, "round": round}
        results = []
        named_results = {}
        
        for idx, check in enumerate(checks):
            # Check for missing variables in condition
            condition_missing = RuleEvaluator.find_missing_variables(check.get("condition", ""), context)
            amount_missing = RuleEvaluator.find_missing_variables(check.get("amount_owed", ""), context)
            all_missing = sorted(set(condition_missing + amount_missing))
            
            # Initialize result structure
            result = {
                "message": "",
                "amount": 0.0,
                "missing_fields": all_missing,
                "condition_error": None,
                "amount_error": None
            }
            
            # If there are missing variables, mark as inconclusive but still try to evaluate
            if all_missing:
                result["message"] = f"Cannot evaluate - missing fields: {', '.join(all_missing)}"
                print(f"[Missing Variables] Check {idx}: {all_missing}")
            
            # Try to evaluate condition
            try:
                cond = simple_eval(check.get("condition", "False").strip(), names=context, functions=allowed_functions)
            except Exception as e:
                print(f"[Eval Error] Condition: {check.get('condition', 'N/A')}: {e}")
                cond = False
                result["condition_error"] = str(e)
                # If we couldn't evaluate due to missing vars, update the message
                if not all_missing:  # Only update if we haven't already identified missing vars
                    result["message"] = f"Condition evaluation failed: {str(e)}"
            
            # If condition is true, try to evaluate amount
            if cond:
                try:
                    amount = simple_eval(check.get("amount_owed", "0").strip(), names=context, functions=allowed_functions)
                    result["amount"] = amount
                    if not all_missing:  # Only set violation message if no missing fields
                        result["message"] = check.get("violation_message", "Violation detected")
                except Exception as e:
                    print(f"[Eval Error] Underpaid Amount: {check.get('amount_owed', 'N/A')}: {e}")
                    result["amount"] = 0.0
                    result["amount_error"] = str(e)
                    if not all_missing:  # Only update if we haven't already identified missing vars
                        result["message"] = f"Amount calculation failed: {str(e)}"
            
            results.append(result)
            
            # Support named checks
            check_id = check.get("id")
            if check_id:
                named_results[check_id] = result["amount"]
            else:
                named_results[f"check_{idx}"] = result["amount"]
                
        return results, named_results
