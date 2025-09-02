import streamlit as st
import json
import os
import pandas as pd

import sys
import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
import sys
import asyncio
import concurrent.futures

# Add engine directory to path
sys.path.append('engine')

from engine.loader import RuleLoader
from engine.evaluator import RuleEvaluator
from engine.penalty_calculator import PenaltyCalculator
from engine.main import build_context

st.set_page_config(
    page_title="Israeli Labor Law Compliance Engine", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
.violation-card {
    background-color: #ffebee;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #f44336;
    margin: 0.5rem 0;
}
.compliant-card {
    background-color: #e8f5e8;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4caf50;
    margin: 0.5rem 0;
}
.metric-card {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Israeli Labor Law Compliance Engine")
st.markdown("**Advanced Rule Engine with Real-time Violation Detection & Penalty Calculation**")

# Load data functions
@st.cache_data
def load_rules_data():
    try:
        return RuleLoader.load_rules('rules/labor_law_rules.json')
    except Exception as e:
        st.error(f"Error loading rules: {e}")
        return {"rules": []}

def refresh_rules_data():
    """Force refresh of rules data by clearing cache and reloading"""
    st.cache_data.clear()
    return load_rules_data()

@st.cache_data
def load_sample_data():
    try:
        return RuleLoader.load_input('data/sample_input.json')
    except Exception as e:
        return None

def save_rules_data(rules_data):
    """Save rules data to file"""
    try:
        with open('rules/labor_law_rules.json', 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)
        # Clear the cache after saving to ensure fresh data is loaded
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Error saving rules: {e}")
        return False




def test_single_rule(rule, test_payslip, test_attendance, test_contract):
    """Test a single rule against test data with detailed results"""
    context = build_context(test_payslip, test_attendance, test_contract)
    
    if not RuleEvaluator.is_rule_applicable(rule, test_payslip.get('month', datetime.now().strftime('%Y-%m'))):
        return {"applicable": False, "message": "Rule not applicable for this period"}
    
    try:
        check_results, named_results = RuleEvaluator.evaluate_checks(rule['checks'], context)
        penalty = PenaltyCalculator.calculate_penalty(rule['penalty'], check_results, named_results)
        
        violations = [cr for cr in check_results if cr['amount'] > 0]
        
        return {
            "applicable": True,
            "violations": violations,
            "check_results": check_results,
            "named_results": named_results,
            "penalty_calculation": penalty,
            "rule_checks": rule['checks'],
            "rule_penalty": rule['penalty'],
            "context_used": context,
            "total_underpaid_amount": penalty.get('total_underpaid_amount', 0.0),
            "penalty_amount": penalty.get('penalty_amount', 0.0),
            "compliant": len(violations) == 0
        }
    except Exception as e:
        return {"applicable": True, "error": str(e), "context_used": context}

# Initialize session state
if 'rules_data' not in st.session_state:
    st.session_state.rules_data = None

# Initialize session state for new rule builder
if 'new_rule_checks' not in st.session_state:
    st.session_state.new_rule_checks = []

if 'new_rule_penalties' not in st.session_state:
    st.session_state.new_rule_penalties = []

# Initialize session state for Hebrew new rule builder
if 'new_rule_checks_heb' not in st.session_state:
    st.session_state.new_rule_checks_heb = []

if 'new_rule_penalties_heb' not in st.session_state:
    st.session_state.new_rule_penalties_heb = []

# Tab navigation for better UX (Dashboard removed)
# Combined tab1 and tab4 for comprehensive analysis
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Payslip Analysis & Testing", 
    "⚖️ Rule Management",
    "📋 ניתוח תלושי שכר ובדיקה",
    "⚖️ ניהול כללים"
])

# Payslip Analysis Tab
with tab1:
  
    st.header("🎯 Analysis Types Comparison & Testing")
    st.markdown("**Test different analysis types with the same data to compare outputs**")
    

    # Test data setup section with input method selection
    st.subheader("📋 Test Data Setup")
    input_method = st.radio("Choose input method:", ["Manual Entry", "Upload JSON", "Use Sample Data"], key="test_input_method")

    test_payslip_data = None
    test_attendance_data = None
    test_contract_data = None

    if input_method == "Manual Entry":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Employee Information**")
            test_employee_id = st.text_input("Employee ID", value="TEST_001", key="test_emp_id")
            test_month = st.text_input("Month (YYYY-MM)", value="2024-01", key="test_month")
        with col2:
            st.markdown("**Data Sections to Include**")
            include_payslip = st.checkbox("Include Payslip Data", value=True, key="test_include_payslip")
            include_contract = st.checkbox("Include Contract Data", value=True, key="test_include_contract")
            include_attendance = st.checkbox("Include Attendance Data", value=True, key="test_include_attendance")

        if include_payslip:
            st.markdown("---")
            st.markdown("### 💰 Payslip Details")
            col1, col2 = st.columns(2)
            with col1:
                test_base_salary = st.number_input("Base Salary (₪)", min_value=0.0, value=5000.0, step=100.0, key="test_base_salary")
                test_overtime_rate = st.number_input("Overtime Rate Paid (₪/hour)", min_value=0.0, value=30.0, step=0.1, key="test_overtime_rate")
                test_overtime_pay = st.number_input("Total Overtime Pay (₪)", min_value=0.0, value=300.0, step=10.0, key="test_overtime_pay")
            with col2:
                test_hourly_rate = st.number_input("Contract Hourly Rate (₪)", min_value=0.0, value=26.88, step=0.1, key="test_hourly_rate")
                test_regular_hours = st.number_input("Regular Hours", min_value=0, value=186, step=1, key="test_regular_hours")
                test_overtime_hours = st.number_input("Overtime Hours", min_value=0, value=10, step=1, key="test_overtime_hours")
            test_payslip_data = {
                "employee_id": test_employee_id,
                "month": test_month,
                "base_salary": test_base_salary,
                "overtime_hours": test_overtime_hours,
                "overtime_pay": test_overtime_pay,
                "overtime_rate": test_overtime_rate,
                "total_salary": test_base_salary + test_overtime_pay,
                "hours_worked": test_regular_hours + test_overtime_hours,
                "hourly_rate": test_hourly_rate
            }
        if include_attendance:
            st.markdown("---")
            st.markdown("### ⏰ Attendance Details")
            col1, col2 = st.columns(2)
            with col1:
                test_regular_hours = st.number_input("Regular Hours (Attendance)", min_value=0, value=186, step=1, key="test_attendance_regular_hours")
                test_overtime_hours = st.number_input("Overtime Hours (Attendance)", min_value=0, value=10, step=1, key="test_attendance_overtime_hours")
            with col2:
                test_days_worked = st.number_input("Days Worked", min_value=0, value=22, step=1, key="test_days_worked")
                test_sick_days = st.number_input("Sick Days", min_value=0, value=0, step=1, key="test_sick_days")
                test_vacation_days = st.number_input("Vacation Days", min_value=0, value=0, step=1, key="test_vacation_days")
            test_attendance_data = {
                "employee_id": test_employee_id,
                "month": test_month,
                "days_worked": test_days_worked,
                "regular_hours": test_regular_hours,
                "overtime_hours": test_overtime_hours,
                "total_hours": test_regular_hours + test_overtime_hours,
                "sick_days": test_sick_days,
                "vacation_days": test_vacation_days
            }
        if include_contract:
            st.markdown("---")
            st.markdown("### 📋 Contract Details")
            col1, col2 = st.columns(2)
            with col1:
                test_hourly_rate = st.number_input("Hourly Rate (Contract)", min_value=0.0, value=26.88, step=0.1, key="test_contract_hourly_rate")
                test_minimum_wage_monthly = st.number_input("Minimum Wage Monthly", min_value=0.0, value=5300.0, step=10.0, key="test_minimum_wage_monthly")
                test_minimum_wage_hourly = st.number_input("Minimum Wage Hourly", min_value=0.0, value=29.12, step=0.1, key="test_minimum_wage_hourly")
            with col2:
                test_overtime_rate_125 = st.number_input("Overtime Rate 125%", min_value=0.0, value=1.25, step=0.01, key="test_overtime_rate_125")
                test_overtime_rate_150 = st.number_input("Overtime Rate 150%", min_value=0.0, value=1.50, step=0.01, key="test_overtime_rate_150")
                test_standard_hours_per_month = st.number_input("Standard Hours Per Month", min_value=0, value=186, step=1, key="test_standard_hours_per_month")
                test_standard_hours_per_day = st.number_input("Standard Hours Per Day", min_value=0, value=8, step=1, key="test_standard_hours_per_day")
                test_vacation_days_per_year = st.number_input("Vacation Days Per Year", min_value=0, value=14, step=1, key="test_vacation_days_per_year")
                test_sick_days_per_year = st.number_input("Sick Days Per Year", min_value=0, value=18, step=1, key="test_sick_days_per_year")
            test_contract_data = {
                "employee_id": test_employee_id,
                "minimum_wage_monthly": test_minimum_wage_monthly,
                "minimum_wage_hourly": test_minimum_wage_hourly,
                "hourly_rate": test_hourly_rate,
                "overtime_rate_125": test_overtime_rate_125,
                "overtime_rate_150": test_overtime_rate_150,
                "standard_hours_per_month": test_standard_hours_per_month,
                "standard_hours_per_day": test_standard_hours_per_day,
                "vacation_days_per_year": test_vacation_days_per_year,
                "sick_days_per_year": test_sick_days_per_year
            }

    elif input_method == "Upload JSON":
        uploaded_file = st.file_uploader("Upload payslip JSON file", type=['json'], key="test_uploaded_file")
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                test_payslip_data = data.get('payslip', [{}])[0]
                test_attendance_data = data.get('attendance', [{}])[0]
                test_contract_data = data.get('contract', [{}])[0]
                st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    elif input_method == "Use Sample Data":
        sample_data = load_sample_data()
        if sample_data:
            test_payslip_data = sample_data['payslip'][0]
            test_attendance_data = sample_data['attendance'][0]
            test_contract_data = sample_data['contract'][0]
            st.info("Using sample data for analysis")
        else:
            st.error("No sample data available")
    
    # Show test data preview
    with st.expander("📋 Preview Test Data", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Payslip Data:**")
            st.json(test_payslip_data)
        with col2:
            st.markdown("**Attendance Data:**")
            st.json(test_attendance_data)
        with col3:
            st.markdown("**Contract Data:**")
            st.json(test_contract_data)

    # Analysis types to test
    analysis_types = [
        ("violations_list", "📋 Simple Violations List"),
        ("easy", "😊 User-Friendly Summary"),
        ("table", "📊 Organized Table Format"),
        ("violation_count_table", "📈 Statistics Table"),
        ("combined", "⚖️ Detailed Legal Analysis"),
        ("report", "📄 Employer Report")
    ]

    st.markdown("---")
    st.subheader("🚀 Test All Analysis Types")
    
    # Test all analysis types
    if st.button("🚀 Test All Analysis Types", type="primary", key="test_all_analysis_types_combined"):
        st.markdown("## 📊 Analysis Results Comparison")
        for analysis_type, description in analysis_types:
            with st.expander(f"{description} ({analysis_type})", expanded=False):
                with st.spinner(f"Running {analysis_type} analysis..."):
                    try:
                        # Import DocumentProcessor
                        from document_processor_pydantic_ques import DocumentProcessor
                        processor = DocumentProcessor()
                        
                        # Convert single dicts to lists for create_report_with_rule_engine
                        payslip_list = [test_payslip_data] if test_payslip_data else []
                        attendance_list = [test_attendance_data] if test_attendance_data else []
                        
                        # Call create_report_with_rule_engine
                        result = asyncio.run(processor.create_report_with_rule_engine(
                            payslip_data=payslip_list,
                            attendance_data=attendance_list,
                            contract_data=test_contract_data,
                            analysis_type=analysis_type
                        ))
                        
                        # Show summary metrics for the new format
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            if result.get('violations_count', 0) > 0:
                                st.metric("Status", "Violations")
                            elif result.get('inconclusive_count', 0) > 0:
                                st.metric("Status", "Inconclusive")
                            else:
                                st.metric("Status", "Compliant")
                        with col2:
                            st.metric("Violations", result.get('violations_count', 0))
                        with col3:
                            st.metric("Inconclusive", result.get('inconclusive_count', 0))
                        with col4:
                            if 'total_combined' in result:
                                combined_total = result['total_combined']
                            else:
                                total_underpaid = result.get('total_underpaid', result.get('total_underpaid_amount', 0.0))
                                total_penalties = result.get('total_penalties', result.get('penalty_amount', 0.0))
                                combined_total = total_underpaid + total_penalties
                            st.metric("Total (Underpaid + Penalties)", f"₪{combined_total:,.2f}")
                        
                        # Show the formatted output
                        st.markdown("### 📋 Analysis Output:")
                        if 'legal_analysis' in result:
                            if analysis_type in ["table", "violation_count_table"]:
                                st.code(result['legal_analysis'], language="")
                            else:
                                st.markdown(result['legal_analysis'])
                        else:
                            st.info("Analysis completed but no formatted output available.")
                        
                        # Add download button for each analysis
                        if 'legal_analysis' in result:
                            st.download_button(
                                label=f"📥 Download {analysis_type} Report",
                                data=result['legal_analysis'],
                                file_name=f"analysis_{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                key=f"download_{analysis_type}_combined"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error running {analysis_type} analysis: {str(e)}")

    # Individual analysis type testing
    st.markdown("---")
    st.subheader("🔍 Test Individual Analysis Type")

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_test_type = st.selectbox(
            "Choose Analysis Type to Test:",
            options=[at[0] for at in analysis_types],
            format_func=lambda x: next(desc for code, desc in analysis_types if code == x),
            key="individual_analysis_type_combined"
        )

    with col2:
        test_individual_button = st.button("🧪 Test Selected Type", key="test_individual_type_combined")

    # Display results outside of columns for full width
    if test_individual_button:
        with st.spinner(f"Running {selected_test_type} analysis..."):
            try:
                # Import DocumentProcessor
                from document_processor_pydantic_ques import DocumentProcessor
                processor = DocumentProcessor()

                # Convert single dicts to lists for create_report_with_rule_engine
                payslip_list = [test_payslip_data] if test_payslip_data else []
                attendance_list = [test_attendance_data] if test_attendance_data else []

                # Call create_report_with_rule_engine
                result = asyncio.run(processor.create_report_with_rule_engine(
                    payslip_data=payslip_list,
                    attendance_data=attendance_list,
                    contract_data=test_contract_data,
                    analysis_type=selected_test_type
                ))

                st.success(f"✅ {selected_test_type} analysis completed!")

                # Show detailed results using the new format
                st.markdown("### 📊 Detailed Results:")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if result.get('violations_count', 0) > 0:
                        st.metric("Status", "Violations Found")
                    elif result.get('inconclusive_count', 0) > 0:
                        st.metric("Status", "Inconclusive")
                    else:
                        st.metric("Status", "Compliant")
                with col2:
                    st.metric("Violations Found", result.get('violations_count', 0))
                with col3:
                    st.metric("Inconclusive Cases", result.get('inconclusive_count', 0))
                with col4:
                    if 'total_combined' in result:
                        combined_total = result['total_combined']
                    else:
                        total_underpaid = result.get('total_underpaid', result.get('total_underpaid_amount', 0.0))
                        total_penalties = result.get('total_penalties', result.get('penalty_amount', 0.0))
                        combined_total = total_underpaid + total_penalties
                    st.metric("Total (Underpaid + Penalties)", f"₪{combined_total:,.2f}")

                # Analysis output
                st.markdown("### 📋 Analysis Output:")
                if 'legal_analysis' in result:
                    if selected_test_type in ["table", "violation_count_table"]:
                        st.code(result['legal_analysis'], language="")
                    else:
                        st.markdown(result['legal_analysis'])
                else:
                    st.info("Analysis completed but no formatted output available.")

                # Technical details
                with st.expander("🔧 Technical Details", expanded=False):
                    st.markdown("**Analysis Result:**")
                    st.json(result)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.code(str(e))

with tab2 :
    st.header("⚖️ Labor Law Rules Management")
    
    # Load rules data fresh each time to ensure we have latest changes
    rules_data = load_rules_data()
    
    # Display current rules in a more organized way
    st.subheader("📋 Current Rules")
    
    if rules_data['rules']:
        # Create a summary table first
        rules_summary = []
        for rule in rules_data['rules']:
            rules_summary.append({
                'Rule ID': rule['rule_id'],
                'Name': rule['name'],
                'Law Reference': rule['law_reference'],
                'Effective From': rule['effective_from'],
                'Effective To': rule.get('effective_to', 'Ongoing'),
                'Checks': len(rule['checks'])
            })
        
        rules_df = pd.DataFrame(rules_summary)
        st.dataframe(rules_df, use_container_width=True)
        
        # Detailed view with better organization
        st.markdown("### 🔍 Detailed Rule View")
        if len(rules_data['rules']) > 0:
            selected_rule = st.selectbox(
                "Select a rule to view details:",
                options=range(len(rules_data['rules'])),
                format_func=lambda x: f"{rules_data['rules'][x]['rule_id']} - {rules_data['rules'][x]['name']}"
            )
        else:
            selected_rule = None
            st.info("No rules available to select.")
        
        if selected_rule is not None and selected_rule < len(rules_data['rules']):
            rule = rules_data['rules'][selected_rule]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Rule ID:** {rule['rule_id']}")
                st.markdown(f"**Name:** {rule['name']}")
                st.markdown(f"**Law Reference:** {rule['law_reference']}")
                st.markdown(f"**Description:** {rule['description']}")
                st.markdown(f"**Effective Period:** {rule['effective_from']} to {rule.get('effective_to', 'ongoing')}")
                
                st.markdown("**Checks:**")
                for j, check in enumerate(rule['checks'], 1):
                    with st.expander(f"Check {j}: {check.get('violation_message', 'No message')}"):
                        st.code(f"Check ID: {check.get('id', 'N/A')}", language="python")
                        st.code(f"Condition: {check['condition']}", language="python")
                        st.code(f"Underpaid Amount: {check['underpaid_amount']}", language="python")
                
                st.markdown("**Penalty Calculation:**")
                for penalty_line in rule['penalty']:
                    st.code(penalty_line, language="python")
            
            with col2:
                st.markdown("**Actions**")
                
                # Edit Rule
                if st.button("📝 Edit Rule", key=f"edit_{selected_rule}"):
                    st.session_state[f'editing_rule_{selected_rule}'] = True
                
                # Delete Rule
                if st.button("🗑️ Delete Rule", key=f"delete_{selected_rule}", type="secondary"):
                    if st.session_state.get(f'confirm_delete_{selected_rule}', False):
                        # Actually delete the rule
                        rule_id_to_delete = rule['rule_id']
                        rules_data['rules'].pop(selected_rule)
                        if save_rules_data(rules_data):
                            st.success(f"✅ Rule '{rule_id_to_delete}' deleted successfully!")
                            # Clear session state
                            st.session_state[f'confirm_delete_{selected_rule}'] = False
                            # Clear any editing states for all rules since indices may have changed
                            for key in list(st.session_state.keys()):
                                if key.startswith('editing_rule_') or key.startswith('testing_rule_') or key.startswith('confirm_delete_'):
                                    del st.session_state[key]
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete rule")
                    else:
                        st.session_state[f'confirm_delete_{selected_rule}'] = True
                        st.warning("⚠️ Click again to confirm deletion")
                
                # Test Rule
                if st.button("🧪 Test Rule", key=f"test_{selected_rule}"):
                    st.session_state[f'testing_rule_{selected_rule}'] = True
                
                # Cancel confirmations
                if st.session_state.get(f'confirm_delete_{selected_rule}', False):
                    if st.button("❌ Cancel Delete", key=f"cancel_delete_{selected_rule}"):
                        st.session_state[f'confirm_delete_{selected_rule}'] = False
                        st.rerun()
        
        # Edit Rule Form
        if selected_rule is not None and selected_rule < len(rules_data['rules']) and st.session_state.get(f'editing_rule_{selected_rule}', False):
            try:
                # Get the current rule data
                current_rule = rules_data['rules'][selected_rule]
                st.markdown("---")
                st.subheader(f"📝 Edit Rule: {current_rule['rule_id']}")
                
                with st.form(f"edit_rule_form_{selected_rule}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        edit_rule_id = st.text_input("Rule ID*", value=current_rule['rule_id'])
                        edit_name = st.text_input("Rule Name*", value=current_rule['name'])
                        edit_law_reference = st.text_input("Law Reference*", value=current_rule['law_reference'])
                        edit_description = st.text_area("Description*", value=current_rule['description'], height=80)
                    
                    with col2:
                        edit_effective_from = st.date_input("Effective From*", 
                                                           value=datetime.strptime(current_rule['effective_from'], '%Y-%m-%d').date())
                        edit_effective_to = st.text_input("Effective To", value=current_rule.get('effective_to', ''))
                        
                        st.markdown("**Available Functions:**")
                        st.code("min(), max(), abs(), round()")
                        
                        st.markdown("**Available Variables:**")
                        st.code("""
payslip.*, attendance.*, contract.*
employee_id, month, hourly_rate, 
overtime_hours, total_hours, etc.
                        """)
                    
                    # Initialize session state for edit rule
                    if f'edit_rule_checks_{selected_rule}' not in st.session_state:
                        st.session_state[f'edit_rule_checks_{selected_rule}'] = current_rule['checks'].copy()

                    if f'edit_rule_penalties_{selected_rule}' not in st.session_state:
                        st.session_state[f'edit_rule_penalties_{selected_rule}'] = current_rule['penalty'].copy()

                    # Check Management within edit form
                    st.markdown("**Rule Checks:**")

                    # Display current checks
                    if st.session_state[f'edit_rule_checks_{selected_rule}']:
                        st.markdown("**Current Checks:**")
                        for i, check in enumerate(st.session_state[f'edit_rule_checks_{selected_rule}']):
                            with st.expander(f"Check {i+1}: {check.get('violation_message', 'No message')}"):
                                st.code(f"Check ID: {check.get('id', 'N/A')}")
                                st.code(f"Condition: {check['condition']}")
                                st.code(f"Underpaid Amount: {check['underpaid_amount']}")
                                # Remove button for each check
                                if st.form_submit_button(f"🗑️ Remove Check {i+1}"):
                                    st.session_state[f'edit_rule_checks_{selected_rule}'].pop(i)

                    # Add new check inputs
                    st.markdown("**Add New Check:**")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        edit_new_check_id = st.text_input("Check ID", key=f"edit_new_check_id_{selected_rule}", help="Unique identifier for this check", placeholder="first_2h")
                        edit_new_condition = st.text_input("Condition", key=f"edit_new_condition_{selected_rule}", help="e.g., attendance.overtime_hours > 0", placeholder="attendance.overtime_hours > 0")
                        edit_new_underpaid_amount = st.text_input("Underpaid Amount Formula", key=f"edit_new_underpaid_amount_{selected_rule}", help="e.g., (contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)", placeholder="(contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)")
                    with col2:
                        edit_new_violation_message = st.text_input("Violation Message", key=f"edit_new_violation_message_{selected_rule}", help="e.g., Overtime rate violation", placeholder="Overtime rate violation")

                    # Penalty Management within edit form
                    st.markdown("**Penalty Calculation:**")

                    # Display current penalties
                    if st.session_state[f'edit_rule_penalties_{selected_rule}']:
                        st.markdown("**Current Penalty Lines:**")
                        for i, penalty in enumerate(st.session_state[f'edit_rule_penalties_{selected_rule}']):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.code(f"{i+1}. {penalty}")
                            with col2:
                                if st.form_submit_button(f"🗑️ Remove Penalty {i+1}"):
                                    st.session_state[f'edit_rule_penalties_{selected_rule}'].pop(i)

                    # Add new penalty line input
                    st.markdown("**Add New Penalty Line:**")
                    edit_new_penalty_line = st.text_input("Penalty Formula", key=f"edit_new_penalty_line_{selected_rule}", help="e.g., total_underpaid_amount = check_results[0]", placeholder="total_underpaid_amount = check_results[0] , penalty_amount= total_underpaid_amount * 5")

                    # Action buttons
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        edit_add_check_btn = st.form_submit_button("➕ Add Check")
                    with col2:
                        edit_add_penalty_btn = st.form_submit_button("➕ Add Penalty")
                    with col3:
                        edit_clear_all_btn = st.form_submit_button("🗑️ Clear All")
                    with col4:
                        edit_save_changes_btn = st.form_submit_button("💾 Save Changes", type="primary")

                    # Handle form submissions
                    if edit_add_check_btn:
                        if edit_new_check_id and edit_new_condition and edit_new_underpaid_amount and edit_new_violation_message:
                            st.session_state[f'edit_rule_checks_{selected_rule}'].append({
                                "id": edit_new_check_id,
                                "condition": edit_new_condition,
                                "underpaid_amount": edit_new_underpaid_amount,
                                "violation_message": edit_new_violation_message
                            })
                            st.success("✅ Check added successfully!")
                        else:
                            st.error("Please fill all fields for the check")

                    if edit_add_penalty_btn:
                        if edit_new_penalty_line:
                            st.session_state[f'edit_rule_penalties_{selected_rule}'].append(edit_new_penalty_line)
                            st.success("✅ Penalty line added successfully!")
                        else:
                            st.error("Please enter a penalty formula")

                    if edit_clear_all_btn:
                        st.session_state[f'edit_rule_checks_{selected_rule}'] = []
                        st.session_state[f'edit_rule_penalties_{selected_rule}'] = []
                        st.success("✅ All checks and penalties cleared!")

                    if edit_save_changes_btn:
                        if not all([edit_rule_id, edit_name, edit_law_reference, edit_description]):
                            st.error("❌ Please fill in all required fields")
                        else:
                            try:
                                # Check if rule ID already exists (excluding current rule)
                                existing_ids = [r['rule_id'] for i, r in enumerate(rules_data['rules']) if i != selected_rule]
                                if edit_rule_id in existing_ids:
                                    st.error(f"❌ Rule ID '{edit_rule_id}' already exists. Please choose a different ID.")
                                else:
                                    # Use session state lists
                                    edit_checks_json = st.session_state[f'edit_rule_checks_{selected_rule}']
                                    edit_penalty_json = st.session_state[f'edit_rule_penalties_{selected_rule}']

                                    # Validate checks structure
                                    validation_passed = True
                                    if not edit_checks_json:
                                        st.error("❌ Please add at least one check")
                                        validation_passed = False
                                    if not edit_penalty_json:
                                        st.error("❌ Please add at least one penalty line")
                                        validation_passed = False

                                    for i, check in enumerate(edit_checks_json):
                                        required_fields = ['id', 'condition', 'underpaid_amount', 'violation_message']
                                        missing_fields = [f for f in required_fields if f not in check]
                                        if missing_fields:
                                            st.error(f"❌ Check {i+1} missing required fields: {missing_fields}")
                                            validation_passed = False

                                    if not validation_passed:
                                        st.stop()

                                    # Update rule
                                    updated_rule = {
                                        "rule_id": edit_rule_id,
                                        "name": edit_name,
                                        "law_reference": edit_law_reference,
                                        "description": edit_description,
                                        "effective_from": edit_effective_from.strftime('%Y-%m-%d'),
                                        "effective_to": edit_effective_to if edit_effective_to else None,
                                        "checks": edit_checks_json,
                                        "penalty": edit_penalty_json,
                                        "created_date": current_rule.get('created_date', datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')),
                                        "updated_date": datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                                    }

                                    rules_data['rules'][selected_rule] = updated_rule
                                    if save_rules_data(rules_data):
                                        st.success(f"✅ Rule '{edit_rule_id}' updated successfully!")
                                        # Clear session state
                                        if f'edit_rule_checks_{selected_rule}' in st.session_state:
                                            del st.session_state[f'edit_rule_checks_{selected_rule}']
                                        if f'edit_rule_penalties_{selected_rule}' in st.session_state:
                                            del st.session_state[f'edit_rule_penalties_{selected_rule}']
                                        st.session_state[f'editing_rule_{selected_rule}'] = False
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to save rule")

                            except Exception as e:
                                st.error(f"❌ Error updating rule: {e}")

                    col1, col2 = st.columns(2)
                    with col1:
                        pass  # Save Changes button moved above
                    with col2:
                        if st.form_submit_button("❌ Cancel Edit"):
                            st.session_state[f'editing_rule_{selected_rule}'] = False
                            # Clear session state
                            if f'edit_rule_checks_{selected_rule}' in st.session_state:
                                del st.session_state[f'edit_rule_checks_{selected_rule}']
                            if f'edit_rule_penalties_{selected_rule}' in st.session_state:
                                del st.session_state[f'edit_rule_penalties_{selected_rule}']
                            st.rerun()
            except (IndexError, KeyError) as e:
                st.error(f"❌ Error accessing rule data: {e}")
                st.session_state[f'editing_rule_{selected_rule}'] = False
                st.rerun()
        
        # Test Rule Section
        if selected_rule is not None and selected_rule < len(rules_data['rules']) and st.session_state.get(f'testing_rule_{selected_rule}', False):
            try:
                # Get the current rule data
                current_rule = rules_data['rules'][selected_rule]
                st.markdown("---")
                st.subheader(f"🧪 Test Rule: {current_rule['rule_id']}")
                
                with st.form(f"test_rule_form_{selected_rule}"):
                    st.markdown("**Enter test data to validate this rule:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        test_employee_id = st.text_input("Employee ID", value="TEST_001")
                        test_month = st.text_input("Month (YYYY-MM)", value="2024-07")
                        test_hourly_rate = st.number_input("Hourly Rate", value=30.0, step=0.1)
                        test_base_salary = st.number_input("Base Salary", value=4800.0, step=10.0)
                    
                    with col2:
                        test_overtime_rate = st.number_input("Overtime Rate Paid", value=35.0, step=0.1)
                        test_overtime_hours = st.number_input("Overtime Hours", value=5, step=1)
                        test_regular_hours = st.number_input("Regular Hours", value=160, step=1)
                        test_total_hours = test_regular_hours + test_overtime_hours
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        run_test = st.form_submit_button("🚀 Run Test", type="primary")
                        if run_test:
                            test_payslip = {
                                "employee_id": test_employee_id,
                                "month": test_month,
                                "base_salary": test_base_salary,
                                "overtime_rate": test_overtime_rate
                            }
                            test_attendance = {
                                "employee_id": test_employee_id,
                                "month": test_month,
                                "overtime_hours": test_overtime_hours,
                                "total_hours": test_total_hours
                            }
                            test_contract = {
                                "employee_id": test_employee_id,
                                "hourly_rate": test_hourly_rate
                            }
                            test_result = test_single_rule(current_rule, test_payslip, test_attendance, test_contract)
                            if not test_result["applicable"]:
                                st.warning(f"⚠️ {test_result['message']}")
                            elif "error" in test_result:
                                st.error(f"❌ Test failed: {test_result['error']}")
                                if "context_used" in test_result:
                                    st.markdown("**Context data at time of error:**")
                                    st.json(test_result["context_used"])
                            elif test_result["compliant"]:
                                st.success("✅ Test passed - No violations found!")
                                # Show detailed calculation even for passing tests
                                with st.expander("📊 Show Calculation Details"):
                                    st.markdown("**Rule Checks Evaluated:**")
                                    for j, check in enumerate(test_result.get('rule_checks', [])):
                                        st.markdown(f"**Check {j+1}:** {check.get('violation_message', 'No message')}")
                                        st.code(f"Check ID: {check.get('id', 'N/A')}")
                                        st.code(f"Condition: {check['condition']}")
                                        st.code(f"Underpaid Amount: {check['underpaid_amount']}")
                                        if j < len(test_result.get('check_results', [])):
                                            check_result = test_result['check_results'][j]
                                            st.info(f"Result: Condition = {check_result.get('condition_result', 'N/A')}, Amount = ₪{check_result['amount']:.2f}")
                                        # Show calculation steps
                                        if 'calculation_steps' in check_result:
                                            for step in check_result['calculation_steps']:
                                                if step['step'] == 'formula_substitution':
                                                    st.code(f"With values: {step['formula']} = {step['result']}")
                                st.markdown("**Context Data Used:**")
                                st.json(test_result.get('context_used', {}))
                            else:
                                st.error(f"❌ Test found violations:")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Underpaid Amount", f"₪{test_result['total_underpaid_amount']:.2f}")
                                with col2:
                                    st.metric("Penalty Amount", f"₪{test_result['penalty_amount']:.2f}")
                                st.markdown("### 🔍 Detailed Violation Analysis")
                                # Show each check calculation
                                for j, check in enumerate(test_result.get('rule_checks', [])):
                                    st.markdown(f"#### Check {j+1}: {check.get('violation_message', 'No message')}")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Formula:**")
                                        st.code(f"Check ID: {check.get('id', 'N/A')}")
                                        st.code(f"Condition: {check['condition']}")
                                        st.code(f"Underpaid Amount: {check['underpaid_amount']}")
                                    with col2:
                                        if j < len(test_result.get('check_results', [])):
                                            check_result = test_result['check_results'][j]
                                            st.markdown("**Result:**")
                                            condition_met = check_result.get('condition_result', False)
                                            if condition_met:
                                                st.success("✅ Condition: TRUE")
                                            else:
                                                st.info("ℹ️ Condition: FALSE")
                                            amount = check_result['amount']
                                            if amount > 0:
                                                st.error(f"💰 Underpaid: ₪{amount:.2f}")
                                            else:
                                                st.success(f"💰 Amount: ₪{amount:.2f}")
                                    # Show calculation steps
                                    if j < len(test_result.get('check_results', [])):
                                        check_result = test_result['check_results'][j]
                                        if 'calculation_steps' in check_result:
                                            st.markdown("**Calculation Steps:**")
                                            for step in check_result['calculation_steps']:
                                                if step['step'] == 'condition_evaluation':
                                                    st.info(f"🔍 {step['description']}")
                                                elif step['step'] == 'amount_calculation':
                                                    st.success(f"💰 {step['description']}")
                                                elif step['step'] == 'formula_substitution':
                                                    st.code(f"With values: {step['formula']} = {step['result']}")
                                        # Show any errors
                                        if check_result.get('evaluation_error'):
                                            st.error(f"⚠️ Error: {check_result['evaluation_error']}")
                                        # Check for missing fields (from engine results)
                                        if 'missing_fields' in check_result and check_result['missing_fields']:
                                            st.warning("⚠️ **Missing Fields in Test Data:**")
                                            for field in check_result['missing_fields']:
                                                st.markdown(f"• `{field}` - Not found in test data")
                                    st.markdown("---")
                                # Show penalty calculation
                                st.markdown("### 💸 Penalty Calculation")
                                for penalty_line in test_result.get('rule_penalty', []):
                                    st.code(penalty_line)
                                penalty_calc = test_result.get('penalty_calculation', {})
                                # Show penalty calculation steps
                                if 'calculation_steps' in penalty_calc:
                                    st.markdown("**Penalty Calculation Steps:**")
                                    for step in penalty_calc['calculation_steps']:
                                        st.markdown(f"**{step['variable']}:**")
                                        st.code(f"Formula: {step['formula']}")
                                        st.code(f"With values: {step['substituted_formula']}")
                                        st.code(f"Result: {step['result']}")
                                # Show final amounts
                                for key, value in penalty_calc.items():
                                    if isinstance(value, (int, float)) and key in ['total_underpaid_amount', 'penalty_amount']:
                                        st.markdown(f"• **{key.replace('_', ' ').title()}:** ₪{value:.2f}")
                                # Show context data
                                st.markdown("### 📊 Test Data Used")
                                st.json(test_result.get('context_used', {}))
                                # Show violations summary
                                st.markdown("### ⚠️ Violations Found")
                                for violation in test_result["violations"]:
                                    st.markdown(f"• **{violation['message']}:** ₪{violation['amount']:.2f}")
                
                with col2:
                    if st.form_submit_button("❌ Close Test"):
                        st.session_state[f'testing_rule_{selected_rule}'] = False
                        st.rerun()
            except (IndexError, KeyError) as e:
                st.error(f"❌ Error accessing rule data: {e}")
                st.session_state[f'testing_rule_{selected_rule}'] = False
                st.rerun()
    else:
        st.info("No rules found. Add a new rule below.")
    
    # Bulk operations
    # if rules_data['rules']:
    #     st.markdown("---")
    #     st.subheader("🔧 Bulk Operations")
        
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         if st.button("📤 Export All Rules", key="export_all_rules"):
    #             rules_json = json.dumps(rules_data, indent=2, ensure_ascii=False)
    #             st.download_button(
    #                 label="💾 Download Rules JSON",
    #                 data=rules_json,
    #                 file_name=f"labor_law_rules_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    #                 mime="application/json"
    #             )
        
    #     with col2:
    #         uploaded_rules = st.file_uploader("📥 Import Rules", type=['json'], key="import_rules")
    #         if uploaded_rules:
    #             try:
    #                 imported_data = json.load(uploaded_rules)
    #                 if 'rules' in imported_data and isinstance(imported_data['rules'], list):
    #                     st.success(f"✅ Found {len(imported_data['rules'])} rules to import")
                        
    #                     if st.button("🔄 Replace All Rules", key="replace_all_rules"):
    #                         if save_rules_data(imported_data):
    #                             st.success("✅ Rules imported successfully!")
    #                             st.cache_data.clear()
    #                             st.rerun()
    #                         else:
    #                             st.error("❌ Failed to import rules")
    #                 else:
    #                     st.error("❌ Invalid rules file format")
    #             except Exception as e:
    #                 st.error(f"❌ Error reading file: {e}")
        
    #     with col3:
    #         if st.button("🧪 Test All Rules", key="test_all_rules"):
    #             st.session_state['testing_all_rules'] = True
        
    #     # Test all rules functionality
    #     if st.session_state.get('testing_all_rules', False):
    #         st.markdown("---")
    #         st.subheader("🧪 Test All Rules")
            
    #         with st.form("test_all_rules_form"):
    #             st.markdown("**Enter test data to validate all rules:**")
                
    #             col1, col2 = st.columns(2)
                
    #             with col1:
    #                 test_all_employee_id = st.text_input("Employee ID", value="TEST_ALL")
    #                 test_all_month = st.text_input("Month (YYYY-MM)", value="2024-07")
    #                 test_all_hourly_rate = st.number_input("Hourly Rate", value=30.0, step=0.1)
    #                 test_all_base_salary = st.number_input("Base Salary", value=4800.0, step=10.0)
                
    #             with col2:
    #                 test_all_overtime_rate = st.number_input("Overtime Rate Paid", value=35.0, step=0.1)
    #                 test_all_overtime_hours = st.number_input("Overtime Hours", value=5, step=1)
    #                 test_all_regular_hours = st.number_input("Regular Hours", value=160, step=1)
                
    #             col1, col2 = st.columns(2)
    #             with col1:
    #                 if st.form_submit_button("🚀 Test All Rules", type="primary"):
    #                     test_payslip = {
    #                         "employee_id": test_all_employee_id,
    #                         "month": test_all_month,
    #                         "base_salary": test_all_base_salary,
    #                         "overtime_rate": test_all_overtime_rate
    #                     }
                        
    #                     test_attendance = {
    #                         "employee_id": test_all_employee_id,
    #                         "month": test_all_month,
    #                         "overtime_hours": test_all_overtime_hours,
    #                         "total_hours": test_all_regular_hours + test_all_overtime_hours
    #                     }
                        
    #                     test_contract = {
    #                         "employee_id": test_all_employee_id,
    #                         "hourly_rate": test_all_hourly_rate
    #                     }
                        
    #                     st.markdown("### 📊 Test Results:")
                        
    #                     total_violations = 0
    #                     total_penalties = 0
                        
    #                     for i, rule in enumerate(rules_data['rules']):
    #                         with st.expander(f"Rule {i+1}: {rule['rule_id']} - {rule['name']}"):
    #                             test_result = test_single_rule(rule, test_payslip, test_attendance, test_contract)
                                
    #                             if not test_result["applicable"]:
    #                                 st.warning(f"⚠️ {test_result['message']}")
    #                             elif "error" in test_result:
    #                                 st.error(f"❌ Test failed: {test_result['error']}")
    #                             elif test_result["compliant"]:
    #                                 st.success("✅ No violations found")
    #                             else:
    #                                 st.error("❌ Violations found:")
    #                                 col1, col2 = st.columns(2)
    #                                 with col1:
    #                                     st.metric("Underpaid", f"₪{test_result['total_underpaid_amount']:.2f}")
    #                                 with col2:
    #                                     st.metric("Penalty", f"₪{test_result['penalty_amount']:.2f}")
                                    
    #                                 for violation in test_result["violations"]:
    #                                     st.markdown(f"• {violation['message']} (₪{violation['amount']:.2f})")
                                    
    #                                 total_violations += test_result['total_underpaid_amount']
    #                                 total_penalties += test_result['penalty_amount']
                        
    #                     if total_violations > 0:
    #                         st.markdown("### 💰 Total Impact:")
    #                         col1, col2, col3 = st.columns(3)
    #                         with col1:
    #                             st.metric("Total Underpaid", f"₪{total_violations:.2f}")
    #                         with col2:
    #                             st.metric("Total Penalties", f"₪{total_penalties:.2f}")
    #                         with col3:
    #                             st.metric("Total Exposure", f"₪{total_violations + total_penalties:.2f}")
    #                     else:
    #                         st.success("🎉 All rules passed! No violations found.")
                
    #             with col2:
    #                 if st.form_submit_button("❌ Close Test"):
    #                     st.session_state['testing_all_rules'] = False
    #                     st.rerun()
    
    # st.markdown("---")
    
    # Add new rule section
    st.subheader("➕ Add New Rule")
    
    # Form for rule creation with integrated check/penalty management
    with st.form("new_rule_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            rule_id = st.text_input("Rule ID*", help="Unique identifier")
            name = st.text_input("Rule Name*", help="Human-readable name")
            law_reference = st.text_input("Law Reference*", help="e.g., Section 16A")
            description = st.text_area("Description*", help="What this rule checks for", height=80)
        
        with col2:
            effective_from = st.date_input("Effective From*")
            effective_to = st.text_input("Effective To", help="YYYY-MM-DD or leave blank")
            
            st.markdown("**Available Functions:**")
            st.code("min(), max(), abs(), round()")
            
            st.markdown("**Available Variables:**")
            st.code("""
payslip.*, attendance.*, contract.*
employee_id, month, hourly_rate, 
overtime_hours, total_hours, etc.
            """)
    
        # Check Management within form
        st.markdown("**Rule Checks:**")
        
        # Display current checks
        if st.session_state.new_rule_checks:
            st.markdown("**Current Checks:**")
            for i, check in enumerate(st.session_state.new_rule_checks):
                with st.expander(f"Check {i+1}: {check.get('violation_message', 'No message')}"):
                    st.code(f"Check ID: {check.get('id', 'N/A')}")
                    st.code(f"Condition: {check['condition']}")
                    st.code(f"Underpaid Amount: {check['underpaid_amount']}")
                    # Note: Remove functionality moved to form submit buttons
        
        # Add new check inputs
        st.markdown("**Add New Check:**")
        col1, col2 = st.columns([1, 1])
        with col1:
            new_check_id = st.text_input("Check ID", key="new_check_id", help="Unique identifier for this check", placeholder="first_2h")
            new_condition = st.text_input("Condition", key="new_condition", help="e.g., attendance.overtime_hours > 0", placeholder="attendance.overtime_hours > 0")
            new_underpaid_amount = st.text_input("Underpaid Amount Formula", key="new_underpaid_amount", help="e.g., (contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)", placeholder="(contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)")
        with col2:
            new_violation_message = st.text_input("Violation Message", key="new_violation_message", help="e.g., Overtime rate violation", placeholder="Overtime rate violation")
        
        # Penalty Management within form
        st.markdown("**Penalty Calculation:**")
        
        # Display current penalties
        if st.session_state.new_rule_penalties:
            st.markdown("**Current Penalty Lines:**")
            for i, penalty in enumerate(st.session_state.new_rule_penalties):
                st.code(f"{i+1}. {penalty}")
        
        # Add new penalty line input
        st.markdown("**Add New Penalty Line:**")
        new_penalty_line = st.text_input("Penalty Formula", key="new_penalty_line", help="e.g., total_underpaid_amount = check_results[0]", placeholder="total_underpaid_amount = check_results[0] , penalty_amount= total_underpaid_amount * 5" )
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            add_check_btn = st.form_submit_button("➕ Add Check")
        with col2:
            add_penalty_btn = st.form_submit_button("➕ Add Penalty")
        with col3:
            clear_all_btn = st.form_submit_button("🗑️ Clear All")
        with col4:
            submit_rule_btn = st.form_submit_button("✅ Create Rule", type="primary")
        
        # Handle form submissions
        if add_check_btn:
            if new_check_id and new_condition and new_underpaid_amount and new_violation_message:
                st.session_state.new_rule_checks.append({
                    "id": new_check_id,
                    "condition": new_condition,
                    "underpaid_amount": new_underpaid_amount,
                    "violation_message": new_violation_message
                })
                st.success("✅ Check added successfully!")
                st.rerun()
            else:
                st.error("Please fill all fields for the check")
        
        if add_penalty_btn:
            if new_penalty_line:
                st.session_state.new_rule_penalties.append(new_penalty_line)
                st.success("✅ Penalty line added successfully!")
                st.rerun()
            else:
                st.error("Please enter a penalty formula")
        
        if clear_all_btn:
            st.session_state.new_rule_checks = []
            st.session_state.new_rule_penalties = []
            st.success("✅ All checks and penalties cleared!")
            st.rerun()
        
        if submit_rule_btn:
            if not all([rule_id, name, law_reference, description]):
                st.error("❌ Please fill in all required fields")
            else:
                try:
                    # Check if rule ID already exists
                    existing_ids = [r['rule_id'] for r in rules_data['rules']]
                    if rule_id in existing_ids:
                        st.error(f"❌ Rule ID '{rule_id}' already exists. Please choose a different ID.")
                    else:
                        # Use session state lists
                        checks_json = st.session_state.new_rule_checks
                        penalty_json = st.session_state.new_rule_penalties
                        
                        # Validate checks structure
                        validation_passed = True
                        if not checks_json:
                            st.error("❌ Please add at least one check")
                            validation_passed = False
                        if not penalty_json:
                            st.error("❌ Please add at least one penalty line")
                            validation_passed = False
                        
                        for i, check in enumerate(checks_json):
                            required_fields = ['id', 'condition', 'underpaid_amount', 'violation_message']
                            missing_fields = [f for f in required_fields if f not in check]
                            if missing_fields:
                                st.error(f"❌ Check {i+1} missing required fields: {missing_fields}")
                                validation_passed = False
                        
                        if not validation_passed:
                            st.stop()
                        
                        # Create new rule
                        new_rule = {
                            "rule_id": rule_id,
                            "name": name,
                            "law_reference": law_reference,
                            "description": description,
                            "effective_from": effective_from.strftime('%Y-%m-%d'),
                            "effective_to": effective_to if effective_to else None,
                            "checks": checks_json,
                            "penalty": penalty_json,
                            "created_date": datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                            "updated_date": datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                        }
                        
                        # Add to rules data
                        rules_data['rules'].append(new_rule)
                        
                        # Save to file
                        if save_rules_data(rules_data):
                            st.success(f"✅ Rule '{rule_id}' added successfully!")
                            st.balloons()
                            
                            # Clear the session state lists
                            st.session_state.new_rule_checks = []
                            st.session_state.new_rule_penalties = []
                            
                            # Show the new rule
                            with st.expander("📋 View Added Rule", expanded=True):
                                st.json(new_rule)
                            
                            # Auto-refresh to show the new rule
                            st.info("💡 The page will refresh automatically to show your new rule in the list above.")
                            st.rerun()
                        else:
                            st.error("❌ Failed to save rule to file")
                        
                except Exception as e:
                    st.error(f"❌ Error creating rule: {e}")
    
    # Rule validation helper
    st.markdown("---")
    st.subheader("🔍 Rule Validation Helper")
    
    with st.expander("📚 Rule Writing Guide"):
        st.markdown("""
        ### Available Variables in Conditions and Calculations:
        
        **Payslip Data:**
        - `payslip.base_salary` - Base monthly salary
        - `payslip.overtime_rate` - Overtime rate paid per hour
        - `payslip.overtime_pay` - Total overtime payment
        - `payslip.total_pay` - Total payment
        
        **Attendance Data:**
        - `attendance.regular_hours` - Regular working hours
        - `attendance.overtime_hours` - Overtime hours worked
        - `attendance.total_hours` - Total hours worked
        
        **Contract Data:**
        - `contract.hourly_rate` - Contracted hourly rate
        - `contract.position` - Employee position
        
        **Flattened Access:**
        - You can also access fields directly: `overtime_hours`, `hourly_rate`, etc.
        
        ### Mathematical Functions:
        - `min(a, b)` - Minimum of two values
        - `max(a, b)` - Maximum of two values
        - `abs(x)` - Absolute value
        - `round(x, n)` - Round to n decimal places
        - `sum(list)` - Sum of list values
        
        ### Example Conditions:
        ```python
        # Check if overtime hours exist
        attendance.overtime_hours > 0
        
        # Check if hourly rate is below minimum wage
        contract.hourly_rate < 32.7
        
        # Complex condition with multiple criteria
        attendance.overtime_hours > 2 and payslip.overtime_rate < (contract.hourly_rate * 1.5)
        ```
        
        ### Example Underpaid Amount Calculations:
        ```python
        # Simple overtime underpayment
        (contract.hourly_rate * 1.25 - payslip.overtime_rate) * attendance.overtime_hours
        
        # Minimum wage shortfall
        (32.7 - contract.hourly_rate) * attendance.total_hours
        
        # Tiered overtime calculation
        max(0, (contract.hourly_rate * 1.5 - payslip.overtime_rate) * max(attendance.overtime_hours - 2, 0))
        ```
        """)
    
    # Quick rule tester
    with st.expander("🧪 Quick Rule Expression Tester"):
        st.markdown("Test your expressions before adding them to a rule:")
        
        col1, col2 = st.columns(2)
        with col1:
            test_expr = st.text_input("Expression to test:", 
                                    value="contract.hourly_rate * 1.25")
        with col2:
            if st.button("🧪 Test Expression", key="test_expression"):
                try:
                    # Create sample context
                    sample_context = {
        'payslip': {
            'employee_id': 'EMP_001',
            'month': '2024-07',
            'base_salary': 4800.0,
            'overtime_rate': 35.0,
            'overtime_pay': 175.0,
            'total_pay': 4975.0
        },
        'attendance': {
            'employee_id': 'EMP_001',
            'month': '2024-07',
            'regular_hours': 160,
            'overtime_hours': 5,
            'total_hours': 165
        },
        'contract': {
            'employee_id': 'EMP_001',
            'hourly_rate': 30.0,
            'position': 'Software Developer'
        }
    }
                    
                    from simpleeval import simple_eval
                    allowed_functions = {"min": min, "max": max, "abs": abs, "round": round}
                    result = simple_eval(test_expr, names=sample_context, functions=allowed_functions)
                    st.success(f"✅ Result: {result}")
                    
                except Exception as e:
                    st.error(f"❌ Expression error: {e}")
    
    # Formula explanation section
    st.markdown("### 🧮 Common Formula Patterns")
    with st.expander("📚 Understanding Labor Law Calculations"):
        st.markdown("""
        #### Overtime Rate Calculations:
        
        **125% Overtime (First 2 hours):**
        ```python
        required_rate = contract.hourly_rate * 1.25
        underpaid_amount = (required_rate - payslip.overtime_rate) * min(attendance.overtime_hours, 2)
        ```
        
        **150% Overtime (Beyond 2 hours):**
        ```python
        required_rate = contract.hourly_rate * 1.5
        overtime_beyond_2h = max(attendance.overtime_hours - 2, 0)
        underpaid_amount = (required_rate - payslip.overtime_rate) * overtime_beyond_2h
        ```
        
        #### Minimum Wage Calculations:
        ```python
        minimum_wage = 32.7  # Current Israeli minimum wage
        if contract.hourly_rate < minimum_wage:
            underpaid_amount = (minimum_wage - contract.hourly_rate) * attendance.total_hours
        ```
        
        #### Penalty Calculations:
        ```python
        # Basic penalty (5% of underpaid amount)
        penalty_amount = total_underpaid_amount * 0.05
        
        # Progressive penalty based on severity
        if total_underpaid_amount > 1000:
            penalty_amount = total_underpaid_amount * 0.10
        ```
        
        #### Example Calculation:
        **Scenario:** Employee worked 5 overtime hours, paid ₪35/hour, contract rate ₪30/hour
        
        **Step 1:** First 2 hours at 125%
        - Required rate: ₪30 × 1.25 = ₪37.50/hour
        - Underpaid: (₪37.50 - ₪35.00) × 2 = ₪5.00
        
        **Step 2:** Remaining 3 hours at 150%
        - Required rate: ₪30 × 1.50 = ₪45.00/hour
        - Underpaid: (₪45.00 - ₪35.00) × 3 = ₪30.00
        
        **Total Underpaid:** ₪5.00 + ₪30.00 = ₪35.00
        **Penalty (5%):** ₪35.00 × 0.05 = ₪1.75
        **Total Impact:** ₪36.75
        """)
    
    # Sample context display
    st.markdown("### 📋 Sample Context Data")
    sample_context = {
        'payslip': {
            'employee_id': 'EMP_001',
            'month': '2024-07',
            'base_salary': 4800.0,
            'overtime_rate': 35.0,
            'overtime_pay': 175.0,
            'total_pay': 4975.0
        },
        'attendance': {
            'employee_id': 'EMP_001',
            'month': '2024-07',
            'regular_hours': 160,
            'overtime_hours': 5,
            'total_hours': 165
        },
        'contract': {
            'employee_id': 'EMP_001',
            'hourly_rate': 30.0,
            'position': 'Software Developer'
        }
    }
    st.json(sample_context)

# Hebrew Payslip Analysis Tab
with tab3:
  
    st.header("🎯 השוואת סוגי ניתוח ובדיקה")
    st.markdown("**בדוק סוגי ניתוח שונים עם אותם נתונים כדי להשוות תוצאות**")
    

    # Test data setup section with input method selection
    st.subheader("📋 הגדרת נתוני בדיקה")
    input_method_heb = st.radio("בחר שיטת קלט:", ["הזנה ידנית", "העלאת JSON", "השתמש בנתונים לדוגמה"], key="test_input_method_heb")

    test_payslip_data_heb = None
    test_attendance_data_heb = None
    test_contract_data_heb = None

    if input_method_heb == "הזנה ידנית":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**מידע על העובד**")
            test_employee_id_heb = st.text_input("מזהה עובד", value="TEST_001", key="test_emp_id_heb")
            test_month_heb = st.text_input("חודש (YYYY-MM)", value="2024-01", key="test_month_heb")
        with col2:
            st.markdown("**חלקי נתונים לכלול**")
            include_payslip_heb = st.checkbox("כלול נתוני תלוש", value=True, key="test_include_payslip_heb")
            include_contract_heb = st.checkbox("כלול נתוני חוזה", value=True, key="test_include_contract_heb")
            include_attendance_heb = st.checkbox("כלול נתוני נוכחות", value=True, key="test_include_attendance_heb")

        if include_payslip_heb:
            st.markdown("---")
            st.markdown("### 💰 פרטי תלוש")
            col1, col2 = st.columns(2)
            with col1:
                test_base_salary_heb = st.number_input("משכורת בסיס (₪)", min_value=0.0, value=5000.0, step=100.0, key="test_base_salary_heb")
                test_overtime_rate_heb = st.number_input("תעריף שעות נוספות ששולם (₪/שעה)", min_value=0.0, value=30.0, step=0.1, key="test_overtime_rate_heb")
                test_overtime_pay_heb = st.number_input("סה\"כ תשלום שעות נוספות (₪)", min_value=0.0, value=300.0, step=10.0, key="test_overtime_pay_heb")
            with col2:
                test_hourly_rate_heb = st.number_input("תעריף שעתי לפי חוזה (₪)", min_value=0.0, value=26.88, step=0.1, key="test_hourly_rate_heb")
                test_regular_hours_heb = st.number_input("שעות רגילות", min_value=0, value=186, step=1, key="test_regular_hours_heb")
                test_overtime_hours_heb = st.number_input("שעות נוספות", min_value=0, value=10, step=1, key="test_overtime_hours_heb")
            test_payslip_data_heb = {
                "employee_id": test_employee_id_heb,
                "month": test_month_heb,
                "base_salary": test_base_salary_heb,
                "overtime_hours": test_overtime_hours_heb,
                "overtime_pay": test_overtime_pay_heb,
                "overtime_rate": test_overtime_rate_heb,
                "total_salary": test_base_salary_heb + test_overtime_pay_heb,
                "hours_worked": test_regular_hours_heb + test_overtime_hours_heb,
                "hourly_rate": test_hourly_rate_heb
            }
        if include_attendance_heb:
            st.markdown("---")
            st.markdown("### ⏰ פרטי נוכחות")
            col1, col2 = st.columns(2)
            with col1:
                test_regular_hours_att_heb = st.number_input("שעות רגילות (נוכחות)", min_value=0, value=186, step=1, key="test_attendance_regular_hours_heb")
                test_overtime_hours_att_heb = st.number_input("שעות נוספות (נוכחות)", min_value=0, value=10, step=1, key="test_attendance_overtime_hours_heb")
            with col2:
                test_days_worked_heb = st.number_input("ימי עבודה", min_value=0, value=22, step=1, key="test_days_worked_heb")
                test_sick_days_heb = st.number_input("ימי מחלה", min_value=0, value=0, step=1, key="test_sick_days_heb")
                test_vacation_days_heb = st.number_input("ימי חופשה", min_value=0, value=0, step=1, key="test_vacation_days_heb")
            test_attendance_data_heb = {
                "employee_id": test_employee_id_heb,
                "month": test_month_heb,
                "days_worked": test_days_worked_heb,
                "regular_hours": test_regular_hours_att_heb,
                "overtime_hours": test_overtime_hours_att_heb,
                "total_hours": test_regular_hours_att_heb + test_overtime_hours_att_heb,
                "sick_days": test_sick_days_heb,
                "vacation_days": test_vacation_days_heb
            }
        if include_contract_heb:
            st.markdown("---")
            st.markdown("### 📋 פרטי חוזה")
            col1, col2 = st.columns(2)
            with col1:
                test_hourly_rate_con_heb = st.number_input("תעריף שעתי (חוזה)", min_value=0.0, value=26.88, step=0.1, key="test_contract_hourly_rate_heb")
                test_minimum_wage_monthly_heb = st.number_input("שכר מינימום חודשי", min_value=0.0, value=5300.0, step=10.0, key="test_minimum_wage_monthly_heb")
                test_minimum_wage_hourly_heb = st.number_input("שכר מינימום שעתי", min_value=0.0, value=29.12, step=0.1, key="test_minimum_wage_hourly_heb")
            with col2:
                test_overtime_rate_125_heb = st.number_input("תעריף שעות נוספות 125%", min_value=0.0, value=1.25, step=0.01, key="test_overtime_rate_125_heb")
                test_overtime_rate_150_heb = st.number_input("תעריף שעות נוספות 150%", min_value=0.0, value=1.50, step=0.01, key="test_overtime_rate_150_heb")
                test_standard_hours_per_month_heb = st.number_input("שעות סטנדרטיות לחודש", min_value=0, value=186, step=1, key="test_standard_hours_per_month_heb")
                test_standard_hours_per_day_heb = st.number_input("שעות סטנדרטיות ליום", min_value=0, value=8, step=1, key="test_standard_hours_per_day_heb")
                test_vacation_days_per_year_heb = st.number_input("ימי חופשה לשנה", min_value=0, value=14, step=1, key="test_vacation_days_per_year_heb")
                test_sick_days_per_year_heb = st.number_input("ימי מחלה לשנה", min_value=0, value=18, step=1, key="test_sick_days_per_year_heb")
            test_contract_data_heb = {
                "employee_id": test_employee_id_heb,
                "minimum_wage_monthly": test_minimum_wage_monthly_heb,
                "minimum_wage_hourly": test_minimum_wage_hourly_heb,
                "hourly_rate": test_hourly_rate_con_heb,
                "overtime_rate_125": test_overtime_rate_125_heb,
                "overtime_rate_150": test_overtime_rate_150_heb,
                "standard_hours_per_month": test_standard_hours_per_month_heb,
                "standard_hours_per_day": test_standard_hours_per_day_heb,
                "vacation_days_per_year": test_vacation_days_per_year_heb,
                "sick_days_per_year": test_sick_days_per_year_heb
            }

    elif input_method_heb == "העלאת JSON":
        uploaded_file_heb = st.file_uploader("העלה קובץ JSON של תלוש", type=['json'], key="test_uploaded_file_heb")
        if uploaded_file_heb:
            try:
                data = json.load(uploaded_file_heb)
                test_payslip_data_heb = data.get('payslip', [{}])[0]
                test_attendance_data_heb = data.get('attendance', [{}])[0]
                test_contract_data_heb = data.get('contract', [{}])[0]
                st.success("✅ הנתונים נטענו בהצלחה!")
            except Exception as e:
                st.error(f"שגיאה בטעינת הקובץ: {e}")

    elif input_method_heb == "השתמש בנתונים לדוגמה":
        sample_data = load_sample_data()
        if sample_data:
            test_payslip_data_heb = sample_data['payslip'][0]
            test_attendance_data_heb = sample_data['attendance'][0]
            test_contract_data_heb = sample_data['contract'][0]
            st.info("משתמש בנתונים לדוגמה לניתוח")
        else:
            st.error("אין נתונים לדוגמה זמינים")
    
    # Show test data preview
    with st.expander("📋 תצוגה מקדימה של נתוני בדיקה", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**נתוני תלוש:**")
            st.json(test_payslip_data_heb)
        with col2:
            st.markdown("**נתוני נוכחות:**")
            st.json(test_attendance_data_heb)
        with col3:
            st.markdown("**נתוני חוזה:**")
            st.json(test_contract_data_heb)

    # Analysis types to test
    analysis_types_heb = [
        ("violations_list", "📋 רשימת הפרות פשוטה"),
        ("easy", "😊 סיכום ידידותי למשתמש"),
        ("table", "📊 פורמט טבלה מאורגן"),
        ("violation_count_table", "📈 טבלת סטטיסטיקות"),
        ("combined", "⚖️ ניתוח משפטי מפורט"),
        ("report", "📄 דוח מעסיק")
    ]

    st.markdown("---")
    st.subheader("🚀 בדוק את כל סוגי הניתוח")
    
    # Test all analysis types
    if st.button("🚀 בדוק את כל סוגי הניתוח", type="primary", key="test_all_analysis_types_heb"):
        st.markdown("## 📊 השוואת תוצאות ניתוח")
        for analysis_type, description in analysis_types_heb:
            with st.expander(f"{description} ({analysis_type})", expanded=False):
                with st.spinner(f"מריץ ניתוח {analysis_type}..."):
                    try:
                        # Import DocumentProcessor
                        from document_processor_pydantic_ques import DocumentProcessor
                        processor = DocumentProcessor()
                        
                        # Convert single dicts to lists for create_report_with_rule_engine
                        payslip_list = [test_payslip_data_heb] if test_payslip_data_heb else []
                        attendance_list = [test_attendance_data_heb] if test_attendance_data_heb else []
                        
                        # Call create_report_with_rule_engine
                        result = asyncio.run(processor.create_report_with_rule_engine(
                            payslip_data=payslip_list,
                            attendance_data=attendance_list,
                            contract_data=test_contract_data_heb,
                            analysis_type=analysis_type
                        ))
                        
                        # Show summary metrics for the new format
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            if result.get('violations_count', 0) > 0:
                                st.metric("סטטוס", "הפרות")
                            elif result.get('inconclusive_count', 0) > 0:
                                st.metric("סטטוס", "לא חד משמעי")
                            else:
                                st.metric("סטטוס", "תואם")
                        with col2:
                            st.metric("הפרות", result.get('violations_count', 0))
                        with col3:
                            st.metric("לא חד משמעי", result.get('inconclusive_count', 0))
                        with col4:
                            if 'total_combined' in result:
                                combined_total = result['total_combined']
                            else:
                                total_underpaid = result.get('total_underpaid', result.get('total_underpaid_amount', 0.0))
                                total_penalties = result.get('total_penalties', result.get('penalty_amount', 0.0))
                                combined_total = total_underpaid + total_penalties
                            st.metric("סה\"כ (חסר תשלום + קנסות)", f"₪{combined_total:,.2f}")
                        
                        # Show the formatted output
                        st.markdown("### 📋 תוצאת ניתוח:")
                        if 'legal_analysis' in result:
                            if analysis_type in ["table", "violation_count_table"]:
                                st.code(result['legal_analysis'], language="")
                            else:
                                st.markdown(result['legal_analysis'])
                        else:
                            st.info("הניתוח הושלם אך אין תוצאה מעוצבת זמינה.")
                        
                        # Add download button for each analysis
                        if 'legal_analysis' in result:
                            st.download_button(
                                label=f"📥 הורד דוח {analysis_type}",
                                data=result['legal_analysis'],
                                file_name=f"analysis_{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                key=f"download_{analysis_type}_heb"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ שגיאה בהרצת ניתוח {analysis_type}: {str(e)}")

    # Individual analysis type testing
    st.markdown("---")
    st.subheader("🔍 בדוק סוג ניתוח בודד")

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_test_type_heb = st.selectbox(
            "בחר סוג ניתוח לבדיקה:",
            options=[at[0] for at in analysis_types_heb],
            format_func=lambda x: next(desc for code, desc in analysis_types_heb if code == x),
            key="individual_analysis_type_heb"
        )

    with col2:
        test_individual_button_heb = st.button("🧪 בדוק סוג נבחר", key="test_individual_type_heb")

    # Display results outside of columns for full width
    if test_individual_button_heb:
        with st.spinner(f"מריץ ניתוח {selected_test_type_heb}..."):
            try:
                # Import DocumentProcessor
                from document_processor_pydantic_ques import DocumentProcessor
                processor = DocumentProcessor()

                # Convert single dicts to lists for create_report_with_rule_engine
                payslip_list = [test_payslip_data_heb] if test_payslip_data_heb else []
                attendance_list = [test_attendance_data_heb] if test_attendance_data_heb else []

                # Call create_report_with_rule_engine
                result = asyncio.run(processor.create_report_with_rule_engine(
                    payslip_data=payslip_list,
                    attendance_data=attendance_list,
                    contract_data=test_contract_data_heb,
                    analysis_type=selected_test_type_heb
                ))

                st.success(f"✅ ניתוח {selected_test_type_heb} הושלם!")

                # Show detailed results using the new format
                st.markdown("### 📊 תוצאות מפורטות:")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if result.get('violations_count', 0) > 0:
                        st.metric("סטטוס", "נמצאו הפרות")
                    elif result.get('inconclusive_count', 0) > 0:
                        st.metric("סטטוס", "לא חד משמעי")
                    else:
                        st.metric("סטטוס", "תואם")
                with col2:
                    st.metric("הפרות שנמצאו", result.get('violations_count', 0))
                with col3:
                    st.metric("מקרים לא חד משמעיים", result.get('inconclusive_count', 0))
                with col4:
                    if 'total_combined' in result:
                        combined_total = result['total_combined']
                    else:
                        total_underpaid = result.get('total_underpaid', result.get('total_underpaid_amount', 0.0))
                        total_penalties = result.get('total_penalties', result.get('penalty_amount', 0.0))
                        combined_total = total_underpaid + total_penalties
                    st.metric("סה\"כ (חסר תשלום + קנסות)", f"₪{combined_total:,.2f}")

                # Analysis output
                st.markdown("### 📋 תוצאת ניתוח:")
                if 'legal_analysis' in result:
                    if selected_test_type_heb in ["table", "violation_count_table"]:
                        st.code(result['legal_analysis'], language="")
                    else:
                        st.markdown(result['legal_analysis'])
                else:
                    st.info("הניתוח הושלם אך אין תוצאה מעוצבת זמינה.")

                # Technical details
                with st.expander("🔧 פרטים טכניים", expanded=False):
                    st.markdown("**תוצאת ניתוח:**")
                    st.json(result)

            except Exception as e:
                st.error(f"❌ שגיאה: {str(e)}")
                st.code(str(e))

with tab4:
    st.header("⚖️ ניהול כללי חוק העבודה")
    
    # Load rules data fresh each time to ensure we have latest changes
    rules_data = load_rules_data()
    
    # Display current rules in a more organized way
    st.subheader("📋 כללים נוכחיים")
    
    if rules_data['rules']:
        # Create a summary table first
        rules_summary = []
        for rule in rules_data['rules']:
            rules_summary.append({
                'מזהה כלל': rule['rule_id'],
                'שם': rule['name'],
                'הפניה לחוק': rule['law_reference'],
                'תקף מתאריך': rule['effective_from'],
                'תקף עד': rule.get('effective_to', 'רציף'),
                'בדיקות': len(rule['checks'])
            })
        
        rules_df = pd.DataFrame(rules_summary)
        st.dataframe(rules_df, use_container_width=True)
        
        # Detailed view with better organization
        st.markdown("### 🔍 תצוגה מפורטת של כלל")
        if len(rules_data['rules']) > 0:
            selected_rule_heb = st.selectbox(
                "בחר כלל לצפייה בפרטים:",
                options=range(len(rules_data['rules'])),
                format_func=lambda x: f"{rules_data['rules'][x]['rule_id']} - {rules_data['rules'][x]['name']}",
                key="selected_rule_heb"
            )
        else:
            selected_rule_heb = None
            st.info("אין כללים זמינים לבחירה.")
        
        if selected_rule_heb is not None and selected_rule_heb < len(rules_data['rules']):
            rule = rules_data['rules'][selected_rule_heb]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**מזהה כלל:** {rule['rule_id']}")
                st.markdown(f"**שם:** {rule['name']}")
                st.markdown(f"**הפניה לחוק:** {rule['law_reference']}")
                st.markdown(f"**תיאור:** {rule['description']}")
                st.markdown(f"**תקופת תוקף:** {rule['effective_from']} עד {rule.get('effective_to', 'רציף')}")
                
                st.markdown("**בדיקות:**")
                for j, check in enumerate(rule['checks'], 1):
                    with st.expander(f"בדיקה {j}: {check.get('violation_message', 'אין הודעה')}"):
                        st.code(f"מזהה בדיקה: {check.get('id', 'לא זמין')}", language="python")
                        st.code(f"תנאי: {check['condition']}", language="python")
                        st.code(f"סכום חסר תשלום: {check['underpaid_amount']}", language="python")
                
                st.markdown("**חישוב קנס:**")
                for penalty_line in rule['penalty']:
                    st.code(penalty_line, language="python")
            
            with col2:
                st.markdown("**פעולות**")
                
                # Edit Rule
                if st.button("📝 ערוך כלל", key=f"edit_{selected_rule_heb}_heb"):
                    st.session_state[f'editing_rule_{selected_rule_heb}_heb'] = True
                
                # Delete Rule
                if st.button("🗑️ מחק כלל", key=f"delete_{selected_rule_heb}_heb", type="secondary"):
                    if st.session_state.get(f'confirm_delete_{selected_rule_heb}_heb', False):
                        # Actually delete the rule
                        rule_id_to_delete = rule['rule_id']
                        rules_data['rules'].pop(selected_rule_heb)
                        if save_rules_data(rules_data):
                            st.success(f"✅ כלל '{rule_id_to_delete}' נמחק בהצלחה!")
                            # Clear session state
                            st.session_state[f'confirm_delete_{selected_rule_heb}_heb'] = False
                            # Clear any editing states for all rules since indices may have changed
                            for key in list(st.session_state.keys()):
                                if key.startswith('editing_rule_') or key.startswith('testing_rule_') or key.startswith('confirm_delete_'):
                                    del st.session_state[key]
                            st.rerun()
                        else:
                            st.error("❌ נכשל במחיקת הכלל")
                    else:
                        st.session_state[f'confirm_delete_{selected_rule_heb}_heb'] = True
                        st.warning("⚠️ לחץ שוב לאישור מחיקה")
                
                # Test Rule
                if st.button("🧪 בדוק כלל", key=f"test_{selected_rule_heb}_heb"):
                    st.session_state[f'testing_rule_{selected_rule_heb}_heb'] = True
                
                # Cancel confirmations
                if st.session_state.get(f'confirm_delete_{selected_rule_heb}_heb', False):
                    if st.button("❌ בטל מחיקה", key=f"cancel_delete_{selected_rule_heb}_heb"):
                        st.session_state[f'confirm_delete_{selected_rule_heb}_heb'] = False
                        st.rerun()
        
        # Edit Rule Form
        if selected_rule_heb is not None and selected_rule_heb < len(rules_data['rules']) and st.session_state.get(f'editing_rule_{selected_rule_heb}_heb', False):
            try:
                # Get the current rule data
                current_rule = rules_data['rules'][selected_rule_heb]
                st.markdown("---")
                st.subheader(f"📝 עריכת כלל: {current_rule['rule_id']}")
                
                with st.form(f"edit_rule_form_{selected_rule_heb}_heb"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        edit_rule_id_heb = st.text_input("מזהה כלל", value=current_rule['rule_id'], key=f"edit_rule_id_{selected_rule_heb}_heb")
                        edit_name_heb = st.text_input("שם כלל", value=current_rule['name'], key=f"edit_name_{selected_rule_heb}_heb")
                        edit_law_reference_heb = st.text_input("הפניה לחוק", value=current_rule['law_reference'], key=f"edit_law_reference_{selected_rule_heb}_heb")
                        edit_description_heb = st.text_area("תיאור", value=current_rule['description'], height=80, key=f"edit_description_{selected_rule_heb}_heb")
                    
                    with col2:
                        edit_effective_from_heb = st.date_input("תקף מתאריך", 
                                                           value=datetime.strptime(current_rule['effective_from'], '%Y-%m-%d').date(), key=f"edit_effective_from_{selected_rule_heb}_heb")
                        edit_effective_to_heb = st.text_input("תקף עד", value=current_rule.get('effective_to', ''), key=f"edit_effective_to_{selected_rule_heb}_heb")
                        
                        st.markdown("**פונקציות זמינות:**")
                        st.code("min(), max(), abs(), round()")
                        
                        st.markdown("**משתנים זמינים:**")
                        st.code("""
payslip.*, attendance.*, contract.*
employee_id, month, hourly_rate, 
overtime_hours, total_hours, etc.
                        """)
                    
                    # Initialize session state for edit rule
                    if f'edit_rule_checks_{selected_rule_heb}_heb' not in st.session_state:
                        st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb'] = current_rule['checks'].copy()

                    if f'edit_rule_penalties_{selected_rule_heb}_heb' not in st.session_state:
                        st.session_state[f'edit_rule_penalties_{selected_rule_heb}_heb'] = current_rule['penalty'].copy()

                    # Check Management within edit form
                    st.markdown("**בדיקות כלל:**")

                    # Display current checks
                    if st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb']:
                        st.markdown("**בדיקות נוכחיות:**")
                        for i, check in enumerate(st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb']):
                            with st.expander(f"בדיקה {i+1}: {check.get('violation_message', 'אין הודעה')}"):
                                st.code(f"מזהה בדיקה: {check.get('id', 'לא זמין')}")
                                st.code(f"תנאי: {check['condition']}")
                                st.code(f"נוסחת סכום חסר תשלום: {check['underpaid_amount']}")
                                # Remove button for each check
                                if st.form_submit_button(f"🗑️ הסר בדיקה {i+1}"):
                                    st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb'].pop(i)

                    # Add new check inputs
                    st.markdown("**הוסף בדיקה חדשה:**")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        edit_new_check_id_heb = st.text_input("מזהה בדיקה", key=f"edit_new_check_id_{selected_rule_heb}_heb", help="מזהה ייחודי לבדיקה זו", placeholder="first_2h")
                        edit_new_condition_heb = st.text_input("תנאי", key=f"edit_new_condition_{selected_rule_heb}_heb", help="למשל: attendance.overtime_hours > 0", placeholder="attendance.overtime_hours > 0")
                        edit_new_underpaid_amount_heb = st.text_input("נוסחת סכום חסר תשלום", key=f"edit_new_underpaid_amount_{selected_rule_heb}_heb", help="למשל: (contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)", placeholder="(contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)")
                    with col2:
                        edit_new_violation_message_heb = st.text_input("הודעת הפרה", key=f"edit_new_violation_message_{selected_rule_heb}_heb", help="למשל: הפרת תעריף שעות נוספות", placeholder="הפרת תעריף שעות נוספות")

                    # Penalty Management within edit form
                    st.markdown("**חישוב קנס:**")

                    # Display current penalties
                    if st.session_state[f'edit_rule_penalties_{selected_rule_heb}_heb']:
                        st.markdown("**שורות קנס נוכחיות:**")
                        for i, penalty in enumerate(st.session_state[f'edit_rule_penalties_{selected_rule_heb}_heb']):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.code(f"{i+1}. {penalty}")
                            with col2:
                                if st.form_submit_button(f"🗑️ Remove Penalty {i+1}"):
                                    st.session_state[f'edit_rule_penalties_{selected_rule_heb}_heb'].pop(i)

                    # Add new penalty line input
                    st.markdown("**הוסף שורת קנס חדשה:**")
                    edit_new_penalty_line_heb = st.text_input("נוסחת קנס", key=f"edit_new_penalty_line_{selected_rule_heb}_heb", help="למשל: total_underpaid_amount = check_results[0]", placeholder="total_underpaid_amount = check_results[0] , penalty_amount= total_underpaid_amount * 5")

                    # Action buttons
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        edit_add_check_btn_heb = st.form_submit_button("➕ הוסף בדיקה")
                    with col2:
                        edit_add_penalty_btn_heb = st.form_submit_button("➕ הוסף קנס")
                    with col3:
                        edit_clear_all_btn_heb = st.form_submit_button("🗑️ נקה הכל")
                    with col4:
                        edit_save_changes_btn_heb = st.form_submit_button("💾 שמור שינויים", type="primary")

                    # Handle form submissions
                    if edit_add_check_btn_heb:
                        if edit_new_check_id_heb and edit_new_condition_heb and edit_new_underpaid_amount_heb and edit_new_violation_message_heb:
                            st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb'].append({
                                "id": edit_new_check_id_heb,
                                "condition": edit_new_condition_heb,
                                "underpaid_amount": edit_new_underpaid_amount_heb,
                                "violation_message": edit_new_violation_message_heb
                            })
                            st.success("✅ בדיקה נוספה בהצלחה!")
                        else:
                            st.error("אנא מלא את כל השדות לבדיקה")

                    if edit_add_penalty_btn_heb:
                        if edit_new_penalty_line_heb:
                            st.session_state[f'edit_rule_penalties_{selected_rule_heb}_heb'].append(edit_new_penalty_line_heb)
                            st.success("✅ שורת קנס נוספה בהצלחה!")
                        else:
                            st.error("אנא הכנס נוסחת קנס")

                    if edit_clear_all_btn_heb:
                        st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb'] = []
                        st.session_state[f'edit_rule_penalties_{selected_rule_heb}_heb'] = []
                        st.success("✅ כל הבדיקות והקנסות נוקו!")

                    if edit_save_changes_btn_heb:
                        if not all([edit_rule_id_heb, edit_name_heb, edit_law_reference_heb, edit_description_heb]):
                            st.error("❌ אנא מלא את כל השדות הנדרשים")
                        else:
                            try:
                                # Check if rule ID already exists (excluding current rule)
                                existing_ids = [r['rule_id'] for i, r in enumerate(rules_data['rules']) if i != selected_rule_heb]
                                if edit_rule_id_heb in existing_ids:
                                    st.error(f"❌ מזהה כלל '{edit_rule_id_heb}' כבר קיים. אנא בחר מזהה אחר.")
                                else:
                                    # Use session state lists
                                    edit_checks_json_heb = st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb']
                                    edit_penalty_json_heb = st.session_state[f'edit_rule_penalties_{selected_rule_heb}_heb']

                                    # Validate checks structure
                                    validation_passed = True
                                    if not edit_checks_json_heb:
                                        st.error("❌ אנא הוסף לפחות בדיקה אחת")
                                        validation_passed = False
                                    if not edit_penalty_json_heb:
                                        st.error("❌ אנא הוסף לפחות שורת קנס אחת")
                                        validation_passed = False

                                    for i, check in enumerate(edit_checks_json_heb):
                                        required_fields = ['id', 'condition', 'underpaid_amount', 'violation_message']
                                        missing_fields = [f for f in required_fields if f not in check]
                                        if missing_fields:
                                            st.error(f"❌ בדיקה {i+1} חסרים שדות נדרשים: {missing_fields}")
                                            validation_passed = False

                                    if not validation_passed:
                                        st.stop()

                                    # Update rule
                                    updated_rule = {
                                        "rule_id": edit_rule_id_heb,
                                        "name": edit_name_heb,
                                        "law_reference": edit_law_reference_heb,
                                        "description": edit_description_heb,
                                        "effective_from": edit_effective_from_heb.strftime('%Y-%m-%d'),
                                        "effective_to": edit_effective_to_heb if edit_effective_to_heb else None,
                                        "checks": edit_checks_json_heb,
                                        "penalty": edit_penalty_json_heb,
                                        "created_date": current_rule.get('created_date', datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')),
                                        "updated_date": datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                                    }

                                    rules_data['rules'][selected_rule_heb] = updated_rule
                                    if save_rules_data(rules_data):
                                        st.success(f"✅ כלל '{edit_rule_id_heb}' עודכן בהצלחה!")
                                        # Clear session state
                                        if f'edit_rule_checks_{selected_rule_heb}_heb' in st.session_state:
                                            del st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb']
                                        if f'edit_rule_penalties_{selected_rule_heb}_heb' in st.session_state:
                                            del st.session_state[f'edit_rule_penalties_{selected_rule_heb}_heb']
                                        st.session_state[f'editing_rule_{selected_rule_heb}_heb'] = False
                                        st.rerun()
                                    else:
                                        st.error("❌ נכשל בשמירת הכלל")

                            except Exception as e:
                                st.error(f"❌ שגיאה בעדכון כלל: {e}")

                    col1, col2 = st.columns(2)
                    with col1:
                        pass  # Save Changes button moved above
                    with col2:
                        if st.form_submit_button("❌ בטל עריכה"):
                            st.session_state[f'editing_rule_{selected_rule_heb}_heb'] = False
                            # Clear session state
                            if f'edit_rule_checks_{selected_rule_heb}_heb' in st.session_state:
                                del st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb']
                            if f'edit_rule_penalties_{selected_rule_heb}_heb' in st.session_state:
                                del st.session_state[f'edit_rule_penalties_{selected_rule_heb}_heb']
                            st.rerun()
            except (IndexError, KeyError) as e:
                st.error(f"❌ שגיאה בגישה לנתוני כלל: {e}")
                st.session_state[f'editing_rule_{selected_rule_heb}_heb'] = False
                st.rerun()
        
        # Test Rule Section
        if selected_rule_heb is not None and selected_rule_heb < len(rules_data['rules']) and st.session_state.get(f'testing_rule_{selected_rule_heb}_heb', False):
            try:
                # Get the current rule data
                current_rule = rules_data['rules'][selected_rule_heb]
                st.markdown("---")
                st.subheader(f"🧪 בדיקת כלל: {current_rule['rule_id']}")
                
                with st.form(f"test_rule_form_{selected_rule_heb}_heb"):
                    st.markdown("**הכנס נתוני בדיקה לאימות הכלל:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        test_employee_id_heb = st.text_input("מזהה עובד", value="TEST_001", key=f"test_employee_id_{selected_rule_heb}_heb")
                        test_month_heb = st.text_input("חודש (YYYY-MM)", value="2024-07", key=f"test_month_{selected_rule_heb}_heb")
                        test_hourly_rate_heb = st.number_input("תעריף שעתי", value=30.0, step=0.1, key=f"test_hourly_rate_{selected_rule_heb}_heb")
                        test_base_salary_heb = st.number_input("משכורת בסיס", value=4800.0, step=10.0, key=f"test_base_salary_{selected_rule_heb}_heb")
                    
                    with col2:
                        test_overtime_rate_heb = st.number_input("תעריף שעות נוספות ששולם", value=35.0, step=0.1, key=f"test_overtime_rate_{selected_rule_heb}_heb")
                        test_overtime_hours_heb = st.number_input("שעות נוספות", value=5, step=1, key=f"test_overtime_hours_{selected_rule_heb}_heb")
                        test_regular_hours_heb = st.number_input("שעות רגילות", value=160, step=1, key=f"test_regular_hours_{selected_rule_heb}_heb")
                        test_total_hours_heb = test_regular_hours_heb + test_overtime_hours_heb
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        run_test_heb = st.form_submit_button("🚀 הרץ בדיקה", type="primary")
                        if run_test_heb:
                            test_payslip = {
                                "employee_id": test_employee_id_heb,
                                "month": test_month_heb,
                                "base_salary": test_base_salary_heb,
                                "overtime_rate": test_overtime_rate_heb
                            }
                            test_attendance = {
                                "employee_id": test_employee_id_heb,
                                "month": test_month_heb,
                                "overtime_hours": test_overtime_hours_heb,
                                "total_hours": test_total_hours_heb
                            }
                            test_contract = {
                                "employee_id": test_employee_id_heb,
                                "hourly_rate": test_hourly_rate_heb
                            }
                            test_result = test_single_rule(current_rule, test_payslip, test_attendance, test_contract)
                            if not test_result["applicable"]:
                                st.warning(f"⚠️ {test_result['message']}")
                            elif "error" in test_result:
                                st.error(f"❌ הבדיקה נכשלה: {test_result['error']}")
                                if "context_used" in test_result:
                                    st.markdown("**נתונים שהיו בשימוש בזמן השגיאה:**")
                                    st.json(test_result["context_used"])
                            elif test_result["compliant"]:
                                st.success("✅ הבדיקה עברה! לא נמצאו הפרות!")
                                # Show detailed calculation even for passing tests
                                with st.expander("📊 הצג פרטי חישוב"):
                                    st.markdown("**בדיקות כלל שהוערכו:**")
                                    for j, check in enumerate(test_result.get('rule_checks', [])):
                                        st.markdown(f"**בדיקה {j+1}:** {check.get('violation_message', 'אין הודעה')}")
                                        st.code(f"מזהה בדיקה: {check.get('id', 'לא זמין')}")
                                        st.code(f"תנאי: {check['condition']}")
                                        st.code(f"נוסחת סכום חסר תשלום: {check['underpaid_amount']}")
                                        if j < len(test_result.get('check_results', [])):
                                            check_result = test_result['check_results'][j]
                                            st.info(f"תוצאה: תנאי = {check_result.get('condition_result', 'לא זמין')}, סכום = ₪{check_result['amount']:.2f}")
                                        # Show calculation steps
                                        if 'calculation_steps' in check_result:
                                            for step in check_result['calculation_steps']:
                                                if step['step'] == 'formula_substitution':
                                                    st.code(f"עם ערכים: {step['formula']} = {step['result']}")
                                st.markdown("**נתוני הקשר שהיו בשימוש:**")
                                st.json(test_result.get('context_used', {}))
                            else:
                                st.error(f"❌ הבדיקה מצאה הפרות:")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("סכום חסר תשלום", f"₪{test_result['total_underpaid_amount']:.2f}")
                                with col2:
                                    st.metric("סכום קנס", f"₪{test_result['penalty_amount']:.2f}")
                                st.markdown("### 🔍 ניתוח הפרה מפורט")
                                # Show each check calculation
                                for j, check in enumerate(test_result.get('rule_checks', [])):
                                    st.markdown(f"#### בדיקה {j+1}: {check.get('violation_message', 'אין הודעה')}")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**נוסחה:**")
                                        st.code(f"מזהה בדיקה: {check.get('id', 'לא זמין')}")
                                        st.code(f"תנאי: {check['condition']}")
                                        st.code(f"סכום חסר תשלום: {check['underpaid_amount']}")
                                    with col2:
                                        if j < len(test_result.get('check_results', [])):
                                            check_result = test_result['check_results'][j]
                                            st.markdown("**תוצאה:**")
                                            condition_met = check_result.get('condition_result', False)
                                            if condition_met:
                                                st.success("✅ תנאי: נכון")
                                            else:
                                                st.info("ℹ️ תנאי: לא נכון")
                                            amount = check_result['amount']
                                            if amount > 0:
                                                st.error(f"💰 חסר תשלום: ₪{amount:.2f}")
                                            else:
                                                st.success(f"💰 סכום: ₪{amount:.2f}")
                                    # Show calculation steps
                                    if j < len(test_result.get('check_results', [])):
                                        check_result = test_result['check_results'][j]
                                        if 'calculation_steps' in check_result:
                                            st.markdown("**שלבי חישוב:**")
                                            for step in check_result['calculation_steps']:
                                                if step['step'] == 'condition_evaluation':
                                                    st.info(f"🔍 {step['description']}")
                                                elif step['step'] == 'amount_calculation':
                                                    st.success(f"💰 {step['description']}")
                                                elif step['step'] == 'formula_substitution':
                                                    st.code(f"עם ערכים: {step['formula']} = {step['result']}")
                                        # Show any errors
                                        if check_result.get('evaluation_error'):
                                            st.error(f"⚠️ שגיאה: {check_result['evaluation_error']}")
                                        # Check for missing fields (from engine results)
                                        if 'missing_fields' in check_result and check_result['missing_fields']:
                                            st.warning("⚠️ **שדות חסרים בנתוני בדיקה:**")
                                            for field in check_result['missing_fields']:
                                                st.markdown(f"• `{field}` - לא נמצא בנתוני בדיקה")
                                    st.markdown("---")
                                # Show penalty calculation
                                st.markdown("### 💸 חישוב קנס")
                                for penalty_line in test_result.get('rule_penalty', []):
                                    st.code(penalty_line)
                                penalty_calc = test_result.get('penalty_calculation', {})
                                # Show penalty calculation steps
                                if 'calculation_steps' in penalty_calc:
                                    st.markdown("**שלבי חישוב קנס:**")
                                    for step in penalty_calc['calculation_steps']:
                                        st.markdown(f"**{step['variable']}:**")
                                        st.code(f"נוסחה: {step['formula']}")
                                        st.code(f"עם ערכים: {step['substituted_formula']}")
                                        st.code(f"תוצאה: {step['result']}")
                                # Show final amounts
                                for key, value in penalty_calc.items():
                                    if isinstance(value, (int, float)) and key in ['total_underpaid_amount', 'penalty_amount']:
                                        st.markdown(f"• **{key.replace('_', ' ').title()}:** ₪{value:.2f}")
                                # Show context data
                                st.markdown("### 📊 נתוני בדיקה שהיו בשימוש")
                                st.json(test_result.get('context_used', {}))
                                # Show violations summary
                                st.markdown("### ⚠️ הפרות שנמצאו")
                                for violation in test_result["violations"]:
                                    st.markdown(f"• **{violation['message']}:** ₪{violation['amount']:.2f}")
                
                with col2:
                    if st.form_submit_button("❌ סגור בדיקה"):
                        st.session_state[f'testing_rule_{selected_rule_heb}_heb'] = False
                        st.rerun()
            except (IndexError, KeyError) as e:
                st.error(f"❌ שגיאה בגישה לנתוני כלל: {e}")
                st.session_state[f'testing_rule_{selected_rule_heb}_heb'] = False
                st.rerun()
    else:
        st.info("לא נמצאו כללים. הוסף כלל חדש למטה.")
    
    # Add new rule section
    st.subheader("➕ הוסף כלל חדש")
    
    # Form for rule creation with integrated check/penalty management
    with st.form("new_rule_form_heb"):
        col1, col2 = st.columns(2)
        
        with col1:
            rule_id_heb = st.text_input("מזהה כלל*", help="מזהה ייחודי", key="rule_id_heb")
            name_heb = st.text_input("שם כלל*", help="שם קריא לבני אדם", key="name_heb")
            law_reference_heb = st.text_input("הפניה לחוק*", help="למשל: סעיף 16A", key="law_reference_heb")
            description_heb = st.text_area("תיאור*", help="מה הכלל בודק", height=80, key="description_heb")
        
        with col2:
            effective_from_heb = st.date_input("תקף מתאריך*", key="effective_from_heb")
            effective_to_heb = st.text_input("תקף עד", help="YYYY-MM-DD או השאר ריק", key="effective_to_heb")
            
            st.markdown("**פונקציות זמינות:**")
            st.code("min(), max(), abs(), round()")
            
            st.markdown("**משתנים זמינים:**")
            st.code("""
payslip.*, attendance.*, contract.*
employee_id, month, hourly_rate, 
overtime_hours, total_hours, etc.
            """)
    
        # Check Management within form
        st.markdown("**בדיקות כלל:**")
        
        # Display current checks
        if st.session_state.new_rule_checks_heb:
            st.markdown("**בדיקות נוכחיות:**")
            for i, check in enumerate(st.session_state.new_rule_checks_heb):
                with st.expander(f"בדיקה {i+1}: {check.get('violation_message', 'אין הודעה')}"):
                    st.code(f"מזהה בדיקה: {check.get('id', 'לא זמין')}")
                    st.code(f"תנאי: {check['condition']}")
                    st.code(f"סכום חסר תשלום: {check['underpaid_amount']}")
                    # Note: Remove functionality moved to form submit buttons
        
        # Add new check inputs
        st.markdown("**הוסף בדיקה חדשה:**")
        col1, col2 = st.columns([1, 1])
        with col1:
            new_check_id_heb = st.text_input("מזהה בדיקה", key="new_check_id_heb", help="מזהה ייחודי לבדיקה זו", placeholder="first_2h")
            new_condition_heb = st.text_input("תנאי", key="new_condition_heb", help="למשל: attendance.overtime_hours > 0", placeholder="attendance.overtime_hours > 0")
            new_underpaid_amount_heb = st.text_input("נוסחת סכום חסר תשלום", key="new_underpaid_amount_heb", help="למשל: (contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)", placeholder="(contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)")
        with col2:
            new_violation_message_heb = st.text_input("הודעת הפרה", key="new_violation_message_heb", help="למשל: הפרת תעריף שעות נוספות", placeholder="הפרת תעריף שעות נוספות")
        
        # Penalty Management within form
        st.markdown("**חישוב קנס:**")
        
        # Display current penalties
        if st.session_state.new_rule_penalties_heb:
            st.markdown("**שורות קנס נוכחיות:**")
            for i, penalty in enumerate(st.session_state.new_rule_penalties_heb):
                st.code(f"{i+1}. {penalty}")
        
        # Add new penalty line input
        st.markdown("**הוסף שורת קנס חדשה:**")
        new_penalty_line_heb = st.text_input("נוסחת קנס", key="new_penalty_line_heb", help="למשל: total_underpaid_amount = check_results[0]", placeholder="total_underpaid_amount = check_results[0] , penalty_amount= total_underpaid_amount * 5" )
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            add_check_btn_heb = st.form_submit_button("➕ הוסף בדיקה")
        with col2:
            add_penalty_btn_heb = st.form_submit_button("➕ הוסף קנס")
        with col3:
            clear_all_btn_heb = st.form_submit_button("🗑️ נקה הכל")
        with col4:
            submit_rule_btn_heb = st.form_submit_button("✅ צור כלל", type="primary")
        
        # Handle form submissions
        if add_check_btn_heb:
            if new_check_id_heb and new_condition_heb and new_underpaid_amount_heb and new_violation_message_heb:
                st.session_state.new_rule_checks_heb.append({
                    "id": new_check_id_heb,
                    "condition": new_condition_heb,
                    "underpaid_amount": new_underpaid_amount_heb,
                    "violation_message": new_violation_message_heb
                })
                st.success("✅ בדיקה נוספה בהצלחה!")
                st.rerun()
            else:
                st.error("אנא מלא את כל השדות לבדיקה")
        
        if add_penalty_btn_heb:
            if new_penalty_line_heb:
                st.session_state.new_rule_penalties_heb.append(new_penalty_line_heb)
                st.success("✅ שורת קנס נוספה בהצלחה!")
                st.rerun()
            else:
                st.error("אנא הכנס נוסחת קנס")
        
        if clear_all_btn_heb:
            st.session_state.new_rule_checks_heb = []
            st.session_state.new_rule_penalties_heb = []
            st.success("✅ כל הבדיקות והקנסות נוקו!")
            st.rerun()
        
        if submit_rule_btn_heb:
            if not all([rule_id_heb, name_heb, law_reference_heb, description_heb]):
                st.error("❌ אנא מלא את כל השדות הנדרשים")
            else:
                try:
                    # Check if rule ID already exists
                    existing_ids = [r['rule_id'] for r in rules_data['rules']]
                    if rule_id_heb in existing_ids:
                        st.error(f"❌ מזהה כלל '{rule_id_heb}' כבר קיים. אנא בחר מזהה אחר.")
                    else:
                        # Use session state lists
                        checks_json_heb = st.session_state.new_rule_checks_heb
                        penalty_json_heb = st.session_state.new_rule_penalties_heb
                        
                        # Validate checks structure
                        validation_passed = True
                        if not checks_json_heb:
                            st.error("❌ אנא הוסף לפחות בדיקה אחת")
                            validation_passed = False
                        if not penalty_json_heb:
                            st.error("❌ אנא הוסף לפחות שורת קנס אחת")
                            validation_passed = False
                        
                        for i, check in enumerate(checks_json_heb):
                            required_fields = ['id', 'condition', 'underpaid_amount', 'violation_message']
                            missing_fields = [f for f in required_fields if f not in check]
                            if missing_fields:
                                st.error(f"❌ בדיקה {i+1} חסרים שדות נדרשים: {missing_fields}")
                                validation_passed = False
                        
                        if not validation_passed:
                            st.stop()
                        
                        # Create new rule
                        new_rule = {
                            "rule_id": rule_id_heb,
                            "name": name_heb,
                            "law_reference": law_reference_heb,
                            "description": description_heb,
                            "effective_from": effective_from_heb.strftime('%Y-%m-%d'),
                            "effective_to": effective_to_heb if effective_to_heb else None,
                            "checks": checks_json_heb,
                            "penalty": penalty_json_heb,
                            "created_date": datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                            "updated_date": datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                        }
                        
                        # Add to rules data
                        rules_data['rules'].append(new_rule)
                        
                        # Save to file
                        if save_rules_data(rules_data):
                            st.success(f"✅ כלל '{rule_id_heb}' נוסף בהצלחה!")
                            st.balloons()
                            
                            # Clear the session state lists
                            st.session_state.new_rule_checks_heb = []
                            st.session_state.new_rule_penalties_heb = []
                            
                            # Show the new rule
                            with st.expander("📋 הצג כלל שנוסף", expanded=True):
                                st.json(new_rule)
                            
                            # Auto-refresh to show the new rule
                            st.info("💡 הדף יתרענן אוטומטית כדי להציג את הכלל החדש שלך ברשימה למעלה.")
                            st.rerun()
                        else:
                            st.error("❌ נכשל בשמירת הכלל לקובץ")
                        
                except Exception as e:
                    st.error(f"❌ שגיאה ביצירת כלל: {e}")
    
    # Rule validation helper
    st.markdown("---")
    st.subheader("🔍 עוזר אימות כללים")
    
    with st.expander("📚 מדריך כתיבת כללים"):
        st.markdown("""
        ### משתנים זמינים בתנאים ובחישובים:
        
        **נתוני תלוש:**
        - `payslip.base_salary` - משכורת בסיס חודשית
        - `payslip.overtime_rate` - תעריף שעות נוספות ששולם
        - `payslip.overtime_pay` - תשלום שעות נוספות כולל
        - `payslip.total_pay` - תשלום כולל
        
        **נתוני נוכחות:**
        - `attendance.regular_hours` - שעות עבודה רגילות
        - `attendance.overtime_hours` - שעות נוספות
        - `attendance.total_hours` - סה\"כ שעות
        
        **נתוני חוזה:**
        - `contract.hourly_rate` - תעריף שעתי לפי חוזה
        - `contract.position` - תפקיד העובד
        
        **גישה ישירה:**
        - ניתן לגשת לשדות ישירות: `overtime_hours`, `hourly_rate`, וכו'.
        
        ### פונקציות מתמטיות:
        - `min(a, b)` - מינימום בין שני ערכים
        - `max(a, b)` - מקסימום בין שני ערכים
        - `abs(x)` - ערך מוחלט
        - `round(x, n)` - עיגול ל-n מקומות עשרוניים
        - `sum(list)` - סכום של רשימת ערכים
        
        ### דוגמאות לתנאים:
        ```python
        # בדיקה אם קיימות שעות נוספות
        attendance.overtime_hours > 0
        
        # בדיקה אם תעריף שעתי נמוך משכר מינימום
        contract.hourly_rate < 32.7
        
        # תנאי מורכב עם מספר קריטריונים
        attendance.overtime_hours > 2 and payslip.overtime_rate < (contract.hourly_rate * 1.5)
        ```
        
        ### דוגמאות לחישובי סכום חסר תשלום:
        ```python
        # חסר תשלום פשוט בגין שעות נוספות
        (contract.hourly_rate * 1.25 - payslip.overtime_rate) * attendance.overtime_hours
        
        # חסר בגין שכר מינימום
        (32.7 - contract.hourly_rate) * attendance.total_hours
        
        # חישוב מדורג של שעות נוספות
        max(0, (contract.hourly_rate * 1.5 - payslip.overtime_rate) * max(attendance.overtime_hours - 2, 0))
        ```
        """)
    
    # Quick rule tester
    with st.expander("🧪 בודק ביטויים של כללים מהיר"):
        st.markdown("בדוק את הביטויים שלך לפני הוספה לכלל:")
        
        col1, col2 = st.columns(2)
        with col1:
            test_expr_heb = st.text_input("ביטוי לבדיקה:", 
                                    value="contract.hourly_rate * 1.25", key="test_expr_heb")
        with col2:
            if st.button("🧪 בדוק ביטוי", key="test_expression_heb"):
                try:
                    # Create sample context
                    sample_context = {
        'payslip': {
            'employee_id': 'EMP_001',
            'month': '2024-07',
            'base_salary': 4800.0,
            'overtime_rate': 35.0,
            'overtime_pay': 175.0,
            'total_pay': 4975.0
        },
        'attendance': {
            'employee_id': 'EMP_001',
            'month': '2024-07',
            'regular_hours': 160,
            'overtime_hours': 5,
            'total_hours': 165
        },
        'contract': {
            'employee_id': 'EMP_001',
            'hourly_rate': 30.0,
            'position': 'Software Developer'
        }
    }
                    
                    from simpleeval import simple_eval
                    allowed_functions = {"min": min, "max": max, "abs": abs, "round": round}
                    result = simple_eval(test_expr_heb, names=sample_context, functions=allowed_functions)
                    st.success(f"✅ תוצאה: {result}")
                    
                except Exception as e:
                    st.error(f"❌ שגיאת ביטוי: {e}")
    
    # Formula explanation section
    st.markdown("### 🧮 דפוסי נוסחאות נפוצים")
    with st.expander("📚 הבנת חישובי חוקי עבודה"):
        st.markdown("""
        #### חישובי תעריף שעות נוספות:
        
        **125% לשעתיים ראשונות:**
        ```python
        required_rate = contract.hourly_rate * 1.25
        underpaid_amount = (required_rate - payslip.overtime_rate) * min(attendance.overtime_hours, 2)
        ```
        
        **150% מעבר לשעתיים:**
        ```python
        required_rate = contract.hourly_rate * 1.5
        overtime_beyond_2h = max(attendance.overtime_hours - 2, 0)
        underpaid_amount = (required_rate - payslip.overtime_rate) * overtime_beyond_2h
        ```
        
        #### חישובי שכר מינימום:
        ```python
        minimum_wage = 32.7  # שכר מינימום ישראלי נוכחי
        if contract.hourly_rate < minimum_wage:
            underpaid_amount = (minimum_wage - contract.hourly_rate) * attendance.total_hours
        ```
        
        #### חישובי קנסות:
        ```python
        # קנס בסיסי (5% מסכום חסר התשלום)
        penalty_amount = total_underpaid_amount * 0.05
        
        # קנס פרוגרסיבי על פי חומרה
        if total_underpaid_amount > 1000:
            penalty_amount = total_underpaid_amount * 0.10
        ```
        
        #### דוגמת חישוב:
        **תרחיש:** עובד עבד 5 שעות נוספות, שולם ₪35/שעה, תעריף חוזה ₪30/שעה
        
        **שלב 1:** שעתיים ראשונות ב-125%
        - תעריף נדרש: ₪30 × 1.25 = ₪37.50/שעה
        - חסר תשלום: (₪37.50 - ₪35.00) × 2 = ₪5.00
        
        **שלב 2:** 3 שעות נותרות ב-150%
        - תעריף נדרש: ₪30 × 1.50 = ₪45.00/שעה
        - חסר תשלום: (₪45.00 - ₪35.00) × 3 = ₪30.00
        
        **סה\"כ חסר תשלום:** ₪5.00 + ₪30.00 = ₪35.00
        **קנס (5%):** ₪35.00 × 0.05 = ₪1.75
        **השפעה כוללת:** ₪36.75
        """)
    
    # Sample context display
    st.markdown("### 📋 נתוני הקשר לדוגמה")
    sample_context = {
        'payslip': {
            'employee_id': 'EMP_001',
            'month': '2024-07',
            'base_salary': 4800.0,
            'overtime_rate': 35.0,
            'overtime_pay': 175.0,
            'total_pay': 4975.0
        },
        'attendance': {
            'employee_id': 'EMP_001',
            'month': '2024-07',
            'regular_hours': 160,
            'overtime_hours': 5,
            'total_hours': 165
        },
        'contract': {
            'employee_id': 'EMP_001',
            'hourly_rate': 30.0,
            'position': 'Software Developer'
        }
    }
    st.json(sample_context)
