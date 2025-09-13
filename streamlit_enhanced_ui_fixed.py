import streamlit as st
import json
import os
import pandas as pd
import sys
import asyncio
import concurrent.futures
import uuid

from datetime import datetime
from engine.dynamic_params import DynamicParams

# Add engine directory to path
sys.path.append('engine')

from engine.loader import RuleLoader
from engine.evaluator import RuleEvaluator
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
st.markdown("**Advanced Rule Engine with Real-time Violation Detection & Amount Owed Calculation**")

# Load data functions
@st.cache_data
def load_rules_data():
    try:
        return RuleLoader.load_rules('rules/labor_law_rules.json')
    except Exception as e:
        st.error(f"Error loading rules: {e}")
        return {"rules": []}

def refresh_rules_data():
    st.cache_data.clear()
    return load_rules_data()

@st.cache_data
def load_sample_data():
    try:
        return RuleLoader.load_input('data/sample_input.json')
    except Exception as e:
        return None

def save_rules_data(rules_data):
    try:
        # Generate random IDs for checks that don't have them
        for rule in rules_data.get('rules', []):
            if 'checks' in rule:
                for check in rule['checks']:
                    if 'id' not in check or not check['id']:
                        check['id'] = f"check_{uuid.uuid4().hex[:8]}"
        
        with open('rules/labor_law_rules.json', 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Error saving rules: {e}")
        return False

# Dynamic parameter management
def get_dynamic_params():
    return DynamicParams.load()

def add_dynamic_param(section, param, label):
    DynamicParams.add_param(section, param, label)

def get_param_labels(section):
    return [p['label'] for p in DynamicParams.get_params(section)]

def get_param_names(section):
    return [p['param'] for p in DynamicParams.get_params(section)]

def get_all_param_labels():
    return DynamicParams.get_all_param_labels()




def test_single_rule(rule, test_payslip, test_attendance, test_contract):
    """Test a single rule against test data with detailed results"""
    context = build_context(test_payslip, test_attendance, test_contract)
    
    if not RuleEvaluator.is_rule_applicable(rule, test_payslip.get('month', datetime.now().strftime('%Y-%m'))):
        return {"applicable": False, "message": "Rule not applicable for this period"}
    
    try:
        check_results, named_results = RuleEvaluator.evaluate_checks(rule['checks'], context)

        # Include zero-amount violations as per rule engine semantics (amount >= 0)
        violations = [cr for cr in check_results if cr.get('amount', 0) >= 0]

        total_amount_owed = sum(cr.get('amount', 0) for cr in check_results if cr.get('amount', 0) >= 0)
        return {
            "applicable": True,
            "violations": violations,
            "check_results": check_results,
            "named_results": named_results,
            "rule_checks": rule['checks'],
            "context_used": context,
            "total_amount_owed": total_amount_owed,
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

# Initialize session state for Hebrew new rule builder
if 'new_rule_checks_heb' not in st.session_state:
    st.session_state.new_rule_checks_heb = []

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

    dynamic_params = get_dynamic_params()

    if input_method == "Manual Entry":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Employee Information**")
            # Show all payslip params
            payslip_inputs = {}
            for p in dynamic_params['payslip']:
                if p['param'] in ['employee_id', 'month']:
                    payslip_inputs[p['param']] = st.text_input(p['label'], value="" if p['param'] == 'month' else "TEST_001", key=f"payslip_{p['param']}")
        with col2:
            st.markdown("**Data Sections to Include**")
            include_payslip = st.checkbox("Include Payslip Data", value=True, key="test_include_payslip")
            include_contract = st.checkbox("Include Contract Data", value=True, key="test_include_contract")
            include_attendance = st.checkbox("Include Attendance Data", value=True, key="test_include_attendance")

        if include_payslip:
            st.markdown("---")
            st.markdown("### 💰 Payslip Details")
            payslip_inputs = {}
            payslip_fields = [p for p in dynamic_params['payslip'] if p['param'] not in ['employee_id', 'month']]
            for i in range(0, len(payslip_fields), 3):
                cols = st.columns(3)
                for j, p in enumerate(payslip_fields[i:i+3]):
                    label = p['label']
                    key = f"payslip_{p['param']}"
                    if '₪' in label or 'Rate' in label:
                        payslip_inputs[p['param']] = cols[j].number_input(label, min_value=0.0, value=0.0, step=0.1, key=key)
                    else:
                        payslip_inputs[p['param']] = cols[j].number_input(label, min_value=0, value=0, step=1, key=key)
            test_payslip_data = {**payslip_inputs}
            # Add employee_id and month
            for p in dynamic_params['payslip']:
                if p['param'] in ['employee_id', 'month']:
                    test_payslip_data[p['param']] = st.session_state.get(f"payslip_{p['param']}", "")
        if include_attendance:
            st.markdown("---")
            st.markdown("### ⏰ Attendance Details")
            attendance_inputs = {}
            attendance_fields = [p for p in dynamic_params['attendance'] if p['param'] not in ['employee_id', 'month']]
            for i in range(0, len(attendance_fields), 3):
                cols = st.columns(3)
                for j, p in enumerate(attendance_fields[i:i+3]):
                    label = p['label']
                    key = f"attendance_{p['param']}"
                    attendance_inputs[p['param']] = cols[j].number_input(label, min_value=0, value=0, step=1, key=key)
            test_attendance_data = {**attendance_inputs}
            for p in dynamic_params['attendance']:
                if p['param'] in ['employee_id', 'month']:
                    test_attendance_data[p['param']] = st.session_state.get(f"payslip_{p['param']}", "")
        if include_contract:
            st.markdown("---")
            st.markdown("### 📋 Contract Details")
            contract_inputs = {}
            contract_fields = [p for p in dynamic_params['contract'] if p['param'] != 'employee_id']
            for i in range(0, len(contract_fields), 3):
                cols = st.columns(3)
                for j, p in enumerate(contract_fields[i:i+3]):
                    label = p['label']
                    key = f"contract_{p['param']}"
                    if '₪' in label or 'Rate' in label or 'Contribution' in label:
                        contract_inputs[p['param']] = cols[j].number_input(label, min_value=0.0, value=0.0, step=0.1, key=key)
                    else:
                        contract_inputs[p['param']] = cols[j].number_input(label, min_value=0, value=0, step=1, key=key)
            test_contract_data = {**contract_inputs}
            for p in dynamic_params['contract']:
                if p['param'] == 'employee_id':
                    test_contract_data[p['param']] = st.session_state.get(f"payslip_employee_id", "")

    # Add new dynamic parameter section

    st.markdown("---")
    st.subheader("➕ Add/Remove Dynamic Parameter")
    with st.form("add_param_form"):
        param_section = st.selectbox("Section", ["payslip", "attendance", "contract"], key="add_param_section")
        param_name = st.text_input("Parameter Name (use snake_case)", key="add_param_name")
        param_label = st.text_input("Parameter Label (shown in UI)", key="add_param_label")
        submit_param_btn = st.form_submit_button("Add Parameter")
        if submit_param_btn:
            if param_name and param_label:
                add_dynamic_param(param_section, param_name, param_label)
                st.success(f"Added parameter '{param_name}' to {param_section}!")
                st.rerun()
            else:
                st.error("Parameter name and label are required.")

    st.markdown("---")
    st.subheader("🗑️ Remove Dynamic Parameter")
    with st.form("remove_param_form"):
        remove_section = st.selectbox("Section", ["payslip", "attendance", "contract"], key="remove_param_section")
        current_params = get_param_names(remove_section)
        remove_param = st.selectbox("Parameter to Remove", current_params, key="remove_param_name")
        submit_remove_btn = st.form_submit_button("Remove Parameter")
        if submit_remove_btn:
            if remove_param:
                DynamicParams.remove_param(remove_section, remove_param)
                st.success(f"Removed parameter '{remove_param}' from {remove_section}!")
                st.rerun()
            else:
                st.error("Select a parameter to remove.")


    if input_method == "Upload JSON":
        uploaded_file = st.file_uploader("Upload payslip JSON file", type=['json'], key="test_uploaded_file")
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                # Use dynamic param keys to extract data
                test_payslip_data = {p['param']: data.get('payslip', [{}])[0].get(p['param'], None) for p in dynamic_params['payslip']}
                test_attendance_data = {p['param']: data.get('attendance', [{}])[0].get(p['param'], None) for p in dynamic_params['attendance']}
                test_contract_data = {p['param']: data.get('contract', [{}])[0].get(p['param'], None) for p in dynamic_params['contract']}
                st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    if input_method == "Use Sample Data":
        sample_data = load_sample_data()
        if sample_data:
            test_payslip_data = {p['param']: sample_data['payslip'][0].get(p['param'], None) for p in dynamic_params['payslip']}
            test_attendance_data = {p['param']: sample_data['attendance'][0].get(p['param'], None) for p in dynamic_params['attendance']}
            test_contract_data = {p['param']: sample_data['contract'][0].get(p['param'], None) for p in dynamic_params['contract']}
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
                        # Returns dict with keys: legal_analysis, status, analysis_type, violations_count, inconclusive_count, compliant_count, total_amount_owed, violations_by_law, inconclusive_results, all_results (if no data)
                        result = asyncio.run(processor.create_report_with_rule_engine(
                            payslip_data=payslip_list,
                            attendance_data=attendance_list,
                            contract_data=test_contract_data,
                            analysis_type=analysis_type
                        ))
                        # Robust key access
                        violations_count = result.get('violations_count', 0)
                        inconclusive_count = result.get('inconclusive_count', 0)
                        compliant_count = result.get('compliant_count', 0)
                        total_amount_owed = result.get('total_amount_owed', 0.0)
                        legal_analysis = result.get('legal_analysis', '')
                        
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
                            total_amount_owed = result.get('total_amount_owed', 0.0)
                            st.metric("Total Amount Owed", f"₪{total_amount_owed:,.2f}")
                        
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
                # Returns dict with keys: legal_analysis, status, analysis_type, violations_count, inconclusive_count, compliant_count, total_amount_owed, violations_by_law, inconclusive_results, all_results (if no data)
                result = asyncio.run(processor.create_report_with_rule_engine(
                    payslip_data=payslip_list,
                    attendance_data=attendance_list,
                    contract_data=test_contract_data,
                    analysis_type=selected_test_type
                ))
                # Robust key access
                violations_count = result.get('violations_count', 0)
                inconclusive_count = result.get('inconclusive_count', 0)
                compliant_count = result.get('compliant_count', 0)
                total_amount_owed = result.get('total_amount_owed', 0.0)
                legal_analysis = result.get('legal_analysis', '')

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
                    total_amount_owed = result.get('total_amount_owed', 0.0)
                    st.metric("Total Amount Owed", f"₪{total_amount_owed:,.2f}")

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
                        st.code(f"Amount Owed: {check.get('amount_owed', 'N/A')}", language="python")
                
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

                    # Check Management within edit form
                    st.markdown("**Rule Checks:**")

                    # Display current checks
                    if st.session_state[f'edit_rule_checks_{selected_rule}']:
                        st.markdown("**Current Checks:**")
                        for i, check in enumerate(st.session_state[f'edit_rule_checks_{selected_rule}']):
                            with st.expander(f"Check {i+1}: {check.get('violation_message', 'No message')}"):
                                st.code(f"Check ID: {check.get('id', 'N/A')}")
                                st.code(f"Condition: {check['condition']}")
                                st.code(f"Amount Owed: {check.get('amount_owed', 'N/A')}")
                                # Remove button for each check
                                if st.form_submit_button(f"🗑️ Remove Check {i+1}"):
                                    st.session_state[f'edit_rule_checks_{selected_rule}'].pop(i)

                    # Add new check inputs
                    st.markdown("**Add New Check:**")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        edit_new_condition = st.text_input("Condition", key=f"edit_new_condition_{selected_rule}", help="e.g., attendance.overtime_hours > 0", placeholder="attendance.overtime_hours > 0")
                        edit_new_amount_owed = st.text_input("Amount Owed Formula", key=f"edit_new_amount_owed_{selected_rule}", help="e.g., (contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)", placeholder="(contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)")
                    with col2:
                        edit_new_violation_message = st.text_input("Violation Message", key=f"edit_new_violation_message_{selected_rule}", help="e.g., Overtime rate violation", placeholder="Overtime rate violation")

                    # Action buttons
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        edit_add_check_btn = st.form_submit_button("➕ Add Check")
                    with col2:
                        edit_clear_all_btn = st.form_submit_button("🗑️ Clear All")
                    with col3:
                        edit_save_changes_btn = st.form_submit_button("💾 Save Changes", type="primary")

                    # Handle form submissions
                    if edit_add_check_btn:
                        if edit_new_condition and edit_new_amount_owed and edit_new_violation_message:
                            st.session_state[f'edit_rule_checks_{selected_rule}'].append({
                                "id": "",  # Will be generated by backend
                                "condition": edit_new_condition,
                                "amount_owed": edit_new_amount_owed,
                                "violation_message": edit_new_violation_message
                            })
                            st.success("✅ Check added successfully!")
                        else:
                            st.error("Please fill all fields for the check")

                    if edit_clear_all_btn:
                        st.session_state[f'edit_rule_checks_{selected_rule}'] = []
                        st.success("✅ All checks cleared!")

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

                                    # Validate checks structure
                                    validation_passed = True
                                    if not edit_checks_json:
                                        st.error("❌ Please add at least one check")
                                        validation_passed = False

                                    for i, check in enumerate(edit_checks_json):
                                        required_fields = ['condition', 'amount_owed', 'violation_message']  # id will be generated by backend
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
                                        "created_date": current_rule.get('created_date', datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')),
                                        "updated_date": datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                                    }

                                    rules_data['rules'][selected_rule] = updated_rule
                                    if save_rules_data(rules_data):
                                        st.success(f"✅ Rule '{edit_rule_id}' updated successfully!")
                                        # Clear session state
                                        if f'edit_rule_checks_{selected_rule}' in st.session_state:
                                            del st.session_state[f'edit_rule_checks_{selected_rule}']
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
                                        st.code(f"Amount Owed: {check.get('amount_owed', 'N/A')}")
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
                                    st.metric("Amount Owed", f"₪{test_result['total_amount_owed']:.2f}")
                                st.markdown("### 🔍 Detailed Violation Analysis")
                                # Show each check calculation
                                for j, check in enumerate(test_result.get('rule_checks', [])):
                                    st.markdown(f"#### Check {j+1}: {check.get('violation_message', 'No message')}")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Formula:**")
                                        st.code(f"Check ID: {check.get('id', 'N/A')}")
                                        st.code(f"Condition: {check['condition']}")
                                        st.code(f"Amount Owed: {check.get('amount_owed', 'N/A')}")
                                    with col2:
                                        if j < len(test_result.get('check_results', [])):
                                            check_result = test_result['check_results'][j]
                                            st.markdown("**Result:**")
                                            condition_met = check_result.get('condition_result', False)
                                            if condition_met:
                                                st.success("✅ Condition: TRUE")
                                            else:
                                                st.info("ℹ️ Condition: FALSE")
                                            amount = check_result.get('amount', 0)
                                            # Treat zero as a reported owed amount (matching engine semantics)
                                            if amount >= 0:
                                                st.error(f"💰 Amount Owed: ₪{amount:.2f}")
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
                                # Penalty calculation UI removed
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
                    st.code(f"Amount Owed: {check.get('amount_owed', 'N/A')}")
                    # Note: Remove functionality moved to form submit buttons
        
        # Add new check inputs
        st.markdown("**Add New Check:**")
        col1, col2 = st.columns([1, 1])
        with col1:
            new_condition = st.text_input("Condition", key="new_condition", help="e.g., attendance.overtime_hours > 0", placeholder="attendance.overtime_hours > 0")
            new_amount_owed = st.text_input("Amount Owed Formula", key="new_amount_owed", help="e.g., (contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)", placeholder="(contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)")
        with col2:
            new_violation_message = st.text_input("Violation Message", key="new_violation_message", help="e.g., Overtime rate violation", placeholder="Overtime rate violation")
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            add_check_btn = st.form_submit_button("➕ Add Check")
        with col2:
            clear_all_btn = st.form_submit_button("🗑️ Clear All")
        with col3:
            submit_rule_btn = st.form_submit_button("✅ Create Rule", type="primary")
        
        # Handle form submissions
        if add_check_btn:
            if new_condition and new_amount_owed and new_violation_message:
                st.session_state.new_rule_checks.append({
                    "id": "",  # Will be generated by backend
                    "condition": new_condition,
                    "amount_owed": new_amount_owed,
                    "violation_message": new_violation_message
                })
                st.success("✅ Check added successfully!")
                st.rerun()
            else:
                st.error("Please fill all fields for the check")
        
        if clear_all_btn:
            st.session_state.new_rule_checks = []
            st.success("✅ All checks cleared!")
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
                        
                        # Validate checks structure
                        validation_passed = True
                        if not checks_json:
                            st.error("❌ Please add at least one check")
                            validation_passed = False
                        
                        for i, check in enumerate(checks_json):
                            required_fields = ['condition', 'amount_owed', 'violation_message']  # id will be generated by backend
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
        """)
        dynamic_params = get_dynamic_params()
        for section in ['payslip', 'attendance', 'contract']:
            st.markdown(f"**{section.capitalize()} Data:**")
            for p in dynamic_params[section]:
                st.markdown(f"- `{section}.{p['param']}` - {p['label']}")
            st.markdown("")
        st.markdown("**Flattened Access:** You can also access fields directly, e.g. `overtime_hours`, `hourly_rate`, etc.")
        st.markdown("""
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
        
        ### Example Amount Owed Calculations:
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
        amount_owed = (required_rate - payslip.overtime_rate) * min(attendance.overtime_hours, 2)
        ```
        
        **150% Overtime (Beyond 2 hours):**
        ```python
        required_rate = contract.hourly_rate * 1.5
        overtime_beyond_2h = max(attendance.overtime_hours - 2, 0)
        amount_owed = (required_rate - payslip.overtime_rate) * overtime_beyond_2h
        ```
        
        #### Minimum Wage Calculations:
        ```python
        minimum_wage = 32.7  # Current Israeli minimum wage
        if contract.hourly_rate < minimum_wage:
            amount_owed = (minimum_wage - contract.hourly_rate) * attendance.total_hours
        ```
        
        #### Example Calculation:
        **Scenario:** Employee worked 5 overtime hours, paid ₪35/hour, contract rate ₪30/hour
        
        **Step 1:** First 2 hours at 125%
        - Required rate: ₪30 × 1.25 = ₪37.50/hour
        - Amount Owed: (₪37.50 - ₪35.00) × 2 = ₪5.00
        
        **Step 2:** Remaining 3 hours at 150%
        - Required rate: ₪30 × 1.50 = ₪45.00/hour
        - Amount Owed: (₪45.00 - ₪35.00) × 3 = ₪30.00
        
        **Total Amount Owed:** ₪5.00 + ₪30.00 = ₪35.00
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
        # --- Hebrew version of tab1 logic ---
        st.header("🎯 ניתוח סוגי בדיקות והשוואה")
        st.markdown("**בדוק סוגי ניתוח שונים עם אותם נתונים להשוואה**")

        st.subheader("📋 הגדרת נתוני בדיקה")
        input_method = st.radio("בחר שיטת הזנה:", ["הזנה ידנית", "העלאת JSON", "השתמש בנתוני דוגמה"], key="test_input_method_heb")

        test_payslip_data = None
        test_attendance_data = None
        test_contract_data = None

        dynamic_params = get_dynamic_params()

        if input_method == "הזנה ידנית":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**פרטי עובד**")
                payslip_inputs = {}
                for p in dynamic_params['payslip']:
                    if p['param'] in ['employee_id', 'month']:
                        payslip_inputs[p['param']] = st.text_input(p['label'], value="" if p['param'] == 'month' else "TEST_001", key=f"payslip_{p['param']}_heb")
            with col2:
                st.markdown("**סעיפי נתונים לכלול**")
                include_payslip = st.checkbox("כלול נתוני תלוש שכר", value=True, key="test_include_payslip_heb")
                include_contract = st.checkbox("כלול נתוני חוזה", value=True, key="test_include_contract_heb")
                include_attendance = st.checkbox("כלול נתוני נוכחות", value=True, key="test_include_attendance_heb")

            if include_payslip:
                st.markdown("---")
                st.markdown("### 💰 פרטי תלוש שכר")
                payslip_inputs = {}
                payslip_fields = [p for p in dynamic_params['payslip'] if p['param'] not in ['employee_id', 'month']]
                for i in range(0, len(payslip_fields), 3):
                    cols = st.columns(3)
                    for j, p in enumerate(payslip_fields[i:i+3]):
                        label = p['label']
                        key = f"payslip_{p['param']}_heb"
                        if '₪' in label or 'Rate' in label:
                            payslip_inputs[p['param']] = cols[j].number_input(label, min_value=0.0, value=0.0, step=0.1, key=key)
                        else:
                            payslip_inputs[p['param']] = cols[j].number_input(label, min_value=0, value=0, step=1, key=key)
                test_payslip_data = {**payslip_inputs}
                for p in dynamic_params['payslip']:
                    if p['param'] in ['employee_id', 'month']:
                        test_payslip_data[p['param']] = st.session_state.get(f"payslip_{p['param']}_heb", "")
            if include_attendance:
                st.markdown("---")
                st.markdown("### ⏰ פרטי נוכחות")
                attendance_inputs = {}
                attendance_fields = [p for p in dynamic_params['attendance'] if p['param'] not in ['employee_id', 'month']]
                for i in range(0, len(attendance_fields), 3):
                    cols = st.columns(3)
                    for j, p in enumerate(attendance_fields[i:i+3]):
                        label = p['label']
                        key = f"attendance_{p['param']}_heb"
                        attendance_inputs[p['param']] = cols[j].number_input(label, min_value=0, value=0, step=1, key=key)
                test_attendance_data = {**attendance_inputs}
                for p in dynamic_params['attendance']:
                    if p['param'] in ['employee_id', 'month']:
                        test_attendance_data[p['param']] = st.session_state.get(f"payslip_{p['param']}_heb", "")
            if include_contract:
                st.markdown("---")
                st.markdown("### 📋 פרטי חוזה")
                contract_inputs = {}
                contract_fields = [p for p in dynamic_params['contract'] if p['param'] != 'employee_id']
                for i in range(0, len(contract_fields), 3):
                    cols = st.columns(3)
                    for j, p in enumerate(contract_fields[i:i+3]):
                        label = p['label']
                        key = f"contract_{p['param']}_heb"
                        if '₪' in label or 'Rate' in label or 'Contribution' in label:
                            contract_inputs[p['param']] = cols[j].number_input(label, min_value=0.0, value=0.0, step=0.1, key=key)
                        else:
                            contract_inputs[p['param']] = cols[j].number_input(label, min_value=0, value=0, step=1, key=key)
                test_contract_data = {**contract_inputs}
                for p in dynamic_params['contract']:
                    if p['param'] == 'employee_id':
                        test_contract_data[p['param']] = st.session_state.get(f"payslip_employee_id_heb", "")

        st.markdown("---")
        st.subheader("➕ הוסף/הסר פרמטר דינמי")
        with st.form("add_param_form_heb"):
            param_section = st.selectbox("סעיף", ["payslip", "attendance", "contract"], key="add_param_section_heb")
            param_name = st.text_input("שם פרמטר (snake_case)", key="add_param_name_heb")
            param_label = st.text_input("תווית פרמטר (מוצג בממשק)", key="add_param_label_heb")
            submit_param_btn = st.form_submit_button("הוסף פרמטר")
            if submit_param_btn:
                if param_name and param_label:
                    add_dynamic_param(param_section, param_name, param_label)
                    st.success(f"הפרמטר '{param_name}' נוסף ל-{param_section}!")
                    st.rerun()
                else:
                    st.error("שם ותווית הפרמטר נדרשים.")

        st.markdown("---")
        st.subheader("🗑️ הסר פרמטר דינמי")
        with st.form("remove_param_form_heb"):
            remove_section = st.selectbox("סעיף", ["payslip", "attendance", "contract"], key="remove_param_section_heb")
            current_params = get_param_names(remove_section)
            remove_param = st.selectbox("פרמטר להסרה", current_params, key="remove_param_name_heb")
            submit_remove_btn = st.form_submit_button("הסר פרמטר")
            if submit_remove_btn:
                if remove_param:
                    DynamicParams.remove_param(remove_section, remove_param)
                    st.success(f"הפרמטר '{remove_param}' הוסר מ-{remove_section}!")
                    st.rerun()
                else:
                    st.error("בחר פרמטר להסרה.")

        if input_method == "העלאת JSON":
            uploaded_file = st.file_uploader("העלה קובץ JSON של תלוש שכר", type=['json'], key="test_uploaded_file_heb")
            if uploaded_file:
                try:
                    data = json.load(uploaded_file)
                    test_payslip_data = {p['param']: data.get('payslip', [{}])[0].get(p['param'], None) for p in dynamic_params['payslip']}
                    test_attendance_data = {p['param']: data.get('attendance', [{}])[0].get(p['param'], None) for p in dynamic_params['attendance']}
                    test_contract_data = {p['param']: data.get('contract', [{}])[0].get(p['param'], None) for p in dynamic_params['contract']}
                    st.success("✅ הנתונים נטענו בהצלחה!")
                except Exception as e:
                    st.error(f"שגיאה בטעינת קובץ: {e}")

        if input_method == "השתמש בנתוני דוגמה":
            sample_data = load_sample_data()
            if sample_data:
                test_payslip_data = {p['param']: sample_data['payslip'][0].get(p['param'], None) for p in dynamic_params['payslip']}
                test_attendance_data = {p['param']: sample_data['attendance'][0].get(p['param'], None) for p in dynamic_params['attendance']}
                test_contract_data = {p['param']: sample_data['contract'][0].get(p['param'], None) for p in dynamic_params['contract']}
                st.info("משתמש בנתוני דוגמה לניתוח")
            else:
                st.error("אין נתוני דוגמה זמינים")

        # Show test data preview
        with st.expander("📋 תצוגה מקדימה של נתוני בדיקה", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**נתוני תלוש שכר:**")
                st.json(test_payslip_data)
            with col2:
                st.markdown("**נתוני נוכחות:**")
                st.json(test_attendance_data)
            with col3:
                st.markdown("**נתוני חוזה:**")
                st.json(test_contract_data)

        # Analysis types to test
        analysis_types = [
            ("violations_list", "📋 רשימת הפרות פשוטה"),
            ("easy", "😊 סיכום ידידותי למשתמש"),
            ("table", "📊 טבלה מאורגנת"),
            ("violation_count_table", "📈 טבלת סטטיסטיקות"),
            ("combined", "⚖️ ניתוח משפטי מפורט"),
            ("report", "📄 דוח למעסיק")
        ]

        st.markdown("---")
        st.subheader("🚀 בדוק את כל סוגי הניתוחים")
        if st.button("🚀 בדוק את כל סוגי הניתוחים", type="primary", key="test_all_analysis_types_combined_heb"):
            st.markdown("## 📊 השוואת תוצאות ניתוח")
            for analysis_type, description in analysis_types:
                with st.expander(f"{description} ({analysis_type})", expanded=False):
                    with st.spinner(f"מריץ ניתוח {analysis_type}..."):
                        try:
                            # Import DocumentProcessor
                            from document_processor_pydantic_ques import DocumentProcessor
                            processor = DocumentProcessor()
                            payslip_list = [test_payslip_data] if test_payslip_data else []
                            attendance_list = [test_attendance_data] if test_attendance_data else []
                            # Returns dict with keys: legal_analysis, status, analysis_type, violations_count, inconclusive_count, compliant_count, total_amount_owed, violations_by_law, inconclusive_results, all_results (if no data)
                            result = asyncio.run(processor.create_report_with_rule_engine(
                                payslip_data=payslip_list,
                                attendance_data=attendance_list,
                                contract_data=test_contract_data,
                                analysis_type=analysis_type
                            ))
                            # Robust key access
                            violations_count = result.get('violations_count', 0)
                            inconclusive_count = result.get('inconclusive_count', 0)
                            compliant_count = result.get('compliant_count', 0)
                            total_amount_owed = result.get('total_amount_owed', 0.0)
                            legal_analysis = result.get('legal_analysis', '')
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                if result.get('violations_count', 0) > 0:
                                    st.metric("סטטוס", "הפרות")
                                elif result.get('inconclusive_count', 0) > 0:
                                    st.metric("סטטוס", "לא חד משמעי")
                                else:
                                    st.metric("סטטוס", "תקין")
                            with col2:
                                st.metric("הפרות", result.get('violations_count', 0))
                            with col3:
                                st.metric("לא חד משמעי", result.get('inconclusive_count', 0))
                            with col4:
                                total_amount_owed = result.get('total_amount_owed', 0.0)
                                st.metric('סה"כ חסר', f"₪{total_amount_owed:,.2f}")
                            st.markdown("### 📋 פלט ניתוח:")
                            if 'legal_analysis' in result:
                                if analysis_type in ["table", "violation_count_table"]:
                                    st.code(result['legal_analysis'], language="")
                                else:
                                    st.markdown(result['legal_analysis'])
                            else:
                                st.info("הניתוח הסתיים אך אין פלט מעוצב.")
                            if 'legal_analysis' in result:
                                st.download_button(
                                    label=f"📥 הורד דוח {analysis_type}",
                                    data=result['legal_analysis'],
                                    file_name=f"analysis_{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    key=f"download_{analysis_type}_combined_heb"
                                )
                        except Exception as e:
                            st.error(f"❌ שגיאה בניתוח {analysis_type}: {str(e)}")

        st.markdown("---")
        st.subheader("🔍 בדוק סוג ניתוח בודד")
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_test_type = st.selectbox(
                "בחר סוג ניתוח לבדיקה:",
                options=[at[0] for at in analysis_types],
                format_func=lambda x: next(desc for code, desc in analysis_types if code == x),
                key="individual_analysis_type_combined_heb"
            )
        with col2:
            test_individual_button = st.button("🧪 בדוק סוג נבחר", key="test_individual_type_combined_heb")

        # Display results outside of columns for full width
        if test_individual_button:
            with st.spinner(f"מריץ ניתוח {selected_test_type}..."):
                try:
                    # Import DocumentProcessor
                    from document_processor_pydantic_ques import DocumentProcessor
                    processor = DocumentProcessor()
                    payslip_list = [test_payslip_data] if test_payslip_data else []
                    attendance_list = [test_attendance_data] if test_attendance_data else []
                    # Returns dict with keys: legal_analysis, status, analysis_type, violations_count, inconclusive_count, compliant_count, total_amount_owed, violations_by_law, inconclusive_results, all_results (if no data)
                    result = asyncio.run(processor.create_report_with_rule_engine(
                        payslip_data=payslip_list,
                        attendance_data=attendance_list,
                        contract_data=test_contract_data,
                        analysis_type=selected_test_type
                    ))
                    # Robust key access
                    violations_count = result.get('violations_count', 0)
                    inconclusive_count = result.get('inconclusive_count', 0)
                    compliant_count = result.get('compliant_count', 0)
                    total_amount_owed = result.get('total_amount_owed', 0.0)
                    legal_analysis = result.get('legal_analysis', '')

                    st.success(f"✅ ניתוח {selected_test_type} הסתיים!")

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
                            st.metric("סטטוס", "תקין")
                    with col2:
                        st.metric("נמצאו הפרות", result.get('violations_count', 0))
                    with col3:
                        st.metric("מקרים לא חד משמעיים", result.get('inconclusive_count', 0))
                    with col4:
                        total_amount_owed = result.get('total_amount_owed', 0.0)
                        st.metric('סה"כ חסר', f"₪{total_amount_owed:,.2f}")

                    # Analysis output
                    st.markdown("### 📋 פלט ניתוח:")
                    if 'legal_analysis' in result:
                        if selected_test_type in ["table", "violation_count_table"]:
                            st.code(result['legal_analysis'], language="")
                        else:
                            st.markdown(result['legal_analysis'])
                    else:
                        st.info("הניתוח הסתיים אך אין פלט מעוצב.")

                    # Technical details
                    with st.expander("🔧 פרטים טכניים", expanded=False):
                        st.markdown("**תוצאת ניתוח:**")
                        st.json(result)

                except Exception as e:
                    st.error(f"❌ שגיאה: {str(e)}")
                    st.code(str(e))
    
with tab4 :
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
                        st.code(f"נוסחת סכום חסר תשלום: {check.get('amount_owed', 'לא זמין')}", language="python")
                
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

                    # Check Management within edit form
                    st.markdown("**בדיקות כלל:**")

                    # Display current checks
                    if st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb']:
                        st.markdown("**בדיקות נוכחיות:**")
                        for i, check in enumerate(st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb']):
                            with st.expander(f"בדיקה {i+1}: {check.get('violation_message', 'אין הודעה')}"):
                                st.code(f"מזהה בדיקה: {check.get('id', 'לא זמין')}")
                                st.code(f"תנאי: {check['condition']}")
                                st.code(f"נוסחת סכום חסר תשלום: {check.get('amount_owed', 'לא זמין')}")
                                # Remove button for each check
                                if st.form_submit_button(f"🗑️ הסר בדיקה {i+1}"):
                                    st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb'].pop(i)

                    # Add new check inputs
                    st.markdown("**הוסף בדיקה חדשה:**")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        edit_new_condition_heb = st.text_input("תנאי", key=f"edit_new_condition_{selected_rule_heb}_heb", help="למשל: attendance.overtime_hours > 0", placeholder="attendance.overtime_hours > 0")
                        edit_new_amount_owed_heb = st.text_input("נוסחת סכום חסר תשלום", key=f"edit_new_amount_owed_{selected_rule_heb}_heb", help="למשל: (contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)", placeholder="(contract.hourly_rate * 1.25 - payslip.overtime_rate) * min(attendance.overtime_hours, 2)")
                    with col2:
                        edit_new_violation_message_heb = st.text_input("הודעת הפרה", key=f"edit_new_violation_message_{selected_rule_heb}_heb", help="למשל: הפרת תעריף שעות נוספות", placeholder="הפרת תעריף שעות נוספות")

                    # Action buttons
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        edit_add_check_btn_heb = st.form_submit_button("➕ הוסף בדיקה")
                    with col2:
                        edit_clear_all_btn_heb = st.form_submit_button("🗑️ נקה הכל")
                    with col3:
                        edit_save_changes_btn_heb = st.form_submit_button("💾 שמור שינויים", type="primary")

                    # Handle form submissions
                    if edit_add_check_btn_heb:
                        if edit_new_condition_heb and edit_new_amount_owed_heb and edit_new_violation_message_heb:
                            st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb'].append({
                                "id": "",  # Will be generated by backend
                                "condition": edit_new_condition_heb,
                                "amount_owed": edit_new_amount_owed_heb,
                                "violation_message": edit_new_violation_message_heb
                            })
                            st.success("✅ בדיקה נוספה בהצלחה!")
                        else:
                            st.error("אנא מלא את כל השדות לבדיקה")

                    if edit_clear_all_btn_heb:
                        st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb'] = []
                        st.success("✅ כל הבדיקות נוקו!")

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

                                    # Validate checks structure
                                    validation_passed = True
                                    if not edit_checks_json_heb:
                                        st.error("❌ אנא הוסף לפחות בדיקה אחת")
                                        validation_passed = False

                                    for i, check in enumerate(edit_checks_json_heb):
                                        required_fields = ['condition', 'amount_owed', 'violation_message']  # id will be generated by backend
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
                                        "created_date": current_rule.get('created_date', datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')),
                                        "updated_date": datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                                    }

                                    rules_data['rules'][selected_rule_heb] = updated_rule
                                    if save_rules_data(rules_data):
                                        st.success(f"✅ כלל '{edit_rule_id_heb}' עודכן בהצלחה!")
                                        # Clear session state
                                        if f'edit_rule_checks_{selected_rule_heb}_heb' in st.session_state:
                                            del st.session_state[f'edit_rule_checks_{selected_rule_heb}_heb']
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
                                        st.code(f"נוסחת סכום חסר תשלום: {check.get('amount_owed', 'לא זמין')}")
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
                                st.metric("סכום חסר תשלום", f"₪{test_result['total_amount_owed']:.2f}")
                                st.markdown("### 🔍 ניתוח הפרה מפורט")
                                # Show each check calculation
                                for j, check in enumerate(test_result.get('rule_checks', [])):
                                    st.markdown(f"#### בדיקה {j+1}: {check.get('violation_message', 'אין הודעה')}")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**נוסחה:**")
                                        st.code(f"מזהה בדיקה: {check.get('id', 'לא זמין')}")
                                        st.code(f"תנאי: {check['condition']}")
                                        st.code(f"סכום חסר תשלום: {check.get('amount_owed', 'לא זמין')}")
                                    with col2:
                                        if j < len(test_result.get('check_results', [])):
                                            check_result = test_result['check_results'][j]
                                            st.markdown("**תוצאה:**")
                                            condition_met = check_result.get('condition_result', False)
                                            if condition_met:
                                                st.success("✅ תנאי: נכון")
                                            else:
                                                st.info("ℹ️ תנאי: לא נכון")
                                            amount = check_result.get('amount', 0)
                                            # Treat zero as a reported owed amount (matching engine semantics)
                                            if amount >= 0:
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
