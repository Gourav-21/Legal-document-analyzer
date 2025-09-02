#!/usr/bin/env python3
"""
Final test to verify Streamlit integration shows different analysis types correctly
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Analysis Types Test", layout="wide")

st.title("🎯 Analysis Types Test - Verification")
st.markdown("**Testing all 6 analysis types to verify they show different outputs**")

# Import the analyze function
try:
    from document_processor_pydantic_ques import DocumentProcessor
    
    def analyze_payslip_direct(payslip_data, attendance_data, contract_data, analysis_type="rule_based"):
        """Direct analysis function for testing"""
        import asyncio
        
        try:
            processor = DocumentProcessor()
            
            payslip_list = [payslip_data] if payslip_data else []
            attendance_list = [attendance_data] if attendance_data else []
            contract_dict = contract_data if contract_data else {}
            
            def run_async_analysis():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(
                        processor.create_report_with_rule_engine(
                            payslip_data=payslip_list,
                            attendance_data=attendance_list,
                            contract_data=contract_dict,
                            analysis_type=analysis_type
                        )
                    )
                finally:
                    loop.close()
            
            result = run_async_analysis()
            
            return {
                'legal_analysis': result.get('legal_analysis', ''),
                'violations_count': result.get('violations_count', 0),
                'total_underpaid': result.get('total_underpaid', 0),
                'total_penalties': result.get('total_penalties', 0),
                'total_combined': result.get('total_combined', 0),
                'analysis_type': analysis_type
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    # Test data
    test_payslip = {
        "employee_id": "TEST_001",
        "month": "2024-01",
        "base_salary": 5000.0,
        "overtime_hours": 10,
        "overtime_pay": 300.0,
        "overtime_rate": 30.0,
        "total_salary": 5300.0,
        "hours_worked": 196,
        "hourly_rate": 26.88
    }
    
    test_attendance = {
        "employee_id": "TEST_001",
        "month": "2024-01",
        "days_worked": 22,
        "regular_hours": 186,
        "overtime_hours": 10,
        "total_hours": 196
    }
    
    test_contract = {
        "minimum_wage_hourly": 29.12,
        "hourly_rate": 26.88,
        "overtime_rate_125": 1.25,
        "overtime_rate_150": 1.50
    }
    
    # Analysis types
    analysis_types = [
        ("violations_list", "📋 Simple Violations List"),
        ("easy", "😊 User-Friendly Summary"),
        ("table", "📊 Organized Table Format"),
        ("violation_count_table", "📈 Statistics Table"),
        ("rule_based", "⚖️ Detailed Legal Analysis"),
        ("report", "📄 Professional Report")
    ]
    
    st.markdown("---")
    
    # Test all types button
    if st.button("🚀 Test All Analysis Types", type="primary"):
        st.markdown("## 📊 Analysis Results")
        
        for analysis_type, description in analysis_types:
            with st.expander(f"{description} ({analysis_type})", expanded=True):
                with st.spinner(f"Running {analysis_type} analysis..."):
                    result = analyze_payslip_direct(test_payslip, test_attendance, test_contract, analysis_type)
                    
                    if 'error' in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        # Show metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Violations", result.get('violations_count', 0))
                        with col2:
                            st.metric("Underpaid", f"₪{result.get('total_underpaid', 0):.2f}")
                        with col3:
                            st.metric("Penalties", f"₪{result.get('total_penalties', 0):.2f}")
                        with col4:
                            st.metric("Total", f"₪{result.get('total_combined', 0):.2f}")
                        
                        # Show analysis
                        st.markdown("### 📋 Analysis Output:")
                        analysis_text = result.get('legal_analysis', 'No analysis available')
                        
                        # Show character count and first line for verification
                        st.caption(f"Length: {len(analysis_text)} characters | Type: {analysis_type}")
                        
                        if analysis_type in ["table", "violation_count_table"]:
                            st.code(analysis_text, language="")
                        else:
                            st.markdown(analysis_text)
    
    # Individual type selector
    st.markdown("---")
    st.subheader("🔍 Test Individual Type")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_type = st.selectbox(
            "Choose Analysis Type:",
            options=[at[0] for at in analysis_types],
            format_func=lambda x: next(desc for code, desc in analysis_types if code == x)
        )
    
    with col2:
        if st.button("🧪 Test Selected Type"):
            result = analyze_payslip_direct(test_payslip, test_attendance, test_contract, selected_type)
            
            if 'error' in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                st.success(f"✅ {selected_type} analysis completed!")
                
                # Show results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Violations", result.get('violations_count', 0))
                with col2:
                    st.metric("Underpaid", f"₪{result.get('total_underpaid', 0):.2f}")
                with col3:
                    st.metric("Penalties", f"₪{result.get('total_penalties', 0):.2f}")
                with col4:
                    st.metric("Total", f"₪{result.get('total_combined', 0):.2f}")
                
                st.markdown("### 📋 Analysis Output:")
                analysis_text = result.get('legal_analysis', 'No analysis available')
                st.caption(f"Length: {len(analysis_text)} characters | Type: {selected_type}")
                
                if selected_type in ["table", "violation_count_table"]:
                    st.code(analysis_text, language="")
                else:
                    st.markdown(analysis_text)

except ImportError as e:
    st.error(f"❌ Failed to import DocumentProcessor: {e}")
    st.info("Make sure document_processor_pydantic_ques.py is available")

except Exception as e:
    st.error(f"❌ Unexpected error: {e}")
    st.code(str(e))