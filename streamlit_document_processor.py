import streamlit as st
import asyncio
import json
import pandas as pd
from io import BytesIO
from typing import List, Dict, Any
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import the document processor
from document_processor_pydantic_ques import DocumentProcessor

# Page configuration
st.set_page_config(
    page_title="Legal Document Processor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.upload-section {
    background-color: #f8f9fa;
    padding: 2rem;
    border-radius: 10px;
    border: 2px dashed #dee2e6;
    margin: 1rem 0;
}
.result-card {
    background-color: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.success-card {
    border-left: 4px solid #28a745;
}
.warning-card {
    border-left: 4px solid #ffc107;
}
.error-card {
    border-left: 4px solid #dc3545;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📄 Legal Document Processor</h1>', unsafe_allow_html=True)
st.markdown("**Extract structured data from legal documents using AI-powered processing**")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Document types
    st.subheader("Document Types")
    doc_types = ["payslip", "attendance", "contract", "employee"]
    selected_types = {}
    for doc_type in doc_types:
        selected_types[doc_type] = st.checkbox(
            f"Enable {doc_type.title()}",
            value=True,
            help=f"Process {doc_type} documents"
        )

    # Processing options
    st.subheader("Processing Options")
    compress_images = st.checkbox(
        "Compress Images",
        value=False,
        help="Compress images before processing to reduce API costs"
    )

    # API Status
    st.subheader("API Status")
    try:
        processor = DocumentProcessor()
        st.success("✅ Document Processor Ready")
    except Exception as e:
        st.error(f"❌ Initialization Error: {str(e)}")
        processor = None

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Document Upload")

    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload legal documents",
        type=["pdf", "png", "jpg", "jpeg", "xlsx", "xls", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload multiple documents for batch processing"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")

        # Display uploaded files
        st.subheader("📋 Uploaded Files")
        file_info = []
        for i, file in enumerate(uploaded_files):
            file_type = file.name.split('.')[-1].upper()
            file_size = f"{file.size / 1024:.1f} KB"
            file_info.append({
                "File": file.name,
                "Type": file_type,
                "Size": file_size,
                "Index": i
            })

        if file_info:
            st.dataframe(pd.DataFrame(file_info), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Document type assignment
    if uploaded_files:
        st.subheader("🏷️ Document Type Assignment")

        # Create a mapping of files to document types
        file_type_mapping = {}

        # Group files by suggested type based on filename
        suggested_mappings = {}
        for i, file in enumerate(uploaded_files):
            filename = file.name.lower()
            suggested_type = "contract"  # default

            if any(keyword in filename for keyword in ["payslip", "salary", "pay"]):
                suggested_type = "payslip"
            elif any(keyword in filename for keyword in ["attendance", "hours", "work"]):
                suggested_type = "attendance"
            elif any(keyword in filename for keyword in ["contract", "agreement"]):
                suggested_type = "contract"

            suggested_mappings[i] = suggested_type

        # Create columns for file type selection
        cols = st.columns(min(3, len(uploaded_files)))
        for i, file in enumerate(uploaded_files):
            col_idx = i % 3
            with cols[col_idx]:
                st.markdown(f"**{file.name}**")
                file_type_mapping[i] = st.selectbox(
                    f"Document Type for {file.name}",
                    options=["payslip", "attendance", "contract", "employee"],
                    index=["payslip", "attendance", "contract", "employee"].index(suggested_mappings[i]),
                    key=f"type_{i}"
                )

with col2:
    st.subheader("📊 Processing Status")

    # Processing button
    if uploaded_files and processor:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents... This may take a few minutes."):

                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Prepare files and types for processing
                    files_to_process = []
                    types_to_process = []

                    for i, file in enumerate(uploaded_files):
                        if selected_types.get(file_type_mapping[i], False):
                            files_to_process.append(file)
                            types_to_process.append(file_type_mapping[i])

                    if not files_to_process:
                        st.error("No files selected for processing. Please enable at least one document type.")
                        st.stop()

                    status_text.text(f"Processing {len(files_to_process)} document(s)...")

                    # Process documents asynchronously
                    async def process_async():
                        # Convert uploaded files to UploadFile-like objects
                        from fastapi import UploadFile
                        upload_files = []

                        for file in files_to_process:
                            # Create a file-like object
                            file_content = file.getvalue()
                            upload_file = type('UploadFile', (), {
                                'filename': file.name,
                                'read': lambda content=file_content: content,
                                'file': BytesIO(file_content)
                            })()
                            upload_files.append(upload_file)

                        # Process documents
                        result = await processor.process_document(
                            files=upload_files,
                            doc_types=types_to_process,
                            compress=compress_images
                        )
                        return result

                    # Run async processing
                    result = asyncio.run(process_async())

                    progress_bar.progress(100)
                    status_text.text("✅ Processing completed!")

                    # Store result in session state for display
                    st.session_state.processing_result = result
                    st.session_state.processed_files = len(files_to_process)

                except Exception as e:
                    progress_bar.progress(0)
                    status_text.text("❌ Processing failed!")
                    st.error(f"Error during processing: {str(e)}")

    elif not processor:
        st.error("Document processor not initialized. Check API keys.")

# Results display section
if 'processing_result' in st.session_state:
    st.header("📈 Processing Results")

    result = st.session_state.processing_result

    # Create tabs for different result types
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Summary", "💰 Payslips", "⏰ Attendance", "📋 Contracts", "👤 Employees"])

    with tab1:
        st.subheader("Processing Summary")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            payslip_count = len(result.get('payslip_data', []))
            st.metric("Payslip Documents", payslip_count)

        with col2:
            attendance_count = len(result.get('attendance_data', []))
            st.metric("Attendance Records", attendance_count)

        with col3:
            contract_count = 1 if result.get('contract_data') else 0
            st.metric("Contract Documents", contract_count)

        with col4:
            total_processed = st.session_state.get('processed_files', 0)
            st.metric("Total Processed", total_processed)

        # Display raw result for debugging
        with st.expander("🔍 Raw Processing Result (JSON)"):
            st.json(result)

    with tab2:
        st.subheader("💰 Payslip Data")

        payslip_data = result.get('payslip_data', [])
        if payslip_data:
            # Convert to DataFrame for better display
            df_payslips = pd.DataFrame(payslip_data)

            # Display as table
            st.dataframe(df_payslips, use_container_width=True)

            # Show individual payslips
            for i, payslip in enumerate(payslip_data):
                with st.expander(f"Payslip #{i+1} - Employee {payslip.get('employee_id', 'Unknown')}"):
                    st.json(payslip)
        else:
            st.info("No payslip data processed")

    with tab3:
        st.subheader("⏰ Attendance Data")

        attendance_data = result.get('attendance_data', [])
        if attendance_data:
            # Convert to DataFrame
            df_attendance = pd.DataFrame(attendance_data)

            # Display as table
            st.dataframe(df_attendance, use_container_width=True)

            # Show individual attendance records
            for i, attendance in enumerate(attendance_data):
                with st.expander(f"Attendance #{i+1} - Employee {attendance.get('employee_id', 'Unknown')}"):
                    st.json(attendance)
        else:
            st.info("No attendance data processed")

    with tab4:
        st.subheader("📋 Contract Data")

        contract_data = result.get('contract_data', {})
        if contract_data:
            # Display contract data
            st.json(contract_data)

            # Try to display as a more readable format
            with st.expander("📄 Contract Details"):
                for key, value in contract_data.items():
                    if key != 'document_number':  # Skip internal fields
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.info("No contract data processed")

    # Export functionality
    st.header("💾 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download JSON", use_container_width=True):
            json_data = json.dumps(result, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="processing_results.json",
                mime="application/json",
                use_container_width=True
            )

    with col2:
        if st.button("📊 Download Excel", use_container_width=True):
            # Create Excel file with multiple sheets
            output = BytesIO()

            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Payslip data
                if result.get('payslip_data'):
                    pd.DataFrame(result['payslip_data']).to_excel(writer, sheet_name='Payslips', index=False)

                # Attendance data
                if result.get('attendance_data'):
                    pd.DataFrame(result['attendance_data']).to_excel(writer, sheet_name='Attendance', index=False)

                # Contract data
                if result.get('contract_data'):
                    pd.DataFrame([result['contract_data']]).to_excel(writer, sheet_name='Contract', index=False)

            output.seek(0)
            st.download_button(
                label="Download Excel",
                data=output,
                file_name="processing_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    with col3:
        if st.button("🧹 Clear Results", use_container_width=True):
            del st.session_state.processing_result
            del st.session_state.processed_files
            st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and AI-powered document processing")
st.markdown("*For support or issues, please check the application logs.*")