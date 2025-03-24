import streamlit as st
from document_processor import DocumentProcessor
from fastapi import UploadFile
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize DocumentProcessor
doc_processor = DocumentProcessor()

# Set page config
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# Title and description
st.title("📄 Legal Document Analyzer")
st.markdown("Upload your legal documents to check compliance with Israeli labor laws.")

# Create document upload sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Payslip Documents")
    payslip_files = st.file_uploader(
        "Upload payslip documents",
        type=["pdf", "png", "jpg", "jpeg", 'webp', 'docx'],
        accept_multiple_files=True,
        key="payslip_upload"
    )

with col2:
    st.subheader("📑 Contract Documents")
    contract_files = st.file_uploader(
        "Upload contract documents",
        type=["pdf", "png", "jpg", "jpeg", 'webp', 'docx'],
        accept_multiple_files=True,
        key="contract_upload"
    )

# Add process button
if (payslip_files or contract_files) and st.button("Process Documents", type="primary"):
    try:
        with st.spinner("Processing documents..."):
            # Process documents
            all_files = []
            all_doc_types = []
            
            # Add payslip files
            if payslip_files:
                for file in payslip_files:
                    file_content = file.read()
                    fastapi_file = UploadFile(
                        filename=file.name,
                        file=BytesIO(file_content)
                    )
                    all_files.append(fastapi_file)
                    all_doc_types.append("payslip")
            
            # Add contract files
            if contract_files:
                for file in contract_files:
                    file_content = file.read()
                    fastapi_file = UploadFile(
                        filename=file.name,
                        file=BytesIO(file_content)
                    )
                    all_files.append(fastapi_file)
                    all_doc_types.append("contract")
            
            # Process all documents
            result = doc_processor.process_document(all_files, all_doc_types)
            
            # Display results
            if result.get('legal_analysis'):
                analysis = result['legal_analysis']
                st.markdown("### Legal Analysis Results")
                st.markdown(analysis)
                
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        st.info("Please ensure all documents are in the correct format and try again.")

# Add footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>Powered by AI & Labor Law Database</div>",
    unsafe_allow_html=True
)