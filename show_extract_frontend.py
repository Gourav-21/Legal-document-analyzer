# Fix for SQLite3 compatibility with ChromaDB on Streamlit Cloud
import sqlite_fix  # This must be imported first

import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from document_processor_gemini import DocumentProcessor
import json

# Set page config
st.set_page_config(
    page_title="Document Processing Demo",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 2rem;
    }
    .section-header {
        background-color: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
    .image-container {
        border: 2px dashed #ccc;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def convert_image_to_base64(image_bytes):
    """Convert image bytes to base64 string for display"""
    return base64.b64encode(image_bytes).decode()

def display_image_comparison(original_bytes, processed_bytes, title1="Original Image", title2="Processed Image"):
    """Display two images side by side for comparison"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {title1}")
        st.image(original_bytes, use_container_width=True)
    
    with col2:
        st.markdown(f"### {title2}")
        st.image(processed_bytes, use_container_width=True)

def main():
    # Header
    st.markdown('<div class="main-header"><h1>📄 Document Processing Demo</h1><p>Test image preprocessing and document extraction functions</p></div>', unsafe_allow_html=True)
    
    # Initialize DocumentProcessor
    try:
        processor = DocumentProcessor()
        st.success("✅ Document processor initialized successfully")
    except Exception as e:
        st.error(f"❌ Error initializing DocumentProcessor: {str(e)}")
        st.stop()
    
    # Sidebar for options
    st.sidebar.header("Options")
    demo_mode = st.sidebar.selectbox(
        "Select Demo Mode",
        ["Image Preprocessing", "Document Processing", "Both"]
    )
    
    # File upload
    st.markdown('<div class="section-header"><h3>📁 File Upload</h3></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file to process",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"],
        help="Upload an image or PDF file for processing"
    )
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        file_bytes = uploaded_file.getvalue()
        
        # Display file info
        st.info(f"📄 **File:** {filename} | **Size:** {len(file_bytes)} bytes | **Type:** {uploaded_file.type}")
        
        # Determine document type
        doc_type = st.sidebar.selectbox(
            "Document Type",
            ["payslip", "attendance", "contract"],
            help="Select the type of document for processing"
        )
        
        if demo_mode in ["Image Preprocessing", "Both"]:
            st.markdown('<div class="section-header"><h3>🔧 Image Preprocessing Demo</h3></div>', unsafe_allow_html=True)
            
            if st.button("🚀 Run Image Preprocessing", key="preprocess_btn"):
                with st.spinner("Processing image..."):
                    try:
                        # Convert to image if needed
                        if filename.lower().endswith('.pdf'):
                            st.warning("PDF detected - converting first page to image for preprocessing demo")
                            import pdfplumber
                            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                                page = pdf.pages[0]
                                img = page.to_image(resolution=300).original
                                buf = BytesIO()
                                img.save(buf, format="PNG")
                                image_bytes = buf.getvalue()
                        else:
                            image_bytes = processor.ensure_image(file_bytes)
                        
                        # Apply preprocessing
                        processed_bytes = processor.preprocess_image(image_bytes)
                        
                        # Display results
                        st.markdown("### 📊 Preprocessing Results")
                        display_image_comparison(image_bytes, processed_bytes, "Original Image", "Preprocessed Image")
                        
                        # Show processing details
                        with st.expander("🔍 Processing Details"):
                            st.markdown("""
                            **Preprocessing Steps Applied:**
                            1. **Color Conversion**: RGB → Grayscale
                            2. **Denoising**: FastNlMeansDenoising (h=25, templateWindowSize=7, searchWindowSize=21)
                            3. **Thresholding**: Adaptive Gaussian threshold (maxValue=255, blockSize=31, C=2)
                            4. **Deskewing**: MinAreaRect angle detection and rotation matrix correction
                            5. **Format**: Final conversion to PNG format
                            """)
                        
                        # Download options
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "💾 Download Original",
                                data=image_bytes,
                                file_name=f"original_{filename}",
                                mime="image/png"
                            )
                        with col2:
                            st.download_button(
                                "💾 Download Processed",
                                data=processed_bytes,
                                file_name=f"processed_{filename}",
                                mime="image/png"
                            )
                            
                    except Exception as e:
                        st.error(f"❌ Error in preprocessing: {str(e)}")
        
        if demo_mode in ["Document Processing", "Both"]:
            st.markdown('<div class="section-header"><h3>📝 Document Processing Demo</h3></div>', unsafe_allow_html=True)
            
            if st.button("🚀 Run Document Processing", key="process_btn"):
                with st.spinner("Processing document..."):
                    try:
                        # Process document
                        result = processor.process(file_bytes, filename, doc_type)
                        
                        # Display results
                        st.markdown("### 📋 Processing Results")
                        
                        # Show extracted text
                        if 'text' in result and result['text']:
                            with st.expander("📄 Extracted Text", expanded=True):
                                st.text_area("Full Text", result['text'], height=300)
                        
                        # Show extracted fields
                        if 'fields' in result and result['fields']:
                            with st.expander("🏷️ Extracted Fields", expanded=True):
                                if isinstance(result['fields'], dict):
                                    for key, value in result['fields'].items():
                                        st.write(f"**{key}:** {value}")
                                else:
                                    st.json(result['fields'])
                          # Show tables with enhanced DataFrame display
                        if 'tables' in result and result['tables']:
                            with st.expander("📊 Extracted Tables", expanded=True):
                                # Add table display options
                                col1, col2, col3 = st.columns([2, 1, 1])
                                with col1:
                                    st.markdown("### Tables Found in Document")
                                with col2:
                                    table_format = st.selectbox(
                                        "Display Format",
                                        ["DataFrame", "JSON", "Raw"],
                                        key="table_format"
                                    )
                                with col3:
                                    show_stats = st.checkbox("Show Stats", value=True, key="show_stats")
                                
                                for i, table in enumerate(result['tables']):
                                    st.markdown(f"#### 📋 Table {i+1}")
                                    
                                    if table:  # Check if table is not empty
                                        if table_format == "DataFrame":
                                            import pandas as pd
                                            try:
                                                # Create DataFrame with proper handling
                                                if len(table) > 1:
                                                    headers = table[0] if table[0] else [f"Column_{j+1}" for j in range(len(table[1]))]
                                                    df = pd.DataFrame(table[1:], columns=headers)
                                                else:
                                                    df = pd.DataFrame(table)
                                                
                                                # Display DataFrame
                                                st.dataframe(
                                                    df, 
                                                    use_container_width=True,
                                                    hide_index=False
                                                )
                                                
                                                # Show statistics if enabled
                                                if show_stats and not df.empty:
                                                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                                                    with col_stats1:
                                                        st.metric("Rows", len(df))
                                                    with col_stats2:
                                                        st.metric("Columns", len(df.columns))
                                                    with col_stats3:
                                                        st.metric("Non-empty cells", df.count().sum())
                                                
                                                # Download options
                                                col_dl1, col_dl2 = st.columns(2)
                                                with col_dl1:
                                                    csv_data = df.to_csv(index=False)
                                                    st.download_button(
                                                        f"💾 Download as CSV",
                                                        data=csv_data,
                                                        file_name=f"table_{i+1}_{filename}.csv",
                                                        mime="text/csv",
                                                        key=f"csv_{i}"
                                                    )
                                                with col_dl2:
                                                    excel_buffer = BytesIO()
                                                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                                        df.to_excel(writer, index=False, sheet_name=f'Table_{i+1}')
                                                    excel_data = excel_buffer.getvalue()
                                                    st.download_button(
                                                        f"📊 Download as Excel",
                                                        data=excel_data,
                                                        file_name=f"table_{i+1}_{filename}.xlsx",
                                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                        key=f"excel_{i}"
                                                    )
                                                
                                            except Exception as e:
                                                st.error(f"Error creating DataFrame: {str(e)}")
                                                st.json(table)
                                        
                                        elif table_format == "JSON":
                                            st.json(table)
                                        
                                        else:  # Raw format
                                            st.code(str(table), language="python")
                                    
                                    else:
                                        st.info("Empty table")
                                    
                                    # Add separator between tables
                                    if i < len(result['tables']) - 1:
                                        st.divider()
                        
                        else:
                            # Show message when no tables found
                            with st.expander("📊 Extracted Tables"):
                                st.info("No tables were detected in this document.")
                                st.markdown("""
                                **Possible reasons:**
                                - The document doesn't contain tabular data
                                - Tables are embedded in images and need preprocessing
                                - Table structure is too complex for automatic detection
                                
                                💡 **Tip:** Try using 'Image Preprocessing' first if tables are not detected.
                                """)
                        
                        # Show full JSON result
                        with st.expander("🔧 Full JSON Result"):
                            st.json(result)
                        
                        # Download option
                        result_json = json.dumps(result, ensure_ascii=False, indent=2)
                        st.download_button(
                            "💾 Download Results (JSON)",
                            data=result_json,
                            file_name=f"results_{filename}.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error in document processing: {str(e)}")
                        st.exception(e)
        
        # Additional analysis section
        if demo_mode == "Both" and st.button("🔬 Run Complete Analysis", key="complete_btn"):
            st.markdown('<div class="section-header"><h3>🔬 Complete Analysis</h3></div>', unsafe_allow_html=True)
            
            with st.spinner("Running complete analysis..."):
                try:
                    # First preprocess if it's an image
                    if not filename.lower().endswith('.pdf'):
                        image_bytes = processor.ensure_image(file_bytes)
                        processed_bytes = processor.preprocess_image(image_bytes)
                        
                        st.markdown("#### 1. Image Preprocessing Comparison")
                        display_image_comparison(image_bytes, processed_bytes)
                    
                    # Then process document
                    result = processor.process(file_bytes, filename, doc_type)
                    
                    st.markdown("#### 2. Document Processing Results")
                    
                    # Create metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        text_length = len(result.get('text', ''))
                        st.metric("Text Length", f"{text_length} chars")
                    
                    with col2:
                        fields_count = len(result.get('fields', {}))
                        st.metric("Fields Extracted", fields_count)
                    
                    with col3:
                        tables_count = len(result.get('tables', []))
                        st.metric("Tables Found", tables_count)
                    
                    with col4:
                        processing_type = "Multi-step" if not filename.lower().endswith('.pdf') else "Direct"
                        st.metric("Processing Type", processing_type)
                    
                    # Detailed results
                    tabs = st.tabs(["📄 Text", "🏷️ Fields", "📊 Tables", "📈 Analysis"])
                    
                    with tabs[0]:
                        if result.get('text'):
                            st.text_area("Extracted Text", result['text'], height=400)
                        else:
                            st.info("No text extracted")
                    
                    with tabs[1]:
                        if result.get('fields'):
                            st.json(result['fields'])
                        else:
                            st.info("No structured fields extracted")
                    
                    with tabs[2]:
                        if result.get('tables'):
                            for i, table in enumerate(result['tables']):
                                st.markdown(f"**Table {i+1}:**")
                                if table:
                                    try:
                                        import pandas as pd
                                        df = pd.DataFrame(table[1:], columns=table[0] if len(table) > 0 else [])
                                        st.dataframe(df, use_container_width=True)
                                    except Exception:
                                        st.json(table)
                        else:
                            st.info("No tables extracted")
                    
                    with tabs[3]:
                        st.markdown("**Processing Summary:**")
                        st.write(f"- Document type: {doc_type}")
                        st.write(f"- File format: {filename.split('.')[-1].upper()}")
                        st.write(f"- Text extracted: {'Yes' if result.get('text') else 'No'}")
                        st.write(f"- Fields identified: {len(result.get('fields', {}))}")
                        st.write(f"- Tables found: {len(result.get('tables', []))}")
                        
                        if result.get('text'):
                            # Basic text analysis
                            text = result['text']
                            words = len(text.split())
                            lines = len(text.split('\n'))
                            st.write(f"- Word count: {words}")
                            st.write(f"- Line count: {lines}")
                    
                except Exception as e:
                    st.error(f"❌ Error in complete analysis: {str(e)}")
                    st.exception(e)
    
    else:
        # Show demo instructions
        st.markdown('<div class="section-header"><h3>🎯 Demo Instructions</h3></div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Welcome to the Document Processing Demo!
        
        This application demonstrates two key functions from the DocumentProcessor:
        
        #### 🔧 Image Preprocessing (`preprocess_image`)
        - Converts images to grayscale
        - Applies denoising algorithms
        - Performs adaptive thresholding
        - Corrects skew/rotation
        - Optimizes for OCR processing
        
        #### 📝 Document Processing (`process`)
        - Extracts text using Google Vision API
        - Falls back to Tesseract OCR if needed
        - Identifies structured fields
        - Extracts tables from documents
        - Supports payslips, attendance records, and contracts
        
        **To get started:**
        1. Upload a document file (image or PDF)
        2. Select the document type
        3. Choose your demo mode
        4. Click the processing buttons
        
        **Supported formats:** PNG, JPG, JPEG, BMP, TIFF, PDF
        """)

if __name__ == "__main__":
    main()