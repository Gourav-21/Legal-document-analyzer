import streamlit as st
from PIL import Image
import io
from document_processor_openai import DocumentProcessor

# Page title and description
st.title("Legal Document Image Text Extractor")
st.write("Upload an image or Excel file to extract text using OpenAI")

# Initialize document processor
try:
    processor = DocumentProcessor()
    st.success("Document processor initialized successfully")
except Exception as e:
    st.error(f"Error initializing DocumentProcessor: {str(e)}")
    st.stop()

# Image upload widget
uploaded_file = st.file_uploader("Choose an image or Excel file", type=["png","pdf","docx", "jpg", "jpeg", "bmp", "tiff", "xlsx"])

if uploaded_file is not None:
    filename = uploaded_file.name
    file_extension = filename.split(".")[-1].lower()

    if file_extension in ["png", "jpg", "jpeg", "bmp", "tiff"]:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Process button
    if st.button("Extract Text"):
        with st.spinner("Extracting text..."):
            try:
                # Get file content as bytes
                file_content = uploaded_file.getvalue()

                if file_extension == "xlsx":
                    # Handle Excel file processing
                    extracted_text = processor._extract_text2(file_content, filename, False)
                else:
                    # Handle image file processing
                    extracted_text = processor._extract_text2(file_content, filename, True)
                
                # Display results in a more prominent way
                st.subheader("Extracted Text:")
                
                # Show text in a better formatted way with a larger text area
                st.text_area("Extracted Content", extracted_text, height=400)
                
                # Also show the text as regular text for better readability
                with st.expander("View as formatted text", expanded=True):
                    st.markdown(extracted_text)
                
                # Keep download as a secondary option
                st.download_button(
                    label="Download as Text File",
                    data=extracted_text,
                    file_name=f"{uploaded_file.name}_extracted_text.txt",
                    mime="text/plain",
                    help="Save the extracted text to a file"
                )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")