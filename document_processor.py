from fastapi import UploadFile, HTTPException
from PIL import Image
import pytesseract
import google.generativeai as genai
import os
import pdfplumber
from io import BytesIO
from typing import List, Dict, Union
from docx import Document

class DocumentProcessor:
    def __init__(self):
        # Initialize Gemini AI
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Configure Tesseract path
        tesseract_path = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if not os.path.exists(tesseract_path):
            raise Exception("Tesseract not found. Please install Tesseract and set TESSERACT_CMD environment variable.")
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
    def process_document(self, files: Union[UploadFile, List[UploadFile]], doc_types: Union[str, List[str]]):
        if not files:
            raise HTTPException(
                status_code=400,
                detail="At least one document must be provided"
            )
        
        # Handle single file case
        if isinstance(files, UploadFile):
            files = [files]
            doc_types = [doc_types]
        elif not isinstance(doc_types, list):
            doc_types = [doc_types] * len(files)
        
        # Initialize text storage
        payslip_text = None
        contract_text = None
        
        # Process each file based on its type
        for file, doc_type in zip(files, doc_types):
            extracted_text = self._extract_text(file.file.read(), file.filename)
            if doc_type.lower() == "payslip":
                payslip_text = extracted_text
            elif doc_type.lower() == "contract":
                contract_text = extracted_text

        # Analyze the documents
        result = self.analyze_violations(payslip_text, contract_text)
        return result
    
    def analyze_violations(self, payslip_text: str = None, contract_text: str = None) -> Dict:
        # Prepare documents for analysis
        documents = {}
        if payslip_text:
            documents['payslip'] = payslip_text
        if contract_text:
            documents['contract'] = contract_text
            
        # Prepare the prompt for Gemini AI
        prompt = f"""Analyze the following employment documents for potential legal violations under current labor laws.

Documents to analyze:
{', '.join(documents.keys())}

"""        
        
        # Add document contents to prompt
        for doc_type, content in documents.items():
            prompt += f"\n{doc_type.upper()} CONTENT:\n{content}\n"
            
        prompt += """
\nPlease provide a comprehensive legal analysis including:
1. Identified violations of labor laws
2. Summary of relevant labor laws and regulations
3. Similar case precedents
4. Legal recommendations (including whether legal action is advisable)
5. Supporting evidence from the documents

Format the response in a structured way."""

        try:
            # Generate analysis using Gemini AI
            response = self.model.generate_content(prompt)
            analysis = response.text
            
            # Structure the result
            result = {
                "legal_analysis": analysis,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating legal analysis: {str(e)}"
            )

            
    def _extract_text(self, content: bytes, filename: str) -> str:
        if filename.lower().endswith('.pdf'):
            try:
                pdf_file = BytesIO(content)
                text = ''
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += page_text + "\n"
                        else:
                            img = page.to_image(resolution=300).original
                            text += pytesseract.image_to_string(img) + "\n"
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing PDF: {str(e)}"
                )
        elif filename.lower().endswith('.docx'):
            doc_file = BytesIO(content)
            doc = Document(doc_file)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            image = Image.open(BytesIO(content))
            text = pytesseract.image_to_string(image, lang='eng+heb')
        return text
    