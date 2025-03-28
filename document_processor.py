from fastapi import UploadFile, HTTPException
from PIL import Image
import pytesseract
import google.generativeai as genai
import os
import pdfplumber
from io import BytesIO
from typing import List, Dict, Union
from docx import Document
from labour_law import LaborLawStorage
from google.cloud import vision
import io
from google.cloud.vision import ImageAnnotatorClient

class DocumentProcessor:
    def __init__(self):
        # Initialize Gemini AI
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize Labor Law Storage
        self.law_storage = LaborLawStorage()
        
        # Configure Tesseract path
        tesseract_path = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if not os.path.exists(tesseract_path):
            raise Exception("Tesseract not found. Please install Tesseract and set TESSERACT_CMD environment variable.")
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Initialize Vision client with API key
        vision_api_key = os.getenv("GOOGLE_CLOUD_VISION_API_KEY")
        if not vision_api_key:
            raise Exception("Google Cloud Vision API key not found. Please set GOOGLE_CLOUD_VISION_API_KEY environment variable.")
        
        self.vision_client = ImageAnnotatorClient(client_options={"api_key": vision_api_key})
        self.image_context = {"language_hints": ["he"]} 

        
    def process_document(self, files: Union[UploadFile, List[UploadFile]], doc_types: Union[str, List[str]],user:str):
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
        
        # Initialize payslip counter
        payslip_counter = 0
        # Process each file based on its type
        for file, doc_type in zip(files, doc_types):
            extracted_text = self._extract_text2(file.file.read(), file.filename)
            if doc_type.lower() == "payslip":
                payslip_counter += 1
                payslip_text = f"Payslip {payslip_counter}:\n{extracted_text}" if payslip_text is None else f"{payslip_text}\n\nPayslip {payslip_counter}:\n{extracted_text}"
            elif doc_type.lower() == "contract":
                contract_text = extracted_text
        # Analyze the documents
        result = self.analyze_violations(payslip_text, contract_text,user)
        return result
    
    def analyze_violations(self, payslip_text: str = None, contract_text: str = None,user:str = None) -> Dict:
        # Prepare documents for analysis
        documents = {}
        if payslip_text:
            documents['payslip'] = payslip_text
        if contract_text:
            documents['contract'] = contract_text
            
        # Prepare the prompt for Gemini AI
        # Get formatted labor laws
        labor_laws = self.law_storage.format_laws_for_prompt()
        
        prompt = f"""You are a legal document analyzer specializing in Israeli labor law compliance based *only* on user-provided information.

LABOR LAWS TO CHECK AGAINST(Your ONLY reference point):
{labor_laws if labor_laws else 'No labor laws provided for analysis.'}

DOCUMENTS PROVIDED FOR ANALYSIS:
{', '.join(documents.keys())}

"""        
        
        # Add document contents to prompt
        for doc_type, content in documents.items():
            prompt += f"\n{doc_type.upper()} CONTENT:\n{content}\n"
            
        prompt += f"""
INSTRUCTIONS:
1. If no labor laws are provided, respond with: "אין חוקים לעבודה זמינים לניתוח התאמה." in Hebrew.
2. If labor laws exist, analyze the documents ONLY against the provided laws.
3. For each violation found, format the response EXACTLY as shown below, with each section on a new line and proper spacing:

Violation Format Template:

{"""[VIOLATION TITLE]

[SPECIFIC VIOLATION DETAILS]

[LAW REFERENCE AND YEAR]

[SIMILAR CASES OR PRECEDENTS](I request to find a similar legal ruling for this case in israel from any sources, describing RESULT)

[LEGAL IMPLICATIONS]

[RECOMMENDED ACTIONS]

---

Example of correctly formatted violation:

Possible Violation of Mandatory Pension Expansion Order

The payslip shows no pension contribution, despite over 6 months of employment.

According to the Mandatory Pension Expansion Order (2008), an employer is required to contribute to pension after 6 months of continuous employment (or 3 months with prior pension history).

In the [NAME OF RULING] ruling, there was a similar case and the employee won 10,000 thousand shekels there

Lack of contribution may entitle the employee to retroactive compensation or legal action.

It is recommended to review the pension fund details, start date of employment, and contract terms.

---
""" if user == 'lawyer' else """
[VIOLATION TITLE]

[SIMPLE EXPLANATION OF WHAT THE EMPLOYER MIGHT OWE/DO]

---

Example of correctly formatted violation:
    
Your hourly wage appears to be below the legal minimum (32.79 ILS/hour).

You may be entitled to compensation from your employer.

---

"""}
IMPORTANT:
- Always Respond in Hebrew
- Format each violation with proper spacing and line breaks as shown above
- Separate multiple violations with '---'
- If no violations are found against the provided laws, respond with: "לא נמצאו הפרות נגד חוקי העבודה שסופקו." in hebrew
- Do not include any additional commentary or explanations outside of the violation format"""
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
    # google vision api
    def _compress_image(self, image_bytes: bytes, max_size_mb: int = 4) -> bytes:
        # Convert size to bytes
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # If image is already smaller than max size, return original
        if len(image_bytes) <= max_size_bytes:
            return image_bytes
        
        # Open image using PIL
        img = Image.open(BytesIO(image_bytes))
        
        # Get original dimensions
        width, height = img.size
        
        # Scale down large images while maintaining aspect ratio
        max_dimension = 3000  # Maximum dimension for width or height
        if width > max_dimension or height > max_dimension:
            scale = min(max_dimension / width, max_dimension / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Preserve original format if possible
        output_format = img.format or 'JPEG'
        
        # Initial quality and compression settings
        quality = 95
        output = BytesIO()
        
        # Progressive quality reduction with format-specific handling
        while quality > 20:  # Increased minimum quality threshold
            output.seek(0)
            output.truncate(0)
            
            if output_format == 'JPEG':
                img.save(output, format=output_format, quality=quality, optimize=True)
            elif output_format == 'PNG':
                img.save(output, format=output_format, optimize=True)
            else:
                # Default to JPEG for unsupported formats
                img.save(output, format='JPEG', quality=quality, optimize=True)
            
            if output.tell() <= max_size_bytes:
                break
            
            # More gradual quality reduction
            quality -= 5
        
        # If still too large, convert to JPEG as last resort
        if output.tell() > max_size_bytes and output_format != 'JPEG':
            output.seek(0)
            output.truncate(0)
            img.save(output, format='JPEG', quality=70, optimize=True)
        
        return output.getvalue()

    def _extract_text2(self, content: bytes, filename: str) -> str:
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
                            # Convert PDF page to image and use Vision API
                            img = page.to_image(resolution=300).original
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            img_content = img_byte_arr.getvalue()
                            
                            vision_image = vision.Image(content=img_content)
                            response = self.vision_client.text_detection(image=vision_image,image_context=self.image_context)
                            if response.text_annotations:
                                text_list = [text_annotation.description for text_annotation in response.text_annotations]
                                text += " ".join(text_list) + "\n"
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
            # Compress image if needed
            content = self._compress_image(content)
            vision_image = vision.Image(content=content)
            response = self.vision_client.document_text_detection(image=vision_image,image_context=self.image_context)
            
            if response.error.message:
                raise HTTPException(
                    status_code=500,
                    detail=f"Vision API Error: {response.error.message}"
                )
            
            if response.text_annotations:
                text = " ".join([text_annotation.description for text_annotation in response.text_annotations])
            else:
                text = ""
       
        return text