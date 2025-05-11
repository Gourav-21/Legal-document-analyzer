from fastapi import UploadFile, HTTPException
from PIL import Image
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import pdfplumber
from io import BytesIO
from typing import List, Dict, Union
from docx import Document
from judgement import JudgementStorage
from labour_law import LaborLawStorage
from routers.letter_format_api import LetterFormatStorage
from google.cloud import vision
import io
from google.cloud.vision import ImageAnnotatorClient
import pandas as pd

# Load environment variables from .env file
load_dotenv()

class DocumentProcessor:
    def __init__(self):
        # Initialize Vision client with API key
        vision_api_key = os.getenv("GOOGLE_CLOUD_VISION_API_KEY")
        if not vision_api_key:
            raise Exception("Google Cloud Vision API key not found. Please set GOOGLE_CLOUD_VISION_in your .env filevariable.")
        
        self.vision_client = ImageAnnotatorClient(client_options={"api_key": vision_api_key})
        self.image_context = {"language_hints": ["he"]} 
        self.letter_format = LetterFormatStorage()
        self.law_storage = LaborLawStorage()
        self.judgement_storage = JudgementStorage()

    def process_document(self, files: Union[UploadFile, List[UploadFile]], doc_types: Union[str, List[str]], compress: bool = False) -> Dict[str, str]:
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
        attendance_text = None
        
        # Initialize payslip counter
        payslip_counter = 0
        # Process each file based on its type
        for file, doc_type in zip(files, doc_types):
            extracted_text = self._extract_text2(file.file.read(), file.filename, compress=compress)
            if doc_type.lower() == "payslip":
                payslip_counter += 1
                payslip_text = f"Payslip {payslip_counter}:\n{extracted_text}" if payslip_text is None else f"{payslip_text}\n\nPayslip {payslip_counter}:\n{extracted_text}"
            elif doc_type.lower() == "contract":
                contract_text = extracted_text
            elif doc_type.lower() == "attendance":
                attendance_text = extracted_text
        # Return the extracted texts
        return {
            "payslip_text": payslip_text,
            "contract_text": contract_text,
            "attendance_text": attendance_text
        }
    
    def create_report(self, payslip_text: str = None, contract_text: str = None, attendance_text: str = None, type: str = "report", context: str = None) -> Dict:
        # Prepare documents for analysis
        documents = {}
        if payslip_text:
            documents['payslip'] = payslip_text
        if contract_text:
            documents['contract'] = contract_text
        if attendance_text:
            documents['attendance report'] = attendance_text
            
        system_prompt = """You are an expert Israeli labour law attorney specializing in labor law compliance.
You have access to the internet and can search for relevant Israeli labor laws and legal judgments.
Respond in Hebrew. Think step-by-step and clearly explain your reasoning for any conclusions you draw.
When using online search, explicitly state what you searched for and how the results (or lack thereof) influenced your analysis.
Include links to all relevant laws, legal sources, and court judgments you reference.
Follow the required response format exactly and apply relevant laws and legal precedents as needed.
Do not include any disclaimers or advice to consult a lawyer; the user understands the nature of the information provided."""


        # Prepare the prompt for Gemini AI
        prompt = f"""
    
    DOCUMENTS PROVIDED FOR ANALYSIS:
    {', '.join(documents.keys())}

    ADDITIONAL CONTEXT:
    {context if context else 'No additional context provided.'}

    """

        # Add document contents to prompt
        for doc_type, content in documents.items():
            prompt += f"\n{doc_type.upper()} CONTENT:\n{content}\n"

            prompt += f"""
    INSTRUCTIONS:
    1. If, after searching, you cannot find relevant labor laws online, respond with: "לא הצלחתי למצוא חוקי עבודה רלוונטיים באינטרנט לניתוח התאמה." in Hebrew.
    2. If, after searching, you cannot find relevant judgements online, respond with: "לא הצלחתי למצוא החלטות משפטיות רלוונטיות באינטרנט לניתוח." in Hebrew."""

        if(type=='report'):
            prompt += f"""
    3. For each payslip provided, analyze and identify any violations. For each violation found, respond in the exact format below, using clear line breaks and spacing:

    Violation Format:

    [VIOLATION TITLE]
    [DESCRIPTION OF SPECIFIC VIOLATION]
    [LAW REFERENCE AND YEAR (based on online search)]
    [SIMILAR CASE OR LEGAL PRECEDENT (based on online search). If none is found, write: "לא נמצאו תקדימים רלוונטיים בחיפוש המקוון."]
    [LEGAL IMPLICATIONS]
    [RECOMMENDED ACTIONS]
    
     ---

    Example of correctly formatted violation:

    Payslip 1:

    Possible Violation of Mandatory Pension Expansion Order

    The payslip shows no pension contribution, despite over 6 months of employment.

    According to the Mandatory Pension Expansion Order (2008) (found through online search).

    In the [NAME OF RULING FOUND ONLINE] ruling, there was a similar case and the employee won 10,000 thousand shekels there (based on online search results).

    Lack of contribution may entitle the employee to retroactive compensation or legal action.

    It is recommended to review the pension fund details, start date of employment, and contract terms.

    ---
    """
        
        elif type == "profitability":
            prompt += f"""
    INSTRUCTIONS:
    2. For each violation, search for similar legal cases in Israeli judgements found online and summarize both successful and unsuccessful outcomes.
    3. For unsuccessful precedents:
       - Explain why the claims were rejected, based only on the judgement information found online.
       - Clearly recommend against pursuing legal action, if applicable.
       - List the risks and potential costs involved.

    4. For successful precedents:
       - Calculate the average compensation awarded (based on judgements found online).
       - Estimate legal fees (30% of potential compensation).
       - Estimate tax deductions (25% of net compensation).
       - Include a time and effort cost estimate if available.

    Use the following format for your analysis:

    ניתוח כדאיות כלכלית:

    הפרות שזוהו (על בסיס החוקים שנמצאו אונליין):
    [List identified violations]

    תקדימים משפטיים (מתוך פסקי הדין שנמצאו אונליין):
    [Summarize similar cases and their outcomes]

    במקרה של תקדימים שליליים:
    - סיבות לדחיית התביעות: [Reasoning based on online judgements]
    - סיכונים אפשריים: [List risks]
    - המלצה: לא מומלץ להגיש תביעה בשל [Explanation based on judgement analysis]

    במקרה של תקדימים חיוביים:
    ניתוח כספי:
    - סכום פיצוי ממוצע (מפסיקות דין שנמצאו אונליין): [AMOUNT] ₪
    - עלות משוערת של עורך דין (30%): [AMOUNT] ₪
    - השלכות מס (25% מהסכום נטו): [AMOUNT] ₪
    - סכום נטו משוער: [AMOUNT] ₪

    המלצה סופית:
    [Final recommendation based on analysis of judgements found online]
    """
        
        elif(type=='professional'):
            prompt += f"""
    Analyze the provided documents for labor law violations based strictly on the Israeli labor laws you find online and the content of the documents. For each violation, calculate the monetary differences using only those laws.
    Provide your analysis in the following format, entirely in Hebrew:

ניתוח מקצועי של הפרות שכר:

הפרה: [כותרת ההפרה]
[תיאור מפורט של ההפרה, כולל תאריכים רלוונטיים, שעות עבודה, שכר שעתי וחישובים, בהתבסס אך ורק על החוקים הישראליים שנמצאו אונליין והמסמכים שסופקו.
דוגמה: העובד עבד X שעות נוספות בין [חודש שנה] ל-[חודש שנה]. לפי שכר שעתי בסיסי של [שכר] ₪ ושיעורי תשלום שעות נוספות ([שיעור1]% עבור X השעות הראשונות, [שיעור2]% לאחר מכן) כפי שמופיע בחוקי העבודה שנמצאו אונליין, העובד היה זכאי ל-[סכום] ₪ לחודש. בפועל קיבל רק [סכום שקיבל] ₪ למשך X חודשים ו-[סכום] ₪ בחודש [חודש].]
סה"כ חוב: [סכום ההפרש עבור הפרה זו] ₪

הפרה: [כותרת ההפרה]
[תיאור מפורט של ההפרה, כולל תאריכים וחישובים, בהתבסס אך ורק על החוקים שנמצאו אונליין והמסמכים. 
דוגמה: בחודש [חודש שנה] לא בוצעה הפקדה לפנסיה. המעסיק מחויב להפקיד [אחוז]% מהשכר בגובה [שכר] ₪ = [סכום] ₪ בהתאם לחוק/צו הרחבה שנמצא אונליין.]
סה"כ חוב פנסיה: [סכום חוב הפנסיה להפרה זו] ₪

---

סה"כ תביעה משפטית (לא כולל ריבית): [הסכום הכולל לתביעה מכלל ההפרות] ₪  
אסמכתאות משפטיות: [רשימת שמות החוק הרלוונטיים מתוך החוקים הישראליים שנמצאו אונליין. לדוגמה: חוק שעות עבודה ומנוחה, צו הרחבה לפנסיה חובה]

"""


        elif(type=='warning_letter'):
            format_content = self.letter_format.get_format().get('content', '')
            prompt += f"""
    INSTRUCTIONS:
    1. Analyze the provided documents for labor law violations *based exclusively on the Israeli LABOR LAWS and JUDGEMENTS you find online*.
    2. If violations are found, generate a formal warning letter using the provided template.
    3. If no violations are found, respond with: "לא נמצאו הפרות המצדיקות מכתב התראה." in Hebrew.

    Warning Letter Template:
    {format_content}

    Please generate the warning letter in Hebrew with the following guidelines:
    - Replace [EMPLOYER_NAME] with the employer's name from the documents
    - Replace [VIOLATION_DETAILS] with specific details of each violation found (based on laws found online)
    - Replace [LAW_REFERENCES] with relevant labor law citations *from the Israeli labor laws found online*.
    - Replace [REQUIRED_ACTIONS] with clear corrective actions needed
    - Replace [DEADLINE] with a reasonable timeframe for corrections (typically 14 days)
    - Maintain a professional and formal tone throughout
    - Include all violations found in the analysis (based on laws found online)
    - Format the letter according to the provided template structure
    """

        elif(type=='easy'):
            prompt += f"""
 Instructions:

1. Language:
   Respond only in Hebrew.

2. Tone:
   Use clear, friendly, and professional language. The tone should feel like a legal assistant helping a non-lawyer understand their rights — simple, practical, and respectful.

3. Formatting Rules:
   - Present findings in short, well-structured sections.
   - Each main section title (e.g., for a document type or a specific payslip) and each sub-point title (e.g., "What the law says") MUST start with an emoji.
   - **Crucial for Readability**: After any line that starts with an emoji and serves as a title/heading (like "📜 Contract Analysis" or "📌 What the law says"), ALWAYS add a newline character before the detailed text or the next heading begins. This ensures proper spacing.
   - If a violation or legal exception is found, explain it using the following sub-points. Each sub-point must adhere to the emoji-heading-newline rule:
     * 📌 What the law says
     * 📉 What was found in the employee’s case
     * ⚠️ Whether it’s a legal violation and what the employee can claim
     * 🗓 Any relevant deadlines or retroactive rights
     * 💰 If relevant, add a clear monetary calculation
   - Separate distinct violations or major findings with "---".
   - Clearly indicate which payslip or document is being analyzed (e.g., "ניתוח תלוש מספר 1:", "ניתוח חוזה עבודה:").

4. Use these as format guides only (not exact content):

Example 1 – Unemployment:

📅 לפי הנתונים שהוזנו:
תאריך התפטרות: 15.09.2023
תאריך התייצבות ראשון: 10.01.2024
הגיל שלך: 32
ותק תעסוקתי: מעל 24 חודשים

🔎 לפי החוק, את/ה זכאי/ת ל־100 ימי אבטלה.
נכון לעכשיו התייצבת 63 פעמים, ולכן נותרו לך 37 ימי זכאות.

⚠️ טרם הוגשה תביעה לביטוח הלאומי.
כדי לא להפסיד כסף – מומלץ להגיש את התביעה עד 09.01.2025 (אפשר רטרואקטיבית עד שנה אחורה).

Example 2 – Overtime Violation:

⏱ שעות עבודה ביום: 10
סוג שכר: שכר חודשי רגיל (ללא שעות נוספות מוגדרות)

📌 לפי חוק שעות עבודה ומנוחה:
את/ה עובד/ת שעתיים נוספות ביום × 5 ימים = 10 שעות נוספות בשבוע.
התשלום המינימלי עבורן צריך להיות כ־1,334 ₪ בחודש (בהנחה שהשכר שלך הוא 7,000 ₪).

⚠️ נמצאה הפרה של החוק.
את/ה זכאי/ת לדרוש תשלום רטרואקטיבי על שעות נוספות.
ייתכן שמדובר בהפרה מתמשכת שמזכה בפיצוי נוסף.

Example 3 – Vacation Redemption:

🗓 ותק בעבודה: שנה וחצי
ימי עבודה בשבוע: 5
משרה מלאה: כן
מספר ימי חופשה שנוצלו בפועל: 0
שובר פדיון חופשה: לא הופיע

📌 לפי החוק, את/ה זכאי/ת ל־14 ימי חופשה (שנה ראשונה: 12, חצי שנה שנייה: 2).

📉 נמצאה חריגה:
נוצלו 0 ימים, ולכן המעסיק מחויב לפדות את מלוא הזכאות בכסף.
זו הפרה של חוק חופשה שנתית, והעובד/ת רשאי/ת לדרוש תשלום מיידי.

5. If no violations are found:
   For each payslip with no issues, respond exactly with:
   "לא נמצאו הפרות בתלוש מספר [X]"

If no violations are found in any payslip or document:
"לא נמצאו הפרות נגד חוקי העבודה הרלוונטיים שנמצאו בחיפוש המקוון."

6. Legal Requirements:

* Match calculations to the laws or rulings found.
* Do not add summaries or introductions.
* Output should be final and ready to show the end user.
  """

        elif(type=='table'):
            prompt += f"""
            [
                {{
                    "month": str,
                    "violation_type": str,
                    "legal_entitlement": str,  
                    "actual_payment": str,
                    "difference": str,
                    "legal_explanation": str
                }}
            ]
            Analyze the documents and return a structured array of violations matching this format. Each month should have its own entry.
            Only include months where violations occurred. Include citations to relevant Israeli labor laws found in your search.
            Return the array as a string that can be parsed as JSON.
            do not include any other text or explanations. directly return the array.
            Example response:
            '[{{"month":"January 2023","violation_type":"Unpaid Overtime","legal_entitlement":"₪1,500","actual_payment":"₪0","difference":"₪1,500","legal_explanation":"According to סעיף 16 לחוק שעות עבודה ומנוחה"}},{{"month":"February 2023","violation_type":"Vacation Pay","legal_entitlement":"₪2,000","actual_payment":"₪1,000","difference":"₪1,000","legal_explanation":"Based on סעיף 9 לחוק חופשה שנתית"}}]'
            """
        
        prompt += f"""
    IMPORTANT:
    - עבור כל הפרה, הצג חישובים ברורים ככל שניתן.
    - חשב סכום כולל עבור כל הפרה בנפרד.
    - חשב את סך סכום התביעה על ידי חיבור כל ההפרות.
    - ציין בסוף את שמות החוקים הרלוונטיים ששימשו לניתוח.
    - Do not guess. Respond only with data that is verifiable through online sources.
    - Format each violation with proper spacing and line breaks as shown above
    - Analyze each payslip separately and clearly indicate which payslip the violations belong to
    - Separate multiple violations with '---'
    - If no violations are found against the relevant laws in a payslip, respond with: "לא נמצאו הפרות בתלוש מספר [X]" in hebrew
    - If no violations are found in any payslip, respond with: "לא נמצאו הפרות נגד חוקי העבודה הרלוונטיים שנמצאו." in hebrew
    """

        try:
            gemini_api_key = os.environ.get("GOOGLE_CLOUD_VISION_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables.")

            client = genai.Client(api_key=gemini_api_key)
            model_name = "gemini-2.5-pro-preview-05-06"
            
            api_contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            tools_config = [
                types.Tool(google_search=types.GoogleSearch()),
            ]
            
            gen_config = types.GenerateContentConfig(
                temperature=0,
                tools=tools_config,
                response_mime_type="text/plain",
                system_instruction=[
                     types.Part.from_text(text=system_prompt),
                ],
            )

            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=api_contents,
                config=gen_config, 
            )
            
            analysis_parts = []
            for chunk in response_stream:
                if chunk.text: # Ensure text exists before appending
                    analysis_parts.append(chunk.text)
            analysis = "".join(analysis_parts)
            
            if(type == 'table'):
                try:
                    # Find start and end of JSON array in response
                    start_idx = analysis.find('[')
                    end_idx = analysis.rfind(']') + 1
                    
                    if start_idx == -1 or end_idx == 0:
                        # No valid JSON array found
                        return {"legal_analysis": "No violations found", "status": "success"}
                        
                    # Extract just the JSON array portion
                    json_str = analysis[start_idx:end_idx]
                    return {"legal_analysis": json_str, "status": "success"}

                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error parsing table data: {str(e)}"
                    )
            
            # Structure the result
            result = {
                "legal_analysis": analysis,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            # Log the full error for debugging if possible, or at least the type and message
            error_detail = f"Error generating legal analysis with new API: {type(e).__name__} - {str(e)}"
            # Check if the error is from genai and has more specific details
            if hasattr(e, 'message'): # For google.api_core.exceptions.GoogleAPIError
                 error_detail = f"Error generating legal analysis with new API: {e.message}"

            raise HTTPException(
                status_code=500,
                detail=error_detail
            )

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
                img.save(output, format='JPEG', quality=70, optimize=True)
            
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

    def _extract_text2(self, content: bytes, filename: str, compress: bool = False) -> str:
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
                            response = self.vision_client.text_detection(image=vision_image, image_context=self.image_context)
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
        elif filename.lower().endswith('.xlsx'):
            try:
                # Load Excel file into a pandas DataFrame
                excel_file = BytesIO(content)
                df = pd.read_excel(excel_file, sheet_name=None)  # Load all sheets
                
                # Extract text from all sheets
                text = ""
                for sheet_name, sheet_data in df.items():
                    text += f"Sheet: {sheet_name}\n"
                    text += sheet_data.to_string(index=False) + "\n\n"
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing Excel file: {str(e)}"
                )
        else:
            # Compress image if needed
            if compress:
                content = self._compress_image(content)
            vision_image = vision.Image(content=content)
            response = self.vision_client.document_text_detection(image=vision_image, image_context=self.image_context)
            
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