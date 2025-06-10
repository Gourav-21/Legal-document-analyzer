from fastapi import UploadFile, HTTPException
from PIL import Image
from openai import OpenAI
import os
from dotenv import load_dotenv
from io import BytesIO
from typing import List, Dict, Union
from docx import Document
from judgement import JudgementStorage
from labour_law import LaborLawStorage
from routers.letter_format_api import LetterFormatStorage
import io
import pandas as pd
from agentic_doc.parse import parse
import base64

# Load environment variables from .env file
load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.letter_format = LetterFormatStorage()
        self.law_storage = LaborLawStorage()
        self.judgement_storage = JudgementStorage()
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _extract_text2(self, content: bytes, filename: str, compress: bool = False) -> str:
        # Use AgenticDoc for PDFs and images, fallback for docx/xlsx
        try:
            if filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                # Pass bytes directly to AgenticDoc
                result = parse(content)
                # Return markdown or structured, as you prefer
                return result[0].markdown
            elif filename.lower().endswith('.docx'):
                doc_file = BytesIO(content)
                doc = Document(doc_file)
                return '\n'.join([p.text for p in doc.paragraphs])
            elif filename.lower().endswith('.xlsx'):
                excel_file = BytesIO(content)
                df = pd.read_excel(excel_file, sheet_name=None)
                text = ""
                for sheet, data in df.items():
                    text += f"Sheet: {sheet}\n"
                    text += data.to_string(index=False) + "\n\n"
                return text
            else:
                # Unknown file type, treat as plain text
                return content.decode('utf-8', errors='ignore')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")

    def process_document(self, files, doc_types, compress: bool = False):
        # No changes needed here, it still works (calls self._extract_text2 for each file)
        if not files:
            raise HTTPException(
                status_code=400,
                detail="At least one document must be provided"
            )
        
        if isinstance(files, UploadFile):
            files = [files]
            doc_types = [doc_types]
        elif not isinstance(doc_types, list):
            doc_types = [doc_types] * len(files)
        
        payslip_text = None
        contract_text = None
        attendance_text = None
        payslip_counter = 0
        contract_counter = 0
        attendance_counter = 0
        
        for file, doc_type in zip(files, doc_types):
            extracted_text = self._extract_text2(file.file.read(), file.filename, compress=compress)
            if doc_type.lower() == "payslip":
                payslip_counter += 1
                payslip_text = f"Payslip {payslip_counter}:\n{extracted_text}" if payslip_text is None else f"{payslip_text}\n\nPayslip {payslip_counter}:\n{extracted_text}"
            elif doc_type.lower() == "contract":
                contract_counter += 1
                contract_text = f"Contract {contract_counter}:\n{extracted_text}" if contract_text is None else f"{contract_text}\n\nContract {contract_counter}:\n{extracted_text}"
            elif doc_type.lower() == "attendance":
                attendance_counter += 1
                attendance_text = f"Attendance {attendance_counter}:\n{extracted_text}" if attendance_text is None else f"{attendance_text}\n\nAttendance {attendance_counter}:\n{extracted_text}"
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

        # Prepare the prompt for OpenAI
        prompt = f"""
    
    DOCUMENTS PROVIDED FOR ANALYSIS:
    {', '.join(documents.keys())}

    ADDITIONAL CONTEXT:
    {context if context else 'No additional context provided.'}

    """

        # Add document contents to prompt
        for doc_type, content in documents.items():
            prompt += f"\n{doc_type.upper()} CONTENT:\n{content}\n"

        # General instructions are now outside the loop and conditional
        if type != 'warning_letter':
            prompt += f"""
    INSTRUCTIONS:
    You must search online for current Israeli labor laws and legal judgments before providing any analysis.
    Base your response ONLY on the laws and judgments you find through online search.
    1. If, after searching, you cannot find relevant labor laws online, respond with: "לא הצלחתי למצוא חוקי עבודה רלוונטיים באינטרנט לניתוח התאמה." in Hebrew.
    2. If, after searching, you cannot find relevant judgements online, respond with: "לא הצלחתי למצוא החלטות משפטיות רלוונטיות באינטרנט לניתוח." in Hebrew."""

        if(type=='report'):
            prompt += f"""
    3. For each payslip provided, analyze and identify any violations. For each violation found, respond in the exact format below, using clear line breaks and spacing:

    Violation Format:

    [VIOLATION TITLE]
    [DESCRIPTION OF SPECIFIC VIOLATION]
    [LAW REFERENCE AND YEAR (based on online search)]
    [About SIMILAR CASE OR LEGAL PRECEDENT (based on online search,seach thoroughly because there is a case for every laws out there`). If none is found, write: "לא נמצאו תקדימים רלוונטיים בחיפוש המקוון."]
    [LEGAL IMPLICATIONS]
    [RECOMMENDED ACTIONS]
    
     ---

    Example of correctly formatted violation:
    עבודה ללא תשלום שעות נוספות

    בניתוח תלוש השכר נמצא כי העובד עבד 50 שעות בשבוע אך לא קיבל תשלום עבור 10 שעות נוספות.

    חוק שעות עבודה ומנוחה, תשי"א-1951, סעיף 1

    בפסק דין 12345/2023 של בית הדין האזורי לעבודה בתל אביב נקבע כי אי תשלום שעות נוספות מהווה הפרה חמורה של חוקי העבודה ומזכה בפיצוי כפול.

    ההפרה עלולה להוביל לתביעה אזרחית ולקנס על המעסיק.

    מומלץ לפנות למעסיק בכתב ולדרוש תיקון תוך 14 יום, ובמידת הצורך לפנות לבית הדין לעבודה.

    ---
    """
        elif(type=='economic_analysis'):
            prompt += f"""
    Analyze the provided documents for labor law violations based strictly on the Israeli labor laws you find online.

    For each violation identified, provide a detailed economic analysis including:

    1. For each violation:
       - Calculate monetary compensation based only on legal judgments found online.
       - Search for similar cases and their financial outcomes.

    2. For each successful precedent:
       - State the court case name and ruling details found online.
       - Calculate average compensation from similar cases found online.

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
    
    – סיכום:  
לאחר בחינה, העלות המרבית שניתן לתבוע היא [MAX_CLAIM] ₪, הוצאות מס מוערכות ב-[TAX_COST] ₪. במקרה של זכייה, הסכום נטו המשוער הוא [NET_AFTER_TAX] ₪.

    """
        
        elif(type=='professional'):
            prompt += f"""
    Analyze the provided documents for labor law violations based strictly on the Israeli labor laws you find online and the content of the documents. For each violation, calculate the monetary differences using only those laws.

    Provide your analysis in the following format, entirely in Hebrew:

ניתוח מקצועי של הפרות שכר:

הפרה: [כותרת ההפרה]
[תיאור מפורט של ההפרה, כולל תאריכים רלוונטיים, שעות עבודה, שכר שעתי וחישובים, בהתבסס אך ורק על החוקים הישראליים שנמצאו אונליין והמסמכים שסופקו.
דוגמה: העובד עבד X שעות נוספות בין [חודש שנה] ל-[חודש שנה]. לפי שכר שעתי בסיסי של [שכר] ₪ ושיעורי תשלום שעות נוספות ([שיעור1]% עבור X השעות הראשונות, [שיעור2]% לאחר מכן) כפי שמופיע בחוקי העבודה שנמצאו אונליין, המעסיק היה מחויב לשלם [סכום] ₪. מכיוון שלא שולם, החוב הוא [חוב] ₪.
דוגמה: בחודש [חודש שנה] לא בוצעה הפקדה לפנסיה. המעסיק מחויב להפקיד [אחוז]% מהשכר בגובה [שכר] ₪ = [סכום] ₪ בהתאם לחוק/צו הרחבה שנמצא אונליין.]
סה"כ חוב: [סכום ההפרש עבור הפרה זו] ₪

הפרה: [כותרת ההפרה]
[תיאור מפורט של ההפרה, כולל תאריכים וחישובים, בהתבסס אך ורק על החוקים שנמצאו אונליין והמסמכים. 
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
    4. directly return the letter only and do not include any other text or explanations.

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
🔒 מטרה: צור סיכום קצר וברור של ההפרות בתלושי השכר של העובד.
📌 כללים מחייבים:
	1. כתוב בעברית בלבד – אל תשתמש באנגלית בכלל.
	2. עבור כל חודש הצג את ההפרות בשורות נפרדות, כל שורה בפורמט הבא:
❌ [סוג ההפרה בקצרה] – [סכום בש"ח עם ₪, כולל פסיק לאלפים]
לדוגמה: ❌ לא שולם החזר נסיעות בפברואר 2025 – 250 ₪
	3. אם יש מספר רכיבי פנסיה (עובד/מעסיק/בריאות) בחודש מסוים – חבר אותם לסכום אחד של פנסיה באותו החודש.
	4. כל הסכומים יוצגו עם פסיקים לאלפים ועם ₪ בסוף.
	5. חישוב הסכום הכולל יופיע בשורה נפרדת:
💰 סה"כ: [סכום כולל] ₪
	6. הוסף המלצה בסוף:
📝 מה לעשות עכשיו:
פנה/י למעסיק עם דרישה לתשלום הסכומים.
אם אין מענה – מומלץ לפנות לייעוץ משפטי.

📍 הנחיות נוספות:
	• אין לכתוב מספרים בלי הקשר, כל שורה חייבת להיות מלווה בחודש.
	• מיזוג שורות: אם באותו חודש יש כמה רכיבים של פנסיה – מיזג אותם לשורה אחת.
	• הסר שורות ללא סכום ברור.
	• ניסוח פשוט, ללא מינוחים משפטיים, הבהרות או הסברים טכניים.
	• אין לציין "רכיב עובד", "רכיב מעסיק", "לא הופקד" – במקום זאת כתוב: "לא שולמה פנסיה".
🎯 פלט רצוי:
	• שורות מסודרות לפי חודשים
	• אין כפילויות
	• סכומים מדויקים בלבד
	• ניסוח ברור ומובן
	• עברית בלבד

🧪 Example of desired output:
📢 סיכום ההפרות:
❌ לא שולם עבור שעות נוספות בנובמבר 2024 – 495 ₪
❌ לא שולמה פנסיה בנובמבר 2024 – 750 ₪
❌ לא שולמה פנסיה בדצמבר 2024 – 1,221 ₪
❌ לא שולמה פנסיה בינואר 2025 – 831 ₪
❌ לא שולם החזר נסיעות בפברואר 2025 – 250 ₪
❌ לא שולמה פנסיה בפברואר 2025 – 858 ₪
❌ לא שולמה פנסיה במרץ 2025 – 866 ₪
💰 סה"כ: 5,271 ₪
📝 מה לעשות עכשיו:
פנה/י למעסיק עם דרישה לתשלום הסכומים.
אם אין מענה – מומלץ לפנות לייעוץ משפטי.
"""

        elif(type=='table'):
            prompt += f"""
אתה עוזר משפטי מיומן. עליך לנתח את רשימת ההפרות ולהפיק רשימת תביעות מסודרת לפי מסמך (לדוגמה: תלוש שכר מס' 1, מסמך שימוע, מכתב פיטורין וכו').

הנחיות:
1. סדר את התביעות לפי מסמך: כל קבוצה מתחילה בכותרת כמו "תלוש שכר מס' 4 – 05/2024:".

2. תחת כל כותרת, צור רשימה ממוספרת באותיות עבריות (א., ב., ג. וכו').

3. השתמש במבנה הקבוע הבא:
   א. סכום של [amount] ש"ח עבור [תיאור קצר של ההפרה].

4. השתמש בפורמט מספרים עם פסיקים לאלפים ושתי ספרות אחרי הנקודה (למשל: 1,618.75 ש"ח).

5. כתוב "ש"ח" אחרי הסכום — לא ₪.

6. אל תסכם את כלל ההפרות – הצג רק את הפירוט לפי מסמך.

7. אל תוסיף בולטים, טבלאות או הערות.

נתוני הקלט לדוגמה:
❌ אי תשלום שעות נוספות (תלוש 6, 11/2024) – 495 ש"ח
❌ העדר רכיב מעביד לפנסיה (תלוש 6, 11/2024) – 750 ש"ח
❌ העדר רכיב עובד לפנסיה (תלוש 7, 12/2024) – 396 ש"ח
❌ העדר נסיעות (תלוש 9, 02/2025) – 250 ש"ח

פלט נדרש לדוגמה:

תלוש שכר מס' 6 – 11/2024:
א. סכום של 495.00 ש"ח עבור אי תשלום שעות נוספות.
ב. סכום של 750.00 ש"ח עבור העדר רכיב מעביד לפנסיה.

תלוש שכר מס' 7 – 12/2024:
א. סכום של 396.00 ש"ח עבור העדר רכיב עובד לפנסיה.

תלוש שכר מס' 9 – 02/2025:
א. סכום של 250.00 ש"ח עבור העדר תשלום עבור נסיעות.

החזר את הפלט בפורמט זה בלבד, מופרד לפי כל מסמך.

"""
        
        elif(type == 'claim'):
            prompt += f"""
משימה:
כתוב טיוטת כתב תביעה לבית הדין האזורי לעבודה, בהתאם למבנה המשפטי הנהוג בישראל.

נתונים:
השתמש במידע מתוך המסמכים שצורפו (כגון תלושי שכר, הסכמי עבודה, הודעות פיטורים, שיחות עם המעסיק) ובממצאים שנמצאו בניתוח קודם של ההפרות.

פורמט לכתיבה:
1. כותרת: "בית הדין האזורי לעבודה ב[שם עיר]"
2. פרטי התובע/ת:
   - שם מלא, ת"ז, כתובת, טלפון, מייל
3. פרטי הנתבע/ת (מעסיק):
   - שם החברה/מעסיק, ח.פ./ת"ז, כתובת, טלפון, מייל (אם קיים)
4. כותרת: "כתב תביעה"
5. פתיח משפטי:
   בית הדין הנכבד מתבקש לזמן את הנתבע/ת לדין ולחייבו/ה לשלם לתובע/ת את הסכומים המפורטים, מהנימוקים הבאים:
6. סעיף 1 – רקע עובדתי:
   תיאור תקופת העבודה, תפקידים, תאריך תחילה וסיום (אם רלוונטי), מהות יחסי העבודה, מקום העבודה.
7. סעיף 2 – עילות התביעה (לפי הפרות):
   לכל הפרה:
     - איזה חוק הופֵר (לדוג': חוק שכר מינימום, חוק שעות עבודה ומנוחה וכו')
     - פירוט העובדות והתקופה הרלוונטית
     - סכום הפיצוי או הנזק
     - אסמכתאות משפטיות אם רלוונטי
8. סעיף 3 – נזקים שנגרמו:
   סכומים כספיים (בפירוט), נזק לא ממוני אם קיים (עוגמת נפש).
9. סעיף 4 – סעדים מבוקשים:
   תשלום הפרשים, פיצויים, ריבית והצמדה, הוצאות משפט, שכר טרחת עו"ד, וכל סעד אחר.
10. סעיף 5 – סמכות שיפוט:
   ציין סמכות בית הדין לעבודה לפי חוק בית הדין לעבודה תשכ"ט–1969.
11. סעיף 6 – דיון מקדים/הליך מהיר (אם רלוונטי).
12. סעיף 7 – כתובות להמצאת כתבי בי-דין:
   כתובת התובע והנתבע.
13. סיום:
   חתימה, תאריך, רשימת נספחים תומכים (תלושי שכר, מכתבים וכו').

שפה:
- כתוב בעברית משפטית, רשמית ומסודרת.
- סדר את התביעה עם כותרות, סעיפים ממוספרים ורווחים ברורים.
- אל תמציא עובדות או חוקים. אם חסר מידע – השאר שדה ריק לצורך השלמה ע"י המשתמש.

אזהרה:
בסיום, כתוב משפט: "הטיוטה נכתבה אוטומטית לצורך עזרה ראשונית בלבד. מומלץ לפנות לעורך/ת דין מוסמך/ת לפני הגשה לבית הדין."

"""

        # Add type-specific prompts for other conditions if needed
        if type not in ['warning_letter', 'easy', 'table']:
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
            # Using OpenAI with web search capabilities
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Use GPT-4 with web browsing if available, otherwise use regular GPT-4
            # response = self.client.chat.completions.create(
            #     model="gpt-4.1",  # or "gpt-4-1106-preview" if you have access
            #     # tools=[{"type": "web_search_preview"}],  # Enable web search
            #     # messages=messages,
            #     messages=[{"role": "user", "content": "who is the cm of odisha as of today?"},],
            #     temperature=0,
            #     # max_tokens=4000
            # )
            
            # analysis = response.choices[0].message.content
            
            res2=self.client.responses.create(
                model="gpt-4.1",
                input=messages,
                # temperature=0,
                # tools=[{"type": "web_search_preview"}],  # Enable web search
                # max_tokens=1000
            )
            
            # Structure the result
            result = {
                "legal_analysis": res2.output_text,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            error_detail = f"Error generating legal analysis with OpenAI: {str(e)}"
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

    def summarise(self, ai_content_text: str) -> str:
        """
        Summarizes the given text using OpenAI GPT-4.
        """
        try:
            prompt = f"Please summarize the following text concisely in Hebrew:\n\n{ai_content_text}"
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides concise summaries in Hebrew."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a faster model for summarization
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            summary = response.choices[0].message.content
            return summary
            
        except Exception as e:
            error_detail = f"Error generating summary with OpenAI: {str(e)}"
            raise HTTPException(
                status_code=500,
                detail=error_detail
            )
