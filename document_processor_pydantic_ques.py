import sqlite_fix  # This must be imported first

# Apply SQLite3 fix before any other imports that might use ChromaDB
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    print("✅ SQLite3 fix applied in document_processor")
except ImportError:
    print("⚠️ pysqlite3-binary not available in document_processor")

from fastapi import UploadFile, HTTPException
from PIL import Image
import os
from dotenv import load_dotenv
import pdfplumber
from io import BytesIO
from typing import List, Dict, Union, Optional
from docx import Document
from google.cloud import vision
import io
from google.cloud.vision import ImageAnnotatorClient
import pandas as pd
from letter_format import LetterFormatStorage
from rag_storage import RAGLegalStorage
# from rag_storage_local import RAGLegalStorage
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.settings import ModelSettings
from pydantic import BaseModel
import asyncio
from agentic_doc.parse import parse
from judgement import JudgementStorage
from labour_law import LaborLawStorage
# nest_asyncio imported conditionally inside __init__ to avoid uvloop conflicts
# Load environment variables from .env 
load_dotenv()



class DocumentAnalysisRequest(BaseModel):
    """Request model for document analysis"""
    payslip_text: Optional[str] = None
    contract_text: Optional[str] = None
    attendance_text: Optional[str] = None
    analysis_type: str = "report"
    context: Optional[str] = None

class DocumentAnalysisResponse(BaseModel):
    """Response model for document analysis"""
    legal_analysis: str
    # relevant_laws: List[Dict]
    # relevant_judgements: List[Dict]
    status: str
    analysis_type: str

def get_error_detail(e):
    try:
        # Try accessing the structured body if it's JSON-like
        if hasattr(e, "body"):
            import json
            body = e.body
            # If it's a string, parse it
            if isinstance(body, str):
                body = json.loads(body)
            return body.get("error", {}).get("message", str(e))

        # Fallback to using .message
        if hasattr(e, "message"):
            return str(e.message)

        return str(e)

    except Exception:
        return repr(e)


class DocumentProcessor:
    def export_to_excel(self, processed_result: dict) -> bytes:
        """
        Process the processed_result, use AI agent to extract employee name, overtime hours, salary, and all relevant data, and generate an Excel file for download.
        Returns the Excel file as bytes.
        """
        import pandas as pd
        import io
        import asyncio

        # Compose a prompt for the AI agent to extract structured data
        prompt = f"""
        Extract a table of all employees from the following legal document analysis results. For each employee, extract the following fields (if available):
        - שם העובד (Employee Name)
        - חודש עבודה (Work Month)
        - שנה (Year)
        - סה"כ ימי עבודה (Total Work Days)
        - סה"כ שעות עבודה (Total Work Hours)
        - תעריף שעתי (Hourly Rate)
        - משכורת 100 % (100% Salary)
        - ש.נ. 125% (Overtime 125%)
        - ש.נ. 150% (Overtime 150%)
        - ש.נ. 175% (Overtime 175%)
        - ש.נ. 200% (Overtime 200%)
        - שעות שבת (Shabbat Hours)
        - שעות חג (Holiday Hours)
        - חופשה (Vacation)
        - ימי מחלה (Sick Days)
        - הבראה (Convalescence)
        - נסיעות (Travel)
        - הפרשות מעסיק לפנסיה (Employer Pension Contributions)
        - הפרשות מעסיק קרן השתלמות (Employer Study Fund Contributions)
        - ניכוי עובד גמל (Employee Pension Deduction)
        - ניכוי עובד קרן התשלמות (Employee Study Fund Deduction)
        - תוספת וותק (Seniority Bonus)
        - ש.נ. גלובלי (Global Overtime)
        - כמות ש.נ גלובלי (Global Overtime Amount)
        - עמלה (Commission)
        - בונוס (Bonus)
        - עמלה (Commission)

        Return the result as a markdown table with columns for all the above fields (in the order listed). If there are multiple payslips or employees, include all rows.

        Payslip Text:
        {processed_result.get('payslip_text', '')}

        Contract Text:
        {processed_result.get('contract_text', '')}

        Attendance Text:
        {processed_result.get('attendance_text', '')}
        """

        # Use the agent to extract the table
        def get_table_sync():
            import asyncio
            try:
                # Try to get the current event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No event loop in this thread (Streamlit), create and set one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                is_uvloop = 'uvloop' in str(type(loop))
                # If loop is running, patch with nest_asyncio only if not uvloop
                if loop.is_running():
                    if not is_uvloop:
                        try:
                            import nest_asyncio
                            nest_asyncio.apply()
                        except Exception:
                            pass
                    # Use run_until_complete via a new task if possible
                    coro = self.agent.run(prompt, model_settings=ModelSettings(temperature=0.0))
                    task = asyncio.ensure_future(coro)
                    return loop.run_until_complete(task)
                else:
                    # Loop is not running, safe to use run_until_complete
                    return loop.run_until_complete(self.agent.run(prompt, model_settings=ModelSettings(temperature=0.0)))
            except Exception as e:
                raise e

        result = get_table_sync()
        print(result)
        table_md = result.data if hasattr(result, 'data') else str(result)

        # Parse the markdown table to a DataFrame
        import re
        from io import StringIO
        def markdown_table_to_df(md):
            # Find the first markdown table in the string
            lines = md.splitlines()
            table_lines = []
            in_table = False
            for line in lines:
                if '|' in line:
                    in_table = True
                    table_lines.append(line)
                elif in_table and line.strip() == '':
                    break
            if not table_lines:
                raise ValueError("No markdown table found in AI output.")
            # Remove markdown separator lines (e.g., | :--- | ... |)
            cleaned_lines = []
            for i, line in enumerate(table_lines):
                # Remove lines where all cells are dashes or colons (markdown separator)
                cells = [c.strip() for c in line.strip().split('|') if c.strip()]
                if all(re.match(r'^:?[-]+:?$', c) for c in cells):
                    continue
                cleaned_lines.append(line)
            table_str = '\n'.join(cleaned_lines)
            # Use pandas to read the markdown table
            try:
                import pandas as pd
                from io import StringIO
                df = pd.read_csv(StringIO(table_str), sep='|', engine='python')
                # Remove unnamed columns and whitespace
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                df.columns = [c.strip() for c in df.columns]
                # Use DataFrame.map for element-wise string strip (applymap is deprecated)
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
                return df
            except Exception as e:
                raise ValueError(f"Failed to parse markdown table: {e}\nTable string:\n{table_str}")

        df = markdown_table_to_df(table_md)

        # Write DataFrame to Excel in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='EmployeeData')
        output.seek(0)
        return output.read()
    
    def qna_sync(self, report: str, questions: str) -> str:
        """Synchronous wrapper for the async qna method."""
        import asyncio
        try:
            return asyncio.run(self.qna(report, questions))
        except RuntimeError as e:
            # If there's already a running event loop (e.g. in Streamlit), use alternative
            try:
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.qna(report, questions))
            except Exception as inner_e:
                raise RuntimeError(f"Failed to run qna async: {inner_e}") from e
            
    def __init__(self):
        # Initialize RAG storage
        self.rag_storage = RAGLegalStorage()
        
        # Initialize backward compatibility storages
        self.letter_format = LetterFormatStorage()
        # self.law_storage = LaborLawStorage()
        # self.judgement_storage = JudgementStorage()
        self.law_storage = self.rag_storage
        self.judgement_storage = self.rag_storage
        
        # Initialize PydanticAI model - use Gemini only
        gemini_api_key = os.getenv("GOOGLE_CLOUD_VISION_API_KEY")
        # Use ModelSettings to set temperature to 0.0 (deterministic output)
        model_settings = ModelSettings(temperature=0.0)
        if gemini_api_key:
            self.model = GeminiModel('gemini-2.5-pro', api_key=gemini_api_key)
            self.model_type = "gemini"
        else:
            raise Exception("GEMINI_API_KEY must be set in environment variables")
        # Initialize PydanticAI Agent
        # Configure Vision API
        vision_api_key = os.getenv("GOOGLE_CLOUD_VISION_API_KEY")
        if not vision_api_key:
            raise Exception("Google Cloud Vision API key not found. Please set GOOGLE_CLOUD_VISION_API_KEY in your .env file")
        
        self.vision_client = ImageAnnotatorClient(client_options={"api_key": vision_api_key})
        self.image_context = {"language_hints": ["he"]} 
        # Only apply nest_asyncio if not running under uvloop (e.g., not under uvicorn)
        _can_patch = True
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if 'uvloop' in str(type(loop)):
                _can_patch = False
        except Exception:
            pass
        if _can_patch and os.environ.get("USE_NEST_ASYNCIO", "0") == "1":
            import nest_asyncio
            nest_asyncio.apply()
        self.agent = Agent(
            model=self.model,
            result_type=str,
            system_prompt="""You are an expert legal document analyzer specializing in Israeli labor law compliance.

You will be provided with:
1. Relevant labor laws retrieved from a legal database
2. Relevant legal judgements and precedents retrieved from a legal database  
3. Document content to analyze (payslips, contracts, attendance records)
4. Additional context if provided

Your analysis must be based STRICTLY on the provided laws and judgements. Do not use external legal knowledge.

🚫 VERY IMPORTANT:
- Read the document carefully and remember its contents like wage per hour, working hours, etc. and Do not recalculate data like wages per hour, sick days, etc. unless the document provides exact values.
- Do not infer or estimate violations without clear proof in the payslip.
- Use **only** the documents provided (e.g., payslip data, employment contracts data, and attendance records data). **Do not extract or reuse any example values (e.g., 6000 ₪, 186 hours, 14 hours overtime) that appear in the legal texts or examples.**
- Do **not invent** missing data. If the document does not include sufficient detail for a violation (e.g., no overtime hours), **do not report a violation**.
- Do not hallucinate sick days, overtime hours, or absences
- think step by step and analyze the documents carefully. do not rush to conclusions.
- while calculating read the whole law and dont miss anything and explain the calculations step by step and how you arrived at the final amounts.

Always respond in Hebrew and follow the specific formatting requirements for each analysis type.

CRITICAL: When providing analysis, do NOT output template text or placeholders. Always replace ALL placeholders with real data from the analysis.""",
        )

        # Only apply nest_asyncio for question_agent if not running under uvloop
        if _can_patch and os.environ.get("USE_NEST_ASYNCIO", "0") == "1":
            import nest_asyncio
            nest_asyncio.apply()
        self.question_agent = Agent(
            model=self.model,
            result_type=List[str],
            system_prompt="""You are an expert legal document analyzer specializing in Israeli labor law compliance.
                            Your task is to analyze document content and generate specific search queries to find relevant and most applicable laws and judgements.
                            break those questions into sub-questions that can be used to search for laws and judgements in the database.
                            Return a list of 3-5 specific search queries that focus on the key legal concepts present in the documents.
            """
        )

        @self.question_agent.system_prompt
        def dynamic_system_prompt():
            """Dynamic system prompt for question agent"""
            # Get all law summaries from RAG storage
            law_summaries = self.rag_storage.get_all_law_summaries()
            if law_summaries:
                print(f"Found {len(law_summaries)} law summaries in the database.")
                summaries_text = "\n".join([f"- {summary}" for summary in law_summaries])
                return f"""
Here is a list of summaries of laws available in the database that you can search for:

{summaries_text}

Based on the document content provided, generate 3-5 specific search queries that will help find the most relevant laws and judgements. Focus on key legal concepts like:
- Salary and wage-related issues
- Working hours and overtime
- Pension and social benefits
- Employment contracts and termination
- Worker rights violations

Return only the search queries as a list of strings.
"""
            else:
                print("no law summaries found in the database.")
                return """
You are an expert legal document analyzer specializing in Israeli labor law compliance.

No law summaries are currently available in the database. Analyze the document content and generate 3-5 specific search queries based on the general legal concepts present. Focus on key areas like wages, working hours, benefits, contracts, and worker rights.

Return only the search queries as a list of strings.
"""

        # --- Review Agent ---
        if _can_patch and os.environ.get("USE_NEST_ASYNCIO", "0") == "1":
            import nest_asyncio
            nest_asyncio.apply()
        self.review_agent = Agent(
            model=self.model,
            result_type=str,
            system_prompt="""
You are a legal analysis review agent specializing in Israeli labor law.
You are given:
1. A set of relevant labor laws (as text)
2. A set of relevant legal judgements (as text)
3. An analysis of a legal case (as text)
4. A set of relevant employee documents (as text)
5. Additional context or information (as text)
Your job is to:
  - Carefully check if the analysis is correct, complete, and strictly based on the provided laws and judgements.
  - Carefully check the documents and laws and judgements and see if the analysis is missing any important legal points or if it contains any errors or is missing any violation.
  - Carefully check if the analysis is based only on the provided laws and judgements and documents and not assuming any external knowledge or making up facts.
  - You must also carefully check that all calculations (amounts, sums, percentages, totals, etc.) are correct and match the provided laws, judgements, and document data. If you find any calculation errors, you must correct them and explain the correction.
  - If the analysis is correct, return it as-is.
  - If the analysis is incorrect, incomplete, or not strictly based on the provided laws and judgements, or if any calculation is wrong, generate a corrected analysis that is fully compliant and mathematically accurate.
  
  🚫 VERY IMPORTANT:
- Do not recalculate data like wages per hour, overtime hours, sick days, etc. unless the document provides exact values.
- Do not infer or estimate violations without clear proof in the payslip.
- Use **only** the documents provided (e.g., payslip data, employment contracts data, and attendance records data). **Do not extract or reuse any example values (e.g., 6000 ₪, 186 hours, 14 hours overtime) that appear in the legal texts or examples.**
- Do **not invent** missing data. If the document does not include sufficient detail for a violation (e.g., no overtime hours), **do not report a violation**.
- Do not hallucinate sick days, overtime hours, or absences
- think step by step and analyze the documents carefully. do not rush to conclusions.
- while calculating read the whole law and dont miss anything and explain the calculations step by step and how you arrived at the final amounts.

Always respond in Hebrew.
Do not use any external knowledge or make up facts.
Always cite the provided laws and judgements in your corrections.
Always check and correct all calculations.
"""
        )


    async def review_analysis(self, laws: str, judgements: str, analysis: str, documents: Dict[str, str], context: str) -> str:
        """
        Review the analysis against the provided laws and judgements. If correct, return as-is. If not, return a corrected analysis.
        Also, carefully check all calculations (amounts, sums, percentages, totals, etc.) and correct any errors found.
        The output must never mention that it was revised, never explain what was fixed, and must simply return the corrected analysis as if it was always correct.
        """
        print("Reviewing analysis against provided laws and judgements...")
        prompt = f"""
הנך מקבל את החוקים, פסקי הדין, והניתוח המשפטי הבא:

📄 חוקים:
{laws}

📚 פסקי דין:
{judgements}

📑 ניתוח משפטי:
{analysis}

🧾 מסמכים שסופקו לבדיקה:
{ "\n\n".join([f"{doc_type.upper()} CONTENT:\n{content}" for doc_type, content in documents.items()]) }

🔍 הקשר נוסף:
{context}

🔍 הנחיות לבדיקה:

1. בדוק בקפידה האם הניתוח **נכון, שלם, ומבוסס אך ורק** על:
   - החוקים שסופקו
   - פסקי הדין שסופקו
   - המסמכים שסופקו (תלוש שכר, חוזה עבודה, דוח נוכחות)
2. אין להשתמש בידע כללי או חיצוני, ואין להניח מידע שאינו מופיע במפורש במסמכים.
3. אין להעתיק או לעשות שימוש בערכים מהחוקים כדוגמה (כגון 6000 ₪, 186 שעות, 5 ימי מחלה), אלא אם הם מופיעים במפורש במסמכים.
4. אם לא מצוינים נתונים מפורשים (כמו שעות נוספות, ימי מחלה, סכום שכר מינימום), יש להניח **שאין עבירה**, ואין לדווח על הפרה או לבצע חישובים משוערים.
5. כל החישובים (סכומים, אחוזים, טבלאות) חייבים להיות מדויקים ולבוסס אך ורק על הערכים המופיעים במסמכים שסופקו.
6. יש לוודא שהמסקנות המשפטיות תואמות את המסמכים ואת החוק, ללא שגיאות, וללא חריגות או תוספות לא מבוססות.

🛠 אם הניתוח תקין – החזר אותו כפי שהוא.  
🛠 אם יש בו שגיאות – תקן אותו כך שיהיה מדויק, מבוסס אך ורק על המסמכים, החוקים ופסקי הדין שסופקו.  


- לציין שבוצע תיקון.
- להסביר מה שונה.
- להזכיר שהניתוח תוקן או נערך מחדש.

✅ **יש להחזיר תמיד את הניתוח הסופי בלבד, בעברית מלאה וללא כל הערה.**

"""
        try:
            result = await self.review_agent.run(prompt,model_settings=ModelSettings(temperature=0.0))
            return result.data if hasattr(result, 'data') else str(result) 
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in review_analysis: {get_error_detail(e)}")

        

    def process_document(self, files: Union[UploadFile, List[UploadFile]], doc_types: Union[str, List[str]], compress: bool = False) -> Dict[str, str]:
        """Process uploaded documents and extract text"""
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
        
        # Initialize counters
        payslip_counter = 0
        contract_counter = 0
        attendance_counter = 0
        # Process each file based on its type
        for file, doc_type in zip(files, doc_types):
            print(f"Processing file: {file.filename} as type: {doc_type}")
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

    def _build_summary_prompt_from_combined(self, combined_report: str, analysis_type: str) -> str:
        """
        Build a prompt to convert the reviewed combined report into the requested summary type.
        """
        if analysis_type == "table":
            instructions = self._get_table_instructions()
        elif analysis_type == "violation_count_table":
            instructions = self._get_violation_count_table_instructions()
        elif analysis_type == "violations_list":
            instructions = self._get_violations_list_instructions()
        elif analysis_type == "easy":
            instructions = self._get_easy_instructions()
        elif analysis_type == "claim":
            instructions = self._get_claim_instructions()
        elif analysis_type == "warning_letter":
            instructions = self._get_warning_letter_instructions()
        else:
            raise ValueError(f"Unsupported summary analysis_type: {analysis_type}")

        prompt = f"""
להלן דוח ניתוח משפטי משולב של מסמכי העובד:

{combined_report}

---

{instructions}
"""
        return prompt

    async def qna(self, report: str, questions: str) -> str:
        """Generate answer of queries based on the provided document content."""
        prompt = f"""
להלן דוח ניתוח משפטי משולב של מסמכי העובד:

{report}

---

שאלות:
{questions}

---

אנא ספק תשובות לשאלות לעיל.
"""
        result = await self.agent.run(prompt, model_settings=ModelSettings(temperature=0.0))
        return result.data

    async def create_report(self, payslip_text: str = None, contract_text: str = None, 
                          attendance_text: str = None, analysis_type: str = "report", 
                          context: str = None) -> DocumentAnalysisResponse:
        """Create legal analysis report using RAG and PydanticAI"""

        # Prepare documents for analysis
        documents = {}
        if payslip_text:
            documents['payslip'] = payslip_text
        if contract_text:
            documents['contract'] = contract_text
        if attendance_text:
            documents['attendance report'] = attendance_text

        # Special handling for table, violation_count_table, violations_list, easy, claim, warning_letter
        special_types = ["table", "violation_count_table", "violations_list", "easy", "claim", "warning_letter"]
        if analysis_type in special_types:
            # Step 1: Run combined analysis and review
            combined_report = await self.create_report(
                payslip_text=payslip_text,
                contract_text=contract_text,
                attendance_text=attendance_text,
                analysis_type="combined",
                context=context
            )
            reviewed_combined = combined_report.legal_analysis
            print(reviewed_combined)
            # Step 2: Build a summary prompt from the reviewed combined report
            prompt = self._build_summary_prompt_from_combined(reviewed_combined, analysis_type)
            try:
                result = await self.agent.run(prompt, model_settings=ModelSettings(temperature=0.0))
                analysis = result.data
            except Exception as pydantic_error:
                raise pydantic_error
            # No review needed for these summary types
            return DocumentAnalysisResponse(
                legal_analysis=analysis,
                status="success",
                analysis_type=analysis_type
            )

        # Normal flow for all other types
        # Pass the documents to the question agent to generate search queries
        docs = "\n\n".join([f"{doc_type.upper()} CONTENT:\n{content}" for doc_type, content in documents.items()])
        try:
            question_result = await self.question_agent.run(docs)
        except Exception as e:
            print(f"Error generating search queries: {get_error_detail(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating search queries: {get_error_detail(e)}")

        # Extract the search queries from the result object
        search_queries = question_result.data if hasattr(question_result, 'data') else question_result
        
        # Ensure search_queries is a list
        # if not isinstance(search_queries, list):
        #     if isinstance(search_queries, str):
        #         # If it's a string, split by lines or use as single query
        #         search_queries = [search_queries]
        #     else:
        #         # Fallback to default queries
        #         search_queries = ["שכר מינימום", "שעות עבודה", "זכויות עובדים", "פנסיה"]
        
        print(f"Generated search queries: {search_queries}")

        # Query laws and judgements using the generated search queries
        all_relevant_laws = []
        all_relevant_judgements = []

        for query in search_queries:
            print(f"Searching for laws with query: {query}")
            laws = self.rag_storage.search_laws(query, n_results=2)  # Fewer results per query
            all_relevant_laws.extend(laws)

            print(f"Searching for judgements with query: {query}")
            judgements = self.rag_storage.search_judgements(query, n_results=1)  # Fewer results per query
            all_relevant_judgements.extend(judgements)

        # Remove duplicates based on ID or content
        unique_laws = []
        seen_law_ids = set()
        for law in all_relevant_laws:
            law_id = law.get('id') or law.get('metadata', {}).get('id')
            if law_id and law_id not in seen_law_ids:
                unique_laws.append(law)
                seen_law_ids.add(law_id)
            elif not law_id and law not in unique_laws:  # Fallback for items without ID
                unique_laws.append(law)

        unique_judgements = []
        seen_judgement_ids = set()
        for judgement in all_relevant_judgements:
            judgement_id = judgement.get('id') or judgement.get('metadata', {}).get('id')
            if judgement_id and judgement_id not in seen_judgement_ids:
                unique_judgements.append(judgement)
                seen_judgement_ids.add(judgement_id)
            elif not judgement_id and judgement not in unique_judgements:  # Fallback for items without ID
                unique_judgements.append(judgement)

        # Format the laws and judgements for the prompt
        formatted_laws = self.rag_storage.format_laws_for_prompt(unique_laws)
        formatted_judgements = self.rag_storage.format_judgements_for_prompt(unique_judgements)

        # Combine formatted content
        combined_context = f"{formatted_laws}\n\n{formatted_judgements}"
        print(f"Retrieved {len(unique_laws)} laws and {len(unique_judgements)} judgements")

        print(combined_context)
        # Build the analysis prompt based on type
        prompt = await self._build_analysis_prompt(
            analysis_type, documents, combined_context, context
        )

        try:
            # Generate analysis using PydanticAI
            try:
                result = await self.agent.run(prompt,model_settings=ModelSettings(temperature=0.0))
                analysis = result.data
            except Exception as pydantic_error:
                # If PydanticAI fails, check if it's an event loop issue
#                 if "Event loop is closed" in str(pydantic_error) or "event loop" in str(pydantic_error).lower():
#                     # Try to recreate the agent with a fresh context
#                     print(pydantic_error)
#                     print("Attempting to recreate PydanticAI agent due to event loop issue...")
#                       # Create a new agent instance
#                     temp_agent = Agent(
#                         model=self.model,
#                         result_type=str,
#                         system_prompt="""You are an expert legal document analyzer specializing in Israeli labor law compliance.

# You will be provided with:
# 1. Relevant labor laws retrieved from a legal database
# 2. Relevant legal judgements and precedents retrieved from a legal database  
# 3. Document content to analyze (payslips, contracts, attendance records)
# 4. Additional context if provided

# Your analysis must be based STRICTLY on the provided laws and judgements. Do not use external legal knowledge.

# 🚫 VERY IMPORTANT:
# - Read the document carefully and remember its contents like wage per hour, working hours, etc. and Do not recalculate data like wages per hour, sick days, etc. unless the document provides exact values.
# - Do not infer or estimate violations without clear proof in the payslip.
# - Use **only** the documents provided (e.g., payslip data, employment contracts data, and attendance records data). **Do not extract or reuse any example values (e.g., 6000 ₪, 186 hours, 14 hours overtime) that appear in the legal texts or examples.**
# - Do **not invent** missing data. If the document does not include sufficient detail for a violation (e.g., no overtime hours), **do not report a violation**.
# - Do not hallucinate sick days, overtime hours, or absences
# - think step by step and analyze the documents carefully. do not rush to conclusions.
# - while calculating read the whole law and dont miss anything and explain the calculations step by step and how you arrived at the final amounts.


# Always respond in Hebrew and follow the specific formatting requirements for each analysis type."""
#                     )
#                       # Try the request again with the fresh agent
#                     result = await temp_agent.run(prompt,model_settings=ModelSettings(temperature=0.0))
#                     analysis = result.data
#                 else:
                    # Re-raise the original error if it's not event loop related
                    raise pydantic_error
            print("Analysis generated successfully.")
            print(f"Analysis type: {analysis}")
            # --- Review the analysis for legal and calculation correctness ---
            try:
                reviewed_analysis = await self.review_analysis(formatted_laws, formatted_judgements, analysis, documents,context)
                print("Review agent completed successfully.")
            except Exception as review_error:
                print(f"Review agent failed: {review_error}")
                reviewed_analysis = analysis  # fallback to original if review fails

            return DocumentAnalysisResponse(
                legal_analysis=reviewed_analysis,
                # relevant_laws=unique_laws,
                # relevant_judgements=unique_judgements,
                status="success",
                analysis_type=analysis_type
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating legal analysis: {get_error_detail(e)}"
            )

    async def _build_analysis_prompt(self, analysis_type: str, documents: Dict, 
                                   laws_and_judgements: str, 
                                   context: str = None) -> str:
        """Build analysis prompt based on type"""
        
        base_prompt = f"""
RELEVANT LABOR LAWS and JUDGEMENTS (Retrieved from legal database):
{laws_and_judgements}


DOCUMENTS PROVIDED FOR ANALYSIS:
{', '.join(documents.keys())}

ADDITIONAL CONTEXT:
{context if context else 'No additional context provided.'}
"""
        # Add document contents to prompt
        for doc_type, content in documents.items():
            base_prompt += f"\n{doc_type.upper()} CONTENT:\n{content}\n"
        
        # Add specific instructions based on analysis type
        if analysis_type == 'report':
            base_prompt += self._get_report_instructions()
        elif analysis_type == 'profitability':
            base_prompt += self._get_profitability_instructions()
        # elif analysis_type == 'professional':
        #     base_prompt += self._get_professional_instructions()
        elif analysis_type == 'warning_letter':
            base_prompt += await self._get_warning_letter_instructions()
        elif analysis_type == 'easy':
            base_prompt += self._get_easy_instructions()
        elif analysis_type == 'table':
            base_prompt += self._get_table_instructions()
        elif analysis_type == 'claim':
            base_prompt += self._get_claim_instructions()
        elif analysis_type == 'combined':
            base_prompt += self._get_combined_instructions()
        elif analysis_type == 'violation_count_table':
            base_prompt += self._get_violation_count_table_instructions()
        elif analysis_type == 'violations_list':
            base_prompt += self._get_violations_list_instructions()
        
        return base_prompt

    def _get_report_instructions(self) -> str:
        return """
INSTRUCTIONS:
1. If no labor laws are provided, respond with: "אין חוקים לעבודה זמינים לניתוח התאמה." in Hebrew.
2. If labor laws exist, analyze the documents ONLY against the provided laws.
3. ONLY refer to the judgements and their results provided above for legal analysis - do not use external cases or knowledge.
4. If no judgements are provided, respond with: "לא קיימות החלטות משפטיות זמינות לניתוח." in Hebrew.

🎯 GOAL: Help the business owner understand exactly what went wrong and where they made mistakes in their employment practices.

For each payslip provided, analyze and identify violations. For each violation found in each payslip, format the response EXACTLY as shown below, with each section on a new line and proper spacing:

Violation Format Template:

🚨 [VIOLATION TITLE - What the employer did wrong]

📋 מה קרה בפועל:
[Describe exactly what the employer did or failed to do, with specific details from the documents]

⚖️ מה היה צריך לקרות לפי החוק:
[Explain what should have been done according to the law, with specific legal requirements]

📖 בסיס חוקי:
[LAW REFERENCE AND YEAR FROM PROVIDED LAWS - cite the specific law that was violated]

💰 הנזק הכספי:
[Calculate the exact financial impact - how much the employee lost due to this violation]

🏛️ תקדימים משפטיים:
[SIMILAR CASES OR PRECEDENTS FROM PROVIDED JUDGEMENTS](Refer *only* to the retrieved judgements. If a relevant judgement is found, describe the case and its result. If no relevant judgement is found, state "לא נמצאו תקדימים רלוונטיים בפסקי הדין שסופקו.")

⚠️ השלכות אפשריות:
[Explain potential legal consequences and risks for the employer]

✅ מה לעשות כדי לתקן:
[Specific actionable steps the employer should take to fix this violation and prevent it in the future]

---

SUMMARY FOR EMPLOYER:
After completing the violation analysis, provide a clear summary for the business owner:

=== סיכום למעסיק - איפה טעיתם ומה לעשות ===

📊 טבלת הפרות שזוהו:
תלוש/מסמך | מה עשיתם לא נכון | כמה זה עולה (₪) | מה לעשות עכשיו
[Add rows with actual data showing: document name | specific mistake made | financial cost | corrective action needed]

💸 סה"כ עלות הטעויות: [total amount] ₪

🔧 צעדים מיידיים לתיקון:
1. [First immediate action needed]
2. [Second immediate action needed]
3. [Third immediate action needed]

📋 איך למנוע טעויות בעתיד:
• [Prevention measure 1]
• [Prevention measure 2]
• [Prevention measure 3]

IMPORTANT:
- Always Respond in Hebrew
- Focus on helping the employer understand their mistakes and how to fix them
- Use clear, business-friendly language that explains the "why" behind each violation
- Format each violation with proper spacing and line breaks as shown above
- Analyze each payslip separately and clearly indicate which payslip the violations belong to
- Separate multiple violations with '---'
- If no violations are found against the provided laws in a payslip, respond with: "לא נמצאו הפרות בתלוש מספר" followed by the payslip number in hebrew
- If no violations are found in any payslip, respond with: "לא נמצאו הפרות נגד חוקי העבודה שסופקו." in hebrew
- DO NOT output template text or placeholders - use real data from the analysis
- Replace ALL placeholders with actual information from the documents
- Make it clear to the employer what they did wrong and how to prevent it happening again
"""

    def _get_violations_list_instructions(self) -> str:
        return """
CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:

You MUST respond with ONLY the format below. NO additional text, explanations, analysis, or commentary.

If violations found, use EXACTLY this format:

**הפרות שזוהו:**

• [הפרה קצרה] - [חוק]
• [הפרה קצרה] - [חוק]
• [הפרה קצרה] - [חוק]

If NO violations found, respond ONLY with: "לא נמצאו הפרות"

ABSOLUTE RESTRICTIONS:
❌ NO "Legal analysis" headers
❌ NO explanations or breakdowns  
❌ NO calculations or formulas
❌ NO "Analysis of the violation" sections
❌ NO "method of calculating" text
❌ NO English words
❌ NO additional paragraphs
❌ NO template text like [Violation Title]

✅ ONLY Hebrew
✅ ONLY the exact format above
✅ Maximum 4 violations
✅ Each violation = ONE short line

EXAMPLE of correct output:
**הפרות שזוהו:**

• לא שולם שכר מינימום - חוק שכר מינימום
• לא שולמו שעות נוספות - חוק שעות עבודה ומנוחה
• לא הופקדה פנסיה - צו הרחבה פנסיה

If you write ANYTHING beyond this format, you FAILED.
"""

    def _get_profitability_instructions(self) -> str:
        return """
INSTRUCTIONS:
1. Analyze the provided documents and identify potential labor law violations, based *exclusively on the retrieved LABOR LAWS and JUDGEMENTS*.
2. For each violation, refer *only* to the retrieved judgements to identify similar legal cases and their outcomes (both successful and unsuccessful).
3. If similar cases (from the retrieved judgements) were unsuccessful:
   - Explain why the cases were unsuccessful based on the retrieved judgement information.
   - Provide a clear recommendation against pursuing legal action based on this.
   - List potential risks and costs.

4. If similar cases (from the retrieved judgements) were successful, calculate using information from those judgements if available:
   - Average compensation amount from successful retrieved cases.
   - Estimated legal fees (30% of potential compensation).
   - Tax implications (25% of net compensation).
   - Time and effort cost estimation.

Provide the analysis in the following format:

ניתוח כדאיות כלכלית:

הפרות שזוהו (על בסיס החוקים שסופקו):
[List identified violations]

תקדימים משפטיים (מתוך פסקי הדין שסופקו):
[Similar cases from retrieved judgements with outcomes - both successful and unsuccessful]

במקרה של תקדימים שליליים (מתוך פסקי הדין שסופקו):
- סיבות לדחיית התביעות: [REASONS BASED ON RETRIEVED JUDGEMENTS]
- סיכונים אפשריים: [RISKS]
- המלצה: לא מומלץ להגיש תביעה בשל [EXPLANATION BASED ON RETRIEVED JUDGEMENTS]

במקרה של תקדימים חיוביים (מתוך פסקי הדין שסופקו):
ניתוח כספי:
- סכום פיצוי ממוצע (מפסיקות דין שסופקו): [AMOUNT] ₪
- עלות משוערת של עורך דין (30%): [AMOUNT] ₪
- השלכות מס (25% מהסכום נטו): [AMOUNT] ₪
- סכום נטו משוער: [AMOUNT] ₪

המלצה סופית:
[Based on analysis of both successful and unsuccessful cases from the retrieved judgements, provide clear recommendation]

SUMMARY TABLE:
After completing the profitability analysis, provide a summary table:
- Use the heading: === טבלת סיכום כלכלי ===
- Create columns for: פריט | סכום (₪)
- Add rows with actual calculated amounts for:
  * סה"כ פיצוי משוער
  * עלות עורך דין (30%)
  * מס (25%)
- End with: סכום נטו משוער: [calculated amount]

CRITICAL: Replace ALL placeholders with actual calculated values from the analysis. Do not output template text.
""""""
"""

#     def _get_professional_instructions(self) -> str:
#         return """
# Analyze the provided documents for labor law violations based strictly on the retrieved Israeli labor laws and the content of the documents. For each violation, calculate the monetary differences using only those laws.
# Provide your analysis in the following format, entirely in Hebrew:

# ניתוח מקצועי של הפרות שכר:

# הפרה: [כותרת ההפרה]
# [תיאור מפורט של ההפרה, כולל תאריכים רלוונטיים, שעות עבודה, שכר שעתי וחישובים, בהתבסס אך ורק על החוקים הישראליים שנמצאו והמסמכים שסופקו.
# דוגמה: העובד עבד X שעות נוספות בין [חודש שנה] ל-[חודש שנה]. לפי שכר שעתי בסיסי של [שכר] ₪ ושיעורי תשלום שעות נוספות ([שיעור1]% עבור X השעות הראשונות, [שיעור2]% לאחר מכן) כפי שמופיע בחוקי העבודה שנמצאו, העובד היה זכאי ל-[סכום] ₪ לחודש. בפועל קיבל רק [סכום שקיבל] ₪ למשך X חודשים ו-[סכום] ₪ בחודש [חודש].]
# סה"כ חוב: [סכום ההפרש עבור הפרה זו] ₪

# הפרה: [כותרת ההפרה]
# [תיאור מפורט של ההפרה, כולל תאריכים וחישובים, בהתבסס אך ורק על החוקים שנמצאו והמסמכים. 
# דוגמה: בחודש [חודש שנה] לא בוצעה הפקדה לפנסיה. המעסיק מחויב להפקיד [אחוז]% מהשכר בגובה [שכר] ₪ = [סכום] ₪ בהתאם לחוק/צו הרחבה שנמצא.]
# סה"כ חוב פנסיה: [סכום חוב הפנסיה להפרה זו] ₪

# ---

# סה"כ תביעה משפטית (לא כולל ריבית): [הסכום הכולל לתביעה מכלל ההפרות] ₪  
# אסמכתאות משפטיות: [רשימת שמות החוק הרלוונטיים מתוך החוקים הישראליים שנמצאו. לדוגמה: חוק שעות עבודה ומנוחה, צו הרחבה לפנסיה חובה]

# SUMMARY TABLE:
# After completing the professional analysis, provide a summary table with actual data:
# - Use the heading: === טבלת סיכום הפרות מקצועי ===
# - Create columns for: סוג הפרה | תקופה | סכום (₪)
# - Add rows with actual violation types, periods, and amounts from your analysis
# - End with a total line showing the total amount in ₪
# - Include legal references from retrieved laws

# CRITICAL: Replace ALL placeholders with actual data from the analysis. Do not output template text.
# """

    async def _get_warning_letter_instructions(self) -> str:
        # Get letter format from storage (we'll need to adapt this)
        # For now, using a default template
        format_content = """
[תאריך]

לכבוד
[שם המעסיק]
[כתובת המעסיק]

הנדון: התראה בגין הפרות חוקי עבודה

בהתבסס על בדיקת המסמכים שבידינו, נמצאו הפרות של חוקי עבודה כמפורט להלן:

[פרטי ההפרות]

הפרות אלו מהוות הפרה של:
[הפניות לחוקים]

בהתאם לכך, אנו דורשים כי תתקנו את ההפרות הנ"ל תוך [מועד] ימים מקבלת מכתב זה.

אי תיקון ההפרות עלול להוביל לנקיטת הליכים משפטיים.

בכבוד,
[חתימה]
"""
        
        return f"""
INSTRUCTIONS:
1. Analyze the provided documents for labor law violations *based exclusively on the retrieved LABOR LAWS and JUDGEMENTS*.
2. If violations are found, generate a formal warning letter using the provided template.
3. If no violations are found, respond with: "לא נמצאו הפרות המצדיקות מכתב התראה." in Hebrew.

Warning Letter Template:
{format_content}

Please generate the warning letter in Hebrew with the following guidelines:
- Replace [שם המעסיק] with the employer's name from the documents
- Replace [פרטי ההפרות] with specific details of each violation found (based on retrieved laws)
- Replace [הפניות לחוקים] with relevant labor law citations *from the retrieved LABOR LAWS*.
- Replace [מועד] with a reasonable timeframe for corrections (typically 14 days)
- Maintain a professional and formal tone throughout
- Include all violations found in the analysis (based on retrieved laws)
- Format the letter according to the provided template structure
"""

    def _get_easy_instructions(self) -> str:
        return """
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

SUMMARY TABLE:
After completing the easy summary, provide a simple visual table:

=== טבלת סיכום ויזואלי ===
חודש/שנה            | סוג הפרה           | סכום (₪)
נובמבר 2024          | שעות נוספות        | 495
נובמבר 2024          | פנסיה             | 750
דצמבר 2024           | פנסיה             | 1,221
ינואר 2025           | פנסיה             | 831
פברואר 2025          | נסיעות            | 250
פברואר 2025          | פנסיה             | 858
מרץ 2025             | פנסיה             | 866
-----------------------------------------
סה"כ: 5,271 ₪
"""

    def _get_table_instructions(self) -> str:
        return """
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

SUMMARY TABLE:
After completing the organized document list, provide a simple summary table:

=== טבלת סיכום לפי מסמך ===
מסמך                | מספר הפרות         | סכום כולל (ש"ח)
תלוש שכר מס' 6       | 2                  | 1,245.00
תלוש שכר מס' 7       | 1                  | 396.00  
תלוש שכר מס' 9       | 1                  | 250.00
------------------------------------------
סה"כ                | 4                  | 1,891.00
"""

    def _get_claim_instructions(self) -> str:
        return """
משימה:
כתוב טיוטת כתב תביעה לבית הדין האזורי לעבודה, בהתאם למבנה המשפטי הנהוג בישראל.

נתונים:
השתמש במידע מתוך המסמכים שצורפו (כגון תלושי שכר, הסכמי עבודה, הודעות פיטורין, שיחות עם המעסיק) ובממצאים שנמצאו בניתוח קודם של ההפרות.

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

SUMMARY TABLE:
After completing the claim draft, provide a simple summary table:

=== טבלת סיכום כתב התביעה ===
עילת תביעה           | תקופה               | סכום נתבע (₪)
אי תשלום שעות נוספות  | נובמבר 2024          | 495
אי הפקדת פנסיה       | נובמבר 2024-מרץ 2025  | 3,525
אי החזר נסיעות       | פברואר 2025          | 250
---------------------------------------------
סה"כ תביעה ראשית: 4,270 ₪
ריבית והצמדה: לפי החלטת בית הדין
הוצאות משפט: לפי החלטת בית הדין
"""

    def _get_combined_instructions(self) -> str:
        return """
COMPREHENSIVE ANALYSIS INSTRUCTIONS:

📌 OVERVIEW
You are analyzing employee documents(payslips, employment contracts, and attendance records) for legal violations based **only** on the provided labor laws and judgements. Do **not** use any default knowledge, assumptions, or examples from the law unless explicitly confirmed in the payslip.
and explain step by step your thought process, calculations, and legal references.
labor laws and judgements are different from employee documents dont use example values from the legal texts or examples. use only real data from the documents provided.

PART 1 - VIOLATION IDENTIFICATION AND ANALYSIS:

1. If no labor laws are provided, respond with: "אין חוקים לעבודה זמינים לניתוח התאמה." in Hebrew.
2. If labor laws exist, analyze the payslip ONLY against the laws provided.
3. Use **only** the documents provided (e.g., payslip data, employment contracts data, and attendance records data). **Do not extract or reuse any example values (e.g., 6000 ₪, 186 hours, 14 hours overtime) that appear in the legal texts or examples.**
4. Do **not invent** missing data. If the document does not include sufficient detail for a violation (e.g., no overtime hours), **do not report a violation**.
5. If no judgements are provided, respond with: "לא קיימות החלטות משפטיות זמינות לניתוח." in Hebrew.


PART 2 - DETAILED PROFESSIONAL ANALYSIS FORMAT:

For each violation found, provide the following, using actual values from the document:

הפרה: [כותרת ההפרה]
[תיאור מפורט של ההפרה, כולל תאריכים רלוונטיים, שעות עבודה בפועל, שכר שעתי, ימי מחלה מאושרים וכדומה. התבסס רק על הנתונים הקיימים בתלוש, ובאופן בלעדי על החוקים שהוצגו.]

[הפניה לחוק הרלוונטי ושנה מתוך החוקים שסופקו]

[אם קיימים – תאר תקדימים משפטיים מתוך פסקי הדין שסופקו. אם אין – כתוב "לא נמצאו תקדימים רלוונטיים בפסקי הדין שסופקו."]

[השלכות משפטיות ופעולות מומלצות]

סה"כ חוב להפרה זו: [סכום] ₪

---

PART 3 - ORGANIZED TABLE FORMAT BY DOCUMENT:

רשימת תביעות מסודרת לפי מסמך:

תלוש שכר מס' [מספר] – [חודש/שנה]:
א. סכום של [amount] ש"ח עבור [תיאור קצר של ההפרה].
ב. סכום של [amount] ש"ח עבור [תיאור קצר של ההפרה].

---

PART 4 - FINAL SUMMARY:

סה"כ תביעה משפטית (לא כולל ריבית): [סכום כולל] ₪  
אסמכתאות משפטיות: [רשימת שמות החוקים הרלוונטיים מתוך החוקים שסופקו]

---

PART 5 - COMPREHENSIVE SUMMARY TABLE:

=== טבלת סיכום מקיפה ===

Create columns:
- תלוש/מסמך
- סוג הפרה
- תקופה
- סכום (₪)

Add rows using actual data per document and violation.

🚨 TOTAL LINE:
Show total number of violations, total documents, and total amount claimed.

🔍 Breakdown by violation type:
- Total number of documents where each violation occurred
- Total amount per violation type

📘 Legal references:
List names of all laws used in calculations.

---

⚠️ FORMATTING & ACCURACY RULES:

- Always respond in Hebrew
- NEVER use sample values or formulas from laws directly
- Use only data provided in the payslip, employment contracts, and attendance records
- Replace ALL placeholders with actual values
- Do not hallucinate sick days, overtime hours, or absences
- If no violations found for a payslip, write:
  "לא נמצאו הפרות בתלוש מספר [X]"
- If no violations in any payslip, write:
  "לא נמצאו הפרות נגד חוקי העבודה שסופקו."
"""

    def _get_violation_count_table_instructions(self) -> str:
        return """
INSTRUCTIONS:
Analyze the provided documents and identify potential labor law violations, based *exclusively on the retrieved LABOR LAWS and JUDGEMENTS*.
Return ONLY a summary table in Hebrew, with NO additional analysis, commentary, or template text. Do not include any explanations, recommendations, or placeholders.

Provide a comprehensive summary table with actual data:
- Use the heading: === טבלת סיכום מקיפה ===
- Create columns for: תלוש/מסמך | סוג הפרה | תקופה | סכום (₪) | מספר הפרות
- Add rows with actual document names, violation types, periods, amounts, and the count of each violation type per document
- For each violation type, calculate and display the total amount for that violation type across all documents, and the total count of documents/slips affected
- For each violation type, explicitly state how many documents/slips had that violation (e.g., "6 מתוך 10 תלושים נמצאה הפרה של אי תשלום פנסיה")
- End with a total line showing total violations, months, and total amount
- Include a breakdown by violation type, showing the total count and total amount for each type of violation
- All amount calculations must be thorough and accurate: sum the amounts for each violation type, each document, and overall totals. Do not estimate or round; use the actual amounts found in the analysis.
- Include legal references from retrieved laws

Formatting requirements:
- Always respond in Hebrew
- Use actual violation types, counts, and amounts from the analysis
- Do not output any template text or placeholders
- Do not include any additional commentary or explanations
- If no violations are found, respond with: "לא נמצאו הפרות נגד חוקי העבודה שסופקו."
"""

    def _extract_text2(self, content: bytes, filename: str, compress: bool = False) -> str:
        """Extract text from various document formats using AgenticDoc for PDFs and images"""
        print("Extracting text from file:", filename.lower())
        
        try:
            if filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                # Pass bytes directly to AgenticDoc
                print("Extracting text from image or PDF...")
                
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
                    text += f"## Sheet: {sheet}\n"
                    # Convert DataFrame to Markdown table
                    text += data.to_markdown(index=False) + "\n\n"
                return text
            else:
                # Unknown file type, treat as plain text
                result = parse(content)
                # Return markdown or structured, as you prefer
                return result[0].markdown
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Extraction error: {get_error_detail(e)}")

    def _extract_text_legacy(self, content: bytes, filename: str, compress: bool = False) -> str:
        """Legacy text extraction method (kept for fallback)"""
        if filename.lower().endswith('.pdf'):
            return self._extract_pdf_text(content)
        elif filename.lower().endswith('.docx'):
            return self._extract_docx_text(content)
        elif filename.lower().endswith('.xlsx'):
            return self._extract_excel_text(content)
        else:
            return self._extract_image_text(content, compress)

    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF using pdfplumber and Vision API"""
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
            return text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {get_error_detail(e)}")

    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX files"""
        doc_file = BytesIO(content)
        doc = Document(doc_file)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

    def _extract_excel_text(self, content: bytes) -> str:
        """Extract text from Excel files"""
        try:
            excel_file = BytesIO(content)
            df = pd.read_excel(excel_file, sheet_name=None)  # Load all sheets
            
            text = ""
            for sheet_name, sheet_data in df.items():
                text += f"Sheet: {sheet_name}\n"
                text += sheet_data.to_string(index=False) + "\n\n"
            return text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing Excel file: {get_error_detail(e)}")

    def _extract_image_text(self, content: bytes, compress: bool = False) -> str:
        """Extract text from images using Vision API"""
        try:
            if compress:
                content = self._compress_image(content)
            
            vision_image = vision.Image(content=content)
            response = self.vision_client.document_text_detection(image=vision_image, image_context=self.image_context)
            
            if response.error.message:
                raise HTTPException(status_code=500, detail=f"Vision API Error: {response.error.message}")
            
            if response.text_annotations:
                return " ".join([text_annotation.description for text_annotation in response.text_annotations])
            else:
                return ""
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {get_error_detail(e)}")

    def _compress_image(self, image_bytes: bytes, max_size_mb: int = 4) -> bytes:
        """Compress image to reduce size for API limits"""
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

    async def summarise(self, ai_content_text: str) -> str:
        """
        Summarizes the given text using PydanticAI.
        """
        try:
            prompt = f"Please summarize the following text concisely in Hebrew:\n\n{ai_content_text}"
            
            # Use the same agent for summarization
            try:
                result = await self.agent.run(prompt)
                return result.data if hasattr(result, 'data') else str(result)
            except Exception as pydantic_error:
                # If PydanticAI fails, check if it's an event loop issue
                if "Event loop is closed" in str(pydantic_error) or "event loop" in str(pydantic_error).lower():
                    # Try to recreate the agent with a fresh context
                    print("Attempting to recreate PydanticAI agent for summarization due to event loop issue...")
                    
                    # Create a new agent instance
                    temp_agent = Agent(
                        model=self.model,
                        result_type=str,
                        system_prompt="You are a helpful assistant that summarizes text concisely."
                    )
                    
                    # Try the request again with the fresh agent
                    result = await temp_agent.run(prompt)
                    return result.data if hasattr(result, 'data') else str(result)
                else:
                    # Re-raise the original error if it's not event loop related
                    raise pydantic_error
            
        except Exception as e:
            error_detail = f"Error generating summary with PydanticAI: {get_error_detail(e)}"
            raise HTTPException(
                status_code=500,
                detail=error_detail
            )
    
    # Sync wrapper methods for backward compatibility
    def create_report_sync(self, payslip_text: str = None, contract_text: str = None, 
                          attendance_text: str = None, type: str = "report", 
                          context: str = None) -> Dict:
        """Synchronous wrapper for create_report method with improved event loop handling"""
        # Handle backward compatibility parameter mapping
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new thread
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    # Create new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.create_report(
                            payslip_text=payslip_text,
                            contract_text=contract_text, 
                            attendance_text=attendance_text,
                            analysis_type=type,  # Map 'type' to 'analysis_type'
                            context=context
                        ))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    result = future.result()
            else:
                # No running loop, use asyncio.run
                result = asyncio.run(self.create_report(
                    payslip_text=payslip_text,
                    contract_text=contract_text, 
                    attendance_text=attendance_text,
                    analysis_type=type,  # Map 'type' to 'analysis_type'
                    context=context
                ))
        except RuntimeError as e:
            if "no current event loop" in str(e).lower() or "event loop is closed" in str(e).lower():
                # Create new event loop
                result = asyncio.run(self.create_report(
                    payslip_text=payslip_text,
                    contract_text=contract_text, 
                    attendance_text=attendance_text,
                    analysis_type=type,  # Map 'type' to 'analysis_type'
                    context=context
                ))
            else:
                raise e
        
        # Convert response to dict for backward compatibility
        return {
            "legal_analysis": result.legal_analysis,
            "status": result.status,
            # "relevant_laws": result.relevant_laws,
            # "relevant_judgements": result.relevant_judgements
        }
    
    def summarise_sync(self, ai_content_text: str) -> str:
        """Synchronous wrapper for summarise method with improved event loop handling"""
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new thread
                import concurrent.futures
                
                def run_in_thread():
                    # Create new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.summarise(ai_content_text))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result()
            else:
                # No running loop, use asyncio.run
                return asyncio.run(self.summarise(ai_content_text))
        except RuntimeError as e:
            if "no current event loop" in str(e).lower() or "event loop is closed" in str(e).lower():
                # Create new event loop
                return asyncio.run(self.summarise(ai_content_text))
            else:
                raise e
