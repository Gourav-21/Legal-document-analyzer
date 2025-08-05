# Fix for SQLite3 compatibility with ChromaDB on Streamlit Cloud
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    print("✅ SQLite3 fix applied in app.py")
except ImportError:
    print("⚠️ pysqlite3-binary not available in app.py")
    import sqlite_fix  # Fallback to sqlite_fix

import streamlit as st
from document_processor_pydantic_ques import DocumentProcessor
from fastapi import UploadFile
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="מנתח מסמכים משפטיים",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize DocumentProcessor
doc_processor = DocumentProcessor()


# Add custom CSS for small buttons
st.markdown("""
<style>
.stButton>button {
    padding: 0.25rem 1rem;
    font-size: 0.875rem;
    min-height: 0;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Add RTL support
st.markdown("""
<style>
    .element-container, .stMarkdown, .stButton, .stTextArea {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for document processing
if 'processed_result' not in st.session_state:
    # st.session_state.processed_result = {'payslip_text': '--- Payslip 1 ---\nשם החברה:\nיצחק אברהם\nאילת, <!-- text, from page 0 (l=0.650,t=0.020,r=0.950,b=0.073), with ID a88bfa4f-9522-47a0-a5c9-98b5f7b8fdc8 -->\n\nתלוש משכורת לחודש 8/2024  \nכתובת: תרומוס 13/10, אילת <!-- text, from page 0 (l=0.343,t=0.019,r=0.648,b=0.064), with ID 963f656e-da6c-4014-80df-b03e824354db -->\n\nתיק ניקויים\n\nב.ל: 950132886\nמ.ה: 950132886\n\n37674066 - מספר תאגיד <!-- text, from page 0 (l=0.032,t=0.024,r=0.159,b=0.077), with ID 112bbee4-9bfc-4ea1-bd3e-754b221c5d0d -->\n\n<table><thead><tr><th>מס\' עובד</th><th>מחלקה</th><th>שם עובד</th><th>תעודת זהות</th><th>תת מחלקה</th><th>דרוג</th><th>דרגה</th><th>ותק</th><th>תחילת עבודה</th></tr></thead><tbody><tr><td>1</td><td>0</td><td>יצחק</td><td>רמה</td><td>035715044</td><td></td><td></td><td>י ח ש</td><td>01/11/22</td></tr></tbody></table> <!-- table, from page 0 (l=0.034,t=0.076,r=0.949,b=0.111), with ID 2ff27458-8824-4910-8923-e5e6b9b89d19 -->\n\n<table><thead><tr><th>תעריף</th><th>תעריף יום</th><th>תעריף שעה</th><th>ימי עבודה</th><th>שעות עבודה 182</th><th>בנק</th><th>סניף</th><th>חשבון</th><th>פיצויים חודשי</th><th>0.00</th></tr></thead><tbody><tr><td>7,000.00</td><td>323.03</td><td>38.46</td><td>26</td><td></td><td></td><td></td><td>פיצויים פטור</td><td>:</td><td>0.00</td></tr><tr><td colspan="4">תאור התשלום</td><td>כמות</td><td>תעריך</td><td>אחוז</td><td>נטו לתשלום</td><td>סכום התשלום</td><td>פיצויים ותיקה :</td><td>0.00</td></tr><tr><td>שכר יסוד</td><td>1.00</td><td>7,000.00</td><td></td><td></td><td></td><td>7,000.00</td><td>שכר לפיצויים</td><td>:</td><td>0.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>קופ"ל מעסיק-חודשי</td><td>:</td><td>0.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>שכר לקופ"ג</td><td></td><td>0.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>קה"ל מעסיק-חודשי</td><td>:</td><td>0.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>שכר לקה"ל</td><td>:</td><td>0.00</td></tr><tr><td>אחוז משרה</td><td>ז דיוני</td><td>מצב משפחתי</td><td>מצב גבר</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>נ</td><td>נ</td><td></td><td></td><td></td><td></td><td></td><td>2.75</td></tr><tr><td>מס שולי</td><td>% מס קבוע</td><td>זיכוי אישי</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>10.00</td><td></td><td>665</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>זיכוי נוסף</td><td>זיכוי נצבר</td><td>זיכוי השתלמות</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>פטור חודשי</td><td>פטור מ 47</td><td>הכנסה יתרות פיצויים</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>700</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>זכאים מס</td><td>שכר לזכאים</td><td>ח.פ. לזכאים</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ל</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>חיוב מה</td><td>7,000.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>חיוב כ\'</td><td>7,000.00</td></tr></tbody></table> <!-- table, from page 0 (l=0.033,t=0.109,r=0.950,b=0.555), with ID 133b2757-224e-4283-aebe-e44b8a6b17a6 -->\n\nאינפורמטיבי: שכר סיכומים לחודש : 5880.02 ימי תקן 21.67\nאינפורמטיבי: שכר סיכומים לשעה : 32.3 שעות תקן 182.00\n<table><thead><tr><th>סכום</th><th>כמות</th><th>יתרה</th><th>ניכויי רשות</th></tr></thead><tbody><tr><td></td><td></td><td></td><td></td></tr><tr><td>סה"כ</td><td></td><td></td><td></td></tr></tbody></table>\n<table><thead><tr><th>הסכום</th><th>ניכויי חובה</th></tr></thead><tbody><tr><td>28.00</td><td>ביטוח לאומי</td></tr><tr><td>217.00</td><td>דמי בריאות</td></tr><tr><td>245.00</td><td>סה"כ</td></tr></tbody></table> <!-- table, from page 0 (l=0.245,t=0.556,r=0.955,b=0.770), with ID 73c128e9-b2fe-409e-ac5a-be2e8df3e177 -->\n\n<table><thead><tr><th>זיכוי אישי</th><th>% מס קבוע</th><th>מס שולי</th></tr></thead><tbody><tr><td>665</td><td></td><td>10.00</td></tr><tr><td>זיכוי משתלמות</td><td>זיכוי גמל</td><td>זיכוי נוסף</td></tr><tr><td></td><td></td><td></td></tr><tr><td>פטור חודשי<br>ישובי פיתוח<br>700</td><td>פטור ס\' 47</td><td></td></tr><tr><td>מ.ה. לתשלום</td><td>שכר לתשלום</td><td>תשלום<br>מס<br>ל</td></tr><tr><td colspan="2">7,000.00</td><td>חייב מ.ה</td></tr><tr><td colspan="2">7,000.00</td><td>חייב ב.ל</td></tr><tr><td colspan="2">7,000.00</td><td>סה"כ תשלומים</td></tr><tr><td colspan="2">245.00</td><td>סה"כ ניכויים</td></tr></tbody></table> <!-- table, from page 0 (l=0.026,t=0.267,r=0.248,b=0.666), with ID 6c3303c7-5aa4-45e9-af3f-f1a2230dd2aa -->\n\n6,755.00\n\nשכר נטו <!-- text, from page 0 (l=0.025,t=0.690,r=0.211,b=0.754), with ID bb876145-ea4e-4df3-9ade-2c6a962e198e -->\n\nנטו לתשלום: 6,755.00 <!-- text, from page 0 (l=0.028,t=0.761,r=0.220,b=0.820), with ID a0bc93ba-dba8-451c-b8f3-762a6e4f8331 -->\n\nהערות: <!-- text, from page 0 (l=0.855,t=0.778,r=0.918,b=0.799), with ID a4e7dce7-1c94-43b4-9dc7-bc0ba2f25435 -->\n\nניהול העדריות\n<table><thead><tr><th>סוג העדרות</th><th>יתרה קודמת</th><th>ניצול</th><th>יתרה</th></tr></thead><tbody><tr><td>חופש</td><td></td><td></td><td></td></tr><tr><td colspan="2">צבירת מחלה :</td><td colspan="2">חודשי עבודה 1 2 3 4 5 6 7 8 9 10 11 12<br>כ כ כ כ כ כ כ כ</td></tr></tbody></table> <!-- table, from page 0 (l=0.614,t=0.808,r=0.956,b=0.971), with ID cc211a16-7e9e-4dfd-8d66-d2922a20afbb -->\n\nנתונים מצטברים\n<table><tr><th>תשלומים</th><th>56,000</th><th>5,324</th></tr><tr><td>שכר שנות נטו</td><td>56,000</td><td>זיכוי אישי</td></tr><tr><td>חייב מ.ה.</td><td>56,000</td><td>זיכוי נוסף</td></tr><tr><td>מס הכנסה</td><td></td><td>זיכוי גמל</td></tr><tr><td>ביטוח לאומי</td><td>1,960</td><td>זיכוי משמרות</td></tr><tr><td>נמל 35%</td><td>0</td><td>פטור</td></tr><tr><td>ק. השתלמות</td><td>0</td><td>פטור ס\' 47</td></tr></table> <!-- table, from page 0 (l=0.284,t=0.810,r=0.619,b=0.973), with ID d05bb59e-e9b3-47a5-87ad-97626d5acb5e -->\n\n<table><tr><td></td><td>56,000</td><td>חיוב כולל</td></tr><tr><td></td><td>0.00</td><td>סה"כ מטפים</td></tr><tr><td></td><td>0.00</td><td>סה"כ מטפים</td></tr><tr><td></td><td>0.00</td><td>פיצויים מטפים</td></tr></table> <!-- table, from page 0 (l=0.032,t=0.824,r=0.286,b=0.975), with ID 3a0fd597-6918-4367-af70-65fc62e05166 -->\n\nבוצע ע"י: ליליאן - הנה"ח <!-- marginalia, from page 0 (l=0.781,t=0.978,r=0.953,b=0.999), with ID 01287ec1-4f6b-43b5-a9a9-bbf13153844b -->\n\nבתאריך 12/09/2024 <!-- marginalia, from page 0 (l=0.300,t=0.979,r=0.438,b=0.999), with ID fc58d8dd-e96a-4194-952b-40e72e60d301 -->', 'contract_text': '', 'attendance_text': ''}
    st.session_state.processed_result = {'payslip_text': '--- Payslip 1 ---\nתלוש שכר - ינואר 2024\nשם העובד: דוד כהן\nת.ז: 123456789\nשם המעסיק: פתרונות חכמים בע"מ\nחודש עבודה: ינואר 2024\nמשרה: מלאה (100%)\nשכר יסוד: 5,100 ₪\nשעות עבודה בפועל: 186 שעות\nתשלום עבור שעות נוספות: 0 ₪\nהפרשה לפנסיה: לא בוצעה\nדמי הבראה: לא שולמו\nימי חופשה שנצברו: 2\nימי חופשה נוצלו: 0\nסך ניכויים: 950 ₪\nשכר נטו לתשלום: 4,150 ₪\nתאריך תשלום בפועל: 15/02/2024 (באיחור)', 'contract_text': '--- Contract 1 ---\nהסכם עבודה\nהסכם זה נחתם ביום 1 בינואר 2024 בין:\nהמעסיק: פתרונות חכמים בע"מ\nלבין: דוד כהן, ת.ז 123456789\n\nתנאי ההעסקה:\n- משרה מלאה (186 שעות חודשיות)\n- שכר חודשי: 6,000 ₪ (גבוה יותר מהתלוש בפועל)\n- התחלת עבודה: 1 בינואר 2024\n- התחייבות להפרשה מלאה לפנסיה מהיום הראשון\n- זכאות לדמי הבראה לאחר 12 חודשי עבודה\n- זכאות לימי חופשה בהתאם לחוק', 'attendance_text': '--- Attendance Report 1 ---\nדוח נוכחות - ינואר 2024\nסה"כ ימי עבודה: 23 ימים\nסה"כ שעות עבודה: 186 שעות\nמספר שעות נוספות: 14 שעות\nימי היעדרות ללא אישור: 0\nימי מחלה: 2 (לא שולמו בתלוש)'}
    # st.session_state.processed_result = None
    
# Title and description
st.title("📄 מנתח מסמכים משפטיים")
st.markdown("העלה את המסמכים המשפטיים שלך לבדיקת תאימות לחוקי העבודה הישראליים.")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 ניתוח מסמכים", "📚 ניהול חוקים", "📝 ניהול תבנית", "⚖️ ניהול פסקי דין"])

# Document Analysis Tab
with tab1:
    # Create document upload sections
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📄 תלושי שכר")
        payslip_files = st.file_uploader(
            "העלה תלושי שכר",
            type=["pdf", "png", "jpg", "jpeg", 'webp', 'docx'],
            accept_multiple_files=True,
            key="payslip_upload"
        )

    with col2:
        st.subheader("📑 חוזי עבודה")
        contract_files = st.file_uploader(
            "העלה חוזי עבודה",
            type=["pdf", "png", "jpg", "jpeg", 'webp', 'docx'],
            accept_multiple_files=True,
            key="contract_upload"
        )
    
    with col3:
        st.subheader("⏰ דוחות נוכחות")
        attendance_files = st.file_uploader(
            "העלה דוחות נוכחות",
            type=["pdf", "png", "jpg", "jpeg", 'webp', 'docx','xlsx'],
            accept_multiple_files=True,
            key="attendance_upload"
        )

    # Add context input field
    context = st.text_area("הקשר נוסף או הערות מיוחדות", height=100)

    # Process documents button
    if (payslip_files or contract_files or attendance_files) and st.button("עבד מסמכים", type="primary"):
        try:
            with st.spinner("מעבד מסמכים..."):
                # Prepare a result dictionary to hold combined texts
                result = {
                    "payslip_text": "",
                    "contract_text": "",
                    "attendance_text": ""
                }

                if payslip_files:
                    payslip_texts = []
                    for i, file in enumerate(payslip_files, 1):
                        file_content = file.read()
                        # Use new AgenticDoc-powered method, returns markdown/text
                        payslip_text = doc_processor._extract_text2(file_content, file.name,True)
                        payslip_texts.append(f"--- Payslip {i} ---\n{payslip_text}")
                    result["payslip_text"] = "\n\n".join(payslip_texts)

                if contract_files:
                    contract_texts = []
                    for i, file in enumerate(contract_files, 1):
                        file_content = file.read()
                        # Use same method for contracts
                        contract_text = doc_processor._extract_text2(file_content, file.name,True)
                        contract_texts.append(f"--- Contract {i} ---\n{contract_text}")
                    result["contract_text"] = "\n\n".join(contract_texts)

                if attendance_files:
                    attendance_texts = []
                    for i, file in enumerate(attendance_files, 1):
                        file_content = file.read()
                        # Use same method for attendance files (returns markdown or plain text)
                        attendance_text = doc_processor._extract_text2(file_content, file.name ,True)
                        attendance_texts.append(f"--- Attendance Report {i} ---\n{attendance_text}")
                    result["attendance_text"] = "\n\n".join(attendance_texts)

                # Save to session state
                st.session_state.processed_result = result
                # print(result)
                st.success("המסמכים עובדו בהצלחה! כעת תוכל לבחור סוג ניתוח.")

        except Exception as e:
            st.error(f"שגיאה בעיבוד המסמכים: {str(e)}")
            st.info("אנא ודא שהמסמכים בפורמט הנכון ונסה שוב.")


    # Analysis buttons (only shown after processing)
    if st.session_state.get('processed_result'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("דוח המעסיק", type="primary", key="report_btn"):
                st.session_state.report_output_title = None
                st.session_state.report_output_content = None
                st.session_state.summary_output_content = None
                st.session_state.last_legal_analysis = None
                st.session_state.qna_result = None
                st.session_state["qna_question_input"] = ""
                try:
                    with st.spinner("מנתח..."):
                        result = doc_processor.create_report_sync(
                            payslip_text=st.session_state.processed_result.get('payslip_text'),
                            contract_text=st.session_state.processed_result.get('contract_text'),
                            attendance_text=st.session_state.processed_result.get('attendance_text'),
                            type="report",
                            context=context
                        )
                        if result.get('legal_analysis'):
                            st.session_state.report_output_title = "### תוצאות ניתוח משפטי"
                            st.session_state.report_output_content = result['legal_analysis']
                            st.session_state.last_legal_analysis = result['legal_analysis']
                except Exception as e:
                    st.error(f"שגיאה בניתוח: {str(e)}")
                    
            if st.button("ניתוח כדאיות כלכלית", type="primary", key="profitability_btn"):
                st.session_state.report_output_title = None
                st.session_state.report_output_content = None
                st.session_state.summary_output_content = None
                st.session_state.last_legal_analysis = None
                st.session_state.qna_result = None
                st.session_state["qna_question_input"] = ""
                try:
                    with st.spinner("מנתח כדאיות כלכלית..."):
                        result = doc_processor.create_report_sync(
                            payslip_text=st.session_state.processed_result.get('payslip_text'),
                            contract_text=st.session_state.processed_result.get('contract_text'),
                            attendance_text=st.session_state.processed_result.get('attendance_text'),
                            type="profitability",
                            context=context
                        )
                        if result.get('legal_analysis'):
                            st.session_state.report_output_title = "### תוצאות ניתוח כדאיות כלכלית"
                            st.session_state.report_output_content = result['legal_analysis']
                            st.session_state.last_legal_analysis = result['legal_analysis']
                except Exception as e:
                    st.error(f"שגיאה בניתוח: {str(e)}")
                    
                    # Button for violation count table
            if st.button("טבלת הפרות", type="primary", key="violation_count_btn"):
                st.session_state.report_output_title = None
                st.session_state.report_output_content = None
                st.session_state.summary_output_content = None
                st.session_state.last_legal_analysis = None
                st.session_state.qna_result = None
                st.session_state["qna_question_input"] = ""
                try:
                    with st.spinner("יוצר טבלת הפרות..."):
                        result = doc_processor.create_report_sync(
                            payslip_text=st.session_state.processed_result.get('payslip_text'),
                            contract_text=st.session_state.processed_result.get('contract_text'),
                            attendance_text=st.session_state.processed_result.get('attendance_text'),
                            type="violation_count_table",
                            context=context
                        )
                        if result.get('legal_analysis'):
                            st.session_state.report_output_title = "### טבלת הפרות"
                            st.session_state.report_output_content = result['legal_analysis']
                            st.session_state.last_legal_analysis = result['legal_analysis']
                        else:
                            st.info("לא התקבל תוכן להצגת טבלת ספירת הפרות.")
                except Exception as e:
                    st.error(f"שגיאה ביצירת טבלת ספירת הפרות: {str(e)}")


        
        with col2:
            if st.button("הכן תביעה", type="primary", key="claim_btn"):
                st.session_state.report_output_title = None
                st.session_state.report_output_content = None
                st.session_state.summary_output_content = None
                st.session_state.last_legal_analysis = None
                st.session_state.qna_result = None
                st.session_state["qna_question_input"] = ""
                try:
                    with st.spinner("מכין טיוטת תביעה..."):
                        result = doc_processor.create_report_sync(
                            payslip_text=st.session_state.processed_result.get('payslip_text'),
                            contract_text=st.session_state.processed_result.get('contract_text'),
                            attendance_text=st.session_state.processed_result.get('attendance_text'),
                            type="claim",
                            context=context
                        )
                        if result.get('claim_draft'):
                            st.session_state.report_output_title = "### טיוטת כתב תביעה"
                            st.session_state.report_output_content = result['claim_draft']
                            st.session_state.last_legal_analysis = result['claim_draft']
                            st.success("טיוטת התביעה הוכנה בהצלחה.")
                        elif result.get('legal_analysis'):
                            st.session_state.report_output_title = "### תוכן התביעה"
                            st.session_state.report_output_content = result['legal_analysis']
                            st.session_state.last_legal_analysis = result['legal_analysis']
                            st.success("טיוטת התביעה הוכנה בהצלחה.")
                        else:
                            st.info("ההכנה הסתיימה, אך לא התקבל תוכן להצגה.")
                except Exception as e:
                    st.error(f"שגיאה בהכנת התביעה: {str(e)}")

            # Button for violations list only
            if st.button("רשימת הפרות", type="primary", key="violations_list_btn"):
                st.session_state.report_output_title = None
                st.session_state.report_output_content = None
                st.session_state.summary_output_content = None
                st.session_state.last_legal_analysis = None
                st.session_state.qna_result = None
                st.session_state["qna_question_input"] = ""
                try:
                    with st.spinner("יוצר רשימת הפרות..."):
                        result = doc_processor.create_report_sync(
                            payslip_text=st.session_state.processed_result.get('payslip_text'),
                            contract_text=st.session_state.processed_result.get('contract_text'),
                            attendance_text=st.session_state.processed_result.get('attendance_text'),
                            type="violations_list",
                            context=context
                        )
                        if result.get('legal_analysis'):
                            st.session_state.report_output_title = "### רשימת הפרות שזוהו"
                            st.session_state.report_output_content = result['legal_analysis']
                            st.session_state.last_legal_analysis = result['legal_analysis']
                        else:
                            st.info("לא התקבל תוכן להצגת רשימת ההפרות.")
                except Exception as e:
                    st.error(f"שגיאה ביצירת רשימת ההפרות: {str(e)}")
                    
            # if st.button("ניתוח מקצועי", type="primary", key="professional_btn"):
            #     st.session_state.report_output_title = None
            #     st.session_state.report_output_content = None
            #     st.session_state.summary_output_content = None
            #     st.session_state.last_legal_analysis = None
            #     try:
            #         with st.spinner("מבצע ניתוח מקצועי..."):
            #             result = doc_processor.create_report_sync(
            #                 payslip_text=st.session_state.processed_result.get('payslip_text'),
            #                 contract_text=st.session_state.processed_result.get('contract_text'),
            #                 attendance_text=st.session_state.processed_result.get('attendance_text'),
            #                 type="professional",
            #                 context=context
            #             )
            #             if result.get('legal_analysis'):
            #                 st.session_state.report_output_title = "### תוצאות ניתוח מקצועי"
            #                 st.session_state.report_output_content = result['legal_analysis']
            #                 st.session_state.last_legal_analysis = result['legal_analysis']
            #     except Exception as e:
            #         st.error(f"שגיאה בניתוח: {str(e)}")
                    
            if st.button("מכתב התראה", type="primary", key="warning_letter_btn"):
                st.session_state.report_output_title = None
                st.session_state.report_output_content = None
                st.session_state.summary_output_content = None
                st.session_state.last_legal_analysis = None
                st.session_state.qna_result = None
                st.session_state["qna_question_input"] = ""
                try:
                    with st.spinner("מכין מכתב התראה..."):
                        result = doc_processor.create_report_sync(
                            payslip_text=st.session_state.processed_result.get('payslip_text'),
                            contract_text=st.session_state.processed_result.get('contract_text'),
                            attendance_text=st.session_state.processed_result.get('attendance_text'),
                            type="warning_letter",
                            context=context
                        )
                        if result.get('legal_analysis'):
                            st.session_state.report_output_title = "### מכתב התראה"
                            st.session_state.report_output_content = result['legal_analysis']
                            st.session_state.last_legal_analysis = result['legal_analysis']
                except Exception as e:
                    st.error(f"שגיאה בהכנת המכתב: {str(e)}")
        
        with col3:
            if st.button("הסבר פשוט", type="primary", key="easy_explanation_btn"):
                st.session_state.report_output_title = None
                st.session_state.report_output_content = None
                st.session_state.summary_output_content = None
                st.session_state.last_legal_analysis = None
                st.session_state.qna_result = None
                st.session_state["qna_question_input"] = ""
                try:
                    with st.spinner("מסביר בפשטות..."):
                        result = doc_processor.create_report_sync(
                            payslip_text=st.session_state.processed_result.get('payslip_text'),
                            contract_text=st.session_state.processed_result.get('contract_text'),
                            attendance_text=st.session_state.processed_result.get('attendance_text'),
                            type="easy",
                            context=context
                        )
                        if result.get('legal_analysis'):
                            st.session_state.report_output_title = "### הסבר פשוט"
                            st.session_state.report_output_content = result['legal_analysis']
                            st.session_state.last_legal_analysis = result['legal_analysis']
                except Exception as e:
                    st.error(f"שגיאה בהסבר: {str(e)}")

            if st.button("דוח תביעה סופי", type="primary", key="table_btn"):
                st.session_state.report_output_title = None
                st.session_state.report_output_content = None
                st.session_state.summary_output_content = None
                st.session_state.last_legal_analysis = None
                try:
                    with st.spinner("יוצר דוח תביעה סופי..."):
                        result = doc_processor.create_report_sync(
                            payslip_text=st.session_state.processed_result.get('payslip_text'),
                            contract_text=st.session_state.processed_result.get('contract_text'),
                            attendance_text=st.session_state.processed_result.get('attendance_text'),
                            type="table", 
                            context=context
                        )
                        if result.get('legal_analysis'):
                            st.session_state.report_output_title = "### דוח תביעה סופי"
                            st.session_state.report_output_content = result['legal_analysis']
                            st.session_state.last_legal_analysis = result['legal_analysis']
                        else:
                            st.info("לא התקבל תוכן להצגת דוח התביעה הסופי.")
                except Exception as e:
                    st.error(f"שגיאה ביצירת דוח התביעה הסופי: {str(e)}")

            if st.button("דו""ח משולב (מקיף)", type="primary", key="combined_btn"):
                st.session_state.report_output_title = None
                st.session_state.report_output_content = None
                st.session_state.summary_output_content = None
                st.session_state.last_legal_analysis = None
                try:
                    with st.spinner("יוצר דו""ח משולב (מקיף)..."):
                        result = doc_processor.create_report_sync(
                            payslip_text=st.session_state.processed_result.get('payslip_text'),
                            contract_text=st.session_state.processed_result.get('contract_text'),
                            attendance_text=st.session_state.processed_result.get('attendance_text'),
                            type="combined",
                            context=context
                        )
                        if result.get('legal_analysis'):
                            st.session_state.report_output_title = "### דו""ח משולב (מקיף)"
                            st.session_state.report_output_content = result['legal_analysis']
                            st.session_state.last_legal_analysis = result['legal_analysis']
                        else:
                            st.info("לא התקבל תוכן להצגת הדו""ח המשולב.")
                except Exception as e:
                    st.error(f"שגיאה ביצירת הדו\"ח המשולב: {str(e)}")

        # Display area for the report and its summary
        if st.session_state.get('report_output_title') and st.session_state.get('report_output_content'):
            st.markdown(st.session_state.report_output_title)
            st.markdown(st.session_state.report_output_content)

            # Display the "Summarize Results" button if there's content to summarize
            if st.session_state.get('last_legal_analysis'):
                if st.button("סכם תוצאות", key="summarise_btn_main_display"):
                    try:
                        with st.spinner("מסכם..."):
                            summary = doc_processor.summarise_sync(st.session_state.last_legal_analysis)
                            st.session_state.summary_output_content = summary
                    except Exception as e:
                        st.error(f"שגיאה בסיכום: {str(e)}")
                        st.session_state.summary_output_content = None # Clear summary on error
            
            # Display the summary if it has been generated and stored
            if st.session_state.get('summary_output_content'):

                st.markdown("### סיכום")
                st.markdown(st.session_state.summary_output_content)

            # --- QnA Section ---
            # Show QnA input below the summary button
            if st.session_state.get('last_legal_analysis'):
                st.markdown("---")
                st.markdown("#### שאל שאלה על הדוח שהופק")
                qna_question = st.text_input("הזן שאלה על הדוח", key="qna_question_input")
                if 'qna_result' not in st.session_state:
                    st.session_state.qna_result = None
                if st.button("שלח שאלה", key="qna_submit_btn"):
                    try:
                        with st.spinner("שולח שאלה..."):
                            result = doc_processor.qna_sync(
                                st.session_state.last_legal_analysis,
                                qna_question
                            )
                            st.session_state.qna_result = result
                    except Exception as e:
                        st.session_state.qna_result = None
                        st.error(f"שגיאה בשליחת השאלה: {str(e)}")
                # Show the QnA result if available
                if st.session_state.qna_result:
                    st.markdown("#### תשובה לשאלה:")
                    st.markdown(st.session_state.qna_result)

# Letter Format Management Tab
with tab3:
    st.subheader("📝 ניהול תבנית מכתב")
    
    # Get current format
    current_format = doc_processor.letter_format.get_format()
    
    # Edit letter format
    letter_format = st.text_area(
        "תבנית מכתב התראה",
        value=current_format.get("content", ""),
        height=300,
        help="הכנס את תבנית מכתב ההתראה כאן. ניתן להשתמש בתגיות מיוחדות שיוחלפו בערכים בפועל."
    )
    
    # Save button
    if st.button("שמור תבנית", type="primary"):
        try:
            doc_processor.letter_format.update_format(letter_format)
            st.success("תבנית המכתב נשמרה בהצלחה!")
        except Exception as e:
            st.error(f"שגיאה בשמירת התבנית: {str(e)}")
            

# Law Management Tab
with tab2:
    # Clear the new law input if a law was just added
    if st.session_state.get("law_added"):
        st.session_state["new_law_text_area"] = ""
        st.session_state["law_added"] = False
    st.subheader("📚 ניהול חוקי עבודה")
    
    # Add new law section
    with st.expander("הוסף חוק עבודה חדש", expanded=False):
        new_law = st.text_area("הכנס טקסט של חוק עבודה חדש", height=150, key="new_law_text_area")
        if st.button("הוסף חוק", type="primary", key="add_law"):
            if new_law.strip():
                try:
                    doc_processor.law_storage.add_law(new_law)
                    st.success("החוק נוסף בהצלחה!")
                    st.session_state["law_added"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"שגיאה בהוספת החוק: {str(e)}")
            else:
                st.warning("אנא הכנס טקסט של חוק לפני הוספה.")

    # Display existing laws, sorted by created_at descending (most recent first)
    st.subheader("חוקי עבודה קיימים")
    existing_laws = doc_processor.law_storage.get_all_laws()
    # Sort by created_at descending if available
    if existing_laws and 'created_at' in existing_laws[0]:
        existing_laws = sorted(existing_laws, key=lambda x: x.get('created_at', ''), reverse=True)

    if not existing_laws:
        st.info("טרם נוספו חוקי עבודה.")
    else:
        for law in existing_laws:
            if f"editing_{law['id']}" not in st.session_state:
                st.session_state[f"editing_{law['id']}"] = False
                st.session_state[f"edited_text_{law['id']}"] = law["text"]
            
            if st.session_state[f"editing_{law['id']}"]:
                edited_text = st.text_area(
                    "ערוך טקסט חוק",
                    value=st.session_state[f"edited_text_{law['id']}"],
                    height=100,
                    key=f"edit_law_{law['id']}"
                )
                if st.button("שמור", key=f"save_{law['id']}"):
                    try:
                        new_summary = doc_processor.law_storage.update_law(law["id"], edited_text)
                        st.session_state[f"editing_{law['id']}"] = False
                        if new_summary:
                            st.session_state[f"edited_summary_{law['id']}"] = new_summary
                            st.success("החוק עודכן בהצלחה! הסיכום החדש:")
                            st.info(new_summary)
                        else:
                            st.success("החוק עודכן בהצלחה!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"שגיאה בעדכון החוק: {str(e)}")
            else:
                # Display summary if available
                if law.get("summary"):
                    st.text_area(
                        "סיכום החוק",
                        value=law["summary"],
                        height=68,
                        key=f"law_summary_{law['id']}",
                        disabled=True
                    )
                
                st.text_area(
                    "טקסט החוק",
                    value=law["text"],
                    height=100,
                    key=f"law_{law['id']}",
                    disabled=True
                )
                
            # Action buttons below text area
            button_cols = st.columns([1, 1])
            with button_cols[0]:
                if not st.session_state[f"editing_{law['id']}"]:
                    if st.button("✏️ ערוך", key=f"edit_{law['id']}", use_container_width=True):
                        st.session_state[f"editing_{law['id']}"] = True
                        st.session_state[f"edited_text_{law['id']}"] = law["text"]
                        st.rerun()
            with button_cols[1]:
                if st.button("🗑️ מחק", key=f"delete_{law['id']}", use_container_width=True):
                    if doc_processor.law_storage.delete_law(law["id"]):
                        st.success("החוק נמחק בהצלחה!")
                        st.rerun()
                    else:
                        st.error("שגיאה במחיקת החוק.")


# Judgement Management Tab
with tab4:
    # Clear the new judgement input if a judgement was just added
    if st.session_state.get("judgement_added"):
        st.session_state["new_judgement_text_area"] = ""
        st.session_state["judgement_added"] = False
    st.subheader("⚖️ ניהול פסקי דין")

    # Add new judgement section
    with st.expander("הוסף פסק דין חדש", expanded=False):
        new_judgement_text = st.text_area("הכנס טקסט של פסק דין חדש", height=150, key="new_judgement_text_area")
        if st.button("הוסף פסק דין", type="primary", key="add_judgement_button"):
            if new_judgement_text.strip():
                try:
                    # Assuming doc_processor has a judgement_storage attribute
                    doc_processor.judgement_storage.add_judgement(new_judgement_text)
                    st.success("פסק הדין נוסף בהצלחה!")
                    st.session_state["judgement_added"] = True
                    st.rerun()
                except AttributeError:
                    st.error("שגיאה: judgement_storage אינו מוגדר ב-doc_processor.")
                except Exception as e:
                    st.error(f"שגיאה בהוספת פסק הדין: {str(e)}")
            else:
                st.warning("אנא הכנס טקסט של פסק דין לפני הוספה.")

    # Display existing judgements
    st.subheader("פסקי דין קיימים")
    try:
        # Assuming doc_processor has a judgement_storage attribute
        existing_judgements = doc_processor.judgement_storage.get_all_judgements()
        # Sort by created_at descending if available
        if existing_judgements and 'created_at' in existing_judgements[0]:
            existing_judgements = sorted(existing_judgements, key=lambda x: x.get('created_at', ''), reverse=True)

        if not existing_judgements:
            st.info("טרם נוספו פסקי דין.")
        else:
            for judgement in existing_judgements:
                judgement_id = judgement['id']
                if f"editing_judgement_{judgement_id}" not in st.session_state:
                    st.session_state[f"editing_judgement_{judgement_id}"] = False
                    st.session_state[f"edited_judgement_text_{judgement_id}"] = judgement["text"]
                
                if st.session_state[f"editing_judgement_{judgement_id}"]:
                    edited_text = st.text_area(
                        "ערוך טקסט פסק דין",
                        value=st.session_state[f"edited_judgement_text_{judgement_id}"],
                        height=100,
                        key=f"edit_judgement_text_area_{judgement_id}"
                    )
                    if st.button("שמור שינויים", key=f"save_judgement_{judgement_id}"):
                        try:
                            doc_processor.judgement_storage.update_judgement(judgement_id, edited_text)
                            st.session_state[f"editing_judgement_{judgement_id}"] = False
                            st.success("פסק הדין עודכן בהצלחה!")
                            st.rerun()
                        except AttributeError:
                            st.error("שגיאה: judgement_storage אינו מוגדר ב-doc_processor.")
                        except Exception as e:
                            st.error(f"שגיאה בעדכון פסק הדין: {str(e)}")
                else:
                    st.text_area(
                        "טקסט פסק הדין",
                        value=judgement["text"],
                        height=100,
                        key=f"display_judgement_text_area_{judgement_id}",
                        disabled=True
                    )
                    
                # Action buttons below text area
                judgement_button_cols = st.columns([1, 1])
                with judgement_button_cols[0]:
                    if not st.session_state[f"editing_judgement_{judgement_id}"]:
                        if st.button("✏️ ערוך פסק דין", key=f"edit_judgement_button_{judgement_id}", use_container_width=True):
                            st.session_state[f"editing_judgement_{judgement_id}"] = True
                            st.session_state[f"edited_judgement_text_{judgement_id}"] = judgement["text"]
                            st.rerun()
                with judgement_button_cols[1]:
                    if st.button("🗑️ מחק פסק דין", key=f"delete_judgement_button_{judgement_id}", use_container_width=True):
                        try:
                            if doc_processor.judgement_storage.delete_judgement(judgement_id):
                                st.success("פסק הדין נמחק בהצלחה!")
                                st.rerun()
                            else:
                                st.error("שגיאה במחיקת פסק הדין.")
                        except AttributeError:
                            st.error("שגיאה: judgement_storage אינו מוגדר ב-doc_processor.")
                        except Exception as e:
                            st.error(f"שגיאה במחיקת פסק הדין: {str(e)}")
    except AttributeError:
        st.error("לא ניתן לטעון פסקי דין: judgement_storage אינו מוגדר ב-doc_processor.")
    except Exception as e:
        st.error(f"שגיאה בטעינת פסקי דין: {str(e)}")

# Add footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>מופעל על ידי בינה מלאכותית ומאגר חוקי עבודה</div>",
    unsafe_allow_html=True
)
