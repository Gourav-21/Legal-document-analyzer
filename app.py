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

# Initialize DocumentProcessor
doc_processor = DocumentProcessor()

# Set page config
st.set_page_config(
    page_title="מנתח מסמכים משפטיים",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.session_state.processed_result = None

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
                    st.error(f"שגיאה ביצירת הדו""ח המשולב: {str(e)}")

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
