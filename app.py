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
tab1, tab2, tab3 = st.tabs(["📄 ניתוח מסמכים", "📚 ניהול חוקים", "📝 ניהול תבנית"])

# Document Analysis Tab
with tab1:
    # Create document upload sections
    col1, col2 = st.columns(2)

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

# Add context input field
context = st.text_area("הקשר נוסף או הערות מיוחדות", height=100)

# Process documents button
if (payslip_files or contract_files) and st.button("עבד מסמכים", type="primary"):
    try:
        with st.spinner("מעבד מסמכים..."):
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
            st.session_state.processed_result = doc_processor.process_document(all_files, all_doc_types,True)
            st.success("המסמכים עובדו בהצלחה! כעת תוכל לבחור סוג ניתוח.")
            
    except Exception as e:
        st.error(f"שגיאה בעיבוד המסמכים: {str(e)}")
        st.info("אנא ודא שהמסמכים בפורמט הנכון ונסה שוב.")

# Analysis buttons (only shown after processing)
if st.session_state.processed_result:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("דוח ניתוח משפטי", type="primary", key="report_btn"):
            try:
                with st.spinner("מנתח..."):
                    result = doc_processor.create_report(
                        st.session_state.processed_result.get('payslip_text'),
                        st.session_state.processed_result.get('contract_text'),
                        type="report",
                        context=context
                    )
                    if result.get('legal_analysis'):
                        st.markdown("### תוצאות ניתוח משפטי")
                        st.markdown(result['legal_analysis'])
            except Exception as e:
                st.error(f"שגיאה בניתוח: {str(e)}")
                
        if st.button("ניתוח כדאיות כלכלית", type="primary", key="profitability_btn"):
            try:
                with st.spinner("מנתח כדאיות כלכלית..."):
                    result = doc_processor.create_report(
                        st.session_state.processed_result.get('payslip_text'),
                        st.session_state.processed_result.get('contract_text'),
                        type="profitability",
                        context=context
                    )
                    if result.get('legal_analysis'):
                        st.markdown("### תוצאות ניתוח כדאיות כלכלית")
                        st.markdown(result['legal_analysis'])
            except Exception as e:
                st.error(f"שגיאה בניתוח: {str(e)}")
    
    with col2:
        if st.button("ניתוח מקצועי", type="primary", key="professional_btn"):
            try:
                with st.spinner("מבצע ניתוח מקצועי..."):
                    result = doc_processor.create_report(
                        st.session_state.processed_result.get('payslip_text'),
                        st.session_state.processed_result.get('contract_text'),
                        type="professional",
                        context=context
                    )
                    if result.get('legal_analysis'):
                        st.markdown("### תוצאות ניתוח מקצועי")
                        st.markdown(result['legal_analysis'])
            except Exception as e:
                st.error(f"שגיאה בניתוח: {str(e)}")
                
        if st.button("מכתב התראה", type="primary", key="warning_letter_btn"):
            try:
                with st.spinner("מכין מכתב התראה..."):
                    result = doc_processor.create_report(
                        st.session_state.processed_result.get('payslip_text'),
                        st.session_state.processed_result.get('contract_text'),
                        type="warning_letter",
                        context=context
                    )
                    if result.get('legal_analysis'):
                        st.markdown("### מכתב התראה")
                        st.markdown(result['legal_analysis'])
            except Exception as e:
                st.error(f"שגיאה בהכנת המכתב: {str(e)}")
    
    with col3:
        if st.button("הסבר פשוט", type="primary", key="easy_explanation_btn"):
            try:
                with st.spinner("מסביר בפשטות..."):
                    result = doc_processor.create_report(
                        st.session_state.processed_result.get('payslip_text'),
                        st.session_state.processed_result.get('contract_text'),
                        type="easy",
                        context=context
                    )
                    if result.get('legal_analysis'):
                        st.markdown("### הסבר פשוט")
                        st.markdown(result['legal_analysis'])
            except Exception as e:
                st.error(f"שגיאה בהסבר: {str(e)}")

# Law Management Tab
with tab2:
    st.subheader("📚 ניהול חוקי עבודה")
    
    # Add new law section
    with st.expander("הוסף חוק עבודה חדש", expanded=False):
        new_law = st.text_area("הכנס טקסט של חוק עבודה חדש", height=150)
        if st.button("הוסף חוק", type="primary", key="add_law"):
            if new_law.strip():
                try:
                    doc_processor.law_storage.add_law(new_law)
                    st.success("החוק נוסף בהצלחה!")
                except Exception as e:
                    st.error(f"שגיאה בהוספת החוק: {str(e)}")
            else:
                st.warning("אנא הכנס טקסט של חוק לפני הוספה.")

    # Display existing laws
    st.subheader("חוקי עבודה קיימים")
    existing_laws = doc_processor.law_storage.get_all_laws()

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
                        doc_processor.law_storage.update_law(law["id"], edited_text)
                        st.session_state[f"editing_{law['id']}"] = False
                        st.success("החוק עודכן בהצלחה!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"שגיאה בעדכון החוק: {str(e)}")
            else:
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

# Add footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>מופעל על ידי בינה מלאכותית ומאגר חוקי עבודה</div>",
    unsafe_allow_html=True
)