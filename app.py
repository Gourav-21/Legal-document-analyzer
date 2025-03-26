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

# Title and description
st.title("📄 מנתח מסמכים משפטיים")
st.markdown("העלה את המסמכים המשפטיים שלך לבדיקת תאימות לחוקי העבודה הישראליים.")

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

# Add process button
if (payslip_files or contract_files) and st.button("נתח מסמכים", type="primary"):
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
            result = doc_processor.process_document(all_files, all_doc_types)
            
            # Display results
            if result.get('legal_analysis'):
                analysis = result['legal_analysis']
                st.markdown("### תוצאות ניתוח משפטי")
                st.markdown(analysis)
                
    except Exception as e:
        st.error(f"שגיאה בעיבוד המסמכים: {str(e)}")
        st.info("אנא ודא שהמסמכים בפורמט הנכון ונסה שוב.")

st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

# Add Law Management Section
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
            st.warning("אנא הכנס טקסט של חוק לפני ההוספה.")

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
        # st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>מופעל על ידי בינה מלאכותית ומאגר חוקי עבודה</div>",
    unsafe_allow_html=True
)