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
        # payslip_text = "Payslip 1:\nIncome-TAX-Deduction\nHealth insurancet\nPension contribution\nNet salary\n211030000\n5:00\n905328\n10:07 000\n00\n42\n00\n00\n00\n01\n12\n00\n03\nמס הכנסה\nק.השתלוארג בינ\"ל\nבטוח לאומי.\nבשוח בריאות\nדמי ניהול פנסיה ת\nקופת גמל רשף\n185467\n50589\n59571\n55491\n24090\n4002\nסה\"כ ניכויים 379210\nסכום\nבננק\nהתחלה של\n#730\n1.00000\n0.10000\nHop T\n3300\nING OLD\nנתוני עזר\n...\n...\nyaw ng\n2 TRDN\n6\n003\n40 3 6\n31 40 3\n176\na uud\n210\nתשלומים\nphen DOR\n1095000\n109500\n61600\n3038\n15400\n(DVT)\nמשולב אופק חדש\nגמול חינוך כיתה\nקצובת נסיעה\nהחזר טלפון\nמק נסיעות בתפק\nסה\"כ תשלומים ש 1284538\n1000\n15\n1522700\n2\n500\n10:0\nUQLU\n60 01 30 09 2008\n70\n89\n01 30 09 2008\nברוטו מצטבר למם.\n10859963\n9\n510300\nתאום מס הכנסה\nבינל\nקרן השת\n420\nGress Income - TAX - Deduction Health insurancet Pension contribution Net salary 211030000 5:00 905328 10:07 000 00 42 00 00 00 01 12 00 03 מס הכנסה ק.השתלוארג בינ\"ל בטוח לאומי . בשוח בריאות דמי ניהול פנסיה ת קופת גמל רשף 185467 50589 59571 55491 24090 4002 סה\"כ ניכויים 379210 סכום בננק התחלה של # 730 1.00000 0.10000 Hop T 3300 ING OLD נתוני עזר ... ... yaw ng 2 TRDN 6 003 40 3 6 31 40 3 176 a uud 210 תשלומים phen DOR 1095000 109500 61600 3038 15400 ( DVT ) משולב אופק חדש גמול חינוך כיתה קצובת נסיעה החזר טלפון מק נסיעות בתפק סה\"כ תשלומים ש 1284538 1000 15 1522700 2 500 10 : 0 UQLU 60 01 30 09 2008 70 89 01 30 09 2008 ברוטו מצטבר למם . 10859963 9 510300 תאום מס הכנסה בינל קרן השת 420 Gress\n\nPayslip 2:\n01/06/2017\n01/06/2017\nDN\nWork Days\n186.000 178.00\n22.00\npm me\nCLU\n21.00 TOY SO\n26.88 ve 'un\n21.20\n164.29\n287.08\n000\nGRoss salary\nNet Salery\n5.299.68\n472.37\n4,827.11\n#.827.11\n.ULU\n1.50\n0.42\nMROCL\n0\nתלוש משכורת 6/17\nundi au S\n1055/7 noon 00\nלופים\n02 CLU\n572\nCOLO\nGC LACY LO\n103120\n197\nל לתשלום\n1207\njam\n2.73@ bo\n00000000-0\n// 000\n4,784.64\n300.00\n215.04\nAZY!\nSPYY\non'\nGOLL G Anu\nGUB\n0\nco.il\nשראלי ישראלה\nMAIL umud\ncanu\ngown\n26.88\n300.00\n215.04\n1.00\npop on\nang po sa\n13097\n4.785.00\n4.784.07\n0.00\n287.60\n0.00\n207 08\n0.00\n00\nWAIL\n6.336\nALUME\n1000\n6.30\nromanas\nWILL!\nnus ways.\n100\nmye)\nnwotn\n130x0\n5.300.00\n5,299.89\n21.00\n164.00\n287.03\n25% *\n30% T**\n>the 7\n47 wo\n$300.00\n5.300.00\n$00.00\n581.25\n100.48\nALU\n900 129\nwhen\n470121\nCRM\n2.70\n#37 on\n** %\nPO D\nברעות\nHealth-Insurance\n06/09/17 200 1\nACL CUCCIO QUILA: 00'000'S act .00 Uhu: 88 92\nadd 14 ©2001 LIXU ual ca\nCocu MUDELOM T 01/06/2017 01/06/2017 DN Work Days 186.000 178.00 22.00 pm me CLU 21.00 TOY SO 26.88 ve ' un 21.20 164.29 287.08 000 GRoss salary Net Salery 5.299.68 472.37 4,827.11 # .827.11 .ULU 1.50 0.42 MROCL 0 תלוש משכורת 6/17 undi au S 1055/7 noon 00 לופים 02 CLU 572 COLO GC LACY LO 103120 197 ל לתשלום 1207 jam 2.73 @ bo 00000000-0 // 000 4,784.64 300.00 215.04 AZY ! SPYY on ' GOLL G Anu GUB 0 co.il שראלי ישראלה MAIL umud canu gown 26.88 300.00 215.04 1.00 pop on ang po sa 13097 4.785.00 4.784.07 0.00 287.60 0.00 207 08 0.00 00 WAIL 6.336 ALUME 1000 6.30 romanas WILL ! nus ways . 100 mye ) nwotn 130x0 5.300.00 5,299.89 21.00 164.00 287.03 25 % * 30 % T ** > the 7 47 wo $ 300.00 5.300.00 $ 00.00 581.25 100.48 ALU 900 129 when 470121 CRM 2.70 # 37 on ** % PO D ברעות Health - Insurance 06/09/17 200 1 ACL CUCCIO QUILA : 00'000'S act .00 Uhu : 88 92 add 14 © 2001 LIXU ual ca Cocu MUDELOM T\n\nPayslip 3:\nHealth - Insurance\nwork Days.\nשם החברה\nבי אל פי אלקטרוניקה בע\"מ\nמס' עובד\nמחלקה\n3\n0\nעזאגווי\nתעריף\nתעריף יום\n6,000.00\n276.88\n32.97\nתאור התשלום\nכמות\nשכר יסוד\n1.00\nאינפורמטיבי - שכר מינימום לחודש\nאינפורמטיבי- שכר מינימום לשעה\nניכויי חובה\nביטוח לאומי\nדמי בריאות\nסה\"כ\nתלוש משכורת לחודש\n1/2020\nכתובת: נחלת הר חב\"ד 144, קריית מלאכי 8303300\nתיק ניכויים\nב.ל\nמ.ה\n951617638\n951617638\nמספר תאגיד - 515919595\nשם עובד\nזלמן שמעון\nתעודת זהות\nתת מחלקה\nדרוג\nדרגה\nות\nתחילת עבודה\n302762729\n01/08/19\nתעריף שעה ימי עבודה שעות עבודה בנק\nסניף\nחשבון\n21.67\n182\nתעריף\nאחוז\nנטו לגילום\nמשולם לעובד\nסכום התשלום\n6.000.00\n6,000.00\nפיצויים חודשי\nפיצויים פטור\nפיצויים ותיקה\nשכר לפיצויים\nקופ\"ג מעביד-חודשי 0.00\nשכר לקופ\"ג\n0.00\nקה\"ל מעביד-חודשי :0.00\nשכר לקהיל\n0.00:\nמצב\n]\nבן זוג\nנושפחתי עובד\nג\n. זיכוי\n4.75\nמס שולי % מס קבוע\n0.00 :\n0.00 :\n0.00 :\n0.00\nאחוז משרה\nזיכוי אישי\n10.00\nזיכוי כסף\nזיכוי גמל\n1,040\nיטי משמרות\nפטור חודשי\nפטר ס 47\nהנחת\nישובי פיתות\nתאום\nל\nשכר לתאום\nמ.ה. לתאום\nחייב מ.ה.\nחייב ב.ל.\n5300\n29.12\nימי תקן\nשעות תקן\n21.67\n182.00\nסה\"כ תשלומים\nהסכום\nניכויי רשות\nיתרה\nכמות\nסכום\n24.00\n186.00\n210.00\nסה\"כ\nהערות:\nניהול העדרויות\nיתרה קודמת\nסוג העדחת\nניצול\nיתרה\nתשלומים\nשכר שווה כסף\nחופש\nחייב מה.\nמס הטסה\nבטוח לאומי\nחודשי\nעבודה\n1 2 3 4 5 6 7 8 91 1 1\nבוצע עייל: אבינועם משה - רואה חשבון\nגמל 35%\nג השתלמות\n6,000\n6,000\n210\nנתונים מצטברים\nכיכוי ס 47\nזיכוי אישי\n1,040\nזיכוי נוסף\nזיכוי גמל\nדמי חבר טיפול\nחייב ב.ל.\nקופיג מעביד\nקהיל מעביר\nזיכוי משמרות\nפיצויים מעסיק\nפטור\nפטור 476\nבתאריך\n0\n05/02/2020\nסה\"כ ניכויים\nשכר נטו\nנטו לתשלום\n6,000\n0.00\n0.00\n0.00\n6,000.00\n6,000.00\n6,000.00\n210.00\n5,790.00\n5,790.00\nבאמצעות שיקלולית מבית ט.מ.ל. - תוכנת השכר המובילה במדינה\nGROSS-22\nNet Salary Health - Insurance work Days . שם החברה בי אל פי אלקטרוניקה בע\"מ מס ' עובד מחלקה 3 0 עזאגווי תעריף תעריף יום 6,000.00 276.88 32.97 תאור התשלום כמות שכר יסוד 1.00 אינפורמטיבי - שכר מינימום לחודש אינפורמטיבי- שכר מינימום לשעה ניכויי חובה ביטוח לאומי דמי בריאות סה\"כ תלוש משכורת לחודש 1/2020 כתובת : נחלת הר חב\"ד 144 , קריית מלאכי 8303300 תיק ניכויים ב.ל מ.ה 951617638 951617638 מספר תאגיד - 515919595 שם עובד זלמן שמעון תעודת זהות תת מחלקה דרוג דרגה ות תחילת עבודה 302762729 01/08/19 תעריף שעה ימי עבודה שעות עבודה בנק סניף חשבון 21.67 182 תעריף אחוז נטו לגילום משולם לעובד סכום התשלום 6.000.00 6,000.00 פיצויים חודשי פיצויים פטור פיצויים ותיקה שכר לפיצויים קופ\"ג מעביד - חודשי 0.00 שכר לקופ\"ג 0.00 קה\"ל מעביד - חודשי : 0.00 שכר לקהיל 0.00 : מצב ] בן זוג נושפחתי עובד ג . זיכוי 4.75 מס שולי % מס קבוע 0.00 : 0.00 : 0.00 : 0.00 אחוז משרה זיכוי אישי 10.00 זיכוי כסף זיכוי גמל 1,040 יטי משמרות פטור חודשי פטר ס 47 הנחת ישובי פיתות תאום ל שכר לתאום מ.ה. לתאום חייב מ.ה. חייב ב.ל. 5300 29.12 ימי תקן שעות תקן 21.67 182.00 סה\"כ תשלומים הסכום ניכויי רשות יתרה כמות סכום 24.00 186.00 210.00 סה\"כ הערות : ניהול העדרויות יתרה קודמת סוג העדחת ניצול יתרה תשלומים שכר שווה כסף חופש חייב מה . מס הטסה בטוח לאומי חודשי עבודה 1 2 3 4 5 6 7 8 91 1 1 בוצע עייל : אבינועם משה - רואה חשבון גמל 35 % ג השתלמות 6,000 6,000 210 נתונים מצטברים כיכוי ס 47 זיכוי אישי 1,040 זיכוי נוסף זיכוי גמל דמי חבר טיפול חייב ב.ל. קופיג מעביד קהיל מעביר זיכוי משמרות פיצויים מעסיק פטור פטור 476 בתאריך 0 05/02/2020 סה\"כ ניכויים שכר נטו נטו לתשלום 6,000 0.00 0.00 0.00 6,000.00 6,000.00 6,000.00 210.00 5,790.00 5,790.00 באמצעות שיקלולית מבית ט.מ.ל. - תוכנת השכר המובילה במדינה GROSS - 22 Net Salary\n\nPayslip 4:\nWork Deys\nשם החברה\nהמשש השקיות בלם\nת.ד 8\nמס' עובד\nמחלקה\nתלוש משכורת לחודש 5/2024\nכתובת:\nנחל ששברת:2, אילת\nמחלקה: נמוי מינימום\nשם עובד\nתעודת זהות\nתת מחלקה\nדרוג\nדרגה\n1.70\n4\nתעריף\nתעריף יום\nתעריף שעה ימי עבודה שעות עבודה\nבנק\nסניף\nחשבון\nפיצויים חודשי\n6,500.00\n250\n32.5\n26\n0/200\n12\n763\nפיצויים פטור\nמשולמות / בפועל\nמשולם לעובד\n03/04/24%\n0.00:\n0.00 :\n29\n1\nב.ל\nמ.ה\nתיק ניכויים\n940095489\nמספר תאגיד - 514730795\nוותק תחילת עבודה\nחש\nתאור התשלום\nכמות\nתעריף\nאחוז\nנטו לגילום\nפיצויים ותיקה\nסכום התשלום\nשכר יסוד\nימי חג\nשעות נוספות גלובליות\nשעות שבת\n6,500.00\n1.00\n250.00\n1.00\n500.00\n1.00\n1,500.00\n1.00\nשכר לפיצויים\n6,500.00\n250.00\n500.00\n1,500.00\nקופ\"ג מעסיק-חודשי\nשכר לקופייג\nשכר לקהייל\nקהיל מעסיק-חודשי\nמצב\nבן זוג\nמשפחתין עובד\nנכ\nנ. זיכוי\n2.25\nמס שולי % מס קבוע\n14.00\nזיכוי נוסף\n0.00\nאחוז משרה\nזיכוי אישי\n544\nזיכוי גמל זיכוי משמרות\n0.00:\n0.00\n0.00\n0.00\n0.00:\novertime Hours\nאינפורמטיבי - שכר מינימום לחודש\nאינפורמטיבי - שכר מינימום לשעה\nניכויי חובה\nביטוח לאומי\nדמי בריאות\nפטור חודשי\nפטור ס' 47\nהנחת\nישובי פיתוח\n875\nשכר לתאום\nמ.ה. לתאום\nתאום\nמס\nל\nחייב מ.ה.\nחייב ב.ל.\n5880.02\n32.3\n:\n:\nימי תקן\nשעות תקן\n26.00\n200.00\n„Gross Balary\nסה\"כ תשלומים\nהסכום\nניכויי רשות\nיתרה\nכמות\nסכומ\n116.00\n295.00\nHealth Insurance\nסה\"כ\n411.00\nהערות:\nניהול העדרויות\nסוג העדרות יתרה קודמת\nניצול\nיתרה\nחופש\n0\n1.17\n2.33\nמחלה\nצבירת חופש\nצבירת מחלה\n0\n0\n1.17:\n1.5:\n1.5\nסה\"כ\nתשלומים\nשכר שווה כסף\nחייב מ.ה.\nמס הכנסה\nביטוח לאומי\nחודשי\nע ב ו ד ה\n1 2 3 4 5 6 7 8 9 101112\nגמל 35%\nק. השתלמות\nכ כ ל ל ל\n18,850\n300\n19,150\n1,020\nסה\"כ ניכויים\nNet salary\nנתונים מצטברים\nניכוי ס' 47\nדמי חבר/טיפול\nזיכוי אישי\nזיכוי נוסף\nזיכוי גמל\nזיכוי משמרות\n1,089\nחייב ב.ל.\nקופי'ג מעסיק\nקהייל מעסיק\nפיצויים מעסיק\nפטור\nפטור ס' 47\n0\n0\nבתאריך 09/06/2024\nשכר נטו\nנטן לתשלום\n19,150\n0.00\n0.00\n0.00\n8,750.00\n8,750.00\n8,750.00\n411.00\n8,339.00\n8,339.00 Work Deys שם החברה המשש השקיות בלם ת.ד 8 מס ' עובד מחלקה תלוש משכורת לחודש 5/2024 כתובת : נחל ששברת : 2 , אילת מחלקה : נמוי מינימום שם עובד תעודת זהות תת מחלקה דרוג דרגה 1.70 4 תעריף תעריף יום תעריף שעה ימי עבודה שעות עבודה בנק סניף חשבון פיצויים חודשי 6,500.00 250 32.5 26 0/200 12 763 פיצויים פטור משולמות / בפועל משולם לעובד 03 / 04 / 24 % 0.00 : 0.00 : 29 1 ב.ל מ.ה תיק ניכויים 940095489 מספר תאגיד - 514730795 וותק תחילת עבודה חש תאור התשלום כמות תעריף אחוז נטו לגילום פיצויים ותיקה סכום התשלום שכר יסוד ימי חג שעות נוספות גלובליות שעות שבת 6,500.00 1.00 250.00 1.00 500.00 1.00 1,500.00 1.00 שכר לפיצויים 6,500.00 250.00 500.00 1,500.00 קופ\"ג מעסיק - חודשי שכר לקופייג שכר לקהייל קהיל מעסיק - חודשי מצב בן זוג משפחתין עובד נכ נ . זיכוי 2.25 מס שולי % מס קבוע 14.00 זיכוי נוסף 0.00 אחוז משרה זיכוי אישי 544 זיכוי גמל זיכוי משמרות 0.00 : 0.00 0.00 0.00 0.00 : overtime Hours אינפורמטיבי - שכר מינימום לחודש אינפורמטיבי - שכר מינימום לשעה ניכויי חובה ביטוח לאומי דמי בריאות פטור חודשי פטור ס ' 47 הנחת ישובי פיתוח 875 שכר לתאום מ.ה. לתאום תאום מס ל חייב מ.ה. חייב ב.ל. 5880.02 32.3 : : ימי תקן שעות תקן 26.00 200.00 „ Gross Balary סה\"כ תשלומים הסכום ניכויי רשות יתרה כמות סכומ 116.00 295.00 Health Insurance סה\"כ 411.00 הערות : ניהול העדרויות סוג העדרות יתרה קודמת ניצול יתרה חופש 0 1.17 2.33 מחלה צבירת חופש צבירת מחלה 0 0 1.17 : 1.5 : 1.5 סה\"כ תשלומים שכר שווה כסף חייב מ.ה. מס הכנסה ביטוח לאומי חודשי ע ב ו ד ה 1 2 3 4 5 6 7 8 9 101112 גמל 35 % ק . השתלמות כ כ ל ל ל 18,850 300 19,150 1,020 סה\"כ ניכויים Net salary נתונים מצטברים ניכוי ס ' 47 דמי חבר / טיפול זיכוי אישי זיכוי נוסף זיכוי גמל זיכוי משמרות 1,089 חייב ב.ל. קופי'ג מעסיק קהייל מעסיק פיצויים מעסיק פטור פטור ס ' 47 0 0 בתאריך 09/06/2024 שכר נטו נטן לתשלום 19,150 0.00 0.00 0.00 8,750.00 8,750.00 8,750.00 411.00 8,339.00 8,339.00\n\nPayslip 5:\nIn Hebrew\nשכר ברוטו\nניכוי מס הכנסה\nהפרשה לפנסיה.\nביטוח בריאות\nשכר נטר\nימי עבודה\nשעות נוספות\nDescription in English\nGroes Salary\nIncome Tax Deduction\nPension Contribution\nHealth Insurance\nNet Salary\nWork: Days\nOvertime Hours\nvalue/ ערך\nשיח 5,200\nש\"ח 300\n6%\nש\"ח 150\nש\"ח 4,500\n22\n0 In Hebrew שכר ברוטו ניכוי מס הכנסה הפרשה לפנסיה . ביטוח בריאות שכר נטר ימי עבודה שעות נוספות Description in English Groes Salary Income Tax Deduction Pension Contribution Health Insurance Net Salary Work : Days Overtime Hours value / ערך שיח 5,200 ש\"ח 300 6 % ש\"ח 150 ש\"ח 4,500 22 0\n\nPayslip 6:\nתלוש שכר לחודש 11/2024\nפרטי המעסיק :עסקים בע\"מ\nפרטי העובד :ישראל ישראלי ת.ז :123456789 כתובת :שלום לכם\n:פירוט תשלומים\nשכר יסוד :6000.00 -\n₪ דמי נסיעות :250.00 -\nשעות נוספות 12 שעות :(0.00 -\n₪ סה\"כ ברוטו :6250.00\nפירוט ניכויים והפרשות\n₪ 390.00 :(6.5%) הפרשה לפנסיה -\n₪ 186.00 :(3.1%) ביטוח בריאות\n₪ 420.00 :(7%) ביטוח לאומי -\nסה\"כ ניכויים :996.00\n₪ סה\"כ לתשלום נטו ** :5254.00** תלוש שכר לחודש 11/2024 פרטי המעסיק : עסקים בע\"מ פרטי העובד : ישראל ישראלי ת.ז : 123456789 כתובת : שלום לכם : פירוט תשלומים שכר יסוד : 6000.00 - ₪ דמי נסיעות : 250.00 - שעות נוספות 12 שעות : ( 0.00 - ₪ סה\"כ ברוטו : 6250.00 פירוט ניכויים והפרשות ₪ 390.00 : ( 6.5 % ) הפרשה לפנסיה - ₪ 186.00 : ( 3.1 % ) ביטוח בריאות ₪ 420.00 : ( 7 % ) ביטוח לאומי - סה\"כ ניכויים : 996.00 ₪ סה\"כ לתשלום נטו ** : 5254.00 **\n\nPayslip 7:\nתלוש שכר לחודש 12/2024\nפרטי המעסיק :עסקים בע''מ\nפרטי העובד :ישראל ישראלי ת.ז :123456789,כתובת :שלום לכם\n:פירוט תשלומים\nשכר יסוד :6000.00 -\nדמי נסיעות :250.00 -\n-\nשעות נוספות )12 שעות :(600.00 -\nסה\"כ ברוטו :6850.00\n:פירוט ניכויים והפרשות\n₪ 0.00 :(6.5%) הפרשה לפנסיה -\n₪ 186.00 :(3.1%) ביטוח בריאות -\n₪ 420.00 :(7%) ביטוח לאומי -\nסה\"כ ניכויים :606.00\n₪ סה\"כ לתשלום נטו ** :6244.00** תלוש שכר לחודש 12/2024 פרטי המעסיק : עסקים בע''מ פרטי העובד : ישראל ישראלי ת.ז : 123456789 , כתובת : שלום לכם : פירוט תשלומים שכר יסוד : 6000.00 - דמי נסיעות : 250.00 - - שעות נוספות ) 12 שעות : ( 600.00 - סה\"כ ברוטו : 6850.00 : פירוט ניכויים והפרשות ₪ 0.00 : ( 6.5 % ) הפרשה לפנסיה - ₪ 186.00 : ( 3.1 % ) ביטוח בריאות - ₪ 420.00 : ( 7 % ) ביטוח לאומי - סה\"כ ניכויים : 606.00 ₪ סה\"כ לתשלום נטו ** : 6244.00 **\n\nPayslip 8:\nתלוש שכר לחודש 01/2025\nפרטי המעסיק :עסקים בע''מ\nפרטי העובד :ישראל ישראלי,ת.ז :123456789,כתובת :שלום לכם\n:פירוט תשלומים\nשכר יסוד :6000.00 -\nדמי נסיעות :250.00\n-\nשעות נוספות )12 שעות :(600.00 -\n₪ סה\"כ ברוטו :6850.00\n-\n:פירוט ניכויים והפרשות\n₪ 390.00 :(6.5%) הפרשה לפנסיה -\n₪ 186.00 :(3.16) ביטוח בריאות -\n₪ 420.00 ;(7%) ביטוח לאומי\nסה\"כ ניכויים :996.00\n-O\n₪ סה\"כ לתשלום נטו ** :5854.00** תלוש שכר לחודש 01/2025 פרטי המעסיק : עסקים בע''מ פרטי העובד : ישראל ישראלי , ת.ז : 123456789 , כתובת : שלום לכם : פירוט תשלומים שכר יסוד : 6000.00 - דמי נסיעות : 250.00 - שעות נוספות ) 12 שעות : ( 600.00 - ₪ סה\"כ ברוטו : 6850.00 - : פירוט ניכויים והפרשות ₪ 390.00 : ( 6.5 % ) הפרשה לפנסיה - ₪ 186.00 : ( 3.16 ) ביטוח בריאות - ₪ 420.00 ; ( 7 % ) ביטוח לאומי סה\"כ ניכויים : 996.00 -O ₪ סה\"כ לתשלום נטו ** : 5854.00 **\n\nPayslip 9:\nתלוש שכר לחודש 02/2025\nפרטי המעסיק :עסקים בע''מ\nפרטי העובד :ישראל ישראלי ת.ז :123456789 כתובת :שלום לכם\n:פירוט תשלומים\nשכר יסוד :6000.00 -\nדמי נסיעות :0.00 -\n-\n₪ שעות נוספות 12 שעות :(600.00 -\nסה\"כ ברוטו :6600.00\n:פירוט ניכויים והפרשות\n₪ 390.00 :(6.5%) הפרשה לפנסיה -\n186.00₪ :(3.1%) ביטוח בריאות -\n₪ 420.00 :(7%) ביטוח לאומי -\nסה\"כ ניכויים :996.00\n₪ סה\"כ לתשלום נטו ** :5604.00** תלוש שכר לחודש 02/2025 פרטי המעסיק : עסקים בע''מ פרטי העובד : ישראל ישראלי ת.ז : 123456789 כתובת : שלום לכם : פירוט תשלומים שכר יסוד : 6000.00 - דמי נסיעות : 0.00 - - ₪ שעות נוספות 12 שעות : ( 600.00 - סה\"כ ברוטו : 6600.00 : פירוט ניכויים והפרשות ₪ 390.00 : ( 6.5 % ) הפרשה לפנסיה - 186.00 ₪ : ( 3.1 % ) ביטוח בריאות - ₪ 420.00 : ( 7 % ) ביטוח לאומי - סה\"כ ניכויים : 996.00 ₪ סה\"כ לתשלום נטו ** : 5604.00 **\n\nPayslip 10:\nתלוש שכר לחודש 03/2025\nפרטי המעסיק :עסקים בע''מ\nפרטי העובד ישראל ישראלי ת.ז :123456789,כתובת :שלום לכם\n:פירוט תשלומים\nשכר יסוד :6000.00 -\n-\nדמי נסיעות :250.00 -\nשעות נוספות )12 שעות (600.00 -\n₪ סה\"כ ברוטו :6850.00\n-\n:פירוט ניכויים והפרשות\n₪ 390.00 :(6.5%) הפרשה לפנסיה -\n186.00₪ :(3.1%) ביטוח בריאות -\n420.00₪ :(7%) ביטוח לאומי\n₪ סה\"כ ניכויים :996.00\n₪ סה\"כ לתשלום נטו ** :5854.00** תלוש שכר לחודש 03/2025 פרטי המעסיק : עסקים בע''מ פרטי העובד ישראל ישראלי ת.ז : 123456789 , כתובת : שלום לכם : פירוט תשלומים שכר יסוד : 6000.00 - - דמי נסיעות : 250.00 - שעות נוספות ) 12 שעות ( 600.00 - ₪ סה\"כ ברוטו : 6850.00 - : פירוט ניכויים והפרשות ₪ 390.00 : ( 6.5 % ) הפרשה לפנסיה - 186.00 ₪ : ( 3.1 % ) ביטוח בריאות - 420.00 ₪ : ( 7 % ) ביטוח לאומי ₪ סה\"כ ניכויים : 996.00 ₪ סה\"כ לתשלום נטו ** : 5854.00 **"
        # contract_text=":שעות נוספות .5\nבמידה והעובד יידרש לעבוד שעות נוספות, ישולם שכר נוסף כדין\n,עבור השעתיים הראשונות הנוספות בכל יום עבודה 125% -\n,עבור כל שעה נוספת מעבר לכך 150% -\nבונוסים .6\nהמעסיק רשאי להעניק לעובד בונוס שנתי בהתאם לשיקול דעתו ולביצועי העובד .סכום הבונוס\nייקבע בהתאם לשיקול דעת המעסיק\n:חופשה ומחלה .7\n,ימי החופשה והמחלה יהיו בהתאם לחוקי העבודה הקיימים במדינת ישראל\n:סיום העסקה .8\n.כל צד יוכל לסיים את ההתקשרות בהודעה מוקדמת של 30 יום מראש בכתב\nהצדדים מסכימים ומתחייבים בזאת לפעול בתום לב ובאופן הוגן\n;ולראיה באו הצדדים על החתום\nחתימת המעסיק\nחתימת העובד\n5.10.24\n5.10.24\n:תאריך\n:תאריך : שעות נוספות .5 במידה והעובד יידרש לעבוד שעות נוספות , ישולם שכר נוסף כדין , עבור השעתיים הראשונות הנוספות בכל יום עבודה 125 % - , עבור כל שעה נוספת מעבר לכך 150 % - בונוסים .6 המעסיק רשאי להעניק לעובד בונוס שנתי בהתאם לשיקול דעתו ולביצועי העובד .סכום הבונוס ייקבע בהתאם לשיקול דעת המעסיק : חופשה ומחלה .7 , ימי החופשה והמחלה יהיו בהתאם לחוקי העבודה הקיימים במדינת ישראל : סיום העסקה .8 .כל צד יוכל לסיים את ההתקשרות בהודעה מוקדמת של 30 יום מראש בכתב הצדדים מסכימים ומתחייבים בזאת לפעול בתום לב ובאופן הוגן ; ולראיה באו הצדדים על החתום חתימת המעסיק חתימת העובד 5.10.24 5.10.24 : תאריך : תאריך"

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

        # General instructions are now outside the loop and conditional
        if type != 'warning_letter':
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
    [About SIMILAR CASE OR LEGAL PRECEDENT (based on online search,seach thoroughly because there is a case for every laws out there`). If none is found, write: "לא נמצאו תקדימים רלוונטיים בחיפוש המקוון."]
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
            prompt = f"""
אתה עוזר משפטי מיומן. עליך לנתח את רשימת ההפרות ולהפיק רשימת תביעות מסודרת לפי מסמך (לדוגמה: תלוש שכר מס' 1, מסמך שימוע, מכתב פיטורין וכו').

הנחיות:

1. סדר את התביעות לפי מסמך: כל קבוצה מתחילה בכותרת כמו "תלוש שכר מס' 4 – 05/2024:".

2. תחת כל כותרת, צור רשימה ממוספרת באותיות עבריות (א., ב., ג. וכו').

3. השתמש במבנה הקבוע הבא:
   א. סכום של [amount] ש"ח עבור \[תיאור קצר של ההפרה].

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
            prompt = f"""
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

        # Conditionally add the IMPORTANT block, excluding it for 'warning_letter' and 'easy'
        # if type != 'warning_letter' and type != 'easy' and type != 'table':
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
            
            # Structure the result
            result = {
                "legal_analysis": analysis,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            # Log the full error for debugging if possible, or at least the type and message
            error_detail = f"Error generating legal analysis with new API: {str(e)}"
            # Check if the error is from genai and has more specific details
            if hasattr(e, 'error'): # For google.api_core.exceptions.GoogleAPIError
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

    def summarise(self, ai_content_text: str) -> str:
        """
        Summarizes the given text using the Gemini 2.5 Flash model.
        """
        try:
            gemini_api_key = os.environ.get("GOOGLE_CLOUD_VISION_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables.")

            client = genai.Client(api_key=gemini_api_key)
            # Using gemini-2.5-flash-preview-05-20 as requested for summarization
            model_name = "gemini-2.5-flash-preview-04-17"

            prompt = f"Please summarize the following text concisely:\n\n{ai_content_text}"
            
            api_contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Minimal config for summarization, no tools needed unless specified
            gen_config = types.GenerateContentConfig(
                temperature=0.1, # Lower temperature for more factual summary
                response_mime_type="text/plain",
            )

            response = client.models.generate_content(
                model=model_name,
                contents=api_contents,
                config=gen_config,
            )
            
            summary = ""
            if response.text:
                summary = response.text
            
            return summary
            
        except Exception as e:
            error_detail = f"Error generating summary with Gemini Flash: {str(e)}"
            if hasattr(e, 'error'): # For google.api_core.exceptions.GoogleAPIError
                error_detail = f"Error generating summary with Gemini Flash: {e.message}"
            # Depending on how you want to handle errors, you might raise an HTTPException
            # or return an error message. For now, let's re-raise to be consistent.
            raise HTTPException(
                status_code=500,
                detail=error_detail
            )