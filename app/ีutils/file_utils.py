# app/utils/file_utils.py
from linebot import LineBotApi
from docx import Document
from PyPDF2 import PdfReader
import io
import tempfile
from app.configs import Configs

cfg = Configs()
line_bot_api = LineBotApi(cfg.LINE_CHANNEL_ACCESS_TOKEN)

def extract_text_from_file_message(file_message, message_id: str) -> str:
    """ดาวน์โหลดไฟล์จาก LINE และแปลงเป็นข้อความ"""
    # ดาวน์โหลดไฟล์จาก LINE server
    file_content = line_bot_api.get_message_content(message_id).content

    # อ่านเนื้อหาไฟล์ตามประเภท
    filename = file_message.file_name.lower()
    if filename.endswith(".txt"):
        return file_content.decode("utf-8", errors="ignore")

    elif filename.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        doc = Document(tmp_file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    elif filename.endswith(".pdf"):
        pdf_reader = PdfReader(io.BytesIO(file_content))
        return "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

    else:
        return "❌ Unsupported file type"
