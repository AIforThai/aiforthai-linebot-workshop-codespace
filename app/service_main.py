import os
import tempfile
from datetime import datetime
import base64
import requests


import ffmpeg
from aift import setting
from aift.multimodal import audioqa, textqa, vqa
from fastapi import APIRouter, Request
from linebot import LineBotApi
from linebot.models import AudioMessage, ImageMessage, MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
# from app import audioqa, textqa, vqa
from app.configs import Configs

router = APIRouter(tags=["Main"], prefix="/message")

cfg = Configs()
setting.set_api_key(cfg.AIFORTHAI_APIKEY)
line_bot_api = LineBotApi(cfg.LINE_CHANNEL_ACCESS_TOKEN)

text_for_audio_append = (
    "คุณต้องการให้ฉันทำอะไรกับเสียงนี้ ? รับเฉพาะข้อความเท่านั้น\n\nยกเลิกให้พิมพ์ ยกเลิก หรือ cancel"
)
mp3_file = []

text_for_visual_append = (
    "คุณต้องการให้ฉันทำอะไรกับภาพนี้ ? รับเฉพาะข้อความเท่านั้น\n\nยกเลิกให้พิมพ์ ยกเลิก หรือ cancel"
)
image_file = []

# ✅ ฟังก์ชันหลักสำหรับให้ webhook เรียกใช้
async def handle_event(event: MessageEvent):
    if isinstance(event.message, TextMessage):
        await handle_text(event)
    elif isinstance(event.message, AudioMessage):
        await handle_audio(event)
    elif isinstance(event.message, ImageMessage):
        await handle_image(event)
    else:
        send_message(event, "❗️ไม่รองรับประเภทข้อความนี้ครับ")



async def handle_text(event):
    user_text = event.message.text.strip().lower()
    res = None

    # ✅ ยกเลิกการทำงาน
    if user_text in ["ยกเลิก", "cancel"]:
        for path in mp3_file:
            if os.path.exists(path):
                os.remove(path)
        mp3_file.clear()

        for path in image_file:
            if os.path.exists(path):
                os.remove(path)
        image_file.clear()

        send_message(event, "เคลียร์ความจำก่อนหน้านี้แล้ว")
        return

    # ✅ ถ้ามี mp3
    if mp3_file:
        res = audioqa.generate(mp3_file[0], user_text, return_json=True)

    # ✅ กรณี maewmong หรือ แมวมอง พร้อม image
    elif user_text.lower() in ["maewmong", "แมวมอง","MAEWMONG","Meawmong"] and image_file:
        messages = call_maewmong_api(event, image_file[0], cfg.AIFORTHAI_APIKEY)
        line_bot_api.reply_message(event.reply_token, messages)
        clear_files(image_file)
        return

    # ✅ ถ้ามีรูปภาพ + คำถามทั่วไป
    elif image_file:
        res = vqa.generate(image_file[0], user_text, return_json=True)

    # ✅ ข้อความทั่วไป
    else:
        # print("Pathumma LLM -- Text ")
        current_time = datetime.now()
        session_id = f"{current_time.day:02}{current_time.month:02}{current_time.hour:02}{(current_time.minute // 10) * 10:02}"
        res = textqa.chat(user_text, session_id + cfg.AIFORTHAI_APIKEY, temperature=0.6)
        # print(res)

    # ✅ ตอบกลับ
    if res and "content" in res:
        reply = res["content"] if isinstance(res["content"], str) else res["content"][0]
    elif res and "response" in res:
        reply = res["response"] if isinstance(res["response"], str) else res["response"][0]
    else:
        reply = "ไม่สามารถประมวลผลข้อความนี้ได้"

    send_message(event, reply)


async def handle_audio(event):
    message_content = line_bot_api.get_message_content(event.message.id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_in:
        for chunk in message_content.iter_content():
            tmp_in.write(chunk)
        tmp_in_path = tmp_in.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        (
            ffmpeg.input(tmp_in_path)
            .output(tmp_out_path, format="mp3", acodec="libmp3lame")
            .run(quiet=True, overwrite_output=True)
        )
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)

        mp3_file.append(tmp_out_path)
        send_message(event, text_for_audio_append)

    except Exception as e:
        send_message(event, f"เกิดข้อผิดพลาดในการแปลงไฟล์เสียง: {e}")
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)


async def handle_image(event):
    image_content = line_bot_api.get_message_content(event.message.id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
        for chunk in image_content.iter_content():
            tmp_img.write(chunk)
        tmp_img_path = tmp_img.name

    image_file.append(tmp_img_path)
    send_message(event, text_for_visual_append)

# ✅ ฟังก์ชันการเคลียร์ค่า
def clear_files(file_list):
    for path in file_list:
        if os.path.exists(path):
            os.remove(path)
    file_list.clear()

# ✅ ฟังก์ชันส่งข้อความ
def send_message(event, message: str):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message))

# ✅ ฟังก์ชันส่งรูปภาพ
def send_image(event, image_url):
    line_bot_api.reply_message(
        event.reply_token,
        ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
    )


# ✅ ฟังก์ชันใหม่: รองรับข้อความจากไฟล์ เช่น docx, pdf, txt
async def handle_text_from_file(event, text_content: str):
    try:
        current_time = datetime.now()
        session_id = f"{current_time.day:02}{current_time.month:02}{current_time.hour:02}{(current_time.minute // 10) * 10:02}"
        
        # ใช้ LLM chat จากข้อความที่แปลงจากไฟล์
        res = textqa.chat(text_content, session_id + cfg.AIFORTHAI_APIKEY, temperature=0.6)

        # เลือกข้อความตอบกลับ
        if res and "content" in res:
            reply = res["content"] if isinstance(res["content"], str) else res["content"][0]
        elif res and "response" in res:
            reply = res["response"] if isinstance(res["response"], str) else res["response"][0]
        else:
            reply = "ไม่สามารถประมวลผลเนื้อหาจากไฟล์ได้"

    except Exception as e:
        reply = f"❌ เกิดข้อผิดพลาดในการประมวลผลไฟล์: {e}"

    send_message(event, reply)


# ฟังก์ชันส่งรูปภาพไปที่แมวมอง
def call_maewmong_api(event, image_path, apikey):
    try:
        # แปลงภาพเป็น base64
        with open(image_path, "rb") as img:
            encoded_image = base64.b64encode(img.read()).decode("utf-8")

        payload = {
            "index": "imagesearch-dara",
            "image": encoded_image,
        }

        headers = {
            "Content-Type": "application/json",
            "Apikey": apikey,
        }
        
        response = requests.post(cfg.URL_MAEWMONG, json=payload, headers=headers)
        data = response.json()

        if "result" in data and data["result"][0]:
            person = data["result"][0][0]
            name = person.get("name", "ไม่ทราบชื่อ")
            image_url = cfg.IMG_RESULT + person["thumb"]

            messages = [
                TextSendMessage(text=f"ดารามาแล้ว\n ชื่อ: {name}"),
                ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
            ]
            return messages
        else:
            return [TextSendMessage(text="ไม่พบข้อมูลจาก Maewmong API")]

    except Exception as e:
        return [TextSendMessage(text=f"เกิดข้อผิดพลาด: {str(e)}")]

