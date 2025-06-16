import os
import tempfile

import PIL
from aift import setting
from fastapi import APIRouter, Request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    ImageMessage,
    MessageEvent,
    TextSendMessage,
)

from app.configs import Configs

from . import infer

router = APIRouter(tags=["NomadML"], prefix="/nomadml")

cfg = Configs()

setting.set_api_key(cfg.AIFORTHAI_APIKEY)  # AIFORTHAI_APIKEY
line_bot_api = LineBotApi(cfg.LINE_CHANNEL_ACCESS_TOKEN)  # CHANNEL_ACCESS_TOKEN
handler = WebhookHandler(cfg.LINE_CHANNEL_SECRET)  # CHANNEL_SECRET

# LOAD MODEL ONCE
session, mode, config = infer.get_onnx_session("model-onnx")


@router.post("")
async def nomadml_demo(request: Request):
    """
    Line Webhook endpoint สำหรับรับข้อความและรูปภาพจาก Line Messaging API และประมวลผลด้วย model จาก NomadML

    ฟังก์ชันนี้ทำหน้าที่:
    1. รับ HTTP POST Request จาก Line Webhook
    2. ตรวจสอบลายเซ็น (X-Line-Signature) เพื่อยืนยันความถูกต้องของข้อความ
    3. ส่งข้อความไปยัง handler เพื่อประมวลผลอีเวนต์ที่ได้รับ
    4. รองรับการประมวลผลรูปภาพ (ImageMessage):
        - สำหรับรูปภาพ (ImageMessage): ประมวลผลรูปภาพด้วยโมเดล AI ที่เลือกไว้ และส่งผลลัพธ์กลับไปยังผู้ใช้
    """
    signature = request.headers["X-Line-Signature"]
    body = await request.body()
    try:
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token or channel secret.")
    return "OK"


@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_id = event.message.id
    image_content = line_bot_api.get_message_content(message_id)
    result = "ERROR"
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "image.jpg")
        # Save user's image in tempdirectory and process it
        with open(img_path, "wb") as f:
            for chunk in image_content.iter_content():
                f.write(chunk)

        # NomadML Inference section
        with PIL.Image.open(img_path) as _img:
            _img = infer.resize_keep_ratio(_img)
            res = infer.predict_image_onnx(session, mode, config, _img, 0.8)
        result = res

    # return text response
    send_message(event, result)


def echo(event):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=event.message.text))


# function for sending message
def send_message(event, message):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message))
