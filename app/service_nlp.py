from fastapi import APIRouter, Request

from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage, AudioSendMessage, AudioMessage

from datetime import datetime
import io, re, json, requests, wave

from app.configs import Configs
from app.user_state_store import set_user_state, get_user_state, clear_user_state

from aift import setting
from aift.nlp import tokenizer, ner, g2p, soundex, similarity, text_cleansing, tag
from aift.nlp.translation import zh2th, th2zh, en2th, th2en
from aift.nlp.longan import sentence_tokenizer, tagger, token_tagger, tokenizer as logan_tokenizer
from aift.nlp import sentiment
from aift.nlp.alignment import en_alignment, zh_alignment
from aift.speech import tts
from aift.speech.stt import partii4, partii5

router = APIRouter(tags=["NLP"], prefix="/nlp")
cfg = Configs()
setting.set_api_key(cfg.AIFORTHAI_APIKEY)
line_bot_api = LineBotApi(cfg.LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(cfg.LINE_CHANNEL_SECRET)

COMMAND_LIST = [
    "#trexplus", "#lexto", "#trex++", "#tner", "#g2p", "#soundex", "#thaiwordsim", "#wordapprox",
    "#textclean", "#tagsuggest", "#mtch2th", "#mtth2ch", "#mten2th", "#mtth2en", "#ssense", "#emonews",
    "#thaimoji", "#cyberbully", "#longan_sentence", "#longan_tagger", "#longan_tokentag", "#longan_tokenizer",
    "#en2th_aligner", "#ch2th_aligner", "#tts", "#vajatts:0|", "#textsum"
]

EMOJI_MAP = {
    0: "😊", 1: "🥲", 2: "😡", 3: "😑", 4: "😱", 5: "😰", 6: "😯", 7: "😴",
    8: "😝", 9: "😍", 10: "😌", 11: "😐", 12: "😬", 13: "😳", 14: "😵",
    15: "💔", 16: "😎", 17: "😭", 18: "😅", 19: "😉", 20: "💜", 21: "😇"
}



def has_user_state(user_id: str) -> bool:
    return get_user_state(user_id) is not None

async def handle_event(event):
    if isinstance(event.message, TextMessage):
        await handle_text(event)
    elif isinstance(event.message, AudioMessage):
        await handle_audio(event)

def extract_command_and_text(message_text: str):
    text = message_text.strip()
    if text.startswith("#nlp"):
        text = text[len("#nlp"):]
    for cmd in COMMAND_LIST:
        if text.startswith(cmd):
            return cmd, text[len(cmd):].strip()
    return None, text

async def handle_text(event):
    user_id = event.source.user_id
    user_input = event.message.text.strip()

    # ✅ ตรวจสอบกรณีผู้พิมพ์ "ยกเลิก"
    if user_input in ["ยกเลิก", "cancel", "Cancel", "CANCEL"]:
        clear_user_state(user_id)
        send_message(event, "✅ ยกเลิกบริการเรียบร้อยแล้ว")
        return

    # ✅ ตรวจสอบกรณีเลือกจาก Rich Menu
    if user_input.startswith("SELECT:"):
        cmd = user_input[len("SELECT:"):].strip().lower()
        if f"#{cmd}" in COMMAND_LIST:
            set_user_state(user_id, f"#{cmd}") # บันทึก command + timestamp
            send_message(event, f"🤖 {cmd.upper()} [พร้อมให้บริการ]\nกรุณาใส่ข้อความหรือคำที่ต้องการ.")
        else:
            send_message(event, f"❗บริการไม่ถูกต้อง: {cmd}")
        return
    
    state_command = get_user_state(user_id)

    # ✅ กรณี image_mode → ไม่ให้ NLP จัดการ
    if state_command == "image_mode" and user_input in ["1", "2", "3", "4", "5"]:
        # Do nothing → ปล่อยให้ service_image.py ดักไปจัดการ
        return

    # ✅ กรณี timeout
    if not state_command and has_user_state(user_id):
        send_message(event, "⏱ หมดเวลา 5 นาทีแล้ว ระบบได้ยกเลิกบริการที่เลือกไว้แล้วครับ")
        clear_user_state(user_id)
        return

    if state_command:
        msg_with_command = f"{state_command}{user_input}"
        clear_user_state(user_id)
    else:
        msg_with_command = user_input

    matched_command, content = extract_command_and_text(msg_with_command)

    if matched_command:
        try:
            await process_command(event, matched_command, content, user_input)
        except Exception as e:
            send_message(event, f"❗Error: {str(e)}")
    else:
        send_message(event, "Service not found")

async def process_command(event, matched_command, content, user_input):
    if matched_command == "#trexplus":
        result = tokenizer.tokenize(content, engine='trexplus', return_json=True)
        send_message(event, str(result))
    elif matched_command == "#lexto":
        result = tokenizer.tokenize(content, engine='lexto', return_json=True)
        send_message(event, str(result))
    elif matched_command == "#trex++":
        result = tokenizer.tokenize(content, engine='trexplusplus', return_json=True)
        send_message(event, str(list(zip(result['words'], result['tags']))))
    elif matched_command == "#tner":
        result = ner.analyze(content, return_json=True)
        send_message(event, str(list(zip(result['words'], result['POS'], result['tags']))))
    elif matched_command == "#longan_sentence":
        send_message(event, str(sentence_tokenizer.tokenize(content)))
    elif matched_command == "#longan_tagger":
        send_message(event, str(tagger.tag(content)))
    elif matched_command == "#longan_tokentag":
        send_message(event, str(token_tagger.tokenize_tag(content)))
    elif matched_command == "#longan_tokenizer":
        send_message(event, str(logan_tokenizer.tokenize(content)))
    elif matched_command == "#g2p":
        result = callG2P(content)
        if "phoneme" in result:
            # print(result['phoneme'])
            send_message(event, str(result["phoneme"]))
        else:
            # print("Error or unexpected response:", result)
            send_message(event, str(result))
    elif matched_command == "#textsum":
        send_message(event, callTextSummarization(content))
    elif matched_command == "#soundex":
        model = "personname"
        result = soundex.analyze(content, model=model)['words']
        send_message(event, str(result))
    elif matched_command.startswith("#thaiwordsim"):
        result = similarity.similarity(content, engine='thaiwordsim', model="thwiki")
        send_message(event, str(result))
    elif matched_command == "#wordapprox":
        result = similarity.similarity(content, engine='wordapprox', model="personname", return_json=True)
        send_message(event, str(result))
    elif matched_command == "#textclean":
        send_message(event, str(text_cleansing.clean(content)))
    elif matched_command == "#tagsuggest":
        send_message(event, str(tag.analyze(content, numtag=5)))
    elif matched_command == "#mtch2th":
        send_message(event, str(zh2th.translate(content, return_json=True)))
    elif matched_command == "#mtth2ch":
        send_message(event, str(th2zh.translate(content, return_json=True)))
    elif matched_command == "#mten2th":
        send_message(event, str(en2th.translate(content)))
    elif matched_command == "#mtth2en":
        send_message(event, str(th2en.translate(content)))
    elif matched_command == "#ssense":
        send_message(event, str(sentiment.analyze(content, engine='ssense')))
    elif matched_command == "#emonews":
        send_message(event, str(sentiment.analyze(content, engine='emonews')))
    elif matched_command == "#thaimoji":
        # send_message(event, str(sentiment.analyze(content, engine='thaimoji')))
        result = sentiment.analyze(content, engine='thaimoji')  # result = dict: {'0': '0.25', '9': '0.31', ...}

        # แปลงเป็นลิสต์ของ (class_id: int, score: float)
        class_scores = [(int(k), float(v)) for k, v in result.items()]
        
        # เรียงจากคะแนนมากไปน้อย
        sorted_classes = sorted(class_scores, key=lambda x: x[1], reverse=True)
        
        # เลือก top 3 class ID
        top_classes = [cid for cid, score in sorted_classes[:3]]
        
        # แปลงเป็น emoji
        emojis = [EMOJI_MAP.get(cid, "❓") for cid in top_classes]
        
        reply = f"🤖 อีโมจิที่ตรงกับข้อความ:\n{' '.join(emojis)}"
        send_message(event, reply)
    elif matched_command == "#cyberbully":
        send_message(event, str(sentiment.analyze(content, engine='cyberbully')))
    elif matched_command == "#en2th_aligner":
        contents = content.split('|')
        send_message(event, str(en_alignment.analyze(contents[0], contents[1], return_json=True)))
    elif matched_command == "#ch2th_aligner":
        contents = content.split('|')
        send_message(event, str(zh_alignment.analyze(contents[0], contents[1], return_json=True)))
    elif matched_command.startswith("#vajatts"):
        await process_vajatts(event, matched_command, content, event.message.text)
    elif matched_command == "#tts":
        response = callVaja9(content, 0)
        if response.json().get('msg') == 'success':
            download_and_play(response.json()['wav_url'])
            send_audio(event, cfg.WAV_URL + cfg.DIR_FILE + cfg.WAV_FILE, response.json()['durations'])
        else:
            send_message(event, "TTS failed")

def send_audio(event, audio_url, duration_sec):
    audio_msg = AudioSendMessage(original_content_url=audio_url, duration=int(duration_sec * 1000))
    line_bot_api.reply_message(event.reply_token, audio_msg)

def send_message(event, message):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message))

def get_wav_duration_in_ms(file_path):
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return int((frames / rate) * 1000)

def callVaja9(text, speaker):
    url = cfg.URL_VAJA
    headers = {'Apikey': cfg.AIFORTHAI_APIKEY, 'Content-Type': 'application/json'}
    response = requests.post(url, json={'input_text': text, 'speaker': speaker}, headers=headers)
    return response

def download_and_play(wav_url):
    file_name = cfg.DIR_FILE + cfg.WAV_FILE
    resp = requests.get(wav_url, headers={'Apikey': cfg.AIFORTHAI_APIKEY})
    if resp.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(resp.content)

def callG2P(content):
    url = "https://api.aiforthai.in.th/g2p"
    headers = {"Apikey": cfg.AIFORTHAI_APIKEY, "Content-Type": "application/json"}
    params = json.dumps({"text": content})
    response = requests.post(url, data=params, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def callTextSummarization(content):
    url = 'https://api.aiforthai.in.th/textsummarize'
    headers = {'Apikey': cfg.AIFORTHAI_APIKEY, 'Content-Type': 'application/json'}
    params = json.dumps([{"id": 100, "comp_rate": 30, "src": content}])
    response = requests.post(url, data=params, headers=headers)
    return bytes(response.text, "utf-8").decode("unicode_escape")

async def handle_audio(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    with open("received_audio.wav", "wb") as f:
        for chunk in message_content.iter_content():
            f.write(chunk)
    text = callPartii("received_audio.wav")
    send_message(event, str(text))

def callPartii(file):
    url = cfg.URL_PARTII
    files = {'wavfile': (file, open(file, 'rb'), 'audio/wav')}
    headers = {'Apikey': cfg.AIFORTHAI_APIKEY}
    param = {"outputlevel": "--uttlevel", "outputformat": "--txt"}
    response = requests.post(url, headers=headers, files=files, data=param)
    return json.loads(response.text)['message']

async def process_vajatts(event, matched_command, content, user_input):
    # print(f'Match Command: {user_input} :: Content=> {event.message.text}')
    # ดึง speaker จาก matched_command เช่น '#vajatts:0'
    match = re.match(r"#vajatts:(\d)", matched_command)
    if not match:
        send_message(event, "❗Error: Use #vajatts:X|ข้อความ (X = 0,1,2,3)")
        return

    speaker = int(match.group(1))
    # message_text = match.group(2).strip()

    if speaker not in [0, 1, 2, 3]:
        send_message(event, "❗Speaker must be 0=ชาย, 1=หญิง, 2=เด็กชาย, 3=เด็กหญิง")
        return

    tts.convert(content, cfg.DIR_FILE + cfg.WAV_FILE, speaker=speaker)
    audio_url = cfg.WAV_URL + cfg.DIR_FILE + cfg.WAV_FILE
    audio_duration = get_wav_duration_in_ms(cfg.DIR_FILE + cfg.WAV_FILE)
    send_audio(event, audio_url, audio_duration / 1000)
