import requests
from aift.setting.setting import PACKAGE_NAME

from app.configs import Configs

cfg = Configs()


def _safe_json_response(res: requests.Response) -> dict:
    try:
        return res.json()
    except ValueError:
        return {
            "content": f"API error {res.status_code}: {res.text[:200] or 'empty response body'}"
        }


def generate(file: str, instruction: str, return_json: bool = True):
    api_key = cfg.AIFORTHAI_APIKEY
    headers = {"Apikey": api_key, "X-lib": PACKAGE_NAME}
    url = cfg.URL_VQA

    payload = {"query": instruction}
    with open(file, "rb") as image_file:
        files = [("file", (file, image_file, "image/jpeg"))]
        res = requests.request("POST", url, headers=headers, data=payload, files=files)

    data = _safe_json_response(res)

    if not return_json:
        return data.get("content") or data.get("response")
    else:
        return data
