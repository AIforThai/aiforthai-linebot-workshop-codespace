import requests
from aift.setting.setting import PACKAGE_NAME

from app.configs import Configs

cfg = Configs()


def _safe_json_response(res: requests.Response) -> dict:
    try:
        return res.json()
    except ValueError:
        # Upstream sometimes returns an empty/non-JSON body on failures.
        return {
            "response": f"API error {res.status_code}: {res.text[:200] or 'empty response body'}"
        }


def generate(
    instruction: str,
    system_prompt: str = "You are Pathumma LLM, created by NECTEC. Your are a helpful assistant.",
    max_new_tokens: int = 512,
    temperature: float = 0.4,
    return_json: bool = True,
):

    api_key = cfg.AIFORTHAI_APIKEY
    headers = {"Apikey": api_key, "X-lib": PACKAGE_NAME}
    url = cfg.URL_TEXTQA
    payload = {
        "instruction": instruction,
        "system_prompt": system_prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "return_json": return_json,
    }

    res = requests.post(url, headers=headers, data=payload)
    data = _safe_json_response(res)
    if not return_json:
        return data.get("content") or data.get("response")
    else:
        return data


def chat(
    instruction: str,
    sessionid: str,
    context: str = "",
    temperature: float = 0.4,
    return_json: bool = True,
):

    api_key = cfg.AIFORTHAI_APIKEY
    headers = {"accept": "application/json", "Apikey": api_key, "X-lib": PACKAGE_NAME}
    url = cfg.URL_PATHUMMA_CHAT
    payload = {
        "context": context,
        "prompt": instruction,
        "sessionid": sessionid,
        "temperature": temperature,
    }

    res = requests.post(url, headers=headers, data=payload)
    data = _safe_json_response(res)
    if not return_json:
        return data.get("response") or data.get("content")
    else:
        return data
