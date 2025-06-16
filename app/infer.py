import json
import logging
import pathlib
import time

import numpy as np
import onnxruntime
import PIL.Image

logging.getLogger().setLevel(level=logging.DEBUG)
logging.info(f"onnxruntime_providers: {onnxruntime.get_available_providers()}")


def load_json(path: pathlib.Path):
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data
    else:
        return None


def preprocess(input_data, config, dtype=np.float32):
    img_data = input_data
    if len(img_data.shape) == 2:
        img_data = np.expand_dims(img_data, axis=2)

    # normalize
    if config["do_normalize"]:
        mean_vec = np.array(config["image_mean"])
        stddev_vec = np.array(config["image_std"])
        norm_img_data = np.zeros(img_data.shape)
        for i in range(img_data.shape[2]):
            norm_img_data[:, :, i] = (img_data[:, :, i] / 255 - mean_vec[i]) / stddev_vec[i]
    else:
        norm_img_data = img_data

    norm_img_data = np.expand_dims(np.transpose(norm_img_data, (2, 0, 1)), axis=0)

    return norm_img_data.astype(dtype)


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def softmax_np(x, axis=0):
    maxes = np.max(x, axis=axis, keepdims=True)[0]
    x_exp = np.exp(x - maxes)
    x_exp_sum = np.sum(x_exp, axis=axis, keepdims=True)
    probs = x_exp / x_exp_sum
    return probs


def postprocess(result):
    return softmax(np.array(result)).tolist()


def center_to_corners_format(bboxes_center: np.ndarray) -> np.ndarray:
    center_x, center_y, width, height = bboxes_center.T
    bboxes_corners = np.stack(
        # top left x, top left y, bottom right x, bottom right y
        [
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ],
        axis=-1,
    )
    return bboxes_corners


def check_model_type(model_dir: pathlib.Path):
    try:
        if model_dir is None:
            return -2
        model_dir = pathlib.Path(model_dir)
        json_path = model_dir.joinpath("preprocessor_config.json")
        if json_path.exists():
            config = load_json(json_path)
            arc = config["task"]
            if "CLASSIFICATION" == arc:
                return 0
            elif "OBJDETECTION" == arc:
                return 1
            elif "SEGMENTATION" == arc:
                return 2
            else:
                return -1
        else:
            return -2
    except Exception as e:
        logging.exception(e)


def resize_keep_ratio(input: PIL.Image.Image, size=(800, 800)):
    """Resize input to maximum size"""
    mp_w = round(size[0] / input.width, 1)
    mp_h = round(size[1] / input.height, 1)
    scale = max(mp_w, mp_h)
    target = (int(input.width * scale), int(input.height * scale))
    input = input.resize(target, resample=PIL.Image.Resampling.LANCZOS)
    input.thumbnail(size)
    return input


def get_onnx_session(model_dir: pathlib.Path):
    if model_dir.joinpath("model.onnx").exists():
        if model_dir.joinpath("preprocessor_config.json").exists():
            mode = check_model_type(model_dir)
            config = load_json(model_dir.joinpath("preprocessor_config.json"))
            session_options = onnxruntime.SessionOptions()
            session_options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            logging.info("ONNX: Load model")
            session = onnxruntime.InferenceSession(
                model_dir.joinpath("model.onnx"),
                sess_options=session_options,
            )
            logging.info(f"ONNX Session providers: {session.get_providers()}")
            TMP_SESSION = session, mode, config
            return TMP_SESSION
    return None, None, None


def predict_image_onnx(session, mode, config, input: PIL.Image.Image, threshold: float):
    if config is None or session is None:
        return None
    # get the name of the first input of the model
    input_name = session.get_inputs()[0].name
    x, y = input.size

    # resize imput to model's input_size
    input = input.resize(config["input_size"], resample=PIL.Image.Resampling.LANCZOS)

    # check for fp16
    input_type = np.float32  # tensor(float)
    if "tensor(float16)" == session.get_inputs()[0].type:
        input_type = np.float16  # tensor(float16)
    input_data = preprocess(np.asarray(input), config, dtype=input_type)

    st = time.time()
    raw_result = session.run([], {input_name: input_data})
    logging.info(f"runtime prediction onnx: {time.time() - st} sec")
    id2labels = config["id2label"]

    if mode == 0:
        # for Classification
        res = postprocess(raw_result)
        idx = np.argmax(res)
        sort_idx = np.flip(np.squeeze(np.argsort(res)))
        predict_class = id2labels[str(sort_idx[0])]
        output = f"{predict_class}: {res[idx]:0.2f}"
    return output
