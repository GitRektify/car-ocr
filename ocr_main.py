import cv2
import re
import json
import os
import numpy as np
import unicodedata
from paddleocr import PaddleOCR


# =========================
# CONFIG
# =========================

IMAGE_PATH = "ocr-car3.jpg"

MODEL_DIR = "models"

DET_MODEL_DIR = f"{MODEL_DIR}/ch_PP-OCRv4_det_infer"
REC_MODEL_DIR = f"{MODEL_DIR}/ch_PP-OCRv4_rec_infer"
CLS_MODEL_DIR = f"{MODEL_DIR}/ch_ppocr_mobile_v2.0_cls_infer"

USE_GPU = False


# =========================
# OCR INIT
# =========================

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="ch",
    det_model_dir=DET_MODEL_DIR,
    rec_model_dir=REC_MODEL_DIR,
    cls_model_dir=CLS_MODEL_DIR,
    use_gpu=USE_GPU,
    show_log=False,
    drop_score=0.3
)


# =========================
# PREPROCESS IMAGE
# =========================

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = img.shape[:2]
    if w < 1000:
        scale = 1000 / w
        img = cv2.resize(img, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    blur = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharp = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)

    return sharp


# =========================
# RUN OCR
# =========================

def run_ocr(img):
    result = ocr.ocr(img, cls=True)

    lines = []

    if not result or not result[0]:
        return lines

    for item in result[0]:
        box = item[0]
        text = item[1][0]
        score = float(item[1][1])

        lines.append({
            "text": text,
            "score": score,
            "box": box
        })

    return lines


# =========================
# DRAW OCR BOXES
# =========================
def draw_ocr_result_from_img(img, image_path, lines):
    draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for idx, line in enumerate(lines, start=1):
        box = line["box"]
        score = line["score"]

        pts = np.array(box, dtype=np.int32)

        cv2.polylines(draw_img, [pts], True, (0, 255, 0), 2)

        x = int(box[0][0])
        y = int(box[0][1]) - 10
        if y < 20:
            y = int(box[0][1]) + 25

        label = f"{idx}: {score:.2f}"

        cv2.putText(
            draw_img,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    output_img = os.path.splitext(image_path)[0] + "-ocr-boxes.jpg"
    cv2.imwrite(output_img, draw_img)

    return output_img

def draw_ocr_result(image_path, lines):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = img.shape[:2]

    for idx, line in enumerate(lines, start=1):
        box = line["box"]
        score = line["score"]

        pts = np.array(box, dtype=np.int32)

        cv2.polylines(
            img,
            [pts],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

        x = int(box[0][0])
        y = int(box[0][1]) - 10

        if y < 20:
            y = int(box[0][1]) + 25

        label = f"{idx}: {score:.2f}"

        cv2.putText(
            img,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    output_img = os.path.splitext(image_path)[0] + "-ocr-boxes.jpg"
    cv2.imwrite(output_img, img)

    return output_img


# =========================
# TEXT CLEANING
# =========================

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)

    text = text.upper()

    text = text.replace(" ", "")
    text = text.replace("：", ":")
    text = text.replace("，", ",")
    text = text.replace("。", ".")
    text = text.replace("－", "-")
    text = text.replace("—", "-")
    text = text.replace("_", "")

    return text


# =========================
# OCR ERROR FIXES
# =========================

def fix_common_ocr_errors(text):
    replacements = {
        # Uppercase letters
        "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E",
        "Ｆ": "F", "Ｇ": "G", "Ｈ": "H", "Ｉ": "1", "Ｊ": "J",
        "Ｋ": "K", "Ｌ": "L", "Ｍ": "M", "Ｎ": "N", "Ｏ": "0",
        "Ｐ": "P", "Ｑ": "Q", "Ｒ": "R", "Ｓ": "S", "Ｔ": "T",
        "Ｕ": "U", "Ｖ": "V", "Ｗ": "W", "Ｘ": "X", "Ｙ": "Y",
        "Ｚ": "Z",

        # Lowercase letters
        # "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e",
        # "ｆ": "f", "ｇ": "g", "ｈ": "h", "ｉ": "i", "ｊ": "j",
        # "ｋ": "k", "ｌ": "l", "ｍ": "m", "ｎ": "n", "ｏ": "o",
        # "ｐ": "p", "ｑ": "q", "ｒ": "r", "ｓ": "s", "ｔ": "t",
        # "ｕ": "u", "ｖ": "v", "ｗ": "w", "ｘ": "x", "ｙ": "y",
        # "ｚ": "z",

        # Numbers
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


# =========================
# FIELD EXTRACTION
# =========================

def extract_fields(lines):
    raw_text = "\n".join([x["text"] for x in lines])
    text = normalize_text(fix_common_ocr_errors(raw_text))

    fields = {}

    vin_match = re.search(r"[A-HJ-NPR-Z0-9]{17}", text)
    if vin_match:
        fields["vin"] = vin_match.group(0)

    m = re.search(r"整车型号[:：]?([A-Z0-9]+)", text)
    if m:
        fields["vehicle_model"] = m.group(1)

    m = re.search(r"发动机型号[:：]?([A-Z0-9]+)", text)
    if m:
        fields["engine_model"] = m.group(1)

    m = re.search(r"驱动电机型号[:：]?([A-Z0-9/]+)", text)
    if m:
        fields["drive_motor_model"] = m.group(1)

    m = re.search(r"发动机排量[:：]?(\d+)\s?ML", text)
    if m:
        fields["engine_displacement_ml"] = m.group(1)

    m = re.search(r"发动机最大净功率[:：]?(\d+)\s?KW", text)
    if m:
        fields["engine_max_net_power_kw"] = m.group(1)

    m = re.search(r"驱动电机峰值功率[:：]?([\d/]+)\s?KW", text)
    if m:
        fields["motor_peak_power_kw"] = m.group(1)

    m = re.search(r"额定电压[:：]?(\d+\.?\d*)\s?V", text)
    if m:
        fields["battery_voltage_v"] = m.group(1)

    m = re.search(r"额定容量[:：]?(\d+\.?\d*)\s?AH", text)
    if m:
        fields["battery_capacity_ah"] = m.group(1)

    m = re.search(r"最大允许总质量[:：]?(\d+)\s?KG", text)
    if m:
        fields["max_allowed_total_mass_kg"] = m.group(1)

    m = re.search(r"乘坐人数[:：]?(\d+)", text)
    if m:
        fields["seats"] = m.group(1)

    m = re.search(r"制造年月[:：]?(\d{4}[-年]\d{2})", text)
    if m:
        fields["manufacture_date"] = m.group(1).replace("年", "-")

    return fields, raw_text


# =========================
# MAIN PROCESS
# =========================

def process_image(image_path):
    img = preprocess_image(image_path)

    lines = run_ocr(img)

    fields, raw_text = extract_fields(lines)

    boxed_image = draw_ocr_result_from_img(img, image_path, lines)

    output = {
        "raw_text": raw_text,
        "fields": fields,
        "ocr_lines": lines,
        "boxed_image": boxed_image
    }

    return output


# =========================
# RUN
# =========================

if __name__ == "__main__":
    result = process_image(IMAGE_PATH)

    # print(json.dumps(result, ensure_ascii=False, indent=2))

    output_file = os.path.splitext(IMAGE_PATH)[0] + "-result.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved JSON to {output_file}")
    print(f"Saved OCR image to {result['boxed_image']}")