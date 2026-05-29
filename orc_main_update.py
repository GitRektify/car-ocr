import cv2
import re
import json
import os
import numpy as np
import unicodedata
from paddleocr import PaddleOCR


IMAGE_PATH = "VINs/VIN (1).jpg"
MODEL_DIR = "models"

DET_MODEL_DIR = f"{MODEL_DIR}/ch_PP-OCRv4_det_infer"
REC_MODEL_DIR = f"{MODEL_DIR}/ch_PP-OCRv4_rec_infer"
CLS_MODEL_DIR = f"{MODEL_DIR}/ch_ppocr_mobile_v2.0_cls_infer"

USE_GPU = False
VIN_REGEX = r"[A-HJ-NPR-Z0-9]{17}"


ocr = PaddleOCR(
    use_angle_cls=True,
    lang="ch",
    det_model_dir=DET_MODEL_DIR,
    rec_model_dir=REC_MODEL_DIR,
    cls_model_dir=CLS_MODEL_DIR,
    use_gpu=USE_GPU,
    show_log=False,
    drop_score=0.25
)


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.upper()
    text = text.replace(" ", "")
    text = text.replace(":", "：")
    text = text.replace("；", "：")
    text = text.replace("_", "")
    text = text.replace("车新识别代号", "车辆识别代号")
    text = text.replace("创造国", "制造国")
    text = text.replace("M L", "ML")
    text = text.replace("K W", "KW")
    return text


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = img.shape[:2]

    if w < 1000:
        scale = 1000 / w
        img = cv2.resize(img, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=8,
        templateWindowSize=7,
        searchWindowSize=21
    )

    blur = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharp = cv2.addWeighted(denoised, 1.4, blur, -0.4, 0)

    return sharp


def load_original_same_size(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = img.shape[:2]

    if w < 1000:
        scale = 1000 / w
        img = cv2.resize(img, None, fx=scale, fy=scale)

    return img


def box_info(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]

    x1 = int(min(xs))
    y1 = int(min(ys))
    x2 = int(max(xs))
    y2 = int(max(ys))

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": (x1 + x2) // 2,
        "cy": (y1 + y2) // 2,
        "w": x2 - x1,
        "h": y2 - y1
    }


def run_ocr(img):
    result = ocr.ocr(img, cls=True)
    lines = []

    if not result or not result[0]:
        return lines

    for item in result[0]:
        box = item[0]
        text = item[1][0]
        score = float(item[1][1])

        if score < 0.25:
            continue

        info = box_info(box)

        lines.append({
            "text": text,
            "norm_text": normalize_text(text),
            "score": score,
            "box": box,
            **info
        })

    lines.sort(key=lambda x: (x["cy"], x["x1"]))
    return lines


def find_nearest_value(lines, label_keywords, value_regex=None, max_y_gap=45):
    label_candidates = []

    for line in lines:
        t = line["norm_text"]
        if any(k in t for k in label_keywords):
            label_candidates.append(line)

    if not label_candidates:
        return None

    best_value = None
    best_dist = 999999

    for label in label_candidates:
        label_text = label["norm_text"]

        if value_regex:
            m = re.search(value_regex, label_text)
            if m:
                return m.group(1) if m.groups() else m.group(0)

        for line in lines:
            if line is label:
                continue

            y_gap = abs(line["cy"] - label["cy"])
            right_side = line["x1"] >= label["x1"] - 10

            if re.search(r"[\u4e00-\u9fff]{3,}", line["norm_text"]):
                continue

            if y_gap <= max_y_gap and right_side:
                candidate = line["norm_text"]

                if value_regex:
                    m = re.search(value_regex, candidate)
                    if not m:
                        continue
                    candidate = m.group(1) if m.groups() else m.group(0)

                dist = abs(line["cy"] - label["cy"]) + max(0, line["x1"] - label["x2"])

                if dist < best_dist:
                    best_dist = dist
                    best_value = candidate

    return best_value


def extract_fields(lines):
    raw_text = "\n".join([x["text"] for x in lines])
    text = normalize_text(raw_text)

    fields = {}

    vin_match = re.search(VIN_REGEX, text)
    if vin_match:
        fields["vin"] = vin_match.group(0)

    vehicle_model = find_nearest_value(lines, ["整车型号"], r"([A-Z0-9]{6,})")
    if vehicle_model:
        fields["vehicle_model"] = vehicle_model

    engine_model = find_nearest_value(lines, ["发动机型号"], r"([A-Z0-9]{3,})")
    if engine_model:
        fields["engine_model"] = engine_model

    engine_displacement = None
    m = re.search(r"发动机排量：?(\d+\.?\d*)\s*(ML|升|L)?", text)
    if m:
        engine_displacement = m.group(1)
    else:
        engine_displacement = find_nearest_value(lines, ["发动机排量"], r"(\d+\.?\d*)")
    if engine_displacement:
        fields["engine_displacement"] = engine_displacement

    engine_power = find_nearest_value(lines, ["发动机最大净功率", "最大净功率"], r"(\d+\.?\d*)")
    if engine_power:
        fields["engine_max_net_power_kw"] = engine_power

    max_mass = find_nearest_value(lines, ["最大允许总质量"], r"(\d{3,5})")
    if max_mass:
        fields["max_allowed_total_mass_kg"] = max_mass

    seats = None
    m = re.search(r"(乘坐人数|乘员数)：?(\d+)", text)
    if m:
        seats = m.group(2)
    else:
        seats = find_nearest_value(lines, ["乘坐人数", "乘员数"], r"(\d+)")
    if seats:
        fields["seats"] = seats

    date = None
    m = re.search(r"(\d{4})[-年](\d{1,2})", text)
    if m:
        date = f"{m.group(1)}-{int(m.group(2)):02d}"
    else:
        date = find_nearest_value(lines, ["制造年月"], r"(\d{4}[-年]\d{1,2})")
        if date:
            date = date.replace("年", "-")

    if date:
        fields["manufacture_date"] = date

    drive_motor = find_nearest_value(lines, ["驱动电机型号"], r"([A-Z0-9/]{5,})")
    if drive_motor:
        fields["drive_motor_model"] = drive_motor

    motor_power = find_nearest_value(lines, ["驱动电机峰值功率"], r"([\d/]+)")
    if motor_power:
        fields["motor_peak_power_kw"] = motor_power

    battery_voltage = find_nearest_value(lines, ["额定电压"], r"(\d+\.?\d*)")
    if battery_voltage:
        fields["battery_voltage_v"] = battery_voltage

    battery_capacity = find_nearest_value(lines, ["额定容量"], r"(\d+\.?\d*)")
    if battery_capacity:
        fields["battery_capacity_ah"] = battery_capacity

    return fields, raw_text


def draw_ocr_result_from_img(img, image_path, lines):
    if len(img.shape) == 2:
        draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        draw_img = img.copy()

    for idx, line in enumerate(lines, start=1):
        pts = np.array(line["box"], dtype=np.int32)
        cv2.polylines(draw_img, [pts], True, (0, 255, 0), 2)

        x = int(line["x1"])
        y = int(line["y1"]) - 8

        if y < 20:
            y = int(line["y1"]) + 25

        label = f"{idx}: {line['score']:.2f}"

        cv2.putText(
            draw_img,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    output_img = os.path.splitext(image_path)[0] + "-ocr-boxes.jpg"
    cv2.imwrite(output_img, draw_img)

    return output_img


def find_vin_line(lines):
    candidates = []

    for line in lines:
        t = normalize_text(line["text"])

        if re.search(VIN_REGEX, t):
            score = 0

            if "VIN" in t:
                score += 100

            if "车辆识别代号" in t:
                score += 100

            if line["x1"] < 400:
                score += 20

            if line["x1"] > 450 and line["cy"] < 250:
                score -= 80

            candidates.append((score, line))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def get_vin_roi_from_line(line, img_shape):
    h_img, w_img = img_shape[:2]

    x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
    t = normalize_text(line["text"])

    if "VIN" in t:
        x1 = int(x1 + (x2 - x1) * 0.22)

    if "车辆识别代号" in t:
        x1 = int(x1 + (x2 - x1) * 0.33)

    pad_x = 8
    pad_y = 5

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w_img, x2 + pad_x)
    y2 = min(h_img, y2 + pad_y)

    return x1, y1, x2, y2


def remove_vin_from_image(img, lines):
    draw_img = img.copy()

    vin_line = find_vin_line(lines)

    if vin_line is None:
        print("VIN line not found")
        return draw_img

    x1, y1, x2, y2 = get_vin_roi_from_line(vin_line, draw_img.shape)

    roi = draw_img[y1:y2, x1:x2].copy()

    if roi.size == 0:
        return draw_img

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    mask1 = cv2.inRange(gray, 145, 255)

    _, mask2 = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = cv2.bitwise_or(mask1, mask2)

    kernel_small = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    kernel_big = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel_big, iterations=2)

    clean_roi = cv2.inpaint(roi, mask, 5, cv2.INPAINT_TELEA)

    blur_roi = cv2.GaussianBlur(clean_roi, (3, 3), 0)

    alpha = mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    alpha = alpha[:, :, None]

    result_roi = (blur_roi * alpha + roi * (1 - alpha)).astype(np.uint8)

    draw_img[y1:y2, x1:x2] = result_roi

    return draw_img


def process_image(image_path):
    img_for_ocr = preprocess_image(image_path)

    lines = run_ocr(img_for_ocr)

    fields, raw_text = extract_fields(lines)

    original_img = load_original_same_size(image_path)

    removed_img = remove_vin_from_image(original_img, lines)

    output_removed_img = os.path.splitext(image_path)[0] + "-vin-removed.jpg"
    cv2.imwrite(output_removed_img, removed_img)

    boxed_image = draw_ocr_result_from_img(img_for_ocr, image_path, lines)

    output = {
        "raw_text": raw_text,
        "fields": fields,
        "ocr_lines": lines,
        "boxed_image": boxed_image,
        "vin_removed_image": output_removed_img
    }

    return output


if __name__ == "__main__":
    for i in range(1, 24):
        image_path = f"VINs/VIN ({i}).jpg"

        if not os.path.exists(image_path):
            print(f"Skip missing file: {image_path}")
            continue

        result = process_image(image_path)

        output_file = os.path.splitext(image_path)[0] + "-result.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(json.dumps(result["fields"], ensure_ascii=False, indent=2))
        print(f"Saved JSON to {output_file}")
        print(f"Saved OCR image to {result['boxed_image']}")
        print(f"Saved VIN removed image to {result['vin_removed_image']}")


    # result = process_image(IMAGE_PATH)

    # output_file = os.path.splitext(IMAGE_PATH)[0] + "-result.json"

    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=2)

    # print(json.dumps(result["fields"], ensure_ascii=False, indent=2))
    # print(f"\nSaved JSON to {output_file}")
    # print(f"Saved OCR image to {result['boxed_image']}")