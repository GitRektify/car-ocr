import cv2
import re
import json
import os
import numpy as np
import unicodedata
from paddleocr import PaddleOCR


IMAGE_PATH = f"VINs/VIN (1).jpg"
MODEL_DIR = "models"
TARGET_VIN = "LNLRR0008SB409736"  # new VIN you want to print

DET_MODEL_DIR = f"{MODEL_DIR}/ch_PP-OCRv4_det_infer"
REC_MODEL_DIR = f"{MODEL_DIR}/ch_PP-OCRv4_rec_infer"
CLS_MODEL_DIR = f"{MODEL_DIR}/ch_ppocr_mobile_v2.0_cls_infer"

USE_GPU = False


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
        info = box_info(box)

        if score < 0.25:
            continue

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

        # Case: label and value are in same OCR box
        if value_regex:
            m = re.search(value_regex, label_text)
            if m:
                return m.group(1) if m.groups() else m.group(0)

        for line in lines:
            if line is label:
                continue

            # same row or very close row
            y_gap = abs(line["cy"] - label["cy"])

            # value should usually be right side of label
            right_side = line["x1"] >= label["x1"] - 10

            # avoid using other Chinese labels as values
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

    # VIN: first search whole OCR text
    vin_match = re.search(r"[A-HJ-NPR-Z0-9]{17}", text)
    if vin_match:
        fields["vin"] = vin_match.group(0)

    vehicle_model = find_nearest_value(
        lines,
        ["整车型号"],
        r"([A-Z0-9]{6,})"
    )
    if vehicle_model:
        fields["vehicle_model"] = vehicle_model

    engine_model = find_nearest_value(
        lines,
        ["发动机型号"],
        r"([A-Z0-9]{3,})"
    )
    if engine_model:
        fields["engine_model"] = engine_model

    engine_displacement = None
    m = re.search(r"发动机排量：?(\d+\.?\d*)\s*(ML|升|L)?", text)
    if m:
        engine_displacement = m.group(1)
    else:
        engine_displacement = find_nearest_value(
            lines,
            ["发动机排量"],
            r"(\d+\.?\d*)"
        )
    if engine_displacement:
        fields["engine_displacement"] = engine_displacement

    engine_power = find_nearest_value(
        lines,
        ["发动机最大净功率", "最大净功率"],
        r"(\d+\.?\d*)"
    )
    if engine_power:
        fields["engine_max_net_power_kw"] = engine_power

    max_mass = find_nearest_value(
        lines,
        ["最大允许总质量"],
        r"(\d{3,5})"
    )
    if max_mass:
        fields["max_allowed_total_mass_kg"] = max_mass

    seats = None
    m = re.search(r"(乘坐人数|乘员数)：?(\d+)", text)
    if m:
        seats = m.group(2)
    else:
        seats = find_nearest_value(
            lines,
            ["乘坐人数", "乘员数"],
            r"(\d+)"
        )
    if seats:
        fields["seats"] = seats

    date = None

    m = re.search(r"(\d{4})[-年](\d{1,2})", text)
    if m:
        date = f"{m.group(1)}-{int(m.group(2)):02d}"
    else:
        date = find_nearest_value(
            lines,
            ["制造年月"],
            r"(\d{4}[-年]\d{1,2})"
        )
        if date:
            date = date.replace("年", "-")

    if date:
        fields["manufacture_date"] = date

    drive_motor = find_nearest_value(
        lines,
        ["驱动电机型号"],
        r"([A-Z0-9/]{5,})"
    )
    if drive_motor:
        fields["drive_motor_model"] = drive_motor

    motor_power = find_nearest_value(
        lines,
        ["驱动电机峰值功率"],
        r"([\d/]+)"
    )
    if motor_power:
        fields["motor_peak_power_kw"] = motor_power

    battery_voltage = find_nearest_value(
        lines,
        ["额定电压"],
        r"(\d+\.?\d*)"
    )
    if battery_voltage:
        fields["battery_voltage_v"] = battery_voltage

    battery_capacity = find_nearest_value(
        lines,
        ["额定容量"],
        r"(\d+\.?\d*)"
    )
    if battery_capacity:
        fields["battery_capacity_ah"] = battery_capacity

    return fields, raw_text


def draw_ocr_result_from_img(img, image_path, lines):
    if len(img.shape) == 2:
        draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        draw_img = img.copy()

    for idx, line in enumerate(lines, start=1):
        box = line["box"]
        score = line["score"]

        pts = np.array(box, dtype=np.int32)
        cv2.polylines(draw_img, [pts], True, (0, 255, 0), 2)

        x = int(line["x1"])
        y = int(line["y1"]) - 8
        if y < 20:
            y = int(line["y1"]) + 25

        label = f"{idx}: {score:.2f}"

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


def process_image(image_path):
    # OCR image
    img = preprocess_image(image_path)
    lines = run_ocr(img)
    fields, raw_text = extract_fields(lines)

    # Use original color image for final output
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resize original same as preprocess if needed
    h, w = original_img.shape[:2]
    if w < 1000:
        scale = 1000 / w
        original_img = cv2.resize(original_img, None, fx=scale, fy=scale)

    removed_img = remove_vin_from_image(original_img, lines)

    output_removed_img = os.path.splitext(image_path)[0] + "-vin-removed.jpg"
    cv2.imwrite(output_removed_img, removed_img)

    boxed_image = draw_ocr_result_from_img(img, image_path, lines)

    output = {
        "raw_text": raw_text,
        "fields": fields,
        "ocr_lines": lines,
        "boxed_image": boxed_image,
        "vin_removed_image": output_removed_img
    }

    return output

def replace_vin_on_image(img, lines, target_vin):
    if not target_vin:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    vin_line = None
    old_vin = None

    for line in lines:
        t = normalize_text(line["text"])
        m = re.search(r"[A-HJ-NPR-Z0-9]{17}", t)
        if m:
            vin_line = line
            old_vin = m.group(0)
            break

    if vin_line is None:
        return draw_img

    x1, y1, x2, y2 = vin_line["x1"], vin_line["y1"], vin_line["x2"], vin_line["y2"]

    # If OCR line includes label + VIN, cut only VIN region
    if "车辆识别代号" in normalize_text(vin_line["text"]):
        x1 = int(x1 + (x2 - x1) * 0.28)

    pad_x = 4
    pad_y = 4

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(draw_img.shape[1], x2 + pad_x)
    y2 = min(draw_img.shape[0], y2 + pad_y)

    roi = draw_img[y1:y2, x1:x2].copy()

    # Create text mask from bright original VIN
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Remove old VIN but keep background texture
    clean_roi = cv2.inpaint(roi, mask, 3, cv2.INPAINT_TELEA)

    h, w = clean_roi.shape[:2]

    # Use original VIN spacing
    char_count = 17
    char_w = w / char_count

    font = cv2.FONT_HERSHEY_TRIPLEX
    thickness = max(2, int(h * 0.08))

    # Find best font scale
    font_scale = 2.5
    while font_scale > 0.3:
        total_width = 0
        max_height = 0

        for ch in target_vin:
            (tw, th), _ = cv2.getTextSize(ch, font, font_scale, thickness)
            total_width += tw
            max_height = max(max_height, th)

        if total_width <= w * 0.95 and max_height <= h * 0.82:
            break

        font_scale -= 0.05

    result_roi = clean_roi.copy()

    for i, ch in enumerate(target_vin):
        cell_x1 = int(i * char_w)
        cell_x2 = int((i + 1) * char_w)

        (tw, th), baseline = cv2.getTextSize(ch, font, font_scale, thickness)

        tx = cell_x1 + int((cell_x2 - cell_x1 - tw) / 2)
        ty = int((h + th) / 2) - 2

        # soft dark shadow
        cv2.putText(
            result_roi,
            ch,
            (tx + 2, ty + 2),
            font,
            font_scale,
            (25, 25, 25),
            thickness + 1,
            cv2.LINE_AA
        )

        # white VIN text
        cv2.putText(
            result_roi,
            ch,
            (tx, ty),
            font,
            font_scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA
        )

    # Slight blur to match camera/photo quality
    result_roi = cv2.GaussianBlur(result_roi, (3, 3), 0)

    draw_img[y1:y2, x1:x2] = result_roi

    return draw_img

def remove_vin_from_image(img, lines):
    draw_img = img.copy()

    vin_line = None

    for line in lines:
        t = normalize_text(line["text"])
        if re.search(r"[A-HJ-NPR-Z0-9]{17}", t):
            vin_line = line
            break

    if vin_line is None:
        print("VIN line not found")
        return draw_img

    x1, y1, x2, y2 = vin_line["x1"], vin_line["y1"], vin_line["x2"], vin_line["y2"]

    if "车辆识别代号" in normalize_text(vin_line["text"]):
        x1 = int(x1 + (x2 - x1) * 0.28)

    pad_x = 12
    pad_y = 8

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(draw_img.shape[1], x2 + pad_x)
    y2 = min(draw_img.shape[0], y2 + pad_y)

    roi_h = y2 - y1
    roi_w = x2 - x1

    # sample clean background below VIN
    sample_y1 = min(draw_img.shape[0] - roi_h, y2 + 8)
    sample_y2 = sample_y1 + roi_h

    bg_patch = draw_img[sample_y1:sample_y2, x1:x2].copy()

    if bg_patch.shape[:2] != (roi_h, roi_w):
        bg_color = np.median(draw_img[y1:y2, x1:x2], axis=(0, 1)).astype(np.uint8)
        bg_patch = np.full((roi_h, roi_w, 3), bg_color, dtype=np.uint8)

    bg_patch = cv2.GaussianBlur(bg_patch, (15, 15), 0)

    # blend edges so rectangle is less obvious
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.rectangle(mask, (4, 4), (roi_w - 5, roi_h - 5), 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    alpha = mask.astype(float) / 255.0
    alpha = alpha[:, :, None]

    original_roi = draw_img[y1:y2, x1:x2].copy()
    blended = (bg_patch * alpha + original_roi * (1 - alpha)).astype(np.uint8)

    draw_img[y1:y2, x1:x2] = blended

    return draw_img

if __name__ == "__main__":
    for i in range(1, 24):
        IMAGE_PATH = f"VINs/VIN ({i}).jpg"
        result = process_image(IMAGE_PATH)

        output_file = os.path.splitext(IMAGE_PATH)[0] + "-result.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(json.dumps(result["fields"], ensure_ascii=False, indent=2))
        print(f"\nSaved JSON to {output_file}")
        print(f"Saved OCR image to {result['boxed_image']}")


    # result = process_image(IMAGE_PATH)

    # output_file = os.path.splitext(IMAGE_PATH)[0] + "-result.json"

    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=2)

    # print(json.dumps(result["fields"], ensure_ascii=False, indent=2))
    # print(f"\nSaved JSON to {output_file}")
    # print(f"Saved OCR image to {result['boxed_image']}")