from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
import os


# =========================
# CONFIG
# =========================

ORIGINAL_IMAGE = "VINs/VIN (1).jpg"
OUTPUT_IMAGE = "generated_plate_photo_style.jpg"

FONT_CN = "C:/Windows/Fonts/simhei.ttf"
FONT_EN = "C:/Windows/Fonts/arialbd.ttf"

TARGET_VIN = "LNLRR0008SB409736"


# =========================
# FONT HELPERS
# =========================

def load_font(path, size):
    return ImageFont.truetype(path, size)


def draw_text(draw, xy, text, font, fill=(245, 245, 245), shadow=True):
    x, y = xy

    if shadow:
        draw.text((x + 2, y + 2), text, font=font, fill=(20, 20, 20))

    draw.text((x, y), text, font=font, fill=fill)


# =========================
# BACKGROUND PROCESSING
# =========================

def prepare_background(original_path, size=(1536, 960)):
    img = Image.open(original_path).convert("RGB")
    img = img.resize(size)

    # Keep original texture but make old text less visible
    dark_overlay = Image.new("RGB", img.size, (0, 0, 0))
    img = Image.blend(img, dark_overlay, 0.48)

    # Slight blur to hide previous text but keep photo texture
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    # Contrast adjustment using OpenCV
    arr = np.array(img)
    arr = cv2.convertScaleAbs(arr, alpha=1.15, beta=-10)

    return Image.fromarray(arr)


# =========================
# MAIN GENERATOR
# =========================

def generate_plate_from_original(original_path, fields, output_path):
    W, H = 1536, 960

    img = prepare_background(original_path, (W, H))
    draw = ImageDraw.Draw(img)

    # Fonts
    cn_title = load_font(FONT_CN, 58)
    cn_label = load_font(FONT_CN, 38)
    cn_value = load_font(FONT_CN, 38)

    en_vin = load_font(FONT_EN, 72)
    en_big = load_font(FONT_EN, 46)
    en_mid = load_font(FONT_EN, 42)

    # Border
    draw.rounded_rectangle(
        [25, 25, W - 25, H - 25],
        radius=38,
        outline=(220, 220, 220),
        width=4
    )

    # Header
    draw_text(
        draw,
        (190, 45),
        "中国北京汽车集团越野车有限公司 制造",
        cn_title
    )

    # VIN row
    draw_text(draw, (120, 145), "车辆识别代号", cn_label)
    draw_text(draw, (460, 128), fields["vin"], en_vin)

    # Coordinates
    left_label_x = 120
    left_value_x = 410

    right_label_x = 850
    right_value_x = 1120

    y = 250
    step = 82

    # Row 1
    draw_text(draw, (left_label_x, y), "品牌:", cn_label)
    draw_text(draw, (left_value_x, y), fields.get("brand", ""), cn_value)

    draw_text(draw, (right_label_x, y), "整车型号:", cn_label)
    draw_text(draw, (right_value_x, y - 5), fields.get("vehicle_model", ""), en_mid)

    # Row 2
    y += step
    draw_text(draw, (left_label_x, y), "发动机型号:", cn_label)
    draw_text(draw, (left_value_x, y - 5), fields.get("engine_model", ""), en_big)

    draw_text(draw, (right_label_x, y), "驱动电机型号:", cn_label)
    draw_text(draw, (right_value_x, y - 5), fields.get("drive_motor_model_1", ""), en_big)

    # Row 3
    y += step
    draw_text(draw, (left_label_x, y), "发动机排量:", cn_label)
    draw_text(draw, (left_value_x, y - 5), fields.get("engine_displacement", ""), en_big)

    draw_text(draw, (right_value_x, y - 5), fields.get("drive_motor_model_2", ""), en_big)

    # Row 4
    y += step
    draw_text(draw, (left_label_x, y), "发动机最大净功率:", cn_label)
    draw_text(draw, (left_value_x + 90, y - 5), fields.get("engine_power", ""), en_big)

    draw_text(draw, (right_label_x, y), "最大允许总质量:", cn_label)
    draw_text(draw, (right_value_x + 80, y - 5), fields.get("max_mass", ""), en_big)

    # Row 5
    y += step
    draw_text(draw, (left_label_x, y), "驱动电机峰值功率:", cn_label)
    draw_text(draw, (left_value_x + 90, y - 5), fields.get("motor_power", ""), en_big)

    # Row 6
    y += step
    draw_text(draw, (left_label_x, y), "动力电池系统额定电压:", cn_label)
    draw_text(draw, (left_value_x + 160, y - 5), fields.get("battery_voltage", ""), en_big)

    # Row 7
    y += step
    draw_text(draw, (left_label_x, y), "动力电池系统额定容量:", cn_label)
    draw_text(draw, (left_value_x + 160, y - 5), fields.get("battery_capacity", ""), en_big)

    # Row 8
    y += step
    draw_text(draw, (left_label_x, y), "乘坐人数:", cn_label)
    draw_text(draw, (left_value_x, y - 5), fields.get("seats", ""), en_big)

    draw_text(draw, (850, y), "制造年月:", cn_label)
    draw_text(draw, (1080, y - 5), fields.get("manufacture_date", ""), en_big)

    # Final photo-like degradation
    arr = np.array(img)

    # small noise
    noise = np.random.normal(0, 2.2, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # small blur
    arr = cv2.GaussianBlur(arr, (3, 3), 0)

    cv2.imwrite(output_path, arr)

    return output_path


# =========================
# RUN
# =========================

if __name__ == "__main__":
    fields = {
        "vin": TARGET_VIN,
        "brand": "北京牌",
        "vehicle_model": "BJ2030V41MSHEV",
        "engine_model": "A156T2H",
        "drive_motor_model_1": "TZ180XS502A",
        "drive_motor_model_2": "/TZ220XY806",
        "engine_displacement": "1498 mL",
        "engine_power": "130 kW",
        "max_mass": "3140 kg",
        "motor_power": "153/250 kW",
        "battery_voltage": "355.2 V",
        "battery_capacity": "113.5 Ah",
        "seats": "5",
        "manufacture_date": "2025-07"
    }

    output = generate_plate_from_original(
        ORIGINAL_IMAGE,
        fields,
        OUTPUT_IMAGE
    )

    print("Saved:", output)