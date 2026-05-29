from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
import os


ORIGINAL_IMAGE = "source/VIN.jpg"
OUTPUT_IMAGE = "generated_VIN.jpg"

TARGET_VIN = "LNBRCFHK4SB421382"
CANVAS_SIZE = (423, 260)


FONT_CN = [
    r"C:/Windows/Fonts/simhei.ttf",
    r"C:/Windows/Fonts/msyhbd.ttc",
    r"C:/Windows/Fonts/msyh.ttc",
]

FONT_EN = [
    r"C:/Windows/Fonts/ARIALNBD.TTF",
    r"C:/Windows/Fonts/ARIALNB.TTF",
    r"C:/Windows/Fonts/arialbd.ttf",
]


def find_font(fonts):
    for f in fonts:
        if os.path.exists(f):
            return f
    raise FileNotFoundError("Font not found")


def font(fonts, size):
    return ImageFont.truetype(find_font(fonts), size)


def draw_compressed_text(
    img,
    xy,
    text,
    font_obj,
    fill=(250, 250, 242),
    x_scale=0.84,
    stroke=0
):
    x, y = xy

    dummy = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font_obj, stroke_width=stroke)

    tw = bbox[2] - bbox[0] + 8
    th = bbox[3] - bbox[1] + 8

    layer = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)

    ld.text(
        (4 - bbox[0], 4 - bbox[1]),
        text,
        font=font_obj,
        fill=fill,
        stroke_width=stroke,
        stroke_fill=fill
    )

    new_w = int(tw * x_scale)
    layer = layer.resize((new_w, th), Image.Resampling.LANCZOS)

    img.alpha_composite(layer, (x, y))


def prepare_background(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(CANVAS_SIZE, Image.Resampling.LANCZOS)

    # sample image is dark but still has visible texture
    img = Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), 0.38)
    img = img.filter(ImageFilter.GaussianBlur(0.18))

    arr = np.array(img)
    arr = cv2.convertScaleAbs(arr, alpha=1.10, beta=-4)

    return Image.fromarray(arr).convert("RGBA")


def generate_plate(original_path, fields, output_path):
    img = prepare_background(original_path)

    cn_title = font(FONT_CN, 20)
    cn_label = font(FONT_CN, 16)
    cn_small = font(FONT_CN, 16)

    vin_font = font(FONT_EN, 20)
    en_big = font(FONT_EN, 16)
    en_mid = font(FONT_EN, 14)

    # no new outline/border

    draw_compressed_text(img, (52, 8), "中国北京汽车集团越野车有限公司 制造", cn_title, x_scale=0.92)

    draw_compressed_text(img, (32, 38), "车辆识别代号", cn_label, x_scale=0.86)
    draw_compressed_text(img, (130, 38), fields["vin"], vin_font, x_scale=1.4)

    lx, lv = 31, 118
    rx, rv = 247, 345

    y = 70
    step = 21

    draw_compressed_text(img, (lx, y), "品牌:", cn_label, x_scale=0.86)
    draw_compressed_text(img, (lv, y + 2), fields["brand"], cn_small, x_scale=0.84)

    draw_compressed_text(img, (rx, y), "整车型号:", cn_label, x_scale=0.86)
    draw_compressed_text(img, (rv - 12, y + 1), fields["vehicle_model"], en_mid, x_scale=0.74)

    y += step
    draw_compressed_text(img, (lx, y), "发动机型号:", cn_label, x_scale=0.86)
    draw_compressed_text(img, (lv, y + 1), fields["engine_model"], en_big, x_scale=0.78)

    draw_compressed_text(img, (rx, y), "驱动电机型号:", cn_label, x_scale=0.86)
    draw_compressed_text(img, (rv, y + 1), fields["drive_motor_model_1"], en_big, x_scale=0.74)

    y += step
    draw_compressed_text(img, (lx, y), "发动机排量:", cn_label, x_scale=0.86)
    draw_compressed_text(img, (lv, y + 1), fields["engine_displacement"], en_big, x_scale=0.78)

    draw_compressed_text(img, (rv, y + 1), fields["drive_motor_model_2"], en_big, x_scale=0.74)

    y += step
    draw_compressed_text(img, (lx, y), "发动机最大净功率:", cn_label, x_scale=0.84)
    draw_compressed_text(img, (160, y + 1), fields["engine_power"], en_big, x_scale=0.78)

    draw_compressed_text(img, (rx, y), "最大允许总质量:", cn_label, x_scale=0.84)
    draw_compressed_text(img, (368, y + 1), fields["max_mass"], en_big, x_scale=0.78)

    y += step
    draw_compressed_text(img, (lx, y), "驱动电机峰值功率:", cn_label, x_scale=0.84)
    draw_compressed_text(img, (160, y + 1), fields["motor_power"], en_big, x_scale=0.78)

    y += step
    draw_compressed_text(img, (lx, y), "动力电池系统额定电压:", cn_label, x_scale=0.84)
    draw_compressed_text(img, (185, y + 1), fields["battery_voltage"], en_big, x_scale=0.78)

    y += step
    draw_compressed_text(img, (lx, y), "动力电池系统额定容量:", cn_label, x_scale=0.84)
    draw_compressed_text(img, (185, y + 1), fields["battery_capacity"], en_big, x_scale=0.78)

    y += step
    draw_compressed_text(img, (lx, y), "乘坐人数:", cn_label, x_scale=0.86)
    draw_compressed_text(img, (118, y + 1), fields["seats"], en_big, x_scale=0.78)

    draw_compressed_text(img, (247, y), "制造年月:", cn_label, x_scale=0.86)
    draw_compressed_text(img, (320, y + 1), fields["manufacture_date"], en_big, x_scale=1.2)

    arr = np.array(img.convert("RGB")).astype(np.int16)

    noise = np.random.normal(0, 1.1, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    arr = cv2.GaussianBlur(arr, (3, 3), 0)

    cv2.imwrite(output_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return output_path


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
        "motor_power": "153 / 250 kW",
        "battery_voltage": "355.2 V",
        "battery_capacity": "113.5 Ah",
        "seats": "5",
        "manufacture_date": "2025-07",
    }

    print("Saved:", generate_plate(ORIGINAL_IMAGE, fields, OUTPUT_IMAGE))