import requests
import os
import base64
import json
import time
from dotenv import load_dotenv

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")


def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def inpaint(
    window_img, reference_img, mask_img, prompt, output_file="images/output.png"
):
    url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "version": "c27ef18a0f5b1c41922278b55289dc2fe4ed4b25a0757df7e0ea891b9223c8d9",  # уточним актуально
        "input": {
            "image": f"data:image/png;base64,{encode_image_base64(window_img)}",
            "mask": f"data:image/png;base64,{encode_image_base64(mask_img)}",
            "ref_image": f"data:image/png;base64,{encode_image_base64(reference_img)}",
            "prompt": prompt,
        },
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code != 201:
        print(f"❌ Ошибка Replicate API: {response.status_code} - {response.text}")
        return

    prediction = response.json()
    prediction_url = f"{url}/{prediction['id']}"
    print("🕐 Ждём результат...")

    # Ожидание результата генерации
    while True:
        prediction = requests.get(prediction_url, headers=headers).json()
        status = prediction["status"]
        if status == "succeeded":
            break
        elif status == "failed":
            print("❌ Ошибка генерации.")
            return
        time.sleep(3)

    output_url = prediction["output"][0]
    img_data = requests.get(output_url).content
    with open(output_file, "wb") as f:
        f.write(img_data)
    print(f"✅ Изображение сохранено как {output_file}")
