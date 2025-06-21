# sam_mask.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def generate_mask(image_path, output_mask="images/mask.png"):
    api_url = "https://api-inference.huggingface.co/models/facebook/sam-vit-huge"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    with open(image_path, "rb") as f:
        response = requests.post(api_url, headers=headers, data=f)

    if response.status_code == 200:
        with open(output_mask, "wb") as out_file:
            out_file.write(response.content)
        print(f"✅ Маска сохранена как {output_mask}")
    else:
        print(f"❌ Ошибка SAM API: {response.status_code} - {response.text}")
