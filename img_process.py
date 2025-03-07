import os
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import requests

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    raise ValueError("Gemini API Key is not set! Please set it as an environment variable.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-pro")

print("done")

def is_leaf(image_path):
    if not os.path.exists(image_path):
        return "Error: Image file not found."

    try:
        img = Image.open(image_path)

        prompt = "Is this image a plant leaf? Respond with 'Yes' if the image is leaf, Respond with 'No'(and about the image) only if the image is not leaf."

        response = model.generate_content([prompt, img])

        if response and response.text:
            result = response.text.strip()
            return result
        else:
            return None

    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        return None