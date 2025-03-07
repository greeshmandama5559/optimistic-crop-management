import os
import google.generativeai as genai
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    raise ValueError("Gemini API Key is not set! Please set it as an environment variable.")

#use gemini models if not work
#gemini-1.5-pro
#gemini-1.5-pro-latest
#gemini-2.0-flash
#gemini-pro

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-pro")

def generate(prompt):
    if not prompt:
        return "Invalid prompt"
    
    try:
        response = model.generate_content(prompt, timeout=8)
        return response.text.strip()
    
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        return None

