"""
multi_model_chatbot.py
-----------------------
A simple chatbot interface to send a message to multiple LLM APIs:
- OpenAI GPT
- Groq API
- Google Gemini API

Usage:
    python multi_model_chatbot.py
"""

import os
import requests
import json

# ==============================
# CONFIGURATION / API KEYS
# ==============================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "<your-openai-key>")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "<your-groq-key>")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "<your-gemini-key>")

# ==============================
# FUNCTION: OpenAI GPT
# ==============================
def chat_with_openai(message: str, model: str = "gpt-4o") -> str:
    """
    Send a chat message to OpenAI GPT-style API and return the response text.
    Uses REST API style.  
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": message}
        ],
        "max_tokens": 256,
        "temperature": 0.7
    }
    resp = requests.post(url, headers=headers, json=body)
    resp.raise_for_status()
    data = resp.json()
    # pick the first choice
    return data["choices"][0]["message"]["content"]

# ==============================
# FUNCTION: Groq API
# ==============================
def chat_with_groq(message: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Send a chat message to Groq API and return the response text.
    Groq supports OpenAI-compatible chat completions endpoint. :contentReference[oaicite:4]{index=4}
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": message}
        ],
        "max_tokens": 256,
        "temperature": 0.7
    }
    resp = requests.post(url, headers=headers, json=body)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ==============================
# FUNCTION: Google Gemini API
# ==============================
def chat_with_gemini(message: str, model: str = "gemini-2.5-pro-exp-03-25") -> str:
    """
    Send a chat message to Google Gemini API and return the response text.  
    Uses Google GenAI SDK or REST endpoint. :contentReference[oaicite:5]{index=5}
    """
    # Example using REST (simplified)
    url = "https://generativelanguage.googleapis.com/v1/models/{model}:generate".format(model=model)
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "prompt": message,
        "temperature": 0.7,
        "maxOutputTokens": 256
    }
    resp = requests.post(url, headers=headers, json=body)
    resp.raise_for_status()
    data = resp.json()
    # The JSON structure may vary — adjust based on actual response
    return data["candidates"][0]["output"]

# ==============================
# MAIN FUNCTION
# ==============================
def main():
    print("Multi-model chatbot: choose a service.")
    message = input("Enter your message: ")
    print("\n→ Asking OpenAI GPT …")
    try:
        response_openai = chat_with_openai(message)
        print("OpenAI GPT response:", response_openai)
    except Exception as e:
        print("Error calling OpenAI:", e)

    print("\n→ Asking Groq API …")
    try:
        response_groq = chat_with_groq(message)
        print("Groq API response:", response_groq)
    except Exception as e:
        print("Error calling Groq:", e)

    print("\n→ Asking Google Gemini …")
    try:
        response_gemini = chat_with_gemini(message)
        print("Google Gemini response:", response_gemini)
    except Exception as e:
        print("Error calling Gemini:", e)

if __name__ == "__main__":
    main()
