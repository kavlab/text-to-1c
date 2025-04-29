import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")

if not OLLAMA_SERVER_URL or not OLLAMA_MODEL_NAME:
    raise RuntimeError(
        "Environment variables OLLAMA_SERVER_URL and OLLAMA_MODEL_NAME must be set"
    )

app = FastAPI(title="Ollama SQL Translator API")


class TranslationRequest(BaseModel):
    schema: str
    question: str


class TranslationResponse(BaseModel):
    content: str


@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest):
    """
    Переводит вопрос на SQL-запрос на основе переданной схемы БД.
    """
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a text to SQL query translator. "
                    "Users will ask you questions in Russian and you will generate a SQL query based on the provided SCHEMA.\n"
                    f"SCHEMA: {req.schema}"
                ),
            },
            {"role": "user", "content": req.question},
        ],
    }

    try:
        response = requests.post(
            f"{OLLAMA_SERVER_URL}/api/chat", json=payload, timeout=90
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502, detail=f"Error communicating with Ollama server: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{e}"
        )

    try:
        data = response.json()
    except requests.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"{e}\n{response.text}"
        )
    
    assistant_message = data.get("message", {}).get("content")
    if assistant_message is None:
        raise HTTPException(
            status_code=500, detail="Invalid response from Ollama server"
        )

    return TranslationResponse(content=assistant_message)
