import os
from contextlib import asynccontextmanager

import api.local_model as local_model
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL")

if not OLLAMA_SERVER_URL:
    raise RuntimeError("Environment variable OLLAMA_SERVER_URL must be set")

model = None
tokenizer = None
table_keyword_sequences = None
dot_sequence = None
normal_keyword_sequences = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    global table_keyword_sequences, dot_sequence, normal_keyword_sequences

    model, tokenizer = local_model.load_model()

    table_keyword_sequences, dot_sequence, normal_keyword_sequences = (
        local_model.prepare_sequences(tokenizer)
    )

    yield

    del model
    del tokenizer


app = FastAPI(title="SQL Translator API", lifespan=lifespan)


class TranslationRequest(BaseModel):
    model: str
    schema_db: str
    question: str


class TranslationResponse(BaseModel):
    content: str


@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest):
    """
    Переводит вопрос на SQL-запрос на основе переданной схемы БД.
    """
    payload = {
        "model": req.model,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a text to SQL query translator. "
                    "Users will ask you questions in Russian and you will generate a SQL query based on the provided SCHEMA.\n"
                    "Example of a condition with a field of type (Булево):\n ГДЕ T1.Клиент = True\n ГДЕ T1.Поставщик = False\n"
                    f"SCHEMA: {req.schema_db}"
                ),
            },
            {"role": "user", "content": req.question},
        ],
    }

    if req.model == "Qwen3-1.7B-Text-to-1CSQL":
        query = local_model.predict(
            payload["messages"],
            req.schema_db,
            model,
            tokenizer,
            table_keyword_sequences,
            dot_sequence,
            normal_keyword_sequences
        )
        return TranslationResponse(content=query)

    print(req)

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
        raise HTTPException(status_code=500, detail=f"{e}")

    try:
        data = response.json()
    except requests.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{response.text}")

    assistant_message = data.get("message", {}).get("content")
    if assistant_message is None:
        raise HTTPException(
            status_code=500, detail="Invalid response from Ollama server"
        )

    return TranslationResponse(content=assistant_message)
