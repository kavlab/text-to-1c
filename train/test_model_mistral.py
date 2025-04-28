import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Путь к директории с LoRA-файлами (из trainer.save_model())
peft_model_id = "mistral-sft-checkpoints"

# Загрузка модели и токенизатора
model = AutoModelForCausalLM.from_pretrained(
    peft_model_id, device_map="auto", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Загрузка тестового датасета
eval_dataset = load_dataset("json", data_files="data/ru_test.json", split="train")

# Функция генерации запроса
def generate(sample):
    messages = sample["messages"]
    prompt_parts = []

    # Добавим system, если есть
    for message in messages:
        if message["role"] == "system":
            prompt_parts.append(f"[SYSTEM]\n{message['content'].strip()}\n")
            break

    # Instruction + Response (первая пара)
    for i, message in enumerate(messages):
        if message["role"] == "user":
            prompt_parts.append(f"### Instruction:\n{message['content'].strip()}\n")
            break

    prompt_parts.append("### Response:\n")  # начало генерации

    prompt = "\n".join(prompt_parts)

    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Убираем префикс prompt, оставляем только сгенерированный ответ
    predicted_query = outputs[0]["generated_text"][len(prompt):].strip()
    reference_query = next(
        (m["content"] for m in messages if m["role"] == "assistant"), ""
    ).strip()

    return {"predicted_query": predicted_query, "reference_query": reference_query}

# Обработка выборки
predicted_queries = []
reference_queries = []

for s in tqdm(eval_dataset):
    result = generate(s)
    predicted_queries.append(result["predicted_query"])
    reference_queries.append(result["reference_query"])

# Очистка GPU
del model
del tokenizer
torch.cuda.empty_cache()

# Сохранение результатов
df = pd.DataFrame({"ref": reference_queries, "pred": predicted_queries})
df.to_csv("data/predicted_queries_mistral.csv", sep=";")
