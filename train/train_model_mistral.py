import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTConfig, SFTTrainer

# Загрузка модели Mistral и токенизатора
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model_pad_token_id = tokenizer.pad_token_id

# Конфигурация квантования
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
)
model.config.pad_token_id = model_pad_token_id

# Загрузка и форматирование датасета
dataset = load_dataset("json", data_files="data/ru_train.json", split="train")

def format_prompt(row):
    messages = row["messages"]
    parts = []

    # Добавим system как контекст в начале, если есть
    for message in messages:
        if message["role"] == "system":
            system_text = message["content"].strip()
            parts.append(f"[SYSTEM]\n{system_text}\n")
            break

    # Теперь обрабатываем user/assistant по парам
    for i, message in enumerate(messages):
        role = message["role"]
        content = message["content"].strip()
        if role == "user":
            parts.append(f"### Instruction:\n{content}\n")
            # ищем следующий assistant
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                assistant_content = messages[i + 1]["content"].strip()
                parts.append(f"### Response:\n{assistant_content}\n")

    formatted = "\n".join(parts)
    return {"text": formatted}


dataset = dataset.map(format_prompt)

# Конфигурация обучения
output_dir = "./checkpoints/mistral-sft"

sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-5,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    max_seq_length=3072,
    dataset_text_field="text",
)

# Настройка LoRA
peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type=TaskType.CAUSAL_LM,
)

# Инициализация SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=peft_config,
)

# Запуск обучения
trainer.train()

# Сохранение модели
trainer.save_model()
