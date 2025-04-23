import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTConfig, SFTTrainer


# Загрузка базовой модели и токенизатора
# model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
model_name = "Qwen/Qwen2.5-Coder-1.5B"
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

# Загружаем модель в 4-битном режиме для QLoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
)
model.config.pad_token_id = model_pad_token_id

# Загрузка и подготовка датасета
dataset = load_dataset("json", data_files="data/ru_train.json", split="train")


# Функция форматирования диалога в текст по шаблону ChatML
def format_prompt(row):
    # Применяем шаблон диалога токенизатора Qwen:
    text = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False
    )
    # В полученной строке находятся все сообщения (system, user, assistant)
    # в формате, понятном модели (включая специальные токены).
    return {"text": text}


# Преобразуем весь датасет: получаем поле "text" с готовой строкой диалога
dataset = dataset.map(format_prompt)

# Конфигурация обучения
output_dir = "./checkpoints/qwen25-coder-1.5b-sft"

sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    max_seq_length=8192,
)

# Настройка QLoRA
peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type=TaskType.CAUSAL_LM,
)

# Инициализация SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

# Запуск обучения
trainer.train()

# Сохранение обученной модели (LoRA адаптеров)
trainer.save_model()

del model
del tokenizer
torch.cuda.empty_cache()
