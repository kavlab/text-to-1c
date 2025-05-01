import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate(row, tokenizer, model):
    prompt = tokenizer.apply_chat_template(
        row["messages"][:2],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return {
        "thinking_content": thinking_content,
        "predicted_query": content,
        "reference_query": row["messages"][2]["content"],
    }


peft_model_id = "checkpoints/qwen3-1_7b-sft"

model = AutoModelForCausalLM.from_pretrained(
    peft_model_id, device_map="auto", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

eval_dataset = load_dataset("json", data_files="dataset/data/ru_test.json", split="train")

predicted_queries = []
reference_queries = []

for s in tqdm(eval_dataset):
    result = generate(s, tokenizer, model)
    predicted_queries.append(result["predicted_query"])
    reference_queries.append(result["reference_query"])

del model
del tokenizer
torch.cuda.empty_cache()

df = pd.DataFrame({"ref": reference_queries, "pred": predicted_queries})
df.to_csv("evaluate/results/pred_qwen3_1.7b.csv", sep=";")
