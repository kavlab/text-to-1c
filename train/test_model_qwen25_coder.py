import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def generate(pipe, sample):
    prompt = pipe.tokenizer.apply_chat_template(
        sample["messages"][:2], tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.pad_token_id,
    )

    predicted_query = (
        outputs[0]["generated_text"][len(prompt) :].strip().replace("  ", " ")
    )
    reference_query = sample["messages"][2]["content"].replace("  ", " ")

    return {"predicted_query": predicted_query, "reference_query": reference_query}


peft_model_id = "./checkpoints/qwen25-coder-1.5b-sft"

model = AutoModelForCausalLM.from_pretrained(
    peft_model_id, device_map="auto", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

eval_dataset = load_dataset("json", data_files="data/ru_test.json", split="train")

predicted_queries = []
reference_queries = []

number_of_eval_samples = eval_dataset.shape[0]

for s in tqdm(eval_dataset.select(range(number_of_eval_samples))):
    result = generate(pipe, s)
    predicted_queries.append(result["predicted_query"])
    reference_queries.append(result["reference_query"])

del model
del tokenizer
torch.cuda.empty_cache()

df = pd.DataFrame({"ref": reference_queries, "pred": predicted_queries})
df.to_csv("data/results/pred_qwen25_coder_1_5b.csv", sep=";")
