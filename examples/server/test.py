import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/Users/Daglas/dalong.datasets/chinese-mixtral-8x7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="cpu")

text = "中国的三皇五帝分别指的是谁？"
inputs = tokenizer(text, return_tensors="pt").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "/Users/Daglas/dalong.datasets/chinese-mixtral-8x7b"
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="cpu")

# text = "中国的三皇五帝分别指的是谁？"
# inputs = tokenizer(text, return_tensors="pt").to(0)

# outputs = model.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))