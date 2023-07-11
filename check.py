from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
device = "cuda"

peft_model_id = "nRuaif/70M_LIMARP"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = model.to(device)
model.eval()
inputs = tokenizer("<<SYSTEM>>\nSandy's Persona: also known as \"Squirt\", she a 7-year-old girl. She's small and agile enough to move around on a bed with ease. Her personality is curious and playful yet naive due to her young age. ", return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50, temperature=0.9)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
