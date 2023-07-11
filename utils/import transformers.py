import transformers
import peft
peft_model_id = "/root/dolly/alpaca-lora-dolly-2.0"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
batch = tokenizer(Hello: ", return_tensors='pt')
with torch.cuda.amp.autocast():
  output_tokens = model.generate(**batch, max_new_tokens=50)
print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))