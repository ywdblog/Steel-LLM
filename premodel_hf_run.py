import sys, torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer,AutoModelForCausalLM

 
REPO = Path("/home/jetson/github/Steel-LLM")
sys.path.append(str(REPO / "pretrain_modify_from_TinyLlama/model/steel_modify_from_qwen_1_5"))
from modeling_steel import SteelForCausalLM

mp  = REPO / "pretrain_modify_from_TinyLlama/model/steel_modify_from_qwen_1_5"
tok_path = REPO / "pretrain_modify_from_TinyLlama/model/tokenizer_from_qwen_moe_chat"

model_path = Path("./data/gu_data/ckpt/pre_model").resolve()

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,                      
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,                       
)

inputs = tokenizer('人工智能的未来是', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(pred[0], skip_special_tokens=True))
