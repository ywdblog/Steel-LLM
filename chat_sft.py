import sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
BASE = Path("/home/jetson/github/Steel-LLM/data/gu_data/ckpt/pre_model")
ADAPTER = Path("/home/jetson/github/Steel-LLM/sft_output")
sys.path.insert(0, str(BASE))
base = AutoModelForCausalLM.from_pretrained(str(BASE), trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
model = PeftModel.from_pretrained(base, str(ADAPTER))
tok = AutoTokenizer.from_pretrained(str(BASE), trust_remote_code=True)
text = tok.apply_chat_template([{"role":"user","content":"你是谁开发的"}], tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids.to("cuda")
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
