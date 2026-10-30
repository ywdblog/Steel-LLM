import sys, shutil, torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

sys.path.append("./pretrain_modify_from_TinyLlama/model/steel_modify_from_qwen_1_5")
from modeling_steel import SteelForCausalLM

mp           = Path("./pretrain_modify_from_TinyLlama/model/steel_modify_from_qwen_1_5")
tok_path     = Path("./pretrain_modify_from_TinyLlama/model/tokenizer_from_qwen_moe_chat")   

ckpt_dir     = Path("./data/gu_data/ckpt/steel_llm/step-017999-iter-144000-ckpt")
ckpt_path    = ckpt_dir / "state.pth"   
                                                    
output_hf_dir = Path("./data/gu_data/ckpt/pre_model")

# 1. 空模型
config = AutoConfig.from_pretrained(mp, trust_remote_code=True)
model = SteelForCausalLM(config)

# 2. 权重
# state["model"]，忽略 optimizer、hparams、iter_num、step_count
sd = torch.load(ckpt_path, weights_only=False)["model"]
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"missing={len(missing)}, unexpected={len(unexpected)}")
if missing: print("  missing:", missing[:5])

model = model.to(torch.bfloat16)

# 3. 保存 HF 格式
output_hf_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(output_hf_dir, safe_serialization=True)

# 4. tokenizer（用本地 tokenizer 路径）
tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
tokenizer.save_pretrained(output_hf_dir)
print(f"tokenizer vocab = {tokenizer.vocab_size}")   

# 5. 把自定义代码文件也拷过去，否则 HF 加载不了（trust_remote_code）
for f in ["configuration_steel.py", "modeling_steel.py", "softmoe_v3.py"]:
    src = mp / f
    if src.exists():
        shutil.copy(src, output_hf_dir / f)
        print(f"copied: {f}")
    else:
        print(f" missing: {f}")

print(f"\n HF 模型已保存到: {output_hf_dir}")
