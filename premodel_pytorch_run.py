# infer_from_pth.py
import sys, torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

REPO = Path("/home/jetson/github/Steel-LLM")
sys.path.append(str(REPO / "pretrain_modify_from_TinyLlama/model/steel_modify_from_qwen_1_5"))
from modeling_steel import SteelForCausalLM

# ─── 路径 ───
mp       = REPO / "pretrain_modify_from_TinyLlama/model/steel_modify_from_qwen_1_5"
tok_path = REPO / "pretrain_modify_from_TinyLlama/model/tokenizer_from_qwen_moe_chat"
ckpt     = REPO / "data/gu_data/ckpt/steel_llm/step-018000-iter-144000-ckpt/state.pth"

# ─── ① 建空模型 ───
config = AutoConfig.from_pretrained(mp, trust_remote_code=True)
model  = SteelForCausalLM(config)
# ─── ② 加载权重 ───
sd = torch.load(ckpt, weights_only=False)["model"]   # ← OrderedDict
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"missing={len(missing)}, unexpected={len(unexpected)}")
if missing:    print("  missing:", missing[:5])
if unexpected: print("  unexpected:", unexpected[:5])

# ─── ③ 精度 + 设备 ───
model = model.to(torch.bfloat16).to("cuda").eval()

# ─── ④ tokenizer（用本地 tokenizer 目录，不是 mp）───
tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
print(f"tokenizer vocab = {tok.vocab_size}")   # 期望 151643 或 151936

# ─── ⑤ 手写 greedy 推理 ───
@torch.no_grad()
def greedy(prompt, n=40):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    prompt_len = ids.shape[1]
    for _ in range(n):
        #ctx = ids[:, -config.max_position_embeddings:]                # 截到 block_size，避免超长
        ctx = ids[:, -256:] #这里有个硬编码
        logits = model(ctx).logits[:, -1, :]             # ★ 必须 .logits
        nid = logits.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nid], dim=-1)
        if nid.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0][prompt_len:], skip_special_tokens=True)

for p in ["人工智能的未来是", "The capital of France is"]:
    print(f"\n[PROMPT] {p}")
    print(f"[OUTPUT] {greedy(p)}")
