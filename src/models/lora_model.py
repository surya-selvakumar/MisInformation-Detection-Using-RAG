from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_lora_model(base_model: str, r=16, alpha=32, dropout=0.05, target_modules=("q_proj","v_proj")):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
    peft_cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM", target_modules=list(target_modules))
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model, tok
