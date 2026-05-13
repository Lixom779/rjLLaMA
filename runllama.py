from Llamafunctions.Functions import MiniLlama
from safetensors.torch import load_file
from transformers import AutoTokenizer
import torch
import json
import os

torch.set_num_threads(48)

def load_weights(folder):
    
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    tensors = {}
    loaded_files = {}
    
    for name, file in weight_map.items():
        if file not in loaded_files:
            loaded_files[file] = load_file(os.path.join(folder, file))
        
        tensors[name] = loaded_files[file][name]
    
    return tensors
    
def map_weights(model, weights):
    #---embedding layer weights---------------------------------------------#
    model.embed.weight.data.copy_(weights["model.embed_tokens.weight"])
    #----------Transformer Layer weights------------------------------------#
    for i, layer in enumerate(model.layers):
        prefix = f"model.layers.{i}"
        
        layer.attn_norm.weight.data.copy_(weights[f"{prefix}.input_layernorm.weight"])
        
        layer.attn.wq.weight.data.copy_(weights[f"{prefix}.self_attn.q_proj.weight"])
        
        layer.attn.wk.weight.data.copy_(weights[f"{prefix}.self_attn.k_proj.weight"])
        
        layer.attn.wv.weight.data.copy_(weights[f"{prefix}.self_attn.v_proj.weight"])
        
        layer.attn.wo.weight.data.copy_(weights[f"{prefix}.self_attn.o_proj.weight"])
        
        layer.ffn_norm.weight.data.copy_(weights[f"{prefix}.post_attention_layernorm.weight"])
        
        layer.ffn.w_up.weight.data.copy_(weights[f"{prefix}.mlp.up_proj.weight"])
        
        layer.ffn.w_down.weight.data.copy_(weights[f"{prefix}.mlp.down_proj.weight"])
        
        layer.ffn.w_gate.weight.data.copy_(weights[f"{prefix}.mlp.gate_proj.weight"])
    #--------------output norm----------------------------------------------#
    model.norm.weight.data.copy_(weights["model.norm.weight"])
    #---------------output layer--------------------------------------------#
    model.lm_head.weight.data.copy_(weights["lm_head.weight"])
    #model.lm_head.weight.data = model.embed.weight.data

def generate(model, input_ids, stop_tokens, max_tokens = 200):
    
    model.eval()
    
    logits, kv_cache = model(input_ids, start_pos = 0)
    cur_pos = input_ids.shape[1]
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim = True)
    generated = [next_token]
    #print(torch.std(logits))
    
    for _ in range(max_tokens-1):
        logits, kv_cache = model(next_token, cur_pos, kv_cache)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim = True)
        if next_token.item() in stop_tokens:
            break
        generated.append(next_token)
        cur_pos += 1
        #print(torch.std(logits))
    
    return torch.cat(generated, dim=1)
    
with open("config.json", "r") as f:
        cfg = json.load(f)
    

stop_tokens = cfg["eos_token_id"]

model = MiniLlama(vocab_size=cfg["vocab_size"], dim=cfg["hidden_size"], n_layers=cfg["num_hidden_layers"], n_heads=cfg["num_attention_heads"], n_kv_heads=cfg["num_key_value_heads"], RoPE_scaling = cfg.get("rope_scaling",None), hidden_dim=cfg["intermediate_size"], RoPE_theta=cfg["rope_theta"])

weights = load_weights("D:/Bhanu/AI/Transformers/Buliding files for llama/Llama_weights_and_tokenizer/Llama3.18B_instruct/")

map_weights(model, weights)

#total = sum(p.numel() for p in model.parameters())
#print(total)

tokenizer = AutoTokenizer.from_pretrained("D:/Bhanu/AI/Transformers/Buliding files for llama/Llama_weights_and_tokenizer/Llama3.18B_instruct/")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Name the seven wonders of the world"}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).input_ids

print(tokenizer.decode(input_ids[0]))

#logits, kv_cache = model(input_ids, start_pos=0)

#topk = torch.topk(logits[:, -1, :], 5)

#print(topk)
output_ids = generate(model, input_ids, stop_tokens, max_tokens = 50)

print(tokenizer.decode(output_ids[0]))