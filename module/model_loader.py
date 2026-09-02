import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

import gc

from module.clip import CLIP
from module.vae import VAE_Encoder, VAE_Decoder
from module.diffusion import DiffusionModel

# 官方 Stable Diffusion 检查点中四个模块的前缀。
# 加载官方权重时需要按这些前缀拆分检查点来匹配键值，每个条目为（匹配的前缀，剥离后的前缀）。
MODULE_PREFIXES: dict[str, tuple[tuple[str, str], ...]] = {
    'diffusion': (('model.diffusion_model.', 'model.diffusion_model.'),),
    'encoder':   (('first_stage_model.encoder.', 'first_stage_model.encoder.'),
                  ('first_stage_model.quant_conv.', 'first_stage_model.')),
    'decoder':   (('first_stage_model.decoder.', 'first_stage_model.decoder.'),
                  ('first_stage_model.post_quant_conv.', 'first_stage_model.')),
    'clip':      (('cond_stage_model.transformer.text_model.', 'cond_stage_model.transformer.text_model.'),),
}

def convert_to_standard_weights(original_model: dict) -> dict[str, dict[str, torch.Tensor]]:
    """根据键前缀将官方Stable Diffusion检查点拆分为四个模块。"""
    converted: dict[str, dict[str, torch.Tensor]] = {}
    for name, prefixes in MODULE_PREFIXES.items():
        converted[name] = {}
        for match_prefix, strip_prefix in prefixes:
            for key, value in original_model.items():
                if key.startswith(match_prefix):
                    converted[name][key[len(strip_prefix):]] = value

    # position_ids 是一个常数缓冲区，将其删除以保证使用 strict=True 有效
    converted['clip'].pop('embeddings.position_ids', None)
    return converted

def convert_to_safetensors(ckps: dict[str, dict[str, torch.Tensor]], output_file: str) -> None:
    """
    根据 MODULE_PREFIXES 将四个模块的 state_dict 反向合并为官方格式检查点，并保存为 safetensors。

    ckps 结构:
        {
            'encoder': Encoder_VAE().state_dict(), 
            'decoder': Decoder_VAE().state_dict(), 
            'clip': CLIP().state_dict(), 
            'diffusion: DiffusionModel.state_dict()'
        }
    """
    original_model: dict[str, torch.Tensor] = {}
    for name, prefixes in MODULE_PREFIXES.items():
        state_dict = ckps[name]
        # 反向转化：为每个键补回前缀。（匹配前缀，剥离前缀）逆推可得——
        # 模块键需以 match_prefix 去掉 strip_prefix 后的局部前缀开头，补回的前缀即为 strip_prefix
        rules = sorted(
            ((match_prefix[len(strip_prefix):], strip_prefix) for match_prefix, strip_prefix in prefixes),
            key=lambda rule: len(rule[0]),
            reverse=True,  # 局部前缀长的规则优先（如 quant_conv 优先于通用规则）
        )
        for key, value in state_dict.items():
            for local_prefix, strip_prefix in rules:
                if key.startswith(local_prefix):
                    original_model[strip_prefix + key] = value.detach().cpu().contiguous()
                    break

    # position_ids 是常数缓冲区（不在 state_dict 中），补回以贴合官方文件格式，
    # convert_to_standard_weights 拆分时会将其删除，不影响严格加载
    original_model['cond_stage_model.transformer.text_model.embeddings.position_ids'] = torch.arange(77).unsqueeze(0)

    save_file(original_model, output_file)
    print(f"💾 Safetensors saved: {output_file}")

def load_weights(model: VAE_Encoder | VAE_Decoder | CLIP | DiffusionModel,
                 ckp_path: str | None = None, 
                 key_name: str | None = None,
                 device: str = "cuda") -> None:

    # 指定路径和键名才会加载权重
    if ckp_path and key_name:
        # 读取权重文件
        print(f"> Loading {key_name} from {ckp_path}...")
        if ckp_path.endswith('.safetensors'):
            ckp = load_file(ckp_path, device=device)
            ckp = convert_to_standard_weights(ckp)
        else:
            ckp = torch.load(ckp_path, map_location=device)
            
        # 加载状态字典
        model.load_state_dict(ckp[key_name], strict=True)
        
        # 释放内存
        del ckp
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print(f"> Skip loading weights...")
    

def get_encoder(ckp_path: str | None = None, 
                device: str = "cuda", 
                dtype: torch.dtype = torch.bfloat16) -> VAE_Encoder:
    
    print(f"🔄 Initializing encoder...")
    model = VAE_Encoder().to(device=device, dtype=dtype)
    load_weights(model=model, ckp_path=ckp_path, key_name="encoder", device=device)
    print("> Finished!")
    return model
    
def get_decoder(ckp_path: str | None = None, 
                device: str = "cuda", 
                dtype: torch.dtype = torch.bfloat16) -> VAE_Decoder:
    
    print(f"🔄 Initializing decoder...")
    model = VAE_Decoder().to(device=device, dtype=dtype)
    load_weights(model=model, ckp_path=ckp_path, key_name="decoder", device=device)
    print("> Finished!")
    return model

def get_clip(ckp_path: str | None = None, 
             device: str = "cuda", 
             dtype: torch.dtype = torch.bfloat16) -> CLIP:
    
    print(f"🔄 Initializing clip...")
    model = CLIP().to(device=device, dtype=dtype)
    load_weights(model=model, ckp_path=ckp_path, key_name="clip", device=device)
    print("> Finished!")
    return model

def get_diffusion(ckp_path: str | None = None, 
                  key_name: str = "diffusion",
                  device: str = "cuda", 
                  dtype: torch.dtype = torch.bfloat16) -> DiffusionModel:
    
    print(f"🔄 Initializing diffusion...")
    model = DiffusionModel().to(device=device, dtype=dtype)
    load_weights(model=model, ckp_path=ckp_path, key_name=key_name, device=device)
    print("> Finished!")
    return model

def get_models(path: str = None,
               device: str = "cuda",
               dtype: torch.dtype = torch.bfloat16) -> dict[str, nn.Module | nn.Sequential]:
    """指定一个统一的模型权重，获取全部模型实例。"""

    print(f"🔄 Initializing vae, clip and diffusion...")
    models = {
        'encoder': VAE_Encoder().to(device=device, dtype=dtype),
        'decoder': VAE_Decoder().to(device=device, dtype=dtype),
        'clip': CLIP().to(device=device, dtype=dtype),
        'diffusion': DiffusionModel().to(device=device, dtype=dtype),
    }
    
    for name in models.keys():
        load_weights(model=models[name], ckp_path=path, key_name=name, device=device)
    print("✅ Initialization successful!")

    return models