import torch
import torch.nn as nn

from safetensors.torch import load_file

from module.clip import CLIP
from module.vae import VAE_Encoder, VAE_Decoder
from module.diffusion import DiffusionModel

##### Prefixes of the four modules in the official Stable Diffusion checkpoint.
##### Since the module architectures follow the official naming, loading official weights
##### only requires splitting the checkpoint by these prefixes — no key conversion is needed.
##### Each entry is (matched prefix, stripped prefix).
MODULE_PREFIXES: dict[str, tuple[tuple[str, str], ...]] = {
    'diffusion': (('model.diffusion_model.', 'model.diffusion_model.'),),
    'encoder':   (('first_stage_model.encoder.', 'first_stage_model.encoder.'),
                  ('first_stage_model.quant_conv.', 'first_stage_model.')),
    'decoder':   (('first_stage_model.decoder.', 'first_stage_model.decoder.'),
                  ('first_stage_model.post_quant_conv.', 'first_stage_model.')),
    'clip':      (('cond_stage_model.transformer.text_model.', 'cond_stage_model.transformer.text_model.'),),
}

def get_model(name: str, 
              ckp_path: str | None = None, 
              device: str = "cuda", 
              dtype: torch.dtype = torch.bfloat16):
    """获取指定模型实例并加载权重。"""
    
    model_classes = {
        'encoder': VAE_Encoder,
        'decoder': VAE_Decoder,
        'clip': CLIP,
        'diffusion': DiffusionModel,
    }
    if name not in model_classes:
        raise ValueError(f"Unknown model name: {name}. Available: {list(model_classes.keys())}")

    # 实例化指定模型
    print(f"🔄 Initializing {name}...")
    model = model_classes[name]().to(device=device, dtype=dtype)

    # 加载检查点/官方权重
    if ckp_path:
        if ckp_path.endswith('.safetensors'):
            print(f"🔄 Loading {ckp_path} for {name}...")
            ckp = load_from_standard_weights(ckp_path, device=device)
        else:
            print(f"🔄 Loading {ckp_path} for {name}...")
            ckp = torch.load(ckp_path, map_location=device)
        model.load_state_dict(ckp[name], strict=True)
        del ckp
    else:
        print(f"⏭️ Skip loading weights for {name}...")
        
    print("✅ Finished!")
    return model

def get_models(path: str | dict | None = None,
               device: str = "cuda",
               dtype: torch.dtype = torch.bfloat16) -> dict[str, nn.Module | nn.Sequential]:
    """获取全部模型实例并加载权重。"""
    
    models = {
        'encoder': None,
        'decoder': None,
        'clip': None,
        'diffusion': None,
    }
    
    if isinstance(path, str) or isinstance(path, None):
        ##### 指定一个统一的模型权重
        for name in models.keys():
            models[name] = get_model(name=name, ckp_path=path, device=device, dtype=dtype)

    elif isinstance(path, dict):
        ##### Assign checkpoints to each of the four models separately
        for name in models.keys():
            ckp_path = path.get(name)
            if ckp_path is None:
                print(f"⏭️ Skipping {name}...")
                continue

            models[name] = get_model(name=name, ckp_path=ckp_path, device=device, dtype=dtype)

    return models

def load_from_standard_weights(input_file: str, device: str) -> dict[str, dict[str, torch.Tensor]]:
    """Split an official Stable Diffusion checkpoint into the four modules by key prefix.

    Returns a dict with keys 'encoder', 'decoder', 'clip' and 'diffusion', whose values are
    state dicts that can be loaded into the corresponding module with `load_state_dict` directly.
    """
    original_model = load_file(input_file, device=device)
    
    converted: dict[str, dict[str, torch.Tensor]] = {}
    for name, prefixes in MODULE_PREFIXES.items():
        converted[name] = {}
        for match_prefix, strip_prefix in prefixes:
            for key, value in original_model.items():
                if key.startswith(match_prefix):
                    # Strip the prefix so that the keys match the module's own state dict
                    converted[name][key[len(strip_prefix):]] = value

    # `position_ids` is a constant buffer (arange), not a learned weight — drop it so that
    # checkpoints with or without this buffer can both be loaded with `strict=True`
    converted['clip'].pop('embeddings.position_ids', None)

    return converted