import sys;from pathlib import Path;sys.path.insert(0, str(Path(__file__).resolve()))

import torch
from transformers import CLIPTokenizer
import gradio as gr
import random
import os
import json

from module import model_loader, pipeline

torch.set_float32_matmul_precision('high')

MODEL_PATH = './model/'
TOKENIZER_PATH = './tokenizer/'

gr.close_all()

model_cache = {}

def clear_model_cache():
    for old_path, old_model in model_cache.items():
        del old_model
    model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_model(model_path, device, dtype):
    if (model_path, device) not in model_cache:
        clear_model_cache()
        model_cache[(model_path, device)] = model_loader.get_models(model_path, device=device, dtype=dtype)
        
    return model_cache[(model_path, device)]

def model_list():
    if not os.path.exists(MODEL_PATH):
        return []
    return [f for f in os.listdir(MODEL_PATH) 
            if os.path.isfile(os.path.join(MODEL_PATH, f)) and (f.endswith('.safetensors'))]

def avaliable_device():
    return "cuda" if torch.cuda.is_available() else 'cpu'

def avaliable_dtype():
    return "bfloat16" if torch.cuda.is_available() else 'float32'
    
def random_seed():
    return random.randint(0, int(1e8))

def generate(prompt, 
             uncond_prompt, 
             img_width,
             img_height,
             input_image,
             strength,
             use_random_seed,
             seed, 
             do_cfg,
             cfg_scale,
             sampler_name,
             n_inference_steps,
             num_training_steps,
             models,
             device,
             dtype,
             ddim_eta,
             decode_interval):

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype = dtype_map[dtype]
    seed = random_seed() if use_random_seed else seed

    config_info = {
        "prompt": prompt, 
        "uncond_prompt": uncond_prompt, 
        "img_width": img_width,
        "img_height": img_height,
        "strength": strength,
        "use_random_seed": use_random_seed,
        "seed": seed, 
        "do_cfg": do_cfg,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "n_inference_steps": n_inference_steps,
        "num_training_steps": num_training_steps,
        "models": models,
        "device": device,
        "dtype": str(dtype),
        "ddim_eta": ddim_eta,
        "decode_interval": decode_interval
    }
    config_info = json.dumps(config_info, indent=2, ensure_ascii=False)
    
    generator = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        img_width=img_width,
        img_height=img_height,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler_name,
        n_inference_steps=n_inference_steps,
        num_training_steps=num_training_steps,
        models=get_model(MODEL_PATH+models, device, dtype),
        seed=seed,
        device=device,
        dtype=dtype,
        tokenizer=CLIPTokenizer(f"{TOKENIZER_PATH}/vocab.json", f"{TOKENIZER_PATH}/merges.txt"),
        ddim_eta=ddim_eta,
        decode_interval=decode_interval
    )
    for output_image in generator:
        yield output_image[0], config_info

with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion 1.0/1.5 Image Gen UI")

    with gr.Row():
        models = gr.Dropdown(
            choices=model_list(),
            label="Model select",
            info="Automatically discover model files in the model folder under the project root."
        )
    
    with gr.Column():
        prompt = gr.Textbox(
            label="Positive prompts",
            value="airfish_(lefko_d), architecture, blue sky, blurry, boat, broken window, building, city, day, depth of field, drum (container), no humans, outdoors, ruins, scenery, science fiction, sign, signature, sky, water, watercraft",
            placeholder="type prompts...",
            lines=2
        )
        uncond_prompt = gr.Textbox(
            label="Negative prompts",
            placeholder="type prompts...",
            lines=2
        )
        
    with gr.Row():
        with gr.Column():
            img_width = gr.Slider(
                label="Width",
                minimum=128,
                maximum=1024,
                value=512,
                step=64,
                info="Width must be multiples of 64."
            )
            img_height = gr.Slider(
                label="Height",
                minimum=128,
                maximum=1024,
                value=512,
                step=64,
                info="Height must be multiples of 64."
            )
            
        with gr.Column():
            sampler_name = gr.Dropdown(
                label="Sampler",
                choices=["ddpm", "ddim"],
                value="ddpm"
            )
            ddim_eta = gr.Slider(
                label="DDIM eta",
                minimum=0.0,
                maximum=1.0,
                value=0.2,
                step=0.1
            )

    with gr.Row():
        with gr.Group():
            use_random_seed = gr.Checkbox(
                label="Random seed",
                value=True
            )
            seed = gr.Number(
                label="Seed", 
                value=random_seed, 
                precision=0,
                info="Uncheck the 'Random seed' option to use a fixed random seed."
            )
        with gr.Group():
            do_cfg = gr.Checkbox(
                label="Use CFG",
                value=True
            )
            cfg_scale = gr.Slider(
                label="CFG scale",
                minimum=0.0,
                maximum=20.0,
                value=7.5,
                step=0.1,
                info="Higher values increase prompt adherence, while lower values allow more creative variation."
            )
    with gr.Row():
        n_inference_steps = gr.Slider(
            label="Inference steps",
            minimum=10,
            maximum=100,
            value=64,
            step=1,
            info="Higher values result in fewer denoising steps."
        )
        decode_interval = gr.Slider(
            label="Decode interval",
            minimum=1,
            maximum=10,
            value=6,
            step=1,
            info="Number of times to decode and visualize the feature maps during denoising. Set to 1 to decode only the final step."
        )
        num_training_steps = gr.Slider(
            label="Training Steps",
            minimum=100,
            maximum=1000,
            value=1000,
            step=1,
            info="Total diffusion timesteps for noising/denoising during training. Default: 1000."
        )

    with gr.Row():
        device = gr.Dropdown(
                label="Device",
                choices=["cuda", "cpu"],
                value=avaliable_device,
                info="Select GPU or CPU device for inference."
            )
        dtype = gr.Dropdown(
                label="Data type",
                choices=["bfloat16", "float32"],
                value=avaliable_dtype,
                info="CPU does not support bfloat16 mixed precision or float16 acceleration."
            )

    
    with gr.Row():
        with gr.Column():
            config_output = gr.Textbox(
                label="Configurations",
                lines=10,
                max_lines=10,
                interactive=False
            )
            generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Group():
            input_image = gr.Image(
                label="Upload image (optional)",
                type="pil"
            )
            strength = gr.Slider(
                label="Image guidance strength",
                    minimum=0,
                    maximum=1,
                    value=0.2,
                    step=0.01,
                    info="Higher values add more noise and result in greater redrawing/regeneration."
            )

        output_image = gr.Image(label="Overview", type="pil")

    # Generate event
    generate_btn.click(
        fn=generate,
        inputs=[prompt, 
                uncond_prompt,
                img_width,
                img_height,
                input_image,
                strength,
                use_random_seed,
                seed, 
                do_cfg,
                cfg_scale,
                sampler_name,
                n_inference_steps,
                num_training_steps,
                models,
                device,
                dtype,
                ddim_eta,
                decode_interval],
        outputs=[output_image, config_output]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Ocean())