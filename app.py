import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import random

# Predefined styles
styles = ["Realistic", "Anime", "Watercolor", "Cyberpunk", "Sketch"]

# Random prompts
random_prompts = [
    "A futuristic city skyline with flying cars",
    "A dreamy forest filled with glowing mushrooms",
    "An astronaut relaxing on Mars with a guitar",
    "A magical dragon flying over mountains",
    "Cyberpunk samurai walking in neon Tokyo"
]

# Load pipeline (Hugging Face will download model automatically)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_art(prompt, style, color, img_size):
    color_text = color.strip() if color.strip() else "any suitable color scheme"
    final_prompt = f"{prompt}, style: {style}, color scheme: {color_text}"
    image = pipe(final_prompt, height=img_size, width=img_size).images[0]
    return image

def surprise_me():
    return random.choice(random_prompts)

with gr.Blocks() as demo:
    gr.Markdown("## 🎨 AI Art Generator")

    with gr.Row():
        prompt_input = gr.Textbox(label="Enter your prompt", placeholder="Describe your scene here...")
        surprise_btn = gr.Button("🎲 Surprise Me")

    with gr.Row():
        style_dropdown = gr.Dropdown(styles, label="Choose Art Style", value="Realistic")
        color_input = gr.Textbox(label="Color Scheme", placeholder="Leave blank for AI to choose or type multiple colors")
        img_size = gr.Dropdown([256, 512, 768], label="Image Size (px)", value=512)

    generate_btn = gr.Button("✨ Generate Art")
    output = gr.Image(label="Result")

    generate_btn.click(generate_art, inputs=[prompt_input, style_dropdown, color_input, img_size], outputs=output)
    surprise_btn.click(surprise_me, outputs=prompt_input)

demo.launch()
