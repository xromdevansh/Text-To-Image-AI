import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import json
import os

# Set custom CSS styles for Streamlit app
def set_custom_style():
    st.markdown("""
        <style>
        /* Change background */
        .stApp {
            background-color: #f1f6f9;
            font-family: 'Poppins', sans-serif;
        }

        /* Header */
        h1 {
            color: #2b6777;
        }

        /* Textarea */
        .stTextArea textarea {
            background-color: #ffffff;
            color: #333333;
            font-size: 16px;
            border-radius: 8px;
        }

        /* Button */
        div.stButton > button {
            background-color: #52ab98;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }

        /* Success & warning messages */
        .stSuccess, .stWarning {
            font-size: 15px;
        }

        </style>
        <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
# Load Stable Diffusion pipeline safely
@st.cache_resource
def load_model():
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

    pipe = pipe.to("cpu", torch.float32)
    return pipe
set_custom_style()
pipe = load_model()

# Streamlit UI
st.title("AI Image Generator with Stable Diffusion")
st.markdown("Enter one prompt per line and generate images.")

prompt_input = st.text_area("✍️ Enter prompts below:", height=150)

if st.button("Generate Images"):
    prompts = [p.strip() for p in prompt_input.strip().split('\n') if p.strip()]
    if not prompts:
        st.warning("⚠️ Please enter at least one prompt.")
    else:
        with st.spinner("Generating images..."):
            results = pipe(
                prompts,
                num_inference_steps=50,
                guidance_scale=3.5,
                height=512,
                width=512
            )

            metadata = []

            for i, img in enumerate(results.images):
                file_path = f"image_{i}.png"
                img.save(file_path)

                # Store metadata
                metadata.append({
                    "prompt": prompts[i],
                    "filename": file_path
                })

                # Show in Streamlit
                st.image(img, caption=f"Prompt: {prompts[i]}", use_column_width=True)
                st.success(f"✅ Saved: `{file_path}`")

            # Save metadata to JSON
            with open("images.json", "w") as f:
                json.dump(metadata, f, indent=4)

            st.info("📁 All image info saved to `images.json`")
