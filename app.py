import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load fine-tuned model
model_path = "/content/drive/MyDrive/fine-tuned-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit UI
st.title("Text-to-Image Generator")
prompt = st.text_input("Enter your prompt:", "A futuristic cityscape at sunset")

if st.button("Generate Image"):
    with st.spinner("Generating..."):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
