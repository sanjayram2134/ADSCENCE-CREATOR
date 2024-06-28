import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def analyze_image(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    # Process image with padding enabled
    inputs = processor(raw_image, return_tensors="pt", padding=True)

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Filter and refine the caption
    words = caption.split()
    seen = set()
    filtered_caption = []
    for word in words:
        if word not in seen:
            filtered_caption.append(word)
            seen.add(word)
    refined_caption = ' '.join(filtered_caption)

    return refined_caption
"""
# Example usage
caption = analyze_image(r'G:\Sanjayram R\postgen\segmented_output_2.png')
print(f"Generated Caption: {caption}")

def generate_prompt(caption, additional_context=""):
    prompt = f"A background for {caption} {additional_context}"
    return prompt

# Example usage
prompt = generate_prompt(caption, "that highlights the product in an appealing way")
print(f"Generated Prompt: {prompt}")
"""
