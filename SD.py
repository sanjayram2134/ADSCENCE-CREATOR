import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login
import sentencepiece

print(torch.cuda.is_available())
print(torch.cuda.device_count())

login(token="##token")

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
#user_prompt = input("Enter the prompt for the image: ")
def generate_background(user_prompt):

    image = pipe(
        prompt=user_prompt,
        negative_prompt="black and white",
        num_inference_steps=28,
        height=1024,
        width=1024,
        guidance_scale=7.0,
    ).images[0]
    return image

background  = generate_background(prompt)
background.save('genback.png')

#image.save("GENERATED BACKGROUND.png")
#DISPLAY THE BACKGROUND
#import matplotlib.pyplot as plt
#plt.imshow(image)
#plt.axis('off')  # Turn off axis
#plt.show()