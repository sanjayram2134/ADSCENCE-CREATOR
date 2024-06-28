import streamlit as st
from objreg import analyze_image
from promtgen import chat_with_template
from SD import generate_background
#st.title("ADVERTISEMENT POSTER GENERATOR")

caption = analyze_image(r'G:\Sanjayram R\postgen\outputimages\segmented_output_2.png')
print(f"Generated Caption: {caption}")

prompt = chat_with_template(caption)
print(prompt)

background  = generate_background(prompt)
background.save('genback.png')
