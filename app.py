import streamlit as st
from ultralytics import YOLO
from PIL import Image
import json
from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
model = YOLO("yolov8n.pt")
with open("breeds_info.json", "r", encoding="utf-8") as f:
    breed_info = json.load(f)
st.title("🐂 Smart Breed Detection & Advisory")
st.write("Upload an image to detect breed and ask questions via chatbot.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    results = model.predict(img)

    for r in results:
        res_plotted = r.plot()
        st.image(res_plotted, caption="Prediction Result", use_column_width=True)
        if len(r.boxes) > 0:
            breed = model.names[int(r.boxes.cls[0])]
            st.subheader(f"🐃 Breed: {breed}")

            if breed in breed_info:
                st.write(breed_info[breed]["en"]) 
            else:
                st.write("Info not available.")
st.subheader("💬 Breed Information Chatbot")

user_q = st.text_input("Ask about treatment, importance, etc.")
if user_q:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI veterinary advisor. Provide accurate, simple advice for farmers about cattle and buffalo breeds."},
            {"role": "user", "content": user_q}
        ]
    )

    st.write("🤖 Chatbot:")
    st.write(response.choices[0].message.content)