import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Quiz Generator", page_icon="🧠", layout="wide")

st.title("🧠 AI Quiz Generator")
st.write("สร้างคำถามแนววิชาการจากข้อความ โดยใช้โมเดลฟรี")

topic = st.text_area("📘 ใส่เนื้อหาที่ต้องการให้ AI สร้างคำถามจากข้อมูลนี้:")

num_q = st.slider("จำนวนคำถามที่ต้องการสร้าง", 1, 5, 3)

if st.button("สร้างคำถาม"):
    if not topic.strip():
        st.warning("กรุณาใส่เนื้อหาก่อนครับ")
    else:
        st.info("⏳ กำลังสร้างคำถามด้วย AI ฟรีหลายโมเดล...")

        generator1 = pipeline("text-generation", model="distilgpt2")
        generator2 = pipeline("text-generation", model="bigscience/bloom-560m")

        prompt = f"สร้างคำถามแนววิชาการจากเนื้อหานี้ จำนวน {num_q} ข้อ พร้อมเฉลย: {topic}"

        res1 = generator1(prompt, max_new_tokens=150)[0]["generated_text"]
        res2 = generator2(prompt, max_new_tokens=150)[0]["generated_text"]

        st.subheader("ผลลัพธ์จากโมเดล 1 (DistilGPT2)")
        st.write(res1)

        st.subheader("ผลลัพธ์จากโมเดล 2 (Bloom)")
        st.write(res2)

        overlap = len(set(res1.split()) & set(res2.split()))
        confidence = min(100, int(overlap / 5))
        st.success(f"🔍 คะแนนความสอดคล้องระหว่างโมเดล: {confidence}%")

        st.caption("✅ ใช้โมเดลโอเพ่นซอร์สฟรีจาก HuggingFace (ไม่ต้องใช้ API key)")
