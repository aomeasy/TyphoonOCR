# -*- coding: utf-8 -*-
# pages/20_Enhanced_AI_Chat.py
import streamlit as st
from rag_addons import ui_enhanced_ai_chat

st.set_page_config(page_title="Enhanced AI Chat", page_icon="💬", layout="wide")

# ใช้ DB path เดียวกับระบบเดิม (ถ้าไม่มีใน session จะ fallback)
db_path = st.session_state.get("rag_db_path", "typhoon_rag_knowledge.db")

st.sidebar.success("หน้านี้เป็น Add-on | ไม่กระทบโค้ดหลัก")
st.sidebar.write("• เลือก Embedding/Top‑K/MMR\n• Extractive/Strict\n• อ้างอิงแหล่งข้อมูลอัตโนมัติ")

ui_enhanced_ai_chat(
    db_path=db_path,
    answer_models=["qwen2.5:14b", "scb10x/llama3.1-typhoon2-8b-instruct:latest", "scb10x/typhoon-ocr-7b:latest"],
    session_key_chat="enhanced_chat_history"
)
