# -*- coding: utf-8 -*-
# pages/21_KB_Maintenance.py
import streamlit as st
from rag_addons import ui_kb_maintenance

st.set_page_config(page_title="KB Maintenance", page_icon="🧰", layout="wide")

db_path = st.session_state.get("rag_db_path", "typhoon_rag_knowledge.db")

st.sidebar.success("Add-on")
st.sidebar.write("• Re-index (เปลี่ยน chunk/overlap)\n• Reset KB (ยืนยัน 2 ชั้น)")

ui_kb_maintenance(db_path=db_path)
