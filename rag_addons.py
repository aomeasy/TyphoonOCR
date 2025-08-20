# -*- coding: utf-8 -*-
"""
rag_addons.py
ส่วนขยาย RAG แบบไม่กระทบไฟล์เดิม:
- RAGKnowledgeBaseX: chunk Markdown รู้หัวข้อ + MMR retrieval + re-index/reset
- generate_rag_response_strict: ตอบเข้มงวด มีอ้างอิง [1],[2] ห้ามเดา
- extractive_answer: ดึงประโยคตรงจากเอกสาร (ไม่ใช้ LLM สรุป)
- UI helper สำหรับใช้งานในหน้า Chat/Maintenance ใหม่
"""
from __future__ import annotations
import streamlit as st
import requests
import json
import sqlite3
import hashlib
import pickle
import re
import math
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ===== Default endpoints (ปรับได้จากหน้า UI) =====
EMBEDDING_API_URL_DEFAULT = "http://209.15.123.47:11434/api/embeddings"
OLLAMA_API_URL_DEFAULT    = "http://209.15.123.47:11434/api/generate"
EMBEDDING_MODEL_DEFAULT   = "nomic-embed-text:latest"

# ==================== KB Class (Enhanced) ====================

class RAGKnowledgeBaseX:
    """
    เวอร์ชันเสริม: ใช้ตาราง documents/chat_sessions เดิมได้, chunk Markdown, MMR, reindex/reset
    ไม่กระทบคลาสเดิมในโปรเจกต์
    """
    def __init__(
        self,
        db_path: str = "typhoon_rag_knowledge.db",
        embedding_api_url: str = EMBEDDING_API_URL_DEFAULT,
        embedding_model: str = EMBEDDING_MODEL_DEFAULT
    ):
        self.db_path = db_path
        self.embedding_api_url = embedding_api_url
        self.embedding_model = embedding_model
        self._init_db()

    def set_embedding_model(self, model: str):
        if model:
            self.embedding_model = model

    def set_embedding_api(self, url: str):
        if url:
            self.embedding_api_url = url

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                chunk_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_hash TEXT,
                metadata TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question TEXT,
                answer TEXT,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit(); conn.close()

    # ---------- Embedding ----------
    def get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            resp = requests.post(
                self.embedding_api_url,
                json={"model": self.embedding_model, "prompt": text},
                timeout=60
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except Exception as e:
            st.error(f"❌ Embedding error: {e}")
            return None

    # ---------- Chunkers ----------
    def chunk_markdown(self, md_text: str, max_chars: int = 1200, overlap: int = 120) -> List[Dict]:
        """
        แบ่ง Markdown ตามหัวข้อ (#..######) ก่อน แล้วค่อยตัดแบบเลื่อนหน้าต่าง (รักษาหัวข้อ)
        return: [{"text": "...", "section": "หัวข้อ"}]
        """
        if not md_text:
            return []
        # แยกเป็นบล็อกตาม header
        blocks = re.split(r'(?m)^(?=\s{0,3}#{1,6}\s)', md_text)
        chunks: List[Dict] = []

        def sliding(text: str, section_path: str):
            text = text.strip()
            if not text:
                return
            if len(text) <= max_chars:
                chunks.append({"text": text, "section": section_path})
                return
            start = 0
            while start < len(text):
                end = min(len(text), start + max_chars)
                piece = text[start:end]
                chunks.append({"text": piece, "section": section_path})
                if end >= len(text):
                    break
                start = max(0, end - overlap)

        for b in blocks:
            lines = b.strip().splitlines()
            if not lines:
                continue
            if re.match(r'^\s{0,3}#{1,6}\s', lines[0]):  # header
                header_line = lines[0].strip()
                header_text = re.sub(r'^\s{0,3}#{1,6}\s', '', header_line).strip()
                body = "\n".join(lines[1:]).strip()
                section_path = header_text
                sliding(body if body else header_text, section_path)
            else:
                sliding(b, "root")

        return chunks

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                sentence_ends = ['. ', '! ', '? ', '\n\n', '。', '！', '？']
                best_break = end
                for i in range(min(100, len(text) - end)):
                    for ending in sentence_ends:
                        if text[end + i:end + i + len(ending)] == ending:
                            best_break = end + i + len(ending)
                            break
                    if best_break != end:
                        break
                end = best_break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    # ---------- Add Document ----------
    def add_document(self, filename: str, content: str, metadata: Dict = None,
                     chunk_size: int = 1200, overlap: int = 120) -> bool:
        """
        เพิ่มเอกสารแบบรักษาหัวข้อหากเป็น .md (ใช้ chunk_markdown)
        """
        try:
            file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            conn = sqlite3.connect(self.db_path); cursor = conn.cursor()
            cursor.execute("SELECT id FROM documents WHERE file_hash=?", (file_hash,))
            if cursor.fetchone():
                conn.close()
                st.warning(f"⚠️ Document {filename} already exists")
                return False

            if filename.lower().endswith(".md"):
                md_chunks = self.chunk_markdown(content, max_chars=chunk_size, overlap=overlap)
                pairs = [(c["text"], c["section"]) for c in md_chunks]
            else:
                pairs = [(x, "") for x in self.chunk_text(content, chunk_size, overlap)]

            progress = st.progress(0.0); msg = st.empty()
            for i, (txt, section) in enumerate(pairs):
                msg.text(f"Embedding chunk {i+1}/{len(pairs)}…")
                emb = self.get_embedding(txt)
                if emb:
                    cursor.execute("""
                        INSERT INTO documents (filename, content, embedding, chunk_id, file_hash, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        filename,
                        txt,
                        pickle.dumps(emb),
                        i,
                        file_hash,
                        json.dumps({**(metadata or {}), "section": section, "embedding_model": self.embedding_model}, ensure_ascii=False)
                    ))
                progress.progress((i+1)/max(1, len(pairs)))
            conn.commit(); conn.close()
            msg.text(f"✅ Added {len(pairs)} chunks from {filename}")
            return True
        except Exception as e:
            st.error(f"❌ Add document error: {e}")
            return False

    # ---------- Search (MMR) ----------
    def _cosine(self, a, b) -> float:
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
        return 0.0 if na == 0 or nb == 0 else dot/(na*nb)

    def _mmr_rank(self, query_vec, doc_vecs, lambda_mult=0.7, top_k=5):
        selected = []
        candidates = list(range(len(doc_vecs)))
        while candidates and len(selected) < top_k:
            best_idx, best_score = None, -1e9
            for i in candidates:
                rel = self._cosine(query_vec, doc_vecs[i])
                div = max((self._cosine(doc_vecs[i], doc_vecs[j]) for j in selected), default=0.0)
                score = lambda_mult*rel - (1 - lambda_mult)*div
                if score > best_score:
                    best_idx, best_score = i, score
            selected.append(best_idx); candidates.remove(best_idx)
        return selected

    def search_similar(self, query: str, top_k: int = 5, use_mmr: bool = True, lambda_mult: float = 0.7) -> List[Tuple[str, str, float]]:
        try:
            q = self.get_embedding(query)
            if not q:
                return []
            conn = sqlite3.connect(self.db_path); cursor = conn.cursor()
            cursor.execute("SELECT filename, content, embedding FROM documents")
            rows = cursor.fetchall(); conn.close()

            if not rows:
                return []

            vecs = [pickle.loads(r[2]) for r in rows]
            items = [(r[0], r[1]) for r in rows]

            if use_mmr:
                idxs = self._mmr_rank(q, vecs, lambda_mult=lambda_mult, top_k=top_k)
                scored = []
                for i in idxs:
                    sim = float(cosine_similarity([q], [vecs[i]])[0][0])
                    fn, txt = items[i]
                    scored.append((fn, txt, sim))
                scored.sort(key=lambda x: x[2], reverse=True)
                return scored
            else:
                sims = cosine_similarity([q], vecs)[0]
                order = np.argsort(-sims)[:top_k]
                return [(items[i][0], items[i][1], float(sims[i])) for i in order]
        except Exception as e:
            st.error(f"❌ Search error: {e}")
            return []

    # ---------- Stats / Maintenance ----------
    def get_stats(self) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path); cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM documents")
            total_docs = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_chunks = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM chat_sessions")
            total_chats = cursor.fetchone()[0]
            conn.close()
            return {"total_documents": total_docs, "total_chunks": total_chunks, "total_chat_sessions": total_chats}
        except Exception as e:
            return {"error": str(e)}

    def reset_kb(self):
        conn = sqlite3.connect(self.db_path); cursor = conn.cursor()
        cursor.execute("DELETE FROM documents")
        cursor.execute("DELETE FROM chat_sessions")
        conn.commit(); conn.close()

    def reindex_documents(self, filenames: Optional[List[str]] = None, chunk_size: int = 1200, overlap: int = 120):
        """
        รวมเนื้อหาเดิมตาม chunk_id แล้วฝังเวกเตอร์ใหม่ด้วย chunker ล่าสุด
        """
        conn = sqlite3.connect(self.db_path); cursor = conn.cursor()
        if not filenames:
            cursor.execute("SELECT DISTINCT filename FROM documents")
            filenames = [r[0] for r in cursor.fetchall()]

        for fn in filenames:
            cursor.execute("SELECT content FROM documents WHERE filename=? ORDER BY chunk_id ASC", (fn,))
            rows = cursor.fetchall()
            if not rows:
                continue
            full_text = "\n\n".join(r[0] for r in rows)
            cursor.execute("DELETE FROM documents WHERE filename=?", (fn,))
            conn.commit()
            # เพิ่มใหม่
            self.add_document(fn, full_text, metadata={"reindexed": True}, chunk_size=chunk_size, overlap=overlap)
        conn.close()

# ==================== Answering Functions ====================

def generate_rag_response_strict(
    query: str,
    context_docs: List[Tuple[str, str, float]],
    llm_model: str = "qwen2.5:14b",
    ollama_api_url: str = OLLAMA_API_URL_DEFAULT
) -> Optional[str]:
    """
    ตอบจากบริบทอย่างเข้มงวด + อ้างอิง [1],[2] / ไม่มีในเอกสารให้ปฏิเสธ
    """
    try:
        cites = []
        for i, d in enumerate(context_docs[:5], 1):
            fn, txt, sim = d
            cites.append(f"[{i}] {fn} (sim={sim:.3f})\n{txt}")
        context_text = "\n\n---\n\n".join(cites) if cites else "(no context)"

        prompt = f"""[SYSTEM]
คุณเป็นผู้ช่วยตอบคำถามจากคลังเอกสารองค์กร ตอบเฉพาะสิ่งที่อยู่ใน "บริบท" เท่านั้น
ห้ามคาดเดาหรือแต่งข้อมูล หากไม่พบ ให้ตอบว่า "ไม่พบข้อมูลในเอกสารที่ให้มา"
คำตอบควรกระชับ และเมื่ออ้างเนื้อหาให้แนบ [1],[2] ตรงข้อความที่อ้างถึง

[USER]
คำถาม: {query}

บริบท:
{context_text}

รูปแบบคำตอบ:
- สรุปคำตอบสั้นและชัดเจน
- เมื่ออ้างเนื้อหาให้ใส่ [index] เช่น [1]
- ถ้าไม่มีข้อมูลที่เกี่ยวข้อง ให้พิมพ์: ไม่พบข้อมูลในเอกสารที่ให้มา

คำตอบ:
"""
        resp = requests.post(
            ollama_api_url,
            json={"model": llm_model, "prompt": prompt, "temperature": 0.1, "top_p": 0.6, "num_predict": 800, "stream": False},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        st.error(f"❌ Strict answer error: {e}")
        return None


def extractive_answer(
    query: str,
    context_docs: List[Tuple[str, str, float]],
    kb: RAGKnowledgeBaseX,
    top_n: int = 3
) -> Optional[str]:
    """
    Extractive QA: ดึงประโยคที่คล้ายกับคำถามมากที่สุด โดยไม่ให้ LLM แต่ง/สรุป
    """
    try:
        qvec = kb.get_embedding(query)
        if not qvec:
            return None
        sent_re = re.compile(r'(?<=[\.\?!。！？])\s+|\n+')
        cands = []
        for fn, text, sim in context_docs[:5]:
            sentences = [s.strip() for s in sent_re.split(text) if s.strip()]
            for s in sentences:
                s_vec = kb.get_embedding(s[:1000])
                if not s_vec:
                    continue
                score = float(cosine_similarity([qvec], [s_vec])[0][0])
                cands.append((score, s, fn))
        cands.sort(key=lambda x: x[0], reverse=True)
        top = cands[:top_n]
        if not top:
            return "ไม่พบข้อมูลในเอกสารที่ให้มา"
        out = "📌 คำตอบจากข้อความในเอกสาร (Extractive):\n"
        for i, (sc, s, fn) in enumerate(top, 1):
            out += f"- {s}  \n  อ้างอิง: [{i}] {fn} (sim={sc:.3f})\n"
        return out
    except Exception as e:
        st.error(f"❌ Extractive answer failed: {e}")
        return None

# ==================== UI Helpers (ใช้ในหน้าที่เพิ่ม) ====================

def ui_kb_maintenance(db_path: str):
    """กล่องเครื่องมือ Re-index / Reset KB (ใช้ตารางเดิม)"""
    st.header("🧰 Knowledge Base Maintenance (Safe Add-on)")
    kb = RAGKnowledgeBaseX(db_path=db_path)

    stats = kb.get_stats()
    if "error" in stats:
        st.error(f"DB Error: {stats['error']}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("📚 Documents", stats["total_documents"])
        c2.metric("📄 Chunks", stats["total_chunks"])
        c3.metric("💬 Chat Sessions", stats["total_chat_sessions"])

    st.markdown("---")
    st.subheader("🔄 Re-index documents")
    # รายการเอกสาร
    conn = sqlite3.connect(db_path); cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT filename FROM documents ORDER BY filename")
    all_fns = [r[0] for r in cursor.fetchall()]
    conn.close()
    re_docs = st.multiselect("เลือกเอกสาร (ว่าง = ทั้งหมด)", options=all_fns, default=[])
    colx, coly = st.columns(2)
    with colx:
        re_chunk = st.number_input("Chunk size", 400, 4000, 1200, 100)
    with coly:
        re_overlap = st.number_input("Overlap", 0, 800, 120, 10)

    if st.button("🔁 Re-index Now"):
        with st.spinner("Re-indexing…"):
            kb.reindex_documents(filenames=re_docs if re_docs else None, chunk_size=int(re_chunk), overlap=int(re_overlap))
        st.success("✅ Re-index completed")
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("🗑️ Reset Knowledge Base (ล้างข้อมูลทั้งหมด)")
    confirm1 = st.checkbox("ฉันเข้าใจว่าจะลบข้อมูลทั้งหมด")
    confirm2 = st.text_input("พิมพ์คำว่า RESET เพื่อยืนยัน")
    if st.button("❗ Reset KB", type="primary", disabled=not (confirm1 and confirm2.strip().upper() == "RESET")):
        kb.reset_kb()
        st.success("✅ Knowledge base reset")
        st.experimental_rerun()


def ui_enhanced_ai_chat(
    db_path: str,
    ollama_api_url: str = OLLAMA_API_URL_DEFAULT,
    answer_models: Optional[List[str]] = None,
    session_key_chat: str = "enhanced_chat_history"
):
    """หน้า Chat เสริม: เลือก Embedding, Top‑K, MMR, Extractive/Strict, อ้างอิง"""
    st.header("💬 Enhanced AI Chat (Safe Add-on)")

    if session_key_chat not in st.session_state:
        st.session_state[session_key_chat] = []

    kb = RAGKnowledgeBaseX(db_path=db_path)

    # Controls
    st.subheader("⚙️ Retrieval & Answer Settings")
    c1, c2 = st.columns(2)
    with c1:
        embedding_api = st.text_input("Embedding API URL", EMBEDDING_API_URL_DEFAULT)
    with c2:
        embedding_model = st.selectbox("Embedding model", ["nomic-embed-text:latest"], index=0)

    kb.set_embedding_api(embedding_api)
    kb.set_embedding_model(embedding_model)

    r1, r2, r3 = st.columns(3)
    with r1:
        top_k = st.slider("Top‑K", 3, 10, 5)
    with r2:
        use_mmr = st.checkbox("Use MMR diversity", value=True)
    with r3:
        lambda_mult = st.slider("MMR λ (relevance↔diversity)", 0.5, 0.9, 0.7, 0.05)

    m1, m2, m3 = st.columns(3)
    with m1:
        extractive_mode = st.checkbox("Extractive answer (quotes)", value=False)
    with m2:
        strict_mode = st.checkbox("Strict LLM (no guessing + citations)", value=True)
    with m3:
        model_choice = st.selectbox(
            "LLM model for generation",
            answer_models or ["qwen2.5:14b", "scb10x/llama3.1-typhoon2-8b-instruct:latest", "scb10x/typhoon-ocr-7b:latest"],
            index=0
        )

    stats = kb.get_stats()
    if stats.get("total_documents", 0) == 0:
        st.warning("⚠️ ยังไม่มีเอกสารใน Knowledge Base — กรุณาเพิ่มก่อน")
        return

    st.markdown("---")
    st.subheader("🗣️ Ask a question from your documents")
    # show history
    for i, (q, a, ctx) in enumerate(st.session_state[session_key_chat]):
        st.markdown(f'<div style="padding:0.75rem;border-left:4px solid #2196f3;background:#e3f2fd;border-radius:8px;margin-bottom:4px;"><b>คุณ:</b><br>{q}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="padding:0.75rem;border-left:4px solid #9c27b0;background:#f3e5f5;border-radius:8px;margin-bottom:8px;"><b>ผู้ช่วย:</b><br>{a}</div>', unsafe_allow_html=True)
        with st.expander(f"📚 Sources (Chat {i+1})"):
            for j, (fn, content, sim) in enumerate(ctx):
                st.markdown(f"**{j+1}. {fn}** (sim={sim:.3f})")
                st.code((content[:400] + "…") if len(content) > 400 else content, language="markdown")

    with st.form("enhanced_chat_form", clear_on_submit=True):
        user_q = st.text_input("พิมพ์คำถาม", placeholder="ตัวอย่าง: สรุปนโยบายจากไฟล์คู่มือฯ พร้อมอ้างอิง")
        cc1, cc2 = st.columns([1,1])
        with cc1:
            ask_btn = st.form_submit_button("🚀 Ask")
        with cc2:
            clear_btn = st.form_submit_button("🗑️ Clear Chat")

    if clear_btn:
        st.session_state[session_key_chat] = []
        st.experimental_rerun()

    if ask_btn and user_q.strip():
        with st.spinner("🔍 Retrieving & answering…"):
            ctx_docs = kb.search_similar(user_q, top_k=top_k, use_mmr=use_mmr, lambda_mult=lambda_mult)

            if not ctx_docs:
                st.warning("⚠️ No relevant information found in knowledge base")
                return

            answer_text = None
            if extractive_mode:
                answer_text = extractive_answer(user_q, ctx_docs, kb, top_n=3)

            if (not answer_text or not answer_text.strip()) and strict_mode:
                answer_text = generate_rag_response_strict(user_q, ctx_docs, llm_model=model_choice, ollama_api_url=OLLAMA_API_URL_DEFAULT)

            # fallback (ไม่ปิดเผื่อผู้ใช้ปิด strict แตะ extractive)
            if not answer_text or not answer_text.strip():
                answer_text = "ไม่พบข้อมูลในเอกสารที่ให้มา"

            st.session_state[session_key_chat].append((user_q, answer_text, ctx_docs[:3]))

            # บันทึกลงตารางเดิม (session_id แยกเป็น 'enhanced')
            try:
                conn = sqlite3.connect(db_path); cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO chat_sessions (session_id, question, answer, context)
                    VALUES (?, ?, ?, ?)
                ''', ("enhanced", user_q, answer_text,
                      json.dumps([(d[0], d[1][:200], d[2]) for d in ctx_docs[:3]], ensure_ascii=False)))
                conn.commit(); conn.close()
            except Exception as e:
                st.warning(f"บันทึกแชทไม่สำเร็จ: {e}")

            st.experimental_rerun()
