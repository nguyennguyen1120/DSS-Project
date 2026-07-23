"""
Dashboard Streamlit — DSS Sinh Câu Hỏi Trắc Nghiệm Lịch Sử

Tính năng:
  1. Sinh câu hỏi: nhập context, chọn Bloom/Wh-type, so sánh 3 phương pháp
  2. Ngân hàng câu hỏi: xem, lọc, xem verifier_score
  3. AHP/TOPSIS: xem kết quả + kéo trọng số realtime
  4. Ráp đề: chọn số câu, Bloom ratio, so sánh ILP vs Greedy vs Random

Cài:
    pip install streamlit pandas plotly pulp

Chạy:
    streamlit run src/app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Setup paths ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src" / "schema"))
sys.path.insert(0, str(_ROOT / "src" / "preprocess"))
sys.path.insert(0, str(_ROOT / "src" / "dss"))

st.set_page_config(
    page_title="DSS — MCQ Generator",
    page_icon="📚",
    layout="wide",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("📚 DSS MCQ Generator")
st.sidebar.markdown("**Hệ hỗ trợ ra quyết định**  \nSinh câu hỏi trắc nghiệm lịch sử")
page = st.sidebar.radio(
    "Chọn trang:",
    ["🏠 Tổng quan", "✍️ Demo Sinh Câu Hỏi",
     "🔍 Ngân hàng câu hỏi",
     "📊 AHP/TOPSIS", "📝 Ráp đề thi"],
)

DATA_DIR = _ROOT / "data"
VERIFIED_DIR = DATA_DIR / "verified"
DSS_DIR      = DATA_DIR / "dss"
EXAM_DIR     = DATA_DIR / "exams"


# ── Helper: load ngân hàng ───────────────────────────────────────────────────
@st.cache_data
def load_bank():
    rows = []
    for f in VERIFIED_DIR.glob("verified_*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            v = item.get("verification") or {}
            rows.append({
                "item_id":        item["item_id"],
                "method":         item["generator"]["method"],
                "question":       item["question"],
                "answer_text":    item["answer_text"],
                "answer_key":     item["answer_key"],
                "options":        item["options"],
                "context_id":     item["source"]["context_id"],
                "title":          item["source"].get("title", ""),
                "is_vietnam":     item["source"].get("is_vietnam", False),
                "bloom":          item["request"]["bloom_requested"],
                "wh_type":        item["request"]["wh_type_requested"],
                "verifier_score": v.get("verifier_score", 0),
                "status":         v.get("status", "unknown"),
                "violations":     v.get("violations", []),
                "evidence_match": v.get("checks", {}).get(
                    "evidence_match", {}).get("score", 0),
                "bloom_fidelity": v.get("checks", {}).get(
                    "bloom_fidelity", {}).get("score", 0),
                "duplicate_check": v.get("checks", {}).get(
                    "duplicate_check", {}).get("score", 0),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data
def load_results():
    p = VERIFIED_DIR / "results_table.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# TRANG 1: TỔNG QUAN
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Tổng quan":
    st.title("🎓 DSS — Sinh Câu Hỏi Trắc Nghiệm Lịch Sử")
    st.markdown("""
    **Đồ án môn Hệ Hỗ Trợ Ra Quyết Định**
    
    So sánh 3 phương pháp sinh câu hỏi trắc nghiệm:
    - 🔧 **Rule-Based**: Template + Knowledge Base thực thể
    - 🤖 **RAG + LLM**: Hybrid retrieval + GPT-4o-mini
    - 🧠 **ViT5 Fine-tuning**: Seq2Seq answer-aware QG
    """)

    df = load_bank()
    results = load_results()

    if df.empty:
        st.warning("Chưa có dữ liệu. Chạy Verifier trước.")
    else:
        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        total   = len(df)
        accepted = (df["status"] == "accepted").sum()
        methods  = df["method"].nunique()
        avg_score = df["verifier_score"].mean()

        col1.metric("Tổng câu hỏi", f"{total:,}")
        col2.metric("Câu accepted", f"{accepted:,}",
                    f"{accepted/total*100:.1f}%")
        col3.metric("Phương pháp", methods)
        col4.metric("Avg score", f"{avg_score:.3f}")

        st.divider()

        # Bảng so sánh
        if not results.empty:
            st.subheader("📊 Kết quả so sánh 3 phương pháp")
            crit_cols = ["vqr", "evidence_match", "single_correct",
                         "distractor_type_match", "historical_correctness",
                         "question_clarity", "bloom_fidelity", "duplicate_check"]
            disp = results[["method"] + crit_cols].copy()
            disp[crit_cols] = disp[crit_cols].applymap(lambda x: f"{x:.3f}")
            st.dataframe(disp, use_container_width=True)

            # Bar chart VQR
            import plotly.graph_objects as go
            fig = go.Figure([go.Bar(
                x=results["method"], y=results["vqr"],
                text=[f"{v:.1%}" for v in results["vqr"]],
                textposition="outside",
                marker_color=["#2196F3", "#4CAF50", "#FF9800"],
            )])
            fig.update_layout(title="Valid Question Rate (VQR) theo phương pháp",
                              yaxis_tickformat=".0%", height=350)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRANG 2: DEMO SINH CÂU HỎI
# ─────────────────────────────────────────────────────────────────────────────
elif page == "✍️ Demo Sinh Câu Hỏi":
    st.title("✍️ Demo — Sinh câu hỏi từ ngữ cảnh")
    st.markdown("Nhập đoạn văn lịch sử và chọn loại câu hỏi — hệ thống sinh từ **3 phương pháp** để so sánh.")

    # ── Input ──────────────────────────────────────────────────────────────
    st.subheader("1. Nhập ngữ cảnh")
    default_ctx = (
        "Chiến thắng Điện Biên Phủ diễn ra ngày 7 tháng 5 năm 1954, "
        "kết thúc chín năm kháng chiến chống thực dân Pháp. "
        "Chiến dịch do Đại tướng Võ Nguyên Giáp chỉ huy tại lòng chảo "
        "Điện Biên Phủ, tỉnh Lai Châu. Đây là thất bại quân sự lớn nhất "
        "của Pháp tại Đông Dương, buộc Pháp ký Hiệp định Genève năm 1954, "
        "công nhận độc lập của Việt Nam, Lào và Campuchia."
    )
    context = st.text_area(
        "Đoạn văn lịch sử:",
        value=default_ctx,
        height=150,
        help="Nhập đoạn văn chứa thông tin lịch sử. Hệ thống sẽ sinh câu hỏi dựa trên đây."
    )

    col1, col2 = st.columns(2)
    wh_type = col1.selectbox(
        "Loại câu hỏi (Wh-type):",
        ["thoi_gian", "nhan_vat", "dia_diem", "su_kien", "nguyen_nhan", "y_nghia"],
        format_func=lambda x: {
            "thoi_gian":   "⏰ Thời gian (Năm nào? Khi nào?)",
            "nhan_vat":    "👤 Nhân vật (Ai? Người nào?)",
            "dia_diem":    "📍 Địa điểm (Ở đâu? Tại đâu?)",
            "su_kien":     "📅 Sự kiện (Điều gì xảy ra?)",
            "nguyen_nhan": "🔍 Nguyên nhân (Vì sao? Do đâu?)",
            "y_nghia":     "💡 Ý nghĩa (Kết quả? Tầm quan trọng?)",
        }[x]
    )
    bloom = col2.selectbox(
        "Mức Bloom:",
        ["nhan_biet", "thong_hieu", "van_dung", "van_dung_cao"],
        format_func=lambda x: {
            "nhan_biet":   "1️⃣ Nhận biết",
            "thong_hieu":  "2️⃣ Thông hiểu",
            "van_dung":    "3️⃣ Vận dụng",
            "van_dung_cao":"4️⃣ Vận dụng cao",
        }[x]
    )

    st.divider()

    # ── Chọn phương pháp ──────────────────────────────────────────────────
    st.subheader("2. Chọn phương pháp")
    col1, col2, col3 = st.columns(3)
    run_rb  = col1.checkbox("🔧 Rule-Based", value=True)
    run_rag = col2.checkbox("🤖 RAG + LLM", value=True,
                             help="Cần OPENAI_API_KEY")
    run_vit = col3.checkbox("🧠 ViT5 Fine-tuning", value=True,
                             help="Cần models/vit5_qg và vit5_dg")

    openai_key = ""
    if run_rag:
        openai_key = st.text_input(
            "OpenAI API Key:", type="password",
            value=st.session_state.get("openai_key", ""),
            help="Chỉ cần cho RAG+LLM"
        )
        if openai_key:
            st.session_state["openai_key"] = openai_key

    if st.button("🚀 Sinh câu hỏi", type="primary", disabled=not context.strip()):

        results_out = {}

        # ── Rule-Based ──────────────────────────────────────────────────────
        if run_rb:
            with st.spinner("Rule-Based đang sinh..."):
                import pickle, time
                pool_path = _ROOT / "data/processed/distractor_pool.pkl"
                ent_path  = _ROOT / "data/processed/entities.parquet"
                if pool_path.exists() and ent_path.exists():
                    try:
                        sys.path.insert(0, str(_ROOT/"src"/"generators"))
                        from rule_based import generate_one
                        from mcq_schema import WhType, Bloom as BloomEnum
                        import pandas as _pd
                        pool = pickle.load(open(pool_path,"rb"))
                        ent_df = _pd.read_parquet(ent_path)
                        # tạo context_id tạm
                        import hashlib
                        cid = "ctx_demo_" + hashlib.md5(context.encode()).hexdigest()[:8]
                        ents = ent_df[ent_df["context_id"]==cid].to_dict("records")
                        # nếu không có entity từ KB, extract tạm
                        if not ents:
                            import re
                            YEAR_RE = re.compile(r"\b(9[0-9]{2}|1[0-9]{3}|20[0-2][0-9])\b")
                            for m in YEAR_RE.finditer(context):
                                ents.append({"surface":m.group(),"type":"YEAR",
                                             "normalized":m.group(),"bucket":"20C",
                                             "char_start":m.start(),"char_end":m.end()})
                        t0 = time.time()
                        item = generate_one(
                            context=context, title="Demo", context_id=cid,
                            is_vietnam=True,
                            wh_type=WhType(wh_type),
                            bloom=BloomEnum(bloom),
                            entities=ents, pool=pool, seed=42,
                        )
                        elapsed = (time.time()-t0)*1000
                        if item:
                            results_out["rule_based"] = {
                                "item": item, "latency": elapsed, "cost": 0.0}
                        else:
                            results_out["rule_based"] = {"error": "Không tìm được entity phù hợp"}
                    except Exception as e:
                        results_out["rule_based"] = {"error": str(e)}
                else:
                    results_out["rule_based"] = {"error": "Thiếu distractor_pool.pkl hoặc entities.parquet"}

        # ── RAG + LLM ───────────────────────────────────────────────────────
        if run_rag:
            if not openai_key:
                results_out["rag_llm"] = {"error": "Chưa nhập OpenAI API Key"}
            else:
                with st.spinner("RAG+LLM đang sinh..."):
                    try:
                        import os, time
                        os.environ["OPENAI_API_KEY"] = openai_key
                        from openai import OpenAI
                        from rag_llm import build_prompt, parse_llm_output, call_openai
                        from mcq_schema import finalize_options, Provenance
                        client = OpenAI(api_key=openai_key)
                        t0 = time.time()
                        prompt = build_prompt(
                            context, __import__('mcq_schema').WhType(wh_type),
                            __import__('mcq_schema').Bloom(bloom), "Demo")
                        parsed, cost, calls = call_openai(prompt)
                        elapsed = (time.time()-t0)*1000
                        if parsed:
                            r = parse_llm_output(
                                parsed, context,
                                __import__('mcq_schema').WhType(wh_type),
                                __import__('mcq_schema').Bloom(bloom))
                            if r:
                                q, ans, dists, ev, found = r
                                opts, key = finalize_options(
                                    ans, dists,
                                    correct_provenance=Provenance.context_span,
                                    distractor_provenance=Provenance.generated,
                                    seed=0)
                                results_out["rag_llm"] = {
                                    "question": q, "options": opts,
                                    "answer_key": key, "answer_text": ans,
                                    "evidence": ev, "found": found,
                                    "latency": elapsed, "cost": cost,
                                }
                            else:
                                results_out["rag_llm"] = {"error": "Parse LLM output thất bại"}
                        else:
                            results_out["rag_llm"] = {"error": "LLM không trả về kết quả"}
                    except Exception as e:
                        results_out["rag_llm"] = {"error": str(e)[:100]}

        # ── ViT5 ────────────────────────────────────────────────────────────
        if run_vit:
            qg_path = _ROOT / "models/vit5_qg"
            dg_path = _ROOT / "models/vit5_dg"
            if not qg_path.exists() or not dg_path.exists():
                results_out["vit5_ft"] = {"error": "Chưa có models/vit5_qg hoặc vit5_dg"}
            else:
                with st.spinner("ViT5 đang sinh..."):
                    try:
                        import time
                        from infer_vit5 import ViT5Generator, find_evidence
                        from mcq_schema import finalize_options, Provenance
                        if "vit5_gen" not in st.session_state:
                            st.session_state["vit5_gen"] = ViT5Generator(
                                str(qg_path), str(dg_path))
                        gen = st.session_state["vit5_gen"]
                        t0 = time.time()
                        q   = gen.generate_question(context, "", wh_type)
                        if not q.endswith("?"): q += "?"
                        # lấy answer từ context (dùng wh_type để đoán)
                        import re
                        if wh_type == "thoi_gian":
                            m = re.search(r"\b(9[0-9]{2}|1[0-9]{3}|20[0-2][0-9])\b", context)
                            ans = m.group() if m else ""
                        else:
                            ans = context.split(".")[0][:50]
                        dists = gen.generate_distractors(q, ans, context)
                        elapsed = (time.time()-t0)*1000
                        if ans and len(dists) >= 3:
                            opts, key = finalize_options(
                                ans, dists[:3],
                                correct_provenance=Provenance.context_span,
                                distractor_provenance=Provenance.generated,
                                seed=0)
                            ev, found = find_evidence(context, ans)
                            results_out["vit5_ft"] = {
                                "question": q, "options": opts,
                                "answer_key": key, "answer_text": ans,
                                "evidence": ev, "found": found,
                                "latency": elapsed, "cost": 0.0,
                            }
                        else:
                            results_out["vit5_ft"] = {
                                "error": f"Sinh thiếu distractor ({len(dists)}/3)"}
                    except Exception as e:
                        results_out["vit5_ft"] = {"error": str(e)[:100]}

        # ── Hiển thị kết quả ────────────────────────────────────────────────
        st.subheader("3. Kết quả so sánh")
        METHOD_LABELS = {
            "rule_based": "🔧 Rule-Based",
            "rag_llm":    "🤖 RAG + LLM",
            "vit5_ft":    "🧠 ViT5",
        }
        cols = st.columns(len(results_out))
        for col, (method, res) in zip(cols, results_out.items()):
            with col:
                st.markdown(f"### {METHOD_LABELS.get(method, method)}")
                if "error" in res:
                    st.error(res["error"])
                    continue

                # lấy question/options từ MCQItem hoặc dict
                if "item" in res:
                    item = res["item"]
                    q    = item.question
                    opts = item.options
                    key  = item.answer_key
                    ev   = item.evidence.sentence
                    found = item.evidence.found_in_context
                else:
                    q    = res["question"]
                    opts = res["options"]
                    key  = res["answer_key"]
                    ev   = res.get("evidence","")
                    found = res.get("found", False)

                st.markdown(f"**{q}**")
                for opt in opts:
                    is_c  = opt["is_correct"] if isinstance(opt, dict) else opt.is_correct
                    text  = opt["text"]  if isinstance(opt, dict) else opt.text
                    label = opt["label"] if isinstance(opt, dict) else opt.label
                    icon  = "✅" if is_c else "○"
                    st.markdown(f"{icon} **{label}.** {text}")

                st.caption(f"⏱ {res['latency']:.0f}ms | 💰 ${res['cost']:.5f}")
                ev_icon = "✅" if found else "⚠️"
                st.caption(f"{ev_icon} Evidence: {ev[:60]}...")



    st.title("🔍 Ngân hàng câu hỏi")

    df = load_bank()
    if df.empty:
        st.warning("Chưa có dữ liệu verified.")
        st.stop()

    # Bộ lọc
    col1, col2, col3, col4 = st.columns(4)
    method_filter = col1.multiselect(
        "Phương pháp", df["method"].unique().tolist(),
        default=df["method"].unique().tolist())
    status_filter = col2.multiselect(
        "Status", ["accepted","needs_review","rejected"],
        default=["accepted"])
    bloom_filter = col3.multiselect(
        "Bloom", df["bloom"].unique().tolist(),
        default=df["bloom"].unique().tolist())
    min_score = col4.slider("Score tối thiểu", 0.0, 1.0, 0.5, 0.05)

    mask = (
        df["method"].isin(method_filter) &
        df["status"].isin(status_filter) &
        df["bloom"].isin(bloom_filter) &
        (df["verifier_score"] >= min_score)
    )
    filtered = df[mask].reset_index(drop=True)
    st.info(f"Hiển thị {len(filtered):,} / {len(df):,} câu hỏi")

    # Bảng
    display_cols = ["method","status","bloom","wh_type",
                    "verifier_score","title","question"]
    st.dataframe(
        filtered[display_cols].style.background_gradient(
            subset=["verifier_score"], cmap="RdYlGn"),
        use_container_width=True, height=400,
    )

    # Xem chi tiết một câu
    st.subheader("Chi tiết câu hỏi")
    if len(filtered):
        idx = st.number_input("Chọn hàng (0-based)", 0,
                              len(filtered)-1, 0, step=1)
        row = filtered.iloc[int(idx)]
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Q:** {row['question']}")
            for opt in row["options"]:
                icon = "✅" if opt["is_correct"] else "○"
                st.markdown(f"{icon} **{opt['label']}.** {opt['text']}")
        with col2:
            st.metric("Verifier score", f"{row['verifier_score']:.3f}")
            st.markdown(f"**Phương pháp:** {row['method']}")
            st.markdown(f"**Bloom:** {row['bloom']}")
            st.markdown(f"**Chủ đề:** {row['title']}")
            if row["violations"]:
                st.warning(f"Vi phạm: {', '.join(row['violations'])}")
            else:
                st.success("Không có vi phạm")


# ─────────────────────────────────────────────────────────────────────────────
# TRANG 3: AHP/TOPSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 AHP/TOPSIS":
    st.title("📊 AHP/TOPSIS — Quyết định chọn phương pháp")

    results = load_results()
    if results.empty:
        st.warning("Chưa có results_table.csv. Chạy Verifier trước.")
        st.stop()

    CRITERIA = ["vqr","evidence_match","single_correct","distractor_type_match",
                "historical_correctness","question_clarity",
                "bloom_fidelity","duplicate_check"]
    LABELS = {
        "vqr": "VQR", "evidence_match": "Evidence Match",
        "single_correct": "Single Correct", "distractor_type_match": "Distractor Type",
        "historical_correctness": "Historical", "question_clarity": "Clarity",
        "bloom_fidelity": "Bloom Fidelity", "duplicate_check": "Duplicate",
    }

    st.subheader("Kéo trọng số để xem thứ hạng TOPSIS thay đổi realtime")
    st.markdown("*(Trọng số sẽ được chuẩn hoá tổng = 1)*")

    # Sliders trọng số
    weights_raw = {}
    cols = st.columns(4)
    defaults = {"vqr":0.25,"evidence_match":0.18,"single_correct":0.13,
                "distractor_type_match":0.10,"historical_correctness":0.13,
                "question_clarity":0.09,"bloom_fidelity":0.07,"duplicate_check":0.05}
    for i, c in enumerate(CRITERIA):
        weights_raw[c] = cols[i % 4].slider(
            LABELS[c], 0.01, 0.50,
            float(defaults.get(c, 0.1)), 0.01, key=f"w_{c}")

    total_w = sum(weights_raw.values())
    weights = {k: v/total_w for k, v in weights_raw.items()}

    # Tính TOPSIS realtime
    import numpy as np
    dm     = results[CRITERIA].values.astype(float)
    w_arr  = np.array([weights[c] for c in CRITERIA])
    norms  = np.sqrt((dm**2).sum(axis=0)); norms[norms==0]=1e-10
    v      = (dm / norms) * w_arr
    pis    = v.max(axis=0)
    nis    = v.min(axis=0)
    d_pis  = np.sqrt(((v-pis)**2).sum(axis=1))
    d_nis  = np.sqrt(((v-nis)**2).sum(axis=1))
    scores = d_nis / (d_pis + d_nis + 1e-10)

    topsis_df = pd.DataFrame({
        "Phương pháp": results["method"],
        "TOPSIS Score": scores.round(4),
        "Hạng": len(scores) - scores.argsort().argsort(),
    }).sort_values("Hạng")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Xếp hạng")
        st.dataframe(topsis_df, use_container_width=True)
        best = topsis_df.iloc[0]["Phương pháp"]
        st.success(f"🏆 Tốt nhất: **{best.upper()}**")

    with col2:
        import plotly.express as px
        fig = px.bar(topsis_df.sort_values("TOPSIS Score"),
                     x="TOPSIS Score", y="Phương pháp",
                     orientation="h", color="TOPSIS Score",
                     color_continuous_scale="RdYlGn",
                     title="TOPSIS Score theo trọng số hiện tại")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart tiêu chí
    st.subheader("So sánh tiêu chí (Radar Chart)")
    import plotly.graph_objects as go
    fig2 = go.Figure()
    colors = {"rule_based": "blue", "rag_llm": "green", "vit5_ft": "orange"}
    for _, row in results.iterrows():
        vals = [row[c] for c in CRITERIA] + [row[CRITERIA[0]]]
        lbls = [LABELS[c] for c in CRITERIA] + [LABELS[CRITERIA[0]]]
        fig2.add_trace(go.Scatterpolar(
            r=vals, theta=lbls,
            fill="toself", name=row["method"],
            line_color=colors.get(row["method"], "gray"),
        ))
    fig2.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
                       height=450)
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRANG 4: RÁP ĐỀ THI
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📝 Ráp đề thi":
    st.title("📝 Ráp đề thi — Tối ưu có ràng buộc")

    df = load_bank()
    if df.empty:
        st.warning("Chưa có dữ liệu verified.")
        st.stop()

    accepted = df[df["status"] == "accepted"].copy()
    st.info(f"Ngân hàng: **{len(accepted)} câu accepted** sẵn sàng để ráp đề")

    col1, col2, col3 = st.columns(3)
    k = col1.number_input("Số câu trong đề", 10, min(80, len(accepted)), 40)
    max_ctx = col2.number_input("Tối đa câu/ngữ cảnh", 1, 5, 2)
    slack   = col3.number_input("Slack Bloom (±)", 0, 3, 1)

    st.subheader("Ma trận đặc tả Bloom")
    c1, c2, c3, c4 = st.columns(4)
    b1 = c1.slider("Nhận biết (%)", 10, 70, 40)
    b2 = c2.slider("Thông hiểu (%)", 10, 60, 35)
    b3 = c3.slider("Vận dụng (%)", 5, 50, 20)
    b4 = c4.slider("Vận dụng cao (%)", 0, 30, 5)
    total_pct = b1 + b2 + b3 + b4
    if total_pct != 100:
        st.warning(f"Tổng tỉ lệ = {total_pct}% (cần = 100%). Sẽ tự chuẩn hoá.")

    bloom_ratio = f"{b1},{b2},{b3},{b4}"

    if st.button("🚀 Ráp đề ngay", type="primary"):
        with st.spinner("Đang tối ưu..."):
            sys.path.insert(0, str(_ROOT / "src" / "dss"))
            from exam_builder import (solve_ilp, solve_greedy, solve_random,
                                      parse_bloom_ratio, build_exam_df)

            bloom_target = parse_bloom_ratio(bloom_ratio, int(k))

            # ILP
            ilp_ids, ilp_score = solve_ilp(
                accepted, int(k), bloom_target, max_ctx, slack)
            g_ids, g_score = solve_greedy(
                accepted, int(k), bloom_target, max_ctx)
            r_ids, r_score = solve_random(accepted, int(k))

        # So sánh
        cmp = pd.DataFrame([
            {"Chiến lược": "ILP (Tối ưu)", "Score": round(ilp_score,3),
             "Số câu": len(ilp_ids)},
            {"Chiến lược": "Greedy",        "Score": round(g_score,3),
             "Số câu": len(g_ids)},
            {"Chiến lược": "Random",         "Score": round(r_score,3),
             "Số câu": len(r_ids)},
        ])
        st.subheader("So sánh chiến lược ráp đề")
        st.dataframe(cmp, use_container_width=True)

        # Hiển thị đề ILP
        best_ids = ilp_ids if ilp_ids else g_ids
        exam_df  = build_exam_df(accepted, best_ids)

        st.subheader(f"Đề thi {'ILP' if ilp_ids else 'Greedy'} ({len(best_ids)} câu)")

        # Bloom distribution
        import plotly.express as px
        bloom_cnt = exam_df["bloom"].value_counts().reset_index()
        bloom_cnt.columns = ["Bloom", "Số câu"]
        fig = px.pie(bloom_cnt, names="Bloom", values="Số câu",
                     title="Phân bố Bloom trong đề")
        st.plotly_chart(fig, use_container_width=True)

        # Danh sách câu hỏi
        for i, row in exam_df.iterrows():
            with st.expander(
                f"Câu {i}. [{row['bloom']}] {row['question'][:80]}...  "
                f"(score={row['verifier_score']:.3f})"
            ):
                st.markdown(f"**{row['question']}**")
                for opt in row["options"]:
                    icon = "✅" if opt["is_correct"] else "○"
                    st.markdown(f"{icon} {opt['label']}. {opt['text']}")
                st.caption(f"Phương pháp: {row['method']} | "
                           f"Chủ đề: {row['title']} | "
                           f"Bloom: {row['bloom']}")

        # Export
        EXAM_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = EXAM_DIR / "exam_dashboard.csv"
        exam_df.to_csv(csv_path, index=True, encoding="utf-8-sig")
        st.success(f"Đã lưu đề thi -> {csv_path}")

        # Download button
        st.download_button(
            "⬇️ Tải đề thi (CSV)",
            exam_df.to_csv(index=True, encoding="utf-8").encode("utf-8"),
            file_name="exam.csv", mime="text/csv",
        )
