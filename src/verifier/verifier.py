"""
Verifier — Kiểm chứng câu hỏi trắc nghiệm theo 8 tiêu chí.

Chạy trên 3 file .jsonl đầu ra của 3 phương pháp sinh.

Tiêu chí:
  1. evidence_match        — đáp án có bằng chứng trong ngữ cảnh (NLI via OpenAI)
  2. single_correct        — chỉ 1 phương án đúng (NLI via OpenAI)
  3. distractor_type_match — distractor cùng loại với đáp án (NER + cosine)
  4. historical_correctness— không sai sự kiện/năm (đối chiếu KB)
  5. question_clarity      — câu hỏi rõ ràng (heuristic + length)
  6. duplicate_check       — không trùng câu hỏi khác (cosine embedding)
  7. bloom_fidelity        — đúng mức Bloom yêu cầu (zero-shot via OpenAI)
  8. answer_position_ok    — phân bố A/B/C/D cân bằng (thống kê batch)

Cài:
    pip install openai sentence-transformers datasketch

Chạy:
    python src/verifier/verifier.py \\
        --inputs data/generated/rule_based.jsonl \\
                 data/generated/rag_llm.jsonl \\
                 data/generated/vit5_ft.jsonl \\
        --entities data/processed/entities.parquet \\
        --outdir data/verified
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "schema"))
sys.path.insert(0, str(_SRC / "preprocess"))

from mcq_schema import (
    MCQItem, Verification, VerdictStatus, Check, Bloom, WhType,
)

# ─────────────────────────── Trọng số 8 tiêu chí ────────────────────────────

WEIGHTS = {
    "evidence_match":         0.20,
    "single_correct":         0.18,
    "distractor_type_match":  0.14,
    "historical_correctness": 0.12,
    "question_clarity":       0.10,
    "duplicate_check":        0.10,
    "bloom_fidelity":         0.10,
    "answer_position_ok":     0.06,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6

ACCEPT_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.55


# ─────────────────────────── 1. Evidence Match (NLI via OpenAI) ──────────────

NLI_CACHE: dict[str, str] = {}   # cache để không gọi API 2 lần cùng input


def _nli_openai(premise: str, hypothesis: str,
                client, model="gpt-4o-mini") -> str:
    """Trả 'entailment' | 'neutral' | 'contradiction'."""
    key = f"{premise[:80]}|||{hypothesis[:80]}"
    if key in NLI_CACHE:
        return NLI_CACHE[key]

    prompt = (
        f"Classify the relationship between the following premise and hypothesis.\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Output exactly one word: entailment, neutral, or contradiction."
    )
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )
            label = resp.choices[0].message.content.strip().lower()
            if label not in ("entailment", "neutral", "contradiction"):
                label = "neutral"
            NLI_CACHE[key] = label
            return label
        except Exception as e:
            if attempt == 0:
                time.sleep(1)
            else:
                return "neutral"
    return "neutral"


def check_evidence_match(item: MCQItem, client) -> Check:
    """
    Hypothesis: câu khẳng định ghép từ câu hỏi + đáp án đúng.
    Premise: context.
    Nếu entailment → passed.
    Nhãn vàng: câu is_impossible trong ViQuAD → phải neutral/contradiction.
    """
    ctx  = item.source.context
    q    = item.question
    ans  = item.answer_text

    # ghép thành câu khẳng định đơn giản
    hypothesis = f"{q.rstrip('?')} là {ans}."

    label = _nli_openai(ctx[:1000], hypothesis, client)
    passed = label == "entailment"
    score  = 1.0 if passed else (0.4 if label == "neutral" else 0.0)

    # bonus: evidence sentence có trong context
    if item.evidence.found_in_context and passed:
        score = min(1.0, score + 0.1)

    return Check(
        score=round(score, 3),
        passed=passed,
        reason=f"NLI={label} | evidence_in_ctx={item.evidence.found_in_context}",
    )


# ─────────────────────────── 2. Single Correct Answer ───────────────────────

def check_single_correct(item: MCQItem, client) -> Check:
    """
    Chạy NLI cho cả 4 phương án.
    Đúng khi: đúng 1 entailment (phương án is_correct),
              3 phương án còn lại là neutral/contradiction.
    """
    ctx = item.source.context[:1000]
    q   = item.question.rstrip("?")

    entail_labels = []
    for opt in item.options:
        hypothesis = f"{q} là {opt.text}."
        label = _nli_openai(ctx, hypothesis, client)
        entail_labels.append((opt.label, opt.is_correct, label))

    n_entail = sum(1 for _, _, l in entail_labels if l == "entailment")
    correct_is_entail = any(
        is_c and l == "entailment"
        for _, is_c, l in entail_labels
    )

    passed = (n_entail == 1 and correct_is_entail)
    if passed:
        score = 1.0
    elif correct_is_entail and n_entail <= 2:
        score = 0.5   # đáp án đúng OK nhưng có distractor cũng entail
    else:
        score = 0.0

    detail = " | ".join(f"{l}:{lbl}" for l, _, lbl in entail_labels)
    return Check(
        score=round(score, 3),
        passed=passed,
        reason=f"n_entail={n_entail} correct_entail={correct_is_entail} | {detail}",
    )


# ─────────────────────────── 3. Distractor Type Match ───────────────────────

def check_distractor_type(item: MCQItem, embedder=None) -> Check:
    """
    Kiểm tra distractor cùng loại thực thể với đáp án (entity_type field).
    Thêm: cosine similarity distractor–đáp án nằm trong [0.4, 0.9].
    """
    correct_opt = next((o for o in item.options if o.is_correct), None)
    if correct_opt is None:
        return Check(score=0.0, passed=False, reason="Không tìm thấy đáp án đúng")

    correct_type = correct_opt.entity_type
    distractors  = [o for o in item.options if not o.is_correct]

    if correct_type is None:
        # không có entity_type → bỏ qua kiểm tra type, chỉ kiểm tra cosine
        type_score = 0.7
        type_reason = "entity_type=None (skip type check)"
    else:
        type_matches = sum(
            1 for o in distractors
            if o.entity_type == correct_type or o.entity_type is None
        )
        type_score  = type_matches / 3
        type_reason = f"type_match={type_matches}/3 correct_type={correct_type}"

    # cosine similarity nếu có embedder
    cosine_score = 0.8   # default nếu không có embedder
    cosine_reason = "no_embedder"
    if embedder:
        try:
            texts = [correct_opt.text] + [o.text for o in distractors]
            embs  = embedder.encode(texts, normalize_embeddings=True)
            sims  = [float(embs[0] @ embs[i+1]) for i in range(3)]
            in_range = sum(1 for s in sims if 0.3 <= s <= 0.95)
            cosine_score  = in_range / 3
            cosine_reason = f"cosine={[round(s,2) for s in sims]}"
        except Exception:
            pass

    score  = 0.6 * type_score + 0.4 * cosine_score
    passed = score >= 0.6

    return Check(
        score=round(score, 3),
        passed=passed,
        reason=f"{type_reason} | {cosine_reason}",
    )


# ─────────────────────────── 4. Historical Correctness ──────────────────────

def check_historical_correctness(item: MCQItem, kb_facts: dict) -> Check:
    """
    Đối chiếu KB: kiểm tra đáp án đúng có tồn tại trong KB của context không.
    Gắn cờ 'unverified' nếu không tìm thấy (không kết luận SAI).
    """
    cid = item.source.context_id
    ans = item.answer_text.strip().lower()

    # lấy các surface trong KB của context này
    surfaces = {s.lower() for s in kb_facts.get(cid, set())}

    if not surfaces:
        return Check(
            score=0.7,
            passed=True,
            reason="unverified: không có KB cho context này",
        )

    if ans in surfaces:
        return Check(score=1.0, passed=True, reason=f"KB match: '{ans}'")

    # fuzzy: kiểm tra answer là substring của bất kỳ surface nào
    partial = any(ans in s or s in ans for s in surfaces)
    if partial:
        return Check(score=0.85, passed=True, reason=f"KB partial match: '{ans}'")

    return Check(
        score=0.5,
        passed=True,   # không kết luận sai, chỉ gắn cờ
        reason=f"unverified: '{ans}' không trong KB ({len(surfaces)} entries)",
    )


# ─────────────────────────── 5. Question Clarity ────────────────────────────

import re as _re

_AMBIGUOUS_PRONOUNS = ["nó", "điều này", "điều đó", "đây", "đó", "họ", "chúng"]
_DOUBLE_NEG = [("không", "không"), ("chưa", "không"), ("không", "chưa")]


def check_question_clarity(item: MCQItem) -> Check:
    q     = item.question.strip()
    score = 1.0
    reasons = []

    # độ dài: 6–40 token
    n_tok = len(q.split())
    if n_tok < 4:
        score -= 0.7   # quá ngắn → chắc chắn fail
        reasons.append(f"quá ngắn ({n_tok} token)")
    elif n_tok < 6:
        score -= 0.4
        reasons.append(f"ngắn ({n_tok} token)")
    elif n_tok > 50:
        score -= 0.2
        reasons.append(f"quá dài ({n_tok} token)")

    # phải có đúng 1 dấu hỏi
    n_q = q.count("?")
    if n_q == 0:
        score -= 0.3; reasons.append("thiếu dấu hỏi")
    elif n_q > 1:
        score -= 0.2; reasons.append("nhiều hơn 1 dấu hỏi")

    # đại từ mơ hồ không có tiền ngữ
    ql = q.lower()
    for p in _AMBIGUOUS_PRONOUNS:
        if ql.startswith(p) or f" {p} " in ql:
            score -= 0.15
            reasons.append(f"đại từ mơ hồ: '{p}'")
            break

    # phủ định kép
    words = ql.split()
    for w1, w2 in _DOUBLE_NEG:
        if w1 in words and w2 in words:
            idx1 = words.index(w1)
            idx2 = words.index(w2) if w2 in words else -1
            if idx2 > idx1 and idx2 - idx1 <= 5:
                score -= 0.2
                reasons.append("phủ định kép")
                break

    score  = max(0.0, min(1.0, score))
    passed = score >= 0.6
    return Check(
        score=round(score, 3),
        passed=passed,
        reason=", ".join(reasons) if reasons else "OK",
    )


# ─────────────────────────── 6. Duplicate Check ─────────────────────────────

def check_duplicate(item: MCQItem, seen_embeddings: list,
                    seen_ids: list, embedder=None,
                    threshold: float = 0.92) -> Check:
    """
    So sánh cosine embedding câu hỏi với mọi câu đã xử lý.
    Nếu không có embedder, dùng Jaccard trên trigram.
    """
    q = item.question.strip().lower()

    def trigram_jaccard(a: str, b: str) -> float:
        def tg(s): return set(s[i:i+3] for i in range(len(s)-2))
        ta, tb = tg(a), tg(b)
        if not ta or not tb: return 0.0
        return len(ta & tb) / len(ta | tb)

    if embedder:
        try:
            import numpy as np
            emb = embedder.encode([item.question], normalize_embeddings=True)[0]
            max_sim = 0.0
            max_idx = -1
            for i, prev_emb in enumerate(seen_embeddings):
                sim = float(np.dot(emb, prev_emb))
                if sim > max_sim:
                    max_sim = sim; max_idx = i
            seen_embeddings.append(emb)
            seen_ids.append(item.item_id)
            passed = max_sim < threshold
            score  = 1.0 - max_sim if passed else 0.0
            reason = (f"max_cosine={max_sim:.3f} vs threshold={threshold}"
                      + (f" (dup of {seen_ids[max_idx]})" if not passed else ""))
            return Check(score=round(score,3), passed=passed, reason=reason)
        except Exception:
            pass

    # fallback Jaccard
    max_j = 0.0
    for prev_q in seen_ids:
        j = trigram_jaccard(q, prev_q.lower())
        max_j = max(max_j, j)
    seen_ids.append(q)
    passed = max_j < 0.85
    return Check(
        score=round(1.0 - max_j, 3),
        passed=passed,
        reason=f"max_jaccard={max_j:.3f}",
    )


# ─────────────────────────── 7. Bloom Fidelity (zero-shot OpenAI) ───────────

BLOOM_PROMPTS = {
    "nhan_biet":   "recall or recognition of facts directly stated in text",
    "thong_hieu":  "comprehension or interpretation beyond literal text",
    "van_dung":    "application of knowledge to new situations",
    "van_dung_cao":"analysis, evaluation or synthesis of information",
}


def check_bloom_fidelity(item: MCQItem, client) -> Check:
    """
    Dùng GPT-4o-mini phân loại mức Bloom của câu hỏi sinh ra.
    So với bloom_requested trong request.
    """
    requested = item.request.bloom_requested.value
    q = item.question

    opts_text = "\n".join(
        f"- {o.label}. {o.text}" for o in item.options)

    prompt = (
        f"Classify the Bloom's taxonomy level of this multiple-choice question.\n\n"
        f"Question: {q}\n"
        f"Options:\n{opts_text}\n\n"
        f"Choose exactly one:\n"
        f"- nhan_biet ({BLOOM_PROMPTS['nhan_biet']})\n"
        f"- thong_hieu ({BLOOM_PROMPTS['thong_hieu']})\n"
        f"- van_dung ({BLOOM_PROMPTS['van_dung']})\n"
        f"- van_dung_cao ({BLOOM_PROMPTS['van_dung_cao']})\n\n"
        f"Output exactly one of: nhan_biet, thong_hieu, van_dung, van_dung_cao"
    )

    predicted = "nhan_biet"
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            raw = resp.choices[0].message.content.strip().lower()
            valid = ["nhan_biet", "thong_hieu", "van_dung", "van_dung_cao"]
            predicted = raw if raw in valid else "nhan_biet"
            break
        except Exception:
            if attempt == 0: time.sleep(1)

    # scoring: exact match = 1.0, adjacent level = 0.6, far = 0.2
    LEVELS = ["nhan_biet", "thong_hieu", "van_dung", "van_dung_cao"]
    r_idx = LEVELS.index(requested) if requested in LEVELS else 0
    p_idx = LEVELS.index(predicted) if predicted in LEVELS else 0
    diff  = abs(r_idx - p_idx)
    score_map = {0: 1.0, 1: 0.6, 2: 0.2, 3: 0.0}
    score  = score_map.get(diff, 0.0)
    passed = diff == 0

    return Check(
        score=round(score, 3),
        passed=passed,
        reason=f"requested={requested} predicted={predicted} diff={diff}",
    )


# ─────────────────────────── 8. Answer Position Balance ─────────────────────

def check_answer_position(item: MCQItem,
                           position_counts: dict) -> Check:
    """
    Theo dõi phân bố A/B/C/D theo batch.
    Passed luôn True cho item đơn lẻ; score dựa trên mức lệch hiện tại.
    """
    key = item.answer_key
    position_counts[key] = position_counts.get(key, 0) + 1
    total = sum(position_counts.values())
    pct   = {k: position_counts.get(k, 0)/total
              for k in ["A","B","C","D"]}
    max_dev = max(abs(p - 0.25) for p in pct.values())
    score   = max(0.0, 1.0 - 4 * max_dev)   # 0 dev → 1.0; 0.25 dev → 0.0
    return Check(
        score=round(score, 3),
        passed=True,
        reason=f"dist={pct} max_dev={max_dev:.2f}",
    )


# ─────────────────────────── Tổng hợp verifier_score ────────────────────────

def compute_verdict(checks: dict[str, Check]) -> tuple[float, VerdictStatus, list[str]]:
    score = sum(WEIGHTS[k] * checks[k].score for k in WEIGHTS if k in checks)
    score = round(score, 4)
    violations = [k for k, c in checks.items() if not c.passed]
    if score >= ACCEPT_THRESHOLD:
        status = VerdictStatus.accepted
    elif score >= REVIEW_THRESHOLD:
        status = VerdictStatus.needs_review
    else:
        status = VerdictStatus.rejected
    return score, status, violations


# ─────────────────────────── Pipeline chính ─────────────────────────────────

def load_kb_facts(entities_path: str) -> dict[str, set[str]]:
    """Tải KB: {context_id -> set(surface)} để kiểm tra historical correctness."""
    import pandas as pd
    df = pd.read_parquet(entities_path)
    kb: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        kb[row["context_id"]].add(str(row["surface"]))
    print(f"KB loaded: {len(kb)} contexts, "
          f"{sum(len(v) for v in kb.values())} entities")
    return dict(kb)


def verify_file(
    jsonl_path: str,
    client,
    kb_facts: dict,
    embedder=None,
    seen_embeddings: list = None,
    seen_dup_ids: list = None,
    position_counts: dict = None,
) -> list[MCQItem]:
    """Verify toàn bộ một file .jsonl, trả về list MCQItem đã điền verification."""
    if seen_embeddings  is None: seen_embeddings  = []
    if seen_dup_ids     is None: seen_dup_ids     = []
    if position_counts  is None: position_counts  = {}

    items_out = []
    path = Path(jsonl_path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    print(f"\nVerifying {path.name} ({len(lines)} câu)...")

    for i, line in enumerate(lines):
        raw  = json.loads(line)
        item = MCQItem.model_validate(raw)

        checks: dict[str, Check] = {}

        # 1. Evidence Match
        checks["evidence_match"] = check_evidence_match(item, client)

        # 2. Single Correct Answer
        checks["single_correct"] = check_single_correct(item, client)

        # 3. Distractor Type Match
        checks["distractor_type_match"] = check_distractor_type(item, embedder)

        # 4. Historical Correctness
        checks["historical_correctness"] = check_historical_correctness(
            item, kb_facts)

        # 5. Question Clarity
        checks["question_clarity"] = check_question_clarity(item)

        # 6. Duplicate Check
        checks["duplicate_check"] = check_duplicate(
            item, seen_embeddings, seen_dup_ids, embedder)

        # 7. Bloom Fidelity
        checks["bloom_fidelity"] = check_bloom_fidelity(item, client)

        # 8. Answer Position Balance
        checks["answer_position_ok"] = check_answer_position(
            item, position_counts)

        # Tổng hợp
        score, status, violations = compute_verdict(checks)
        item.verification = Verification(
            status=status,
            verifier_score=score,
            checks=checks,
            violations=violations,
        )
        items_out.append(item)

        if (i + 1) % 20 == 0 or (i + 1) == len(lines):
            n_acc = sum(1 for it in items_out
                        if it.verification.status == VerdictStatus.accepted)
            print(f"  [{i+1}/{len(lines)}] accepted={n_acc} "
                  f"({n_acc/(i+1):.0%})", flush=True)

    return items_out


# ─────────────────────────── CLI ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs",   nargs="+",
                    default=["data/generated/rule_based.jsonl",
                             "data/generated/rag_llm.jsonl",
                             "data/generated/vit5_ft.jsonl"])
    ap.add_argument("--entities", default="data/processed/entities.parquet")
    ap.add_argument("--outdir",   default="data/verified")
    ap.add_argument("--no_embed", action="store_true",
                    help="Bỏ qua sentence embedder (nhanh hơn, kém hơn)")
    args = ap.parse_args()

    # OpenAI client
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("LỖI: thiếu OPENAI_API_KEY")
        return
    client = OpenAI(api_key=api_key)

    # Sentence embedder cho distractor cosine + duplicate check
    embedder = None
    if not args.no_embed:
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(
                "bkai-foundation-models/vietnamese-bi-encoder")
            print("Embedder loaded.")
        except Exception as e:
            print(f"Không load được embedder ({e}). Dùng Jaccard fallback.")

    # KB
    kb_facts = load_kb_facts(args.entities)

    # State dùng chung qua 3 file (duplicate check cross-file)
    seen_embeddings: list = []
    seen_dup_ids:    list = []
    position_counts: dict = {}

    # Kết quả tổng hợp
    import csv
    import pandas as pd
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for jsonl_path in args.inputs:
        if not Path(jsonl_path).exists():
            print(f"Bỏ qua (không tìm thấy): {jsonl_path}")
            continue

        items = verify_file(
            jsonl_path, client, kb_facts,
            embedder=embedder,
            seen_embeddings=seen_embeddings,
            seen_dup_ids=seen_dup_ids,
            position_counts=position_counts,
        )

        # Lưu file verified
        method = Path(jsonl_path).stem
        out_path = Path(args.outdir) / f"verified_{method}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")

        # Thống kê
        n = len(items)
        n_acc = sum(1 for i in items
                    if i.verification.status == VerdictStatus.accepted)
        n_rev = sum(1 for i in items
                    if i.verification.status == VerdictStatus.needs_review)
        n_rej = sum(1 for i in items
                    if i.verification.status == VerdictStatus.rejected)
        avg_score = sum(i.verification.verifier_score for i in items) / n

        # Điểm trung bình từng tiêu chí
        crit_scores = {}
        for k in WEIGHTS:
            vals = [i.verification.checks[k].score
                    for i in items if k in i.verification.checks]
            crit_scores[k] = round(sum(vals)/len(vals), 3) if vals else 0.0

        print(f"\n=== {method.upper()} ===")
        print(f"  Tổng câu        : {n}")
        print(f"  Accepted  (VQR) : {n_acc} ({n_acc/n:.0%})")
        print(f"  Needs review    : {n_rev} ({n_rev/n:.0%})")
        print(f"  Rejected        : {n_rej} ({n_rej/n:.0%})")
        print(f"  Avg score       : {avg_score:.3f}")
        print("  Điểm từng tiêu chí:")
        for k, v in crit_scores.items():
            bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
            print(f"    {k:25s}: {v:.3f} |{bar}|")

        row = {"method": method, "n_items": n,
               "vqr": n_acc/n, "avg_score": avg_score,
               **crit_scores}
        summary_rows.append(row)

    # Bảng so sánh chính → results_table.csv
    if summary_rows:
        results_df = pd.DataFrame(summary_rows)
        results_path = Path(args.outdir) / "results_table.csv"
        results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
        print(f"\n\n{'='*60}")
        print(f"BẢNG SO SÁNH 3 PHƯƠNG PHÁP → {results_path}")
        print('='*60)
        print(results_df.to_string(index=False))
        print(f"\n✅ Xong. File results_table.csv dùng làm input cho AHP/TOPSIS.")


if __name__ == "__main__":
    main()
