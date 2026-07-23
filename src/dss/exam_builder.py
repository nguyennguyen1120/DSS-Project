"""
ILP Exam Builder — Ráp đề thi từ ngân hàng câu hỏi đã qua Verifier.

Bài toán tối ưu:
    max  Σ verifier_score(i) · x(i)
    s.t. Σ x(i) = K                          (đúng K câu)
         phân bố Bloom khớp ma trận đặc tả   (±1 câu)
         mỗi title (chủ đề) ≥ 1 câu          (phủ đủ chủ đề)
         mỗi context_id đóng góp ≤ 2 câu     (không lặp ngữ cảnh)
         x(i) ∈ {0,1}

So sánh 3 chiến lược: ILP (tối ưu) vs Greedy vs Random.

Cài:
    pip install pulp pandas

Chạy:
    python src/dss/exam_builder.py \\
        --verified_dir data/verified \\
        --n_questions 40 \\
        --bloom_ratio 0.4,0.35,0.25 \\
        --outdir data/exams
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ─────────────────────────── Load ngân hàng câu hỏi ─────────────────────────

def load_bank(verified_dir: str) -> pd.DataFrame:
    """Gộp 3 file verified_*.jsonl, chỉ lấy câu accepted."""
    rows = []
    for f in Path(verified_dir).glob("verified_*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            v = item.get("verification") or {}
            if v.get("status") != "accepted":
                continue
            rows.append({
                "item_id":        item["item_id"],
                "method":         item["generator"]["method"],
                "question":       item["question"],
                "answer_key":     item["answer_key"],
                "answer_text":    item["answer_text"],
                "options":        item["options"],
                "context_id":     item["source"]["context_id"],
                "title":          item["source"].get("title", ""),
                "is_vietnam":     item["source"].get("is_vietnam", False),
                "bloom":          item["request"]["bloom_requested"],
                "wh_type":        item["request"]["wh_type_requested"],
                "verifier_score": v.get("verifier_score", 0.5),
            })
    df = pd.DataFrame(rows)
    print(f"Ngân hàng: {len(df)} câu accepted "
          f"từ {df['method'].nunique()} phương pháp")
    print(df["bloom"].value_counts().to_string())
    return df


# ─────────────────────────── Phân bố Bloom target ───────────────────────────

BLOOM_ORDER = ["nhan_biet", "thong_hieu", "van_dung", "van_dung_cao"]


def parse_bloom_ratio(ratio_str: str, k: int) -> dict[str, int]:
    """
    Chuyển chuỗi tỉ lệ (vd '0.4,0.35,0.25') thành số câu mỗi mức Bloom.
    Hỗ trợ 3 mức (bỏ van_dung_cao) hoặc 4 mức.
    """
    parts = [float(x) for x in ratio_str.split(",")]
    # chuẩn hoá tổng = 1
    total = sum(parts)
    parts = [p / total for p in parts]
    # gán cho các mức Bloom
    levels = BLOOM_ORDER[:len(parts)]
    counts = {}
    allocated = 0
    for i, (lv, ratio) in enumerate(zip(levels, parts)):
        if i == len(parts) - 1:
            counts[lv] = k - allocated
        else:
            counts[lv] = round(k * ratio)
            allocated += counts[lv]
    # mức còn lại (van_dung_cao nếu chỉ có 3 tỉ lệ)
    for lv in BLOOM_ORDER:
        if lv not in counts:
            counts[lv] = 0
    return counts


# ─────────────────────────── ILP ─────────────────────────────────────────────

def solve_ilp(
    df: pd.DataFrame,
    k: int,
    bloom_target: dict[str, int],
    max_per_context: int = 2,
    bloom_slack: int = 1,
) -> tuple[list[str], float]:
    """
    Giải ILP bằng PuLP. Trả về (danh sách item_id, tổng score).
    """
    try:
        import pulp
    except ImportError:
        print("Cần cài: pip install pulp")
        return [], 0.0

    prob = pulp.LpProblem("ExamBuilder", pulp.LpMaximize)
    n    = len(df)
    ids  = df["item_id"].tolist()
    x    = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Mục tiêu: tối đa tổng verifier_score
    prob += pulp.lpSum(df.iloc[i]["verifier_score"] * x[i] for i in range(n))

    # Ràng buộc 1: đúng K câu
    prob += pulp.lpSum(x) == k

    # Ràng buộc 2: phân bố Bloom (±slack)
    for bloom, target in bloom_target.items():
        if target == 0:
            continue
        indices = [i for i in range(n) if df.iloc[i]["bloom"] == bloom]
        if not indices:
            continue
        total_b = pulp.lpSum(x[i] for i in indices)
        prob += total_b >= max(0, target - bloom_slack)
        prob += total_b <= target + bloom_slack

    # Ràng buộc 3: mỗi context_id ≤ max_per_context câu
    ctx_groups = defaultdict(list)
    for i in range(n):
        ctx_groups[df.iloc[i]["context_id"]].append(i)
    for ctx, idxs in ctx_groups.items():
        prob += pulp.lpSum(x[i] for i in idxs) <= max_per_context

    # Ràng buộc 4: mỗi title ≥ 1 câu (chỉ áp với title có >= 2 câu)
    title_groups = defaultdict(list)
    for i in range(n):
        t = df.iloc[i]["title"]
        if t:
            title_groups[t].append(i)
    n_titles = len([t for t, idxs in title_groups.items() if len(idxs) >= 2])
    if n_titles <= k:   # chỉ áp khi số title <= K (khả thi)
        for title, idxs in title_groups.items():
            if len(idxs) >= 2:
                prob += pulp.lpSum(x[i] for i in idxs) >= 1

    # Giải
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    status = prob.solve(solver)

    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        print(f"  ILP status: {pulp.LpStatus[status]} — thử nới slack")
        return [], 0.0

    selected = [ids[i] for i in range(n) if pulp.value(x[i]) == 1]
    total_score = pulp.value(prob.objective)
    return selected, total_score


# ─────────────────────────── Greedy ──────────────────────────────────────────

def solve_greedy(
    df: pd.DataFrame,
    k: int,
    bloom_target: dict[str, int],
    max_per_context: int = 2,
) -> tuple[list[str], float]:
    """Greedy: sắp giảm dần verifier_score, chọn câu nào thoả ràng buộc."""
    df_sorted = df.sort_values("verifier_score", ascending=False)
    selected, ctx_count, bloom_count = [], defaultdict(int), defaultdict(int)

    for _, row in df_sorted.iterrows():
        if len(selected) >= k:
            break
        cid   = row["context_id"]
        bloom = row["bloom"]
        if ctx_count[cid] >= max_per_context:
            continue
        if bloom_count[bloom] >= bloom_target.get(bloom, 0) + 1:
            continue
        selected.append(row["item_id"])
        ctx_count[cid]   += 1
        bloom_count[bloom] += 1

    # nếu chưa đủ K (Bloom ràng buộc quá chặt), nới bloom
    if len(selected) < k:
        for _, row in df_sorted.iterrows():
            if len(selected) >= k:
                break
            if row["item_id"] not in selected:
                if ctx_count[row["context_id"]] < max_per_context:
                    selected.append(row["item_id"])
                    ctx_count[row["context_id"]] += 1

    score = df[df["item_id"].isin(selected)]["verifier_score"].sum()
    return selected, score


# ─────────────────────────── Random ──────────────────────────────────────────

def solve_random(
    df: pd.DataFrame,
    k: int,
    seed: int = 42,
) -> tuple[list[str], float]:
    """Baseline: chọn ngẫu nhiên K câu."""
    sample = df.sample(n=min(k, len(df)), random_state=seed)
    score  = sample["verifier_score"].sum()
    return sample["item_id"].tolist(), score


# ─────────────────────────── Báo cáo ─────────────────────────────────────────

def build_exam_df(df: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    """Tạo DataFrame đề thi từ danh sách item_id."""
    exam = df[df["item_id"].isin(selected)].copy()
    exam = exam.sort_values(["bloom", "title"]).reset_index(drop=True)
    exam.index += 1
    return exam


def print_exam_stats(label: str, df: pd.DataFrame,
                     selected: list[str], score: float,
                     bloom_target: dict, elapsed: float):
    exam = build_exam_df(df, selected)
    print(f"\n{'='*55}")
    print(f"{label}")
    print(f"{'='*55}")
    print(f"  Số câu chọn   : {len(selected)}")
    print(f"  Tổng score    : {score:.4f} (avg={score/max(len(selected),1):.4f})")
    print(f"  Thời gian     : {elapsed*1000:.0f}ms")
    print(f"  Phân bố Bloom :")
    for b in BLOOM_ORDER:
        n = (exam["bloom"] == b).sum()
        t = bloom_target.get(b, 0)
        ok = "✅" if abs(n - t) <= 1 else "⚠️"
        print(f"    {b:15s}: {n:2d} / {t:2d} {ok}")
    print(f"  Phân bố phương pháp:")
    for m, cnt in exam["method"].value_counts().items():
        print(f"    {m:12s}: {cnt}")
    print(f"  Số chủ đề phủ : {exam['title'].nunique()}")
    return exam


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified_dir", default="data/verified")
    ap.add_argument("--n_questions",  type=int, default=40)
    ap.add_argument("--bloom_ratio",  default="0.4,0.35,0.25",
                    help="Tỉ lệ Bloom: nhan_biet,thong_hieu,van_dung[,van_dung_cao]")
    ap.add_argument("--max_per_ctx",  type=int, default=2)
    ap.add_argument("--bloom_slack",  type=int, default=1)
    ap.add_argument("--outdir",       default="data/exams")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    k = args.n_questions

    # Load ngân hàng
    df = load_bank(args.verified_dir)
    if len(df) < k:
        print(f"[CẢNH BÁO] Chỉ có {len(df)} câu accepted < {k} câu yêu cầu.")
        k = len(df)

    bloom_target = parse_bloom_ratio(args.bloom_ratio, k)
    print(f"\nMa trận đặc tả ({k} câu):")
    for b, cnt in bloom_target.items():
        if cnt > 0:
            print(f"  {b:15s}: {cnt} câu ({cnt/k*100:.0f}%)")

    results = []

    # ── ILP ──────────────────────────────────────────────────────────────────
    print("\n[ILP] Đang giải...")
    t0 = time.time()
    ilp_ids, ilp_score = solve_ilp(
        df, k, bloom_target, args.max_per_ctx, args.bloom_slack)
    ilp_time = time.time() - t0

    if ilp_ids:
        ilp_exam = print_exam_stats(
            "ILP (Tối ưu)", df, ilp_ids, ilp_score, bloom_target, ilp_time)
        ilp_exam.to_csv(Path(args.outdir) / "exam_ilp.csv",
                        index=True, encoding="utf-8-sig")
        results.append({"method": "ILP", "score": ilp_score,
                        "n": len(ilp_ids), "time_ms": ilp_time*1000})
    else:
        print("  ILP không tìm được nghiệm — dùng Greedy thay thế")

    # ── Greedy ───────────────────────────────────────────────────────────────
    print("\n[Greedy] Đang chọn...")
    t0 = time.time()
    g_ids, g_score = solve_greedy(df, k, bloom_target, args.max_per_ctx)
    g_time = time.time() - t0

    g_exam = print_exam_stats(
        "Greedy", df, g_ids, g_score, bloom_target, g_time)
    g_exam.to_csv(Path(args.outdir) / "exam_greedy.csv",
                  index=True, encoding="utf-8-sig")
    results.append({"method": "Greedy", "score": g_score,
                    "n": len(g_ids), "time_ms": g_time*1000})

    # ── Random ───────────────────────────────────────────────────────────────
    print("\n[Random] Đang chọn...")
    t0 = time.time()
    r_ids, r_score = solve_random(df, k, args.seed)
    r_time = time.time() - t0

    r_exam = print_exam_stats(
        "Random (baseline)", df, r_ids, r_score, bloom_target, r_time)
    r_exam.to_csv(Path(args.outdir) / "exam_random.csv",
                  index=True, encoding="utf-8-sig")
    results.append({"method": "Random", "score": r_score,
                    "n": len(r_ids), "time_ms": r_time*1000})

    # ── So sánh 3 chiến lược ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("SO SÁNH 3 CHIẾN LƯỢC RÁP ĐỀ")
    print(f"{'='*55}")
    cmp_df = pd.DataFrame(results)
    cmp_df["avg_score"] = cmp_df["score"] / cmp_df["n"]
    cmp_df["gain_vs_random_%"] = (
        (cmp_df["score"] - cmp_df.loc[cmp_df.method=="Random","score"].values[0])
        / cmp_df.loc[cmp_df.method=="Random","score"].values[0] * 100
    ).round(1)
    print(cmp_df.to_string(index=False))
    cmp_df.to_csv(Path(args.outdir) / "strategy_comparison.csv",
                  index=False, encoding="utf-8-sig")

    print(f"\nĐã lưu vào {args.outdir}/:")
    print("  exam_ilp.csv, exam_greedy.csv, exam_random.csv, strategy_comparison.csv")

    # In mẫu đề thi ILP
    best_exam = ilp_exam if ilp_ids else g_exam
    print(f"\n{'='*55}")
    print("MẪU ĐỀ THI (5 câu đầu)")
    print(f"{'='*55}")
    for i, row in best_exam.head(5).iterrows():
        print(f"\nCâu {i}. [{row['bloom']}] {row['question']}")
        for opt in row["options"]:
            mark = " ←" if opt["is_correct"] else ""
            print(f"   {opt['label']}. {opt['text']}{mark}")


if __name__ == "__main__":
    main()
