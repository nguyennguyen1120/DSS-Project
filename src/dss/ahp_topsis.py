"""
AHP + TOPSIS — Hệ hỗ trợ ra quyết định chọn phương pháp sinh MCQ.

Input : data/verified/results_table.csv
Output:
  - data/dss/ahp_weights.csv       — trọng số AHP + Consistency Ratio
  - data/dss/topsis_ranking.csv    — xếp hạng TOPSIS
  - data/dss/sensitivity.csv       — phân tích độ nhạy
  - data/dss/dss_report.txt        — báo cáo tổng hợp

Chạy:
    python src/dss/ahp_topsis.py --results data/verified/results_table.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────── Tiêu chí và cấu hình ───────────────────────────

# 8 tiêu chí từ results_table.csv
CRITERIA = [
    "vqr",
    "evidence_match",
    "single_correct",
    "distractor_type_match",
    "historical_correctness",
    "question_clarity",
    "bloom_fidelity",
    "duplicate_check",
]

# Tên hiển thị đẹp hơn
CRITERIA_LABELS = {
    "vqr":                    "VQR (Valid Question Rate)",
    "evidence_match":         "Evidence Match",
    "single_correct":         "Single Correct Answer",
    "distractor_type_match":  "Distractor Type Match",
    "historical_correctness": "Historical Correctness",
    "question_clarity":       "Question Clarity",
    "bloom_fidelity":         "Bloom Fidelity",
    "duplicate_check":        "Duplicate Check",
}

# Hướng: benefit (càng cao càng tốt) hay cost (càng thấp càng tốt)
CRITERIA_TYPE = {
    "vqr":                    "benefit",
    "evidence_match":         "benefit",
    "single_correct":         "benefit",
    "distractor_type_match":  "benefit",
    "historical_correctness": "benefit",
    "question_clarity":       "benefit",
    "bloom_fidelity":         "benefit",
    "duplicate_check":        "benefit",
}

# Ma trận so sánh cặp AHP (8×8)
# Thang Saaty 1–9: 1=ngang nhau, 3=hơi quan trọng hơn, 5=quan trọng hơn,
#                  7=rất quan trọng, 9=tuyệt đối quan trọng
# Quan điểm: giáo viên ưu tiên chất lượng câu hỏi > distractor > tốc độ
#
# Hàng/cột theo thứ tự CRITERIA ở trên
AHP_MATRIX = np.array([
    # vqr  ev   sc   dt   hc   qc   bf   dup
    [1,    2,   3,   3,   2,   2,   3,   4   ],  # vqr
    [1/2,  1,   2,   2,   2,   2,   2,   3   ],  # evidence_match
    [1/3,  1/2, 1,   2,   1,   2,   2,   3   ],  # single_correct
    [1/3,  1/2, 1/2, 1,   1,   1,   2,   3   ],  # distractor_type_match
    [1/2,  1/2, 1,   1,   1,   2,   2,   3   ],  # historical_correctness
    [1/2,  1/2, 1/2, 1,   1/2, 1,   1,   2   ],  # question_clarity
    [1/3,  1/2, 1/2, 1/2, 1/2, 1,   1,   2   ],  # bloom_fidelity
    [1/4,  1/3, 1/3, 1/3, 1/3, 1/2, 1/2, 1   ],  # duplicate_check
], dtype=float)

# Chỉ số ngẫu nhiên RI theo Saaty (theo số tiêu chí n)
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
      6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


# ─────────────────────────── AHP ─────────────────────────────────────────────

def compute_ahp(matrix: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Tính trọng số AHP và Consistency Ratio.
    Trả về (weights, lambda_max, CR).
    CR < 0.1 → ma trận nhất quán.
    """
    n = matrix.shape[0]

    # Bước 1: chuẩn hoá cột
    col_sum = matrix.sum(axis=0)
    norm    = matrix / col_sum

    # Bước 2: vector trọng số = trung bình hàng
    weights = norm.mean(axis=1)

    # Bước 3: lambda_max
    weighted = matrix @ weights
    lambda_max = (weighted / weights).mean()

    # Bước 4: Consistency Index và Ratio
    ci = (lambda_max - n) / (n - 1)
    ri = RI.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0

    return weights, lambda_max, cr


# ─────────────────────────── TOPSIS ──────────────────────────────────────────

def compute_topsis(
    decision_matrix: np.ndarray,   # shape (n_alt, n_crit)
    weights: np.ndarray,            # shape (n_crit,)
    criteria_types: list[str],      # "benefit" | "cost"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TOPSIS chuẩn. Trả về (scores, pis_dist, nis_dist).
    score cao → phương án tốt hơn.
    """
    # Bước 1: chuẩn hoá vector
    norms = np.sqrt((decision_matrix ** 2).sum(axis=0))
    norms[norms == 0] = 1e-10
    r = decision_matrix / norms

    # Bước 2: ma trận có trọng số
    v = r * weights

    # Bước 3: PIS và NIS
    pis = np.array([
        v[:, j].max() if criteria_types[j] == "benefit" else v[:, j].min()
        for j in range(v.shape[1])
    ])
    nis = np.array([
        v[:, j].min() if criteria_types[j] == "benefit" else v[:, j].max()
        for j in range(v.shape[1])
    ])

    # Bước 4: khoảng cách Euclidean
    d_pis = np.sqrt(((v - pis) ** 2).sum(axis=1))
    d_nis = np.sqrt(((v - nis) ** 2).sum(axis=1))

    # Bước 5: điểm gần gũi
    scores = d_nis / (d_pis + d_nis + 1e-10)

    return scores, d_pis, d_nis


# ─────────────────────────── Phân tích độ nhạy ───────────────────────────────

def sensitivity_analysis(
    decision_matrix: np.ndarray,
    base_weights: np.ndarray,
    criteria_types: list[str],
    criteria_names: list[str],
    vary_criterion: int = 0,   # index của tiêu chí thay đổi (mặc định: vqr)
    n_steps: int = 20,
) -> pd.DataFrame:
    """
    Cho trọng số của tiêu chí `vary_criterion` chạy từ 0.01 → 0.50,
    phân phối lại trọng số còn lại theo tỉ lệ.
    Trả về DataFrame: weight_varied × score mỗi phương án.
    """
    rows = []
    w_range = np.linspace(0.01, 0.50, n_steps)

    for w_new in w_range:
        w = base_weights.copy()
        old_w = w[vary_criterion]
        remaining = w_new if w_new < 1.0 else 0.99
        scale = (1.0 - remaining) / (1.0 - old_w) if (1.0 - old_w) > 0 else 1.0
        w = w * scale
        w[vary_criterion] = remaining
        w = w / w.sum()   # renormalize

        scores, _, _ = compute_topsis(decision_matrix, w, criteria_types)
        row = {"weight_varied": round(w_new, 3)}
        for idx, score in enumerate(scores):
            row[f"alt_{idx}"] = round(score, 4)
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/verified/results_table.csv")
    ap.add_argument("--outdir",  default="data/dss")
    ap.add_argument("--vary",    default="vqr",
                    help="Tiêu chí thay đổi trong phân tích độ nhạy")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load kết quả
    df = pd.read_csv(args.results)
    methods = df["method"].tolist()
    n_alt   = len(methods)
    print(f"Phương pháp: {methods}")
    print(f"Tiêu chí   : {CRITERIA}\n")

    # Ma trận quyết định (n_alt × n_crit)
    dm = df[CRITERIA].values.astype(float)
    crit_types = [CRITERIA_TYPE[c] for c in CRITERIA]

    # ── AHP ──────────────────────────────────────────────────────────────────
    weights, lmax, cr = compute_ahp(AHP_MATRIX)

    print("=" * 60)
    print("AHP — TRỌNG SỐ TIÊU CHÍ")
    print("=" * 60)
    ahp_rows = []
    for i, c in enumerate(CRITERIA):
        print(f"  {CRITERIA_LABELS[c]:30s}: {weights[i]:.4f}")
        ahp_rows.append({"criterion": c, "label": CRITERIA_LABELS[c],
                          "weight": round(weights[i], 4)})
    print(f"\n  lambda_max = {lmax:.4f}")
    print(f"  CR = {cr:.4f}  {'✅ Nhất quán (CR<0.1)' if cr < 0.1 else '❌ CHƯA NHẤT QUÁN'}")

    ahp_df = pd.DataFrame(ahp_rows)
    ahp_df["lambda_max"] = round(lmax, 4)
    ahp_df["CR"]         = round(cr, 4)
    ahp_df.to_csv(Path(args.outdir) / "ahp_weights.csv",
                  index=False, encoding="utf-8-sig")

    if cr >= 0.1:
        print("\n[CẢNH BÁO] CR >= 0.1. Cần điều chỉnh ma trận AHP_MATRIX.")
        print("  Gợi ý: giảm độ chênh lệch giữa các tiêu chí kề nhau.")

    # ── TOPSIS ───────────────────────────────────────────────────────────────
    scores, d_pis, d_nis = compute_topsis(dm, weights, crit_types)
    ranks  = len(scores) - scores.argsort().argsort()

    print("\n" + "=" * 60)
    print("TOPSIS — XẾP HẠNG PHƯƠNG ÁN")
    print("=" * 60)
    topsis_rows = []
    for i, m in enumerate(methods):
        print(f"  {m:12s}: score={scores[i]:.4f}  "
              f"d_pis={d_pis[i]:.4f}  d_nis={d_nis[i]:.4f}  "
              f"rank={ranks[i]}")
        topsis_rows.append({
            "method": m, "topsis_score": round(scores[i], 4),
            "d_pis": round(d_pis[i], 4), "d_nis": round(d_nis[i], 4),
            "rank": int(ranks[i]),
        })
    best = methods[scores.argmax()]
    print(f"\n  => Phương pháp tốt nhất: {best.upper()}")

    topsis_df = pd.DataFrame(topsis_rows).sort_values("rank")
    topsis_df.to_csv(Path(args.outdir) / "topsis_ranking.csv",
                     index=False, encoding="utf-8-sig")

    # ── Phân tích độ nhạy ────────────────────────────────────────────────────
    vary_idx = CRITERIA.index(args.vary) if args.vary in CRITERIA else 0
    print(f"\n{'='*60}")
    print(f"PHÂN TÍCH ĐỘ NHẠY — thay đổi trọng số '{CRITERIA[vary_idx]}'")
    print("=" * 60)

    sens_df = sensitivity_analysis(dm, weights, crit_types, CRITERIA,
                                   vary_criterion=vary_idx)

    # tìm điểm giao (thứ hạng đổi)
    crossover_points = []
    for step in range(1, len(sens_df)):
        prev = sens_df.iloc[step-1]
        curr = sens_df.iloc[step]
        # so sánh mọi cặp phương án
        for a in range(n_alt):
            for b in range(a+1, n_alt):
                ka, kb = f"alt_{a}", f"alt_{b}"
                if (prev[ka] > prev[kb]) != (curr[ka] > curr[kb]):
                    w_cross = (prev["weight_varied"] + curr["weight_varied"]) / 2
                    crossover_points.append({
                        "weight": round(w_cross, 3),
                        "swap": f"{methods[a]} <-> {methods[b]}",
                    })
                    print(f"  Điểm giao w={w_cross:.3f}: "
                          f"{methods[a]} và {methods[b]} đổi thứ hạng")

    if not crossover_points:
        print(f"  Không có điểm giao — thứ hạng ổn định khi "
              f"trọng số '{CRITERIA[vary_idx]}' thay đổi 1%→50%")

    # lưu sensitivity
    # đổi tên cột alt_0/1/2 → tên phương pháp
    rename = {f"alt_{i}": methods[i] for i in range(n_alt)}
    sens_df = sens_df.rename(columns=rename)
    sens_df.to_csv(Path(args.outdir) / "sensitivity.csv",
                   index=False, encoding="utf-8-sig")

    # ── Báo cáo tổng hợp ─────────────────────────────────────────────────────
    report_lines = [
        "=" * 60,
        "BÁO CÁO DSS — SO SÁNH PHƯƠNG PHÁP SINH MCQ",
        "=" * 60,
        "",
        "1. TRỌNG SỐ TIÊU CHÍ (AHP)",
    ]
    for i, c in enumerate(CRITERIA):
        report_lines.append(
            f"   {CRITERIA_LABELS[c]:30s}: {weights[i]:.4f} ({weights[i]*100:.1f}%)")
    report_lines += [
        f"   CR = {cr:.4f} ({'dat' if cr < 0.1 else 'CHUA DAT'} nguong 0.1)",
        "",
        "2. XEP HANG TOPSIS",
    ]
    for row in sorted(topsis_rows, key=lambda x: x["rank"]):
        report_lines.append(
            f"   Hang {row['rank']}: {row['method']:12s} "
            f"score={row['topsis_score']:.4f}")
    report_lines += [
        "",
        f"3. KET LUAN DSS",
        f"   Phuong phap tot nhat (tong the): {best.upper()}",
        "",
        "   Diem manh theo phuong phap:",
    ]
    # điểm mạnh từng phương pháp
    for i, m in enumerate(methods):
        best_crit = CRITERIA[dm[i].argmax()]
        report_lines.append(
            f"   - {m:12s}: manh nhat o '{CRITERIA_LABELS[best_crit]}' "
            f"({dm[i].max():.3f})")
    report_lines += [
        "",
        "4. PHAN TICH DO NHAY",
        f"   Tieu chi thay doi: {CRITERIA[vary_idx]}",
    ]
    if crossover_points:
        for cp in crossover_points:
            report_lines.append(
                f"   Diem giao w={cp['weight']}: {cp['swap']}")
    else:
        report_lines.append("   Thu hang on dinh (khong co diem giao)")
    report_lines += [
        "",
        "5. DE XUAT",
        "   - Su dung RAG+LLM khi chat luong la uu tien (VQR cao nhat)",
        "   - Su dung Rule-based khi can kiem soat lich su + offline + $0",
        "   - ViT5 phu hop sinh cau hoi (QG) nhung nen ket hop voi",
        "     RAG distractor de nang chat luong phuong an nhieu",
        "   - Chien luoc lai (hybrid routing) toi uu cho moi loai cau hoi",
    ]

    report = "\n".join(report_lines)
    print("\n" + report)
    (Path(args.outdir) / "dss_report.txt").write_text(
        report, encoding="utf-8")

    print(f"\n4 file da luu vao {args.outdir}/:")
    print(f"  ahp_weights.csv, topsis_ranking.csv, "
          f"sensitivity.csv, dss_report.txt")


if __name__ == "__main__":
    main()
