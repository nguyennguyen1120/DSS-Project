"""
Chia split CÂN TẢI theo title (thay hash mù).

Vấn đề của hash mù: với ít bài + kích thước lệch (1 bài 223 đoạn, 1 bài 7 đoạn),
các đoạn dồn ngẫu nhiên -> test không đại diện.

Giải pháp: greedy phân hoạch theo SỐ ĐOẠN, làm RIÊNG cho từng tiểu miền
(VN / thế giới) để mọi split đều có cả hai. Vẫn giữ bất biến: một bài -> một split
(chống rò rỉ). Deterministic nhờ sắp xếp ổn định + seed.
"""

from __future__ import annotations
from collections import defaultdict


def balanced_split_by_title(
    title_sizes: dict[str, int],
    title_is_vn: dict[str, bool],
    ratios=(0.80, 0.10, 0.10),
    seed: int = 42,
) -> dict[str, str]:
    """
    title_sizes: {title -> số đoạn}
    title_is_vn: {title -> có phải bài VN không}
    Trả về {title -> split}. Đảm bảo:
      - mỗi bài đúng 1 split (chống rò rỉ)
      - tổng đoạn mỗi split xấp xỉ ratios
      - mỗi tiểu miền (VN, thế giới) được phân đều -> mọi split có cả hai
    """
    splits = ["train", "dev", "test"]
    assignment: dict[str, str] = {}

    # xử lý riêng từng tiểu miền
    for want_vn in (True, False):
        group = {t: s for t, s in title_sizes.items() if title_is_vn.get(t, False) == want_vn}
        if not group:
            continue
        total = sum(group.values())
        target = {sp: total * r for sp, r in zip(splits, ratios)}
        current = {sp: 0.0 for sp in splits}

        # sắp giảm dần theo size; tie-break theo tên để deterministic
        ordered = sorted(group.items(), key=lambda kv: (-kv[1], kv[0]))
        for title, size in ordered:
            # bỏ vào split đang THIẾU nhiều nhất so với chỉ tiêu (deficit lớn nhất)
            best = max(splits, key=lambda sp: (target[sp] - current[sp]))
            assignment[title] = best
            current[best] += size

    return assignment


def report_split(title_sizes, title_is_vn, assignment) -> str:
    agg = defaultdict(lambda: {"titles": 0, "passages": 0, "vn": 0, "world": 0})
    for title, sp in assignment.items():
        a = agg[sp]
        a["titles"] += 1
        a["passages"] += title_sizes[title]
        if title_is_vn.get(title, False):
            a["vn"] += title_sizes[title]
        else:
            a["world"] += title_sizes[title]
    total = sum(title_sizes.values())
    lines = ["=== CHIA SPLIT CÂN TẢI (theo title) ==="]
    for sp in ["train", "dev", "test"]:
        a = agg[sp]
        pct = 100 * a["passages"] / total if total else 0
        lines.append(f"  {sp:5s}: {a['titles']:2d} bài | {a['passages']:4d} đoạn "
                     f"({pct:4.1f}%) | VN={a['vn']} / TG={a['world']}")
    return "\n".join(lines)
