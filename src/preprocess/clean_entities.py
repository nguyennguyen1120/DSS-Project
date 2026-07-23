"""
Lớp lọc luật làm sạch thực thể NER (dùng chung cho mọi backend NER).

Bắt 4 khuôn mẫu lỗi phát hiện khi chấm mẫu:
  1. danh từ chung + tên  : "dân tộc Nga" -> "Nga";  "người Bồ Đào Nha" -> "Bồ Đào Nha"
  2. chức danh + tên      : "chủ tịch Hồ Chí Minh" -> "Hồ Chí Minh" (đổi sang PER)
  3. năm bị gán LOC/PER   : loại bỏ (đã có bản YEAR đúng từ regex)
  4. rác ranh giới / cụm  : bắt đầu bằng dấu câu, chứa động từ, quá ngắn, stopword
"""

from __future__ import annotations
import re

# danh từ chung đứng trước tên -> cần bóc
COMMON_PREFIXES = [
    "dân tộc", "người", "binh sĩ", "bộ đội", "tiếng", "thủ đô", "tỉnh",
    "thành phố", "sông", "núi", "đảo", "bán đảo", "miền", "vùng", "đất",
    "thái ấp", "bố chính", "quân", "quận", "huyện", "làng", "xã", "phủ",
    "vương quốc", "đế quốc", "nhà nước", "chính phủ", "quốc hội",
    "ủy ban", "hội đồng",
]
# chức danh đứng trước tên người -> bóc, và ép type PER
TITLE_PREFIXES = [
    "chủ tịch", "tổng thống", "đại tướng", "thủ tướng", "tổng bí thư",
    "giáo hoàng", "nữ hoàng", "quốc vương", "hoàng đế", "vua", "hoàng hậu",
    "thái tử", "công tước", "bá tước", "thủ lĩnh", "tướng", "đại tá",
    "trung tá", "thượng tá", "bộ trưởng", "thống đốc", "toàn quyền",
]
# động từ/từ nối -> nếu chứa thì là cụm, không phải thực thể tên
VERB_MARKERS = [
    "khinh thường", "chống", "kế vị", "đánh", "chiếm", "giết", "nên ",
    "liên danh", "liên quân", "viện trợ", "quan điểm",
    "và ", "hoặc ", "của ", "cho ", "với ", "tại ", "về ",
]
STOPWORDS = {"bộ", "thứ 2", "tờ 2", "người", "bác", "đảng", "nhà nước",
             "chính phủ", "ông", "bà", "họ", "nó", "mình",
             # Khuôn mẫu B: 1-token hay bị cắt nhầm
             "nam", "bắc", "đông", "tây", "quốc", "thành", "công", "hội"}

# Khuôn mẫu C: từ tôn giáo/hệ tư tưởng hay bị gán ORG nhầm
RELIGION_MARKERS = ["hồi giáo", "phật giáo", "thiên chúa", "cơ đốc",
                    "tin lành", "ấn độ giáo", "nho giáo", "đạo giáo"]

# Khuôn mẫu A: subword artifact ELECTRA
SUBWORD_RE = re.compile(r"^##|^[A-Z]{1,3}$")   # "##s", "FD", "BT"...

YEAR_ONLY = re.compile(r"^(năm\s+)?\d{1,4}(\s*(tr(ước)?\.?\s*C(ông)?\.?N(guyên)?|TCN))?$",
                       re.IGNORECASE)
PUNCT_START = re.compile(r'^[\s,."\'();:\-–]')


def _strip_prefix(surface: str, prefixes: list[str]) -> tuple[str, bool]:
    """Bóc tiền tố (không phân biệt hoa/thường). Trả về (còn lại, có_bóc)."""
    low = surface.lower()
    for p in sorted(prefixes, key=len, reverse=True):
        if low.startswith(p + " "):
            return surface[len(p):].strip(), True
    return surface, False


def clean_entity(surface: str, typ: str) -> tuple[str, str] | None:
    """
    Làm sạch một thực thể. Trả về (surface_mới, type_mới) hoặc None nếu loại bỏ.
    KHÔNG áp cho YEAR/DATE/CENTURY (đã đúng từ regex).
    """
    if typ in ("YEAR", "DATE", "CENTURY"):
        return surface, typ

    s = surface.strip().strip('.,;:"\'()')
    if not s:
        return None

    # 4a. rác ranh giới: bắt đầu bằng dấu câu (đã strip ở trên nhưng phòng hờ)
    if PUNCT_START.match(surface):
        s = surface.strip().strip('.,;:"\'()[]')
        if not s:
            return None

    # 3. năm bị gán LOC/PER -> loại (đã có bản YEAR)
    if YEAR_ONLY.match(s):
        return None

    # Khuôn mẫu A: subword artifact / fragment ELECTRA
    if SUBWORD_RE.match(s):
        return None

    # 4b. stopword / rác
    if s.lower() in STOPWORDS:
        return None

    # Khuôn mẫu A (bổ sung): quá ngắn sau khi đã strip (≤ 3 ký tự = fragment)
    # ngoại trừ tên quốc gia ngắn hợp lệ: Mỹ, Anh, Nga, Đức, Pháp, Ý...
    SHORT_WHITELIST = {"mỹ", "anh", "nga", "đức", "pháp", "ý", "lào", "hàn",
                       "nhật", "tàu", "hán", "minh", "thanh", "tần", "hán",
                       "trần", "lý", "lê", "hồ", "mạc", "tây"}
    if len(s) <= 3 and typ in ("PER", "LOC", "ORG") and s.lower() not in SHORT_WHITELIST:
        return None

    # Khuôn mẫu C: ORG chứa từ tôn giáo/hệ tư tưởng → loại
    if typ == "ORG" and any(r in s.lower() for r in RELIGION_MARKERS):
        return None

    # 4c. chứa động từ/từ nối -> cụm, loại
    low = s.lower()
    if any(v in low for v in VERB_MARKERS):
        return None

    # 2. chức danh + tên người -> bóc, ép PER
    stripped, did = _strip_prefix(s, TITLE_PREFIXES)
    if did:
        s = stripped
        typ = "PER"

    # 1. danh từ chung + tên -> bóc (giữ nguyên type gốc thường là LOC)
    s2, did2 = _strip_prefix(s, COMMON_PREFIXES)
    if did2:
        s = s2

    s = s.strip().strip('.,;:"\'()')
    # sau khi bóc: quá ngắn hoặc lại là năm -> loại
    if len(s) < 2 or YEAR_ONLY.match(s) or s.lower() in STOPWORDS:
        return None
    # còn dấu câu giữa hoặc toàn số -> loại
    if s.isdigit():
        return None

    return s, typ


def clean_entities(entities: list[dict]) -> list[dict]:
    """Áp clean_entity lên list bản ghi {surface, type, ...}. Bỏ bản None,
    cập nhật surface/type/normalized."""
    out = []
    for e in entities:
        # thời gian: giữ NGUYÊN bản ghi (normalized đã chuẩn từ regex)
        if e["type"] in ("YEAR", "DATE", "CENTURY"):
            out.append(e)
            continue
        res = clean_entity(e["surface"], e["type"])
        if res is None:
            continue
        new_surface, new_type = res
        e2 = dict(e)
        e2["surface"] = new_surface
        e2["type"] = new_type
        e2["normalized"] = new_surface.lower().replace(" ", "_")
        out.append(e2)
    return out
