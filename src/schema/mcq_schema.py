"""
Schema chuẩn cho MỘT câu hỏi trắc nghiệm — DÙNG CHUNG cho cả 3 phương pháp
(Rule-based / RAG+LLM / ViT5) và cho Verifier + DSS.

Nguyên tắc:
  - Generator điền phần "sinh" (question, options, evidence, generator, source, request).
  - Verifier điền phần "kiểm chứng" (verification).
  - DSS đọc phần metadata + verifier_score.

Ràng buộc CỨNG (bất biến, chặn lỗi tại nguồn):
  - đúng 4 phương án, nhãn A/B/C/D, không trùng
  - đúng 1 đáp án đúng -> 3 distractor
  - answer_key / answer_text khớp phương án is_correct
  - evidence.sentence phải là substring của context khi found_in_context=True

Ngoài ra cung cấp finalize_options(): mọi generator gọi để chuẩn hoá 4 phương án.
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ============================ Enums =========================================

class Method(str, Enum):
    rule_based = "rule_based"
    rag_llm = "rag_llm"
    vit5_ft = "vit5_ft"


class Bloom(str, Enum):
    nhan_biet = "nhan_biet"
    thong_hieu = "thong_hieu"
    van_dung = "van_dung"
    van_dung_cao = "van_dung_cao"


class WhType(str, Enum):
    thoi_gian = "thoi_gian"
    nhan_vat = "nhan_vat"
    dia_diem = "dia_diem"
    su_kien = "su_kien"
    nguyen_nhan = "nguyen_nhan"
    y_nghia = "y_nghia"


class Provenance(str, Enum):
    context_span = "context_span"   # trích từ ngữ cảnh (đáp án đúng thường là đây)
    kb = "kb"                       # từ Knowledge Base (rule-based distractor)
    retrieved = "retrieved"         # từ retrieval pool
    generated = "generated"         # LLM/ViT5 sinh
    gold = "gold"                   # distractor người viết (khi dùng gold)


class VerdictStatus(str, Enum):
    accepted = "accepted"
    needs_review = "needs_review"
    rejected = "rejected"


# ============================ Thành phần ====================================

class Option(BaseModel):
    label: str                              # "A".."D"
    text: str
    is_correct: bool
    entity_type: Optional[str] = None       # YEAR/PER/LOC/... (để kiểm Distractor Type Match)
    provenance: Provenance


class Generator(BaseModel):
    method: Method
    variant: str = "base"                   # vd "few_shot_v2", "template_dep"
    model_name: Optional[str] = None        # vd "gpt-4o-mini", "vit5-base"
    params_hash: Optional[str] = None


class Source(BaseModel):
    dataset: str = "uit_viquad2"
    context_id: str
    context: str                            # đoạn ngữ cảnh -> INPUT nguồn 1
    title: Optional[str] = None
    is_vietnam: Optional[bool] = None


class Request(BaseModel):
    """ECHO của INPUT nguồn 2 — không được sửa sau khi sinh.
    Dùng để Verifier chấm Bloom Fidelity + chứng minh 3 phương pháp nhận cùng input."""
    bloom_requested: Bloom
    wh_type_requested: WhType


class Evidence(BaseModel):
    sentence: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    found_in_context: bool


class Metadata(BaseModel):
    wh_type_detected: Optional[WhType] = None
    bloom_predicted: Optional[Bloom] = None
    difficulty_score: Optional[float] = None
    difficulty_band: Optional[str] = None       # de / trung_binh / kho
    topic: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class GenerationTrace(BaseModel):
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    n_llm_calls: int = 0
    retrieved_context_ids: list[str] = Field(default_factory=list)


class Check(BaseModel):
    score: float
    passed: bool
    reason: str = ""


class Verification(BaseModel):
    status: VerdictStatus
    verifier_score: float
    checks: dict[str, Check] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)


# ============================ Item chính ====================================

class MCQItem(BaseModel):
    schema_version: str = "1.0"
    item_id: str
    generator: Generator
    source: Source
    request: Request
    question: str                               # thân câu hỏi (stem)
    options: list[Option] = Field(min_length=4, max_length=4)
    answer_key: str                             # "A".."D"
    answer_text: str
    evidence: Evidence
    metadata: Metadata = Field(default_factory=Metadata)
    generation_trace: GenerationTrace = Field(default_factory=GenerationTrace)
    verification: Optional[Verification] = None     # None lúc sinh; Verifier điền
    dedup: dict = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def _invariants(self):
        # đúng 4 phương án, nhãn A/B/C/D
        assert len(self.options) == 4, "Phải có đúng 4 phương án"
        labels = sorted(o.label for o in self.options)
        assert labels == ["A", "B", "C", "D"], f"Nhãn phải A,B,C,D; nhận {labels}"
        # không trùng nội dung
        texts = [o.text.strip() for o in self.options]
        assert len(set(texts)) == 4, f"Phương án trùng nội dung: {texts}"
        # đúng 1 đáp án đúng
        corrects = [o for o in self.options if o.is_correct]
        assert len(corrects) == 1, "Phải có đúng 1 đáp án đúng (=> 3 distractor)"
        assert corrects[0].label == self.answer_key, "answer_key lệch is_correct"
        assert corrects[0].text == self.answer_text, "answer_text lệch phương án đúng"
        # evidence phải nằm trong context nếu khai báo có
        if self.evidence.found_in_context:
            assert self.evidence.sentence in self.source.context, \
                "evidence.sentence phải là substring nguyên văn của context"
        return self


# ============================ Tiện ích dùng chung ===========================

def make_item_id(context_id: str, question: str, method: str) -> str:
    raw = f"{context_id}|{question}|{method}"
    return "itm_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def finalize_options(
    correct_text: str,
    distractors: list[str],
    correct_provenance: Provenance = Provenance.context_span,
    distractor_provenance: Provenance = Provenance.generated,
    correct_entity_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> tuple[list[Option], str]:
    """
    Chuẩn hoá thành ĐÚNG 4 phương án cho MỌI generator. Trả về (options, answer_key).

    - khử trùng distractor (và trùng với đáp án đúng), so khớp không phân biệt hoa/thường
    - nếu < 3 distractor hợp lệ -> raise (generator phải cấp fallback trước khi gọi)
    - nếu > 3 -> lấy 3 đầu tiên
    - xáo trộn vị trí A/B/C/D (khử thiên lệch vị trí), deterministic nếu có seed
    """
    correct_text = correct_text.strip()
    seen = {correct_text.lower()}
    clean_distractors = []
    for d in distractors:
        d = d.strip()
        if d and d.lower() not in seen:
            seen.add(d.lower())
            clean_distractors.append(d)

    if len(clean_distractors) < 3:
        raise ValueError(
            f"Chỉ có {len(clean_distractors)} distractor hợp lệ (<3). "
            f"Generator phải bổ sung fallback trước khi gọi finalize_options. "
            f"correct={correct_text!r}, distractors={distractors!r}")

    chosen = clean_distractors[:3]

    entries = [(correct_text, True)] + [(d, False) for d in chosen]
    rng = random.Random(seed)
    rng.shuffle(entries)

    labels = ["A", "B", "C", "D"]
    options, answer_key = [], None
    for lab, (text, is_corr) in zip(labels, entries):
        prov = correct_provenance if is_corr else distractor_provenance
        etype = correct_entity_type if is_corr else correct_entity_type
        options.append(Option(label=lab, text=text, is_correct=is_corr,
                              entity_type=etype, provenance=prov))
        if is_corr:
            answer_key = lab
    return options, answer_key


if __name__ == "__main__":
    # sinh file JSON Schema chính thức để nộp kèm báo cáo
    import json
    from pathlib import Path
    schema = MCQItem.model_json_schema()
    out = Path("mcq_item.schema.json")
    out.write_text(json.dumps(schema, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"Đã sinh JSON Schema -> {out}")
