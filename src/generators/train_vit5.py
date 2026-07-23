"""
Train ViT5-base — Stage A (Question Generation) + Stage B (Distractor Generation)
Chạy trên máy local RTX 3050 4GB VRAM.

Cài đặt:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.44.2 datasets==2.20.0 accelerate sentencepiece pandas pyarrow

Chạy:
    python src/generators/train_vit5.py --outdir models
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import json
from collections import defaultdict
from pathlib import Path

# fix import
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "schema"))
sys.path.insert(0, str(_SRC / "preprocess"))

MAX_SRC = 256    # nhỏ hơn Colab vì VRAM ít hơn
MAX_TGT = 64


# ─────────────────────────── Data prep ───────────────────────────────────────

def infer_wh_type(question: str, answer: str) -> str:
    YEAR_RE = re.compile(r"^\d{3,4}$")
    if YEAR_RE.match(str(answer).strip()):
        return "thoi_gian"
    q = str(question).lower()
    WH = {
        "thoi_gian":   ["khi nào","năm nào","bao giờ","thời gian","ngày nào"],
        "nhan_vat":    ["ai ","người nào","nhân vật","ông nào","vua nào"],
        "dia_diem":    ["ở đâu","tại đâu","địa điểm","nơi nào"],
        "nguyen_nhan": ["vì sao","tại sao","nguyên nhân","lý do"],
        "y_nghia":     ["ý nghĩa","kết quả","hệ quả","dẫn đến"],
    }
    for wh, kws in WH.items():
        if any(k in q for k in kws):
            return wh
    return "su_kien"


def load_data(data_dir: str):
    import pandas as pd
    qa  = pd.read_parquet(f"{data_dir}/qa_pairs.parquet")
    ctx = pd.read_parquet(f"{data_dir}/corpus.parquet")

    qa_clean = qa[(qa.answer_span_ok == True) & (qa.is_impossible == False)].copy()

    # Drop các cột đã có trong corpus để tránh _x/_y khi merge
    drop_cols = [c for c in ["context", "title", "is_vietnam"]
                 if c in qa_clean.columns]
    if drop_cols:
        qa_clean = qa_clean.drop(columns=drop_cols)

    df = qa_clean.merge(
        ctx[["context_id","context","title","is_vietnam"]],
        on="context_id", how="inner"
    )
    df["wh_type"] = df.apply(
        lambda r: infer_wh_type(r["question"], r["answer_text"]), axis=1)

    # Stage A: QG
    df["qg_input"] = df.apply(
        lambda r: f"<type> {r['wh_type']} </type> "
                  f"<ans> {r['answer_text']} </ans> "
                  f"<ctx> {r['context'][:500]} </ctx>",
        axis=1)
    df["qg_target"] = df["question"]

    # Stage B: DG — distractor gold từ các answer khác cùng context
    ctx_answers = defaultdict(set)
    for _, row in df.iterrows():
        ctx_answers[row["context_id"]].add(str(row["answer_text"]))
    for _, row in qa[qa.is_impossible == True].iterrows():
        if row.get("plausible_answer"):
            ctx_answers[row["context_id"]].add(str(row["plausible_answer"]))

    dg_rows = []
    for _, row in df.iterrows():
        pool = [a for a in ctx_answers[row["context_id"]]
                if a != row["answer_text"] and a.strip()]
        if pool:
            dg_rows.append({
                "dg_input":  f"<q> {row['question']} </q> "
                             f"<ans> {row['answer_text']} </ans> "
                             f"<ctx> {row['context'][:350]} </ctx>",
                "dg_target": " | ".join(pool[:3]),
                "split":     row["split"],
            })
    import pandas as pd
    dg_df = pd.DataFrame(dg_rows)

    splits = {
        "train_qg": df[df["split"]=="train"][["qg_input","qg_target"]].reset_index(drop=True),
        "dev_qg":   df[df["split"]=="dev" ][["qg_input","qg_target"]].reset_index(drop=True),
        "train_dg": dg_df[dg_df["split"]=="train"][["dg_input","dg_target"]].reset_index(drop=True),
        "dev_dg":   dg_df[dg_df["split"]=="dev" ][["dg_input","dg_target"]].reset_index(drop=True),
        "test_df":  df[df["split"]=="test"].reset_index(drop=True),
    }
    print(f"QG train={len(splits['train_qg']):,} dev={len(splits['dev_qg']):,}")
    print(f"DG train={len(splits['train_dg']):,} dev={len(splits['dev_dg']):,}")
    print(f"Test : {len(splits['test_df']):,}")
    return splits


# ─────────────────────────── Tokenize ────────────────────────────────────────

def make_tokenize_fn(tokenizer, vocab_size, max_src, max_tgt):
    def tokenize_fn(examples, src_col, tgt_col):
        model_inputs = tokenizer(
            examples[src_col],
            max_length=max_src, truncation=True, padding="max_length",
        )
        labels_enc = tokenizer(
            text_target=examples[tgt_col],
            max_length=max_tgt, truncation=True, padding="max_length",
        )
        # Clamp input_ids và labels về [0, vocab_size-1]
        model_inputs["input_ids"] = [
            [min(max(t, 0), vocab_size-1) for t in ids]
            for ids in model_inputs["input_ids"]
        ]
        model_inputs["labels"] = [
            [(-100 if t == tokenizer.pad_token_id
              else min(max(t, 0), vocab_size-1))
             for t in lbl]
            for lbl in labels_enc["input_ids"]
        ]
        return model_inputs
    return tokenize_fn


# ─────────────────────────── Training loop ───────────────────────────────────

def train_stage(
    train_df, dev_df, src_col, tgt_col,
    tokenizer, vocab_size, output_dir,
    model_name="VietAI/vit5-base",
    epochs=5, batch_size=4, grad_accum=8,   # batch 4 × accum 8 = eff. 32
    lr=3e-5, patience=2,
    max_src=MAX_SRC, max_tgt=MAX_TGT,
):
    import torch
    from torch.utils.data import DataLoader
    from torch.amp import autocast, GradScaler
    from transformers import AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
    from datasets import Dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    tokenize_fn = make_tokenize_fn(tokenizer, vocab_size, max_src, max_tgt)

    # Tokenize
    ds_tr = Dataset.from_pandas(train_df)
    ds_dv = Dataset.from_pandas(dev_df)
    tok_tr = ds_tr.map(
        lambda x: tokenize_fn(x, src_col, tgt_col),
        batched=True, remove_columns=ds_tr.column_names,
        load_from_cache_file=False,
    )
    tok_dv = ds_dv.map(
        lambda x: tokenize_fn(x, src_col, tgt_col),
        batched=True, remove_columns=ds_dv.column_names,
        load_from_cache_file=False,
    )
    tok_tr.set_format("torch")
    tok_dv.set_format("torch")

    # Verify clamp
    max_id = tok_tr[0]["input_ids"].max().item()
    assert max_id < vocab_size, f"Clamp thất bại: {max_id} >= {vocab_size}"
    print(f"Clamp OK (max_id={max_id} < vocab_size={vocab_size})")

    tr_loader = DataLoader(tok_tr, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=(device.type=="cuda"))
    dv_loader = DataLoader(tok_dv, batch_size=batch_size*2, shuffle=False,
                           num_workers=0)

    # Model ban đầu (có thể bị ghi đè khi resume)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.resize_token_embeddings(vocab_size)
    model = model.to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    best_loss, no_improve = float("inf"), 0
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(output_dir) / "checkpoint_latest"

    # Resume từ checkpoint nếu có
    start_epoch = 0
    if checkpoint_dir.exists() and (checkpoint_dir / "trainer_state.json").exists():
        state = json.load(open(checkpoint_dir / "trainer_state.json"))
        start_epoch = state["epoch"] + 1
        best_loss   = state["best_loss"]
        no_improve  = state["no_improve"]
        print(f"Resume từ checkpoint epoch {state['epoch']} "
              f"(best_loss={best_loss:.4f}, no_improve={no_improve})")
        model = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint_dir))
        model.resize_token_embeddings(vocab_size)
        model = model.to(device)
    else:
        print("Không có checkpoint — train từ đầu.")

    # Tạo optimizer SAU khi biết model (quan trọng khi resume)
    total_steps = (len(tr_loader) // grad_accum) * epochs
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler  = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05*total_steps)),
        num_training_steps=total_steps,
    )
    use_amp = device.type == "cuda"
    scaler  = GradScaler("cuda") if use_amp else None

    # Load optimizer state nếu resume
    if start_epoch > 0:
        opt_path = checkpoint_dir / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(
                torch.load(str(opt_path), map_location=device))
            print("  Optimizer state restored.")

    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        tr_loss, steps = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(tr_loader):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["labels"].to(device)

            if use_amp:
                with autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=ids, attention_mask=mask,
                                 labels=lbl).loss / grad_accum
                scaler.scale(loss).backward()
            else:
                loss = model(input_ids=ids, attention_mask=mask,
                             labels=lbl).loss / grad_accum
                loss.backward()

            tr_loss += loss.item() * grad_accum
            steps   += 1

            if (step+1) % grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step+1) % 100 == 0:
                print(f"  step {step+1}/{len(tr_loader)} "
                      f"loss={tr_loss/steps:.4f}", flush=True)

        # Eval
        model.eval()
        dv_loss, dv_steps = 0.0, 0
        with torch.no_grad():
            for batch in dv_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbl  = batch["labels"].to(device)
                if use_amp:
                    with autocast("cuda", dtype=torch.float16):
                        loss = model(input_ids=ids, attention_mask=mask, labels=lbl).loss
                else:
                    loss = model(input_ids=ids, attention_mask=mask, labels=lbl).loss
                dv_loss += loss.item(); dv_steps += 1
        dv_loss /= dv_steps

        print(f"Epoch {epoch+1}/{epochs} | "
              f"train={tr_loss/steps:.4f} | dev={dv_loss:.4f}", flush=True)

        # Bước 1: luôn lưu checkpoint TRƯỚC (dù có improve hay không)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for param in model.parameters():
            param.data = param.data.contiguous()
        model.save_pretrained(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))
        torch.save(optimizer.state_dict(), str(checkpoint_dir / "optimizer.pt"))

        # Bước 2: kiểm tra improve và lưu best model
        if dv_loss < best_loss:
            best_loss  = dv_loss
            no_improve = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  ✅ Saved best (dev={dv_loss:.4f}) -> {output_dir}")
        else:
            no_improve += 1
            print(f"  No improve ({no_improve}/{patience})")

        # Bước 3: lưu state SAU khi đã save xong cả hai
        json.dump({
            "epoch":      epoch,
            "best_loss":  best_loss,
            "no_improve": no_improve,
            "train_loss": round(tr_loss/steps, 4),
            "dev_loss":   round(dv_loss, 4),
        }, open(checkpoint_dir / "trainer_state.json", "w"), indent=2)
        print(f"  💾 Checkpoint saved (epoch={epoch})")

        if no_improve >= patience:
            print("  Early stopping.")
            break

    del model
    if device.type == "cuda":
        import torch; torch.cuda.empty_cache()
    return best_loss


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--outdir",   default="models")
    ap.add_argument("--model",    default="VietAI/vit5-base")
    ap.add_argument("--epochs",   type=int, default=5)
    ap.add_argument("--batch",    type=int, default=4)
    ap.add_argument("--accum",    type=int, default=8)
    ap.add_argument("--lr",       type=float, default=3e-5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--stage",    choices=["A","B","AB"], default="AB",
                    help="A=QG only, B=DG only, AB=cả hai")
    args = ap.parse_args()

    import os
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    from transformers import AutoTokenizer

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, legacy=True)
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size:,}")

    # Load data
    print("\nLoading data...")
    splits = load_data(args.data_dir)

    # Stage A — Question Generation
    if args.stage in ("A","AB"):
        qg_out = str(Path(args.outdir) / "vit5_qg")
        print(f"\n{'='*50}")
        print(f"STAGE A: Question Generation -> {qg_out}")
        print(f"{'='*50}")
        best = train_stage(
            splits["train_qg"], splits["dev_qg"],
            src_col="qg_input", tgt_col="qg_target",
            tokenizer=tokenizer, vocab_size=vocab_size,
            output_dir=qg_out,
            model_name=args.model,
            epochs=args.epochs, batch_size=args.batch,
            grad_accum=args.accum, lr=args.lr, patience=args.patience,
        )
        print(f"Stage A done. Best dev loss: {best:.4f}")

    # Stage B — Distractor Generation
    if args.stage in ("B","AB"):
        dg_out = str(Path(args.outdir) / "vit5_dg")
        print(f"\n{'='*50}")
        print(f"STAGE B: Distractor Generation -> {dg_out}")
        print(f"{'='*50}")
        best = train_stage(
            splits["train_dg"], splits["dev_dg"],
            src_col="dg_input", tgt_col="dg_target",
            tokenizer=tokenizer, vocab_size=vocab_size,
            output_dir=dg_out,
            model_name=args.model,
            epochs=args.epochs, batch_size=args.batch,
            grad_accum=args.accum, lr=args.lr, patience=args.patience,
            max_tgt=48,
        )
        print(f"Stage B done. Best dev loss: {best:.4f}")

    print("\n✅ Training hoàn tất.")


if __name__ == "__main__":
    main()
