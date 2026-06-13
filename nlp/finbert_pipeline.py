# nlp/finbert_pipeline.py

import pandas as pd
import numpy as np
import re
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))
from config import SUPPLY_CHAIN_KEYWORDS

# ─────────────────────────────────────────────
# Load FinBERT model
# Downloads once (~500MB), cached after that
# ─────────────────────────────────────────────

def load_finbert():
    print("Loading FinBERT model (downloads once, ~500MB)...")
    model_name = "ProsusAI/finbert"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    print("✅ FinBERT loaded")
    return tokenizer, model


# ─────────────────────────────────────────────
# Score a single text chunk
# Returns: {"positive": float, "negative": float, "neutral": float}
# ─────────────────────────────────────────────

def score_text(text: str, tokenizer, model) -> dict:
    inputs = tokenizer(
        text,
        return_tensors  = "pt",
        truncation      = True,
        max_length      = 512,
        padding         = True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1).squeeze()

    # FinBERT label order: positive, negative, neutral
    return {
        "positive": probs[0].item(),
        "negative": probs[1].item(),
        "neutral":  probs[2].item(),
    }


# ─────────────────────────────────────────────
# Split MD&A into paragraphs
# ─────────────────────────────────────────────

def split_paragraphs(text: str) -> list:
    # Split on double spaces or sentence boundaries
    paragraphs = re.split(r'\s{2,}|(?<=[.!?])\s+(?=[A-Z])', text)
    # Keep only substantial paragraphs
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 100]
    return paragraphs


# ─────────────────────────────────────────────
# Filter paragraphs to supply chain relevant ones
# ─────────────────────────────────────────────

def filter_supply_chain_paragraphs(paragraphs: list, keywords: list) -> list:
    relevant = []
    for para in paragraphs:
        para_lower = para.lower()
        if any(kw in para_lower for kw in keywords):
            relevant.append(para)
    return relevant


# ─────────────────────────────────────────────
# Compute Supply Chain Stress Score for one filing
#
# Score logic:
#   - Higher negative sentiment in SC paragraphs = higher stress
#   - Also track keyword density as a signal
#   - Final score: 0 (no stress) to 1 (max stress)
# ─────────────────────────────────────────────

def compute_stress_score(mda_text: str, tokenizer, model, keywords: list) -> dict:
    paragraphs    = split_paragraphs(mda_text)
    sc_paragraphs = filter_supply_chain_paragraphs(paragraphs, keywords)

    total_paragraphs = len(paragraphs)
    sc_count         = len(sc_paragraphs)

    if sc_count == 0:
        return {
            "stress_score":       0.0,
            "avg_negative":       0.0,
            "avg_positive":       0.0,
            "avg_neutral":        0.0,
            "sc_paragraph_count": 0,
            "sc_paragraph_ratio": 0.0,
            "keyword_density":    0.0,
        }

    # Score each SC paragraph
    sentiments = []
    for para in sc_paragraphs:
        # Truncate to 512 tokens worth of text (~1800 chars)
        para_truncated = para[:1800]
        scores = score_text(para_truncated, tokenizer, model)
        sentiments.append(scores)

    avg_negative = np.mean([s["negative"] for s in sentiments])
    avg_positive = np.mean([s["positive"] for s in sentiments])
    avg_neutral  = np.mean([s["neutral"]  for s in sentiments])

    # Keyword density: total keyword mentions per 1000 words
    full_text_lower = mda_text.lower()
    total_words     = len(mda_text.split())
    keyword_hits    = sum(full_text_lower.count(kw) for kw in keywords)
    keyword_density = (keyword_hits / total_words) * 1000 if total_words > 0 else 0

    # Stress score: weighted combo of negative sentiment + keyword density
    # Normalise keyword density (cap at 20 per 1000 words)
    kd_normalised = min(keyword_density / 20.0, 1.0)
    stress_score  = (0.7 * avg_negative) + (0.3 * kd_normalised)

    return {
        "stress_score":       round(stress_score, 4),
        "avg_negative":       round(avg_negative, 4),
        "avg_positive":       round(avg_positive, 4),
        "avg_neutral":        round(avg_neutral,  4),
        "sc_paragraph_count": sc_count,
        "sc_paragraph_ratio": round(sc_count / total_paragraphs, 4) if total_paragraphs > 0 else 0,
        "keyword_density":    round(keyword_density, 4),
    }


# ─────────────────────────────────────────────
# Process all filings
# ─────────────────────────────────────────────

def process_all_filings(
    input_path:  str = "data/mda_raw.csv",
    output_path: str = "data/stress_scores.csv"
) -> pd.DataFrame:

    print(f"Loading filings from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  {len(df)} filings to process across {df['company'].nunique()} companies")

    tokenizer, model = load_finbert()

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring filings"):
        try:
            scores = compute_stress_score(
                mda_text  = row["mda_text"],
                tokenizer = tokenizer,
                model     = model,
                keywords  = SUPPLY_CHAIN_KEYWORDS
            )

            results.append({
                "company":            row["company"],
                "cik":                row["cik"],
                "date":               row["date"],
                "accession":          row["accession"],
                **scores
            })

        except Exception as e:
            print(f"  [ERROR] {row['company']} {row['date']}: {e}")
            continue

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df["date"] = pd.to_datetime(results_df["date"])
        results_df = results_df.sort_values(["company", "date"]).reset_index(drop=True)
        os.makedirs("data", exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved stress scores to {output_path}")

        # Quick summary
        print("\nAverage stress score by company:")
        summary = results_df.groupby("company")["stress_score"].agg(["mean", "max", "min"])
        print(summary.round(4).to_string())

    return results_df


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = process_all_filings(
        input_path  = "data/mda_raw.csv",
        output_path = "data/stress_scores.csv"
    )
