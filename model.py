"""
model.py
Transformer-based Psychological Manipulation Detector
Uses facebook/bart-large-mnli for zero-shot multi-label classification
No fine-tuning needed — works out of the box for hackathon
"""

import torch
import numpy as np
import re
from transformers import pipeline

# ── Manipulation Tactic Labels ──────────────────────────────────────────────
TACTIC_LABELS = [
    "Fear & Urgency",
    "False Social Proof",
    "Identity Attack",
    "Emotional Hijacking",
    "Scarcity Illusion",
    "Gaslighting"
]

TACTIC_DESCRIPTIONS = {
    "Fear & Urgency":      "Creates panic or time pressure to bypass rational thinking",
    "False Social Proof":  "Uses fake or exaggerated popularity to influence behavior",
    "Identity Attack":     "Targets personal identity to shame or manipulate",
    "Emotional Hijacking": "Overloads emotions to override logical decision-making",
    "Scarcity Illusion":   "Fabricates limited availability to trigger impulsive action",
    "Gaslighting":         "Contradicts reality to make the reader doubt their perception"
}

TACTIC_COLORS = {
    "Fear & Urgency":      "#ff4d6a",
    "False Social Proof":  "#ffb547",
    "Identity Attack":     "#a78bfa",
    "Emotional Hijacking": "#f472b6",
    "Scarcity Illusion":   "#38bdf8",
    "Gaslighting":         "#fb923c"
}


class ManipulationDetector:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"[Model] Loading zero-shot classifier on {'GPU' if self.device == 0 else 'CPU'}...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )
        print("[Model] Ready!")

    # ── Main Analysis ────────────────────────────────────────────────────────
    def analyze_text(self, text: str) -> dict:
        if not text or len(text.strip()) < 10:
            return self._empty_result()

        chunks = self._chunk_text(text)
        all_scores = []

        for chunk in chunks[:5]:          # cap at 5 chunks for speed
            result = self.classifier(
                chunk,
                candidate_labels=TACTIC_LABELS,
                multi_label=True
            )
            scores = dict(zip(result["labels"], result["scores"]))
            all_scores.append(scores)

        # Average across chunks
        tactic_scores = {
            label: float(np.mean([s.get(label, 0) for s in all_scores]))
            for label in TACTIC_LABELS
        }

        # Scale 0-100 with sensitivity boost
        raw = float(np.mean(list(tactic_scores.values())))
        overall = min(100.0, raw * 180)

        sorted_tactics = sorted(tactic_scores.items(), key=lambda x: x[1], reverse=True)
        top_tactics = [t for t, s in sorted_tactics if s > 0.25]

        if overall >= 70:
            severity, sev_color = "HIGH",     "#ef4444"
        elif overall >= 40:
            severity, sev_color = "MODERATE", "#f59e0b"
        else:
            severity, sev_color = "LOW",      "#22c55e"

        return {
            "overall_score":       round(overall, 1),
            "severity":            severity,
            "severity_color":      sev_color,
            "tactic_scores":       {k: round(v * 100, 1) for k, v in tactic_scores.items()},
            "tactic_colors":       TACTIC_COLORS,
            "top_tactics":         top_tactics[:3],
            "tactic_descriptions": TACTIC_DESCRIPTIONS,
            "chunks_analyzed":     len(chunks[:5]),
            "word_count":          len(text.split())
        }

    # ── Sentence Heatmap ─────────────────────────────────────────────────────
    def highlight_sentences(self, text: str) -> list:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        results = []

        for sent in sentences[:20]:
            if len(sent.split()) < 4:
                results.append({"text": sent, "score": 0.0, "tactic": None})
                continue
            try:
                res = self.classifier(
                    sent,
                    candidate_labels=TACTIC_LABELS,
                    multi_label=False
                )
                top_score = float(res["scores"][0])
                top_label = res["labels"][0]
                scaled    = min(1.0, top_score * 1.8)
                results.append({
                    "text":   sent,
                    "score":  round(scaled, 3),
                    "tactic": top_label if top_score > 0.3 else None
                })
            except Exception:
                results.append({"text": sent, "score": 0.0, "tactic": None})

        return results

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _chunk_text(self, text: str, size: int = 400) -> list:
        words  = text.split()
        chunks = [" ".join(words[i:i+size]) for i in range(0, len(words), size)]
        return [c for c in chunks if c.strip()] or [text]

    def _empty_result(self) -> dict:
        return {
            "overall_score":       0.0,
            "severity":            "LOW",
            "severity_color":      "#22c55e",
            "tactic_scores":       {k: 0.0 for k in TACTIC_LABELS},
            "tactic_colors":       TACTIC_COLORS,
            "top_tactics":         [],
            "tactic_descriptions": TACTIC_DESCRIPTIONS,
            "chunks_analyzed":     0,
            "word_count":          0
        }


# Module-level singleton loaded once at import
detector = ManipulationDetector()