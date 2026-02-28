"""
app.py
PsychoGuard AI — FastAPI backend
Run: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from model import detector
from llm_analyzer import generate_explanation
from multimodal import analyze_image_caption, analyze_clickbait
from report_gen import generate_report

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="PsychoGuard AI", version="1.0.0", description="Psychological Manipulation Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response schemas ──────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str
    url: Optional[str] = ""
    image_url: Optional[str] = ""
    caption: Optional[str] = ""
    include_explanation: bool = True
    include_highlights: bool = True


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "PsychoGuard AI running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "facebook/bart-large-mnli"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.text or len(req.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text must be at least 10 characters.")

    start = time.time()

    # 1. Transformer classification
    analysis = detector.analyze_text(req.text)

    # 2. Sentence-level heatmap
    highlights = []
    if req.include_highlights:
        highlights = detector.highlight_sentences(req.text)

    # 3. LLM explanation via Claude
    explanation = {"success": False, "full_explanation": "LLM disabled", "sections": {}}
    if req.include_explanation:
        explanation = generate_explanation(req.text, analysis)

    # 4. Multimodal analysis
    multimodal: dict = {"available": False}
    if req.image_url:
        if req.caption:
            multimodal = analyze_image_caption(req.image_url, req.caption)
        else:
            multimodal = analyze_clickbait(req.image_url)

    return {
        "analysis":        analysis,
        "highlights":      highlights,
        "explanation":     explanation,
        "multimodal":      multimodal,
        "processing_time": round(time.time() - start, 2),
    }


@app.post("/quick-score")
def quick_score(req: AnalyzeRequest):
    """Lightweight endpoint used by the browser extension."""
    analysis = detector.analyze_text(req.text)
    return {
        "overall_score": analysis["overall_score"],
        "severity":      analysis["severity"],
        "severity_color":analysis["severity_color"],
        "top_tactics":   analysis["top_tactics"],
    }


@app.post("/report")
def download_report(req: AnalyzeRequest):
    """Generate and return a PDF analysis report."""
    if not req.text or len(req.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text too short.")

    analysis    = detector.analyze_text(req.text)
    explanation = generate_explanation(req.text, analysis)
    pdf_buf     = generate_report(req.text, analysis, explanation, req.url or "")

    return StreamingResponse(
        pdf_buf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="psychoguard-report.pdf"'}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
