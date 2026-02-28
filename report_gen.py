"""
report_gen.py
Generates a professional PDF analysis report using ReportLab.
"""

import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

# ── Colour palette ────────────────────────────────────────────────────────────
C_DARK    = HexColor("#0f1628")
C_ACCENT  = HexColor("#4f7cff")
C_RED     = HexColor("#ef4444")
C_AMBER   = HexColor("#f59e0b")
C_GREEN   = HexColor("#22c55e")
C_SURFACE = HexColor("#f8fafc")
C_GRAY    = HexColor("#64748b")
C_BORDER  = HexColor("#e2e8f0")


def _sty(name, **kw) -> ParagraphStyle:
    return ParagraphStyle(name, **kw)


def generate_report(text: str, analysis: dict, explanation: dict, url: str = "") -> BytesIO:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "PsychoGuard AI — Manipulation Analysis Report",
        _sty("H", fontSize=22, fontName="Helvetica-Bold",
             textColor=C_DARK, spaceAfter=4)
    ))
    story.append(Paragraph(
        f"Generated {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}",
        _sty("Sub", fontSize=9, fontName="Helvetica", textColor=C_GRAY, spaceAfter=14)
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=C_ACCENT))
    story.append(Spacer(1, 0.4*cm))

    # ── Score overview table ──────────────────────────────────────────────────
    score    = analysis.get("overall_score", 0)
    severity = analysis.get("severity", "LOW")
    sev_col  = C_RED if severity == "HIGH" else C_AMBER if severity == "MODERATE" else C_GREEN
    n_tact   = len(analysis.get("top_tactics", []))

    hdr_sty  = _sty("th", fontSize=9,  fontName="Helvetica-Bold", textColor=C_GRAY, alignment=TA_CENTER)
    val_sty  = _sty("tv", fontSize=34, fontName="Helvetica-Bold", textColor=sev_col, alignment=TA_CENTER)

    ov_data = [
        [Paragraph("MANIPULATION SCORE", hdr_sty),
         Paragraph("SEVERITY", hdr_sty),
         Paragraph("TACTICS FOUND", hdr_sty)],
        [Paragraph(f"{score}/100", val_sty),
         Paragraph(severity, val_sty),
         Paragraph(str(n_tact),
                   _sty("tv2", fontSize=34, fontName="Helvetica-Bold",
                        textColor=C_ACCENT, alignment=TA_CENTER))]
    ]
    ov_table = Table(ov_data, colWidths=["33%","34%","33%"])
    ov_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), C_SURFACE),
        ("BOX",          (0,0), (-1,-1), 1, C_BORDER),
        ("INNERGRID",    (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",   (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0), (-1,-1), 10),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
    ]))
    story.append(ov_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Tactic breakdown ──────────────────────────────────────────────────────
    story.append(Paragraph(
        "Tactic Breakdown",
        _sty("SH", fontSize=13, fontName="Helvetica-Bold",
             textColor=C_DARK, spaceAfter=8)
    ))

    tactic_scores = analysis.get("tactic_scores", {})
    trow = [["Tactic", "Score", "Risk"]]
    for tactic, val in tactic_scores.items():
        risk = "HIGH" if val >= 60 else "MODERATE" if val >= 35 else "LOW"
        trow.append([tactic, f"{val:.1f}/100", risk])

    bt = Table(trow, colWidths=["55%","20%","25%"])
    bt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),  (-1,0),  C_DARK),
        ("TEXTCOLOR",     (0,0),  (-1,0),  white),
        ("FONTNAME",      (0,0),  (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),  (-1,-1), 10),
        ("ALIGN",         (1,0),  (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1),  (-1,-1), [white, C_SURFACE]),
        ("BOX",           (0,0),  (-1,-1), 1, C_BORDER),
        ("INNERGRID",     (0,0),  (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0),  (-1,-1), 7),
        ("BOTTOMPADDING", (0,0),  (-1,-1), 7),
        ("LEFTPADDING",   (0,0),  (-1,-1), 10),
    ]))
    story.append(bt)
    story.append(Spacer(1, 0.5*cm))

    # ── LLM explanation ───────────────────────────────────────────────────────
    body_sty = _sty("body", fontSize=10, fontName="Helvetica",
                    textColor=HexColor("#374151"), leading=15, spaceAfter=6)

    if explanation.get("success"):
        story.append(Paragraph(
            "AI Analysis (Claude)",
            _sty("SH2", fontSize=13, fontName="Helvetica-Bold",
                 textColor=C_DARK, spaceAfter=8)
        ))
        sections = explanation.get("sections", {})
        for heading, content in sections.items():
            if content:
                story.append(Paragraph(
                    heading,
                    _sty(f"sec_{heading}", fontSize=10, fontName="Helvetica-Bold",
                         textColor=C_ACCENT, spaceAfter=3)
                ))
                story.append(Paragraph(content, body_sty))
        if not sections and explanation.get("full_explanation"):
            story.append(Paragraph(explanation["full_explanation"], body_sty))
        story.append(Spacer(1, 0.3*cm))

    # ── Analysed text excerpt ─────────────────────────────────────────────────
    if text:
        story.append(HRFlowable(width="100%", thickness=1, color=C_BORDER))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(
            "Analysed Content (excerpt)",
            _sty("SH3", fontSize=11, fontName="Helvetica-Bold",
                 textColor=C_DARK, spaceAfter=6)
        ))
        excerpt = text[:800] + ("…" if len(text) > 800 else "")
        story.append(Paragraph(
            excerpt,
            _sty("exc", fontSize=9, fontName="Helvetica",
                 textColor=C_GRAY, leading=13,
                 backColor=C_SURFACE, leftIndent=8, rightIndent=8,
                 borderPadding=8)
        ))

    # ── Source URL ────────────────────────────────────────────────────────────
    if url:
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(
            f"Source: {url}",
            _sty("url", fontSize=8, textColor=C_GRAY)
        ))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BORDER))
    story.append(Paragraph(
        "PsychoGuard AI  ·  Protecting minds from digital manipulation  ·  For research & educational use",
        _sty("foot", fontSize=7, textColor=C_GRAY, alignment=TA_CENTER)
    ))

    doc.build(story)
    buf.seek(0)
    return buf