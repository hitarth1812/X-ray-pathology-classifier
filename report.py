"""
PDF report generator for Chest X-Ray pathology results.

Returns an in-memory ``io.BytesIO`` PDF — nothing is written to disk.
"""

import io
from datetime import datetime
from typing import Dict, Optional

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


_RED = colors.HexColor("#e4572e")
_GREY = colors.HexColor("#9ca3af")
_DARK = colors.HexColor("#10243f")
_LIGHT_BG = colors.HexColor("#f8fbff")
_BORDER = colors.HexColor("#d6e3f3")


def generate_report(
    image_path: str,
    confidences: Dict[str, float],
    threshold: float,
    patient_label: str = "Anonymous",
) -> io.BytesIO:
    """
    Build a single-page PDF pathology report and return it as BytesIO.

    Parameters
    ----------
    image_path:
        Absolute path to the uploaded X-ray image.
    confidences:
        Dict mapping class label → sigmoid probability (0–1).
    threshold:
        Decision threshold used for positive/negative classification.
    patient_label:
        Placeholder identifier shown in the report header.

    Returns
    -------
    io.BytesIO positioned at offset 0, ready for download.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Header ───────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=_DARK,
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=10,
        textColor=_GREY,
        spaceAfter=6,
    )
    story.append(Paragraph("Chest X-Ray Pathology Report", title_style))
    story.append(
        Paragraph(
            f"Patient: {patient_label} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Threshold: {threshold:.2f}",
            sub_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=_BORDER, spaceAfter=10))

    # ── Thumbnail ─────────────────────────────────────────────────────────────
    try:
        pil_img = Image.open(image_path).convert("RGB")
        pil_img.thumbnail((200, 200))
        thumb_buf = io.BytesIO()
        pil_img.save(thumb_buf, format="PNG")
        thumb_buf.seek(0)
        thumb_w, thumb_h = pil_img.size
        rl_img = RLImage(thumb_buf, width=thumb_w, height=thumb_h)
        story.append(rl_img)
        story.append(Spacer(1, 8))
    except Exception:
        story.append(Paragraph("[Image could not be embedded]", styles["Normal"]))

    # ── Results table ─────────────────────────────────────────────────────────
    sorted_items = sorted(confidences.items(), key=lambda x: x[1], reverse=True)

    header_style = ParagraphStyle(
        "TH",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.white,
        fontName="Helvetica-Bold",
    )
    cell_style = ParagraphStyle("TD", parent=styles["Normal"], fontSize=9, textColor=_DARK)

    table_data = [
        [
            Paragraph("Finding", header_style),
            Paragraph("Confidence", header_style),
            Paragraph("Status", header_style),
        ]
    ]

    NO_FINDING = "No Finding"
    other_positive_labels = {
        lbl for lbl, sc in confidences.items() if lbl != NO_FINDING and sc >= threshold
    }
    suppress_no_finding = len(other_positive_labels) > 0

    for label, score in sorted_items:
        is_suppressed = label == NO_FINDING and suppress_no_finding
        is_positive = (score >= threshold) and not is_suppressed
        bullet = "●" if is_positive else "○"
        bullet_color = _RED if is_positive else _GREY
        status_text = "Positive" if is_positive else ("Suppressed" if is_suppressed else "Negative")

        table_data.append(
            [
                Paragraph(label, cell_style),
                Paragraph(f"{score * 100:.2f}%", cell_style),
                Paragraph(
                    f"<font color='{bullet_color.hexval()}'>{bullet}</font> {status_text}",
                    cell_style,
                ),
            ]
        )

    col_widths = [90 * mm, 40 * mm, 40 * mm]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _DARK),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT_BG, colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.4, _BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 10))

    # ── Footer disclaimer ─────────────────────────────────────────────────────
    disclaimer_style = ParagraphStyle(
        "Disc",
        parent=styles["Normal"],
        fontSize=8,
        textColor=_GREY,
        fontName="Helvetica-Oblique",
    )
    story.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER, spaceAfter=6))
    story.append(
        Paragraph(
            "For research/simulation only. Not for clinical use. "
            "This report was generated automatically by an AI model and must not be "
            "used for medical decision-making.",
            disclaimer_style,
        )
    )

    doc.build(story)
    buf.seek(0)
    return buf
