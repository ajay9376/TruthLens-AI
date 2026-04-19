"""
TruthLens AI — Forensic PDF Report Generator
=============================================
Generates a professional forensic report for deepfake analysis results.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import os

# ─── Colors ───
PURPLE     = HexColor('#7c3aed')
CYAN       = HexColor('#06b6d4')
DARK       = HexColor('#0d0b1a')
GREEN      = HexColor('#10b981')
RED        = HexColor('#ef4444')
YELLOW     = HexColor('#f59e0b')
LIGHT_GRAY = HexColor('#f8f9fa')
MID_GRAY   = HexColor('#6b7280')
WHITE      = white
BLACK      = black

def get_verdict_color(verdict: str):
    if verdict == "REAL":
        return GREEN
    elif verdict == "DEEPFAKE":
        return RED
    else:
        return YELLOW

def get_score_color(score: float):
    if score >= 60:
        return GREEN
    elif score >= 40:
        return YELLOW
    else:
        return RED

def generate_report(results: dict, video_name: str, output_path: str = None):
    """
    Generate a professional forensic PDF report.
    
    Args:
        results: dict with keys: verdict, final_score, syncnet_score,
                 texture_score, blink_score, lip_score
        video_name: name of analyzed video file
        output_path: where to save the PDF
    
    Returns:
        path to generated PDF
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"TruthLens_Report_{timestamp}.pdf"

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    story  = []

    # ─── Header ───
    header_style = ParagraphStyle(
        'Header',
        fontSize=28,
        fontName='Helvetica-Bold',
        textColor=WHITE,
        alignment=TA_CENTER,
        spaceAfter=4,
    )

    sub_header_style = ParagraphStyle(
        'SubHeader',
        fontSize=12,
        fontName='Helvetica',
        textColor=HexColor('#c4b5fd'),
        alignment=TA_CENTER,
    )

    header_table = Table(
        [[Paragraph("🔍 TruthLens AI", header_style)],
         [Paragraph("Forensic Deepfake Analysis Report", sub_header_style)]],
        colWidths=[17*cm]
    )
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), DARK),
        ('ROUNDEDCORNERS', [10]),
        ('TOPPADDING', (0,0), (-1,-1), 20),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
        ('LEFTPADDING', (0,0), (-1,-1), 20),
        ('RIGHTPADDING', (0,0), (-1,-1), 20),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.5*cm))

    # ─── Meta Info ───
    meta_style = ParagraphStyle(
        'Meta',
        fontSize=10,
        fontName='Helvetica',
        textColor=MID_GRAY,
        alignment=TA_LEFT,
    )

    now = datetime.now()
    meta_data = [
        ['📅 Date:', now.strftime("%B %d, %Y")],
        ['⏰ Time:', now.strftime("%H:%M:%S")],
        ['🎬 Video File:', video_name],
        ['🤖 System:', 'TruthLens AI v2.0'],
        ['📊 Signals Used:', 'SyncNet · Face Texture · Blink Pattern · Lip Reader'],
    ]

    meta_table = Table(meta_data, colWidths=[4*cm, 13*cm])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('TEXTCOLOR', (0,0), (0,-1), PURPLE),
        ('TEXTCOLOR', (1,0), (1,-1), BLACK),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [LIGHT_GRAY, WHITE]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#e5e7eb')),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5*cm))

    # ─── Verdict ───
    verdict       = results.get('verdict', 'UNKNOWN')
    final_score   = results.get('final_score', 0)
    verdict_color = get_verdict_color(verdict)

    verdict_icons = {'REAL': '✅', 'DEEPFAKE': '❌', 'SUSPICIOUS': '⚠️'}
    verdict_icon  = verdict_icons.get(verdict, '❓')

    verdict_style = ParagraphStyle(
        'Verdict',
        fontSize=32,
        fontName='Helvetica-Bold',
        textColor=WHITE,
        alignment=TA_CENTER,
    )

    score_style = ParagraphStyle(
        'Score',
        fontSize=14,
        fontName='Helvetica',
        textColor=HexColor('#e0e7ff'),
        alignment=TA_CENTER,
    )

    verdict_table = Table(
        [[Paragraph(f"{verdict_icon}  {verdict}", verdict_style)],
         [Paragraph(f"Combined Confidence Score: {final_score:.1f} / 100", score_style)]],
        colWidths=[17*cm]
    )
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), verdict_color),
        ('TOPPADDING', (0,0), (-1,-1), 20),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
        ('LEFTPADDING', (0,0), (-1,-1), 20),
        ('RIGHTPADDING', (0,0), (-1,-1), 20),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 0.5*cm))

    # ─── Signal Breakdown ───
    section_style = ParagraphStyle(
        'Section',
        fontSize=14,
        fontName='Helvetica-Bold',
        textColor=PURPLE,
        spaceBefore=10,
        spaceAfter=6,
    )

    story.append(Paragraph("📊 Signal Breakdown", section_style))

    signals = [
        ('🎵 SyncNet Lip-Sync', results.get('syncnet_score', 50)),
        ('🎨 Face Texture Analysis', results.get('texture_score', 50)),
        ('👁️ Blink Pattern Detection', results.get('blink_score', 50)),
        ('👄 Lip Reader', results.get('lip_score', 50)),
    ]

    signal_data = [['Signal', 'Score', 'Status', 'Bar']]

    for name, score in signals:
        color  = get_score_color(score)
        status = '✅ REAL' if score >= 60 else '⚠️ SUSPICIOUS' if score >= 40 else '❌ FAKE'
        bar    = '█' * int(score / 10) + '░' * (10 - int(score / 10))
        signal_data.append([name, f"{score:.1f}/100", status, bar])

    signal_table = Table(
        signal_data,
        colWidths=[6*cm, 3*cm, 3.5*cm, 4.5*cm]
    )

    signal_table.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0,0), (-1,0), PURPLE),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        # Body
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT_GRAY, WHITE]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#e5e7eb')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (1,0), (2,-1), 'CENTER'),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
    ]))
    story.append(signal_table)
    story.append(Spacer(1, 0.5*cm))

    # ─── Interpretation ───
    story.append(Paragraph("📝 Analysis Interpretation", section_style))

    interp_style = ParagraphStyle(
        'Interp',
        fontSize=11,
        fontName='Helvetica',
        textColor=BLACK,
        leading=18,
        spaceAfter=6,
    )

    if verdict == "REAL":
        interpretation = """
        <b>This video appears to be GENUINE.</b> All analyzed signals indicate natural 
        human behavior consistent with authentic video content. The lip movements align 
        naturally with the audio track, the facial texture shows realistic skin patterns, 
        the eye blink rate falls within normal human ranges, and the speech articulation 
        appears natural and unforced.
        <br/><br/>
        <b>Recommendation:</b> This video can be considered authentic based on our 
        4-signal analysis. No deepfake manipulation was detected.
        """
    elif verdict == "DEEPFAKE":
        interpretation = """
        <b>This video shows strong indicators of DEEPFAKE manipulation.</b> Multiple 
        signals have flagged anomalies consistent with AI-generated or manipulated content. 
        The analysis detected unnatural patterns in facial texture, abnormal eye blink 
        behavior, and/or inconsistencies between lip movements and audio.
        <br/><br/>
        <b>Recommendation:</b> Treat this video with extreme caution. Do not use as 
        evidence without further verification from a certified forensic expert.
        """
    else:
        interpretation = """
        <b>This video returned INCONCLUSIVE results.</b> Some signals indicate authentic 
        content while others show potential anomalies. This could be due to video 
        compression artifacts, unusual lighting conditions, or subtle manipulation.
        <br/><br/>
        <b>Recommendation:</b> Manual review by a forensic expert is recommended before 
        making any decisions based on this video.
        """

    story.append(Paragraph(interpretation, interp_style))
    story.append(Spacer(1, 0.3*cm))

    # ─── Disclaimer ───
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#e5e7eb')))
    story.append(Spacer(1, 0.3*cm))

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        fontSize=8,
        fontName='Helvetica',
        textColor=MID_GRAY,
        alignment=TA_CENTER,
        leading=12,
    )

    story.append(Paragraph(
        "⚠️ DISCLAIMER: This report is generated by an AI system and should not be used as sole "
        "evidence in legal proceedings. TruthLens AI provides probabilistic analysis only. "
        "Always consult a certified forensic expert for legal matters.",
        disclaimer_style
    ))

    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph(
        "Generated by TruthLens AI v2.0 | github.com/ajay9376/TruthLens-AI",
        disclaimer_style
    ))

    # ─── Build PDF ───
    doc.build(story)
    print(f"✅ Report saved: {output_path}")
    return output_path


# ─── CLI Test ───
if __name__ == "__main__":
    test_results = {
        'verdict':       'REAL',
        'final_score':   69.8,
        'syncnet_score': 34.0,
        'texture_score': 45.0,
        'blink_score':   85.0,
        'lip_score':     100.0,
    }
    path = generate_report(test_results, "test_video.mp4")
    print(f"📄 Report generated: {path}")