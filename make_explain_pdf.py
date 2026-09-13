# -*- coding: utf-8 -*-
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

FONT = Path(r"C:\Windows\Fonts\LeelawUI.ttf")
FONT_B = Path(r"C:\Windows\Fonts\LeelaUIb.ttf")
OUT = Path(__file__).resolve().parent / "workshop-lecture10-guide.pdf"

pdfmetrics.registerFont(TTFont("Lee", str(FONT)))
pdfmetrics.registerFont(TTFont("LeeBold", str(FONT_B)))

INK = HexColor("#3d2b1f")
HEAD = HexColor("#6b4f3a")
ASK = HexColor("#8b4513")


def styles():
    return {
        "h1": ParagraphStyle("h1", fontName="LeeBold", fontSize=18, leading=26, textColor=HEAD, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName="LeeBold", fontSize=14, leading=22, textColor=HEAD, spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("body", fontName="Lee", fontSize=12, leading=20, textColor=INK, spaceAfter=6),
        "q": ParagraphStyle("q", fontName="LeeBold", fontSize=12, leading=20, textColor=ASK, spaceBefore=4, spaceAfter=2),
        "a": ParagraphStyle("a", fontName="Lee", fontSize=12, leading=20, textColor=INK, leftIndent=12, spaceAfter=8),
    }


def build():
    s = styles()
    story = []
    add = story.append

    add(Paragraph("อธิบายโค้ด Workshop Lecture 10", s["h1"]))
    add(Paragraph("Morphological Processing · โฟลเดอร์ workshop-morph · อ่านก่อนเจอจารย์", s["body"]))
    add(Paragraph("จารย์ข้อนี้จะไล่ตาม 6 ข้อในสไลด์ ไม่ค่อยไล่หน้าเว็บหรือสีน้ำตาล จำไฟล์หลัก 3 ตัวนี้พอ", s["body"]))
    add(Paragraph("• ai/dataset.py = ข้อ 1–2 หาโจทย์และ Dataset", s["body"]))
    add(Paragraph("• ai/process.py = ข้อ 3–4 อัลกอริทึมแยกวัตถุ + Morphology", s["body"]))
    add(Paragraph("• ai/evaluate.py = ข้อ 5 วัด Confusion Matrix และ ROC", s["body"]))
    add(Paragraph("กดวัดผลบนเว็บแค่เรียก evaluate() ผ่าน /api/evaluate", s["body"]))

    add(Paragraph("1) Dataset มาจากไหน — dataset.py", s["h2"]))
    add(Paragraph("โหลดชุดเหรียญบนทรายจาก Zenodo แล้วเก็บ 50 คู่ ย่อเหลือ 256×256 พิกเซล รูปอยู่ dataset/images  mask อยู่ dataset/masks", s["body"]))
    add(Paragraph("จารย์ถาม: Dataset อะไร Ground truth ใครทำ", s["q"]))
    add(Paragraph("ตอบ: ชื่อชุด Labeled Images of Sand and Coins v2 (Buscombe 2022, CC BY 4.0) ลิงก์ https://doi.org/10.5281/zenodo.6232246 mask เขาทำมาให้แล้ว ขาว = เหรียญ ดำ = ทราย เราใช้ 50 คู่", s["a"]))
    add(Paragraph("จารย์ถาม: เอารูปมาทำไม ต้องมี 50 ทำไม", s["q"]))
    add(Paragraph("ตอบ: จะวัด Confusion Matrix กับ ROC ต้องมีรูปจริงคู่กับคำตอบ ถ้าวัดรูปเดียวตัวเลขเด้งง่าย ไม่น่าเชื่อ จารย์เลยให้อย่างน้อย 50", s["a"]))

    add(Paragraph("2) อัลกอริทึมแยกวัตถุ — to_binary() ใน process.py", s["h2"]))
    add(Paragraph("ลำดับสั้น ๆ: เปลี่ยนเป็นเทา → Otsu ตัดขาวดำ → ถ้าขาวเยอะไปให้กลับสี เพราะเหรียญควรเป็นกลุ่มเล็ก", s["body"]))
    add(Paragraph("จารย์ถาม: แยกยังไง Positive คืออะไร", s["q"]))
    add(Paragraph("ตอบ: ใช้ Otsu หาค่าตัดอัตโนมัติ เหรียญ = Positive คลาส ทราย = Negative คลาส", s["a"]))
    add(Paragraph("จารย์ถาม: ทำไมต้องกลับสี", s["q"]))
    add(Paragraph("ตอบ: ทรายสว่างกว่าเหรียญบ่อย Otsu จะได้ขาวทั้งพื้น ถ้าไม่กลับ Positive จะกลายเป็นทราย", s["a"]))

    add(Paragraph("3) Morphology — apply_morph() ใน process.py", s["h2"]))
    add(Paragraph("Structuring Element เป็นสี่เหลี่ยม 5×5 ตามเลคเชอร์ที่บอกว่ารูปทรงและขนาดของ SE คุมว่าจะยืดหรือยุบทิศไหน แค่ไหน ตอนวัดผลเรียกแค่ Opening", s["body"]))
    add(Paragraph("จารย์ถาม: Opening คืออะไร ทำไมใช้ตัวนี้", s["q"]))
    add(Paragraph("ตอบ: ทำ Erosion แล้วตามด้วย Dilation ลบจุดเล็กที่ไม่ใช่เหรียญ Accuracy จากประมาณ 0.69 ขึ้นเป็น 0.84", s["a"]))
    add(Paragraph("จารย์ถาม: Dilation / Erosion / Closing ต่างกันยังไง", s["q"]))
    add(Paragraph("ตอบ: Dilation ทำให้วัตถุพอง · Erosion ทำให้วัตถุยุบ · Closing คือ Dilation แล้ว Erosion ใช้อุดรูเล็กในวัตถุ", s["a"]))

    add(Paragraph("4) เทียบกับ Ground truth — scores() และลูปใน evaluate()", s["h2"]))
    add(Paragraph("ไล่ทุกไฟล์ใน dataset/images อ่าน mask ชื่อเดียวกัน จากนั้นตัดด้วย to_binary แล้ว Opening แล้วเทียบทีละพิกเซลกับ mask", s["body"]))
    add(Paragraph("จารย์ถาม: เลขแสนล้านนับจากไหน", s["q"]))
    add(Paragraph("ตอบ: นับพิกเซลทั้ง 50 รูป รูปละ 256×256 รวม 3,276,800 จุด ไม่ได้นับเหรียญ และไม่ได้นับทีละภาพ", s["a"]))
    add(Paragraph("จารย์ถาม: TP FP TN FN คืออะไร", s["q"]))
    add(Paragraph("ตอบ: TP จุดเหรียญที่ทายถูก · FP ทรายแต่โปรแกรมว่าเหรียญ · FN เหรียญแต่ตัดไม่เจอ · TN ทรายที่ทายถูก", s["a"]))
    add(Paragraph("จารย์ถาม: คือเอารูปจริงมาตัดเหรียญแล้วเทียบกับ ground truth ใช่ไหม", s["q"]))
    add(Paragraph("ตอบ: ใช่ รูปถ่าย → โปรแกรมตัดว่าจุดไหนเป็นเหรียญ → เทียบกับ mask ที่เป็นคำตอบ", s["a"]))

    add(Paragraph("5) สูตรใน metrics() ตรงสไลด์", s["h2"]))
    add(Paragraph("Accuracy = (TP + TN) / ทั้งหมด", s["body"]))
    add(Paragraph("Precision = TP / (TP + FP)", s["body"]))
    add(Paragraph("Recall = TPR = TP / P = TP / (TP + FN)", s["body"]))
    add(Paragraph("Fall-out = FPR = FP / N = FP / (FP + TN)", s["body"]))
    add(Paragraph("FNR = FN / P", s["body"]))
    add(Paragraph("TNR = TN / N", s["body"]))
    add(Paragraph("F1 = 2 · Precision · Recall / (Precision + Recall)", s["body"]))
    add(Paragraph("สูตรฝั่งขวาบางช่องในสไลด์สลับตัวหาร ใช้สูตรซ้ายของสไลด์กับนิยามมาตรฐานจะตรงกับ Fall-out และ Recall ในแผ่นเดียวกัน", s["body"]))
    add(Paragraph("จารย์ถาม: Accuracy 0.84 แปลว่าดีไหม", s["q"]))
    add(Paragraph("ตอบ: สูงเพราะ TN ทรายเยอะ Precision ประมาณ 0.27 ยังพลาดทรายปนเหรียญ เป็น baseline สำหรับงานคลาส ไม่ใช่โมเดลพร้อมใช้จริง", s["a"]))
    add(Paragraph("แทนเลข Opening ล่าสุด", s["body"]))
    add(Paragraph("TP 159,031 · FN 102,018 · FP 421,497 · TN 2,594,254", s["body"]))
    add(Paragraph("Accuracy = (159,031 + 2,594,254) / (261,049 + 3,015,751) = 0.8402", s["body"]))
    add(Paragraph("Precision = 159,031 / (159,031 + 421,497) = 0.2739", s["body"]))
    add(Paragraph("Recall = 159,031 / 261,049 = 0.6092", s["body"]))
    add(Paragraph("Fall-out = 421,497 / 3,015,751 = 0.1398", s["body"]))
    add(Paragraph("F1 = 0.3779", s["body"]))

    add(Paragraph("6) ROC — make_roc()", s["h2"]))
    add(Paragraph("ไล่ threshold 0 ถึง 255 ทีละ 8 แต่ละค่าคำนวณ TPR กับ FPR แล้วลากกราฟ เส้นประคือเดาสุ่ม โมเดลดีจะชิดมุมซ้ายบน", s["body"]))
    add(Paragraph("จารย์ถาม: แกนอะไร", s["q"]))
    add(Paragraph("ตอบ: แกน X = FPR (Fall-out) · แกน Y = TPR (Recall)", s["a"]))
    add(Paragraph("จารย์ถาม: ทำไมเส้นไม่สวย", s["q"]))
    add(Paragraph("ตอบ: ช่วงกลางอยู่ใต้เส้นสุ่ม แยกเหรียญกับทรายยังไม่คม ตรงกับที่ Precision ต่ำ", s["a"]))

    add(Paragraph("อย่าไปลึกถ้าจารย์ไม่ถาม", s["h2"]))
    add(Paragraph("หน้าเว็บ Flask พอร์ต 5001 เป็นแค่ตัวโชว์ผล ข้อ 6 ไม่ได้บังคับว่าต้องมีเว็บ", s["body"]))
    add(Paragraph("ฟังก์ชัน process() กับ /api/process เป็นของตอนยังมีช่องอัปโหลด หน้าเว็บตอนนี้ไม่เรียก", s["body"]))
    add(Paragraph("user_id ใน process เป็นตามโครงงานเก่า ไม่ได้ใช้ตอนวัดผล", s["body"]))

    add(Paragraph("ชี้ไฟล์ให้จารย์ยังไง", s["h2"]))
    add(Paragraph("ชี้ to_binary แล้วชี้ apply_morph แบบ opening แล้วชี้ scores แล้วชี้ metrics กับ make_roc จบข้อ 3 ถึงข้อ 5", s["body"]))

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="อธิบายโค้ด Workshop Lecture 10",
    )
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()
