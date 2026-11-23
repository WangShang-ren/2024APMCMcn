"""按官方 Word 模板生成论文。

流程：
1. 从 main.tex 生成精简的 word_source.tex（去掉目录、标题页、部分图；
   重写参考文献与附录）；
2. pandoc -> docx（使用官方模板作为 reference-doc，继承页面与样式）；
3. python-docx 后处理：插入官方“选题/赛项名称/参赛编号”表格、套用宋体小四、
   标题/图表编号、页脚页码、图片缩放等；
4. LibreOffice -> PDF。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
SOLUTION = ROOT.parent
TEMPLATE = Path(
    "/home/shaun/static/work/modeling/APMCM/"
    "2024年第十四届APMCM亚太地区大学生数学建模竞赛/"
    "参赛纪律、论文提交方式、承诺书及论文模板/"
    "2024年第十四届APMCM亚太地区大学生数学建模竞赛（中文赛项）论文模板.docx"
)
WORD_SRC = ROOT / "word_source.tex"
RAW_DOCX = ROOT / "_official_raw.docx"
OUT_DOCX = ROOT / "Bapmcm241*****.docx"
OUT_PDF = ROOT / "Bapmcm241*****.pdf"
def _find_tool(local: str, name: str) -> str:
    p = Path(local)
    if p.exists():
        return str(p)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"未找到 {name}。请安装 pandoc / LibreOffice，或修改 build_official.py 中的路径。"
    )


_HOME = Path.home()
PANDOC = Path(_find_tool(
    str(_HOME / ".local/pandoc/pandoc-3.14/bin/pandoc"), "pandoc"
))
SOFFICE = Path(_find_tool(
    str(_HOME / ".local/libreoffice/squashfs-root/opt/libreoffice26.2/program/soffice"),
    "soffice",
))

# 保留的图（按 label）
KEEP_FIGS = {
    "fig:structcorr", "fig:structrowsum", "fig:p1", "fig:p1inference", "fig:p2profiles",
    "fig:p2weights", "fig:p2topsis", "fig:p3models", "fig:p3five",
    "fig:p4hist", "fig:genppc", "fig:genpred", "fig:causaldag",
    "fig:causalcorr",
}



def number_refs(body: str) -> str:
    """把 \\ref{tab:...}/\\ref{fig:...} 替换为按出现顺序的静态编号（Word 用）。"""
    labels = {}
    for env, prefix in (("table", "tab"), ("figure", "fig")):
        pos = 0
        n = 0
        endtag = f"\\end{{{env}}}"
        while True:
            a = body.find(f"\\begin{{{env}}}", pos)
            if a < 0:
                break
            b = body.find(endtag, a)
            if b < 0:
                break
            b += len(endtag)
            block = body[a:b]
            n += 1
            m = re.search(rf"\\label{{({prefix}:[^}}]+)}}", block)
            if m:
                labels[m.group(1)] = n
            pos = b
    for lab, num in labels.items():
        body = body.replace(f"\\ref{{{lab}}}", str(num))
    return body

def make_word_source() -> None:
    tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    body = tex.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]

    # 1) 去掉目录与标题页
    body = body.replace("\\maketitle", "")
    body = body.replace("\\tableofcontents", "")
    body = re.sub(r"\n\s*\\newpage\s*\n", "\n", body)

    # 2) 摘要改为普通“摘要”标题
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    if m:
        abstract = m.group(1)
        abstract = abstract.replace("\\vspace{0.5em}", "")
        body = body[:m.start()] + "\\section*{摘要}\n" + abstract + body[m.end():]

    # 3) 删除不进入 Word 版的图；图 3（问题 1）改为左右合并图
    p1_block = r"""\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{../figures/fig_p1_combined.png}
\caption{20 个指标的相关性几乎相同，排名不可区分}
\label{fig:p1}
\end{figure}"""
    fig_pattern = re.compile(r"\\begin\{figure\}.*?\\end\{figure\}", re.S)
    def process_fig(match: re.Match) -> str:
        block = match.group(0)
        if "\\label{fig:p1}" in block:
            return p1_block
        labels = re.findall(r"\\label\{(fig:[^}]+)\}", block)
        return block if any(l in KEEP_FIGS for l in labels) else ""
    body = fig_pattern.sub(process_fig, body)

    # 5) 参考文献改为普通列表（pandoc 对 thebibliography 支持不好）
    bm = re.search(r"\\begin\{thebibliography\}\{99\}(.*?)\\end\{thebibliography\}", body, re.S)
    if bm:
        items = re.findall(r"\\bibitem\{[^}]*\}(.*?)(?=\\bibitem|$)", bm.group(1), re.S)
        lines = ["\\section*{参考文献}", ""]
        for i, it in enumerate(items, 1):
            text = " ".join(it.split())
            lines.append(f"\\noindent\\textbf{{[{i}]}}\\ {text}")
            lines.append("")
        body = body[:bm.start()] + "\n".join(lines) + body[bm.end():]

    # 6) 附录改为不编号标题
    body = body.replace("\\appendix", "")
    body = body.replace("\\section{复现说明}", "\\section*{附录 A\\quad 复现说明}")
    body = body.replace("\\section{主要文件}", "\\section*{附录 B\\quad 主要文件}")
    # Word 版不保留“主要文件”附录，避免最后一页几乎空白
    body = body.split("\\section*{附录 B")[0]

    # 7) 展开表格 \input
    def read_table(m: re.Match) -> str:
        return (ROOT / (m.group(1) + ".tex")).read_text(encoding="utf-8")
    body = re.sub(r"\\input\{(tables/[^}]+)\}", read_table, body)

    # 8) 去掉 \label/\ref 的残留（pandoc 对 section 引用会解析，图引用未使用）
    body = number_refs(body)

    src = r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\title{基于总量与等权结构的洪水灾害数据分析与概率预测}
\author{}
\date{}
\begin{document}
\maketitle
""" + body + "\n\\end{document}\n"
    WORD_SRC.write_text(src, encoding="utf-8")
    print(f"word_source.tex: {len(src)} chars")


def run_pandoc() -> None:
    cmd = [
        str(PANDOC), str(WORD_SRC),
        "-o", str(RAW_DOCX),
        "--resource-path", str(ROOT),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    print("pandoc docx:", RAW_DOCX, RAW_DOCX.stat().st_size)


# ---------------- docx 后处理 ----------------

def set_run_font(run, size=12, east="宋体", ascii_="Times New Roman", bold=None):
    run.font.name = ascii_
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), east)
    rfonts.set(qn("w:ascii"), ascii_)
    rfonts.set(qn("w:hAnsi"), ascii_)
    rfonts.set(qn("w:hint"), "eastAsia")


def style_paragraph(p, kind="body"):
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    if kind == "body":
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.first_line_indent = Pt(24)  # 小四 2 字符
        for r in p.runs:
            set_run_font(r, 12)
    elif kind == "list":
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.first_line_indent = Pt(0)
        pf.left_indent = Cm(0.74)
        for r in p.runs:
            set_run_font(r, 12)
    elif kind == "h1":
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(12)
        pf.space_after = Pt(6)
        pf.keep_with_next = True
        for r in p.runs:
            set_run_font(r, 16, east="黑体", bold=True)
    elif kind == "h2":
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(8)
        pf.space_after = Pt(4)
        pf.keep_with_next = True
        for r in p.runs:
            set_run_font(r, 14, east="黑体", bold=True)
    elif kind == "h3":
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(6)
        pf.space_after = Pt(3)
        pf.keep_with_next = True
        for r in p.runs:
            set_run_font(r, 12, east="黑体", bold=True)
    elif kind == "title":
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(6)
        pf.space_after = Pt(10)
        for r in p.runs:
            set_run_font(r, 18, east="黑体", bold=True)
    elif kind == "abstract_title":
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(6)
        pf.space_after = Pt(4)
        for r in p.runs:
            set_run_font(r, 16, east="黑体", bold=True)
    elif kind == "caption":
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(3)
        pf.space_after = Pt(6)
        for r in p.runs:
            set_run_font(r, 10.5)
    elif kind == "quote":
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.first_line_indent = Pt(0)
        pf.left_indent = Cm(0.74)
        pf.right_indent = Cm(0.74)
        for r in p.runs:
            set_run_font(r, 12)


def add_borders(table) -> None:
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tblPr.append(borders)


def set_cell_text(cell, lines, size=12, east="宋体", bold=False):
    """清空单元格并写入文本（避免 python-docx 的 text setter 产生裸文本节点）。"""
    p = cell.paragraphs[0]
    for child in list(p._p):
        if child.tag != qn("w:pPr"):
            p._p.remove(child)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    for i, line in enumerate(lines):
        run = p.add_run(line)
        set_run_font(run, size, east=east, bold=bold)
        if i < len(lines) - 1:
            run.add_break()


def make_cover_table(doc) -> None:
    table = doc.add_table(rows=2, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    add_borders(table)
    widths = [Cm(3.05), Cm(10.16), Cm(2.79)]

    # 固定表格与列宽（官方模板：1728 / 5760 / 1579 twips）
    tblPr = table._tbl.tblPr
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW"); tblPr.append(tblW)
    tblW.set(qn("w:w"), "9067"); tblW.set(qn("w:type"), "dxa")
    layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    for gc, w in zip(table._tbl.tblGrid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(w.twips)))

    for row in table.rows:
        row.allow_break = False
        for i, cell in enumerate(row.cells):
            cell.width = widths[i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tcPr = cell._tc.get_or_add_tcPr()
            tcW = tcPr.find(qn("w:tcW"))
            if tcW is None:
                tcW = OxmlElement("w:tcW"); tcPr.append(tcW)
            tcW.set(qn("w:w"), str(int(widths[i].twips)))
            tcW.set(qn("w:type"), "dxa")

    merged = table.cell(0, 1).merge(table.cell(1, 1))
    set_cell_text(table.cell(0, 0), ["选题"], 12)
    set_cell_text(merged,
                  ["2024年第十四届APMCM", "亚太地区大学生数学建模竞赛（中文赛项）"],
                  14, east="黑体", bold=True)
    set_cell_text(table.cell(0, 2), ["参赛编号"], 12)
    set_cell_text(table.cell(1, 0), ["B"], 14, bold=True)
    set_cell_text(table.cell(1, 2), ["apmcm241*****"], 9)
    table.rows[0].height = Cm(1.4)
    table.rows[1].height = Cm(1.2)

    body = doc.element.body
    body.insert(0, table._tbl)
    spacer = OxmlElement("w:p")
    body.insert(1, spacer)


def clean_direct_text(doc) -> None:
    """pandoc 会在 w:p 下写入裸文本节点，Word 会与 w:r 重复显示；这里清除。"""
    from lxml.etree import _Element
    for p in doc.element.body.iter(qn("w:p")):
        _Element.text.__set__(p, None)
        for child in p:
            child.tail = None


def add_page_number_footer(doc) -> None:
    sec = doc.sections[0]
    footer = sec.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run("第 "), 9)
    run = p.add_run()
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = " PAGE "
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1); run._r.append(instr); run._r.append(fld2)
    set_run_font(run, 9)
    set_run_font(p.add_run(" 页"), 9)


def postprocess() -> None:
    doc = Document(RAW_DOCX)
    clean_direct_text(doc)

    # 页面：A4，2.5cm 页边距
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    for attr in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(sec, attr, Cm(2.5))

    # 基础样式
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")

    # 逐段处理
    h1 = h2 = h3 = 0
    fig_no = tab_no = 0
    paragraphs = list(doc.paragraphs)
    for p in paragraphs:
        st = p.style.name if p.style is not None else ""
        text = p.text.strip()

        if st == "Title":
            style_paragraph(p, "title")
        elif st == "Author":
            p._element.getparent().remove(p._element)
        elif st == "Abstract Title":
            for r in p.runs:
                r.text = ""
            if p.runs:
                p.runs[0].text = "摘要"
            else:
                p.add_run("摘要")
            style_paragraph(p, "abstract_title")
        elif st == "Abstract":
            style_paragraph(p, "body")
            if text.startswith("关键词"):
                p.paragraph_format.first_line_indent = Pt(0)
        elif st == "Heading 1":
            if text.startswith("摘要"):
                style_paragraph(p, "abstract_title")
            elif text.startswith("参考文献") or text.startswith("附录"):
                if text.startswith("附录 A") and "复现" in text:
                    for r in p.runs:
                        r.text = ""
                    if p.runs:
                        p.runs[0].text = "附录 A　复现说明"
                style_paragraph(p, "h1")
            else:
                h1 += 1; h2 = h3 = 0
                if p.runs:
                    p.runs[0].text = f"{h1} " + p.runs[0].text
                style_paragraph(p, "h1")
        elif st == "Heading 2":
            h2 += 1; h3 = 0
            if p.runs:
                p.runs[0].text = f"{h1}.{h2} " + p.runs[0].text
            style_paragraph(p, "h2")
        elif st == "Heading 3":
            h3 += 1
            if p.runs:
                p.runs[0].text = f"{h1}.{h2}.{h3} " + p.runs[0].text
            style_paragraph(p, "h3")
        elif st == "Image Caption":
            fig_no += 1
            if p.runs:
                p.runs[0].text = f"图 {fig_no}  " + p.runs[0].text
            style_paragraph(p, "caption")
        elif st == "Table Caption":
            tab_no += 1
            if p.runs:
                p.runs[0].text = f"表 {tab_no}  " + p.runs[0].text
            style_paragraph(p, "caption")
            p.paragraph_format.keep_with_next = True
        elif st == "Captioned Figure":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(0)
        elif text.startswith("[") and len(text) > 3 and text[1].isdigit():
            p.paragraph_format.first_line_indent = Cm(-0.74)
            p.paragraph_format.left_indent = Cm(0.74)
            p.paragraph_format.space_after = Pt(0)
            for r in p.runs:
                set_run_font(r, 10)
        elif st == "Block Text":
            style_paragraph(p, "quote")
        elif st == "Source Code":
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.left_indent = Cm(0.5)
            for r in p.runs:
                set_run_font(r, 9, ascii_="Consolas")
        elif st in ("First Paragraph", "Body Text", "Compact"):
            style_paragraph(p, "body")
            # 列表项：pandoc 用 numPr 表示，取消首行缩进并左缩进
            pPr = p._p.pPr
            if pPr is not None and pPr.numPr is not None:
                style_paragraph(p, "list")
        else:
            style_paragraph(p, "body")

    # 关键词后分页
    for p in doc.paragraphs:
        if p.text.strip().startswith("关键词"):
            p.add_run().add_break(WD_BREAK.PAGE)
            break

    # 图片缩放（最大宽度 15.5cm）
    for shape in doc.inline_shapes:
        w, h = shape.width, shape.height
        if w and w > Cm(15.5):
            ratio = Cm(15.5) / w
            shape.width = Cm(15.5)
            shape.height = int(h * ratio)

    # 表格格式
    for ti, table in enumerate(doc.tables):
        if ti == 0:
            continue
        add_borders(table)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = table._tbl.tblPr
        tblW = OxmlElement("w:tblW")
        tblW.set(qn("w:w"), "5000")
        tblW.set(qn("w:type"), "pct")
        tblPr.append(tblW)
        if len(table.rows) <= 12:
            for row in table.rows[:-1]:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        p.paragraph_format.keep_with_next = True
        for ri, row in enumerate(table.rows):
            row.allow_break = False
            if ri == 0:
                trPr = row._tr.get_or_add_trPr()
                trPr.append(OxmlElement("w:tblHeader"))
            for cell in row.cells:
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                for p in cell.paragraphs:
                    p.paragraph_format.first_line_indent = Pt(0)
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:
                        set_run_font(r, 9, bold=(ri == 0))

    # 封面表格插到最前面
    make_cover_table(doc)

    # 页脚页码
    add_page_number_footer(doc)

    doc.save(OUT_DOCX)
    print("docx:", OUT_DOCX, OUT_DOCX.stat().st_size)


def to_pdf() -> None:
    cmd = [
        str(SOFFICE), "--headless",
        "-env:UserInstallation=file:///tmp/opencode/lo_profile2",
        "--convert-to", "pdf", str(OUT_DOCX), "--outdir", str(ROOT),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    print("pdf:", OUT_PDF, OUT_PDF.stat().st_size if OUT_PDF.exists() else "missing")


if __name__ == "__main__":
    make_word_source()
    run_pandoc()
    postprocess()
    to_pdf()
