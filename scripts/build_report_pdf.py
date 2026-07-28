"""Build and validate the polished CrisisForge technical-report PDF.

The report source remains Markdown. This script converts the registered research
text and figures into a publication-style PDF, then performs structural checks.
Visual page rendering is handled separately with Poppler during release QA.
"""

from __future__ import annotations

import argparse
import html
import re
from collections.abc import Iterable
from pathlib import Path

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "reports" / "crisisforge_research_report.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "crisisforge_research_report.pdf"

NAVY = colors.HexColor("#10273F")
NAVY_2 = colors.HexColor("#183E5B")
CYAN = colors.HexColor("#00A9CE")
TEAL = colors.HexColor("#21C4B5")
AMBER = colors.HexColor("#F1B24A")
INK = colors.HexColor("#1C2733")
MUTED = colors.HexColor("#5D6A78")
PALE = colors.HexColor("#EDF3F7")
PALE_CYAN = colors.HexColor("#E8F7FA")
RULE = colors.HexColor("#CED9E2")
WHITE = colors.white


REFERENCES = [
    (
        "Hamilton, J. D. (1989). A New Approach to the Economic Analysis of "
        "Nonstationary Time Series and the Business Cycle. Econometrica, 57(2), 357-384.",
        "https://www.jstor.org/stable/1912559",
    ),
    (
        "Kim, C.-J. (1994). Dynamic Linear Models with Markov-Switching. "
        "Journal of Econometrics, 60(1-2), 1-22.",
        "https://www.sciencedirect.com/science/article/pii/0304407694900361",
    ),
    (
        "Urga, G., and Wang, F. (2024). High-dimensional regime-switching factor models.",
        "https://openaccess.city.ac.uk/id/eprint/32040/",
    ),
    (
        "Barigozzi, M., and Massacci, D. (2025). Large-panel regime-switching factor methods.",
        "https://cris.unibo.it/handle/11585/1000607",
    ),
    (
        "Shen, L., et al. (2023). TimeDiff: Non-autoregressive multi-step probabilistic "
        "forecasting with diffusion models. ICML 2023.",
        "https://proceedings.mlr.press/v202/shen23d.html",
    ),
    (
        "FTS-Diffusion (2024). Financial time-series generation with diffusion models. ICLR 2024.",
        "https://proceedings.iclr.cc/paper_files/paper/2024/hash/"
        "f90fc76b199fe6b0ec2a51aaf72c3277-Abstract-Conference.html",
    ),
    (
        "Chen et al. (2025). Factor-structured diffusion for high-dimensional financial "
        "scenario generation. arXiv preprint 2504.06566.",
        "https://arxiv.org/abs/2504.06566",
    ),
    (
        "Cont, R., et al. (2024). Tail-GAN: Learning to simulate tail risk scenarios. "
        "Management Science.",
        "https://pubsonline.informs.org/doi/10.1287/mnsc.2023.00936",
    ),
    (
        "McNeil, A. J., and Frey, R. (2000). Estimation of tail-related risk measures "
        "for heteroscedastic financial time series. Journal of Empirical Finance, 7(3-4).",
        "https://www.sciencedirect.com/science/article/abs/pii/S0927539800000128",
    ),
    (
        "Esfahani, P. M., and Kuhn, D. (2018). Data-driven distributionally robust "
        "optimization using the Wasserstein metric. Mathematical Programming, 171, 115-166.",
        "https://link.springer.com/article/10.1007/s10107-017-1172-1",
    ),
    (
        "Wilder, B., et al. (2023). Decision-focused learning: Foundations and "
        "applications. Management Science.",
        "https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922",
    ),
    (
        "DoFlow (2026). Time-series intervention modeling with normalizing flows. ICLR 2026.",
        "https://iclr.cc/virtual/2026/poster/10011565",
    ),
    (
        "DiffCATS (2026). Diffusion-based counterfactual generation for time series. OpenReview.",
        "https://openreview.net/forum?id=FwC6CyaHop",
    ),
]


def _font_file(name: str) -> str:
    """Return a macOS font path, falling back to a bundled common path."""
    candidates = {
        "Arial": [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
        "Arial-Bold": [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
        "Arial-Italic": [
            "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        ],
        "Arial-BoldItalic": [
            "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
        ],
        "Georgia": [
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ],
        "Georgia-Bold": [
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ],
    }
    for path in candidates[name]:
        if Path(path).exists():
            return path
    raise FileNotFoundError(f"No usable font found for {name}")


def register_fonts() -> None:
    """Register font families with broad Unicode coverage."""
    for name in (
        "Arial",
        "Arial-Bold",
        "Arial-Italic",
        "Arial-BoldItalic",
        "Georgia",
        "Georgia-Bold",
    ):
        pdfmetrics.registerFont(TTFont(name, _font_file(name)))
    pdfmetrics.registerFontFamily(
        "Arial",
        normal="Arial",
        bold="Arial-Bold",
        italic="Arial-Italic",
        boldItalic="Arial-BoldItalic",
    )
    pdfmetrics.registerFontFamily(
        "Georgia",
        normal="Georgia",
        bold="Georgia-Bold",
    )


def normalize_typography(text: str) -> str:
    """Normalize typography to PDF-safe characters and ASCII hyphens."""
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u202f": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def latex_to_text(text: str) -> str:
    """Convert the small LaTeX subset used by the report to readable text."""
    text = text.replace(
        r"L_j=s_j\sum_{h=1}^{20}X_{j,h}",
        "L_j = s_j x sum(h = 1..20) X_(j,h)",
    )
    text = text.replace(
        r"\operatorname{ES}_{0.95}(L_j^{\mathrm{treated}})"
        r"-\operatorname{ES}_{0.95}(L_j^{\mathrm{reference}})",
        "ES_0.95(L_j, treated) - ES_0.95(L_j, reference)",
    )
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = re.sub(r"\\operatorname\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    substitutions = {
        r"\alpha": "alpha",
        r"\epsilon": "epsilon",
        r"\operatorname": "",
        r"\log": "log",
        r"\exp": "exp",
        r"\expm1": "expm1",
        r"\operatorname{Cov}": "Cov",
        r"\operatorname{ES}": "ES",
        r"\ell": "l",
        r"\infty": "infinity",
        r"\sum": "sum",
        r"\qquad": "   ",
        r"\quad": " ",
        r"\,": " ",
        r"\\": " ",
    }
    for source, target in substitutions.items():
        text = text.replace(source, target)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def inline_markup(text: str) -> str:
    """Translate basic Markdown inline markup into ReportLab paragraph markup."""
    text = normalize_typography(latex_to_text(text))
    tokens: dict[str, str] = {}

    def token(value: str) -> str:
        key = f"@@CF_TOKEN_{len(tokens)}@@"
        tokens[key] = value
        return key

    def link_repl(match: re.Match[str]) -> str:
        label = html.escape(match.group(1))
        url = html.escape(match.group(2), quote=True)
        return token(f'<link href="{url}" color="#007E9B"><u>{label}</u></link>')

    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", link_repl, text)
    text = re.sub(
        r"\*\*([^*]+)\*\*",
        lambda match: token(f"<b>{html.escape(match.group(1))}</b>"),
        text,
    )
    text = re.sub(
        r"`([^`]+)`",
        lambda match: token(f'<font name="Courier">{html.escape(match.group(1))}</font>'),
        text,
    )
    escaped = html.escape(text)
    for key, value in tokens.items():
        escaped = escaped.replace(key, value)
    return escaped


def plain_heading(text: str) -> str:
    """Return a heading without Markdown or ReportLab tags."""
    stripped = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    stripped = stripped.replace("**", "").replace("`", "")
    return normalize_typography(latex_to_text(stripped))


class SectionRule(Flowable):
    """Thin accent rule used below major headings."""

    def __init__(self, width: float = 42 * mm) -> None:
        super().__init__()
        self.width = width
        self.height = 4

    def draw(self) -> None:
        self.canv.setStrokeColor(CYAN)
        self.canv.setLineWidth(2.2)
        self.canv.line(0, 2, self.width, 2)


class CrisisForgeDocTemplate(BaseDocTemplate):
    """Document template with cover, TOC registration, and running section titles."""

    def __init__(self, filename: str, **kwargs: object) -> None:
        super().__init__(filename, **kwargs)
        self.current_section = "Technical Report"

    def beforeDocument(self) -> None:  # noqa: N802
        """Reset pass-specific state before each Table-of-Contents build pass."""
        super().beforeDocument()
        self.current_section = "Technical Report"

    def afterFlowable(self, flowable: Flowable) -> None:  # noqa: N802
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        if style_name not in {"CF-H1", "CF-H2"}:
            return
        level = 0 if style_name == "CF-H1" else 1
        title = flowable.getPlainText()
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        key = f"section-{level}-{slug}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(title, key, level=level, closed=False)
        self.notify("TOCEntry", (level, title, self.page, key))
        if level == 0:
            self.current_section = title


def draw_cover(canvas: object, doc: BaseDocTemplate) -> None:
    """Draw the full-bleed cover background."""
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(NAVY_2)
    canvas.circle(width + 10 * mm, height - 16 * mm, 65 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#0C334B"))
    canvas.circle(-15 * mm, 16 * mm, 58 * mm, fill=1, stroke=0)
    canvas.setStrokeColor(CYAN)
    canvas.setLineWidth(3)
    canvas.line(24 * mm, height - 37 * mm, 74 * mm, height - 37 * mm)
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(1)
    canvas.line(24 * mm, 29 * mm, width - 24 * mm, 29 * mm)

    chip_specs = [
        ("PUBLIC-CORE VALIDATION", 24 * mm, 37 * mm, 48 * mm),
        ("SEALED TEST", 76 * mm, 37 * mm, 30 * mm),
        ("NEGATIVE RESULTS RETAINED", 110 * mm, 37 * mm, 61 * mm),
    ]
    canvas.setFont("Arial-Bold", 6.7)
    for label, x, y, chip_width in chip_specs:
        canvas.setFillColor(colors.HexColor("#204B63"))
        canvas.roundRect(x, y, chip_width, 8 * mm, 2 * mm, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#BFEFF3"))
        canvas.drawCentredString(x + chip_width / 2, y + 2.8 * mm, label)
    canvas.restoreState()


def draw_body(canvas: object, doc: CrisisForgeDocTemplate) -> None:
    """Draw running header, footer, and page number."""
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, height - 16 * mm, width - 20 * mm, height - 16 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Arial-Bold", 7.2)
    canvas.drawString(20 * mm, height - 12.2 * mm, "CRISISFORGE")
    canvas.setFont("Arial", 7.2)
    canvas.drawRightString(width - 20 * mm, height - 12.2 * mm, "Technical Research Report")

    canvas.setStrokeColor(RULE)
    canvas.line(20 * mm, 15 * mm, width - 20 * mm, 15 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Arial", 7)
    canvas.drawString(20 * mm, 10.7 * mm, "Public-core validation release v0.3.0")
    canvas.drawRightString(width - 20 * mm, 10.7 * mm, f"{doc.page}")
    canvas.restoreState()


def make_styles() -> dict[str, ParagraphStyle]:
    """Create the report's typography system."""
    sample = getSampleStyleSheet()
    return {
        "cover_kicker": ParagraphStyle(
            "CF-CoverKicker",
            parent=sample["Normal"],
            fontName="Arial-Bold",
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#8DE5EA"),
            spaceAfter=15,
        ),
        "cover_title": ParagraphStyle(
            "CF-CoverTitle",
            parent=sample["Title"],
            fontName="Georgia-Bold",
            fontSize=35,
            leading=39,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=18,
        ),
        "cover_subtitle": ParagraphStyle(
            "CF-CoverSubtitle",
            parent=sample["Normal"],
            fontName="Arial",
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#D7E6EF"),
            spaceAfter=18,
        ),
        "cover_meta": ParagraphStyle(
            "CF-CoverMeta",
            parent=sample["Normal"],
            fontName="Arial",
            fontSize=9,
            leading=14,
            textColor=colors.HexColor("#A9C0CF"),
        ),
        "h1": ParagraphStyle(
            "CF-H1",
            parent=sample["Heading1"],
            fontName="Georgia-Bold",
            fontSize=20,
            leading=24,
            textColor=NAVY,
            spaceBefore=15,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "CF-H2",
            parent=sample["Heading2"],
            fontName="Arial-Bold",
            fontSize=13,
            leading=16,
            textColor=NAVY_2,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "CF-Body",
            parent=sample["BodyText"],
            fontName="Arial",
            fontSize=9.2,
            leading=13.3,
            textColor=INK,
            spaceAfter=7,
            allowWidows=0,
            allowOrphans=0,
        ),
        "summary": ParagraphStyle(
            "CF-Summary",
            parent=sample["BodyText"],
            fontName="Arial",
            fontSize=9.4,
            leading=13.7,
            textColor=INK,
            backColor=PALE_CYAN,
            borderColor=colors.HexColor("#A8DDE5"),
            borderWidth=0.6,
            borderPadding=(7, 9, 7, 9),
            spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "CF-Bullet",
            parent=sample["BodyText"],
            fontName="Arial",
            fontSize=9.1,
            leading=13,
            textColor=INK,
            leftIndent=0,
            firstLineIndent=0,
            spaceAfter=2.5,
        ),
        "caption": ParagraphStyle(
            "CF-Caption",
            parent=sample["BodyText"],
            fontName="Arial-Italic",
            fontSize=7.8,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "equation": ParagraphStyle(
            "CF-Equation",
            parent=sample["BodyText"],
            fontName="Arial-Italic",
            fontSize=9.4,
            leading=14,
            alignment=TA_CENTER,
            textColor=NAVY,
            backColor=PALE,
            borderColor=RULE,
            borderWidth=0.5,
            borderPadding=8,
            spaceBefore=5,
            spaceAfter=9,
        ),
        "toc_title": ParagraphStyle(
            "CF-TOCTitle",
            parent=sample["Title"],
            fontName="Georgia-Bold",
            fontSize=25,
            leading=30,
            textColor=NAVY,
            spaceAfter=16,
        ),
        "toc0": ParagraphStyle(
            "CF-TOC0",
            fontName="Arial-Bold",
            fontSize=9.5,
            leading=13,
            leftIndent=0,
            firstLineIndent=0,
            textColor=NAVY,
            spaceBefore=3,
        ),
        "toc1": ParagraphStyle(
            "CF-TOC1",
            fontName="Arial",
            fontSize=8.5,
            leading=11,
            leftIndent=11,
            firstLineIndent=0,
            textColor=MUTED,
        ),
        "reference": ParagraphStyle(
            "CF-Reference",
            parent=sample["BodyText"],
            fontName="Arial",
            fontSize=8,
            leading=11,
            textColor=INK,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "CF-Small",
            parent=sample["BodyText"],
            fontName="Arial",
            fontSize=7.8,
            leading=11,
            textColor=MUTED,
            spaceAfter=5,
        ),
    }


def table_from_rows(
    rows: list[list[str]],
    styles: dict[str, ParagraphStyle],
    available_width: float,
) -> Table:
    """Build a styled, splittable report table."""
    columns = len(rows[0])
    if columns == 2:
        ratios = [0.43, 0.57]
    elif columns == 4:
        ratios = [0.34, 0.22, 0.22, 0.22]
    elif columns == 5:
        ratios = [0.31, 0.1725, 0.1725, 0.1725, 0.1725]
    else:
        ratios = [1 / columns] * columns
    col_widths = [available_width * ratio for ratio in ratios]
    body_style = ParagraphStyle(
        "CF-TableBody",
        parent=styles["small"],
        fontSize=7.25 if columns >= 5 else 7.8,
        leading=9.2 if columns >= 5 else 10,
        textColor=INK,
        spaceAfter=0,
    )
    header_style = ParagraphStyle(
        "CF-TableHeader",
        parent=body_style,
        fontName="Arial-Bold",
        textColor=WHITE,
    )
    prepared: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        style = header_style if row_index == 0 else body_style
        prepared.append([Paragraph(inline_markup(cell), style) for cell in row])
    table = Table(
        prepared,
        colWidths=col_widths,
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), 1, CYAN),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
    ]
    for row_index in range(1, len(rows)):
        background = colors.white if row_index % 2 else PALE
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), background))
    table.setStyle(TableStyle(commands))
    return table


def figure_flowables(
    image_path: Path,
    alt_text: str,
    figure_number: int,
    styles: dict[str, ParagraphStyle],
    available_width: float,
) -> list[Flowable]:
    """Return a size-constrained figure and caption."""
    with PILImage.open(image_path) as image:
        pixel_width, pixel_height = image.size
    width = min(available_width, 169 * mm)
    height = width * pixel_height / pixel_width
    max_height = 102 * mm
    if height > max_height:
        scale = max_height / height
        width *= scale
        height *= scale
    figure = Image(str(image_path), width=width, height=height)
    figure.hAlign = "CENTER"
    caption = Paragraph(
        f"<b>Figure {figure_number}.</b> {inline_markup(alt_text)}",
        styles["caption"],
    )
    return [KeepTogether([Spacer(1, 4), figure, caption])]


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    """Parse a Markdown pipe table and return rows plus the next index."""
    raw_rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].lstrip().startswith("|"):
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        raw_rows.append(cells)
        index += 1
    if len(raw_rows) > 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in raw_rows[1]):
        raw_rows.pop(1)
    return raw_rows, index


def parse_list(
    lines: list[str],
    start: int,
    ordered: bool,
) -> tuple[list[str], int]:
    """Parse a wrapped Markdown list."""
    marker = re.compile(r"^\s*\d+\.\s+" if ordered else r"^\s*-\s+")
    items: list[str] = []
    index = start
    current = ""
    while index < len(lines):
        line = lines[index]
        if marker.match(line):
            if current:
                items.append(current.strip())
            current = marker.sub("", line).strip()
            index += 1
            continue
        if not line.strip():
            break
        if (
            line.startswith("#")
            or line.lstrip().startswith("|")
            or line.startswith("![")
            or line.startswith(r"\[")
            or re.match(r"^\s*(?:-|\d+\.)\s+", line)
        ):
            break
        if current:
            current += " " + line.strip()
            index += 1
            continue
        break
    if current:
        items.append(current.strip())
    return items, index


def parse_markdown(
    source: Path,
    styles: dict[str, ParagraphStyle],
    available_width: float,
) -> list[Flowable]:
    """Convert the report Markdown body into ReportLab flowables."""
    lines = source.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line == "## Technical summary")
    lines = lines[start:]
    story: list[Flowable] = []
    index = 0
    figure_number = 0
    current_section = ""

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("## "):
            current_section = plain_heading(stripped[3:])
            story.extend(
                [
                    CondPageBreak(42 * mm),
                    Spacer(1, 4),
                    Paragraph(inline_markup(current_section), styles["h1"]),
                    SectionRule(),
                    Spacer(1, 5),
                ]
            )
            index += 1
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(inline_markup(stripped[4:]), styles["h2"]))
            index += 1
            continue
        image_match = re.fullmatch(r"!\[([^\]]+)\]\(([^)]+)\)", stripped)
        if image_match:
            figure_number += 1
            image_path = source.parent / image_match.group(2)
            if not image_path.exists():
                raise FileNotFoundError(f"Figure does not exist: {image_path}")
            story.extend(
                figure_flowables(
                    image_path,
                    image_match.group(1),
                    figure_number,
                    styles,
                    available_width,
                )
            )
            index += 1
            continue
        if stripped.startswith("|"):
            rows, index = parse_table(lines, index)
            story.extend(
                [
                    Spacer(1, 3),
                    table_from_rows(rows, styles, available_width),
                    Spacer(1, 9),
                ]
            )
            continue
        if stripped.startswith(r"\["):
            equation_lines: list[str] = []
            if stripped != r"\[":
                equation_lines.append(stripped.removeprefix(r"\["))
            index += 1
            while index < len(lines) and lines[index].strip() != r"\]":
                equation_lines.append(lines[index].strip())
                index += 1
            index += 1
            raw_equation = " ".join(equation_lines)
            if "y_t=" in raw_equation and r"\operatorname{Cov}" in raw_equation:
                equation = (
                    "y_t = log(1 + r_t) = alpha_z_t + B_z_t f_t + D_z_t epsilon_t"
                    "<br/>Cov(D_z epsilon_t) = D_z R_z D_z"
                    "<br/>r_t = expm1(y_t)"
                )
            else:
                equation = html.escape(latex_to_text(raw_equation))
            story.append(Paragraph(equation, styles["equation"]))
            continue
        if re.match(r"^\s*-\s+", line):
            items, index = parse_list(lines, index, ordered=False)
            bullets = [
                ListItem(Paragraph(inline_markup(item), styles["bullet"]), leftIndent=2)
                for item in items
            ]
            story.append(
                ListFlowable(
                    bullets,
                    bulletType="bullet",
                    bulletFontName="Arial",
                    bulletFontSize=7,
                    bulletColor=CYAN,
                    leftIndent=15,
                    bulletOffsetY=1.5,
                    spaceAfter=7,
                )
            )
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            items, index = parse_list(lines, index, ordered=True)
            numbered = [
                ListItem(Paragraph(inline_markup(item), styles["bullet"]), leftIndent=2)
                for item in items
            ]
            story.append(
                ListFlowable(
                    numbered,
                    bulletType="1",
                    bulletFontName="Arial-Bold",
                    bulletFontSize=8,
                    bulletColor=NAVY,
                    leftIndent=18,
                    bulletOffsetY=1.5,
                    spaceAfter=7,
                )
            )
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                break
            if (
                candidate.startswith("#")
                or candidate.startswith("![")
                or candidate.startswith("|")
                or candidate.startswith(r"\[")
                or re.match(r"^(?:-|\d+\.)\s+", candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        paragraph_text = " ".join(paragraph_lines)
        style = styles["summary"] if current_section == "Technical summary" else styles["body"]
        story.append(Paragraph(inline_markup(paragraph_text), style))

    return story


def reference_flowables(styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    """Build the explicit reference section appended to the PDF."""
    flowables: list[Flowable] = [
        PageBreak(),
        Paragraph("References", styles["h1"]),
        SectionRule(),
        Spacer(1, 7),
        Paragraph(
            "Primary and archival sources cited in the report. Publication status and "
            "claim boundaries are documented in docs/literature_review.md. URLs were "
            "verified for the 28 July 2026 public-core release.",
            styles["body"],
        ),
    ]
    items = []
    for citation, url in REFERENCES:
        markup = (
            f"{html.escape(normalize_typography(citation))} "
            f'<link href="{html.escape(url, quote=True)}" color="#007E9B">'
            f"<u>{html.escape(url)}</u></link>"
        )
        items.append(ListItem(Paragraph(markup, styles["reference"]), leftIndent=3))
    flowables.append(
        ListFlowable(
            items,
            bulletType="1",
            bulletFontName="Arial-Bold",
            bulletFontSize=7.5,
            bulletColor=NAVY,
            leftIndent=18,
            bulletOffsetY=1,
            spaceAfter=10,
        )
    )
    flowables.extend(
        [
            Paragraph("Release note", styles["h2"]),
            Paragraph(
                "This PDF is a presentation layer over the registered Markdown report. "
                "The Python source, exact configurations, experiment receipts, lockfile, "
                "and machine-readable artifacts remain the authoritative reproducibility "
                "record. No post-2019 model evaluation is introduced by this document.",
                styles["body"],
            ),
        ]
    )
    return flowables


def build_pdf(source: Path, output: Path) -> None:
    """Build the PDF with a cover, dynamic TOC, body, and references."""
    register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()

    page_width, page_height = A4
    left_margin = 20 * mm
    right_margin = 20 * mm
    top_margin = 22 * mm
    bottom_margin = 20 * mm
    available_width = page_width - left_margin - right_margin
    body_frame = Frame(
        left_margin,
        bottom_margin,
        available_width,
        page_height - top_margin - bottom_margin,
        id="body-frame",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    cover_frame = Frame(
        24 * mm,
        42 * mm,
        page_width - 48 * mm,
        page_height - 72 * mm,
        id="cover-frame",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc = CrisisForgeDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title="CrisisForge: Decision-Focused Market Simulation under Regime Shifts",
        author="康智雄",
        subject="Public-core validation technical research report v0.3.0",
        creator="CrisisForge Python report builder",
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="Cover", frames=[cover_frame], onPage=draw_cover),
            PageTemplate(id="Body", frames=[body_frame], onPage=draw_body),
        ]
    )

    toc = TableOfContents()
    toc.levelStyles = [styles["toc0"], styles["toc1"]]
    toc.dotsMinLevel = 0

    story: list[Flowable] = [
        Spacer(1, 53 * mm),
        Paragraph("CRISISFORGE", styles["cover_kicker"]),
        Paragraph(
            "Decision-Focused Market Simulation under Regime Shifts",
            styles["cover_title"],
        ),
        Paragraph(
            "Factor-Structured Temporal Diffusion, Asset-Level Tail Risk, "
            "and Robust Portfolio Control",
            styles["cover_subtitle"],
        ),
        Paragraph(
            "Technical research report<br/>Public-core validation release v0.3.0<br/>28 July 2026",
            styles["cover_meta"],
        ),
        NextPageTemplate("Body"),
        PageBreak(),
        Paragraph("Contents", styles["toc_title"]),
        Paragraph(
            "This report separates implemented evidence, engineering pilots, and "
            "unidentified extensions. Lower distribution and risk scores are better "
            "unless explicitly stated.",
            styles["small"],
        ),
        toc,
        PageBreak(),
    ]
    story.extend(parse_markdown(source, styles, available_width))
    story.extend(reference_flowables(styles))
    doc.multiBuild(story)


def validate_pdf(output: Path, expected_figures: int = 8) -> dict[str, int]:
    """Run structural checks on the generated report."""
    if not output.exists() or output.stat().st_size < 100_000:
        raise RuntimeError(f"PDF missing or unexpectedly small: {output}")
    reader = PdfReader(str(output))
    page_count = len(reader.pages)
    if page_count < 15:
        raise RuntimeError(f"Expected a substantial technical report, got {page_count} pages")
    extracted = "\n".join((page.extract_text() or "") for page in reader.pages)
    required_fragments = [
        "CrisisForge",
        "Technical summary",
        "0.094315",
        "0.0731",
        "3.4354",
        "No reported result uses the post-2019 test set",
        "References",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in extracted]
    if missing:
        raise RuntimeError(f"Required report text missing from PDF: {missing}")
    image_count = sum(len(getattr(page, "images", [])) for page in reader.pages)
    if image_count < expected_figures:
        raise RuntimeError(
            f"Expected at least {expected_figures} embedded figures, found {image_count}"
        )
    return {
        "pages": page_count,
        "bytes": output.stat().st_size,
        "images": image_count,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    source = args.source.resolve()
    output = args.output.resolve()
    build_pdf(source, output)
    result = validate_pdf(output)
    print(
        f"Built {output} | pages={result['pages']} | "
        f"images={result['images']} | bytes={result['bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
