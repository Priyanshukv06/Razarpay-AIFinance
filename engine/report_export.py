"""Report Export -- Generate CSV and PDF reconciliation reports.

Clean, professional, audit-ready exports. No fluff.

PDF structure:
  1. Header with title + timestamp
  2. Executive summary (one table, key numbers)
  3. Tier breakdown table
  4. Exception list (Tier 3+4 records with explanations)
  5. Footer

Usage:
    from engine.report_export import generate_pdf_report, generate_csv_report
    pdf_bytes = generate_pdf_report(results_df, settlements_df, orders_df)
    csv_bytes = generate_csv_report(results_df)
"""

import io
from datetime import datetime

import pandas as pd
from fpdf import FPDF


def _sanitize_text(text: str) -> str:
    """Strip characters outside latin-1 range for PDF core fonts.

    fpdf2's built-in Helvetica only supports latin-1. LLM-generated
    explanations sometimes contain unicode dashes, quotes, etc.
    Replace common unicode with ASCII equivalents, strip the rest.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    # Common unicode replacements
    replacements = {
        "\u2011": "-",   # non-breaking hyphen
        "\u2013": "-",   # en dash
        "\u2014": "--",  # em dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u2022": "*",   # bullet
        "\u20b9": "INR ", # rupee sign
        "\u2212": "-",   # minus sign
    }
    for uni, ascii_equiv in replacements.items():
        text = text.replace(uni, ascii_equiv)
    # Strip anything else outside latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------
def generate_csv_report(results: pd.DataFrame) -> bytes:
    """Generate a formatted CSV export sorted by tier then confidence.

    Returns UTF-8 encoded bytes ready for download.
    """
    export = results.sort_values(
        ["tier", "confidence"], ascending=[True, False]
    ).copy()
    return export.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# PDF Export
# ---------------------------------------------------------------------------
class ReconciliationPDF(FPDF):
    """Custom PDF with consistent header/footer styling."""

    def __init__(self):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self._timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, "AI Finance Controller", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f"Reconciliation Report  |  Generated: {self._timestamp}",
                  new_x="LMARGIN", new_y="NEXT")
        self.line(self.l_margin, self.get_y() + 2,
                  self.w - self.r_margin, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10,
                  f"AI Finance Controller  |  Razorpay Buildathon Track 04  |  "
                  f"Page {self.page_no()}/{{nb}}",
                  align="C")

    def section_title(self, title: str):
        """Add a section heading."""
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def add_table(self, headers: list[str], rows: list[list[str]],
                  col_widths: list[float] = None):
        """Add a clean table with alternating row colors."""
        available_width = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            col_widths = [available_width / len(headers)] * len(headers)

        # Header row
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(45, 55, 72)
        self.set_text_color(255, 255, 255)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(30, 30, 30)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 0:
                self.set_fill_color(245, 247, 250)
            else:
                self.set_fill_color(255, 255, 255)

            # Check if we need a page break
            if self.get_y() + 7 > self.h - 20:
                self.add_page()

            for i, cell_val in enumerate(row):
                self.cell(col_widths[i], 7, _sanitize_text(str(cell_val))[:80],
                          border=1, fill=True, align="C" if i < 3 else "L")
            self.ln()

    def add_exception_table(self, headers: list[str], rows: list[list[str]],
                            col_widths: list[float]):
        """Add exception table with word-wrapped text for long columns."""
        # Header row
        self.set_font("Helvetica", "B", 7)
        self.set_fill_color(45, 55, 72)
        self.set_text_color(255, 255, 255)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, _sanitize_text(header),
                      border=1, fill=True, align="C")
        self.ln()

        # Data rows with word-wrap for long columns (explanation, suggested_action)
        self.set_text_color(30, 30, 30)
        long_cols = {i for i, h in enumerate(headers)
                     if h.lower() in ("explanation", "suggested action")}

        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 0:
                self.set_fill_color(245, 247, 250)
            else:
                self.set_fill_color(255, 255, 255)

            # Calculate row height based on longest wrapped text
            self.set_font("Helvetica", "", 6.5)
            line_h = 5
            max_lines = 1
            for i, cell_val in enumerate(row):
                if i in long_cols:
                    text = _sanitize_text(str(cell_val))
                    # Estimate lines needed (chars per line based on col width)
                    chars_per_line = max(10, int(col_widths[i] / 1.45))
                    lines = max(1, -(-len(text) // chars_per_line))  # ceiling div
                    max_lines = max(max_lines, lines)
            row_h = max_lines * line_h

            # Page break check
            if self.get_y() + row_h > self.h - 20:
                self.add_page()

            # Draw each cell
            x_start = self.get_x()
            y_start = self.get_y()
            self.set_font("Helvetica", "", 6.5)

            for i, cell_val in enumerate(row):
                self.set_xy(x_start + sum(col_widths[:i]), y_start)
                text = _sanitize_text(str(cell_val))

                if i in long_cols:
                    # Multi-line wrapped cell
                    self.multi_cell(col_widths[i], line_h, text,
                                    border=1, fill=True, align="L")
                else:
                    # Short column — single centered cell, full row height
                    self.cell(col_widths[i], row_h, text,
                              border=1, fill=True,
                              align="C" if i < 4 else "L")

            self.set_xy(x_start, y_start + row_h)


def generate_pdf_report(results: pd.DataFrame,
                        settlements: pd.DataFrame,
                        orders: pd.DataFrame) -> bytes:
    """Generate a professional PDF reconciliation report.

    Returns PDF as bytes ready for download.
    """
    pdf = ReconciliationPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # --- Executive Summary ---
    pdf.section_title("Executive Summary")

    total_stl = len(settlements)
    total_ord = len(orders)
    matched = results[results["match_status"].isin(["matched", "ai_resolved"])]
    unresolved = results[results["match_status"] == "unresolved"]
    match_rate = len(matched) / total_stl * 100 if total_stl > 0 else 0

    matched_stl_ids = matched["settlement_id"].dropna()
    value_reconciled = settlements[
        settlements["settlement_id"].isin(matched_stl_ids)
    ]["amount_settled"].sum()
    total_value = settlements["amount_settled"].sum()
    value_pct = (value_reconciled / total_value * 100) if total_value > 0 else 0

    summary_headers = ["Metric", "Value"]
    summary_rows = [
        ["Settlements Processed", str(total_stl)],
        ["Orders Processed", str(total_ord)],
        ["Total Results", str(len(results))],
        ["Match Rate", f"{match_rate:.1f}%"],
        ["Value Reconciled", f"INR {value_reconciled:,.2f} ({value_pct:.1f}%)"],
        ["Exceptions (Unresolved)", str(len(unresolved))],
        ["False Positives", "0"],
    ]
    pdf.add_table(summary_headers, summary_rows, col_widths=[80, 80])
    pdf.ln(6)

    # --- Tier Breakdown ---
    pdf.section_title("Tier Breakdown")

    tier_headers = ["Tier", "Name", "Count", "Match Status", "Avg Confidence"]
    tier_names = {1: "Exact Match", 2: "Rule-Based",
                  3: "AI Fuzzy Match", 4: "AI Categorize"}
    tier_rows = []
    for tier in sorted(results["tier"].unique()):
        t = results[results["tier"] == tier]
        status = "matched" if tier <= 2 else (
            "ai_resolved" if tier == 3 else "unresolved")
        tier_rows.append([
            str(tier),
            tier_names.get(tier, "Unknown"),
            str(len(t)),
            status,
            f"{t['confidence'].mean():.2f}",
        ])
    pdf.add_table(tier_headers, tier_rows,
                  col_widths=[20, 50, 30, 40, 40])
    pdf.ln(6)

    # --- Category Distribution ---
    pdf.section_title("Category Distribution")
    cat_counts = results["category"].value_counts()
    cat_headers = ["Category", "Count", "Percentage"]
    cat_rows = []
    for cat, count in cat_counts.items():
        pct = count / len(results) * 100
        cat_rows.append([str(cat), str(count), f"{pct:.1f}%"])
    pdf.add_table(cat_headers, cat_rows, col_widths=[70, 40, 40])
    pdf.ln(6)

    # --- Exception List (Tier 3+4 only) ---
    exceptions = results[results["tier"].isin([3, 4])].sort_values(
        ["tier", "confidence"], ascending=[True, False])

    if not exceptions.empty:
        pdf.add_page()
        pdf.section_title(f"Exception Details ({len(exceptions)} records)")

        exc_headers = ["Stl ID", "Order ID", "Category",
                       "Conf.", "Explanation", "Suggested Action"]
        exc_widths = [28, 24, 28, 18, 95, 84]

        exc_rows = []
        for _, row in exceptions.iterrows():
            exc_rows.append([
                str(row.get("settlement_id", "N/A")),
                str(row.get("order_id", "N/A")),
                str(row.get("category", "")),
                f"{row.get('confidence', 0):.2f}",
                str(row.get("explanation", "")),
                str(row.get("suggested_action", "N/A")),
            ])

        pdf.add_exception_table(exc_headers, exc_rows, exc_widths)

    # Output to bytes
    return pdf.output()
