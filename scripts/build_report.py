"""Render report/REPORT.md to a self-contained, print-ready HTML file.

Figures are inlined as base64 so the HTML is a single file, and the CSS is
tuned so the result fits the assignment's 8-page limit when printed to PDF.

    python -m pip install markdown
    python scripts/build_report.py .
    # then, to get the PDF:
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
        --disable-gpu --no-pdf-header-footer \
        --print-to-pdf="$PWD/report/REPORT.pdf" "file://$PWD/report/REPORT.html"
"""

import base64
import pathlib
import re
import sys

import markdown

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
src = (root / "report/REPORT.md").read_text(encoding="utf-8")

def embed(match):
    alt, rel = match.group(1), match.group(2)
    path = (root / "report" / rel).resolve()
    if not path.exists():
        return match.group(0)
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><img alt="{alt}" src="data:image/png;base64,{data}">'
            f'<figcaption>{alt}</figcaption></figure>')

body = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", embed, src)
html_body = markdown.markdown(body, extensions=["tables", "fenced_code"])

CSS = """
@page { size: A4; margin: 13mm 12mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 8.6pt;
       line-height: 1.30; color: #111; }
h1 { font-size: 16pt; margin: 0 0 2pt; border-bottom: 2px solid #333; padding-bottom: 3pt; }
h2 { font-size: 11.4pt; margin: 9pt 0 3.5pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; }
h3 { font-size: 9.8pt; margin: 6.5pt 0 2.5pt; }
h2, h3 { page-break-after: avoid; }
p { margin: 4pt 0; }
/* Fixed layout + aggressive wrapping: long file paths were overflowing the page. */
table { border-collapse: collapse; width: 100%; table-layout: fixed;
        margin: 3.5pt 0 5.5pt; font-size: 7.2pt; page-break-inside: avoid; }
th, td { border: 1px solid #c4c4c4; padding: 2.5pt 4pt; text-align: left; vertical-align: top;
         word-break: break-word; overflow-wrap: anywhere; }
th { background: #ececec; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
code { background: #f0f0f0; padding: 0.4pt 2pt; border-radius: 2px; font-size: 7.6pt;
       word-break: break-word; overflow-wrap: anywhere; }
td code, th code { background: none; padding: 0; font-size: 7.4pt; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 4pt; font-size: 7.8pt;
      white-space: pre-wrap; word-break: break-word; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
/* Plots are tall (3 stacked panels); cap the height so two share a page
   instead of each taking one, which pushed the report past the 8-page limit. */
figure { margin: 5pt 0; page-break-inside: avoid; text-align: center; }
figure img { max-width: 52%; max-height: 84mm; width: auto; height: auto;
             border: 1px solid #ddd; }
figcaption { font-size: 7.2pt; color: #555; margin-top: 1.5pt; font-style: italic; }
ul, ol { margin: 3pt 0 5pt; padding-left: 15pt; }
li { margin: 1.2pt 0; }
hr { border: none; border-top: 1px solid #ddd; margin: 6pt 0; }
"""

html = (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<title>PA2 - Instruction Tuning and DPO on a Small Language Model</title>'
        f'<style>{CSS}</style></head><body>{html_body}</body></html>')
out = root / "report/REPORT.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size/1024:.0f} kB)")
