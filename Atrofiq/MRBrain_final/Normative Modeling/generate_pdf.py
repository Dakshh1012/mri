
import hashlib
# Monkeypatch md5 to ignore usedforsecurity
_real_md5 = hashlib.md5
def _wrapped_md5(*args, **kwargs):
    kwargs.pop('usedforsecurity', None)
    return _real_md5(*args, **kwargs)
hashlib.md5 = _wrapped_md5

import markdown
import re
import os
from xhtml2pdf import pisa

def preprocess_markdown(text):
    """
    Clean up custom tags for PDF rendering.
    """
    # Remove carousel tags
    text = text.replace("````carousel", "").replace("````", "")
    text = text.replace("<!-- slide -->", "")
    
    # Handle Alerts (GitHub style) -> Convert to Blockquotes with bold headers
    def replace_alert(match):
        level = match.group(1)
        return f"> **{level}**"
        
    text = re.sub(r"> \[!(\w+)\]", replace_alert, text)
    
    return text

def convert_to_pdf(md_path, pdf_path):
    print(f"Reading {md_path}...")
    with open(md_path, 'r') as f:
        text = f.read()
        
    # Pre-process
    print("Pre-processing markdown...")
    text = preprocess_markdown(text)
    
    # Convert to HTML
    print("Converting to HTML...")
    html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    
    # Add CSS Styling
    # xhtml2pdf supports basic CSS
    full_html = f"""
    <html>
    <head>
    <style>
        body {{ font-family: Helvetica, sans-serif; font-size: 10pt; }}
        img {{ max-width: 100%; height: auto; }}
        blockquote {{ border-left: 4px solid #3498db; padding-left: 10px; background: #eee; }}
        pre {{ background: #eee; padding: 5px; border: 1px solid #ddd; }}
        table {{ border-collapse: collapse; width: 100%; }}
        td, th {{ border: 1px solid #ddd; padding: 5px; }}
        th {{ background-color: #f2f2f2; }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """
    
    # Render PDF
    print(f"Rendering PDF to {pdf_path}...")
    with open(pdf_path, "wb") as output_file:
        pisa_status = pisa.CreatePDF(full_html, dest=output_file)
        
    if pisa_status.err:
        print("Error converting PDF")
    else:
        print("Done!")

if __name__ == "__main__":
    md_file = "/home/anirudh/.gemini/antigravity/brain/ae28d41b-a95b-40ba-a987-86c191e416ca/walkthrough.md"
    pdf_file = "/home/anirudh/.gemini/antigravity/brain/ae28d41b-a95b-40ba-a987-86c191e416ca/Walkthrough_Report.pdf"
    
    convert_to_pdf(md_file, pdf_file)
