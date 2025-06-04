# utils/pdf_utils.py
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file.
    Returns a single string containing the text.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text