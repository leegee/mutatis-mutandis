import re

OCR_FIXES = {
    r'[\|¦]': '',                 # remove OCR bars
    r'\bberty\b': 'liberty',
}

def apply_ocr_fixes(text: str) -> str:
    for pattern, replacement in OCR_FIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text
