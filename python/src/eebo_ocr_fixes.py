import re

OCR_FIXES = {
    r'\bberty\b': 'liberty',
    r'\bli\|berty\b': 'liberty',  # catches literal bars in OCR
    r'\bli\¦berty\b': 'liberty',  # some OCR use different bar symbols
    r'\bliſerty\b': 'liberty',    # long s inside 'liberty'
}

def apply_ocr_fixes(text: str) -> str:
    for pattern, replacement in OCR_FIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text
