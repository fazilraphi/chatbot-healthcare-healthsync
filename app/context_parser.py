import re


def extract_duration(text: str):

    patterns = [
        r"(\d+)\s*days?",
        r"(\d+)\s*hours?",
        r"(\d+)\s*weeks?"
    ]

    text = text.lower()

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    return None


def detect_severity(text: str):

    text = text.lower()

    if "severe" in text or "unbearable" in text:
        return "severe"

    if "moderate" in text:
        return "moderate"

    if "mild" in text:
        return "mild"

    return None