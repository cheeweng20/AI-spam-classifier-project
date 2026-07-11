"""Shared text preprocessing used during training and prediction."""

import re
import string


def clean_text(text):
    """Return normalized text using the preprocessing expected by the models."""
    text = str(text)
    text = re.sub(r"\w*\d\w*", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()
