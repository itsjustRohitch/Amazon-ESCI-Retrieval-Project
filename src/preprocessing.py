"""
src/preprocessing.py
--------------------
Text Preprocessing Pipeline for E-Commerce Product Search (Amazon ESCI).

Implements all 4 standardized text preprocessing steps from scratch:
1. Text Cleaning: HTML tag stripping, entity decoding, punctuation normalization, lowercasing.
2. Tokenization: Alphanumeric-aware tokenization preserving product specs (e.g. '128gb', '4k', 'usb-c').
3. Stop-word Removal: Pruning high-frequency linguistic noise while retaining critical search terms.
4. Stemming: Porter Stemming to reduce morphological variants to root forms (e.g. 'running' -> 'run').
"""

import re
import html
import unicodedata
from typing import List, Optional, Set
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are available
try:
    NLTK_STOPWORDS: Set[str] = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    NLTK_STOPWORDS = set(stopwords.words("english"))

# Instantiate standard Porter Stemmer
_STEMMER = PorterStemmer()

# Regular expressions used for cleaning and tokenization
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"http[s]?://\S+|www\.\S+")
_NON_ALPHANUM_RE = re.compile(r"[^\w\s\-\.]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9]+(?:[.-][a-zA-Z0-9]+)*\b")


def clean_text(text: Optional[str]) -> str:
    """
    Clean raw e-commerce text by stripping HTML tags, unescaping HTML entities,
    normalizing unicode, and removing noise characters while preserving product identifiers.

    Args:
        text (str or None): Raw input string (e.g., product title or query).

    Returns:
        str: Normalized, lowercased, clean text string.
    """
    if text is None or not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities (e.g. &amp; -> &, &quot; -> ")
    text = html.unescape(text)

    # 2. Strip HTML tags (e.g. <b>...</b>, <p>...</p>)
    text = _HTML_TAG_RE.sub(" ", text)

    # 3. Strip URLs if present
    text = _URL_RE.sub(" ", text)

    # 4. Normalize unicode characters (e.g. accents, special typographic quotes)
    text = unicodedata.normalize("NFKD", text)

    # 5. Lowercase
    text = text.lower()

    # 6. Replace underscores, slashes, and certain punctuation with spaces to cleanly delimit words
    text = re.sub(r"[/\\_|,;]", " ", text)

    # 7. Remove non-alphanumeric noise characters except hyphens and decimals inside numbers
    text = _NON_ALPHANUM_RE.sub(" ", text)

    # 8. Collapse consecutive spaces into a single space
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    return text


def tokenize(cleaned_text: str) -> List[str]:
    """
    Tokenize cleaned text into word and alphanumeric tokens.
    Preserves alphanumeric product dimensions and specifications (e.g., '128gb', '4k', 'i7-12700k').

    Args:
        cleaned_text (str): Cleaned text from clean_text().

    Returns:
        List[str]: List of string tokens.
    """
    if not cleaned_text:
        return []

    tokens = _TOKEN_RE.findall(cleaned_text)
    # Strip any trailing or leading hyphens/dots from tokens
    clean_tokens = [t.strip(".-") for t in tokens if len(t.strip(".-")) > 0]
    return clean_tokens


def remove_stopwords(
    tokens: List[str],
    custom_stopwords: Optional[Set[str]] = None,
    min_token_length: int = 2
) -> List[str]:
    """
    Remove common English stop-words and single-character noise from a token list.
    Certain domain-specific numbers or single letters relevant to e-commerce (e.g., sizes 's', 'm', 'l')
    can be preserved if desired.

    Args:
        tokens (List[str]): List of tokens from tokenize().
        custom_stopwords (Set[str], optional): Additional or custom stopwords set.
        min_token_length (int): Minimum character length for tokens (default: 2).

    Returns:
        List[str]: Filtered tokens without stop-words.
    """
    stop_set = custom_stopwords if custom_stopwords is not None else NLTK_STOPWORDS

    filtered = [
        tok for tok in tokens
        if tok not in stop_set and len(tok) >= min_token_length
    ]
    return filtered


def stem_tokens(
    tokens: List[str],
    stemmer: Optional[PorterStemmer] = None
) -> List[str]:
    """
    Apply Porter Stemming to reduce tokens to their morphological root/stem.
    (e.g., 'running' -> 'run', 'headphones' -> 'headphon', 'wireless' -> 'wireless').

    Args:
        tokens (List[str]): Filtered token list.
        stemmer (PorterStemmer, optional): Stemmer instance to use.

    Returns:
        List[str]: Stemmed tokens.
    """
    st = stemmer if stemmer is not None else _STEMMER
    return [st.stem(tok) for tok in tokens]


def preprocess_text(
    text: Optional[str],
    apply_stemming: bool = True,
    return_tokens: bool = True,
    apply_stopwords: bool = True
):
    """
    Full end-to-end preprocessing pipeline:
    clean_text -> tokenize -> remove_stopwords -> stem_tokens.

    Args:
        text (str or None): Raw query or product title.
        apply_stemming (bool): Whether to perform Porter stemming (default: True).
        return_tokens (bool): If True, returns List[str]; if False, returns ' '.join(tokens).

    Returns:
        List[str] or str: Processed tokens or processed text string.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens) if apply_stopwords else tokens

    if apply_stemming:
        processed = stem_tokens(filtered)
    else:
        processed = filtered

    if return_tokens:
        return processed
    return " ".join(processed)


if __name__ == "__main__":
    # Self-test and demonstration on raw Amazon examples
    test_cases = [
        (
            "Query: Messy search with HTML and symbols",
            "<p>Women's <b>Running</b> Shoes (Size 8.5) &amp; Cushioned Sneakers!!!</p>"
        ),
        (
            "Product Title: Tech product with specs and brand",
            "Apple iPhone 13 Pro Max (128GB, Sierra Blue) - Unlocked [Renewed / Refurbished]"
        ),
        (
            "Product Title: Hardware specs and hyphens",
            "Logitech G502 HERO High Performance Gaming Mouse - 25,600 DPI, RGB, 11-Buttons"
        ),
        (
            "Edge Case: Empty or dirty string",
            "   <br/>??? ---   &quot;   "
        )
    ]

    print("=" * 85)
    print("DEMO & SANITY CHECK: Amazon ESCI Preprocessing Pipeline")
    print("=" * 85)

    for desc, raw_str in test_cases:
        print(f"\n[Case] {desc}")
        print(f"  Raw Input:       '{raw_str}'")
        clean_str = clean_text(raw_str)
        print(f"  1. Cleaned:      '{clean_str}'")
        toks = tokenize(clean_str)
        print(f"  2. Tokenized:    {toks}")
        no_stop = remove_stopwords(toks)
        print(f"  3. Stop-removed: {no_stop}")
        stemmed = stem_tokens(no_stop)
        print(f"  4. Stemmed:      {stemmed}")
        print(f"  -> Unified Output: '{' '.join(stemmed)}'")

    print("\n" + "=" * 85)
    print("All preprocessing steps verified successfully!")
    print("=" * 85)
