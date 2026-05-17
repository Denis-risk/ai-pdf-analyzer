import html
import io
import re
from collections import Counter

import streamlit as st
from pypdf import PdfReader

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="AI PDF Analyzer",
    page_icon="📄",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetric"] label {
        color: #475569 !important;
        font-weight: 600;
    }
    .summary-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4f46e5;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        line-height: 1.75;
        color: #1e293b;
    }
    .summary-box h4 {
        margin: 0 0 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #334155;
    }
    .summary-box p {
        margin: 0 0 0.85rem 0;
    }
    .summary-box p:last-child {
        margin-bottom: 0;
    }
    .upload-hint {
        color: #94a3b8;
        font-size: 0.9rem;
    }
</style>
"""

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall", "can",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "she", "his", "her", "not", "no",
}


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    """Read a PDF from bytes and return full text plus page count."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    full_text = "\n\n".join(pages).strip()
    return full_text, len(reader.pages)


def count_words(text: str) -> int:
    """Approximate word count using whitespace splitting."""
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def clean_document_text(text: str) -> str:
    """Normalize PDF text: collapse whitespace and fix broken line wraps."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split cleaned text into full sentences, skipping very short fragments."""
    text = clean_document_text(text)
    if not text:
        return []

    raw_chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(]|$)", text)
    sentences: list[str] = []
    min_chars = 40
    min_words = 8

    for chunk in raw_chunks:
        sentence = chunk.strip()
        if not sentence:
            continue
        word_count = len(re.findall(r"\b\w+\b", sentence))
        if len(sentence) < min_chars or word_count < min_words:
            continue
        if sentence[-1] not in ".!?":
            sentence += "."
        if sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        sentences.append(sentence)

    return sentences


def meaningful_tokens(sentence: str) -> list[str]:
    """Words that count toward importance (letters only, not stopwords)."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
    return [w for w in words if w not in STOP_WORDS]


def choose_summary_length(sentence_count: int) -> int:
    """Pick how many sentences to include (between 5 and 7 when possible)."""
    if sentence_count <= 5:
        return sentence_count
    return min(7, max(5, sentence_count))


def score_sentences(sentences: list[str]) -> list[float]:
    """
    Score each sentence by how often its meaningful words appear in the document.
    Frequent non-stopwords indicate central topics.
    """
    all_meaningful = [w for s in sentences for w in meaningful_tokens(s)]
    if not all_meaningful:
        return [0.0] * len(sentences)

    word_freq = Counter(all_meaningful)
    scores: list[float] = []

    for sentence in sentences:
        tokens = meaningful_tokens(sentence)
        if not tokens:
            scores.append(0.0)
            continue
        # Sum of word frequencies; divide slightly by length to avoid bias toward long lines
        total = sum(word_freq[w] for w in tokens)
        scores.append(total / (len(tokens) ** 0.5))

    return scores


def summarize_text(text: str) -> list[str]:
    """
    Extractive summary: pick 5–7 high-scoring sentences, keep document order.
    No external APIs — only word frequency and stopword filtering.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    pick_count = choose_summary_length(len(sentences))
    if len(sentences) <= pick_count:
        return sentences

    scores = score_sentences(sentences)
    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda i: scores[i],
        reverse=True,
    )
    selected = sorted(ranked_indices[:pick_count])
    return [sentences[i] for i in selected]


def format_summary_html(sentences: list[str]) -> str:
    """Build readable HTML for the summary panel."""
    if not sentences:
        return (
            '<div class="summary-box">'
            "<h4>Document Summary</h4>"
            "<p>Not enough readable text to generate a summary.</p>"
            "</div>"
        )
    body = "".join(f"<p>{html.escape(s)}</p>" for s in sentences)
    return f'<div class="summary-box"><h4>Document Summary</h4>{body}</div>'


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown('<p class="main-header">AI PDF Analyzer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload a PDF to extract text, view stats, and get a quick summary — no API keys required.</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Select a PDF document from your computer.",
    )

    if uploaded_file is None:
        st.markdown(
            '<p class="upload-hint">👆 Upload a PDF to get started.</p>',
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Reading your PDF..."):
        file_bytes = uploaded_file.read()
        text, page_count = extract_text_from_pdf(file_bytes)

    if not text:
        st.warning(
            "No readable text was found in this PDF. "
            "It may be scanned as images only — try a text-based PDF."
        )
        return

    char_count = len(text)
    word_count = count_words(text)

    st.success(f"Processed **{uploaded_file.name}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Pages", page_count)
    col2.metric("Characters", f"{char_count:,}")
    col3.metric("Words (approx.)", f"{word_count:,}")

    st.divider()

    summary_sentences = summarize_text(text)
    st.markdown(format_summary_html(summary_sentences), unsafe_allow_html=True)
    st.caption(
        "Extractive summary: the most informative sentences from your document, "
        "shown in original order. Generated locally — no API keys."
    )

    st.divider()

    with st.expander("View extracted text", expanded=False):
        st.text_area(
            "Full document text",
            value=text,
            height=400,
            label_visibility="collapsed",
        )


if __name__ == "__main__":
    main()
