"""Translate a question into English before the lexical search (D-021).

The pages are in English and full-text search only matches words, so a Spanish question finds
nothing but its acronyms (D-019). Vector search does not need this: bge-m3 already places a
Spanish question next to the English text that answers it (D-003).
"""

from sentinel1_rag.config import CHAT_MODEL, CHAT_NUM_CTX
from sentinel1_rag.ollama import Ollama

# Detection comes first: placed after the translation instructions, it was ignored for 3 of the
# 7 English eval questions, which came back reworded (D-021). British spelling, because the
# documentation uses it: "polarization" and "polarisation" stem to different lexemes.
TRANSLATE_PROMPT = """\
You receive a question about the Sentinel-1 radar satellite mission.
- If the question is written in English, reply with the single word ENGLISH and nothing else.
- Otherwise, translate it into English, using the standard terminology of radar remote sensing \
and British spelling, as the documentation does. Keep acronyms, product names and numbers \
exactly as they are. Reply with the translation only."""

# Temperature 0 makes the translation repeatable, so the eval stays deterministic.
TRANSLATE_OPTIONS = {"num_ctx": CHAT_NUM_CTX, "temperature": 0, "num_predict": 128}


def to_english(ollama: Ollama, question: str) -> str:
    """The question in English: the question itself when it already is, or when the
    translation was cut off."""
    messages = [{"role": "system", "content": TRANSLATE_PROMPT}, {"role": "user", "content": question}]
    response = ollama.chat(CHAT_MODEL, messages, TRANSLATE_OPTIONS)
    text = response["message"]["content"].strip()
    if response["done_reason"] == "length" or text.rstrip(".").upper() == "ENGLISH":
        return question
    return text


def use_translation(method: str, option: bool | None) -> bool:
    """Resolve --translate / --no-translate: translation is on by default wherever there is a
    lexical search to translate for (D-022), and it cannot apply to vector search alone."""
    if option is None:
        return method != "vector"
    if option and method == "vector":
        raise SystemExit("--translate only changes the lexical search: use it with --method lexical or hybrid")
    return option
