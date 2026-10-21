# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ukrainian alphabet, Unicode normalization and character dictionary helpers.

The helpers in this module are dependency free (standard library only) so that
they can be used by dataset tools, metrics and tests without importing paddle.
"""

from __future__ import annotations

import os
import unicodedata
from typing import Dict, Iterable, List, Sequence

UK_UPPER = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
UK_LOWER = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
UK_ALPHABET = UK_UPPER + UK_LOWER

# Letters that distinguish Ukrainian from other Cyrillic alphabets and are the
# most common source of OCR errors (confusions with Г/г, Е/е, Latin I/i, ...).
UK_CRITICAL_CHARS = "ҐґЄєІіЇї"

# U+0027 APOSTROPHE and U+2019 RIGHT SINGLE QUOTATION MARK are both used as the
# Ukrainian apostrophe and are both kept as separate classes.
APOSTROPHE_ASCII = "'"
APOSTROPHE_TYPOGRAPHIC = "’"
# U+02BC MODIFIER LETTER APOSTROPHE is the orthographically recommended
# Ukrainian apostrophe. It is rendered identically to U+2019, so an image based
# model cannot distinguish them; labels are canonicalized to U+2019.
APOSTROPHE_MODIFIER = "ʼ"
UK_APOSTROPHES = APOSTROPHE_ASCII + APOSTROPHE_TYPOGRAPHIC + APOSTROPHE_MODIFIER

# Russian / Belarusian letters which are not part of the Ukrainian alphabet.
NON_UK_CYRILLIC = "ЁёЪъЫыЭэЎў"

# Punctuation and symbols kept in the Ukrainian dictionary in addition to
# letters and digits: quotes, dashes, brackets, currency, dates, addresses and
# simple math. Every character must exist in the base (eslav) dictionary so
# that pretrained weights can be transferred to it.
UK_EXTRA_SYMBOLS = (
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "«»„“”‘’‚‹›"
    "–—‐‑−…·•"
    "№₴€£¥¢°§©®™"
    "×÷±≤≥≠≈∙⋅"
    "²³¹½¼¾"
)

_UNICODE_REPLACEMENTS = {
    APOSTROPHE_MODIFIER: APOSTROPHE_TYPOGRAPHIC,
    " ": " ",  # no-break space
    " ": " ",  # narrow no-break space
    " ": " ",  # thin space
    "﻿": "",  # BOM / zero width no-break space
    "​": "",  # zero width space
    "­": "",  # soft hyphen
}


def normalize_uk_text(text: str, canonicalize_apostrophe: bool = True) -> str:
    """Normalize a Ukrainian transcription.

    * Applies Unicode NFC so that decomposed letters (``и`` + U+0306,
      ``і`` + U+0308, ``г`` + U+0491...) become the single code points used by
      the character dictionary. Without this step the combining marks are
      silently dropped by the label encoder and ``й`` turns into ``и``.
    * Replaces U+02BC with U+2019 (visually identical apostrophes).
    * Replaces exotic spaces with a regular space and removes invisible chars.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str, got {}".format(type(text).__name__))
    text = unicodedata.normalize("NFC", text)
    for src, dst in _UNICODE_REPLACEMENTS.items():
        if not canonicalize_apostrophe and src == APOSTROPHE_MODIFIER:
            continue
        text = text.replace(src, dst)
    return text


def read_char_dict(dict_path: str) -> List[str]:
    """Read a PaddleOCR character dictionary exactly like ``BaseRecLabelDecode``.

    Lines are decoded as UTF-8 and only the line terminator is stripped.
    """
    if not os.path.isfile(dict_path):
        raise FileNotFoundError("character dict not found: {}".format(dict_path))
    chars = []
    with open(dict_path, "rb") as fin:
        for line in fin.readlines():
            chars.append(line.decode("utf-8").strip("\n").strip("\r\n"))
    return chars


def validate_char_dict(chars: Sequence[str]) -> Dict[str, list]:
    """Return problems of a character dictionary (empty lists mean "ok")."""
    seen = {}
    duplicates = []
    for idx, ch in enumerate(chars):
        if ch in seen:
            duplicates.append((ch, seen[ch], idx))
        else:
            seen[ch] = idx
    return {
        "duplicates": duplicates,
        "multi_char_entries": [
            (i, c) for i, c in enumerate(chars) if len(c) != 1 and c != ""
        ],
        "empty_entries": [i for i, c in enumerate(chars) if c == ""],
        "whitespace_entries": [
            (i, c) for i, c in enumerate(chars) if c and c.isspace()
        ],
        "non_nfc_entries": [
            (i, c) for i, c in enumerate(chars) if unicodedata.normalize("NFC", c) != c
        ],
    }


def missing_chars(text: str, charset: Iterable[str], use_space_char=True) -> List[str]:
    """Return the sorted unique characters of ``text`` absent from ``charset``."""
    charset = set(charset)
    if use_space_char:
        charset.add(" ")
    return sorted({c for c in text if c not in charset})


def is_uk_dict_char(ch: str) -> bool:
    """Whether a character belongs to the Ukrainian recognition dictionary."""
    if len(ch) != 1:
        return False
    if ch in UK_ALPHABET or ch in UK_EXTRA_SYMBOLS:
        return True
    return ch.isascii() and ch.isalnum()


def build_uk_char_list(base_chars: Sequence[str]) -> List[str]:
    """Select the Ukrainian subset of a base dictionary, preserving its order.

    Keeping the base order and only removing entries lets
    ``Global.pretrained_character_dict_path`` transfer every row of the output
    layers from the base model (see ``ppocr.utils.save_load``).
    """
    result = []
    seen = set()
    for ch in base_chars:
        if is_uk_dict_char(ch) and ch not in seen:
            result.append(ch)
            seen.add(ch)
    missing = [c for c in UK_ALPHABET + UK_EXTRA_SYMBOLS if c not in seen]
    if missing:
        raise ValueError(
            "base dictionary misses required characters: {}".format("".join(missing))
        )
    return result


def mixed_script_words(text: str) -> List[str]:
    """Return words that mix Cyrillic and Latin letters (e.g. Latin ``i`` in ``мiсто``)."""
    words = []
    for word in text.split():
        has_cyr = any("CYRILLIC" in unicodedata.name(c, "") for c in word)
        has_lat = any(c.isalpha() and "LATIN" in unicodedata.name(c, "") for c in word)
        if has_cyr and has_lat:
            words.append(word)
    return words
