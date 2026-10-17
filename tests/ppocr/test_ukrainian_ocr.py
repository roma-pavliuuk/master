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

"""Ukrainian OCR: dictionary, Unicode, label encoding/decoding, head weight
transfer and CER/WER metrics.

These tests check how Ukrainian characters flow through the code. They do NOT
measure recognition quality of a trained model on real images; see
tests/models/test_ukrainian_text_recognition.py for that.
"""

import copy
import unicodedata
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DICT_DIR = REPO_ROOT / "ppocr" / "utils" / "dict"
UK_DICT = DICT_DIR / "ppocrv5_uk_dict.txt"
ESLAV_DICT = DICT_DIR / "ppocrv5_eslav_dict.txt"
UK_PRINTED_CONFIG = (
    REPO_ROOT
    / "configs"
    / "rec"
    / "PP-OCRv5"
    / "multi_language"
    / "uk_PP-OCRv5_mobile_rec.yml"
)
UK_HANDWRITTEN_CONFIG = UK_PRINTED_CONFIG.with_name(
    "uk_handwritten_PP-OCRv5_mobile_rec.yml"
)

from ppocr.utils.ukrainian import (  # noqa: E402
    UK_ALPHABET,
    UK_CRITICAL_CHARS,
    build_uk_char_list,
    mixed_script_words,
    normalize_uk_text,
    read_char_dict,
    validate_char_dict,
)
from ppocr.utils.ocr_error_rate import (  # noqa: E402
    ErrorRateAccumulator,
    cer,
    char_edit_counts,
    wer,
    word_edit_counts,
)

UK_TEST_PHRASES = [
    "Ґанок",
    "ґудзик",
    "Європа",
    "єдність",
    "Україна",
    "українська мова",
    "Їжак",
    "їхній",
    "під'їзд",
    "під’їзд",
    "Об'єктно-орієнтоване програмування",
]


# --------------------------------------------------------------------------- #
# Dictionary
# --------------------------------------------------------------------------- #
class TestUkrainianDict:
    def test_alphabet_is_complete(self):
        assert len(UK_ALPHABET) == 66
        assert len(set(UK_ALPHABET)) == 66
        for ch in "ҐґЄєІіЇїЙйЩщЬь":
            assert ch in UK_ALPHABET

    def test_dict_file_is_utf8_without_bom_and_lf(self):
        raw = UK_DICT.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert b"\r" not in raw
        raw.decode("utf-8")

    def test_dict_contains_alphabet_digits_apostrophes(self):
        chars = set(read_char_dict(str(UK_DICT)))
        missing = [c for c in UK_ALPHABET + "0123456789" + "'’" if c not in chars]
        assert missing == []
        for ch in UK_CRITICAL_CHARS:
            assert ch in chars, ch
        for ch in "«»„“”–—-()[]{}.,:;!?/№%₴€$°§":
            assert ch in chars, ch

    def test_dict_has_no_duplicates_or_bad_entries(self):
        problems = validate_char_dict(read_char_dict(str(UK_DICT)))
        assert problems == {
            "duplicates": [],
            "multi_char_entries": [],
            "empty_entries": [],
            "whitespace_entries": [],
            "non_nfc_entries": [],
        }

    def test_dict_excludes_non_ukrainian_cyrillic(self):
        chars = set(read_char_dict(str(UK_DICT)))
        for ch in "ЁёЪъЫыЭэЎў":
            assert ch not in chars

    def test_dict_is_ordered_subset_of_eslav_dict(self):
        uk = read_char_dict(str(UK_DICT))
        eslav = read_char_dict(str(ESLAV_DICT))
        positions = [eslav.index(c) for c in uk]
        assert positions == sorted(positions)

    def test_dict_file_matches_generator(self):
        expected = build_uk_char_list(read_char_dict(str(ESLAV_DICT)))
        assert read_char_dict(str(UK_DICT)) == expected

    def test_build_rejects_incomplete_base(self):
        with pytest.raises(ValueError, match="misses required characters"):
            build_uk_char_list(list("абв"))

    def test_validate_detects_duplicates(self):
        problems = validate_char_dict(["а", "б", "а", "", " ", "ab"])
        assert problems["duplicates"] == [("а", 0, 2)]
        assert problems["empty_entries"] == [3]
        assert problems["whitespace_entries"] == [(4, " ")]
        assert problems["multi_char_entries"] == [(5, "ab")]

    def test_missing_dict_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_char_dict(str(tmp_path / "nope.txt"))

    @pytest.mark.parametrize("phrase", UK_TEST_PHRASES)
    def test_test_phrases_are_covered(self, phrase):
        chars = set(read_char_dict(str(UK_DICT))) | {" "}
        assert [c for c in phrase if c not in chars] == []

    def test_legacy_uk_dict_is_incomplete(self):
        # Documents why ppocrv5_uk_dict.txt is needed: the legacy uk_dict.txt
        # (and cyrillic_dict.txt) lack Ї/ї; uk_dict.txt also lacks Й/й/Щ.
        legacy = set(read_char_dict(str(DICT_DIR / "uk_dict.txt")))
        assert {"Ї", "ї", "Й", "й", "Щ"} - legacy == {"Ї", "ї", "Й", "й", "Щ"}


# --------------------------------------------------------------------------- #
# Unicode normalization
# --------------------------------------------------------------------------- #
class TestNormalization:
    def test_decomposed_letters_are_composed(self):
        decomposed = unicodedata.normalize("NFD", "Їжак йде, ґанок")
        assert decomposed != "Їжак йде, ґанок"
        assert normalize_uk_text(decomposed) == "Їжак йде, ґанок"

    def test_modifier_apostrophe_is_canonicalized(self):
        assert normalize_uk_text("підʼїзд") == "під’їзд"
        assert normalize_uk_text("підʼїзд", canonicalize_apostrophe=False) == "підʼїзд"

    def test_ascii_and_typographic_apostrophes_are_kept(self):
        assert normalize_uk_text("під'їзд") == "під'їзд"
        assert normalize_uk_text("під’їзд") == "під’їзд"

    def test_invisible_chars_and_spaces(self):
        assert normalize_uk_text("﻿Київ —​столиця") == "Київ —столиця"

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            normalize_uk_text(None)

    def test_mixed_script_words(self):
        # Latin "i" inside a Cyrillic word is a common annotation error
        assert mixed_script_words("мiсто Київ IBAN UA21") == ["мiсто"]


# --------------------------------------------------------------------------- #
# Label encoding / CTC decoding
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ctc_decoder():
    from ppocr.postprocess.rec_postprocess import CTCLabelDecode

    return CTCLabelDecode(character_dict_path=str(UK_DICT), use_space_char=True)


@pytest.fixture(scope="module")
def ctc_encoder():
    from ppocr.data.imaug.label_ops import CTCLabelEncode

    return CTCLabelEncode(75, character_dict_path=str(UK_DICT), use_space_char=True)


def _ctc_logits(indices, num_classes, repeat=2):
    """Frame sequence with repeats and blanks, as a CTC model would emit."""
    frames = [0]
    for i, idx in enumerate(indices):
        if i > 0 and indices[i - 1] == idx:
            frames.append(0)  # blank between identical chars
        frames.extend([idx] * repeat)
    frames.append(0)
    logits = np.full((1, len(frames), num_classes), 0.01, dtype=np.float32)
    for t, idx in enumerate(frames):
        logits[0, t, idx] = 0.95
    return logits


class TestEncodeDecode:
    def test_num_classes_matches_dict(self, ctc_decoder):
        dict_len = len(read_char_dict(str(UK_DICT)))
        # blank + dictionary + space
        assert len(ctc_decoder.character) == dict_len + 2
        assert ctc_decoder.character[0] == "blank"
        assert ctc_decoder.character[-1] == " "
        assert ctc_decoder.get_ignored_tokens() == [0]

    def test_train_and_infer_use_same_table(self, ctc_decoder, ctc_encoder):
        assert ctc_encoder.character == ctc_decoder.character

    def test_ukrainian_dict_path_is_not_reversed(self, ctc_decoder):
        assert ctc_decoder.reverse is False

    @pytest.mark.parametrize("phrase", UK_TEST_PHRASES)
    def test_roundtrip(self, phrase, ctc_encoder, ctc_decoder):
        data = ctc_encoder({"label": phrase})
        assert data is not None
        length = int(data["length"])
        assert length == len(phrase)  # no character was dropped
        indices = data["label"][:length].tolist()
        logits = _ctc_logits(indices, len(ctc_decoder.character))
        text, score = ctc_decoder(logits)[0]
        assert text == phrase
        assert 0.0 < score <= 1.0

    def test_unknown_modifier_apostrophe_is_dropped_without_normalization(
        self, ctc_encoder
    ):
        raw = "підʼїзд"
        data = ctc_encoder({"label": raw})
        assert int(data["length"]) == len(raw) - 1  # U+02BC silently dropped
        data = ctc_encoder({"label": normalize_uk_text(raw)})
        assert int(data["length"]) == len(raw)

    def test_decomposed_label_loses_breve_without_normalization(
        self, ctc_encoder, ctc_decoder
    ):
        raw = unicodedata.normalize("NFD", "йод")
        data = ctc_encoder({"label": raw})
        decoded = ctc_decoder.decode([data["label"][: int(data["length"])]])[0][0]
        assert decoded == "иод"
        data = ctc_encoder({"label": normalize_uk_text(raw)})
        assert ctc_decoder.decode([data["label"][: int(data["length"])]])[0][0] == "йод"

    def test_multilabel_encode_ctc_and_nrtr(self):
        from ppocr.data.imaug.label_ops import MultiLabelEncode

        enc = MultiLabelEncode(
            25,
            character_dict_path=str(UK_DICT),
            use_space_char=True,
            gtc_encode="NRTRLabelEncode",
        )
        out = enc({"label": "Об'єкт ґанку", "image": np.zeros((1, 1, 3))})
        assert int(out["length"]) == len("Об'єкт ґанку")
        gtc = out["label_gtc"].tolist()
        assert gtc[0] == 2 and 3 in gtc  # <s> ... </s>

    def test_too_long_label_is_rejected(self):
        from ppocr.data.imaug.label_ops import CTCLabelEncode

        enc = CTCLabelEncode(25, character_dict_path=str(UK_DICT), use_space_char=True)
        assert enc({"label": "а" * 26}) is None
        assert enc({"label": ""}) is None


class TestWordInfo:
    @pytest.mark.parametrize(
        "word", ["під'їзд", "під’їзд", "підʼїзд", "Об'єктно-орієнтоване"]
    )
    def test_apostrophe_and_hyphen_words_stay_whole(self, word):
        from ppocr.postprocess.rec_postprocess import BaseRecLabelDecode

        decoder = BaseRecLabelDecode()
        words, cols, states = decoder.get_word_info(
            word, np.ones(len(word), dtype=bool)
        )
        assert ["".join(w) for w in words] == [word]
        assert states == ["en&num"]

    def test_ukrainian_sentence_split(self):
        from ppocr.postprocess.rec_postprocess import BaseRecLabelDecode

        text = "Їхній під’їзд, Ґанок"
        words, _, _ = BaseRecLabelDecode().get_word_info(
            text, np.ones(len(text), dtype=bool)
        )
        assert ["".join(w) for w in words] == ["Їхній", "під’їзд", "Ґанок"]


# --------------------------------------------------------------------------- #
# Output layer transfer between dictionaries
# --------------------------------------------------------------------------- #
class TestClassTransfer:
    def test_transfer_rows_by_character(self):
        from ppocr.utils.save_load import (
            build_rec_class_transfer,
            transfer_class_params,
        )

        old_chars, new_chars = ["а", "б", "в"], ["в", "а", "ґ", "є"]
        layouts = build_rec_class_transfer(old_chars, new_chars, use_space_char=True)
        # CTC fc weight [in, classes]: blank, а, б, в, space
        old = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
        new = np.full((2, 6), -1.0, dtype=np.float32)  # blank, в, а, ґ, є, space
        value, copied, total = transfer_class_params(old, new, layouts)
        assert (copied, total) == (4, 6)
        np.testing.assert_array_equal(value[:, 0], old[:, 0])  # blank
        np.testing.assert_array_equal(value[:, 1], old[:, 3])  # в
        np.testing.assert_array_equal(value[:, 2], old[:, 1])  # а
        np.testing.assert_array_equal(value[:, 3], new[:, 3])  # ґ is new
        np.testing.assert_array_equal(value[:, 4], new[:, 4])  # є is new
        np.testing.assert_array_equal(value[:, 5], old[:, 4])  # space

    def test_same_size_dicts_are_not_remapped(self):
        from ppocr.utils.save_load import (
            build_rec_class_transfer,
            transfer_class_params,
        )

        layouts = build_rec_class_transfer(["а", "б"], ["б", "а"], use_space_char=True)
        assert (
            transfer_class_params(np.zeros((2, 4)), np.zeros((2, 4)), layouts) is None
        )

    def test_nrtr_layout_with_tail_class(self):
        from ppocr.utils.save_load import (
            build_rec_class_transfer,
            transfer_class_params,
        )

        layouts = build_rec_class_transfer(["а", "б"], ["б"], use_space_char=True)
        # Transformer vocab: blank,<unk>,<s>,</s>, а, б, space, tail -> 8 rows
        old = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
        new = np.zeros((7, 3), dtype=np.float32)
        value, copied, total = transfer_class_params(old, new, layouts)
        assert (copied, total) == (7, 7)
        np.testing.assert_array_equal(value[4], old[5])  # б
        np.testing.assert_array_equal(value[6], old[7])  # tail

    def test_unrelated_shape_is_not_transferred(self):
        from ppocr.utils.save_load import (
            build_rec_class_transfer,
            transfer_class_params,
        )

        layouts = build_rec_class_transfer(["а", "б"], ["б"], use_space_char=True)
        assert (
            transfer_class_params(np.zeros((3, 9)), np.zeros((3, 11)), layouts) is None
        )
        assert (
            transfer_class_params(np.zeros((4, 9)), np.zeros((3, 11)), layouts) is None
        )

    def test_config_without_option_disables_transfer(self):
        from ppocr.utils.save_load import build_rec_class_transfer_from_config

        assert (
            build_rec_class_transfer_from_config({"character_dict_path": str(UK_DICT)})
            is None
        )

    def test_config_with_missing_dict_raises(self, tmp_path):
        from ppocr.utils.save_load import build_rec_class_transfer_from_config

        with pytest.raises(FileNotFoundError):
            build_rec_class_transfer_from_config(
                {
                    "character_dict_path": str(UK_DICT),
                    "pretrained_character_dict_path": str(tmp_path / "missing.txt"),
                }
            )

    def test_real_model_head_transfer(self, tmp_path):
        """Build the PP-OCRv5 rec model with the eslav dict, save it, load it
        into the Ukrainian-dict model and check that class rows moved."""
        import paddle

        from ppocr.modeling.architectures import build_model
        from ppocr.utils.save_load import (
            build_rec_class_transfer_from_config,
            load_pretrained_params,
        )

        cfg = yaml.safe_load(UK_PRINTED_CONFIG.read_text(encoding="utf-8"))

        def make_model(dict_path):
            n = len(read_char_dict(str(dict_path))) + 2  # blank + space
            arch = copy.deepcopy(cfg["Architecture"])
            arch["Head"]["out_channels_list"] = {
                "CTCLabelDecode": n,
                "NRTRLabelDecode": n + 3,
            }
            return build_model(arch)

        paddle.seed(0)
        src = make_model(ESLAV_DICT)
        paddle.save(src.state_dict(), str(tmp_path / "src.pdparams"))
        dst = make_model(UK_DICT)
        global_cfg = {
            "character_dict_path": str(UK_DICT),
            "pretrained_character_dict_path": str(ESLAV_DICT),
            "use_space_char": True,
        }
        load_pretrained_params(
            dst,
            str(tmp_path / "src"),
            class_transfer=build_rec_class_transfer_from_config(global_cfg),
        )

        eslav = ["blank"] + read_char_dict(str(ESLAV_DICT)) + [" "]
        uk = ["blank"] + read_char_dict(str(UK_DICT)) + [" "]
        w_src = src.state_dict()["head.ctc_head.fc.weight"].numpy()
        w_dst = dst.state_dict()["head.ctc_head.fc.weight"].numpy()
        assert w_dst.shape[1] == len(uk)
        for ch in ["blank", "Ґ", "ї", "’", "'", "№", " "]:
            np.testing.assert_array_equal(
                w_dst[:, uk.index(ch)], w_src[:, eslav.index(ch)]
            )
        # backbone weights are loaded unchanged
        for key, value in src.state_dict().items():
            if key.startswith("backbone."):
                np.testing.assert_array_equal(
                    dst.state_dict()[key].numpy(), value.numpy()
                )
                break


# --------------------------------------------------------------------------- #
# CER / WER
# --------------------------------------------------------------------------- #
class TestErrorRates:
    def test_cer_formula_counts(self):
        # ref "кіт" -> hyp "кит": 1 substitution
        c = char_edit_counts("кіт", "кит")
        assert (c.substitutions, c.deletions, c.insertions, c.ref_len) == (1, 0, 0, 3)
        assert cer("кіт", "кит") == pytest.approx(1 / 3)

    def test_cer_deletion_and_insertion(self):
        c = char_edit_counts("під'їзд", "підїзд")
        assert (c.substitutions, c.deletions, c.insertions) == (0, 1, 0)
        c = char_edit_counts("Їжак", "Їжаки")
        assert (c.substitutions, c.deletions, c.insertions) == (0, 0, 1)
        assert cer("Їжак", "Їжаки") == pytest.approx(0.25)

    def test_cer_can_exceed_one(self):
        assert cer("я", "якась довга фраза") > 1.0

    def test_wer(self):
        c = word_edit_counts("українська мова державна", "українська мова державна")
        assert c.errors == 0
        assert wer("українська мова", "украiнська мова") == pytest.approx(0.5)
        assert wer("українська мова", "українська") == pytest.approx(0.5)
        assert wer("а б", "а в б") == pytest.approx(0.5)

    def test_empty_reference(self):
        assert cer("", "") == 0.0
        assert cer("", "а") == float("inf")

    def test_corpus_level_is_not_mean_of_lines(self):
        acc = ErrorRateAccumulator()
        acc.update("а", "б")  # 1 / 1
        acc.update("абвгґдеєжз", "абвгґдеєжз")  # 0 / 10
        s = acc.summary()
        assert s["cer"] == pytest.approx(1 / 11)
        assert s["line_acc"] == pytest.approx(0.5)
        assert s["num_ref_chars"] == 11

    def test_rec_error_rate_metric(self):
        from ppocr.metrics import build_metric

        metric = build_metric({"name": "RecErrorRateMetric", "main_indicator": "acc"})
        preds = [("Ґанок", 0.9), ("під'їзд", 0.8), ("Европа", 0.7)]
        labels = [("Ґанок", 1.0), ("під’їзд", 1.0), ("Європа", 1.0)]
        batch = metric((preds, labels))
        assert batch["acc"] == pytest.approx(1 / 3, abs=1e-4)
        result = metric.get_metric()
        assert result["cer"] == pytest.approx(2 / 18)
        assert result["wer"] == pytest.approx(2 / 3)
        assert result["acc"] == pytest.approx(1 / 3, abs=1e-4)
        assert result["char_acc"] == pytest.approx(1 - 2 / 18)
        assert metric.get_metric()["cer"] == 0.0  # reset after get_metric


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #
class TestConfigs:
    @pytest.mark.parametrize("path", [UK_PRINTED_CONFIG, UK_HANDWRITTEN_CONFIG])
    def test_rec_config_consistency(self, path):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        g = cfg["Global"]
        assert g["model_name"] == "eslav_PP-OCRv5_mobile_rec"  # registered in PaddleX
        assert (REPO_ROOT / g["character_dict_path"]).is_file()
        assert (REPO_ROOT / g["pretrained_character_dict_path"]).is_file()
        assert g["use_space_char"] is True
        nrtr = cfg["Architecture"]["Head"]["head_list"][1]["NRTRHead"]
        assert nrtr["max_text_length"] == g["max_text_length"]
        assert cfg["Metric"]["name"] == "RecErrorRateMetric"

    def test_handwritten_config_selects_best_by_cer(self):
        cfg = yaml.safe_load(UK_HANDWRITTEN_CONFIG.read_text(encoding="utf-8"))
        assert cfg["Metric"]["main_indicator"] == "char_acc"

    def test_handwritten_config_allows_long_lines(self):
        cfg = yaml.safe_load(UK_HANDWRITTEN_CONFIG.read_text(encoding="utf-8"))
        assert cfg["Global"]["max_text_length"] >= 70
        # aspect-ratio aware training must not squeeze long lines ...
        assert cfg["Train"]["dataset"]["ds_width"] is True
        assert cfg["Train"]["sampler"]["max_w"] >= 1280
        # ... and evaluation must resize like PaddleX inference (no squeeze)
        resize = cfg["Eval"]["dataset"]["transforms"][2]["RecResizeImg"]
        assert resize["eval_mode"] is True
        assert cfg["Eval"]["loader"]["batch_size_per_card"] == 1


# --------------------------------------------------------------------------- #
# Public API language mapping
# --------------------------------------------------------------------------- #
def test_uk_lang_resolves_to_eslav_model():
    pytest.importorskip("paddlex")
    from paddleocr import PaddleOCR

    engine = object.__new__(PaddleOCR)
    assert engine._get_ocr_model_names("uk", None) == (
        "PP-OCRv5_server_det",
        "eslav_PP-OCRv5_mobile_rec",
    )
