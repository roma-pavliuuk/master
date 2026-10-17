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

"""Tests of the Ukrainian dataset tools in tools/ukrainian/."""

import importlib.util
import json
import os
import random
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "tools" / "ukrainian"
UK_DICT = REPO_ROOT / "ppocr" / "utils" / "dict" / "ppocrv5_uk_dict.txt"


def _load(name):
    spec = importlib.util.spec_from_file_location("uk_" + name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_dataset = _load("check_dataset")
split_dataset = _load("split_dataset")
gen_synth = _load("gen_synth_printed")
convert_rukopys = _load("convert_rukopys")
eval_rec = _load("eval_rec")
fetch_rukopys = _load("fetch_rukopys")
lexicon_correct = _load("lexicon_correct")


def _write_img(path, w=120, h=32):
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(img, "ok", (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


@pytest.fixture
def broken_dataset(tmp_path):
    (tmp_path / "img").mkdir()
    for i in range(6):
        _write_img(tmp_path / "img" / "{}.jpg".format(i))
    (tmp_path / "img" / "corrupt.jpg").write_bytes(b"not an image")
    _write_img(tmp_path / "img" / "tiny.jpg", w=5, h=4)
    lines = [
        "img/0.jpg\tҐанок",  # 1 ok
        "img/missing.jpg\tЄвропа",  # 2 missing image
        "img/corrupt.jpg\tЇжак",  # 3 corrupted
        "img/1.jpg\t",  # 4 empty label
        "img/2.jpg\tщось ы",  # 5 non-uk cyrillic + unknown char
        "img/3.jpg\t" + unicodedata.normalize("NFD", "їхній"),  # 6 non NFC
        "img/4.jpg\tпідʼїзд",  # 7 modifier apostrophe
        "img/0.jpg\tҐанок",  # 8 duplicate line
        "img/0.jpg\tГанок",  # 9 conflicting label
        "img/5.jpg\tÐ¿Ñ–Ð´",  # 10 mojibake
        "img/tiny.jpg\tа",  # 11 tiny image
        "img/1.jpg no tab",  # 12 bad format
        "img/5.jpg\t" + "дуже довгий рядок " * 3,  # 13 too long
        "img/4.jpg\tмiсто",  # 14 mixed script (Latin i)
    ]
    label = tmp_path / "train.txt"
    label.write_bytes(
        ("\n".join(lines) + "\n").encode("utf-8") + b"img/0.jpg\t\xff\xfe\n"
    )
    return tmp_path, label


class TestCheckDataset:
    def test_detects_all_problem_types(self, broken_dataset):
        root, label = broken_dataset
        parser_args = [
            "--label_file",
            str(label),
            "--data_dir",
            str(root),
            "--dict",
            str(UK_DICT),
            "--max_text_length",
            "25",
            "--report",
            str(root / "report.json"),
        ]
        assert check_dataset.main(parser_args) == 1
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))[
            str(label)
        ]
        found = set(report["issues"])
        expected = {
            "missing_image",
            "corrupted_image",
            "empty_label",
            "non_uk_cyrillic",
            "unknown_chars",
            "non_nfc",
            "modifier_apostrophe",
            "duplicate_line",
            "conflicting_label",
            "mojibake",
            "tiny_image",
            "bad_format",
            "too_long",
            "mixed_script",
            "invalid_utf8",
        }
        assert expected <= found, expected - found
        assert "ы (U+044B)" in report["unknown_chars"]
        assert report["num_records"] == 15

    def test_fix_output_normalizes_and_drops_errors(self, broken_dataset):
        root, label = broken_dataset
        out = root / "clean.txt"
        check_dataset.main(
            [
                "--label_file",
                str(label),
                "--data_dir",
                str(root),
                "--fix_output",
                str(out),
            ]
        )
        clean = out.read_text(encoding="utf-8").splitlines()
        assert "img/4.jpg\tпід’їзд" in clean  # U+02BC canonicalized
        assert "img/3.jpg\tїхній" in clean  # NFC
        assert not any("missing.jpg" in line or "corrupt.jpg" in line for line in clean)

    def test_clean_dataset_passes(self, tmp_path):
        _write_img(tmp_path / "a.jpg", w=200, h=32)
        (tmp_path / "l.txt").write_text(
            "a.jpg\tОб'єктно-орієнтоване\n", encoding="utf-8"
        )
        assert (
            check_dataset.main(
                ["--label_file", str(tmp_path / "l.txt"), "--data_dir", str(tmp_path)]
            )
            == 0
        )

    def test_leakage_between_splits(self, tmp_path):
        _write_img(tmp_path / "a.jpg")
        _write_img(tmp_path / "b.jpg")
        (tmp_path / "train.txt").write_text("a.jpg\tа\nb.jpg\tб\n", encoding="utf-8")
        (tmp_path / "test.txt").write_text("a.jpg\tа\n", encoding="utf-8")
        report_path = tmp_path / "r.json"
        rc = check_dataset.main(
            [
                "--label_file",
                str(tmp_path / "train.txt"),
                str(tmp_path / "test.txt"),
                "--data_dir",
                str(tmp_path),
                "--report",
                str(report_path),
            ]
        )
        assert rc == 1
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report[str(tmp_path / "test.txt")]["issues"]["leak"]["count"] == 1

    def test_det_mode(self, tmp_path):
        _write_img(tmp_path / "p.jpg", w=100, h=100)
        good = [
            {"transcription": "Їжак", "points": [[1, 1], [50, 1], [50, 20], [1, 20]]},
            {"transcription": "###", "points": [[1, 30], [50, 30], [50, 40], [1, 40]]},
        ]
        bad = [{"transcription": "x", "points": [[1, 1], [2, 2]]}]
        lines = [
            "p.jpg\t" + json.dumps(good, ensure_ascii=False),
            "p.jpg\t" + json.dumps(bad),
            "p.jpg\t{not json",
            "p.jpg\t"
            + json.dumps(
                [{"transcription": "а", "points": [[1, 1], [500, 1], [500, 9], [1, 9]]}]
            ),
        ]
        (tmp_path / "det.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        report_path = tmp_path / "r.json"
        check_dataset.main(
            [
                "--mode",
                "det",
                "--label_file",
                str(tmp_path / "det.txt"),
                "--data_dir",
                str(tmp_path),
                "--report",
                str(report_path),
            ]
        )
        issues = json.loads(report_path.read_text(encoding="utf-8"))[
            str(tmp_path / "det.txt")
        ]["issues"]
        assert {"bad_det_points", "bad_det_json", "det_points_outside"} <= set(issues)
        assert issues["bad_det_points"]["count"] == 1


class TestSplitDataset:
    def test_groups_are_not_split(self):
        records = [
            ("img/page{}__{}.jpg".format(p, k), "img/page{}__{}.jpg\tт".format(p, k))
            for p in range(20)
            for k in range(5)
        ]
        train, val, test = split_dataset.split_records(
            records, [0.8, 0.1, 0.1], "prefix", "__", seed=3
        )
        assert len(train) + len(val) + len(test) == 100
        pages = [{line.split("__")[0] for line in s} for s in (train, val, test)]
        assert (
            not (pages[0] & pages[1])
            and not (pages[0] & pages[2])
            and not (pages[1] & pages[2])
        )
        assert len(test) > 0 and len(val) > 0

    def test_deterministic(self):
        records = [("{}.jpg".format(i), str(i)) for i in range(50)]
        a = split_dataset.split_records(records, [0.5, 0.5], seed=1)
        b = split_dataset.split_records(records, [0.5, 0.5], seed=1)
        assert a == b

    def test_invalid_ratios(self):
        with pytest.raises(ValueError):
            split_dataset.split_records([("a", "a")], [0, 0])


def _ukrainian_font():
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]:
        if os.path.isfile(path) and gen_synth.font_covers(path):
            return path
    return None


class TestSyntheticGeneration:
    def test_sample_text_respects_length_and_dict(self):
        rng = random.Random(0)
        corpus = ["Об'єктно-орієнтоване програмування вивчають на другому курсі."]
        for _ in range(50):
            text = gen_synth.sample_text(rng, corpus, 25)
            assert 0 < len(text) <= 25

    def test_glyph_coverage_rejects_missing_glyphs(self):
        font = _ukrainian_font()
        if font is None:
            pytest.skip("no system font with Ukrainian glyphs")
        cov = gen_synth.GlyphCoverage()
        assert cov.covers(font, "Ґанок Їжак під’їзд")
        assert not cov.covers(font, "\U0010fffd")

    def test_generation_end_to_end(self, tmp_path):
        font = _ukrainian_font()
        if font is None:
            pytest.skip("no system font with Ukrainian glyphs")
        rc = gen_synth.main(
            [
                "--output_dir",
                str(tmp_path),
                "--num",
                "12",
                "--fonts",
                font,
                "--label_name",
                "train.txt",
                "--seed",
                "5",
            ]
        )
        assert rc == 0
        lines = (tmp_path / "train.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 12
        assert (
            check_dataset.main(
                [
                    "--label_file",
                    str(tmp_path / "train.txt"),
                    "--data_dir",
                    str(tmp_path),
                ]
            )
            == 0
        )


class TestTrainingDataPipeline:
    """Rendered Ukrainian words pass the real training transforms unchanged."""

    @pytest.mark.parametrize(
        "phrase",
        [
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
        ],
    )
    def test_render_encode_decode(self, tmp_path, phrase):
        font = _ukrainian_font()
        if font is None:
            pytest.skip("no system font with Ukrainian glyphs")
        from ppocr.data.imaug import create_operators, transform
        from ppocr.postprocess.rec_postprocess import CTCLabelDecode

        img = gen_synth.render_line(phrase, font, random.Random(0))
        path = tmp_path / "x.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        global_cfg = {
            "character_dict_path": str(UK_DICT),
            "use_space_char": True,
            "max_text_length": 40,
        }
        ops = create_operators(
            [
                {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
                {"RecResizeImg": {"image_shape": [3, 48, 320]}},
                {
                    "KeepKeys": {
                        "keep_keys": [
                            "image",
                            "label_ctc",
                            "label_gtc",
                            "length",
                            "valid_ratio",
                        ]
                    }
                },
            ],
            global_cfg,
        )
        out = transform(
            {"img_path": str(path), "label": phrase, "image": path.read_bytes()}, ops
        )
        assert out is not None
        image, label_ctc, _, length, _ = out
        assert image.shape == (3, 48, 320)
        decoder = CTCLabelDecode(character_dict_path=str(UK_DICT), use_space_char=True)
        assert decoder.decode([label_ctc[: int(length)]])[0][0] == phrase


class TestConvertRukopys:
    def test_convert(self, tmp_path):
        split = tmp_path / "test"
        (split / "images").mkdir(parents=True)
        page = np.full((400, 600, 3), 255, np.uint8)
        cv2.imwrite(str(split / "images" / "p1.jpg"), page)
        regions = [
            {
                "bbox": [10, 10, 300, 50],
                "type": "handwritten",
                "language": "uk",
                "legibility": "legible",
                "text": "Їжак підʼїздом",
            },
            {
                "bbox": [10, 60, 300, 100],
                "type": "handwritten",
                "language": "uk",
                "legibility": "illegible",
                "text": "",
            },
            {
                "bbox": [10, 110, 300, 150],
                "type": "handwritten",
                "language": "uk",
                "legibility": "legible",
                "text": "~~стара~~{нова} думка",
            },
            {
                "bbox": [10, 160, 300, 200],
                "type": "printed",
                "language": "uk",
                "legibility": "legible",
                "text": "Друкований",
            },
            {
                "bbox": [10, 210, 300, 250],
                "type": "image",
                "language": "uk",
                "legibility": "legible",
                "text": "",
            },
            {
                "bbox": [10, 260, 300, 300],
                "type": "formula",
                "language": "uk",
                "legibility": "legible",
                "text": "x²",
            },
            {
                "bbox": [10, 310, 300, 350],
                "type": "handwritten",
                "language": "uk",
                "legibility": "legible",
                "text": "Δ кут",
            },
        ]
        meta = [
            {
                "file_name": "images/p1.jpg",
                "image_width": 600,
                "image_height": 400,
                "regions": regions,
            },
            {
                "file_name": "images/not_downloaded.jpg",
                "image_width": 10,
                "image_height": 10,
                "regions": [],
            },
        ]
        (split / "metadata.jsonl").write_text(
            "\n".join(json.dumps(m, ensure_ascii=False) for m in meta), encoding="utf-8"
        )
        out = tmp_path / "out"
        det = tmp_path / "det.txt"
        assert (
            convert_rukopys.main(
                [
                    "--split_dir",
                    str(split),
                    "--output_dir",
                    str(out),
                    "--name",
                    "t",
                    "--det_label",
                    str(det),
                ]
            )
            == 0
        )
        rec = (out / "t.txt").read_text(encoding="utf-8").splitlines()
        assert rec == ["images/p1__0000.jpg\tЇжак під’їздом"]
        assert (out / "images" / "p1__0000.jpg").is_file()
        det_lines = det.read_text(encoding="utf-8").splitlines()
        assert len(det_lines) == 1
        boxes = json.loads(det_lines[0].split("\t")[1])
        texts = [b["transcription"] for b in boxes]
        assert texts == ["Їжак під’їздом", "###", "###", "Друкований", "###", "Δ кут"]

    def test_size_mismatch_is_reported(self, tmp_path):
        cv2.imwrite(str(tmp_path / "p.jpg"), np.zeros((10, 20, 3), np.uint8))
        with pytest.raises(ValueError):
            convert_rukopys.load_page(str(tmp_path / "p.jpg"), 30, 30)

    def test_select_pages_stratified(self):
        records = [
            {"file_name": str(i), "source": "school" if i % 3 else "archive"}
            for i in range(30)
        ]
        pages = fetch_rukopys.select_pages(records, 6, seed=0)
        assert len(pages) == 6
        assert sum(p["source"] == "archive" for p in pages) == 3


class TestEvalRec:
    def test_evaluate_with_pred_file(self, tmp_path):
        (tmp_path / "test.txt").write_text(
            "a.jpg\tпідʼїзд\nb.jpg\tЄвропа\nc.jpg\tукраїнська мова\n", encoding="utf-8"
        )
        (tmp_path / "pred.txt").write_text(
            "a.jpg\tпід’їзд\nb.jpg\tЕвропа\nc.jpg\tукраїнська мова\n", encoding="utf-8"
        )
        out = tmp_path / "out"
        assert (
            eval_rec.main(
                [
                    "--pred_file",
                    str(tmp_path / "pred.txt"),
                    "--label_file",
                    "hw=" + str(tmp_path / "test.txt"),
                    "--output_dir",
                    str(out),
                ]
            )
            == 0
        )
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))[
            "subsets"
        ]["hw"]
        # U+02BC == U+2019 in canonical mode, Є->Е is one substitution of 28 chars
        assert report["cer"] == pytest.approx(1 / 28)
        assert report["wer"] == pytest.approx(1 / 4)
        assert report["line_acc"] == pytest.approx(2 / 3)
        assert report["critical_char_recall"]["Є"] == {"total": 1, "recall": 0.0}

    def test_strict_apostrophe_mode(self):
        res = eval_rec.evaluate([("підʼїзд", "під’їзд")], apostrophe_mode="strict")
        assert res["cer"] == pytest.approx(1 / 7)
        res = eval_rec.evaluate([("під'їзд", "під’їзд")], apostrophe_mode="merge")
        assert res["cer"] == 0.0

    def test_missing_predictions_fail(self, tmp_path):
        (tmp_path / "test.txt").write_text("a.jpg\tа\n", encoding="utf-8")
        (tmp_path / "pred.txt").write_text("b.jpg\tа\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            eval_rec.main(
                [
                    "--pred_file",
                    str(tmp_path / "pred.txt"),
                    "--label_file",
                    str(tmp_path / "test.txt"),
                ]
            )

    def test_bad_label_file_fails(self, tmp_path):
        (tmp_path / "test.txt").write_text("no tab here\n", encoding="utf-8")
        with pytest.raises(ValueError):
            eval_rec.read_label_file(str(tmp_path / "test.txt"))


def test_convert_marker_filter_is_case_insensitive():
    assert convert_rukopys.clean_text("[Illegible] слово") is None
    assert convert_rukopys.clean_text("[illegible] слово") is None
    assert convert_rukopys.clean_text("Їжак  підʼїздом ") == "Їжак під’їздом"


class TestPipelineHelpers:
    @pytest.fixture(autouse=True)
    def _path(self, monkeypatch):
        monkeypatch.syspath_prepend(str(TOOLS / "examples"))

    @staticmethod
    def _line(text, x, y, w=100, h=20, angle_dy=0):
        poly = [[x, y], [x + w, y + angle_dy], [x + w, y + h + angle_dy], [x, y + h]]
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        return {
            "text": text,
            "score": 1.0,
            "poly": poly,
            "box": [min(xs), min(ys), max(xs), max(ys)],
        }

    def test_group_rows_on_skewed_table(self):
        import math
        from uk_ocr_common import group_rows, lines_to_text

        # table rotated by ~3 degrees: cells of one row drift down to the right
        slope = math.tan(math.radians(3))
        cells = []
        for r, row in enumerate(
            [["№", "Документ", "Дата"], ["1", "Заява", "24.08.2024"]]
        ):
            for c, text in enumerate(row):
                x = 50 + 300 * c
                cells.append(
                    self._line(
                        text,
                        x,
                        100 + 40 * r + int(x * slope),
                        w=250,
                        angle_dy=int(250 * slope),
                    )
                )
        random.Random(0).shuffle(cells)
        text = lines_to_text(group_rows(cells))
        assert text == "№\tДокумент\tДата\n1\tЗаява\t24.08.2024"

    def test_build_uk_ocr_validates_inputs(self, tmp_path):
        pytest.importorskip("paddlex")
        from uk_ocr_common import build_uk_ocr

        with pytest.raises(ValueError):
            build_uk_ocr("cursive")
        with pytest.raises(FileNotFoundError):
            build_uk_ocr("handwritten", rec_model_dir=str(tmp_path))

    def test_eval_pipeline_matching(self):
        eval_pipeline = _load("eval_pipeline")
        assert eval_pipeline.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
        assert eval_pipeline.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
        from ppocr.utils.ocr_error_rate import ErrorRateAccumulator

        gt = [
            {
                "transcription": "Ґанок",
                "points": [[0, 0], [100, 0], [100, 20], [0, 20]],
            },
            {
                "transcription": "Їжак",
                "points": [[0, 40], [100, 40], [100, 60], [0, 60]],
            },
            {
                "transcription": "###",
                "points": [[0, 80], [100, 80], [100, 100], [0, 100]],
            },
        ]
        pred = [
            ([[0, 1], [100, 1], [100, 21], [0, 21]], [0, 1, 100, 21], "Ганок"),
            ([[0, 81], [100, 81], [100, 99], [0, 99]], [0, 81, 100, 99], "ignored"),
        ]
        line, page, det = (
            ErrorRateAccumulator(),
            ErrorRateAccumulator(),
            {"tp": 0, "gt": 0, "pred": 0},
        )
        eval_pipeline.evaluate_page(gt, pred, 0.5, line, page, det)
        assert det == {"tp": 1, "gt": 2, "pred": 1}
        s = line.summary()
        # 1 substitution in "Ґанок" + 4 deletions of the missed "Їжак" over 9 chars
        assert s["cer"] == pytest.approx(5 / 9)
        assert page.summary()["num_ref_chars"] == len("Ґанок Їжак")


class TestDetectorCropGrouping:
    @staticmethod
    def _poly(x1, y1, x2, y2):
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def test_split_line_pieces_are_merged(self):
        # line 0 detected as two pieces, line 1 as one piece; boxes overlap vertically
        boxes = [[0, 0, 300, 60], [0, 40, 300, 100]]
        polys = [
            self._poly(5, 10, 140, 45),
            self._poly(150, 12, 295, 48),
            self._poly(5, 60, 290, 95),
        ]
        groups = convert_rukopys.assign_polys_to_lines(boxes, polys)
        assert sorted(len(g) for g in groups.values()) == [1, 2]
        assert len(groups[0]) == 2 and groups[1] == [polys[2]]
        quad = convert_rukopys.merge_polys(groups[0])
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        assert min(xs) <= 5 and max(xs) >= 295 and min(ys) <= 10 and max(ys) >= 48

    def test_partially_detected_line_is_skipped(self):
        boxes = [[0, 0, 300, 60]]
        polys = [self._poly(5, 10, 120, 45)]  # covers ~40% of the width
        assert convert_rukopys.assign_polys_to_lines(boxes, polys) == {}

    def test_piece_goes_to_line_containing_it(self):
        boxes = [[0, 0, 300, 60], [0, 30, 300, 120]]
        polys = [self._poly(5, 70, 295, 110)]  # fully inside the second box only
        assert list(convert_rukopys.assign_polys_to_lines(boxes, polys)) == [1]


def test_add_image_sizes(tmp_path):
    add_image_sizes = _load("add_image_sizes")
    _write_img(tmp_path / "a.jpg", w=150, h=30)
    (tmp_path / "l.txt").write_text("a.jpg\tЇжак\nmissing.jpg\tб\n", encoding="utf-8")
    add_image_sizes.main(
        [
            "--label_file",
            str(tmp_path / "l.txt"),
            "--data_dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "o.txt"),
        ]
    )
    assert (tmp_path / "o.txt").read_text(encoding="utf-8") == "a.jpg\tЇжак\t150\t30\n"


def test_convert_det_label_resume(tmp_path):
    split = tmp_path / "split"
    (split / "images").mkdir(parents=True)
    meta = []
    for name in ("p1", "p2"):
        cv2.imwrite(
            str(split / "images" / (name + ".jpg")),
            np.full((100, 200, 3), 255, np.uint8),
        )
        meta.append(
            {
                "file_name": "images/{}.jpg".format(name),
                "image_width": 200,
                "image_height": 100,
                "regions": [
                    {
                        "bbox": [10, 10, 150, 40],
                        "type": "handwritten",
                        "language": "uk",
                        "legibility": "legible",
                        "text": name,
                    }
                ],
            }
        )
    (split / "metadata.jsonl").write_text(
        "\n".join(json.dumps(m) for m in meta), encoding="utf-8"
    )
    det = tmp_path / "det.txt"
    # interrupted run: page 1 finished (marked text), a truncated line for page 2
    (tmp_path / "det.txt.partial").write_text(
        'images/p1.jpg\t[{"transcription": "DONE", "points": []}]\nimages/p2.jp',
        encoding="utf-8",
    )
    convert_rukopys.main(
        [
            "--split_dir",
            str(split),
            "--output_dir",
            str(tmp_path / "o"),
            "--no_rec",
            "--det_label",
            str(det),
            "--resume",
        ]
    )
    lines = det.read_text(encoding="utf-8").splitlines()
    assert [ln.split("\t")[0] for ln in lines] == ["images/p1.jpg", "images/p2.jpg"]
    assert "DONE" in lines[0]  # page 1 was not processed again
    assert json.loads(lines[1].split("\t")[1])[0]["transcription"] == "p2"
    assert not (tmp_path / "det.txt.partial").exists()


def test_convert_rec_resume(tmp_path):
    split = tmp_path / "split"
    (split / "images").mkdir(parents=True)
    meta = []
    for name in ("p1", "p2", "p3"):
        cv2.imwrite(
            str(split / "images" / (name + ".jpg")),
            np.full((100, 200, 3), 255, np.uint8),
        )
        meta.append(
            {
                "file_name": "images/{}.jpg".format(name),
                "image_width": 200,
                "image_height": 100,
                "regions": [
                    {
                        "bbox": [10, 10, 150, 40],
                        "type": "handwritten",
                        "language": "uk",
                        "legibility": "legible",
                        "text": "рядок " + name,
                    }
                ],
            }
        )
    (split / "metadata.jsonl").write_text(
        "\n".join(json.dumps(m) for m in meta), encoding="utf-8"
    )
    out = tmp_path / "o"
    out.mkdir()
    # interrupted run: page 1 finished, page 2 only half written
    (out / "r.txt.partial").write_text(
        "images/p1__0000.jpg\tDONE p1\n#DONE\timages/p1.jpg\n"
        "images/p2__0000.jpg\tрядок p2\n#DONE\timages/p2",
        encoding="utf-8",
    )
    argv = ["--split_dir", str(split), "--output_dir", str(out), "--name", "r"]
    convert_rukopys.main(argv + ["--resume"])
    lines = (out / "r.txt").read_text(encoding="utf-8").splitlines()
    assert lines == [
        "images/p1__0000.jpg\tDONE p1",  # page 1 was not processed again
        "images/p2__0000.jpg\tрядок p2",
        "images/p3__0000.jpg\tрядок p3",
    ]
    assert not (out / "r.txt.partial").exists()


def test_handwritten_mode_limits_detection_size(monkeypatch):
    pytest.importorskip("paddlex")
    monkeypatch.syspath_prepend(str(TOOLS / "examples"))
    import paddleocr
    import uk_ocr_common

    captured = {}

    class FakeOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(paddleocr, "PaddleOCR", FakeOCR)
    uk_ocr_common.build_uk_ocr("handwritten")
    assert captured["text_det_limit_side_len"] == 960
    assert captured["text_det_limit_type"] == "max"
    assert captured["text_det_unclip_ratio"] == 1.8
    captured.clear()
    uk_ocr_common.build_uk_ocr("handwritten", text_det_limit_side_len=1280)
    assert captured["text_det_limit_side_len"] == 1280
    captured.clear()
    uk_ocr_common.build_uk_ocr("printed")
    assert "text_det_limit_side_len" not in captured



@pytest.fixture
def lexicon_file(tmp_path):
    lex = tmp_path / "lex.txt"
    lex.write_text(
        "україна\nукраїни\nпід'їзд\nграматика\nлист\nлис\n", encoding="utf-8"
    )
    return lex


@pytest.fixture
def corrector(tmp_path, lexicon_file):
    freq = tmp_path / "freq.txt"
    freq.write_text("лист 50\nлис 3\nнеіснуюче 9\n", encoding="utf-8")
    return lexicon_correct.LexiconCorrector([str(lexicon_file)], str(freq), min_len=4)


def test_lexicon_corrects_single_edit_and_keeps_case(corrector):
    assert corrector.correct("Украіна") == "Україна"  # substitution
    assert corrector.correct("УКРАЇНАА") == "УКРАЇНА"  # deletion, upper case
    assert corrector.correct("грамтаика") == "граматика"  # transposition
    assert corrector.correct("граммматика") == "граммматика"  # distance 2: kept


def test_lexicon_keeps_known_short_and_mixed_words(corrector):
    text = "під'їзд під’їзд підʼїзд"  # every apostrophe variant is known
    assert corrector.correct(text) == text
    assert corrector.correct("кот") == "кот"  # shorter than min_len
    assert corrector.correct("Iкраїна") == "Iкраїна"  # Latin I: not Cyrillic
    assert corrector.correct("Украіна2") == "Украіна2"


def test_lexicon_ambiguity_and_frequency(corrector, lexicon_file):
    # "лиса": "лис" (deletion) and "лист" (substitution); frequency decides
    assert corrector.correct("лиса") == "лист"
    plain = lexicon_correct.LexiconCorrector([str(lexicon_file)], min_len=4)
    assert plain.correct("лиса") == "лиса"  # ambiguous without frequencies


def test_lexicon_keeps_word_hyphenated_at_line_end(corrector):
    assert corrector.correct("Украіна -") == "Украіна -"
    assert corrector.correct("Украіна - це") == "Україна - це"


def test_filter_silver_keeps_agreeing_lines():
    filter_silver = _load("filter_silver")
    rows = [
        ("a.jpg", "Україна", "Україна"),  # agree
        ("b.jpg", "Украіна", "Україна"),  # 1 error / 7 chars > 0.1
        ("c.jpg", "Двадцять друге", "Двадцтттт друге"),  # silver garbage
        ("d.jpg", "і", "і"),  # too short
    ]
    kept = list(filter_silver.select(rows, max_cer=0.1, min_chars=3))
    assert [k[0] for k in kept] == ["a.jpg"]
    assert kept[0][1] == "Україна"  # the silver label is kept
    loose = list(filter_silver.select(rows, max_cer=0.15, min_chars=3))
    assert [k[0] for k in loose] == ["a.jpg", "b.jpg"]


def test_filter_silver_lexicon_band():
    filter_silver = _load("filter_silver")
    lexicon = {"україна", "велика", "держава"}
    rows = [
        # (image, prediction, silver): 3 edits / 20 chars in both rows
        ("a.jpg", "Укрвїна велака дежава", "Україна велика держава"),
        ("b.jpg", "Україна велика держава", "Укрвїна велака дежава"),  # not words
    ]
    assert list(filter_silver.select(rows, max_cer=0.1)) == []
    kept = list(filter_silver.select(rows, 0.1, lexicon=lexicon, max_cer_lexicon=0.35))
    assert [k[0] for k in kept] == ["a.jpg"]
