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

"""Real recognition of rendered Ukrainian words by an actual model.

Unlike tests/ppocr/test_ukrainian_ocr.py these tests run a trained model on
images, so a failure means the MODEL misrecognizes the word. They download
the model and are therefore marked ``resource_intensive``:

    pytest tests/models/test_ukrainian_text_recognition.py -m resource_intensive
    UK_REC_MODEL_DIR=output/uk_printed_rec_ppocrv5/best_accuracy/inference \\
        pytest tests/models/test_ukrainian_text_recognition.py -m resource_intensive

Without UK_REC_MODEL_DIR the official eslav_PP-OCRv5_mobile_rec weights are used.
"""

import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

pytestmark = [pytest.mark.resource_intensive, pytest.mark.py38_incompatible]

PHRASES = [
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
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MODEL_NAME = "eslav_PP-OCRv5_mobile_rec"


def _render(text, size=40):
    font = ImageFont.truetype(FONT, size)
    l, t, r, b = font.getbbox(text)
    img = Image.new("RGB", (r - l + 2 * size // 2, b - t + size // 2), "white")
    ImageDraw.Draw(img).text(
        (size // 2 - l, size // 4 - t), text, font=font, fill="black"
    )
    return np.asarray(img)[:, :, ::-1].copy()  # BGR like cv2


@pytest.fixture(scope="module")
def recognizer():
    if not os.path.isfile(FONT):
        pytest.skip("DejaVuSans font is not installed")
    from paddleocr import TextRecognition

    model_dir = os.environ.get("UK_REC_MODEL_DIR")
    return TextRecognition(model_name=MODEL_NAME, model_dir=model_dir, device="cpu")


@pytest.mark.parametrize("phrase", PHRASES)
def test_recognizes_rendered_phrase(recognizer, phrase):
    result = recognizer.predict(_render(phrase))[0]
    assert result["rec_text"] == phrase


def test_model_dictionary_covers_ukrainian(recognizer):
    # character table that the loaded model actually decodes with
    chars = set(recognizer.paddlex_predictor.post_op.character)
    missing = [c for c in "ҐґЄєІіЇї'’" if c not in chars]
    assert missing == []


def test_invalid_model_dir_raises(tmp_path):
    from paddleocr import TextRecognition

    with pytest.raises(Exception):
        TextRecognition(model_name=MODEL_NAME, model_dir=str(tmp_path), device="cpu")
