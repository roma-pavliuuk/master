# Українське OCR у PaddleOCR: друкований і рукописний текст

Цей документ описує доопрацювання PaddleOCR для розпізнавання української мови:
що змінено, як користуватися, як готувати дані, навчати й оцінювати моделі, а
також **що реально перевірено, а що ще ні**.

> Коротко про стан
>
> * Друкований текст: працює «з коробки» через `PaddleOCR(lang="uk")`, але офіційна
>   модель систематично плутає українські літери з російськими (`і`→`и`, `є`→`е`,
>   `ї`→`й`, `Ґ`→`Г`). Для якісного результату потрібне донавчання — конфігурація,
>   дані та скрипти для цього додані й перевірені коротким CPU-запуском.
> * Рукописний текст: офіційні ваги **не підходять** (CER ≈ 75 % на реальному
>   рукописі). Додано окрему конфігурацію, конвертер реального датасету RUKOPYS і
>   перевірений повний цикл донавчання, але повноцінне навчання потребує GPU.
> * Результати в розділі [Результати експериментів](#результати-експериментів)
>   отримані фактичними запусками; там же зазначено обсяг даних і обмеження.

## Зміст

1. [Що змінено](#що-змінено)
2. [Структура нових файлів](#структура-нових-файлів)
3. [Підтримувані символи](#підтримувані-символи)
4. [Архітектури та обґрунтування](#архітектури-та-обґрунтування)
5. [Встановлення](#встановлення)
6. [Ваги моделей](#ваги-моделей)
7. [Інференс: приклади](#інференс-приклади)
8. [Формат даних і підготовка датасету](#формат-даних-і-підготовка-датасету)
9. [Навчання](#навчання)
10. [Оцінювання якості](#оцінювання-якості)
11. [Результати експериментів](#результати-експериментів)
12. [Тести](#тести)
13. [Вимоги до обладнання](#вимоги-до-обладнання)
14. [Відомі обмеження](#відомі-обмеження)
15. [Наступні кроки](#наступні-кроки)

## Що змінено

Дослідження репозиторію (PaddleOCR 3.x з цього репозиторію, PaddleX 3.7.2,
PaddlePaddle 3.2.0) показало:

* `PaddleOCR(lang="uk")` вже вибирає `PP-OCRv5_server_det` +
  `eslav_PP-OCRv5_mobile_rec` (модель для російської, білоруської, української).
  Її словник `ppocrv5_eslav_dict.txt` (517 символів) містить увесь український
  алфавіт, включно з `Ґ ґ Є є І і Ї ї` та апострофами `'` і `’`.
* Застарілий словник `ppocr/utils/dict/uk_dict.txt` **неповний**: у ньому немає
  `Ї ї Й й Щ`; у `cyrillic_dict.txt` немає `Ї ї`. Ці файли не змінювалися (до них
  прив'язані старі ваги), натомість додано новий словник.
* Під час навчання символи, яких немає у словнику, **мовчки відкидаються**
  (`BaseRecLabelEncode.encode`), а мітки довші за `max_text_length` мовчки
  пропускаються (`MultiLabelEncode` повертає `None`). Звідси інструмент перевірки
  датасету.
* `load_pretrained_params` пропускав параметри з невідповідною формою, тобто
  при зміні словника вихідні шари CTC і NRTR ініціалізувалися випадково.
* У репозиторії не було CER/WER (лише `acc` і `norm_edit_dis`).

Зміни в наявному коді (мінімальні, зворотно сумісні):

| Файл | Зміна |
|---|---|
| `ppocr/utils/save_load.py` | Новий необов'язковий параметр `Global.pretrained_character_dict_path`: при зміні словника рядки вихідних шарів CTC/NRTR переносяться **посимвольно** (`build_rec_class_transfer`, `transfer_class_params`). Без параметра поведінка не змінилася. |
| `ppocr/postprocess/rec_postprocess.py` | `get_word_info` не розриває слова на апострофах `’` (U+2019) і `ʼ` (U+02BC), як раніше вже було для `'` (важливо для `return_word_box`). |
| `ppocr/metrics/rec_metric.py`, `ppocr/metrics/__init__.py` | Нова метрика `RecErrorRateMetric` = `RecMetric` + корпусні CER і WER. |

## Структура нових файлів

```
ppocr/utils/ukrainian.py                 # алфавіт, Unicode-нормалізація, перевірка словника
ppocr/utils/ocr_error_rate.py            # CER/WER: (S + D + I) / N
ppocr/utils/dict/ppocrv5_uk_dict.txt     # український словник (204 символи)
configs/rec/PP-OCRv5/multi_language/uk_PP-OCRv5_mobile_rec.yml              # друкований текст
configs/rec/PP-OCRv5/multi_language/uk_handwritten_PP-OCRv5_mobile_rec.yml  # рукописний текст
configs/det/PP-OCRv5/uk_handwritten_PP-OCRv5_server_det.yml                 # детекція рукописних рядків
tools/ukrainian/
  build_uk_dict.py        # відтворювана генерація словника
  check_dataset.py        # перевірка датасету (rec і det)
  split_dataset.py        # train/val/test із групуванням (сторінка, шрифт)
  gen_synth_printed.py    # синтетичні рядки друкованого тексту
  fetch_rukopys.py        # обмежене завантаження підмножини RUKOPYS
  convert_rukopys.py      # RUKOPYS -> кропи рядків (rec) і полігони (det)
  eval_rec.py             # CER / WER / точність рядків для моделі розпізнавання
  eval_pipeline.py        # наскрізна оцінка детекція + розпізнавання на сторінках
  data/uk_corpus_sample.txt
  examples/
    uk_ocr_common.py           # створення конвеєра, порядок читання
    ocr_printed.py             # приклад 1: друкований текст (фото, скан, PDF)
    ocr_handwritten.py         # приклад 2: рукописний текст
    ocr_batch_to_files.py      # приклади 3–5: кілька зображень, збереження, координати
    train_printed.sh           # приклад 6a: навчання (друкований)
    train_handwritten.sh       # приклад 6b: навчання (рукописний)
    evaluate.sh                # приклад 7: оцінювання на test
    make_sample_document.py    # синтетична сторінка документа з розміткою
tests/ppocr/test_ukrainian_ocr.py              # словник, Unicode, кодування/декодування, перенесення ваг, CER/WER
tests/tools/test_ukrainian_tools.py            # інструменти даних і оцінювання
tests/models/test_ukrainian_text_recognition.py  # реальне розпізнавання (resource_intensive)
docs/ukrainian_ocr.md                          # цей документ
```

## Підтримувані символи

`ppocr/utils/dict/ppocrv5_uk_dict.txt` — 204 символи + пробіл (`use_space_char: true`)
+ службовий `blank` для CTC = **206 класів CTC**:

* український алфавіт (66 літер): `АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ`
  `абвгґдеєжзиіїйклмнопрстуфхцчшщьюя`;
* цифри `0–9`, латиниця `A–Z a–z` (адреси e-mail, URL, IBAN, абревіатури, римські цифри);
* апострофи `'` (U+0027) і `’` (U+2019);
* лапки `« » „ “ ” ‘ ’ ‚ ‹ › "`, дефіс і тире `- ‐ ‑ – — −`, дужки `( ) [ ] { }`;
* символи для дат, адрес, сум: `. , : ; ! ? / \ № % ₴ € $ £ ¥ ¢ ° § © ® ™ @ # & * + = < > _ | ~ ^`,
  гравіс (U+0060), `… · • × ÷ ± ≤ ≥ ≠ ≈ ∙ ⋅ ² ³ ¹ ½ ¼ ¾`.

Особливості:

* Словник — **впорядкована підмножина** `ppocrv5_eslav_dict.txt` (генерується
  `tools/ukrainian/build_uk_dict.py`, перевіряється тестом). Тому всі 206 рядків
  вихідного шару CTC переносяться з офіційної моделі.
* Російські/білоруські `Ё ё Ъ ъ Ы ы Э э Ў ў` не включені — модель не може їх вивести.
* `ʼ` (U+02BC, рекомендований правописом апостроф) виглядає так само, як `’`, тож
  розрізнити їх на зображенні неможливо; у мітках він канонізується до `’`
  (`normalize_uk_text`, `check_dataset.py --fix_output`, `convert_rukopys.py`).
* Unicode NFC обов'язкова: розкладені `й` = `и` + U+0306 чи `ї` = `і` + U+0308
  енкодер перетворив би на `и` / `і` (комбінувальний знак відкидається). Тест
  `test_decomposed_label_loses_breve_without_normalization` це демонструє.
* Таблиця символів для навчання й інференсу одна: при експорті вона записується в
  `inference.yml` (`PostProcess.character_dict`), а PaddleX будує декодер саме з неї.

## Архітектури та обґрунтування

| Сценарій | Детекція | Розпізнавання | Ваги для старту |
|---|---|---|---|
| Друкований | `PP-OCRv5_server_det` (DB, PPHGNetV2_B4) | `SVTR_LCNet`: PPLCNetV3 + MultiHead (CTC + NRTR), вхід 48×320, `max_text_length` 25 | `eslav_PP-OCRv5_mobile_rec` |
| Рукописний | `PP-OCRv5_server_det`, за потреби донавчений (`uk_handwritten_PP-OCRv5_server_det.yml`) | той самий `SVTR_LCNet`, вхід 48×960, `max_text_length` 75, сильніші геометричні аугментації | `eslav_PP-OCRv5_mobile_rec` |

Чому так:

* Нова архітектура не потрібна: PP-OCRv5 `SVTR_LCNet` уже підтримує CTC з
  довільною шириною рядка, а `eslav` — єдина офіційна модель, навчена на
  українській кирилиці. Використання її ваг дає перенесення всіх шарів.
* Функція втрат — `MultiLoss` (CTCLoss + NRTRLoss), як у базовій моделі; на
  інференсі використовується лише CTC-гілка (`CTCLabelDecode`).
* Рукопис: у RUKOPYS 63 % рукописних рядків довші за 25 символів (медіана 31,
  99-й перцентиль ≈ 70), співвідношення сторін рядка — медіана ≈ 12, 90 % ≤ 19.
  З `max_text_length: 25` такі рядки мовчки відкидалися б, а при ширині 320
  CTC мав би лише 40 кроків. Тому 75 символів і ширина 960 (120 кроків CTC).
* `Global.model_name` у конфігураціях лишається `eslav_PP-OCRv5_mobile_rec`:
  PaddleX завантажує лише зареєстровані назви моделей і перевіряє збіг назви з
  `inference.yml`. Український словник береться з `inference.yml`.

## Встановлення

```bash
cd PaddleOCR
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip setuptools wheel
# CPU-збірка (для GPU див. https://www.paddlepaddle.org.cn/install/quick)
pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install -r requirements.txt "paddlex[ocr-core]>=3.7.0,<3.8.0" aiohttp pytest
pip install -e . --no-deps
```

Перевірено з Python 3.12, PaddlePaddle 3.2.0 (CPU), PaddleX 3.7.2 на Linux.
Нових залежностей не додано: інструменти використовують numpy, OpenCV, Pillow,
rapidfuzz, requests, PyYAML, які вже є в залежностях.

Для синтетики потрібні шрифти з українськими гліфами (на Ubuntu: DejaVu,
Liberation, Noto, URW base35 — `fonts-dejavu`, `fonts-liberation`, `fonts-noto-core`).

## Ваги моделей

* Інференс: офіційні моделі завантажуються PaddleX автоматично
  (`~/.paddlex/official_models/`). Щоб пропустити перевірку джерела:
  `export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`.
* Навчання: у конфігураціях `Global.pretrained_model` — URL офіційних
  навчальних ваг `eslav_PP-OCRv5_mobile_rec_pretrained.pdparams`
  (з `docs/version3.x/module_usage/text_recognition.md`), він завантажується
  автоматично. Офлайн:

  ```bash
  mkdir -p pretrain_models
  wget -P pretrain_models https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/eslav_PP-OCRv5_mobile_rec_pretrained.pdparams
  # далі: -o Global.pretrained_model=./pretrain_models/eslav_PP-OCRv5_mobile_rec_pretrained
  ```

* Детекція рукопису: `PP-OCRv5_server_det_pretrained.pdparams` (URL у конфігурації).

### Перенесення ваг при зміні словника

```yaml
Global:
  pretrained_model: <ваги моделі зі словником A>
  pretrained_character_dict_path: ./ppocr/utils/dict/ppocrv5_eslav_dict.txt  # словник A
  character_dict_path: ./ppocr/utils/dict/ppocrv5_uk_dict.txt                 # словник B
```

Параметри, форма яких відрізняється лише кількістю класів (CTC `fc`, NRTR
`embedding` і `tgt_word_prj`), заповнюються рядками відповідних символів;
нові символи отримують початкову ініціалізацію. У лозі:

```
Transferred head.ctc_head.fc.weight [120, 519] -> [120, 206]: 206/206 classes copied from pretrained params, 0 newly initialized
Transferred head.gtc_head.embedding.embedding.weight [523, 384] -> [210, 384]: 210/210 classes copied ...
```

Підтримуються класові розкладки голів CTC (`CTCLabelDecode`) і NRTR (`MultiHead` у PP-OCRv3–v5, `SVTR_LCNet` / `SVTR_HGNet`); для інших голів (SAR, Attention тощо) параметри з іншою формою, як і раніше, пропускаються з попередженням.
Якщо словники однакового розміру, але різні, параметри мають однакову форму й
вантажаться позиційно — виводиться попередження (розмір словника треба змінити).
`checkpoints` (продовження навчання) працює як раніше, без перенесення.

## Інференс: приклади

Усі приклади використовують лише публічний API (`paddleocr.PaddleOCR`,
`paddleocr.TextRecognition`) і запускаються з кореня репозиторію.

### 1. Друкований текст

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="uk",                          # PP-OCRv5_server_det + eslav_PP-OCRv5_mobile_rec
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in ocr.predict("ukrainian_document.png"):   # також .pdf: кожна сторінка окремо
    print(res["rec_texts"])
    res.save_to_json("output/")
    res.save_to_img("output/")
```

Донавчена модель друкованого тексту:

```python
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",   # зареєстрована назва архітектури
    text_recognition_model_dir="output/uk_printed_rec_ppocrv5/best_accuracy/inference",
    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
)
```

```bash
python tools/ukrainian/examples/make_sample_document.py --output_dir output/uk_demo
python tools/ukrainian/examples/ocr_printed.py output/uk_demo/uk_printed_document.png
python tools/ukrainian/examples/ocr_printed.py output/uk_demo/uk_printed_document.pdf
python tools/ukrainian/examples/ocr_printed.py output/uk_demo/uk_printed_document_photo.jpg \
    --rec_model_dir output/uk_printed_rec_ppocrv5/best_accuracy/inference
```

Для сканів зі значним поворотом увімкніть `use_doc_orientation_classify=True`,
для зігнутих сторінок — `use_doc_unwarping=True`, для перевернутих рядків —
`use_textline_orientation=True` (додаткові моделі PaddleX).

### 2. Рукописний текст

```python
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_server_det",
    text_detection_model_dir=None,   # або донавчений детектор
    text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
    text_recognition_model_dir="output/uk_handwritten_rec_ppocrv5/best_accuracy/inference",
    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
)
```

```bash
python tools/ukrainian/examples/ocr_handwritten.py page.jpg \
    --rec_model_dir output/uk_handwritten_rec_ppocrv5/best_accuracy/inference
# лише розпізнавання вже вирізаних рядків
python tools/ukrainian/examples/ocr_handwritten.py train_data/uk_handwritten/images \
    --rec_model_dir output/uk_handwritten_rec_ppocrv5/best_accuracy/inference --lines
```

### 3–5. Кілька зображень, збереження тексту, координати

```bash
python tools/ukrainian/examples/ocr_batch_to_files.py \
    output/uk_demo/uk_printed_document.png output/uk_demo/uk_printed_document_photo.jpg \
    output/uk_demo/uk_printed_document.pdf --output_dir output/uk_demo/results
```

Для кожної сторінки: `<name>.txt` (текст у порядку читання; клітинки одного рядка
таблиці розділені табуляцією; нахил сторінки компенсується), `<name>.lines.json`
(`text`, `score`, `box` = `[x_min, y_min, x_max, y_max]`, `poly`), нативні
`*_res.json` і візуалізація. У консоль виводяться координати кожної області.

### 6–7. Навчання та оцінювання

Див. [Навчання](#навчання) і [Оцінювання якості](#оцінювання-якості), а також
`tools/ukrainian/examples/train_printed.sh`, `train_handwritten.sh`, `evaluate.sh`.

## Формат даних і підготовка датасету

### Формат

Розпізнавання (`SimpleDataSet` / `MultiScaleDataSet`), UTF-8, один рядок на зразок:

```
images/0001.jpg	Об'єктно-орієнтоване програмування
images/0002.jpg	під’їзд
```

шлях до зображення відносно `data_dir`, табуляція, транскрипція одного рядка тексту.

Детекція, один рядок на сторінку:

```
images/page1.jpg	[{"transcription": "Їжак", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, {"transcription": "###", "points": [...]}]
```

`###` — область, яку слід ігнорувати (нерозбірливий текст, таблиця тощо).

Рекомендована структура:

```
train_data/uk_printed/       images/  train.txt  val.txt  test.txt
train_data/uk_handwritten/   images/  train.txt  val.txt  test.txt
train_data/rukopys_raw/{train,test}/  images/  metadata.jsonl  det_*.txt
```

### Як додати власні дані

1. Виріжте рядки тексту (або сторінки для детекції) і запишіть мітки у формат вище.
   Транскрибуйте **точно те, що видно**, включно з апострофами й регістром.
2. Перевірте й очистіть:

   ```bash
   python tools/ukrainian/check_dataset.py --label_file my/all.txt --data_dir my \
       --max_text_length 25 --hash_images --report my/report.json --fix_output my/all.clean.txt
   ```

   Виявляються: відсутні та пошкоджені зображення, некоректні рядки, порожні
   транскрипції, символи поза словником, невалідний UTF-8 і «кракозябри»
   (UTF-8, прочитаний як cp1251/latin-1), не-NFC Unicode, `ʼ`, російські літери,
   змішування кирилиці й латиниці в слові (`мiсто` з латинською `i`), дублікати
   записів і зображень, конфліктні мітки, задовгі мітки (> `max_text_length`),
   замалі зображення, незвичні пропорції, забагато символів для ширини (CTC),
   перетин між train/val/test (кілька `--label_file`). Режим `--mode det` перевіряє
   JSON і полігони. Код виходу 1 означає помилки. `--fix_output` записує
   нормалізовані мітки без зламаних рядків.
3. Розбийте на train/val/test **за групами** (сторінка, автор, шрифт), щоб рядки
   однієї сторінки не потрапили в різні вибірки:

   ```bash
   python tools/ukrainian/split_dataset.py --label_file my/all.clean.txt --output_dir my \
       --ratios 0.8 0.1 0.1 --group_by prefix   # або dir
   ```

### Синтетичні дані (лише друкований текст)

```bash
python tools/ukrainian/gen_synth_printed.py --output_dir train_data/uk_printed --num 6000 \
    --font_subset train --label_name train.txt --seed 0
python tools/ukrainian/gen_synth_printed.py --output_dir train_data/uk_printed --num 600 \
    --font_subset train --label_name val.txt --seed 1
python tools/ukrainian/gen_synth_printed.py --output_dir train_data/uk_printed --num 1000 \
    --font_subset test --label_name test.txt --seed 2
```

* Шрифти шукаються в `/usr/share/fonts` і залишаються лише ті, що мають гліфи всього
  українського алфавіту; для кожного рядка додатково перевіряється наявність гліфів
  (рядки, які б рендерились «тофу» □, відкидаються).
* Родини шрифтів детерміновано діляться на train/test (25 % родин лише в test) —
  test вимірює узагальнення на **невідомі шрифти**.
* Аугментації: розмір шрифту 18–64 px, кольори, фони (суцільні, градієнт, «папір»,
  кольоровий), нерівномірне освітлення, нахил ±3°, перспектива, низька роздільність,
  розмиття, шум, JPEG, лінії таблиць.
* Тексти: `tools/ukrainian/data/uk_corpus_sample.txt` (161 власне речення, усі
  літери, апострофи, дати, адреси, суми) + шаблони дат/сум/адрес/телефонів;
  власний корпус — `--corpus file1.txt file2.txt`.

Синтетичний друкований текст **не замінює** рукописний датасет.

### Рукописний датасет: RUKOPYS

Перевірене відкрите джерело: **RUKOPYS** (Ukrainian Catholic University),
<https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys>. За
карткою датасету: ліцензія CC BY 4.0; частина джерел (Національний диктант,
архів, КНТЕУ, ліцей) надана за угодою для академічних досліджень і публікацій —
перевірте актуальні умови перед використанням і цитуйте датасет. Формат:
сторінки + `metadata.jsonl` з регіонами `bbox [x1, y1, x2, y2]`, `type`
(`handwritten`, `printed`, `formula`, `table`, …), `legibility`, `text`;
спліти `train` (1 330 сторінок, ручна розмітка), `silver` (9 020, автоматична),
`test` (385). Маркери правок `~~закреслене~~`, `~~старе~~{нове}`, `[illegible]`.

```bash
python tools/ukrainian/fetch_rukopys.py --split train --max_pages 120 --output_dir train_data/rukopys_raw
python tools/ukrainian/fetch_rukopys.py --split test  --max_pages 40  --output_dir train_data/rukopys_raw
python tools/ukrainian/convert_rukopys.py --split_dir train_data/rukopys_raw/train \
    --output_dir train_data/uk_handwritten --name train_all --det_label train_data/rukopys_raw/train/det_train.txt
python tools/ukrainian/convert_rukopys.py --split_dir train_data/rukopys_raw/test \
    --output_dir train_data/uk_handwritten --name test --det_label train_data/rukopys_raw/test/det_test.txt
python tools/ukrainian/split_dataset.py --label_file train_data/uk_handwritten/train_all.txt \
    --output_dir train_data/uk_handwritten --names train val --ratios 0.9 0.1 --group_by prefix
```

Нічого не завантажується без явного запуску; `--max_pages` обмежує обсяг
(сторінка ≈ 2 МБ; весь `test` ≈ 0,7 ГБ). Конвертер ураховує EXIF-орієнтацію,
пропускає нерозбірливі регіони, регіони з маркерами правок і символами поза
словником, нормалізує Unicode; для детекції нечитабельні області, формули й
таблиці позначаються `###`. Офіційний `test` RUKOPYS використовується лише як test.

Інші відомі набори (не перевірені й не використані в цій роботі): Kaggle
«Ukrainian Handwritten Text» (за описом — рядки, CC BY-SA 4.0), HKR (рукописний
казахсько-російський, не український). Перед використанням перевірте ліцензію й формат.

## Навчання

Загальна схема (див. також `tools/ukrainian/examples/train_*.sh`):

```bash
# друкований текст (GPU)
python tools/train.py -c configs/rec/PP-OCRv5/multi_language/uk_PP-OCRv5_mobile_rec.yml
# рукописний текст (GPU)
python tools/train.py -c configs/rec/PP-OCRv5/multi_language/uk_handwritten_PP-OCRv5_mobile_rec.yml
# кілька GPU
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c <config>
# CPU: додайте -o Global.use_gpu=False
```

* **Попередньо навчені ваги**: `Global.pretrained_model` (URL або локальний шлях без `.pdparams`).
* **Адаптація вихідного шару**: `Global.pretrained_character_dict_path` (див. вище).
* **Checkpoint**: кожні `save_epoch_step` епох і `latest` після кожної епохи в
  `Global.save_model_dir` (`.pdparams`, `.pdopt`, `.states`).
* **Продовження перерваного навчання**:
  `-o Global.checkpoints=./output/uk_printed_rec_ppocrv5/latest` (відновлює ваги, оптимізатор, епоху).
* **Validation і найкраща модель**: оцінка кожні `eval_batch_step` ітерацій на
  `Eval.dataset` (`RecErrorRateMetric`: acc, CER, WER); найкраща за `acc`
  зберігається в `best_accuracy/` і, оскільки `uniform_output_enabled: true`,
  одразу експортується в `best_accuracy/inference/`.
* **Експорт вручну**:

  ```bash
  python tools/export_model.py -c configs/rec/PP-OCRv5/multi_language/uk_PP-OCRv5_mobile_rec.yml \
      -o Global.use_gpu=False Global.pretrained_model=./output/uk_printed_rec_ppocrv5/best_accuracy/best_accuracy \
      Global.save_inference_dir=./output/uk_printed_rec_ppocrv5/inference
  ```

* **Інференс після навчання**: `text_recognition_model_dir=<…>/inference` (див. приклади).
* **Test-набір** у конфігураціях не використовується: `Train` → `train.txt`,
  `Eval` → `val.txt`; `test.txt` лише для `eval_rec.py` / `eval_pipeline.py`.

Детекція рукописних рядків. Стандартний детектор розбиває рукописні рядки на
шматки, зливає сусідні й пропускає частину тексту.

1. Розмітка: прямокутні рамки RUKOPYS перетинаються на похилих рядках, тому
   де можливо вони замінюються щільними повернутими чотирикутниками цілих рядків
   (об'єднані шматки, знайдені детектором):

   ```bash
   python tools/ukrainian/convert_rukopys.py --split_dir train_data/rukopys_raw/train \
       --output_dir output/tmp_det_prep --name unused --crop_source det \
       --refine_det_label --no_rec --det_label train_data/rukopys_raw/train/det_train_refined.txt
   ```

   Повна підготовка (розмітка, поділ train/val за сторінками, базова оцінка;
   можна переривати — повторний запуск продовжує з файлів `*.partial`):
   `bash tools/ukrainian/examples/train_handwritten_det_prepare.sh` (≈ 2 год на CPU).
2. Навчання: на CPU реалістичний лише мобільний детектор
   (`configs/det/PP-OCRv5/uk_handwritten_PP-OCRv5_mobile_det.yml`, ≈ 13 с на
   ітерацію з батчем 8, ≈ 35 хв на епоху). Серверний
   (`uk_handwritten_PP-OCRv5_server_det.yml`) на CPU у ≈ 7 разів повільніший,
   для нього потрібен GPU.

   ```bash
   bash tools/ukrainian/examples/train_handwritten_det.sh   # повторний запуск продовжує з latest
   ```

3. Використання донавченого детектора:
   `--det_model_name PP-OCRv5_mobile_det --det_model_dir <...>/best_accuracy/inference`
   у `eval_pipeline.py`, `ocr_handwritten.py`, `ocr_batch_to_files.py`.

## Оцінювання якості

Метрики (`ppocr/utils/ocr_error_rate.py`):

```
CER = (S + D + I) / N      S, D, I — заміни, видалення, вставки символів мінімального
                           (Левенштейн) вирівнювання; N — символів у еталоні
WER = те саме для слів (розділених пробілами)
Точність рядка = частка рядків, розпізнаних без жодної помилки
```

Рахуються на рівні корпусу (`ΣS+ΣD+ΣI / ΣN`), а не як середнє по рядках; можуть
перевищувати 1 при великій кількості вставок. Обидві сторони нормалізуються
однаково: NFC, `ʼ`→`’` (режим `--apostrophe_mode canonical`; `strict` — без
змін, `merge` — усі апострофи рівні), стиснення пробілів; `--ignore_case` за бажанням.

```bash
# модель розпізнавання на окремих test-наборах, окремо друкований і рукописний
python tools/ukrainian/eval_rec.py --model_dir <inference_dir> --output_dir output/eval/x \
    --label_file printed=train_data/uk_printed/test.txt handwritten=train_data/uk_handwritten/test.txt
# готові передбачення з іншої системи: "<image>\t<text>"
python tools/ukrainian/eval_rec.py --pred_file preds.txt --label_file test.txt
# повний конвеєр на сторінках: детекція P/R/H (IoU 0.5) + CER/WER рядків і сторінок
python tools/ukrainian/eval_pipeline.py --det_label train_data/rukopys_raw/test/det_test.txt \
    --data_dir train_data/rukopys_raw/test --mode handwritten --rec_model_dir <inference_dir>
```

`eval_rec.py` порівнює з **сирими** транскрипціями (на відміну від `tools/eval.py`,
де еталон уже відфільтровано словником), рахує recall критичних символів
`Ґ ґ Є є І і Ї ї ' ’`, зберігає `report.json` і `pred_*.txt`, попереджає, якщо
рядків менше 100.

## Результати експериментів

Усі числа нижче отримані фактичними запусками 10.10.2026 на CPU (без GPU).
Звіти: `output/eval/*/report.json`, логи навчання: `output/uk_*_rec_cpu.log`
(каталоги `output/` і `train_data/` не входять у git).

### Дані

| Набір | Походження | Рядків | Призначення |
|---|---|---|---|
| `uk_printed/train.txt` | синтетика, 19 родин шрифтів | 6 000 | навчання (друкований) |
| `uk_printed/val.txt` | синтетика, ті самі шрифти, інше зерно | 600 | валідація |
| `uk_printed/test.txt` | синтетика, **6 інших родин шрифтів** | 1 000 | test (друкований) |
| `uk_handwritten/train.txt` | RUKOPYS train, 104 сторінки | 1 821 | навчання (рукописний) |
| `uk_handwritten/val.txt` | RUKOPYS train, інші 11 сторінок | 164 | валідація |
| `uk_handwritten/test.txt` | RUKOPYS **test**, 39 сторінок із придатними рядками | 647 | test (рукописний) |
| `rukopys_raw/test/det_test.txt` | ті самі 40 сторінок test | 40 сторінок | наскрізна оцінка |
| `uk_demo/uk_printed_document*` | синтетична сторінка документа (абзац + таблиця) | 23 рядки | наскрізна оцінка |

Синтетичний друкований test має невідомі моделі шрифти, але той самий корпус
текстів і генератор, тому він оптимістичніший за реальні документи. Рукописний
test — справжні рукописи інших сторінок (окремий спліт RUKOPYS). Обсяг рукописних
даних (≈ 1,8 тис. рядків) **замалий** для повноцінного навчання.

### Проведене навчання (короткі CPU-запуски, не повноцінне навчання)

| Модель | Старт | Дані | Режим | Час |
|---|---|---|---|---|
| друкована | `eslav_PP-OCRv5_mobile_rec`, перенесення голови на `ppocrv5_uk_dict.txt` (206/206 класів CTC) | 6 000 синт. рядків | 2 епохи, 374 ітерації, батч 32, 48×320, lr 2e-4 (cosine) | ≈ 1 год 31 хв |
| рукописна | те саме | 1 821 рядок RUKOPYS | 3 епохи, 342 ітерації, батч 16, 48×960, lr 1e-4 (cosine) | ≈ 2 год 26 хв |

Команди — `tools/train.py` з конфігураціями `uk_PP-OCRv5_mobile_rec.yml` /
`uk_handwritten_PP-OCRv5_mobile_rec.yml` і перевизначеннями
`Global.use_gpu=False Global.epoch_num=2|3 "Train.sampler.scales=[[320,48]]"|"[[960,48]]"
Train.sampler.first_bs=32|16 Optimizer.lr.warmup_epoch=0.1|0.2`. Детектор не донавчався.

Отримані inference-моделі (локально, не в git):
`output/uk_printed_rec_cpu/best_accuracy/inference` і
`output/uk_handwritten_rec_cpu/best_accuracy/inference` — їх можна передати як
`--rec_model_dir` / `text_recognition_model_dir` у прикладах замість шляхів за
замовчуванням `output/uk_*_rec_ppocrv5/...`.

**Рукописна v2** (додатковий запуск): ще 200 сторінок RUKOPYS train (усього
320 завантажених; train — 5 297 рядків з 293 сторінок, val і test ті самі), старт
з ваг рукописної моделі, 2 епохи, 662 ітерації, батч 16, lr 1e-4, вибір найкращої
моделі за `char_acc`, ≈ 4 год 50 хв на CPU (`output/uk_handwritten_rec_cpu_v2`).

Валідація під час навчання:

| Модель | Крок | acc | CER | WER |
|---|---|---|---|---|
| друкована | 150 | 93,3 % | 1,07 % | 3,70 % |
| друкована | 300 (найкраща) | 96,8 % | 0,54 % | 1,85 % |
| рукописна | 150 | 1,8 % | 37,4 % | 80,8 % |
| рукописна | 300 (збережена як best) | 1,8 % | 43,0 % | 86,8 % |
| рукописна v2 | 200 | 0,0 % | 48,5 % | 99,2 % |
| рукописна v2 | 400 (best) | 0,6 % | 44,8 % | 100 % |
| рукописна v2 | 600 | 0,6 % | 45,3 % | 104,9 % |

У рукописному запуску найкраща модель обиралася за `acc` (`>=`), а точність
рядків однакова (1,8 %), тож модель кроку 300 перезаписала кращу за CER модель
кроку 150. Після цього рукописна конфігурація переведена на
`main_indicator: char_acc` (= 1 − CER); повторний запуск із цим виправленням
не проводився.

### Розпізнавання рядків (`tools/ukrainian/eval_rec.py`, test-набори)

| Модель | Друкований test (1 000 рядків): CER / WER / точн. рядків | Рукописний test (647 рядків): CER / WER / точн. рядків |
|---|---|---|
| офіційна `eslav_PP-OCRv5_mobile_rec` | 9,21 % / 34,5 % / 47,7 % | 74,9 % / 96,5 % / 0,8 % |
| офіційна + український словник, **без навчання** | 9,17 % / 34,3 % / 47,7 % | — |
| донавчена друкована | **1,48 % / 4,95 % / 91,5 %** | 56,8 % / 93,3 % / 0,8 % |
| донавчена рукописна (best) | 17,7 % / 53,8 % / 30,2 % | **42,1 % / 86,6 % / 1,6 %** |
| донавчена рукописна (остання епоха) | 19,5 % / 61,1 % / 23,2 % | 42,7 % / 87,3 % / 1,7 % |
| рукописна v2: + 2 епохи на 5 297 рядках (best, крок 400) | — | 40,5 % / 93,6 % / 0,9 % |
| рукописна v2 (остання епоха) | — | 41,3 % / 94,4 % / 0,6 % |

Recall критичних символів на друкованому test (офіційна → донавчена друкована):
`і` 0,48 → 0,99; `І` 0,36 → 0,99; `ґ` 0,38 → 1,00; `Ґ` 0,58 → 1,00; `є` 0,44 → 0,88;
`Є` 0,40 → 0,76; `ї` 0,67 → 0,99; `’` 0,00 → 0,94; `'` 0,86 → 0,94.

Найчастіші помилки офіційної моделі на друкованому test: `і`→`и` (184 рази),
`і`→латинська `i` (46), `р`→`p` (35), `а`→`a` (31), `І`→`I` (20), `ї`→`й` (20),
`№`→`N` (16), `’`→`'` (16), `є`→`е` (15).

Швидкість інференсу на CPU (PaddleX): ≈ 9–16 мс на друкований рядок, ≈ 30–48 мс
на рукописний (довший вхід); значення залежать від завантаження CPU під час вимірювання.

### Контрольні фрази (`tests/models/test_ukrainian_text_recognition.py`)

Чистий рендер шрифтом DejaVu Sans, вимагається точний збіг:

* офіційна модель: 6 з 11 фраз **не розпізнано** (`Европа`, `едність`, `Украйна`,
  `украйнська мова`, `під'їзд` замість `під’їзд`, `Об'єктно-ориєнтоване …`);
* донавчена друкована: 11 з 11 фраз розпізнано (13/13 тестів).

### Повний конвеєр: детекція + розпізнавання (`tools/ukrainian/eval_pipeline.py`)

| Дані | Розпізнавання | Детекція P / R / H | CER / WER рядків | CER / WER сторінки |
|---|---|---|---|---|
| демо-документ (23 рядки) | офіційна | 1,00 / 1,00 / 1,00 | 3,92 % / 19,3 % | 3,70 % / 19,3 % |
| демо-документ (23 рядки) | донавчена друкована | 1,00 / 1,00 / 1,00 | **0,26 % / 1,75 %** | **0,25 % / 1,75 %** |
| RUKOPYS test, 40 сторінок | офіційна | 0,50 / 0,66 / 0,57 | 89,9 % / 113 % | 70,4 % / 94,5 % |
| RUKOPYS test, 40 сторінок | донавчена рукописна | 0,50 / 0,66 / 0,57 | 74,6 % / 111 % | **45,3 % / 86,6 %** |
| RUKOPYS test, 40 сторінок | рукописна v2 | 0,50 / 0,66 / 0,57 | 79,2 % / 119 % | 47,3 % / 92,6 % |

Детекція на рукописних сторінках оцінювалась за IoU ≥ 0,5 відносно прямокутних
рамок RUKOPYS (похилі рядки → рамки перетинаються), тому P/R є нижньою оцінкою;
CER/WER сторінки від зіставлення не залежать. Демо-документ — одна синтетична
сторінка, це перевірка працездатності, а не статистично значуща оцінка.

### Рукописна модель v5: повний train RUKOPYS, кропи детектора (11–12.10.2026)

Найкраща рукописна модель: `output/uk_handwritten_rec_cpu_v5/best_accuracy/inference`.

Що змінилося порівняно з v1/v2:

* **Дані:** весь `train` RUKOPYS (1 329 сторінок завантажено, ≈ 3 ГБ). Train —
  19 515 рядків із 1 243 сторінок; val — 413 рядків із 35 інших сторінок; test —
  ті самі 40 сторінок офіційного `test`.
* **Кропи детектора** (`convert_rukopys.py --crop_source det --gt_fallback`):
  детектор `PP-OCRv5_server_det` з тими самими параметрами, що й у конвеєрі,
  часто розбиває рукописний рядок на кілька шматків. Тому всі шматки одного
  рядка розмітки об'єднуються в один щільний, вирівняний за нахилом кроп. Пряме
  зіставлення 1:1 давало кропи частини рядка з міткою всього рядка. Рядки, які
  детектор покрив менш ніж на 85 % ширини, беруться з рамки розмітки, якщо вона
  не охоплює кілька рядків. Відкинуто 548 таких «абзаців» під міткою одного рядка.
  Разом 13 878 кропів детектора та 5 637 кропів розмітки.
* **М'якші аугментації:** `tia_prob` 0,6 → 0,2 (TIA-деформації обрізали частину
  рядка, і мітка ставала неправильною), `reverse_prob` 0,4 → 0,1.
* **Групування за пропорціями** (`ds_width: true`, `max_w: 1280`): рядки однієї
  пропорції йдуть одним батчем без доповнення до 960, навчання в 2,3 раза швидше.
  Це потребує міток із шириною й висотою (`tools/ukrainian/add_image_sizes.py`).
* **Виправлена розбіжність навчання й інференсу.** Модель, навчена без
  доповнення, на рядках, доповнених до ширини 960 (так раніше робили і валідація,
  і `inference.yml`), видавала сміття: CER валідації 69,6 %, хоча за правильної
  передобробки 24 %. Тепер валідація й експорт використовують
  `RecResizeImg: {image_shape: [3, 48, 160], eval_mode: true}`: висота 48, ширина
  за пропорцією рядка, доповнення лише до 160. Це збігається з PaddleX, який
  сортує рядки за пропорціями перед розпізнаванням.
* **Навчання:** старт із ваг v4 (крок 400; v4 — перший запуск на цих даних, який
  ще мав неправильну валідацію), 2 епохи × 1 220 ітерацій, батч 16, lr 5e-5
  (cosine), ≈ 1,4 зразка/с, ≈ 9 год на CPU з паузою між епохами (продовження з
  `Global.checkpoints=.../latest/latest`).

Валідація v5 (413 рядків кропів детектора):

| Епоха / крок | CER | WER | Точність рядків |
|---|---|---|---|
| 1 / 400 | 25,5 % | 69,7 % | 6,1 % |
| 1 / 800 | 24,5 % | 67,8 % | 5,1 % |
| 1 / 1200 | 23,1 % | 65,5 % | 5,8 % |
| 2 / 400 | 22,6 % | 64,5 % | 7,7 % |
| 2 / 800 | 22,1 % | 63,2 % | 7,0 % |
| 2 / 1200 (best) | **21,9 %** | 63,7 % | 8,0 % |

Test (рядки) — `eval_rec.py --batch_size 1`:

| Модель | `test_det`: кропи детектора, 489 рядків (CER / WER / точн. рядків) | `test`: рамки розмітки, 647 рядків (CER / WER) |
|---|---|---|
| офіційна | 69,8 % / 95,8 % / 0,6 % | 74,9 % / 96,5 % |
| рукописна v1 | 32,8 % / 81,3 % / 1,0 % | 42,1 % / 86,6 % |
| рукописна v2 | 34,5 % / 87,1 % / 1,0 % | 40,5 % / 93,6 % |
| v5, епоха 1 | 20,6 % / 61,9 % / 4,1 % | 28,0 % / 67,2 % |
| **v5, епоха 2 (best)** | **19,7 % / 60,5 % / 4,9 %** | **27,1 % / 65,7 %** |

Recall `і` на `test_det`: 0,87 (v5) проти 0,00 в офіційної моделі.

Повний конвеєр на 40 рукописних сторінках (`eval_pipeline.py`):

| Розпізнавання | CER сторінки | WER сторінки |
|---|---|---|
| офіційна | 70,4 % | 94,5 % |
| рукописна v1 | 45,3 % | 86,6 % |
| **рукописна v5** | **38,9 %** | **76,9 %** |

Висновок: на окремих рядках v5 удвічі-втричі точніша за офіційну модель, проте
на цілих сторінках CER (38,9 %) помітно вищий, ніж на рядках (19,7–27,1 %). Цю
різницю дає детектор, який не донавчався: він розбиває й зливає рукописні рядки
та пропускає частину тексту (recall за IoU ≥ 0,5 — 0,66). Наступне вузьке місце —
детекція.

### Рукописна модель v6: донавчання на GPU (Kaggle, 15.10.2026)

Найкраща рукописна модель: `output/uk_handwritten_rec_gpu_v6/best_accuracy/inference`.

* Безкоштовний GPU Kaggle (NVIDIA Tesla T4, 15 ГБ), приватні датасети з кодом і
  даними (ті самі 19 515 рядків train, 413 val, test без змін), скрипт
  `tools/ukrainian/kaggle/train_rec_v6.py` (запуск через Kaggle CLI, див. `tools/ukrainian/kaggle/README.md`).
* Старт із ваг v5, 12 епох, батч 10 (батчі 32/64 у fp32 і 24 зі змішаною
  точністю не вміщуються в пам'ять T4), lr 7e-5 (cosine), ≈ 15 хв на епоху,
  ≈ 23 зразки/с (у ≈ 15 разів швидше за CPU цієї машини), разом 3 год 10 хв.
* Помилка, яку варто знати: значення lr у вигляді `7e-05` у `-o ...` YAML-парсер
  PaddleOCR читає як рядок, і навчання падає на старті — передавайте десятковий
  запис (`0.00007`).

Валідація (413 рядків кропів детектора):

| Епоха | 1 | 3 | 5 | 6 | 9 (best) | 12 |
|---|---|---|---|---|---|---|
| CER | 21,8 % | 18,4 % | 17,6 % | 16,5 % | **15,9 %** | 16,1 % |
| WER | 63,7 % | 56,9 % | 55,0 % | 53,9 % | **51,6 %** | 52,0 % |

Test (рядки, `eval_rec.py --batch_size 1`):

| Модель | `test_det` (489): CER / WER / точн. рядків | `test` (647): CER / WER / точн. рядків |
|---|---|---|
| v5 | 19,7 % / 60,5 % / 4,9 % | 27,1 % / 65,7 % / 5,3 % |
| **v6** | **15,8 % / 52,0 % / 8,8 %** | **21,3 % / 57,4 % / 11,0 %** |

Повний конвеєр (серверний детектор + розпізнавання) на 40 рукописних сторінках:

| Розпізнавання | CER сторінки | WER сторінки |
|---|---|---|
| офіційна | 70,4 % | 94,5 % |
| v5 | 38,9 % | 76,8 % |
| **v6** | **34,3 %** | **70,1 %** |

Друкований синтетичний test моделлю v6 — 14,6 % CER (v5: 10,2 %), тобто для
друкованого тексту й надалі використовуйте окрему друковану модель.

Детектор: короткий (2 епохи) CPU-донавчання мобільного детектора погіршило
конвеєр (CER сторінки 49,9 % проти 46,3 % з офіційним мобільним і 38,9 % із
серверним), тому використовується офіційний серверний детектор.

### Детекція рукописних сторінок: роздільність і донавчений серверний детектор (16–17.10.2026)

**Роздільність детекції.** Фото й скани рукописних сторінок мають 3000–4000 px.
Конвеєр PaddleOCR за замовчуванням подає детектору сторінку майже в повному
розмірі (`text_det_limit_side_len=64`, `limit_type="min"`, до 4000 px), і на
такому масштабі детектор розбиває та пропускає рукописні рядки. Зменшення
сторінки до ~1000 px за найбільшою стороною різко покращує результат без
жодного навчання. Тому рукописний режим прикладів (`uk_ocr_common.build_uk_ocr`)
тепер використовує `text_det_limit_side_len=960, text_det_limit_type="max"`
(можна перевизначити; у `eval_pipeline.py` — `--det_limit_side_len`).
Друкований режим не змінено.

**Донавчений серверний детектор.** `PP-OCRv5_server_det` донавчено на Kaggle
(T4): 1 293 сторінки train RUKOPYS із уточненою розміткою, 30 епох, батч 8,
lr 0,0001, ≈ 5,5 хв на епоху, 3 год 20 хв разом; сторінки завантажуються з
Hugging Face прямо в ноутбуці. Найкраща епоха 18 за Hmean на 35 val-сторінках
(валідація при найбільшій стороні 2048 px): офіційний 0,676 (P 0,62 / R 0,75) →
донавчений 0,707 (P 0,76 / R 0,66). Модель:
`output/uk_handwritten_det_server/best_accuracy/inference`.

Повний конвеєр на 40 тестових рукописних сторінках, розпізнавання v6:

| Детектор | Роздільність (найбільша сторона) | Hmean детекції | CER сторінки | WER сторінки |
|---|---|---|---|---|
| офіційний | повна (до 4000 px, як раніше) | 0,569 | 34,3 % | 70,1 % |
| офіційний | 2560 px | 0,646 | 31,2 % | 69,0 % |
| офіційний | 2048 px | 0,750 | 28,9 % | 68,8 % |
| офіційний | 1600 px | 0,816 | 26,9 % | 65,7 % |
| офіційний | 1280 px | 0,817 | 26,9 % | 66,3 % |
| **офіційний** | **960 px (новий типовий)** | 0,852 | 26,6 % | **64,2 %** |
| офіційний | 768 px | 0,845 | 26,6 % | 65,8 % |
| донавчений | повна | 0,640 | 53,2 % | 81,4 % |
| донавчений | 2048 px | 0,821 | 35,0 % | 73,8 % |
| донавчений | 1280 px | **0,927** | 26,2 % | 68,6 % |
| **донавчений** | **960 px** | 0,914 | **25,6 %** | 67,7 % |

Висновки:

* Найбільший приріст дала правильна роздільність детекції: CER сторінки
  34,3 % → 26,6 % з офіційним детектором.
* Донавчений детектор знаходить рядки значно краще (Hmean 0,91–0,93 проти 0,85),
  дає найнижчий CER сторінки (25,6 %), але вищий WER (67,7 % проти 64,2 %).
  Імовірна причина — модель розпізнавання v6 навчалась на кропах офіційного
  детектора, а донавчений вирізає рядки інакше (щільніше). Наступний крок —
  переріз рядків донавченим детектором і коротке донавчання розпізнавання.

### Рукописна модель v7 і фінальний конвеєр (17.10.2026)

**v7** — розпізнавання, донавчене з v6 на рядках, вирізаних **донавченим
детектором на 960 px**, тобто так само, як під час використання
(`convert_rukopys.py --crop_source det --gt_fallback --det_model_dir
output/uk_handwritten_det_server/best_accuracy/inference --det_limit_side_len 960
--det_limit_type max`). Цей детектор покриває більше рядків: 19 116 кропів
детектора замість 14 176 на train і 600 замість 489 на test. Train — 19 858
рядків (без 34 val-сторінок), val — 522 рядки, test — 600. Навчання на Kaggle
T4: 6 епох із v6, батч 10, lr 0,00005, ≈ 2 год; найкраща епоха 4 (val CER
15,3 %, WER 53,5 %). Модель: `output/uk_handwritten_rec_gpu_v7/best_accuracy/inference`.

Рядки (`eval_rec.py --batch_size 1`):

| Модель | нові кропи детектора (600): CER / WER | старі кропи детектора (489): CER / WER | рамки розмітки (647): CER / WER |
|---|---|---|---|
| v6 | 17,8 % / 56,6 % | 15,8 % / 52,0 % | 21,3 % / 57,4 % |
| v7 | **16,7 % / 53,4 %** | 16,1 % / 52,0 % | 22,2 % / 57,9 % |

**Фінальний конвеєр для рукопису: донавчений детектор (960 px) + v7.**
40 тестових сторінок RUKOPYS:

| Конвеєр | CER сторінки | WER сторінки |
|---|---|---|
| `PaddleOCR(lang="uk")` за замовчуванням | 70,4 % | 94,5 % |
| офіційний детектор (повна роздільність) + v6 | 34,3 % | 70,1 % |
| офіційний детектор 960 px + v6 | 26,6 % | 64,2 % |
| донавчений детектор 960 px + v6 | 25,6 % | 67,7 % |
| офіційний детектор 960 px + v7 | 26,6 % | 63,5 % |
| **донавчений детектор 960 px + v7** | **24,3 %** | **62,5 %** |

За типом джерела (по 10 сторінок):

| Джерело | `PaddleOCR(lang="uk")`: CER / WER | фінальний конвеєр: CER / WER |
|---|---|---|
| диктанти | 61,0 % / 93,9 % | **17,9 % / 47,2 %** |
| шкільні зошити | 85,1 % / 99,0 % | **27,5 % / 65,2 %** |
| університетські конспекти | 82,0 % / 97,9 % | **33,3 % / 81,2 %** |
| архівні документи 1919–1935 | 67,7 % / 91,3 % | **25,1 % / 69,2 %** |

**Повний тест RUKOPYS — усі 385 сторінок** (`det_test_full.txt`; жодна не використовувалась у навчанні;
40 з них — ті, на яких порівнювались конвеєри вище, решта 345 раніше не оцінювались):

| Конвеєр | детекція P / R / H | CER рядків | CER сторінки | WER сторінки |
|---|---|---|---|---|
| `PaddleOCR(lang="uk")` за замовчуванням | 0,412 / 0,662 / 0,508 | 96,9 % | 69,2 % | 93,5 % |
| **донавчений детектор 960 px + v7** | **0,919 / 0,930 / 0,924** | **25,2 %** | **22,9 %** | **60,0 %** |

Результат на 385 сторінках (22,9 %) узгоджується з оцінкою на 40 сторінках
(24,3 %), тобто вибір конвеєра на малій вибірці не був випадковим.

**Пороги постобробки детектора (DB).** Підібрано сіткою на 35 val-сторінках
(`eval_pipeline.py --det_thresh --det_box_thresh --det_unclip_ratio`, 18
комбінацій thresh ∈ {0,2; 0,3; 0,4}, box_thresh ∈ {0,5; 0,6}, unclip ∈ {1,5; 1,8;
2,1}). Розширення рамок unclip 1,8 стабільно краще за типове 1,5 (рамка
захоплює виносні елементи літер), 2,1 — вже гірше (зачіпає сусідні рядки).
На val: 25,7 % → 24,2 % CER сторінки. Перевірка трьох кращих комбінацій на
всіх 385 тестових сторінках:

| thresh / box_thresh / unclip | детекція H | CER рядків | CER сторінки | WER сторінки |
|---|---|---|---|---|
| 0,3 / 0,6 / 1,5 (типові PaddleOCR) | 0,924 | 25,2 % | 22,9 % | 60,0 % |
| 0,3 / 0,6 / 1,8 | 0,918 | 24,3 % | 22,7 % | 58,4 % |
| 0,4 / 0,6 / 1,8 | 0,919 | 24,1 % | 22,6 % | 58,2 % |
| **0,4 / 0,5 / 1,8** (найкраща на val) | 0,916 | **23,7 %** | **22,2 %** | **58,3 %** |

Ці значення тепер типові для рукописного режиму (`HANDWRITTEN_DET_LIMIT` у
`tools/ukrainian/examples/uk_ocr_common.py`).

**Словникова постобробка** (`tools/ukrainian/lexicon_correct.py`). Слово, якого
немає у словнику, замінюється словом зі словника на відстані редагування 1
(вставка, видалення, заміна, перестановка сусідніх літер), якщо вибір
однозначний: кандидат один або найчастотніший за частотним списком. Слова,
коротші за 4 літери, слова з цифрами/латиницею й фрагмент слова, перенесеного
дефісом у кінці рядка, не змінюються; регістр зберігається. Джерела, **не
пов’язані з RUKOPYS**: словник `/usr/share/dict/ukrainian` (пакет Debian/Ubuntu
`wukrainian`, ≈ 1,56 млн словоформ проєкту dict_uk) і частотний список
FrequencyWords `uk_50k.txt` (OpenSubtitles 2018,
https://github.com/hermitdave/FrequencyWords, завантажено в
`train_data/lexicon/`). Частоти з транскрипцій RUKOPYS train давали ще кращий
результат на val (WER 44,8 %), але **не використані**: диктанти в train і test
мають однакові тексти, тож це було б підглядання.

| Набір | без корекції: CER / WER | з корекцією: CER / WER |
|---|---|---|
| рукопис, val-рядки (522) | 15,3 % / 53,5 % | 14,4 % / 45,9 % |
| рукопис, test-рядки детектора (600) | 16,7 % / 53,4 % | 15,6 % / 44,8 % |
| рукопис, test-рядки розмітки (647) | 22,2 % / 57,9 % | 21,4 % / 51,1 % |
| **рукопис, повний конвеєр, 385 сторінок** | 22,2 % / 58,3 % | **21,1 % / 50,7 %** |
| друк, реальні рядки RUKOPYS (557), друкована модель | 8,4 % / 34,2 % | 8,8 % / 35,8 % |

Для рукопису корекція виправляє переважно одиночні помилки в словах (WER
−7…8 п.п.). Для друкованого тексту вона **шкодить**: реальні друковані рядки
RUKOPYS — це машинопис 1920-х років з тодішнім правописом («Презідія»,
«Плятформа», «Інститута»), який сучасний словник «виправляє». Тому корекція
вимикається за замовчуванням і вмикається лише явно:

```bash
python tools/ukrainian/examples/ocr_handwritten.py сторінка.jpg \
    --det_model_dir output/uk_handwritten_det_server/best_accuracy/inference \
    --rec_model_dir output/uk_handwritten_rec_gpu_v7/best_accuracy/inference \
    --lexicon /usr/share/dict/ukrainian --lexicon_freq train_data/lexicon/uk_50k.txt
```

Обмеження: власні назви поза словником можуть бути «виправлені» (у тестових
рядках «Маск» → «Макс»), а архівні тексти зі старим правописом — осучаснені.

```bash
python tools/ukrainian/examples/ocr_handwritten.py сторінка.jpg \
    --det_model_dir output/uk_handwritten_det_server/best_accuracy/inference \
    --rec_model_dir output/uk_handwritten_rec_gpu_v7/best_accuracy/inference
```

**Демо для доповіді:** `tools/ukrainian/examples/make_demo.py` будує для
тестових сторінок візуалізації «офіційна модель / наша модель», тексти й
`summary.md` з CER/WER по сторінках (приклад — `output/demo`, 7 сторінок із
чотирьох типів джерел; одна сторінка з хімічними формулами виключена, бо
формули не входять в еталон). Типові помилки, що лишаються: зайві дефіси
(лінії зошита читаються як штрихи), сплутування схожих рукописних літер
(«Треба» → «Греба»), складні почерки конспектів.

### Реальний друкований текст (17.10.2026)

Відкритого набору сучасних друкованих українських документів із вирівняним
еталонним текстом знайти не вдалося, тому реальний друкований test узято з
**test-спліту RUKOPYS**: області типу `printed` (559 у мові `uk`; 557 рядків
після фільтрів) на 35 сторінках, — переважно машинописні й друковані архівні
документи 1919–1935 рр. (українська еміграція в Празі: потертий шрифт,
розрядка, архівна орфографія на кшталт «орґанізації»), а також друковані
частини диктантів і шкільних бланків. Друкована модель на цих даних не
навчалась (вона навчена лише на синтетиці).

```bash
python tools/ukrainian/convert_rukopys.py --split_dir train_data/rukopys_raw/test \
    --output_dir train_data/uk_printed_real --name test_printed --types printed
python tools/ukrainian/eval_rec.py --batch_size 1 --model_dir <модель> \
    --label_file real_printed=train_data/uk_printed_real/test_printed.txt
```

| Модель | CER | WER | Рядків без помилок | recall `і` |
|---|---|---|---|---|
| офіційна `eslav_PP-OCRv5_mobile_rec` | 11,8 % | 46,1 % | 15,6 % | 0,37 |
| **друкована (донавчена на синтетиці)** | **8,4 %** | **34,2 %** | **35,9 %** | **0,97** |
| рукописна v7 (для порівняння) | 19,1 % | 67,7 % | 4,5 % | 0,78 |

Найчастіші помилки друкованої моделі тут: пропущені пробіли (машинописна
розрядка), латинська `I` в еталоні римських цифр проти кириличної `І`, `-`
замість `—`, а також майже повний провал на дрібному або нахиленому тексті фото
підручників. Синтетичний test (1,5 % CER) суттєво оптимістичніший за реальний
(8,4 %).

**Зайві дефіси в рукописі.** У v6 траплялися дефіси перед словами («-госщу»),
імовірно, через лінії зошита. На валідації v7 дефісів, навпаки, менше, ніж в
еталоні (130 проти 153), а шаблон «пробіл-дефіс-слово» трапляється 6 разів
(в еталоні 4), тому окрему постобробку не додано — вона погіршила б результат.

### Висновки

* Друкований текст: донавчення лише на синтетиці усуває «російський ухил»:
  синтетичний test із невідомими шрифтами — CER 9,2 % → 1,5 %, реальні
  друковані/машинописні рядки RUKOPYS — 11,8 % → 8,4 % (рядків без помилок
  15,6 % → 35,9 %). Для сучасних документів потрібен окремий реальний test.
* Рукописний текст: донавчення на 1,8 тис. реальних рядків знижує CER з 74,9 % до
  42,1 %, але модель **ще не придатна для практичного використання**. Потрібно
  повноцінне навчання на значно більшому обсязі даних (GPU).
* Додаткове навчання v2 на втричі більшій кількості рядків дало змішаний
  результат: CER на рядках розмітки трохи кращий (42,1 % → 40,5 %), але модель
  частіше вставляє зайві символи й пробіли (вставок пробілів 189 → 436), тому WER
  гірший, а на повному конвеєрі CER сторінки погіршився (45,3 % → 47,3 %).
  Імовірна причина — прямокутні кропи RUKOPYS містять шматки сусідніх рядків, а на
  інференсі рядки вирізає детектор інакше. Це підтвердилося у v5 (див. вище).
* Найкращий рукописний конвеєр — донавчений серверний детектор (960 px) +
  розпізнавання v7: CER сторінки 24,3 %, WER 62,5 % на 40 сторінках
  (`PaddleOCR(lang="uk")`: 70,4 % / 94,5 %); на всіх 385 тестових сторінках
  22,9 % / 60,0 % (офіційний 69,2 % / 93,5 %), з підібраними порогами детектора
  22,2 % / 58,3 %, а зі словниковою постобробкою — 21,1 % / 50,7 %. Текст здебільшого читабельний, але не безпомилковий;
  найкраще — диктанти (CER 17,9 %), найважче — університетські конспекти (33,3 %).
* Одна модель не покриває обидва сценарії: рукописне донавчання погіршує
  друкований текст (1,5 % → 17,7 % CER) — використовуйте окремі моделі.


## Тести

```bash
# швидкі тести (без завантаження моделей)
python -m pytest tests/ppocr/test_ukrainian_ocr.py tests/tools/test_ukrainian_tools.py -q
# реальне розпізнавання контрольних слів моделлю (завантажує модель)
python -m pytest tests/models/test_ukrainian_text_recognition.py -m resource_intensive -q
UK_REC_MODEL_DIR=output/uk_printed_rec_ppocrv5/best_accuracy/inference \
    python -m pytest tests/models/test_ukrainian_text_recognition.py -m resource_intensive -q
```

Покрито: завантаження й валідність словника (UTF-8, без дублікатів, NFC, відповідність
генератору, підмножина eslav у тому ж порядку), наявність `Ґ ґ Є є І і Ї ї`, `'` `’`,
кількість класів (blank + словник + пробіл), однакова таблиця для енкодера й
декодера, CTC-кодування/декодування всіх контрольних фраз («Ґанок», «ґудзик»,
«Європа», «єдність», «Україна», «українська мова», «Їжак», «їхній», «під'їзд»,
«під’їзд», «Об'єктно-орієнтоване програмування»), втрати через `ʼ` і розкладений
Unicode без нормалізації, межі слів з апострофами, перенесення ваг (у т. ч. на
реальній моделі PP-OCRv5), CER/WER з відомими S/D/I, метрика навчання,
конфігурації, мапінг `lang="uk"`, інструменти даних (усі типи проблем), синтетика,
проходження відрендерених українських слів через реальні трансформації навчання,
конвертер RUKOPYS, порядок читання на нахиленій таблиці, наскрізні метрики.

Тести словника й кодування **не** доводять якість розпізнавання — для цього
`tests/models/test_ukrainian_text_recognition.py` і метрики на test-наборах.

## Вимоги до обладнання

Виміряно на цій машині (16 потоків x86-64 CPU, 30 ГБ RAM, без GPU):

| Операція | Швидкість |
|---|---|
| Інференс розпізнавання (PaddleX, статичний граф), рядок 48×320 | ≈ 9–10 мс/рядок |
| Інференс розпізнавання, рукописний рядок (довший) | ≈ 30 мс/рядок |
| Навчання `SVTR_LCNet`, 48×320, батч 32, CPU | ≈ 2 зразки/с (≈ 15 с/ітерацію) |
| Навчання 48×960, батч 16, CPU | ≈ 0,6 зразка/с (≈ 26 с/ітерацію) |
| Навчання з `ds_width` (ширина за пропорцією), батч 16, CPU | ≈ 1,4 зразка/с (≈ 12 с/ітерацію) |
| Пам'ять процесу навчання 48×320, батч 32, CPU | ≈ 8 ГБ RSS |

Рекомендації:

* Інференс: CPU достатньо; 4+ ГБ RAM.
* Донавчання розпізнавання: GPU з 8+ ГБ (батч 64–128 для 48×320, 16–32 для 48×960);
  на CPU — лише для перевірки конвеєра (години на епоху).
* Донавчання детектора `PP-OCRv5_server_det`: GPU з 16+ ГБ (батч 4–8, кропи 640×640).
* Диск: ≈ 70 МБ навчальні ваги rec, ≈ 0,7 ГБ test RUKOPYS, ≈ 3 ГБ train RUKOPYS
  (+ ≈ 2 ГБ кропів рядків).

## Відомі обмеження

* Офіційна `eslav_PP-OCRv5_mobile_rec` має «російський ухил» на українському
  тексті (див. результати); звуження словника цього не виправляє, бо `і`/`и`,
  `є`/`е`, `ї`/`й` — усі українські літери. Потрібне донавчання на українських даних.
* Проведене донавчання — короткий CPU-запуск на малих даних, а не повноцінне
  навчання; синтетичний друкований train/test має обмежений набір шрифтів і текстів.
* Найкраща рукописна модель (v5) навчалася на CPU лише 2 епохи на ≈ 19,5 тис.
  рядків RUKOPYS (1 243 сторінки); точність рядків — лише ≈ 5–8 %.
* Рукописна модель v5 навчена без доповнення рядків. Використовуйте її з
  власним `inference.yml` (базова ширина 160); якщо примусово доповнювати
  рядки до ширини 960 (наприклад, `text_rec_input_shape`), вона видає сміття.
* У RUKOPYS рамки рядків прямокутні (не повернуті); у похилих рядках кроп
  захоплює частини сусідніх рядків.
* Детектор для рукопису не донавчався; наскрізна якість на рукописних сторінках
  обмежена і детекцією, і розпізнаванням.
* `get_word_info` виправлено в `ppocr`; у PaddleX (інференс через `PaddleOCR`) є
  власна реалізація, яку цей репозиторій не змінює (впливає лише на `return_word_box`).
* Модель з українським словником не виводить `Ё Ы Э Ъ Ў` — для російської/білоруської
  використовуйте `lang="ru"` / `lang="be"` (офіційна eslav-модель).
* Застарілі `uk_dict.txt`, `cyrillic_dict.txt` і моделі `ocr_version="PP-OCRv3"`
  для `uk` не містять `Ї ї` (і `Й Щ` в `uk_dict.txt`) — не використовуйте їх для української.

## Наступні кроки

1. Донавчити друковану модель на GPU на реальних українських документах (скани,
   фото, PDF) + синтетиці з більшою кількістю шрифтів і корпусом реальних текстів.
2. Рукописна модель: більше епох на повному `train` (зроблено 2), обережно —
   автоматично розмічений `silver` (≈ 9 тис. сторінок); оцінити на всьому `test`;
   розглянути більшу архітектуру (`PP-OCRv5_server_rec` з посимвольним
   перенесенням голови).
3. ~~Узгодити кропи навчання й інференсу~~ — зроблено у v5 (кропи детектора).
4. Донавчити детектор на рукописних сторінках (`uk_handwritten_PP-OCRv5_server_det.yml`)
   і оцінити `eval_pipeline.py`.
5. Зібрати незалежний test реальних друкованих документів різної якості.
6. Опційно — словникова/мовна модель для пост-корекції.
