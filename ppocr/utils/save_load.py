# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os
import pickle
import json
from packaging import version

import paddle

from ppocr.utils.logging import get_logger
from ppocr.utils.network import maybe_download_params

try:
    import encryption  # Attempt to import the encryption module for AIStudio's encryption model

    encrypted = encryption.is_encryption_needed()
except ImportError:
    print("Skipping import of the encryption module.")
    encrypted = False  # Encryption is not needed if the module cannot be imported

__all__ = ["load_model"]


# just to determine the inference model file format
def get_FLAGS_json_format_model():
    # json format by default
    return os.environ.get("FLAGS_json_format_model", "1").lower() in ("1", "true", "t")


FLAGS_json_format_model = get_FLAGS_json_format_model()


def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    "be happy if some process has already created {}".format(path)
                )
            else:
                raise OSError("Failed to mkdir {}".format(path))


def load_model(config, model, optimizer=None, model_type="det", ema=None):
    """
    load model from checkpoint or pretrained_model
    """
    logger = get_logger()
    global_config = config["Global"]
    checkpoints = global_config.get("checkpoints")
    pretrained_model = global_config.get("pretrained_model")
    best_model_dict = {}
    is_float16 = False
    is_nlp_model = model_type == "kie" and config["Architecture"]["algorithm"] not in [
        "SDMGR"
    ]

    if is_nlp_model is True:
        # NOTE: for kie model dsitillation, resume training is not supported now
        if config["Architecture"]["algorithm"] in ["Distillation"]:
            return best_model_dict
        checkpoints = config["Architecture"]["Backbone"]["checkpoints"]
        # load kie method metric
        if checkpoints:
            if os.path.exists(os.path.join(checkpoints, "metric.states")):
                with open(os.path.join(checkpoints, "metric.states"), "rb") as f:
                    states_dict = pickle.load(f, encoding="latin1")
                best_model_dict = states_dict.get("best_model_dict", {})
                if "epoch" in states_dict:
                    best_model_dict["start_epoch"] = states_dict["epoch"] + 1
            logger.info("resume from {}".format(checkpoints))

            if optimizer is not None:
                if checkpoints[-1] in ["/", "\\"]:
                    checkpoints = checkpoints[:-1]
                if os.path.exists(checkpoints + ".pdopt"):
                    optim_dict = paddle.load(checkpoints + ".pdopt")
                    optimizer.set_state_dict(optim_dict)
                else:
                    logger.warning(
                        "{}.pdopt is not exists, params of optimizer is not loaded".format(
                            checkpoints
                        )
                    )

        return best_model_dict

    if checkpoints:
        if checkpoints.endswith(".pdparams"):
            checkpoints = checkpoints.replace(".pdparams", "")
        assert os.path.exists(
            checkpoints + ".pdparams"
        ), "The {}.pdparams does not exists!".format(checkpoints)

        # load params from trained model
        params = paddle.load(checkpoints + ".pdparams")
        state_dict = model.state_dict()
        new_state_dict = {}
        for key, value in state_dict.items():
            if key not in params:
                logger.warning(
                    "{} not in loaded params {} !".format(key, params.keys())
                )
                continue
            pre_value = params[key]
            if pre_value.dtype == paddle.float16:
                is_float16 = True
            if pre_value.dtype != value.dtype:
                pre_value = pre_value.astype(value.dtype)
            if list(value.shape) == list(pre_value.shape):
                new_state_dict[key] = pre_value
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params shape {} !".format(
                        key, value.shape, pre_value.shape
                    )
                )
        model.set_state_dict(new_state_dict)
        if is_float16:
            logger.info(
                "The parameter type is float16, which is converted to float32 when loading"
            )
        if optimizer is not None:
            if os.path.exists(checkpoints + ".pdopt"):
                optim_dict = paddle.load(checkpoints + ".pdopt")
                optimizer.set_state_dict(optim_dict)
            else:
                logger.warning(
                    "{}.pdopt is not exists, params of optimizer is not loaded".format(
                        checkpoints
                    )
                )

        if os.path.exists(checkpoints + ".states"):
            with open(checkpoints + ".states", "rb") as f:
                states_dict = pickle.load(f, encoding="latin1")
            best_model_dict = states_dict.get("best_model_dict", {})
            best_model_dict["acc"] = 0.0
            if "epoch" in states_dict:
                best_model_dict["start_epoch"] = states_dict["epoch"] + 1
        logger.info("resume from {}".format(checkpoints))

        # Restore EMA state if available
        if ema is not None:
            pdema_path = checkpoints + ".pdema"
            if os.path.exists(pdema_path):
                ema_data = paddle.load(pdema_path)
                # .pdparams contains EMA weights; restore original training weights
                if "train_state" in ema_data:
                    train_sd = ema_data["train_state"]
                    cur_sd = model.state_dict()
                    for k in cur_sd:
                        if k in train_sd:
                            if list(cur_sd[k].shape) == list(train_sd[k].shape):
                                cur_sd[k] = train_sd[k]
                    model.set_state_dict(cur_sd)
                    logger.info(
                        "EMA: restored training weights from {}".format(pdema_path)
                    )
                # Restore EMA shadow weights + step
                if "ema_state" in ema_data and "step" in ema_data:
                    ema.set_state_dict(ema_data)
                    logger.info(
                        "EMA: restored shadow weights (step={}) from {}".format(
                            ema_data["step"], pdema_path
                        )
                    )

    elif pretrained_model:
        class_transfer = build_rec_class_transfer_from_config(global_config)
        is_float16 = load_pretrained_params(
            model, pretrained_model, class_transfer=class_transfer
        )
    else:
        logger.info("train from scratch")
    best_model_dict["is_float16"] = is_float16
    return best_model_dict


_SPECIAL_TOKENS = ("<blank>", "<unk>", "<s>", "</s>", "<tail>")


def _special(name):
    # Tuples never collide with dictionary characters (which are str).
    return ("special", name)


def _read_dict_chars(dict_path):
    chars = []
    with open(dict_path, "rb") as fin:
        for line in fin.readlines():
            chars.append(line.decode("utf-8").strip("\n").strip("\r\n"))
    return chars


def build_rec_class_transfer(old_chars, new_chars, use_space_char=True):
    """Describe how output classes of a recognition model map between two dicts.

    Returns a list of ``(old_tokens, new_tokens)`` class layouts used by the
    supported recognition heads:

    * CTC (``CTCLabelDecode``): ``blank`` + chars [+ space]
    * NRTR (``NRTRLabelDecode``, GTC branch of ``MultiHead``):
      ``blank, <unk>, <s>, </s>`` + chars [+ space]
    * NRTR vocabulary of ``Transformer`` which adds one extra trailing class.
    """
    space = [" "] if use_space_char else []
    old_chars = list(old_chars) + space
    new_chars = list(new_chars) + space
    blank, unk, bos, eos, tail = [_special(t) for t in _SPECIAL_TOKENS]
    layouts = [
        ([blank] + old_chars, [blank] + new_chars),
        ([blank, unk, bos, eos] + old_chars, [blank, unk, bos, eos] + new_chars),
        (
            [blank, unk, bos, eos] + old_chars + [tail],
            [blank, unk, bos, eos] + new_chars + [tail],
        ),
    ]
    return layouts


def build_rec_class_transfer_from_config(global_config):
    """Build class layouts when ``Global.pretrained_character_dict_path`` is set."""
    old_dict_path = global_config.get("pretrained_character_dict_path")
    if not old_dict_path:
        return None
    new_dict_path = global_config.get("character_dict_path")
    if not new_dict_path:
        raise ValueError(
            "Global.pretrained_character_dict_path requires Global.character_dict_path"
        )
    for p in (old_dict_path, new_dict_path):
        if not os.path.isfile(p):
            raise FileNotFoundError("character dict not found: {}".format(p))
    old_chars = _read_dict_chars(old_dict_path)
    new_chars = _read_dict_chars(new_dict_path)
    if old_chars == new_chars:
        return None
    if len(old_chars) == len(new_chars):
        get_logger().warning(
            "pretrained_character_dict_path and character_dict_path have the same "
            "size but different content; output layers have identical shapes and "
            "are loaded positionally. Change the dict size or re-initialize the head."
        )
    return build_rec_class_transfer(
        old_chars, new_chars, global_config.get("use_space_char", False)
    )


def transfer_class_params(old_value, new_value, class_transfer):
    """Copy the rows of ``old_value`` into ``new_value`` along the class axis.

    ``old_value`` and ``new_value`` are numpy arrays that differ only in the size
    of one axis, which must equal the number of classes of one layout in
    ``class_transfer``. Classes present in both dictionaries keep their
    pretrained weights; new classes keep the initial values of ``new_value``.

    Returns ``(array, num_copied, num_new_classes)`` or ``None`` when the
    parameter is not a class layer of a known layout.
    """
    if old_value.ndim != new_value.ndim:
        return None
    diff_axes = [
        i for i in range(old_value.ndim) if old_value.shape[i] != new_value.shape[i]
    ]
    if len(diff_axes) != 1:
        return None
    axis = diff_axes[0]
    for old_tokens, new_tokens in class_transfer:
        if old_value.shape[axis] != len(old_tokens) or new_value.shape[axis] != len(
            new_tokens
        ):
            continue
        old_index = {}
        for i, tok in enumerate(old_tokens):
            old_index.setdefault(tok, i)
        result = new_value.copy()
        copied = 0
        for j, tok in enumerate(new_tokens):
            i = old_index.get(tok)
            if i is None:
                continue
            dst = [slice(None)] * new_value.ndim
            src = [slice(None)] * old_value.ndim
            dst[axis], src[axis] = j, i
            result[tuple(dst)] = old_value[tuple(src)]
            copied += 1
        return result, copied, len(new_tokens)
    return None


def load_pretrained_params(model, path, class_transfer=None):
    logger = get_logger()
    path = maybe_download_params(path)
    if path.endswith(".pdparams"):
        path = path.replace(".pdparams", "")
    assert os.path.exists(
        path + ".pdparams"
    ), "The {}.pdparams does not exists!".format(path)

    params = paddle.load(path + ".pdparams")

    state_dict = model.state_dict()

    new_state_dict = {}
    is_float16 = False

    for k1 in params.keys():
        if k1 not in state_dict.keys():
            logger.warning("The pretrained params {} not in model".format(k1))
        else:
            if params[k1].dtype == paddle.float16:
                is_float16 = True
            if params[k1].dtype != state_dict[k1].dtype:
                params[k1] = params[k1].astype(state_dict[k1].dtype)
            if list(state_dict[k1].shape) == list(params[k1].shape):
                new_state_dict[k1] = params[k1]
                continue
            transferred = None
            if class_transfer is not None:
                transferred = transfer_class_params(
                    params[k1].numpy(), state_dict[k1].numpy(), class_transfer
                )
            if transferred is not None:
                value, copied, total = transferred
                new_state_dict[k1] = paddle.to_tensor(value, dtype=state_dict[k1].dtype)
                logger.info(
                    "Transferred {} {} -> {}: {}/{} classes copied from pretrained "
                    "params, {} newly initialized".format(
                        k1,
                        params[k1].shape,
                        state_dict[k1].shape,
                        copied,
                        total,
                        total - copied,
                    )
                )
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params {} {} !".format(
                        k1, state_dict[k1].shape, k1, params[k1].shape
                    )
                )

    model.set_state_dict(new_state_dict)
    if is_float16:
        logger.info(
            "The parameter type is float16, which is converted to float32 when loading"
        )
    logger.info("load pretrain successful from {}".format(path))
    return is_float16


def save_model(
    model,
    optimizer,
    model_path,
    logger,
    config,
    is_best=False,
    prefix="ppocr",
    ema=None,
    train_state=None,
    **kwargs,
):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)

    if prefix == "best_accuracy":
        best_model_path = os.path.join(model_path, "best_model")
        _mkdir_if_not_exist(best_model_path, logger)

    paddle.save(optimizer.state_dict(), model_prefix + ".pdopt")
    if prefix == "best_accuracy":
        paddle.save(
            optimizer.state_dict(), os.path.join(best_model_path, "model.pdopt")
        )

    is_nlp_model = config["Architecture"]["model_type"] == "kie" and config[
        "Architecture"
    ]["algorithm"] not in ["SDMGR"]
    if is_nlp_model is not True:
        paddle.save(model.state_dict(), model_prefix + ".pdparams")
        metric_prefix = model_prefix

        # Save EMA state for training resumption
        if ema is not None and train_state is not None:
            paddle.save(
                {
                    "train_state": train_state,
                    "ema_state": ema.state_dict,
                    "step": ema.step,
                },
                model_prefix + ".pdema",
            )

        if prefix == "best_accuracy":
            paddle.save(
                model.state_dict(), os.path.join(best_model_path, "model.pdparams")
            )
            if ema is not None and train_state is not None:
                paddle.save(
                    {
                        "train_state": train_state,
                        "ema_state": ema.state_dict,
                        "step": ema.step,
                    },
                    os.path.join(best_model_path, "model.pdema"),
                )

    else:  # for kie system, we follow the save/load rules in NLP
        if config["Global"]["distributed"]:
            arch = model._layers
        else:
            arch = model
        if config["Architecture"]["algorithm"] in ["Distillation"]:
            arch = arch.Student
        arch.backbone.model.save_pretrained(model_prefix)
        metric_prefix = os.path.join(model_prefix, "metric")

        if prefix == "best_accuracy":
            arch.backbone.model.save_pretrained(best_model_path)

    save_model_info = kwargs.pop("save_model_info", False)
    if save_model_info:
        with open(os.path.join(model_path, f"{prefix}.info.json"), "w") as f:
            json.dump(kwargs, f)
        logger.info("Already save model info in {}".format(model_path))
        if prefix != "latest":
            done_flag = kwargs.pop("done_flag", False)
            update_train_results(config, prefix, save_model_info, done_flag=done_flag)

    # save metric and config
    with open(metric_prefix + ".states", "wb") as f:
        pickle.dump(kwargs, f, protocol=2)
    if is_best:
        logger.info("save best model is to {}".format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))


def update_train_results(config, prefix, metric_info, done_flag=False, last_num=5):
    if paddle.distributed.get_rank() != 0:
        return

    assert last_num >= 1
    train_results_path = os.path.join(
        config["Global"]["save_model_dir"], "train_result.json"
    )
    save_model_tag = ["pdparams", "pdopt", "pdstates"]
    paddle_version = version.parse(paddle.__version__)
    if FLAGS_json_format_model or paddle_version >= version.parse("3.0.0"):
        save_inference_files = {
            "inference_config": "inference.yml",
            "pdmodel": "inference.json",
            "pdiparams": "inference.pdiparams",
        }
    else:
        save_inference_files = {
            "inference_config": "inference.yml",
            "pdmodel": "inference.pdmodel",
            "pdiparams": "inference.pdiparams",
            "pdiparams.info": "inference.pdiparams.info",
        }
    if os.path.exists(train_results_path):
        with open(train_results_path, "r") as fp:
            train_results = json.load(fp)
    else:
        train_results = {}
        train_results["model_name"] = config["Global"]["model_name"]
        label_dict_path = config["Global"].get("character_dict_path", "")
        if label_dict_path != "":
            label_dict_path = os.path.abspath(label_dict_path)
            if not os.path.exists(label_dict_path):
                label_dict_path = ""
        train_results["label_dict"] = label_dict_path
        train_results["train_log"] = "train.log"
        train_results["visualdl_log"] = ""
        train_results["config"] = "config.yaml"
        train_results["models"] = {}
        for i in range(1, last_num + 1):
            train_results["models"][f"last_{i}"] = {}
        train_results["models"]["best"] = {}
    train_results["done_flag"] = done_flag
    if "best" in prefix:
        if "acc" in metric_info["metric"]:
            metric_score = metric_info["metric"]["acc"]
        elif "precision" in metric_info["metric"]:
            metric_score = metric_info["metric"]["precision"]
        elif "exp_rate" in metric_info["metric"]:
            metric_score = metric_info["metric"]["exp_rate"]
        else:
            raise ValueError("No metric score found.")
        train_results["models"]["best"]["score"] = metric_score
        for tag in save_model_tag:
            if tag == "pdparams" and encrypted:
                train_results["models"]["best"][tag] = os.path.join(
                    prefix,
                    (
                        f"{prefix}.encrypted.{tag}"
                        if tag != "pdstates"
                        else f"{prefix}.states"
                    ),
                )
            else:
                train_results["models"]["best"][tag] = os.path.join(
                    prefix,
                    f"{prefix}.{tag}" if tag != "pdstates" else f"{prefix}.states",
                )
        for key in save_inference_files:
            train_results["models"]["best"][key] = os.path.join(
                prefix, "inference", save_inference_files[key]
            )
    else:
        for i in range(last_num - 1, 0, -1):
            train_results["models"][f"last_{i + 1}"] = train_results["models"][
                f"last_{i}"
            ].copy()
        if "acc" in metric_info["metric"]:
            metric_score = metric_info["metric"]["acc"]
        elif "precision" in metric_info["metric"]:
            metric_score = metric_info["metric"]["precision"]
        elif "exp_rate" in metric_info["metric"]:
            metric_score = metric_info["metric"]["exp_rate"]
        else:
            metric_score = 0
        train_results["models"][f"last_{1}"]["score"] = metric_score
        for tag in save_model_tag:
            if tag == "pdparams" and encrypted:
                train_results["models"][f"last_{1}"][tag] = os.path.join(
                    prefix,
                    (
                        f"{prefix}.encrypted.{tag}"
                        if tag != "pdstates"
                        else f"{prefix}.states"
                    ),
                )
            else:
                train_results["models"][f"last_{1}"][tag] = os.path.join(
                    prefix,
                    f"{prefix}.{tag}" if tag != "pdstates" else f"{prefix}.states",
                )
        for key in save_inference_files:
            train_results["models"][f"last_{1}"][key] = os.path.join(
                prefix, "inference", save_inference_files[key]
            )

    with open(train_results_path, "w") as fp:
        json.dump(train_results, fp)
