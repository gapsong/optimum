# coding=utf-8
# Copyright 2023 HuggingFace Inc. team and GPTQ and AutoGPTQ authors.
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

# Neu: benötigte Quant-Funktionen importieren
from __future__ import annotations
import importlib
import json
import os
from enum import Enum
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union
from safetensors import safe_open  # Falls safetensors verwendet wird; sonst load_file aus transformers
import torch
from packaging import version
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D
from transformers.utils.quantization_config import QuantizationMethod

from ..utils import is_accelerate_available, is_auto_gptq_available, is_gptqmodel_available
from ..utils.modeling_utils import recurse_getattr
from ..version import __version__ as optimum_version
from .constants import GPTQ_CONFIG
from .data import get_dataset, prepare_dataset
from .utils import (
    get_block_name_with_pattern,
    get_device,
    get_layers,
    get_preceding_modules,
    get_seqlen,
    nested_move_to,
)


if is_accelerate_available():
    from accelerate import (
        cpu_offload_with_hook,
        load_checkpoint_and_dispatch,
    )
    from accelerate.hooks import remove_hook_from_module

if is_auto_gptq_available():
    from auto_gptq import __version__ as autogptq_version
    from auto_gptq import exllama_set_max_input_length
    from auto_gptq.modeling._utils import autogptq_post_init as gptq_post_init
    from auto_gptq.quantization import GPTQ
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear as hf_select_quant_linear

if is_gptqmodel_available():
    from gptqmodel import exllama_set_max_input_length
    from gptqmodel.quantization import GPTQ
    from gptqmodel.utils.importer import hf_select_quant_linear
    from gptqmodel.utils.model import hf_convert_gptq_v1_to_v2_format, hf_convert_gptq_v2_to_v1_format
    from gptqmodel.utils.model import hf_gptqmodel_post_init as gptq_post_init
    from gptqmodel.version import __version__ as gptqmodel_version

logger = getLogger(__name__)


def has_device_more_than_cpu():
    return torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())


class ExllamaVersion(int, Enum):
    ONE = 1
    TWO = 2



import math
from typing import NamedTuple, Optional, Union

import torch
from tqdm.auto import tqdm

from .weight_permutation import get_permutation_order

from .quant_groups import quantize, dequantize, Quantizer



class QuantizationResult(NamedTuple):
    """A collection of codebooks, indices and assorted statistics produced by SPQRUtil; not memory-optimized!"""

    weight: torch.FloatTensor  # dequantized(quantized(weight)), same shape as the original
    perm: Optional[torch.LongTensor]  # optional input permutation indices that were used during quantization
    # NOTE: if permutation_order != identity, all subsequent tensors (incl. outlier indices) are permuted in that order!

    quantization_errors: torch.Tensor  # per-element quantization errors, defined as (weight - quantized_weight) / diag(inverse_hessian_cholesky)
    unstructured_outlier_threshold: float  # threshold on squared error increase used for determining *UNSTRUCTURED* outliers
    unstructured_outlier_mask: torch.Tensor  # bool mask where True means that this is an individual outlier
    save_quant_dict: dict


class SPQRUtil:
    """Learns GPTQ for a single linear layer"""

    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.columns = self.layer.weight.data.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        assert self.H is not None, "Already ran quantization; cannot add more data batches"
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def quantize(
        self,
        *,
        bits: int = 2,
        blocksize: int = 128,
        percdamp: float = 1e-2,
        groupsize: Optional[int] = None,
        keep_last_columns: int = 0,
        outlier_relative_threshold: float = float("inf"),
        permutation_order: Union[str, torch.Tensor] = "identity",
        keep_H: bool = True,
        simplified_outliers: bool = False,
        verbose=True,
        perchannel: bool = True,
        sym: bool = False,
        save_quantization: bool = False,
        **kwargs,
    ) -> QuantizationResult:
        """
        :param bits: number of bits used at the lowest level (the full model size will be different!)
        :param blocksize: take blocks of this many input features at a time for GPTQ
        :note: blocksize affects runtime and memory, but does not affect the resulting matrix (up to machine precision)
        :param groupsize: fit quantization scaling / statistics to each group of this many input features
        :param percdamp: relative regularizer added to hessian diagonal before inversion
        :note: if groupsize_in_dim* is None, use the same quantization statistics across all input features
        :param keep_last_columns: if not None, keep the last (this many) input features un_quantized and return them
        :note: the un-quantized columns will be a part of the first returned result
        :param outlier_relative_threshold: threshold used for *UNSTRUCTURED* outliers, relative to
        :note: if keep_last_columns > 0, quantized_dequantized_weights[-keep_last_columns:] will be non-quantized
        :param permutation_order: re-order input features using a certain policy
        :param keep_H: if False, delete the accumulated hessian during quantize; if False, keep the accumulated hessian
        :param simplified_outliers: if True,do not perform leave-one-out evaluation when detecting outliers;
            works faster, but generally worse in perplexity
        :param verbose: if True, display a tqdm progressbar over input columns
        :param sym: if True, base weight quantization is symmetric
        :param perchannel: if True, base weight quantization will learn statistics for each output dimension separately
        :return: a QuantizationResult tuple that contains(
            weight, perm, _unused, _unused, _unused, _unused, quantization_errors, outlier_unstructured_mask
        ), see class QuantizationResult below for details
        """
        weight = self.layer.weight.detach().to(dtype=torch.float, copy=True)
        save_quant_dict = {}
        perm = get_permutation_order(self.H, weight, permutation_order)

        if save_quantization:
            save_quant_dict["quant_weights"] = []
            save_quant_dict["quant_layer_scale"] = []
            save_quant_dict["quant_layer_zeros"] = []
            save_quant_dict["quant_layer_scale_qq_scale"] = []
            save_quant_dict["quant_layer_scale_qq_zero"] = []
            save_quant_dict["quant_layer_zero_qq_scale"] = []
            save_quant_dict["quant_layer_zero_qq_zero"] = []
            save_quant_dict["save_float_dtype"] = self.layer.weight.dtype
            save_quant_dict["outliers_matrix"] = torch.zeros(
                weight.shape, dtype=save_quant_dict["save_float_dtype"]
            ).to(
                weight.device
            )  # shape = [out_features, in_features]

        weight = weight[:, perm]  # note: weight is modified
        H = self.H
        if keep_H:
            H = H.clone()  # protect from in-place changes
        else:
            self.H = None

        H = H[perm][:, perm]
        self.dead = torch.diag(H) == 0  # indices of input features that do not affect outputs
        if percdamp > 0:
            ix = torch.arange(len(H), device=weight.device)
            H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
            del ix
        H[self.dead, self.dead] = 1
        weight[:, self.dead] = 0
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)
        H_inv_cho_diag = torch.diag(H_inv_cho)
        del H

        quantizer = Quantizer()
        quantizer.configure(bits, perchannel=perchannel, sym=sym, **kwargs)
        assert H_inv_cho.shape[0] == H_inv_cho.shape[1] == weight.shape[1], "weight must be [out_features, in_features]"
        out_dim, in_dim = weight.shape  # [out_features, in_features]

        if groupsize is None:
            groupsize = in_dim

        # prepare outlier detection
        outlier_column_indices = torch.empty(0, dtype=torch.int64, device=weight.device)
        del H_inv

        outlier_scale = (weight.var(dim=0) / torch.diag(H_inv_cho).square()).mean().item()
        unstructured_outlier_threshold = outlier_relative_threshold * outlier_scale
        in_group_index = -1  # index of current group of input features, for group quantizer purposes

        quantization_errors = torch.zeros_like(weight)
        unstructured_outlier_mask = torch.zeros_like(weight, dtype=torch.bool)

        block_start_iter = range(0, in_dim - keep_last_columns, blocksize)
        block_start_iter = tqdm(block_start_iter, leave=False) if verbose else block_start_iter
        for block_start in block_start_iter:
            block_end = min(block_start + blocksize, in_dim)
            for column_index in range(block_start, block_end):
                if column_index % groupsize == 0:
                    # fit weight quantizer on the upcoming group of weight columns (inputs), across all rows (outputs)
                    in_group_index += 1
                    group_weight = weight[:, column_index : column_index + groupsize]

                    if simplified_outliers or (unstructured_outlier_threshold == float("inf")):
                        quantizer.find_params(group_weight, weight=True)

                    else:
                        # objective: detect which weights will be designated as outliers, fit quantizer *without* these weights
                        # step 1: fit quantizer on a leave-one-out version of weights, i.e. in each group, drop one weight at a time
                        assert perchannel, "refitting quantizer is only implemented for perchannel=True"
                        group_diag_hessian_inv_cho = H_inv_cho_diag[column_index : column_index + groupsize]
                        loo_quantization_error_sq = get_leave_one_out_error(
                            group_weight, group_diag_hessian_inv_cho, bits=bits, sym=sym
                        )
                        # ^-- dequantized(quantized(group_weight)) using a quantizer trained on all weights except the reconstructed one

                        likely_unstructured_outlier_mask = (
                            loo_quantization_error_sq > unstructured_outlier_threshold
                        ).float()

                        non_outlier_mask = 1 - likely_unstructured_outlier_mask
                        mean_over_non_outliers = torch.sum(
                            group_weight * non_outlier_mask, dim=1, keepdim=True
                        ) / torch.sum(non_outlier_mask, dim=1, keepdim=True).clamp_min(1)
                        group_weight_without_outliers = group_weight * non_outlier_mask + mean_over_non_outliers * (
                            1 - non_outlier_mask
                        )
                        quantizer.find_params(group_weight_without_outliers, weight=True)
                        del group_diag_hessian_inv_cho, loo_quantization_error_sq
                        del mean_over_non_outliers, group_weight_without_outliers, non_outlier_mask

                    if save_quantization:
                        if quantizer.qq_scale_bits is not None:
                            save_quant_dict["quant_layer_scale"].append(quantizer.quant_scale.to(torch.int8))
                            save_quant_dict["quant_layer_scale_qq_scale"].append(
                                quantizer.qq_scale.scale.to(save_quant_dict["save_float_dtype"])
                            )
                            save_quant_dict["quant_layer_scale_qq_zero"].append(
                                quantizer.qq_scale.zero.to(save_quant_dict["save_float_dtype"])
                            )
                        else:
                            save_quant_dict["quant_layer_scale"].append(
                                quantizer.scale.to(save_quant_dict["save_float_dtype"])
                            )

                        if quantizer.qq_zero_bits is not None and (
                            (not quantizer.round_zero) or quantizer.qq_zero_bits < quantizer.bits
                        ):
                            save_quant_dict["quant_layer_zeros"].append(quantizer.quant_zero.to(torch.int8))
                            save_quant_dict["quant_layer_zero_qq_scale"].append(
                                quantizer.qq_zero.scale.to(save_quant_dict["save_float_dtype"])
                            )
                            save_quant_dict["quant_layer_zero_qq_zero"].append(
                                quantizer.qq_zero.zero.to(save_quant_dict["save_float_dtype"])
                            )
                        else:
                            save_quant_dict["quant_layer_zeros"].append(
                                quantizer.zero.to(save_quant_dict["save_float_dtype"])
                            )
                    del group_weight

                weight_quant_i = quantize(
                    weight[:, column_index].unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
                )
                weight_i_quantized = dequantize(weight_quant_i, quantizer.scale, quantizer.zero).reshape_as(
                    weight[:, column_index]
                )

                delta_weight_i = weight[:, column_index] - weight_i_quantized  # [out_dim]
                quantization_errors[:, column_index] = (
                    delta_weight_i / H_inv_cho[column_index, column_index]
                )  # [out_dim]

                if unstructured_outlier_threshold != float("inf"):
                    unstructured_outlier_mask[:, column_index] = (
                        quantization_errors[:, column_index].square() > unstructured_outlier_threshold
                    )
                    # re-quantize without outliers
                    is_outlier = unstructured_outlier_mask[:, column_index].float()

                    weight_quant_i = quantize(
                        (weight[:, column_index] * (1 - is_outlier)).unsqueeze(1),
                        quantizer.scale,
                        quantizer.zero,
                        quantizer.maxq,
                    )
                    weight_i_quantized_wo_outliers = dequantize(
                        weight_quant_i, quantizer.scale, quantizer.zero
                    ).reshape_as(weight[:, column_index])
                    weight_i_quantized = (
                        weight_i_quantized_wo_outliers * (1 - is_outlier) + weight[:, column_index] * is_outlier
                    )  # [out_dim]

                    if save_quantization:
                        save_quant_dict["outliers_matrix"][:, column_index] = weight[:, column_index] * is_outlier

                    del weight_i_quantized_wo_outliers

                    delta_weight_i = weight[:, column_index] - weight_i_quantized  # [out_dim]
                    quantization_errors[:, column_index] = (
                        delta_weight_i / H_inv_cho[column_index, column_index]
                    )  # [out_dim]

                if save_quantization:
                    save_quant_dict["quant_weights"].append(weight_quant_i.to(torch.int8))

                weight[:, column_index] = weight_i_quantized
                weight[:, column_index + 1 : block_end].addr_(
                    quantization_errors[:, column_index],
                    H_inv_cho[column_index, column_index + 1 : block_end],
                    alpha=-1,
                )

            weight[:, block_end:].addmm_(
                quantization_errors[:, block_start:block_end],
                H_inv_cho[block_start:block_end, block_end:],
                alpha=-1,
            )

        if permutation_order != "identity":
            invperm = torch.argsort(perm)
            weight = weight[:, invperm]

        if save_quantization:
            save_quant_dict["perm"] = perm.to(torch.int32)
            save_quant_dict["keep_last_columns"] = 0
            save_quant_dict["g_idx"] = in_group_index
            save_quant_dict["weight_shape"] = weight.shape
            save_quant_dict["groupsize"] = groupsize if groupsize else weight.shape[1]
            save_quant_dict["quant_weights"] = torch.cat(save_quant_dict["quant_weights"], dim=1)
            save_quant_dict["outliers_matrix"] = save_quant_dict["outliers_matrix"].to_sparse()

        return QuantizationResult(
            weight=weight,
            perm=perm,
            quantization_errors=quantization_errors,
            unstructured_outlier_threshold=unstructured_outlier_threshold,
            unstructured_outlier_mask=unstructured_outlier_mask,
            save_quant_dict=save_quant_dict,
        )
        
def get_leave_one_out_error(group_weight: torch.Tensor,
                            group_diag_hinv_cho: torch.Tensor,
                            *,
                            bits: int,
                            sym: bool = True):
    """
    Berechnet pro Element die Fehlerreduktion, wenn dieses Element als Outlier behandelt wird.
    (SPQR-ähnlicher Leave-One-Out Score)

    Args:
        group_weight: [out_dim, g] Gewichte der aktuellen Gruppe (g = group_size).
        group_diag_hinv_cho: [g] diag(cholesky(H_inv)) für die g Spalten dieser Gruppe.
        bits: Basis-Bitbreite für die Quantisierung.
        sym: Symmetrische Quantisierung (wie in deinem Quantizer üblich).

    Returns:
        reduction_in_squared_error: [out_dim, g]
            (baseline_error_sq - loo_error_sq) pro Gewicht. Höher = nützlicher als Outlier.
    """
    assert group_weight.ndim == 2, f"Erwartet 2D, bekam {group_weight.shape}"
    out_dim, g = group_weight.shape
    assert group_diag_hinv_cho.ndim == 1 and group_diag_hinv_cho.numel() == g, \
        f"group_diag_hinv_cho muss 1D Länge g sein, bekam {group_diag_hinv_cho.shape}"

    # 1) Indizes für Leave-One-Out: für jede Spalte j, alle anderen Spalten
    #    Erzeuge eine [g, g-1]-Indexmatrix: in Zeile j stehen die Spalten != j.
    loo_indices = []
    for j in range(g):
        idx = torch.cat([torch.arange(0, j, device=group_weight.device),
                         torch.arange(j + 1, g, device=group_weight.device)])
        loo_indices.append(idx)
    loo_indices = torch.stack(loo_indices, dim=0)  # [g, g-1]

    # 2) Leave-One-Out Daten: [out_dim, g, g-1]
    groupwise_loo_data = group_weight[:, loo_indices]  # advanced indexing erzeugt neue Achse

    # 3) Quantizer für Leave-One-Out fitten (ein Fit für alle g LOO-Fälle gleichzeitig)
    #    Verwende deinen Quantizer (perchannel=True) auf [out_dim*g, g-1].
    from .quant_groups import Quantizer  # passe ggf. Import an
    fast_quantizer = Quantizer(shape=groupwise_loo_data.flatten(0, 1).shape)
    fast_quantizer.configure(bits, perchannel=True, sym=sym)
    fast_quantizer.find_params(groupwise_loo_data.flatten(0, 1), weight=True)

    # Rekonstruiere leave-one-out Gewichte und forme zurück zu [out_dim, g, g-1]
    loo_reconstructed = fast_quantizer.quantize_dequantize(
        groupwise_loo_data.flatten(0, 1)
    ).reshape_as(groupwise_loo_data)

    # 4) Whitening-Faktoren für die LOO-Batches: [g, g-1]
    loo_group_diag_hinv_cho = group_diag_hinv_cho[loo_indices]

    # LOO-Fehler: Summe der hessian-gewichteten MSE über die verbleibenden (g-1) Spalten
    # Ergebnis: [out_dim, g]
    loo_errors_sq = ((loo_reconstructed - groupwise_loo_data) / loo_group_diag_hinv_cho).square().sum(dim=-1)

    # 5) Baseline: normal quantisieren ohne Outliers auf [out_dim, g]
    base_quantizer = Quantizer(shape=group_weight.shape)
    base_quantizer.configure(bits, perchannel=True, sym=sym)
    base_quantizer.find_params(group_weight, weight=True)
    baseline_reconstructed = base_quantizer.quantize_dequantize(group_weight)

    baseline_errors_sq = ((baseline_reconstructed - group_weight) / group_diag_hinv_cho).square().sum(dim=1, keepdim=True)
    # 6) Nutzen als Outlier = wie stark sinkt der Fehler, wenn dieses Gewicht ein Outlier bleibt
    reduction_in_squared_error = baseline_errors_sq - loo_errors_sq  # [out_dim, g]
    return reduction_in_squared_error



class GPTQQuantizer(object):
    r"""
    A simple API for GPTQ Quantization
    """

    def __init__(
        self,
        bits: int,
        dataset: Optional[Union[List[str], str]] = None,
        group_size: int = 128,
        damp_percent: float = 0.1,
        desc_act: bool = False,
        sym: bool = True,
        true_sequential: bool = True,
        use_cuda_fp16: bool = False,
        model_seqlen: Optional[int] = None,
        block_name_to_quantize: Optional[str] = None,
        module_name_preceding_first_block: Optional[List[str]] = None,
        batch_size: int = 1,
        pad_token_id: Optional[int] = None,
        disable_exllama: bool = False,
        exllama_config: Optional[Dict[str, Any]] = None,
        max_input_length: Optional[int] = None,
        cache_block_outputs: Optional[bool] = True,
        modules_in_block_to_quantize: Optional[List[List[str]]] = None,
        checkpoint_format: str = "gptq",
        meta: Optional[Dict[str, any]] = None,
        backend: Optional[str] = None,
        quant_engine: str = "gptq",
        *args,
        **kwargs,
    ):
        """
        Args:
            bits (`int`):
                The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
            dataset (`Union[List[str], str, Any]`, defaults to `None`):
                The dataset used for quantization. You can provide your own dataset in a list of string or in a list of tokenized data
                (e.g. [{ "input_ids": [ 1, 100, 15, ... ],"attention_mask": [ 1, 1, 1, ... ]},...])
                or just use the original datasets used in GPTQ paper ['wikitext2','c4','c4-new'].
            group_size (int, defaults to 128):
                The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
            damp_percent (`float`, defaults to `0.1`):
                The percent of the average Hessian diagonal to use for dampening, recommended value is 0.1.
            desc_act (`bool`, defaults to `False`):
                Whether to quantize columns in order of decreasing activation size.
                Setting it to False can significantly speed up inference but the perplexity may become slightly worse.
                Also known as act-order.
            sym (`bool`, defaults to `True`):
                Whether to use symetric quantization.
            true_sequential (`bool`, defaults to `True`):
                Whether to perform sequential quantization even within a single Transformer block.
                Instead of quantizing the entire block at once, we perform layer-wise quantization.
                As a result, each layer undergoes quantization using inputs that have passed through the previously quantized layers.
            use_cuda_fp16 (`bool`, defaults to `False`):
                Whether or not to use optimized cuda kernel for fp16 model. Need to have model in fp16.
            model_seqlen (`Optional[int]`, defaults to `None`):
                The maximum sequence length that the model can take.
            block_name_to_quantize (`Optional[str]`, defaults to `None`):
                The transformers block name to quantize. If None, we will infer the block name using common patterns (e.g. model.layers)
            module_name_preceding_first_block (`Optional[List[str]]`, defaults to `None`):
                The layers that are preceding the first Transformer block.
            batch_size (`int`, defaults to `1`):
                The batch size of the dataset
            pad_token_id (`Optional[int]`, defaults to `None`):
                The pad token id. Needed to prepare the dataset when `batch_size` > 1.
            disable_exllama (`bool`, defaults to `False`):
                Whether to use exllama backend. Only works with `bits` = 4.
            exllama_config (`Dict[str, Any]`, *optional*):
                The exllama config. You can specify the version of the exllama kernel through the `version` key. Defaults to `{"version": 2}` if unset.
            max_input_length (`Optional[int]`, defaults to `None`):
                The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input length.
                It is specific to the exllama backend with act-order.
            cache_block_outputs (`bool`, defaults to `True`):
                Whether to cache block outputs to reuse as inputs for the succeeding block. It allows optimization of non-standard models
                (e.g. ChatGLM) but can require more time.
            modules_in_block_to_quantize (`Optional[List[List[str]]]`, defaults to `None`):
                List list of module names to quantize in the block specified. This argument is useful to exclude certain linear modules from being quantized.
                The block to quantize can be specified by setting `block_name_to_quantize`. We will quantize each list sequentially.
                If not set, we will quantize all linear layers. Example: `inside_layer_modules=[["self_attention.query_key_value"], ["mlp.dense_h_to_4h"]]`
            checkpoint_format (`str`, *optional*, defaults to `gptq`):
                GPTQ weight format. `gptq`(v1) is supported by both gptqmodel and auto-gptq. `gptq_v2` is gptqmodel only.
            meta (`Dict[str, any]`, *optional*):
                Properties, such as tooling:version, that do not directly contributes to quantization or quant inference are stored in meta.
                i.e. `meta.quantizer`: ["optimum:_version_", "gptqmodel:_version_"]
            backend (`str`, *optional*):
                Controls which gptq kernel to be used. Valid values for gptqmodel are `auto`, `auto_trainable` and more. For auto-gptq, only valid value is None and `auto_trainable`. Ref gptqmodel backends: https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/utils/backend.py
        """

        self.bits = bits
        self.dataset = dataset
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.sym = sym
        self.true_sequential = true_sequential
        self.checkpoint_format = checkpoint_format.lower()
        self.meta = meta
        self.backend = backend.lower() if backend is not None else None
        self.quant_engine = (quant_engine or "spqr").lower()
        self.use_cuda_fp16 = use_cuda_fp16
        self.model_seqlen = model_seqlen
        self.block_name_to_quantize = block_name_to_quantize
        self.module_name_preceding_first_block = module_name_preceding_first_block
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.disable_exllama = disable_exllama
        self.exllama_config = exllama_config
        self.max_input_length = max_input_length
        self.quant_method = QuantizationMethod.GPTQ
        self.cache_block_outputs = cache_block_outputs
        self.modules_in_block_to_quantize = modules_in_block_to_quantize
        self.lr_layers = {}  # Dict zum Speichern aller LR-Matrizen: {"layer_name": lr_tensor}
        self.apply_error_correction = True 
        
        self.serialization_keys = [
            "bits",
            "dataset",
            "group_size",
            "damp_percent",
            "desc_act",
            "sym",
            "true_sequential",
            "quant_method",
            "modules_in_block_to_quantize",
            "checkpoint_format",
            "meta",
            "quant_engine",
        ]

        if self.bits not in [2, 3, 4, 8]:
            raise ValueError("only support quantize to [2,3,4,8] bits.")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

        if self.exllama_config is None:
            self.exllama_config = {"version": ExllamaVersion.TWO}
        else:
            if "version" not in self.exllama_config:
                raise ValueError("`exllama_config` needs to have a `version` key")
            elif self.exllama_config["version"] not in [ExllamaVersion.ONE, ExllamaVersion.TWO]:
                version = self.exllama_config["version"]
                raise ValueError(
                    f"Only supported versions are in [ExllamaVersion.ONE, ExllamaVersion.TWO] - not recognized version {version}"
                )
        self.exllama_version = self.exllama_config["version"]
        self.lr_layers = None

    def select_quant_linear(self, device_map: Union[str, dict], pack: bool = False):
        if is_gptqmodel_available():
            self.quant_linear = hf_select_quant_linear(
                bits=self.bits,
                group_size=self.group_size,
                desc_act=self.desc_act,
                sym=self.sym,
                checkpoint_format=self.checkpoint_format,
                meta=self.meta,
                device_map=device_map,
                backend=self.backend,
                pack=pack,
            )
        else:
            self.quant_linear = hf_select_quant_linear(
                use_triton=False,
                desc_act=self.desc_act,
                group_size=self.group_size,
                bits=self.bits,
                disable_exllama=self.disable_exllama or self.exllama_version != ExllamaVersion.ONE,
                disable_exllamav2=self.disable_exllama or self.exllama_version != ExllamaVersion.TWO,
            )

    def to_dict(self):
        """
        Returns the args in dict format.
        """
        gptq_dict = {}
        for key in self.serialization_keys:
            gptq_dict[key] = getattr(self, key)

        if gptq_dict.get("meta") is None:
            gptq_dict["meta"] = {}

        meta = gptq_dict["meta"]
        # store both optimum:version and gptq_lib:version into quantize_config.meta.quantizer
        if meta.get("quantizer") is None:
            meta["quantizer"] = [f"optimum:{optimum_version}"]

            if is_gptqmodel_available():
                meta["quantizer"].append(f"gptqmodel:{gptqmodel_version}")
            elif is_auto_gptq_available():
                meta["quantizer"].append(f"auto_gptq:{autogptq_version}")

        return gptq_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Instantiates a `GPTQQuantizer` using config_dict as kwargs

        Args:
            config_dict (`Dict[str,Any]`):
                quantization config

        Returns:
            `GPTQQuantizer`:  The quantizer object instantiated from those parameters.
        """
        return cls(**config_dict)

    def convert_model(self, model: nn.Module, **kwargs):
        """
        Convert the model to a GPTQ model by getting and replacing the layers.

        Args:
            model (`nn.Module`):
                Model to be converted

        """
        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name_with_pattern(model)
        block_name = self.block_name_to_quantize
        layers_to_be_replaced = get_layers(model, prefix=block_name)
        if self.modules_in_block_to_quantize is not None:
            layers_to_keep = sum(self.modules_in_block_to_quantize, [])
            for name in list(layers_to_be_replaced.keys()):
                if not any(name.endswith(layer) for layer in layers_to_keep):
                    logger.info(
                        f"Quantization disabled for {name} (only modules_in_block_to_quantize={self.modules_in_block_to_quantize} are quantized)"
                    )
                    del layers_to_be_replaced[name]

        self.select_quant_linear(device_map=kwargs.get("device_map", None), pack=False)

        self._replace_by_quant_layers(model, layers_to_be_replaced)

        return model

    def get_no_split_module_classes(self, model):
        """
        Get the modules that should not be split across multiple devices.
        Args:
            model (`nn.Module`):
                The input model
        """

        block_class_name = recurse_getattr(model, self.block_name_to_quantize)[0].__class__.__name__
        no_split_module_classes = [block_class_name]
        return no_split_module_classes

    def _replace_by_quant_layers(self, module: nn.Module, names: List[str], name: str = ""):
        """
        Replaces linear layers in `module` by `QuantLinear`

        Args:
            module (`nn.Module`):
                Module to quantize
            names (`List[str]`):
                List of names of the module to quantize
            name (`str`, defaults to `""`):
                To keep track of the name of the current module
        """
        if isinstance(module, self.quant_linear):
            return
        for attr in dir(module):
            layer = getattr(module, attr)
            name1 = name + "." + attr if name != "" else attr
            if name1 in names:
                device = get_device(layer)
                delattr(module, attr)
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    out_features = layer.out_features
                elif isinstance(layer, nn.Conv2d):
                    in_features = layer.in_channels
                    out_features = layer.out_channels
                elif isinstance(layer, Conv1D):
                    in_features = layer.weight.shape[0]
                    out_features = layer.weight.shape[1]
                bias = layer.bias is not None
                if is_gptqmodel_available():
                    new_layer = self.quant_linear(
                        self.bits,
                        self.group_size,
                        self.desc_act,
                        self.sym,
                        in_features,
                        out_features,
                        bias,
                        weight_dtype=layer.weight.dtype,
                    )
                else:
                    if not (self.desc_act) or self.group_size == -1:
                        new_layer = self.quant_linear(
                            self.bits,
                            self.group_size,
                            in_features,
                            out_features,
                            bias,
                            use_cuda_fp16=self.use_cuda_fp16,
                            weight_dtype=layer.weight.dtype,
                        )
                    else:
                        new_layer = self.quant_linear(
                            self.bits,
                            self.group_size,
                            in_features,
                            out_features,
                            bias,
                            weight_dtype=layer.weight.dtype,
                        )
                new_layer.device = device
                setattr(module, attr, new_layer.to(device))
        for name1, child in module.named_children():
            self._replace_by_quant_layers(child, names, name + "." + name1 if name != "" else name1)
            
    def _quantize_block_with_spqr(
            self,
            block_index: int,
            block: nn.Module,
            block_device: Union[torch.device, int],
            layers_name_list: List[List[str]],
            dataset: List[Dict[str, torch.Tensor]],
            layer_inputs: List[Tuple[torch.Tensor]],
            layer_input_kwargs: List[Dict[str, torch.Tensor]],
            quantizers: dict
        ):
            layers = get_layers(block)
            logger.info(f"SPQR: Module to quantize {layers_name_list}")
            for subset_name_list in tqdm(layers_name_list, leave=False, desc="SPQR quantizing layers inside the block"):
                subset_layers = {name: layers[name] for name in subset_name_list if name in layers}
                spqr = {}
                handles = []

                # Hooks, um Eingaben der Layer zu akkumulieren (Hessian via X^T X)
                for name, layer in subset_layers.items():
                    if not isinstance(layer, nn.Linear):
                        logger.info(f"SPQR: Überspringe {name} (Typ {layer.__class__.__name__}), nur nn.Linear wird unterstützt.")
                        continue
                    spqr[name] = SPQRUtil(layer)

                    def add_batch_spqr(n):
                        def _hook(_, inp, out):
                            try:
                                x = inp[0].detach()
                            except Exception:
                                x = inp[0]
                            spqr[n].add_batch(x)
                        return _hook

                    handles.append(layer.register_forward_hook(add_batch_spqr(name)))

                # Hessian-Akkumulation durch Vorwärtsläufe
                for j in range(len(dataset)):
                    layer_inputs[j] = nested_move_to(layer_inputs[j], block_device)
                    for k, v in layer_input_kwargs[j].items():
                        layer_input_kwargs[j][k] = nested_move_to(v, block_device)
                    block(*layer_inputs[j], **layer_input_kwargs[j])

                # Hooks entfernen
                for h in handles:
                    h.remove()

                # Quantisierung und Gewichtsüberschreibung
                for name in subset_name_list:
                    logger.info(f"SPQR: Quantisiere {name} ...")
                    res = spqr[name].quantize(
                        bits=self.bits,
                        # kleiner Blocksize für Stabilität; später tunen
                        blocksize=self.group_size,
                        percdamp=self.damp_percent,
                        # GPTQ: group_size==-1 => per-column; SPQR groupsize=None => ein Group; minimal kompatibel
                        groupsize=None if self.group_size == -1 else self.group_size,
                        permutation_order="identity" if not self.desc_act else "identity",  # TODO: act_order später
                        perchannel=True,
                        sym=self.sym,
                        verbose=False,
                        save_quantization=True
                    )

                    scale, zero, g_idx  = res.save_quant_dict["quant_layer_scale"], res.save_quant_dict["quant_layer_zeros"], res.save_quant_dict["g_idx"]
                    
                    quantizers[f"{self.block_name_to_quantize}.{block_index}.{name}"] = (
                        spqr[name],
                        scale,
                        zero,
                        g_idx,
                    )

                del subset_layers, spqr
                
    def load_all_lr_layers(self, adapter_path):
        """
        Lädt einen gespeicherten LoRA-Adapter, extrahiert lora_A und lora_B,
        berechnet den Skalierungsfaktor und rekonstruiert die Low-Rank-Update-Matrix für alle Layer.
        Speichert alles in self.lr_layers.
        """
        # 1. Pfade zur Konfigurations- und Gewichtsdatei definieren
        config_path = os.path.join(adapter_path, "adapter_config.json")
        weights_path = os.path.join(adapter_path, "adapter_model.safetensors")

        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            print(f"❌ Fehler: Konnte 'adapter_config.json' oder 'adapter_model.safetensors' nicht in {adapter_path} finden.")
            return

        # 2. Konfiguration laden, um 'r' und 'lora_alpha' zu erhalten
        print(f"🔍 Lade Konfiguration von: {config_path}")
        with open(config_path, "r") as f:
            adapter_config = json.load(f)

        r = adapter_config.get("r")
        lora_alpha = adapter_config.get("lora_alpha")
        use_rslora = adapter_config.get("use_rslora", False)

        if r is None or lora_alpha is None:
            print("❌ Fehler: 'r' oder 'lora_alpha' nicht in der Konfiguration gefunden.")
            return

        # 3. Den 'scaling'-Parameter berechnen
        if use_rslora:
            scaling = lora_alpha / torch.sqrt(torch.tensor(r))
            print(f"✅ Konfiguration geladen: r={r}, lora_alpha={lora_alpha} (rsLoRA-Skalierung verwendet)")
        else:
            scaling = lora_alpha / r
            print(f"✅ Konfiguration geladen: r={r}, lora_alpha={lora_alpha}")
        
        print(f"   Berechneter Skalierungsfaktor: {scaling:.4f}\n")

        # 4. Das State Dictionary des Adapters laden
        print(f"🔍 Lade Gewichte von: {weights_path}")
        # Verwende safe_open für safetensors
        adapter_state_dict = {}
        lr_layers = {}
        with safe_open(weights_path, framework="pt") as f:
            for key in f.keys():
                adapter_state_dict[key] = f.get_tensor(key)
        print(f"✅ {len(adapter_state_dict)} Tensoren geladen.\n")

        # 5. Durch alle Layer iterieren und die Low-Rank-Matrix rekonstruieren
        # Finde alle lora_A Gewichte, um die Layer zu identifizieren
        lora_a_keys = [key for key in adapter_state_dict if key.endswith(".lora_A.weight")]

        for lora_a_key in lora_a_keys:
            base_key = lora_a_key.replace(".lora_A.weight", "")
            lora_b_key = base_key + ".lora_B.weight"

            if lora_b_key in adapter_state_dict:
                lora_A = adapter_state_dict[lora_a_key]
                lora_B = adapter_state_dict[lora_b_key]
                
                # Dies ist die Kernlogik Ihrer Anfrage
                # LR = scaling * (B @ A)
                low_rank_update = scaling * (lora_B @ lora_A)
                
                # Speichere im Dict (base_key ist der Layer-Name, z.B. "q_proj")
                # layer_name = base_key.split(".")[-1]  # Extrahiere z.B. "q_proj" aus "model.layers.0.self_attn.q_proj"
                cleaned_keys = base_key.replace("base_model.model.model.layers.","")
                lr_layers[cleaned_keys] = low_rank_update
                
                print(f"--- Rekonstruiert für Layer: {base_key} ---")
                print(f"  - lora_A Form: {lora_A.shape}")
                print(f"  - lora_B Form: {lora_B.shape}")
                print(f"  - Rekonstruierte LR-Matrix Form: {low_rank_update.shape}\n")

        print(f"✅ Alle LR-Layer geladen und in self.lr_layers gespeichert: {list(lr_layers.keys())}")
        
        return lr_layers

    @torch.no_grad()
    def update_block_weights(self, block, block_index, layers_in_block, add=True):
        """
        Aktualisiert die Gewichte eines Blocks, indem die Low-Rank-Updates aus self.lr_layers addiert (oder subtrahiert) werden.

        Args:
            block (nn.Module): Der Transformer-Block, der aktualisiert werden soll.
            block_index (int): Der Index des Blocks im Modell (z.B. 0 für den ersten Block).
            layers_in_block (List[List[str]]): Eine Liste von Layernamen innerhalb des Blocks, die aktualisiert werden könnten.
                                              Beispiel: [['self_attn.q_proj'], ['self_attn.v_proj']]
            add (bool): Wenn True, werden die Updates addiert. Wenn False, werden sie subtrahiert.
        """
        # 1. Hole das state_dict des Blocks. Wir werden dieses modifizieren und dann wieder laden.
        sd = block.state_dict()
        
        # 2. Iteriere durch alle Layer, die potenziell quantisiert werden sollen.
        #    Die `layers_in_block` ist eine Liste von Listen, also flachen wir sie ab.
        all_layer_names = [name for sublist in layers_in_block for name in sublist]

        for layer_name in all_layer_names:
            # 3. Konstruiere den Schlüssel für dein `lr_layers` Dictionary.
            #    Beispiel: "0.self_attn.q_proj"
            lr_key = f"{block_index}.{layer_name}"

            # 4. Prüfe, ob für diesen Layer ein Low-Rank-Update existiert.
            if lr_key in self.lr_layers:
                
                # 5. Konstruiere den Schlüssel für das `state_dict` des Blocks.
                #    Dies ist der entscheidende Schritt: Wir hängen ".weight" an.
                #    Beispiel: "self_attn.q_proj.weight"
                param_key = f"{layer_name}.weight"

                if param_key in sd:
                    # 6. Hole das Low-Rank-Delta und das Ziel-Parameter-Tensor.
                    delta = self.lr_layers[lr_key]
                    param = sd[param_key]
                    
                    # Stelle sicher, dass beide auf demselben Gerät und vom selben Typ sind.
                    delta = delta.to(device=param.device, dtype=param.dtype)
                    
                    # 7. Addiere oder subtrahiere das Update.
                    if add:
                        sd[param_key].add_(delta)
                        print(f"✅ Update für Block {block_index}, Layer {param_key} HINZUGEFÜGT.")
                    else:
                        sd[param_key].sub_(delta)
                        print(f"✅ Update für Block {block_index}, Layer {param_key} ENTFERNT.")
                else:
                    print(f"⚠️ Warnung: Parameter '{param_key}' nicht im state_dict von Block {block_index} gefunden.")
            
        # 8. Lade das modifizierte state_dict zurück in den Block.
        #    `strict=False` ist sicherer, falls das state_dict zusätzliche Schlüssel enthält.
        block.load_state_dict(sd, strict=False)

            
    @torch.no_grad()
    def quantize_model(self, model: nn.Module, tokenizer: Optional[Any] = None, adapter_path:str = None):
        """
        Quantizes the model using the dataset

        Args:
            model (`nn.Module`):
                The model to quantize
            tokenizer (Optional[`Any`], defaults to `None`):
                The tokenizer to use in order to prepare the dataset. You can pass either:
                    - A custom tokenizer object.
                    - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                      using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        Returns:
            `nn.Module`: The quantized model
        """

        if self.quant_engine == "gptq":
            if not is_auto_gptq_available() and not is_gptqmodel_available():
                raise RuntimeError(
                    "gptqmodel oder auto-gptq ist erforderlich für GPTQ-Quantisierung. Installiere `pip install gptqmodel` oder `pip install auto-gptq`."
                )
            elif is_gptqmodel_available() and is_auto_gptq_available():
                logger.warning("Beide gptqmodel und auto-gptq erkannt, nutze gptqmodel.")
            gptq_supports_cpu = (
                is_auto_gptq_available()
                and version.parse(importlib.metadata.version("auto-gptq")) > version.parse("0.4.2")
            ) or is_gptqmodel_available()
            if not gptq_supports_cpu and not torch.cuda.is_available():
                raise RuntimeError("Keine CUDA- oder CPU(IPEX)-Unterstützung für GPTQ-Quantisierung gefunden.")
            if not self.sym and not is_gptqmodel_available():
                raise ValueError("Asymmetrisch (sym=False) wird nur von gptqmodel unterstützt.")
            if self.checkpoint_format == "gptq_v2" and not is_gptqmodel_available():
                raise ValueError("gptq_v2 Format nur mit gptqmodel.")
        else:
            raise ValueError(f"Unbekannter quant_engine: {self.quant_engine}")

        model.eval()

        # gptqmodel internal is gptq_v2 for asym support, gptq(v1) can only support sym=True
        if is_gptqmodel_available() and self.checkpoint_format != "gptq_v2":
            self.checkpoint_format = "gptq_v2"

        # For Transformer model
        has_config = False
        has_device_map = False
        if hasattr(model, "config"):
            has_config = True
            use_cache = model.config.use_cache
            model.config.use_cache = False

        # If the model has a device_map, we don't move to model. We have already dispatched the hook that will do the work
        if hasattr(model, "hf_device_map"):
            devices = list(model.hf_device_map.values())
            has_device_map = True
            if "disk" in devices:
                raise ValueError("disk offload is not supported with GPTQ quantization")
            if "cpu" in devices or torch.device("cpu") in devices:
                if len(model.hf_device_map) > 1:
                    logger.info("Cpu offload is not recommended. There might be some issues with the memory")
                    hook = None
                    for name, device in model.hf_device_map.items():
                        if device == "cpu":
                            module = recurse_getattr(model, name)
                            remove_hook_from_module(module, recurse=True)
                            module, hook = cpu_offload_with_hook(module, prev_module_hook=hook)
                else:
                    has_device_map = False

        if hasattr(model, "dtype"):
            self.use_cuda_fp16 = model.dtype == torch.float16

        if self.model_seqlen is None:
            # We allow a max value of 4028 to avoid passing data with huge length to the model during the calibration step
            self.model_seqlen = min(4028, get_seqlen(model))

        device = get_device(model)

        # Step 1: Prepare the data
        if isinstance(self.dataset, list) and not isinstance(self.dataset[0], str):
            dataset = self.dataset
            logger.info("GPTQQuantizer dataset appears to be already tokenized. Skipping tokenization.")
        else:
            if isinstance(tokenizer, str):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
                except Exception:
                    raise ValueError(
                        f"""We were not able to get the tokenizer using `AutoTokenizer.from_pretrained`
                        with the string that you have passed {tokenizer}. If you have a custom tokenizer, you can pass it as input.
                        For now, we only support quantization for text model. Support for vision, speech and multimodel will come later."""
                    )
            if self.dataset is None:
                raise ValueError("You need to pass `dataset` in order to quantize your model")
            elif isinstance(self.dataset, str):
                dataset = get_dataset(self.dataset, tokenizer, seqlen=self.model_seqlen, split="train")
            elif isinstance(self.dataset, list):
                dataset = [tokenizer(data, return_tensors="pt") for data in self.dataset]
            else:
                raise ValueError(
                    f"You need to pass a list of string, a list of tokenized data or a string for `dataset`. Found: {type(self.dataset)}."
                )

        dataset = prepare_dataset(dataset, pad_token_id=self.pad_token_id, batch_size=self.batch_size)

        # Step 2: get the input of the 1st block
        # To do that, we need to put the modules preceding the first block on the same device as the first bloc.
        # Then we run the model and it will stop at the first bloc as we added a prehook that raise an Exception after storing the inputs.

        layer_inputs = []
        layer_outputs = []
        layer_input_kwargs = []

        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name_with_pattern(model)

        if self.module_name_preceding_first_block is None:
            self.module_name_preceding_first_block = get_preceding_modules(model, self.block_name_to_quantize)

        blocks = recurse_getattr(model, self.block_name_to_quantize)

        cur_layer_device = get_device(blocks[0])
        if not is_gptqmodel_available() and cur_layer_device.type == "cpu":
            cur_layer_device = 0

        if not has_device_map:
            # put modules from module_name_preceding_first_block on cuda or xpu or cpu
            to_device = cur_layer_device
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)
                if module is None:
                    raise ValueError(f"Module {module_name} was not found in model")
                module = module.to(to_device)
            blocks[0] = blocks[0].to(to_device)
        import matplotlib.pyplot as plt


        @torch.no_grad()
        def detect_spqr_outliers_weightwise_loo(W, H, bits=4, lambda_=1e-2, group_size=16, target_rate=0.000001, sym=True):
            """
            Markiert Elemente mit größter Fehlerreduktion (Leave-One-Out) als Outliers.
            target_rate bezieht sich auf die Elemente in der Gruppe (nicht nur Spalten).
            """
            # Whitening
            I = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
            L = torch.linalg.cholesky(H + lambda_ * I)
            Hinv = torch.cholesky_inverse(L)
            Hinv_cho = torch.linalg.cholesky(Hinv, upper=True)
            d = torch.diag(Hinv_cho)

            out, inp = W.shape
            mask = torch.zeros_like(W, dtype=torch.bool)

            for c0 in range(0, inp, group_size):
                c1 = min(c0 + group_size, inp)
                G = W[:, c0:c1]  # [out, g]
                # Leave-One-Out Nutzen (höher = stärkerer Outlier-Kandidat)
                reduction = get_leave_one_out_error(G, d[c0:c1], bits=bits, sym=sym)  # [out, g]

                # Wähle Top-k Elemente in dieser Gruppe nach target_rate
                k = max(1, int(reduction.numel() * target_rate))
                flat = reduction.reshape(-1)
                topk_idx = torch.topk(flat, k, largest=True).indices
                rows = topk_idx // (c1 - c0)
                cols = topk_idx %  (c1 - c0)
                mask[rows, c0 + cols] = True

            return mask

        @torch.no_grad()
        def detect_spqr_outliers(W, H, outlier_percentile=0.99, bits=4, percdamp=0.01):
            """
            FINALE VERSION: Verwendet einen adaptiven Schwellenwert, um einen festen Prozentsatz
            der Spalten mit dem höchsten Fehler als Ausreißer zu identifizieren.

            Args:
                W (torch.Tensor): Die originale Gewichtsmatrix (float32).
                H (torch.Tensor): Die ORIGINALE Hessian-Matrix (float32).
                outlier_percentile (float): Das Perzentil, das als Schwellenwert verwendet wird.
                                            0.99 bedeutet: "Markiere die Top 1% als Ausreißer".
                                            0.98 bedeutet: "Markiere die Top 2% als Ausreißer".
                bits (int): Anzahl der Bits für die Quantisierungssimulation.
                percdamp (float): Dämpfung für den Hessian.
            """
            
            # --- 1. Quantisierte Gewichte für die gesamte Matrix vorberechnen ---
            q_max = 2**(bits - 1) - 1
            scales = W.abs().max(dim=0, keepdim=True)[0] / q_max
            scales.clamp_(min=1e-5)
            W_q = torch.round(W / scales).clamp(-q_max, q_max) * scales
            
            # --- 2. Fehlerbeitrag für JEDE SPALTE berechnen und speichern ---
            num_cols = W.shape[1]
            column_errors = torch.zeros(num_cols, device=W.device)

            for j in range(num_cols):
                h_val = H[j, j].clone()
                h_val += percdamp * H.diagonal().abs().mean()
                if h_val == 0: h_val = 1.0
                
                transform_scalar = torch.sqrt(h_val)
                error_col = W[:, j] - W_q[:, j]
                whitened_error = error_col * transform_scalar
                column_error_contribution = torch.sum(whitened_error**2)
                column_errors[j] = column_error_contribution

            # --- 3. ADAPTIVEN Schwellenwert berechnen ---
            # Der Schwellenwert ist das 99. Perzentil aller Fehlerbeiträge.
            # Alle Spalten, deren Fehler größer ist, gehören zu den Top 1%.
            adaptive_threshold = torch.quantile(column_errors, outlier_percentile)

            # --- 4. Maske basierend auf dem adaptiven Schwellenwert erstellen ---
            # Dies ist eine Vektor-Operation und viel schneller als eine Schleife.
            is_outlier_column = column_errors > adaptive_threshold
            outlier_mask = torch.zeros_like(W, dtype=torch.bool)
            outlier_mask[:, is_outlier_column] = True

            # --- 5. Optional: Visualisierung der Fehler und des adaptiven Schwellenwerts ---
            # plt.figure(figsize=(12, 5))
            # plt.plot(column_errors.cpu().numpy(), marker='.', linestyle='-')
            # plt.axhline(y=adaptive_threshold.cpu().item(), color='r', linestyle='--', label=f'Adaptiver Threshold ({outlier_percentile*100:.0f}. Perzentil)')
            # plt.xlabel("Spalten-Index (Column Index)")
            # plt.ylabel("Fehlerbeitrag der Spalte")
            # plt.title(f"Fehlerbeiträge pro Spalte (Adaptive Methode)")
            # plt.legend()
            # plt.yscale('log')
            # plt.tight_layout()
            # plt.show()

            if outlier_mask.any():
                num_outlier_cols = outlier_mask.sum().item() // W.shape[0]
                percent_outliers = (num_outlier_cols / num_cols) * 100
                print(f"INFO: V6 (Adaptive) detected {outlier_mask.sum()} outliers in {num_outlier_cols} columns ({percent_outliers:.2f}%).")
                
            return outlier_mask

        @torch.no_grad()
        def process_outliers_for_saving(outlier_mask, adjusted_outliers_W, svd_rank=32):
            """
            Processes the outlier matrix to create two saveable formats: sparse and SVD.

            Args:
                outlier_mask (torch.Tensor): The boolean mask of outlier positions.
                adjusted_outliers_W (torch.Tensor): The full weight matrix containing adjusted outlier values.
                svd_rank (int): The rank for the SVD approximation.

            Returns:
                dict: A dictionary containing the processed data for both formats.
            """
            # Create the dense outlier matrix O
            O = torch.zeros_like(adjusted_outliers_W)
            O[outlier_mask] = adjusted_outliers_W[outlier_mask]
            
            # --- 1. Sparse Format (for PEFT-like adapter) ---
            # Convert the boolean mask to COO format indices
            sparse_indices = outlier_mask.nonzero().t()  # Shape: (2, num_outliers)
            sparse_values = O[outlier_mask]
            
            # --- 2. SVD Format ---
            try:
                U, S, Vh = torch.linalg.svd(O.to(torch.float32), full_matrices=False)
                
                # Truncate to the specified rank
                U_k = U[:, :svd_rank]
                S_k = S[:svd_rank]
                Vh_k = Vh[:svd_rank, :]
                
                svd_components = {
                    "U": U_k.cpu(),
                    "S": S_k.cpu(),
                    "Vh": Vh_k.cpu()
                }
            except torch.linalg.LinAlgError:
                print("WARNING: SVD did not converge. Skipping SVD format for this layer.")
                svd_components = None

            return {
                "sparse": {
                    "outlier_mask": outlier_mask.cpu(),
                    "values": sparse_values.cpu()
                },
                "svd": svd_components
            }

        def visualize_and_save_weights(
            W: torch.Tensor, 
            outlier_mask: torch.Tensor, 
            layer_name: str, 
            base_save_dir: str = "spqr_visualizations"
        ):
            """
            Erstellt und speichert verbesserte Visualisierungen der Gewichtsmatrizen und Ausreißer
            mit adaptiven Farbskalen.

            Args:
                W (torch.Tensor): Die originale, volle Gewichtsmatrix (auf der CPU oder GPU).
                outlier_mask (torch.Tensor): Die boolesche Maske, die die Positionen der Ausreißer anzeigt.
                layer_name (str): Ein eindeutiger Name für die Schicht (z.B. "block_16_o_proj").
                base_save_dir (str): Das Hauptverzeichnis, in dem die Ergebnisse gespeichert werden.
            """
            print(f"Erstelle verbesserte Visualisierungen für Schicht: {layer_name}...")
            
            # --- 1. Daten vorbereiten und auf die CPU verschieben ---
            W_cpu = W.detach().float().cpu()
            mask_cpu = outlier_mask.detach().cpu()
            
            W_for_gptq = W_cpu.clone()
            W_for_gptq[mask_cpu] = 0.0
            
            O_sparse = torch.zeros_like(W_cpu)
            outlier_values = W_cpu[mask_cpu]
            O_sparse[mask_cpu] = outlier_values
            
            layer_dir = os.path.join(base_save_dir, layer_name)
            os.makedirs(layer_dir, exist_ok=True)
            
            # --- 2. Plots mit angepassten Skalen erstellen ---

            # Skala 1: Für die dichten Matrizen (ignoriert die Top/Bottom 1% der extremsten Werte)
            # Dies enthüllt die Struktur der "normalen" Gewichte.
            vmin_dense = torch.quantile(W_cpu, 0.01).item()
            vmax_dense = torch.quantile(W_cpu, 0.99).item()
            
            # Plot 1: Originale Gewichte (mit adaptiver Skala)
            plt.figure(figsize=(12, 9))
            plt.imshow(W_cpu.numpy(), cmap='coolwarm', aspect='auto', vmin=vmin_dense, vmax=vmax_dense)
            plt.colorbar(label="Gewichtswert")
            plt.title(f"1. Originale Gewichte (Angepasste Skala)\n{layer_name}")
            plt.xlabel("Eingangs-Features (Input Features)")
            plt.ylabel("Ausgangs-Features (Output Features)")
            plt.savefig(os.path.join(layer_dir, "01_original_weights.png"), bbox_inches='tight', dpi=150)
            plt.close()

            # Plot 2: Gewichte für GPTQ (mit adaptiver Skala)
            plt.figure(figsize=(12, 9))
            plt.imshow(W_for_gptq.numpy(), cmap='coolwarm', aspect='auto', vmin=vmin_dense, vmax=vmax_dense)
            plt.colorbar(label="Gewichtswert")
            plt.title(f"2. Gewichte für GPTQ (Ausreißer auf Null)\n{layer_name}")
            plt.xlabel("Eingangs-Features")
            plt.ylabel("Ausgangs-Features")
            plt.savefig(os.path.join(layer_dir, "02_weights_for_gptq.png"), bbox_inches='tight', dpi=150)
            plt.close()

            # Skala 2: Für die Sparse-Matrix (fokussiert auf die Ausreißer)
            # Macht die Skala symmetrisch um Null, um die Ausreißer hervorzuheben.
            if outlier_values.numel() > 0:
                v_abs_max = outlier_values.abs().max().item()
                vmin_sparse, vmax_sparse = -v_abs_max, v_abs_max
            else:
                vmin_sparse, vmax_sparse = -1, 1 # Fallback, falls keine Ausreißer da sind
            import numpy as np
            # Plot 3: Nur die Ausreißer (mit fokussierter Skala)
            plt.figure(figsize=(12, 9))
            # Wir verwenden hier np.ma.masked_where, um die Nullen transparent zu machen.
            # Das hebt die Struktur der Ausreißer noch besser hervor.
            masked_sparse = np.ma.masked_where(O_sparse.numpy() == 0, O_sparse.numpy())
            
            # Hintergrundfarbe des Plots auf neutrales Grau setzen
            ax = plt.gca()
            ax.set_facecolor('gray')
            
            plt.imshow(masked_sparse, cmap='coolwarm', aspect='auto', vmin=vmin_sparse, vmax=vmax_sparse)
            plt.colorbar(label="Wert des Ausreißers")
            plt.title(f"3. Nur die Ausreißer (Fokussierte Skala)\n{layer_name}")
            plt.xlabel("Eingangs-Features")
            plt.ylabel("Ausgangs-Features")
            plt.savefig(os.path.join(layer_dir, "03_outliers_sparse.png"), bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"Verbesserte Visualisierungen für '{layer_name}' wurden in '{layer_dir}' gespeichert.")
    
        def store_input_hook(_, input, *args):
            kwargs = args[0]
            if input is None:
                if "hidden_states" in kwargs:
                    input = (nested_move_to(kwargs["hidden_states"], cur_layer_device),)
                else:
                    raise ValueError("No input value found in the foward pass")
            layer_inputs.append(input)
            other_kwargs = {}
            for k, v in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states"]:
                    other_kwargs[k] = nested_move_to(v, cur_layer_device)
            layer_input_kwargs.append(other_kwargs)
            raise ValueError
        # NEW: SPQR pro-Block Quantisierung
        
                
        if self.cache_block_outputs:
            handle = blocks[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
            for data in dataset:
                for k, v in data.items():
                    data[k] = nested_move_to(v, cur_layer_device)
                try:
                    model(**data)
                except ValueError:
                    pass
            handle.remove()

        if not has_device_map:
            blocks[0].to(device)
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)
                if module is None:
                    raise ValueError(f"Module {module_name} was not found in model")

        torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

        # Step 3: Quantize the blocks
        quantizers = {}
        for i, block in enumerate(tqdm(blocks, desc=f"Quantizing {self.block_name_to_quantize} blocks ")):
            logger.info(f"Start quantizing block {self.block_name_to_quantize} {i + 1}/{len(blocks)}")
            if not self.cache_block_outputs:
                handle = block.register_forward_pre_hook(store_input_hook, with_kwargs=True)
                for data in dataset:
                    for k, v in data.items():
                        data[k] = nested_move_to(v, cur_layer_device)
                    try:
                        model(**data)
                    except ValueError:
                        pass
                handle.remove()
            if (not has_device_map or get_device(block) == torch.device("cpu")) and has_device_more_than_cpu():
                block = block.to(0)
            layers = get_layers(block)
            block_device = get_device(block)
            if not is_gptqmodel_available() and block_device.type == "cpu":
                block_device = 0
            if isinstance(self.modules_in_block_to_quantize, list) and len(self.modules_in_block_to_quantize) > 0:
                if self.true_sequential:
                    layers_name_list = self.modules_in_block_to_quantize
                else:
                    layers_name_list = [sum(self.modules_in_block_to_quantize, [])]
            else:
                if self.true_sequential:
                    layers_name_list = [[key] for key in layers.keys()]
                else:
                    layers_name_list = [list(layers.keys())]
                
            self._quantize_block_with_spqr(
                block_index=i,
                block=block,
                block_device=block_device,
                layers_name_list=layers_name_list,
                dataset=dataset,
                layer_inputs=layer_inputs,
                layer_input_kwargs=layer_input_kwargs,
                quantizers=quantizers
            )
            
            if self.cache_block_outputs:
                for j in range(len(dataset)):
                    layer_output = block(*layer_inputs[j], **layer_input_kwargs[j])
                    layer_outputs.append(layer_output)
                if not has_device_map:
                    blocks[i] = block.to(device)
                del layers
                del layer_inputs
                layer_inputs, layer_outputs = layer_outputs, []
            else:
                del layers
                del layer_inputs
                layer_inputs = []
            torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
        if self.quant_engine == "gptq":
            if self.bits == 4:
                pass
            self.pack_model(model=model, quantizers=quantizers)
        else:
            logger.info("SPQR: Pack-Schritt übersprungen (Gewichte sind bereits quantisiert).")
        model.is_quantized = True
        model.quantization_method = QuantizationMethod.GPTQ
        if has_config:
            model.config.use_cache = use_cache
            model.config.quantization_config = self.to_dict()
        model = self.post_init_model(model)
        torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        return model

    def post_init_model(self, model):
        if self.quant_engine == "spqr":
            return model
        """
        Post-initialization that require device information, for example buffers initialization on device.

        Args:
            model (`nn.Module`):
                The input model
        """
        if self.bits == 4 and not self.disable_exllama:
            if get_device(model).type != "cuda" or (
                hasattr(model, "hf_device_map") and any(d in model.hf_device_map for d in ["cpu", "disk", "hpu"])
            ):
                if not self.disable_exllama:
                    logger.warning(
                        "Found modules on cpu/disk. Using Exllama/Exllamav2 backend requires all the modules to be on GPU. Setting `disable_exllama=True`"
                    )
                    self.disable_exllama = True

        class StoreAttr(object):
            pass

        if is_gptqmodel_available():
            model, _ = hf_convert_gptq_v1_to_v2_format(
                model, self.bits, self.quant_linear, self.checkpoint_format, self.meta
            )

        model.quantize_config = StoreAttr()
        model.quantize_config.desc_act = self.desc_act
        model = gptq_post_init(model, use_act_order=self.desc_act)
        if (
            self.desc_act
            and (not self.disable_exllama and self.exllama_version == ExllamaVersion.ONE)
            and self.max_input_length is not None
        ):
            model = exllama_set_max_input_length(model, self.max_input_length)
        return model

    def pack_model(
        self,
        model: nn.Module,
        quantizers: Dict[str, Tuple],
    ):
        """
        Pack the model by replacing the layers by quantized layers

        Args:
            model (`nn.Module`):
                The model to pack
            quantizers (`Dict[str,Tuple]`):
                A mapping of the layer name and the data needed to pack the layer
        """
        logger.info("Packing model...")
        layers = get_layers(model)
        layers = {n: layers[n] for n in quantizers}

        self.select_quant_linear(device_map=model.hf_device_map, pack=True)

        self._replace_by_quant_layers(model, quantizers)
        qlayers = get_layers(model, [self.quant_linear])
        for name in qlayers:
            logger.info(name)
            quantizers[name], scale, zero, g_idx = quantizers[name]
            # so far can only pack layer on CPU
            layer_device = qlayers[name].device
            qlayers[name].to("cpu")
            layers[name], scale, zero, g_idx = layers[name].to("cpu"), scale.to("cpu"), zero.to("cpu"), g_idx.to("cpu")
            qlayers[name].pack(layers[name], scale, zero, g_idx)
            qlayers[name].to(layer_device)

        logger.info("Model packed.")

    def save(self, model: nn.Module, save_dir: str, max_shard_size: str = "10GB", safe_serialization: bool = True):
        """
        Save model state dict and configs

        Args:
            model (`nn.Module`):
                Model to be saved. The model can be wrapped or unwraped.
            save_dir (`str`):
                Directory to which to save. Will be created if it doesn't exist.
            max_shard_size (`str`, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>
            safe_serialization (`bool`, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).

        """

        # convert gptqmodel internal gptq_v2 format to v1 for max compatibility
        if is_gptqmodel_available():
            model, converted = hf_convert_gptq_v2_to_v1_format(
                model, self.sym, self.bits, self.quant_linear, self.checkpoint_format, self.meta
            )
            if converted:
                self.checkpoint_format = "gptq"

        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
        with open(os.path.join(save_dir, GPTQ_CONFIG), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_quantized_model(
    model: nn.Module,
    save_folder: str,
    quant_config_name: str = GPTQ_CONFIG,
    state_dict_name: Optional[str] = None,
    device_map: Optional[str] = None,
    max_memory: Optional[Dict] = None,
    no_split_module_classes: Optional[Dict] = None,
    offload_folder: Optional[str] = None,
    offload_buffers: Optional[str] = None,
    offload_state_dict: bool = False,
    disable_exllama: bool = False,
    exllama_config: Optional[Dict[str, Any]] = None,
    max_input_length: Optional[int] = None,
):
    """
    Load quantized weights from the save_folder into the converted model and dispatch the weights according to the device_map.

    Args:
        model (`nn.Module`):
            The model can be enpty or not.
        save_folder (`str`):
            Directory to which to load the weights.
        quant_config_name (`str`, defaults to `GPTQ_CONFIG`):
            Name of the quantization config file
        state_dict_name (`Optional[str]`, defaults to `None`):
            Name of the state dict file
        device_map (`Optional[str]`, defaults to `None`):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
            To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`.
        max_memory (`Optional[Dict]`, defaults to `None`):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each GPU
            and the available CPU RAM if unset.
        no_split_module_classes (`Optional[Dict]`, defaults to `None`):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        offload_folder (`Optional[str]`, defaults to `None`):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_buffers (`Optional[str]`, defaults to `None`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
        offload_state_dict (`bool`, defaults to `False`):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit. Will default to `True` if the device map
            picked contains `"disk"` values.
        disable_exllama (`Optional[bool]`, defaults to `None`):
            Whether to use exllama backend. Only works with `bits` = 4.
        exllama_config (`Optional[Dict[str, Any]]`, defaults to `None`):
            The exllama config. You can specify the version of the exllama kernel through the `version` key. Defaults to `{"version": 2}` if unset.
        max_input_length (`Optional[int]`, defaults to `None`):
            The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input length.
            It is specific to the exllama backend with act-order.

    Returns:
        `nn.Module`: The quantized model
    """
    if not torch.cuda.is_available() and not is_gptqmodel_available():
        raise RuntimeError("No GPU found. A GPU is needed to run quantized model by auto_gptq.")
    if not is_auto_gptq_available() and not is_gptqmodel_available():
        raise RuntimeError(
            "gptqmodel (`pip install gptqmodel`) or auto-gptq (`pip install auto-gptq`) is required in order to load quantized weights. Please notice that auto-gptq will be deprecated in the future."
        )
    if not is_accelerate_available():
        raise RuntimeError(
            "You need to install accelerate in order to load and dispatch weights to"
            "a quantized model. You can do it with `pip install accelerate`"
        )
    if device_map is None:
        device_map = {"": torch.cuda.current_device()}
        logger.info("The device_map was not initialized." "Setting device_map to `{'':torch.cuda.current_device()}`.")

    if exllama_config is None:
        exllama_config = {"version": ExllamaVersion.TWO}
    else:
        if "version" not in exllama_config:
            raise ValueError("`exllama_config` needs to have a `version` key")
        elif exllama_config["version"] not in [ExllamaVersion.ONE, ExllamaVersion.TWO]:
            version = exllama_config["version"]
            raise ValueError(
                f"Only supported versions are in [ExllamaVersion.ONE, ExllamaVersion.TWO] - not recognized version {version}"
            )

    # this branch will check if model is from huggingface
    try:
        if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
            quantize_config_dict = model.config.quantization_config.to_dict()
        else:
            with open(os.path.join(save_folder, quant_config_name), "r", encoding="utf-8") as f:
                quantize_config_dict = json.load(f)
    except Exception as err:
        raise ValueError(
            f"Failed to load quantization config from {save_folder} (lookup for traceback): {err}\nTip: If the save directory is saved from a transformers.PreTrainedModel, make sure that `config.json` contains a 'quantization_config' key."
        ) from err
    quantizer = GPTQQuantizer.from_dict(quantize_config_dict)
    quantizer.disable_exllama = disable_exllama
    quantizer.exllama_config = exllama_config
    quantizer.exllama_version = quantizer.exllama_config["version"]
    quantizer.max_input_length = max_input_length

    model = quantizer.convert_model(model, device_map=device_map)

    if no_split_module_classes is None:
        no_split_module_classes = quantizer.get_no_split_module_classes(model)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(save_folder, state_dict_name) if state_dict_name is not None else save_folder,
        device_map=device_map,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes,
        offload_folder=offload_folder,
        offload_buffers=offload_buffers,
        offload_state_dict=offload_state_dict,
    )

    model = quantizer.post_init_model(model)
    model.is_quantized = True
    model.quantization_method = QuantizationMethod.GPTQ
    model.eval()
    return model
