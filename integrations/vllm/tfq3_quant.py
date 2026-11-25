"""
vLLM QuantizationConfig & LinearMethodBase plugin for TFQ3 (`TFPN` v3, 3/4/5-bit).

Registers `--quantization tfq3` using vLLM's standard quantization plugin API and dispatches
`apply()` directly to `tfq3_cuda_ext.tfq3_linear` (`src/kernels/cuda/tfq3_cuda.cuh`).

Status: [SOURCE-INSPECTED / INTERFACE-TESTED] — verified against vLLM's QuantizationConfig
and LinearMethodBase contract; live end-to-end serving with vLLM requires installing both
`vllm` and `integrations/pytorch` (`pip install -e integrations/pytorch`) on a CUDA host.
"""
from __future__ import annotations

from typing import Any, Optional
import torch
from torch.nn.parameter import Parameter

try:
    from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs  # type: ignore
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig  # type: ignore
    HAS_VLLM = True
except Exception:
    HAS_VLLM = False

    class QuantizationConfig:  # type: ignore
        pass

    class LinearMethodBase:  # type: ignore
        pass

    def set_weight_attrs(weight: torch.Tensor, attrs: dict[str, Any]) -> None:
        for k, v in attrs.items():
            setattr(weight, k, v)


class Tfq3LinearMethod(LinearMethodBase):
    """vLLM LinearMethod dispatching TFQ3 packed tensors to `tfq3_cuda_ext.tfq3_linear`."""

    def __init__(self, quant_config: "Tfq3Config") -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        out_part = sum(output_partition_sizes)
        pad_r = ((out_part + 127) // 128) * 128
        pad_c = ((input_size_per_partition + 127) // 128) * 128
        bits = self.quant_config.bits
        n_words = (pad_r // 16) * (pad_c // 128) * (bits * 2) * 32
        n_scales = (pad_r // 128) * (pad_c // 128)

        qweight = Parameter(torch.empty(n_words, dtype=torch.int32), requires_grad=False)
        scales = Parameter(torch.empty(n_scales, dtype=torch.float16), requires_grad=False)
        tables = Parameter(torch.empty(16384, dtype=torch.uint8), requires_grad=False)
        set_weight_attrs(qweight, {"packed_dim": 0, **extra_weight_attrs})
        set_weight_attrs(scales, extra_weight_attrs)
        layer.register_parameter("tfq3_words", qweight)
        layer.register_parameter("tfq3_scales", scales)
        layer.register_parameter("tfq3_tables", tables)
        layer.tfq3_pad_r = pad_r
        layer.tfq3_pad_c = pad_c
        layer.tfq3_orig_r = out_part

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        import tfq3_cuda_ext  # type: ignore

        y = tfq3_cuda_ext.tfq3_linear(
            x.to(torch.bfloat16),
            layer.tfq3_words,
            layer.tfq3_scales,
            layer.tfq3_tables,
            self.quant_config.bits,
            layer.tfq3_pad_r,
            layer.tfq3_pad_c,
            layer.tfq3_orig_r,
        )
        if bias is not None:
            y = y + bias.to(dtype=y.dtype)
        return y.to(dtype=x.dtype)


class Tfq3Config(QuantizationConfig):
    """vLLM QuantizationConfig for TFQ3 checkpoints (`pw_v3_tables.bin` + `.tffpw`)."""

    def __init__(self, bits: int = 4) -> None:
        if bits not in (3, 4, 5):
            raise ValueError(f"TFQ3 supports bits in {{3, 4, 5}}, got {bits}")
        self.bits = bits

    def get_name(self) -> str:
        return "tfq3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # NVIDIA Ampere (sm_80) and newer

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["tfq3_manifest.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Tfq3Config":
        bits = int(config.get("bits", 4))
        return cls(bits=bits)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[LinearMethodBase]:
        if isinstance(layer, torch.nn.Linear) or layer.__class__.__name__.endswith("Linear"):
            return Tfq3LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []
