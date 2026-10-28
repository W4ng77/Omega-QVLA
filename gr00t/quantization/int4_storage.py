"""Lossless nibble packing for weights that have already been quantized to W4."""

from __future__ import annotations

import torch


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.to(torch.uint8)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def _unpack_nibbles(packed: torch.Tensor, group_size: int) -> torch.Tensor:
    codes = torch.empty((packed.shape[0], group_size), dtype=torch.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    return codes


def pack_codebook(weight: torch.Tensor, group_size: int, batch_groups: int = 8192) -> dict:
    if weight.ndim != 2 or weight.shape[1] % group_size or group_size % 2:
        raise ValueError("Expected a matrix with an even group size dividing its input width")

    groups = weight.contiguous().view(-1, group_size)
    packed_parts = []
    codebook_parts = []
    for start in range(0, groups.shape[0], batch_groups):
        values = groups[start : start + batch_groups]
        sorted_values, order = values.sort(dim=1)
        new_value = torch.ones_like(sorted_values, dtype=torch.bool)
        new_value[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
        unique_count = new_value.sum(dim=1)
        if (unique_count > 16).any():
            raise ValueError("A group contains more than 16 distinct W4 values")
        sorted_codes = new_value.cumsum(dim=1) - 1
        codes = torch.empty_like(sorted_codes, dtype=torch.uint8)
        codes.scatter_(1, order, sorted_codes.to(torch.uint8))
        codebook = torch.zeros((values.shape[0], 16), dtype=weight.dtype)
        rows, columns = new_value.nonzero(as_tuple=True)
        codebook[rows, sorted_codes[rows, columns]] = sorted_values[rows, columns]
        packed_parts.append(_pack_nibbles(codes))
        codebook_parts.append(codebook)

    return {
        "encoding": "int4_codebook_v1",
        "shape": tuple(weight.shape),
        "group_size": group_size,
        "packed": torch.cat(packed_parts),
        "codebook": torch.cat(codebook_parts),
    }


def pack_rtn(weight: torch.Tensor, group_size: int) -> dict:
    if weight.ndim != 2 or weight.shape[1] % group_size or group_size % 2:
        raise ValueError("Expected a matrix with an even group size dividing its input width")

    groups = weight.detach().float().view(-1, group_size)
    minimum = groups.amin(dim=1)
    maximum = groups.amax(dim=1)
    scales = (maximum - minimum).clamp_min(1e-8) / 15
    zero_points = (-minimum / scales).round().clamp(0, 15)
    codes = (groups / scales[:, None] + zero_points[:, None]).round().clamp(0, 15).to(torch.uint8)
    return {
        "encoding": "int4_rtn_v1",
        "shape": tuple(weight.shape),
        "group_size": group_size,
        "dtype": str(weight.dtype),
        "packed": _pack_nibbles(codes),
        "scales": scales,
        "zero_points": zero_points.to(torch.uint8),
    }


def pack_symmetric(weight: torch.Tensor, group_size: int = 128, batch_groups: int = 8192) -> dict:
    if weight.ndim != 2 or weight.shape[1] % group_size or group_size % 2:
        raise ValueError("Expected a matrix with an even group size dividing its input width")

    groups = weight.contiguous().view(-1, group_size)
    packed_parts = []
    scale_parts = []
    fallback_indices = []
    fallback_codebooks = []
    for start in range(0, groups.shape[0], batch_groups):
        values = groups[start : start + batch_groups]
        float_values = values.float()
        maximum = float_values.abs().amax(dim=1).clamp_min(1e-12)
        previous = torch.nextafter(values, torch.full_like(values, -float("inf"))).float()
        following = torch.nextafter(values, torch.full_like(values, float("inf"))).float()
        codes = torch.zeros_like(values, dtype=torch.uint8)
        scales = torch.zeros(values.shape[0], dtype=torch.float32)
        matched = torch.zeros(values.shape[0], dtype=torch.bool)

        for maximum_code in range(1, 9):
            candidate = maximum / maximum_code
            signed = (float_values / candidate[:, None]).round().clamp(-8, 7)
            divisor = signed.masked_fill(signed == 0, 1)
            lower = torch.minimum((previous + float_values) / (2 * divisor), (following + float_values) / (2 * divisor))
            upper = torch.maximum((previous + float_values) / (2 * divisor), (following + float_values) / (2 * divisor))
            lower.masked_fill_(signed == 0, -float("inf"))
            upper.masked_fill_(signed == 0, float("inf"))
            lower_bound = lower.amax(dim=1)
            upper_bound = upper.amin(dim=1)
            scale = (lower_bound + upper_bound) / 2
            valid = (
                ~matched
                & (lower_bound < upper_bound)
                & (((signed != 0) | (float_values == 0)).all(dim=1))
                & ((signed * scale[:, None]).to(values.dtype) == values).all(dim=1)
            )
            codes[valid] = (signed[valid] + 8).to(torch.uint8)
            scales[valid] = scale[valid]
            matched |= valid
            if matched.all():
                break

        missing = (~matched).nonzero(as_tuple=True)[0]
        if missing.numel():
            fallback = pack_codebook(values[missing], group_size)
            codes[missing] = _unpack_nibbles(fallback["packed"], group_size)
            fallback_indices.append(missing.to(torch.int32) + start)
            fallback_codebooks.append(fallback["codebook"])

        packed_parts.append(_pack_nibbles(codes))
        scale_parts.append(scales)

    return {
        "encoding": "int4_symmetric_v1",
        "shape": tuple(weight.shape),
        "group_size": group_size,
        "packed": torch.cat(packed_parts),
        "scales": torch.cat(scale_parts),
        "fallback_indices": torch.cat(fallback_indices) if fallback_indices else torch.empty(0, dtype=torch.int32),
        "fallback_codebooks": torch.cat(fallback_codebooks) if fallback_codebooks else torch.empty((0, 16), dtype=weight.dtype),
    }


def unpack_int4(record: dict) -> torch.Tensor:
    group_size = int(record["group_size"])
    codes = _unpack_nibbles(record["packed"], group_size)
    if record["encoding"] == "int4_rtn_v1":
        dtype = getattr(torch, record["dtype"].removeprefix("torch."))
        values = (codes.float() - record["zero_points"][:, None]) * record["scales"][:, None]
        return values.to(dtype).reshape(record["shape"])
    codebooks = record.get("codebook")
    if codebooks is not None:
        return codebooks.gather(1, codes.long()).reshape(record["shape"])

    fallback_codebooks = record["fallback_codebooks"]
    dtype = fallback_codebooks.dtype
    values = ((codes.float() - 8) * record["scales"][:, None]).to(dtype)
    fallback_indices = record["fallback_indices"].long()
    if fallback_indices.numel():
        values[fallback_indices] = fallback_codebooks.gather(1, codes[fallback_indices].long())
    return values.reshape(record["shape"])
