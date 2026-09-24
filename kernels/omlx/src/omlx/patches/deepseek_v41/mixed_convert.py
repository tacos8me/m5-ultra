# SPDX-License-Identifier: MIT
"""Stream 3-bit experts, native 8-bit projections, q8 embed/head, FP8 SSD Engram.

Run this inside lockf -k ~/llm/locks/gpu.lock. Source checkpoints remain immutable.
"""

import argparse, json, math, os, re, shutil, struct, time
from pathlib import Path
import mlx.core as mx
from .convert import iter_source_weights, source_engram_tables
from .sharding import ShardWriter

PROFILES = {
    "q3g128": {"expert_group": 128, "cap_gib": 224},
    "mixed23g64": {"expert_group": 64, "cap_gib": 210},
}


def policy(name, n_layers=40, profile="q3g128"):
    match = re.fullmatch(r"language_model\.layers\.(\d+)\.ffn\.experts\.w([123])", name)
    if match:
        layer, projection = map(int, match.groups())
        if profile == "q3g128":
            return 3
        # Protect every down projection and boundary/diagnostically fragile layers.
        # This is a memory-constrained starting policy, not calibrated importance.
        return (
            3
            if projection == 2 or layer < 4 or layer >= n_layers - 4 or layer == 15
            else 2
        )
    if name in ("language_model.embed", "language_model.head"):
        return 8
    return None


def requantize(values, name, spec, bits, group_size=64):
    weight = values[name + ".weight"]
    nexpert = weight.shape[0] if weight.ndim == 3 else 1
    width = weight.shape[-1] * 32 // spec["bits"] if spec else weight.shape[-1]
    # Multiply before integer division: affine3 packing has 3 words per 32 values.
    chunk_rows = max(1, (64 * 1024**2) // (2 * width))
    packed = []
    scales = []
    biases = []
    for expert in range(nexpert):
        part = weight[expert] if weight.ndim == 3 else weight
        qs = []
        ss = []
        bs = []
        for start in range(0, part.shape[0], chunk_rows):
            stop = min(part.shape[0], start + chunk_rows)
            x = part[start:stop]
            if spec:

                def sl(suffix):
                    a = values.get(name + suffix)
                    if a is None:
                        return None
                    return (a[expert] if weight.ndim == 3 else a)[start:stop]

                x = mx.dequantize(
                    x,
                    sl(".scales"),
                    sl(".biases"),
                    group_size=spec.get("group_size", 32),
                    bits=spec["bits"],
                    mode=spec["mode"],
                )
            if (
                bits == 3
                and group_size == 128
                and os.environ.get("DS41_QUANT_LSQ") == "1"
            ):
                from .lsq_quant import quantize

                q, s, b = quantize(x.astype(mx.bfloat16))
            else:
                q, s, b = mx.quantize(
                    x.astype(mx.bfloat16), group_size=group_size, bits=bits
                )
            mx.eval(q, s, b)
            qs.append(q)
            ss.append(s)
            bs.append(b)
        packed.append(mx.concatenate(qs))
        scales.append(mx.concatenate(ss))
        biases.append(mx.concatenate(bs))
    join = mx.stack if weight.ndim == 3 else lambda xs: xs[0]
    result = {
        name + ".weight": join(packed),
        name + ".scales": join(scales),
        name + ".biases": join(biases),
    }
    mx.eval(result)
    return result, {
        "bits": bits,
        "group_size": group_size,
        "mode": "affine",
        "quantize_input": bool(spec and spec.get("quantize_input", True)),
    }


def inventory(source, mapping, n_layers=40, profile="q3g128"):
    expert_group = PROFILES[profile]["expert_group"]
    headers = {}
    for file in set(mapping.values()):
        with (source / file).open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            headers[file] = json.loads(f.read(n))
    resident = 0
    engram_files = set()
    details = {}
    for name, file in mapping.items():
        t = headers[file][name]
        if ".engram.embed." in name:
            engram_files.add(file)
            continue
        if name.endswith(".scale"):
            continue
        shape = t["shape"]
        count = math.prod(shape)
        dtype = t["dtype"]
        elements = count * (2 if dtype in ("I8", "U8") else 1)
        if re.fullmatch(r"layers.\d+.ffn.experts.\d+.w[123].weight", name):
            module = "language_model." + re.sub(
                r".experts.\d+.", ".experts.", name.removesuffix(".weight")
            )
            bits = policy(module, n_layers, profile)
            size = elements * bits // 8 + elements // expert_group * 4
        elif name in ("embed.weight", "head.weight"):
            size = elements + elements // 64 * 4
        elif dtype.startswith("F8_E4M3"):
            size = (
                elements * 2
                if name.endswith("wo_a.weight")
                or ".markov_head." in name
                or (name.removesuffix(".weight") + ".bias") in mapping
                else elements + elements // 32
            )
        elif dtype in ("I8", "U8"):
            size = (
                elements * 2
                if (name.removesuffix(".weight") + ".bias") in mapping
                else elements // 2 + elements // 32
            )
        else:
            size = t["data_offsets"][1] - t["data_offsets"][0]
        resident += size
        family = "draft" if name.startswith("mtp.") else "backbone"
        details[family] = details.get(family, 0) + size
    return {
        "estimated_resident_tensor_bytes": resident,
        "families": details,
        "engram_link_bytes": sum((source / f).stat().st_size for f in engram_files),
    }


def convert(source, destination, dry_run=False, profile="q3g128"):
    options = PROFILES[profile]
    source, destination = Path(source), Path(destination)
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "deepseek_v41" or "omlx_deepseek_v41" in config:
        raise ValueError("Expected the original DeepSeek V4.1 checkpoint")
    mapping = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    text = config.get("text_config", config)
    n_layers = text.get("num_hidden_layers", text.get("n_layers", 40))
    budget = inventory(source, mapping, n_layers, profile)
    print(json.dumps(budget), flush=True)
    if budget["estimated_resident_tensor_bytes"] > options["cap_gib"] * 1024**3:
        raise ValueError(
            f"Quant exceeds the {options['cap_gib']} GiB resident tensor cap"
        )
    if dry_run:
        return budget
    if destination.exists():
        raise FileExistsError(destination)
    if (
        shutil.disk_usage(destination.parent).free
        < budget["estimated_resident_tensor_bytes"] + 16 * 1024**3
    ):
        raise RuntimeError("Insufficient disk headroom for mixed quant")
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    destination.mkdir()
    (destination / "engram").mkdir()
    marker = destination / "conversion.inprogress.json"
    marker.write_text(
        json.dumps(
            {
                "source": str(source),
                "policy": profile + "-native8-embedhead8-engramfp8",
                "budget": budget,
            },
            indent=2,
        )
    )
    tables = source_engram_tables(mapping, config)
    for table in tables.values():
        for key in ("weight_file", "scale_file"):
            file = table.get(key)
            if not file:
                continue
            output = destination / "engram" / file
            if not output.exists():
                os.link(source / file, output)
            table[key] = str(output.relative_to(destination))
    writer = ShardWriter(destination, max_shard_bytes=3_000_000_000)
    quantized = {}
    written = 0
    start = time.monotonic()
    for i, (values, specs) in enumerate(
        iter_source_weights(source, config, mapping, preserve_mtp=True)
    ):
        for key in list(values):
            if not key.endswith(".weight"):
                continue
            name = key[:-7]
            bits = policy(name, n_layers, profile)
            if bits is None:
                continue
            group_size = options["expert_group"] if ".ffn.experts." in name else 64
            converted, spec = requantize(
                values, name, specs.get(name), bits, group_size
            )
            values.update(converted)
            specs[name] = spec
        mx.eval(values)
        size = sum(v.nbytes for v in values.values())
        written += size
        writer.add(values)
        quantized.update(specs)
        mx.clear_cache()
        print(
            json.dumps(
                {
                    "groups": i + 1,
                    "written_bytes": written,
                    "elapsed_s": time.monotonic() - start,
                    "peak_gb": mx.get_peak_memory() / 1e9,
                }
            ),
            flush=True,
        )
    if written != budget["estimated_resident_tensor_bytes"]:
        raise ValueError(
            f"Exported bytes {written} differ from independent inventory {budget['estimated_resident_tensor_bytes']}"
        )
    result_map = writer.finish()
    config.pop("quantization_config", None)
    config["omlx_deepseek_v41"] = {
        "version": 1,
        "quantized_modules": quantized,
        "engram_tables": tables,
        "preserve_mtp": True,
        "excluded_draft_tensors": 0,
    }
    for name in ("tokenizer.json", "tokenizer_config.json", "LICENSE"):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": result_map}, indent=2) + "\n"
    )
    report = {
        "policy": profile + "-native8-embedhead8-engramfp8",
        "source": str(source),
        "budget": budget,
        "actual_resident_tensor_bytes": written,
        "quantizer": "lsq" if os.environ.get("DS41_QUANT_LSQ") == "1" else "minmax",
        "groups": len(quantized),
        "elapsed_s": time.monotonic() - start,
    }
    (destination / "ds41-conversion.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    marker.unlink()
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--profile", choices=tuple(PROFILES), default="q3g128")
    args = p.parse_args()
    print(
        json.dumps(convert(args.source, args.destination, args.dry_run, args.profile)),
        flush=True,
    )
