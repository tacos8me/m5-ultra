# SPDX-License-Identifier: MIT
"""Fit the verified oQ3e model in RAM: affine3/g128 experts and affine8/g64 embed/head.

Run only under the shared GPU lock. This candidate introduces a second rounding;
its quality must be checked. Original-weight conversion remains the preferred source.
"""

import argparse, copy, json, math, os, re, shutil, struct, time
from pathlib import Path
import mlx.core as mx
from .storage import TensorFile, decode_array
from .sharding import ShardWriter
from .mixed_convert import requantize


def target(name):
    if re.fullmatch(r"language_model\.layers\.\d+\.ffn\.experts\.w[123]", name):
        return 3, 128
    if name in ("language_model.embed", "language_model.head"):
        return 8, 64
    return None


def plan(source):
    source = Path(source)
    config = json.loads((source / "config.json").read_text())
    fmt = config["omlx_deepseek_v41"]
    if fmt["version"] != 1 or not fmt.get("preserve_mtp"):
        raise ValueError("Expected converted oMLX checkpoint with DSpark")
    mapping = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    headers = {}
    for file in set(mapping.values()):
        with (source / file).open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            headers[file] = json.loads(f.read(n))
    tables = copy.deepcopy(fmt["engram_tables"])
    excluded = set()
    for table in tables.values():
        excluded.update(
            table[k] for k in ("weight_key", "scale_key", "bias_key") if table.get(k)
        )
    modules = fmt["quantized_modules"]
    replaced = {name for name in modules if target(name)} | {
        "language_model.embed",
        "language_model.head",
    }
    size = 0
    for name, file in mapping.items():
        if name in excluded:
            continue
        module = name.rsplit(".", 1)[0]
        if module in replaced:
            if not name.endswith(".weight"):
                continue
            bits, group = target(module)
            old = modules.get(module)
            shape = headers[file][name]["shape"]
            width = shape[-1] * 32 // old["bits"] if old else shape[-1]
            if width % group:
                raise ValueError("Unsupported target group size: " + module)
            count = math.prod(shape[:-1]) * width
            size += count * bits // 8 + count // group * 4
        else:
            offsets = headers[file][name]["data_offsets"]
            size += offsets[1] - offsets[0]
    return config, mapping, tables, excluded, size


def convert(source, destination, dry_run=False):
    source, destination = Path(source), Path(destination)
    config, mapping, tables, excluded, size = plan(source)
    print(
        json.dumps(
            {
                "target_resident_bytes": size,
                "target_resident_gib": size / 1024**3,
                "guard_105pct_gib": size * 1.05 / 1024**3,
            }
        ),
        flush=True,
    )
    if size > 224 * 1024**3:
        raise ValueError("Candidate exceeds 224 GiB tensor cap")
    if dry_run:
        return
    if destination.exists():
        raise FileExistsError(destination)
    if shutil.disk_usage(destination.parent).free < size + 32 * 1024**3:
        raise RuntimeError("Insufficient disk headroom")
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    destination.mkdir()
    marker = destination / "conversion.inprogress.json"
    marker.write_text(
        json.dumps(
            {"source": str(source), "policy": "oq3e-to-q3g128-q8", "bytes": size}
        )
    )
    for table in tables.values():
        for key in ("weight_file", "scale_file"):
            file = table.get(key)
            if file:
                out = destination / "engram" / file
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    os.link(source / file, out)
                table[key] = str(out.relative_to(destination))
    specs = copy.deepcopy(config["omlx_deepseek_v41"]["quantized_modules"])
    writer = ShardWriter(destination, max_shard_bytes=3_000_000_000)
    consumed = set(excluded)
    total = 0
    started = time.monotonic()

    def read(names):
        readers = {}
        result = {}
        try:
            for name in names:
                file = mapping[name]
                if file not in readers:
                    readers[file] = TensorFile(source / file)
                result[name] = decode_array(*readers[file].read(name))
            mx.eval(result)
            return result
        finally:
            for r in readers.values():
                r.close()

    # Process each complete projection together so scales/biases cannot cross output shards.
    all_names = [k for k in mapping if k.endswith(".weight")] + [
        k for k in mapping if not k.endswith(".weight")
    ]
    for name in all_names:
        if name in consumed:
            continue
        module = name.removesuffix(".weight")
        old = specs.get(module) if name.endswith(".weight") else None
        names = (
            [name]
            + ([module + ".scales"] if old else [])
            + ([module + ".biases"] if old and old["mode"] == "affine" else [])
        )
        values = read(names)
        consumed.update(names)
        rule = target(module) if name.endswith(".weight") else None
        if rule:
            values, specs[module] = requantize(values, module, old, rule[0], rule[1])
        mx.eval(values)
        total += sum(x.nbytes for x in values.values())
        writer.add(values)
        mx.clear_cache()
        print(
            json.dumps(
                {
                    "bytes": total,
                    "elapsed_s": time.monotonic() - started,
                    "peak_gb": mx.get_peak_memory() / 1e9,
                }
            ),
            flush=True,
        )
    if total != size:
        raise ValueError(f"Inventory mismatch: {size} expected vs {total} exported")
    index = writer.finish()
    config["omlx_deepseek_v41"].update(
        quantized_modules=specs, engram_tables=tables, engram_in_index=False
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "LICENSE"):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index}, indent=2) + "\n"
    )
    (destination / "ds41-conversion.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "policy": "oq3e-to-q3g128-q8",
                "resident_tensor_bytes": total,
                "elapsed_s": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )
    marker.unlink()
    print("CONVERSION_COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    convert(args.source, args.destination, args.dry_run)
