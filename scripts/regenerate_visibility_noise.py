"""Regenerate noisy visibility files as clean visibility plus Gaussian noise."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np


VISIBILITY_RE = re.compile(
    r"^(?P<prefix>.*visibility_(?P<scene>\d+)_expand_(?P<expand>\d+)_)(?P<suffix>noised_\d+)\.npy$"
)


@dataclass(frozen=True)
class NoiseSummary:
    path: str
    scene: int
    expand: int
    sigma_real: float
    sigma_imag: float
    mean_real: float
    mean_imag: float
    clean_abs_delta_abs_corr: float


def parse_int_set(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def stable_seed(base_seed: int, path: str) -> int:
    digest = hashlib.blake2b(f"{base_seed}:{path}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def load_npy_from_zip(archive: ZipFile, name: str) -> np.ndarray:
    with archive.open(name) as handle:
        return np.load(handle)


def write_npy_bytes(array: np.ndarray) -> bytes:
    with tempfile.SpooledTemporaryFile() as handle:
        np.save(handle, array)
        handle.seek(0)
        return handle.read()


def should_regenerate(name: str, noise_suffix: str, expand_ids: set[int] | None) -> tuple[int, int] | None:
    match = VISIBILITY_RE.match(PurePosixPath(name).name)
    if match is None or match.group("suffix") != noise_suffix:
        return None
    expand = int(match.group("expand"))
    if expand_ids is not None and expand not in expand_ids:
        return None
    return int(match.group("scene")), expand


def clean_name_for(noisy_name: str, clean_suffix: str) -> str:
    path = PurePosixPath(noisy_name)
    match = VISIBILITY_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unsupported visibility filename: {noisy_name}")
    return str(path.with_name(f"{match.group('prefix')}{clean_suffix}.npy"))


def normalized_last_dim(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2 and array.shape[0] == 2 and array.shape[1] != 2:
        return array.T
    if array.ndim == 3 and array.shape[1] == 2 and array.shape[-1] != 2:
        return np.moveaxis(array, 1, -1).reshape(-1, 2)
    if array.ndim == 3 and array.shape[-1] == 2:
        return array.reshape(-1, 2)
    return array.reshape(-1, 2)


def noise_summary(path: str, scene: int, expand: int, clean: np.ndarray, noisy: np.ndarray) -> NoiseSummary:
    clean_ri = normalized_last_dim(clean).astype(np.float64)
    delta_ri = normalized_last_dim(noisy - clean).astype(np.float64)
    clean_amp = np.linalg.norm(clean_ri, axis=1)
    delta_amp = np.linalg.norm(delta_ri, axis=1)
    corr = float(np.corrcoef(clean_amp, delta_amp)[0, 1]) if clean_amp.size > 1 else float("nan")
    return NoiseSummary(
        path=path,
        scene=scene,
        expand=expand,
        sigma_real=float(delta_ri[:, 0].std()),
        sigma_imag=float(delta_ri[:, 1].std()),
        mean_real=float(delta_ri[:, 0].mean()),
        mean_imag=float(delta_ri[:, 1].mean()),
        clean_abs_delta_abs_corr=corr,
    )


def make_noisy(clean: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=clean.shape)
    return (clean.astype(np.float64) + noise).astype(clean.dtype, copy=False)


def aggregate_summaries(summaries: Iterable[NoiseSummary]) -> dict[str, float | int]:
    items = list(summaries)
    if not items:
        return {"num_regenerated": 0}
    return {
        "num_regenerated": len(items),
        "sigma_real_mean": float(np.mean([item.sigma_real for item in items])),
        "sigma_imag_mean": float(np.mean([item.sigma_imag for item in items])),
        "mean_real_mean": float(np.mean([item.mean_real for item in items])),
        "mean_imag_mean": float(np.mean([item.mean_imag for item in items])),
        "clean_abs_delta_abs_corr_mean": float(np.mean([item.clean_abs_delta_abs_corr for item in items])),
        "clean_abs_delta_abs_corr_max": float(np.max([item.clean_abs_delta_abs_corr for item in items])),
    }


def regenerate_zip(args: argparse.Namespace, expand_ids: set[int] | None) -> list[NoiseSummary]:
    compression = ZIP_STORED if args.compression == "stored" else ZIP_DEFLATED
    with ZipFile(args.input) as src:
        names = src.namelist()
        name_set = set(names)
        summaries: list[NoiseSummary] = []
        with ZipFile(args.output, "w", compression=compression, compresslevel=args.compresslevel) as dst:
            for name in names:
                parsed = should_regenerate(name, args.noise_suffix, expand_ids)
                if parsed is None:
                    dst.writestr(name, src.read(name))
                    continue
                clean_name = clean_name_for(name, args.clean_suffix)
                if clean_name not in name_set:
                    raise FileNotFoundError(f"Missing clean pair for {name}: {clean_name}")
                scene, expand = parsed
                clean = load_npy_from_zip(src, clean_name)
                noisy = make_noisy(clean, args.sigma, stable_seed(args.seed, name))
                dst.writestr(name, write_npy_bytes(noisy))
                summaries.append(noise_summary(name, scene, expand, clean, noisy))
    return summaries


def regenerate_directory(args: argparse.Namespace, expand_ids: set[int] | None) -> list[NoiseSummary]:
    input_root = args.input
    output_root = args.output
    summaries: list[NoiseSummary] = []
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace noisy files there.")
    output_root.mkdir(parents=True, exist_ok=True)
    for src_path in input_root.rglob("*"):
        rel = src_path.relative_to(input_root)
        dst_path = output_root / rel
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        parsed = should_regenerate(rel.as_posix(), args.noise_suffix, expand_ids)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if parsed is None:
            if not dst_path.exists() or args.overwrite:
                shutil.copy2(src_path, dst_path)
            continue
        clean_path = input_root / Path(clean_name_for(rel.as_posix(), args.clean_suffix))
        if not clean_path.exists():
            raise FileNotFoundError(f"Missing clean pair for {src_path}: {clean_path}")
        scene, expand = parsed
        clean = np.load(clean_path)
        noisy = make_noisy(clean, args.sigma, stable_seed(args.seed, rel.as_posix()))
        np.save(dst_path, noisy)
        summaries.append(noise_summary(rel.as_posix(), scene, expand, clean, noisy))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input srdata directory or zip.")
    parser.add_argument("--output", type=Path, required=True, help="Output srdata directory or zip.")
    parser.add_argument("--sigma", type=float, default=0.05, help="Gaussian std per real/imag component.")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--clean-suffix", default="noised_0")
    parser.add_argument("--noise-suffix", default="noised_1")
    parser.add_argument("--expand-ids", default=None, help="Comma-separated expand ids to regenerate. Default: all.")
    parser.add_argument("--compression", choices=["deflated", "stored"], default="deflated")
    parser.add_argument("--compresslevel", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.sigma <= 0.0:
        raise ValueError("--sigma must be positive.")
    if args.output.exists() and args.output.is_file() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists. Pass --overwrite to replace it.")

    expand_ids = parse_int_set(args.expand_ids)
    if args.input.suffix.lower() == ".zip":
        summaries = regenerate_zip(args, expand_ids)
    else:
        summaries = regenerate_directory(args, expand_ids)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "sigma": args.sigma,
        "seed": args.seed,
        "clean_suffix": args.clean_suffix,
        "noise_suffix": args.noise_suffix,
        "expand_ids": sorted(expand_ids) if expand_ids is not None else "all",
        **aggregate_summaries(summaries),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
