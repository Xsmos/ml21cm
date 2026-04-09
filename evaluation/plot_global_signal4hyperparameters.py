import os
import sys
import argparse
import re
import textwrap
from typing import Dict, Any, List, Tuple, Set, Optional

import h5py
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from torch.utils.data import DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.load_h5 import Dataset4h5, ranges_dict

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 17,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 13,
        "legend.title_fontsize": 13,
        "figure.titlesize": 20,
    }
)

FS_TITLE = plt.rcParams["axes.titlesize"]
FS_LABEL = plt.rcParams["axes.labelsize"]
FS_TICK = plt.rcParams["xtick.labelsize"]
FS_LEGEND = plt.rcParams["legend.fontsize"]
FS_TEXT = plt.rcParams["font.size"]
FS_GROUP = plt.rcParams["legend.title_fontsize"]


# Manually specify the experiment registry here.
# key: jobID, value: primary hyperparameters for legend labels.
JOBID_HPARAMS: Dict[int, Dict[str, Any]] = {
    49654214: {
        "num_res_blocks": 1,
    # 46941305: {
    #     "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 2,
        "epochs": 120,
        "transform": "arcsinh",
    },
    49654134: {
        "num_res_blocks": 1,
    # 49299747: {
    #     "num_res_blocks": 3,
    # 46941303: {
        # "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 2,
        "epochs": 120,
        "transform": "min_max",
    },
    49654128: {
        "num_res_blocks": 1,
    # 46941293: {
    #     "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 2,
        "epochs": 120,
        "transform": "z_score",
    },
    49654199: {
        "num_res_blocks": 1,
    # 46941286: {
    #     "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 2,
        "epochs": 120,
        "transform": "pt_inv",
    },
    49653881: {
        "num_res_blocks": 1,
    # 49299542: {
    #     "num_res_blocks": 3,
    # 47032706: {
        # "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 3,
        "epochs": 120,
        "transform": "min_max",
    },
    49653904: {
        "num_res_blocks": 1,
    # 47032672: {
    #     "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 3,
        "epochs": 120,
        "transform": "z_score",
    },
    # 47032656: {
    #     "num_res_blocks": 3,
    #     "squish": "1,0",
    #     "dim": 3,
    #     "epochs": 120,
    #     "transform": "pt_inv",
    # },
    48820329: {
        "num_res_blocks": 1,
    # 48057253: {
    #     "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 3,
        "epochs": 120,
        "transform": "pt_inv",
    },
    48902106: {
        "num_res_blocks": 1,
    # 48057168: {
    #     "num_res_blocks": 3,
        "squish": "0.5,0",
        "dim": 3,
        "epochs": 120,
        "transform": "pt_inv",
    },
    49325389: {
        "num_res_blocks": 1,
        "squish": "0.01,0",
        "dim": 3,
        "epochs": 120,
        "transform": "pt_inv",
    },
    48057143: {
        "num_res_blocks": 3,
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 120,
        "transform": "pt_inv",
    },
    48580330: {
        "num_res_blocks": 2,
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 120,
        "transform": "pt_inv",
    },
    48820652: {
    # 48436662: {
        "num_res_blocks": 1,
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 240,
        "transform": "pt_inv",
    },
    48436662: {
        "num_res_blocks": 1, # baseline
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 120,
        "transform": "pt_inv",
    },
    # 48902183: {
    #     "num_res_blocks": 1,
    #     "squish": "0.1,1",
    #     # "squish": "0.1,0",
    #     "dim": 3,
    #     "epochs": 120,
    #     "transform": "pt_inv",
    # },
    48820480: {
        "num_res_blocks": 1,
    # 47908550: {
    #     "num_res_blocks": 3,
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 60,
        "transform": "pt_inv",
    },
    48820424: {
    # 47356556: {
        # "num_res_blocks": 3,
        "num_res_blocks": 1,
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 30,
        "transform": "pt_inv",
    },
}
BASELINE_JOBID = 48436662
# Selection by 1-based index in JOBID_HPARAMS insertion order.
PDF_JOB_INDICES_1BASED = [5, 6, 7]
MAIN_PLOT_JOB_INDICES_1BASED = [5, 6, 7, 8, 9, 10, 11, 13]

# MAE trend grouping (1-based job index as shown on x-axis):
# used only for visualization aids in global_signal_hparams_mae_trend.pdf.
MAE_GROUPS = [
    {"name": "2D Transform", "indices_1based": [1, 2, 3, 4], "color": "#4C78A8", "row": 1},
    # job index 7 was removed from the original 16-job layout; indices below are
    # the current 15-job layout shown on x-axis (1..15).
    {"name": "3D Transform", "indices_1based": [5, 6, 7], "color": "#F58518", "row": 2},
    {"name": "3D + Amplitude Scale", "indices_1based": [7, 8, 9, 13], "color": "#54A24B", "row": 1},
    {"name": "3D + ResBlocks", "indices_1based": [10, 11, 13], "color": "#B279A2", "row": 2},
    {"name": "3D + Epoch", "indices_1based": [12, 13, 14, 15], "color": "#E45756", "row": 0},
]
# Optional section separators (1-based boundary after index i).
# Helps readability without changing existing job order.
MAE_SECTION_BOUNDARIES_1BASED = [4, 7]


def _jobids_from_indices_1based(
    indices_1based: List[int],
    registry: Dict[int, Dict[str, Any]],
    source_name: str,
) -> List[int]:
    ordered_jobids = list(registry.keys())
    selected = []
    for idx1 in indices_1based:
        idx0 = idx1 - 1
        if 0 <= idx0 < len(ordered_jobids):
            selected.append(ordered_jobids[idx0])
        else:
            print(
                f"Skipped {source_name} index={idx1}: out of range for {len(ordered_jobids)} configured jobs."
            )
    return selected


def load_h5_as_tensor(
    dir_name: str,
    num_image: int,
    num_redshift: int = 1024,
    HII_DIM: int = 64,
    z_step: int = 1,
    scale_path: bool = False,
    dim: int = 3,
    startat: int = 0,
):
    if not os.path.isabs(dir_name):
        dir_name = os.path.join(os.environ["SCRATCH"], dir_name)

    dataset = Dataset4h5(
        dir_name,
        num_image=num_image,
        num_redshift=num_redshift,
        HII_DIM=HII_DIM,
        z_step=z_step,
        scale_path=scale_path,
        dim=dim,
        startat=startat,
    )

    with h5py.File(dir_name, "r") as f:
        los = f["redshifts_distances"][:, startat : startat + dataset.num_redshift : z_step]

    dataloader = DataLoader(dataset, batch_size=min(num_image, 1024), shuffle=False)
    x, c = next(iter(dataloader))

    return x, c, los


def load_x_ml(
    target_pattern: str,
    jobid: int,
    num_image: int,
    ema: int = 0,
    outputs_dir: str = "../training/outputs",
    pt_fname: str = None,
    transform: str = "pt_inv",
    apply_inverse: bool = True,
):
    fnames = [
        fname
        for fname in os.listdir(outputs_dir)
        if target_pattern in fname and str(jobid) in fname and f"-ema{ema}" in fname
    ]

    if len(fnames) == 0:
        raise ValueError(
            f"No files found for target_pattern={target_pattern}, jobid={jobid}, ema={ema} in {outputs_dir}"
        )

    x_ml = []
    loaded = 0
    for fname in sorted(fnames):
        data = np.load(os.path.join(outputs_dir, fname))
        if loaded >= num_image:
            break
        remaining = num_image - loaded
        if data.shape[0] > remaining:
            data = data[:remaining]
        x_ml.append(data)
        loaded += data.shape[0]

    if loaded == 0:
        raise ValueError(
            f"Matched files exist but no data loaded for target_pattern={target_pattern}, jobid={jobid}."
        )

    x_ml = np.concatenate(x_ml, axis=0)
    original_shape = x_ml.shape

    if apply_inverse and transform == "pt_inv" and pt_fname is not None:
        pt = joblib.load(pt_fname)
        x_ml = pt.inverse_transform(x_ml.reshape(-1, 1))
    elif apply_inverse and transform == "min_max":
        min_val, max_val = ranges_dict[transform]
        x_ml = (x_ml + 1) / 2
        x_ml = x_ml * (max_val - min_val) + min_val
    elif apply_inverse and transform == "z_score":
        mean_val, std_val = ranges_dict[transform]
        x_ml = x_ml * std_val + mean_val
    elif apply_inverse and transform == "arcsinh":
        x_ml = np.sinh(x_ml)

    x_ml = torch.from_numpy(x_ml.reshape(*original_shape))
    return x_ml


def x2Tb(x: torch.Tensor):
    if x.ndim == 4:
        return x[:, 0].mean(axis=1)
    if x.ndim == 5:
        return x[:, 0].mean(axis=(1, 2))
    raise ValueError(f"Unsupported tensor ndim: {x.ndim}")


def derive_target_pattern_from_real_h5(real_h5: str) -> str:
    base = os.path.basename(real_h5)
    m = re.search(r"Tvir([0-9]*\.?[0-9]+)-zeta([0-9]*\.?[0-9]+)", base)
    if m is None:
        raise ValueError(
            "Cannot derive target pattern from --real_h5. Expected filename containing "
            "'Tvir<value>-zeta<value>', e.g. LEN...-Tvir4.699-zeta30-....h5"
        )
    tvir = float(m.group(1))
    zeta = float(m.group(2))
    return f"Tvir{tvir:.3f}-zeta{zeta:.3f}"


def _normalize_num_str(v: Any) -> str:
    s = str(v).strip()
    try:
        return f"{float(s):g}"
    except ValueError:
        return s


def _parse_squish(v: Any) -> tuple[str, str]:
    # squish is stored as "A,k" in current experiments.
    s = str(v).replace(" ", "")
    parts = s.split(",")
    if len(parts) >= 2:
        return _normalize_num_str(parts[0]), _normalize_num_str(parts[1])
    return _normalize_num_str(s), "0"


def _abbr_transform(v: Any) -> str:
    # Short aliases for legend compactness.
    mapping = {
        "pt_inv": "YJ",   # Yeo-Johnson inverse
        "min_max": "MM",
        "z_score": "ZS",
        "arcsinh": "AS",
    }
    return mapping.get(str(v), str(v))


def _param_cmp_value(k: str, v: Any) -> Any:
    if k == "squish":
        return _parse_squish(v)
    if k == "transform":
        return _abbr_transform(v)
    return str(v)


def _param_token(k: str, v: Any) -> str:
    # Legend shorthand:
    # RB=num_res_blocks, A=squish(A,k; k=0 omitted), D=dim, Ep=epochs, T=transform.
    if k == "num_res_blocks":
        return f"RB{v}"
    if k == "squish":
        a, squish_k = _parse_squish(v)
        if squish_k == "0":
            return f"A{a}"
        return f"A{a},k{squish_k}"
    if k == "dim":
        return f"D{v}"
    if k == "epochs":
        return f"Ep{v}"
    if k == "transform":
        return f"T{_abbr_transform(v)}"
    return f"{k}={v}"


def _ordered_keys(hyperparams: Dict[str, Any]) -> list[str]:
    preferred = ["num_res_blocks", "squish", "dim", "epochs", "transform"]
    keys = [k for k in preferred if k in hyperparams]
    keys.extend([k for k in hyperparams.keys() if k not in preferred])
    return keys


def _format_model_label(
    jobid: int,
    hyperparams: Dict[str, Any],
    baseline_hyperparams: Dict[str, Any],
    show_jobid: bool = False,
) -> str:
    prefix = f"job={jobid}: " if show_jobid else ""
    if not hyperparams:
        return f"job={jobid}" if show_jobid else "config"

    if jobid == BASELINE_JOBID:
        tokens = [_param_token(k, hyperparams[k]) for k in _ordered_keys(hyperparams)]
        return f"{prefix}{' '.join(tokens)}"

    diff_tokens = []
    for k in _ordered_keys(hyperparams):
        v = hyperparams[k]
        if k not in baseline_hyperparams or _param_cmp_value(k, v) != _param_cmp_value(k, baseline_hyperparams[k]):
            diff_tokens.append(_param_token(k, v))

    if not diff_tokens:
        return f"{prefix}same as BASE" if show_jobid else "same as BASE"
    return f"{prefix}{' '.join(diff_tokens)}"


def _format_job_diff_text(
    jobid: int,
    hyperparams: Dict[str, Any],
    baseline_hyperparams: Dict[str, Any],
    show_jobid: bool = False,
) -> str:
    prefix = f"{jobid}: " if show_jobid else ""
    if jobid == BASELINE_JOBID:
        if not hyperparams:
            return f"{jobid}" if show_jobid else "BASE"
        tokens = [_param_token(k, hyperparams[k]) for k in _ordered_keys(hyperparams)]
        return (f"{jobid}: " if show_jobid else "") + " ".join(tokens)
    if not hyperparams:
        return f"{jobid}" if show_jobid else "config"

    diff_items = []
    for k in _ordered_keys(hyperparams):
        v = hyperparams[k]
        if k not in baseline_hyperparams or _param_cmp_value(k, v) != _param_cmp_value(k, baseline_hyperparams[k]):
            diff_items.append(_param_token(k, v))

    if not diff_items:
        return f"{jobid} (same)" if show_jobid else "same as BASE"
    return prefix + " ".join(diff_items)


def _wrap_label_text(s: str, width: int = 28) -> str:
    parts = [p.strip() for p in s.split(",")]
    lines = []
    cur = ""
    for p in parts:
        token = p if cur == "" else ", " + p
        if len(cur) + len(token) <= width:
            cur += token
        else:
            if cur:
                lines.append(cur)
            if len(p) > width:
                wrapped = textwrap.wrap(p, width=width)
                lines.extend(wrapped[:-1])
                cur = wrapped[-1] if wrapped else ""
            else:
                cur = p
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _generate_distinct_colors(n: int):
    if n <= 0:
        return []
    if n <= 20:
        cmap = plt.get_cmap("tab20", n)
        return [cmap(i) for i in range(n)]
    # Fallback for many jobs: sample a continuous map uniformly.
    cmap = plt.get_cmap("turbo")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def _symlog_forward(x, linthresh: float = 1.0):
    x = np.asarray(x)
    ax = np.abs(x)
    y = np.where(ax <= linthresh, ax / linthresh, 1.0 + np.log10(ax / linthresh))
    return np.sign(x) * y


def _symlog_inverse(y, linthresh: float = 1.0):
    y = np.asarray(y)
    ay = np.abs(y)
    x = np.where(ay <= 1.0, ay * linthresh, (10 ** (ay - 1.0)) * linthresh)
    return np.sign(y) * x


def _make_symlog_bins(xmin: float, xmax: float, n_bins: int = 180, linthresh: float = 1.0):
    umin = _symlog_forward(xmin, linthresh=linthresh)
    umax = _symlog_forward(xmax, linthresh=linthresh)
    if np.isclose(umin, umax):
        umax = umin + 1e-6
    u_edges = np.linspace(umin, umax, n_bins + 1)
    return _symlog_inverse(u_edges, linthresh=linthresh)


def _infer_job_dim(meta: Dict[str, Any], fallback_dim: int = 3) -> int:
    if "dim" in meta:
        try:
            return int(meta["dim"])
        except Exception:
            pass
    stride = meta.get("stride", None)
    if isinstance(stride, (list, tuple)):
        return len(stride)
    if isinstance(stride, str):
        return len([s for s in stride.split("-") if s.strip() != ""])
    return int(fallback_dim)


def _forward_transform_truth_for_job(x_true: torch.Tensor, transform: str, pt_fname: str = None) -> torch.Tensor:
    x_np = x_true.numpy()
    orig_shape = x_np.shape
    if transform == "pt_inv":
        if pt_fname is None:
            raise ValueError("pt_fname is required to forward-transform truth for 'pt_inv' jobs.")
        pt = joblib.load(pt_fname)
        x_np = pt.transform(x_np.reshape(-1, 1)).reshape(orig_shape)
    elif transform == "min_max":
        min_val, max_val = ranges_dict["min_max"]
        x_np = (x_np - min_val) / (max_val - min_val)
        x_np = x_np * 2 - 1
    elif transform == "z_score":
        mean_val, std_val = ranges_dict["z_score"]
        x_np = (x_np - mean_val) / std_val
    elif transform == "arcsinh":
        x_np = np.arcsinh(x_np)
    return torch.from_numpy(x_np.astype(np.float32))


def _inverse_transform_sampled_for_job(x_ml_raw: torch.Tensor, transform: str, pt_fname: str = None) -> torch.Tensor:
    x_np = x_ml_raw.numpy()
    orig_shape = x_np.shape
    if transform == "pt_inv":
        if pt_fname is None:
            raise ValueError("pt_fname is required to inverse-transform sampled data for 'pt_inv' jobs.")
        pt = joblib.load(pt_fname)
        x_np = pt.inverse_transform(x_np.reshape(-1, 1)).reshape(orig_shape)
    elif transform == "min_max":
        min_val, max_val = ranges_dict["min_max"]
        x_np = (x_np + 1) / 2
        x_np = x_np * (max_val - min_val) + min_val
    elif transform == "z_score":
        mean_val, std_val = ranges_dict["z_score"]
        x_np = x_np * std_val + mean_val
    elif transform == "arcsinh":
        x_np = np.sinh(x_np)
    return torch.from_numpy(x_np.astype(np.float32))


def plot_pixel_pdf_by_job_transform(
    x_true_by_job: Dict[int, torch.Tensor],
    x_ml_raw_by_job: Dict[int, torch.Tensor],
    model_meta: Dict[int, Dict[str, Any]],
    pt_fname: str = None,
    savename: str = "hparams_pdf.pdf",
    show_jobid: bool = False,
):
    if not x_ml_raw_by_job:
        print("No selected dim=3 PDF jobs found; skipped.")
        return

    jobids = list(x_ml_raw_by_job.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), dpi=220, sharey=True)
    ax_r, ax_l = axes
    testing_lw = 1.5 + 1.0
    diffusion_lw = 1.2 + 1.0
    raw_testing_lw = 1.8 + 1.0
    job_handles = []
    linthresh_tf = 1.0
    linthresh_raw = 1.0
    first_jobid = jobids[0]
    x_true_ref = x_true_by_job[first_jobid].numpy().reshape(-1)
    t_by_job = {}
    m_by_job = {}
    m_inv_by_job = {}

    for i, jobid in enumerate(jobids):
        color = f"C{i}"
        transform = model_meta.get(jobid, {}).get("transform", "pt_inv")
        x_true = x_true_by_job[jobid]
        x_ml_raw = x_ml_raw_by_job[jobid]

        x_true_t = _forward_transform_truth_for_job(x_true, transform=transform, pt_fname=pt_fname)
        t = x_true_t.numpy().reshape(-1)
        m = x_ml_raw.numpy().reshape(-1)
        t_by_job[jobid] = t
        m_by_job[jobid] = m

        transform_label = {
            "pt_inv": "yeo-johnson",
            "min_max": "min-max",
            "z_score": "z-score",
            "arcsinh": "arcsinh",
        }.get(str(transform), str(transform))
        label = f"job={jobid}: {transform_label}" if show_jobid else transform_label
        job_handles.append(Line2D([0], [0], color=color, lw=diffusion_lw, linestyle="-", label=label))

        # Left subplot: inverse-transformed sampled vs raw 21cmfast space.
        m_inv = _inverse_transform_sampled_for_job(x_ml_raw, transform=transform, pt_fname=pt_fname).numpy().reshape(-1)
        m_inv_by_job[jobid] = m_inv

    tf_lows = []
    tf_highs = []
    for jobid in jobids:
        tf_lows.append(np.percentile(t_by_job[jobid], 0.1))
        tf_lows.append(np.percentile(m_by_job[jobid], 0.1))
        tf_highs.append(np.percentile(t_by_job[jobid], 99.9))
        tf_highs.append(np.percentile(m_by_job[jobid], 99.9))
    bins_l = _make_symlog_bins(min(tf_lows), max(tf_highs), n_bins=180, linthresh=linthresh_tf)

    for i, jobid in enumerate(jobids):
        color = f"C{i}"
        ax_l.hist(t_by_job[jobid], bins=bins_l, density=True, histtype="step", linewidth=testing_lw, linestyle="--", color=color)
        ax_l.hist(m_by_job[jobid], bins=bins_l, density=True, histtype="step", linewidth=diffusion_lw, linestyle="-", color=color)

    raw_lows = [np.percentile(x_true_ref, 0.1)]
    raw_highs = [np.percentile(x_true_ref, 99.9)]
    for jobid in jobids:
        raw_lows.append(np.percentile(m_inv_by_job[jobid], 0.1))
        raw_highs.append(np.percentile(m_inv_by_job[jobid], 99.9))
    bins_r = _make_symlog_bins(min(raw_lows), max(raw_highs), n_bins=180, linthresh=linthresh_raw)

    for i, jobid in enumerate(jobids):
        color = f"C{i}"
        ax_r.hist(m_inv_by_job[jobid], bins=bins_r, density=True, histtype="step", linewidth=diffusion_lw, linestyle="-", color=color)

    # Raw testing-set reference (same space as inverse-transformed sampled).
    ax_r.hist(
        x_true_ref,
        bins=bins_r,
        density=True,
        histtype="step",
        linewidth=raw_testing_lw,
        linestyle="--",
        color="black",
        label="21cmfast",
    )

    ax_l.set_yscale("log")
    ax_l.set_ylim(bottom=1e-3)
    ax_l.set_xscale("symlog", linthresh=linthresh_tf)
    ax_l.grid(alpha=0.35)
    ax_l.set_xlabel("voxel value", fontsize=FS_LABEL)
    ax_l.set_title("Transformed voxel space", fontsize=FS_TITLE)
    ax_r.set_ylabel("PDF", fontsize=FS_LABEL)
    ax_l.tick_params(axis="both", labelsize=FS_TICK)

    ax_r.set_yscale("log")
    ax_r.set_ylim(bottom=1e-3)
    ax_r.set_xscale("symlog", linthresh=linthresh_raw)
    ax_r.grid(alpha=0.35)
    ax_r.set_xlabel("voxel value", fontsize=FS_LABEL)
    ax_r.set_title("Raw voxel space", fontsize=FS_TITLE)
    ax_r.tick_params(axis="both", labelsize=FS_TICK)

    legend_jobs = ax_r.legend(
        handles=job_handles,
        fontsize=FS_LEGEND,
        loc="upper right",
        framealpha=0.85,
    )
    ax_r.add_artist(legend_jobs)

    style_handles = [
        Line2D([0], [0], color="black", lw=testing_lw, linestyle="--", label="21cmfast"),
        Line2D([0], [0], color="black", lw=diffusion_lw, linestyle="-", label="diffusion"),
    ]
    legend_style = ax_r.legend(
        handles=style_handles,
        fontsize=FS_LEGEND,
        loc="upper left",
        framealpha=0.85,
    )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)
    if savename:
        plt.savefig(savename, bbox_inches="tight")
        print(f"Saved figure to {savename}")
        plt.close()
    else:
        plt.show()


def plot_global_signal_hyperparameters(
    x_true_by_job: Dict[int, torch.Tensor],
    x_ml_by_job: Dict[int, torch.Tensor],
    los_by_job: Dict[int, np.ndarray],
    model_meta: Dict[int, Dict[str, Any]],
    sigma_level: float = 68.27,
    y_eps: float = 1.0,
    lw: float = 0.8,
    alpha_true: float = 0.20,
    z_idx: int = None,
    savename: str = None,
    show_jobid: bool = False,
    main_plot_jobids: Optional[List[int]] = None,
):
    low = (100 - sigma_level) / 2
    high = 100 - low

    fig, (ax_left, ax_delta) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11, 8.5),
        dpi=220,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    jobids_in_order = []
    mae_rel_by_job = []
    mae_std_by_job = []
    mae_sigma_by_job = []

    line_lw = lw + 1.0
    ref_line_lw = 1.7 + 1.0
    legend_line_lw = 1.5 + 1.0
    marker_line_lw = 1.5 + 1.0
    zero_line_lw = 1.0 + 1.0

    baseline_hyperparams = model_meta.get(BASELINE_JOBID, {})
    handles_for_legend = [Line2D([0], [0], color="black", linestyle=":", lw=ref_line_lw, label="21cmFAST median")]
    baseline_label = _format_model_label(
        BASELINE_JOBID,
        model_meta.get(BASELINE_JOBID, {}),
        baseline_hyperparams,
        show_jobid=show_jobid,
    )

    z_other_band = 1
    z_other_line = 2
    z_baseline_band = 5
    z_baseline_line = 6
    z_true_band = 7
    z_true_line = 8
    skipped_in_main_plot = []
    main_plot_jobids_set: Optional[Set[int]] = set(main_plot_jobids) if main_plot_jobids is not None else None
    delta_plot_records = []
    plotted_color_idx = 0
    delta_ref_x_axis = None
    delta_ref_low = None
    delta_ref_high = None

    for idx, (jobid, x_ml) in enumerate(x_ml_by_job.items()):
        x_true = x_true_by_job[jobid]
        los = los_by_job[jobid]

        tb_true = x2Tb(x_true)
        y_true = np.median(tb_true, axis=0)
        perc_true = np.percentile(tb_true, [low, high], axis=0)
        sigma_true = 0.5 * (perc_true[1] - perc_true[0])
        x_axis = los[1, : y_true.shape[0]]

        if idx == 0:
            ax_left.fill_between(
                x_axis,
                perc_true[0],
                perc_true[1],
                alpha=alpha_true,
                facecolor="black",
                edgecolor="black",
                label="21cmFAST CI",
                zorder=z_true_band,
            )
            ax_left.plot(
                x_axis,
                y_true,
                linestyle=":",
                c="black",
                lw=ref_line_lw,
                label="21cmFAST median",
                zorder=z_true_line,
            )
            delta_ref_x_axis = x_axis.copy()
            delta_ref_low = (perc_true[0] - y_true).copy()
            delta_ref_high = (perc_true[1] - y_true).copy()

        tb_ml = x2Tb(x_ml)
        y_ml = np.median(tb_ml, axis=0)
        perc_ml = np.percentile(tb_ml, [low, high], axis=0)
        sigma_ml = 0.5 * (perc_ml[1] - perc_ml[0])
        tb_delta = tb_ml - tb_true
        y_delta = np.median(tb_delta, axis=0)
        perc_delta = np.percentile(tb_delta, [low, high], axis=0)

        show_in_main_plot = main_plot_jobids_set is None or jobid in main_plot_jobids_set
        if show_in_main_plot:
            color = f"C{plotted_color_idx}"
            plotted_color_idx += 1
            ax_left.fill_between(
                x_axis,
                perc_ml[0],
                perc_ml[1],
                color=color,
                alpha=0.10,
                linewidth=0,
                zorder=z_baseline_band if jobid == BASELINE_JOBID else z_other_band,
            )
            ax_left.plot(
                x_axis,
                y_ml,
                linestyle="-",
                c=color,
                linewidth=line_lw,
                zorder=z_baseline_line if jobid == BASELINE_JOBID else z_other_line,
            )
            delta_plot_records.append(
                {
                    "jobid": jobid,
                    "x_axis": x_axis.copy(),
                    "y_delta": y_delta.copy(),
                    "perc_delta": perc_delta.copy(),
                    "color": color,
                }
            )
        else:
            skipped_in_main_plot.append(jobid)

        mask_rel = np.abs(y_true) > y_eps
        mask_std = sigma_true > y_eps

        eps_rel = ((y_ml - y_true) / np.abs(y_true))[mask_rel]
        eps_std = ((y_ml - y_true) / sigma_true)[mask_std]
        eps_sigma = (sigma_ml / sigma_true - 1)[mask_std]

        jobids_in_order.append(jobid)
        mae_rel_by_job.append(np.abs(eps_rel).mean() if eps_rel.size > 0 else np.nan)
        mae_std_by_job.append(np.abs(eps_std).mean() if eps_std.size > 0 else np.nan)
        mae_sigma_by_job.append(np.abs(eps_sigma).mean() if eps_sigma.size > 0 else np.nan)

        if show_in_main_plot:
            model_label = _format_model_label(
                jobid,
                model_meta.get(jobid, {}),
                baseline_hyperparams,
                show_jobid=show_jobid,
            )
            handles_for_legend.append(Line2D([0], [0], color=color, lw=legend_line_lw, label=model_label))

    if z_idx is not None:
        first_jobid = next(iter(los_by_job.keys()))
        los = los_by_job[first_jobid]
        if z_idx < len(los[1]):
            x_mark = los[1][z_idx]
            ax_left.axvline(x=x_mark, color="red", linestyle="--", linewidth=marker_line_lw)
            ax_delta.axvline(x=x_mark, color="red", linestyle="--", linewidth=marker_line_lw)

    for rec in delta_plot_records:
        jobid = rec["jobid"]
        x_axis = rec["x_axis"]
        y_delta = rec["y_delta"]
        perc_delta = rec["perc_delta"]
        color = rec["color"]

        ax_delta.fill_between(
            x_axis,
            perc_delta[0],
            perc_delta[1],
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=z_baseline_band if jobid == BASELINE_JOBID else z_other_band,
        )
        ax_delta.plot(
            x_axis,
            y_delta,
            linestyle="-",
            c=color,
            linewidth=line_lw,
            zorder=z_baseline_line if jobid == BASELINE_JOBID else z_other_line,
        )

    if delta_ref_x_axis is not None:
        ax_delta.fill_between(
            delta_ref_x_axis,
            delta_ref_low,
            delta_ref_high,
            alpha=alpha_true,
            facecolor="black",
            edgecolor="black",
            zorder=z_true_band,
        )
        ax_delta.plot(
            delta_ref_x_axis,
            np.zeros_like(delta_ref_x_axis),
            linestyle=":",
            c="black",
            lw=ref_line_lw,
            zorder=z_true_line,
        )
    ax_delta.axhline(y=0.0, color="gray", linestyle="--", linewidth=zero_line_lw, alpha=0.8, zorder=0)

    ax_left.set_ylabel(r"$\langle T_b \rangle$ [mK]", fontsize=FS_LABEL)
    ax_left.grid()
    ax_left.tick_params(axis="both", labelsize=FS_TICK)
    ax_delta.set_ylabel(r"$\Delta\langle T_b \rangle$ [mK]", fontsize=FS_LABEL)
    ax_delta.set_xlabel("distance [Gpc]", fontsize=FS_LABEL)
    ax_delta.set_yscale("symlog", linthresh=10.0)
    ax_delta.grid()
    ax_delta.tick_params(axis="both", labelsize=FS_TICK)
    ax_delta.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1000:.1f}"))

    ax_twin = ax_left.secondary_xaxis("top")
    ax_twin.set_xlim(ax_left.get_xlim())
    ax_twin.set_xlabel("redshift", fontsize=FS_LABEL)
    first_jobid = next(iter(los_by_job.keys()))
    los = los_by_job[first_jobid]
    z_ticks_x = ax_twin.get_xticks()
    z_ticks = np.interp(z_ticks_x, los[1], los[0])
    ax_twin.set_xticks(z_ticks_x)
    ax_twin.set_xticklabels([f"{zv:.1f}" for zv in z_ticks])
    ax_twin.tick_params(axis="x", labelsize=FS_TICK)

    legend = ax_left.legend(handles=handles_for_legend, fontsize=FS_LEGEND, loc="best", framealpha=0.85)
    for txt in legend.get_texts():
        if txt.get_text() == baseline_label:
            txt.set_fontweight("bold")
            txt.set_fontstyle("italic")
    if skipped_in_main_plot:
        print(f"Skipped jobIDs in global_signal_hparams.pdf: {sorted(set(skipped_in_main_plot))}")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    if savename:
        fig.savefig(savename, bbox_inches="tight")
        print(f"Saved figure to {savename}")
        plt.close()
    else:
        plt.show()

    fig_mae, ax_mae = plt.subplots(
        1,
        1,
        figsize=(max(6.8, 0.45 * len(jobids_in_order) + 1.6), 4.8),
        dpi=220,
    )
    fs_mae_title = 18
    fs_mae_label = 16
    fs_mae_tick = 12
    fs_mae_legend = 11
    fs_mae_group = 10
    fs_mae_text = 8
    x = np.arange(len(jobids_in_order))
    ax_mae.scatter(x, mae_rel_by_job, marker="o", s=30, label=r"$\mathrm{MAE}_{rel}$")
    ax_mae.scatter(x, mae_std_by_job, marker="s", s=30, label=r"$\mathrm{MAE}_{std}$")
    ax_mae.scatter(x, mae_sigma_by_job, marker="^", s=34, label=r"$\mathrm{MAE}_{\sigma}$")
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels([str(i + 1) for i in range(len(x))])
    ax_mae.set_xlabel("job index", fontsize=fs_mae_label)
    ax_mae.set_ylabel("MAE", fontsize=fs_mae_label)
    ax_mae.set_yscale("log")
    ax_mae.grid()
    ax_mae.tick_params(axis="both", labelsize=fs_mae_tick)

    mae_rel_arr = np.asarray(mae_rel_by_job, dtype=float)
    mae_std_arr = np.asarray(mae_std_by_job, dtype=float)
    text_transform = ax_mae.get_xaxis_transform()
    y_min_raw = np.nanmin(np.concatenate([mae_rel_arr, mae_std_arr, np.asarray(mae_sigma_by_job, dtype=float)]))
    y_max_raw = np.nanmax(np.concatenate([mae_rel_arr, mae_std_arr, np.asarray(mae_sigma_by_job, dtype=float)]))
    y_min = max(y_min_raw * 0.9, 1e-12)
    y_max = y_max_raw * 1.35
    ax_mae.set_ylim(y_min, y_max)
    y_span = np.log10(y_max) - np.log10(y_min)
    row_base = y_max / (10 ** (0.10 * y_span))
    row_step = 10 ** (0.08 * y_span)
    lane_step = 10 ** (0.055 * y_span)
    row_lane_intervals: Dict[int, List[List[Tuple[float, float]]]] = {}
    best_in_group_labeled = False
    best_markers_by_pos: Dict[int, List[Tuple[str, float]]] = {}
    for group in MAE_GROUPS:
        valid_pos = []
        for idx1 in group["indices_1based"]:
            idx0 = idx1 - 1
            if 0 <= idx0 < len(x):
                valid_pos.append(idx0)
        if not valid_pos:
            continue

        valid_pos_arr = np.array(sorted(valid_pos), dtype=int)
        x0 = float(valid_pos_arr.min())
        x1 = float(valid_pos_arr.max())
        color = group["color"]
        runs = []
        start = valid_pos_arr[0]
        prev = valid_pos_arr[0]
        for pos in valid_pos_arr[1:]:
            if pos == prev + 1:
                prev = pos
            else:
                runs.append((start, prev))
                start = pos
                prev = pos
        runs.append((start, prev))
        for run_start, run_end in runs:
            ax_mae.axvspan(float(run_start) - 0.45, float(run_end) + 0.45, color=color, alpha=0.06, zorder=0)

        row = int(group.get("row", 0))
        span_left = float(x0) - 0.45
        span_right = float(x1) + 0.45
        lanes = row_lane_intervals.setdefault(row, [])
        lane_idx = 0
        while True:
            if lane_idx >= len(lanes):
                lanes.append([])
            overlaps = any(not (span_right < l or span_left > r) for l, r in lanes[lane_idx])
            if not overlaps:
                lanes[lane_idx].append((span_left, span_right))
                break
            lane_idx += 1

        y_row = row_base / (row_step ** row) / (lane_step ** lane_idx)
        ax_mae.plot([x0, x1], [y_row, y_row], color=color, lw=1.2, ls="--", alpha=0.8, clip_on=True)
        for run_start, run_end in runs:
            ax_mae.plot(
                [float(run_start), float(run_end)],
                [y_row, y_row],
                color=color,
                lw=2.2,
                clip_on=True,
            )
        for xi in valid_pos_arr:
            ax_mae.plot(
                [float(xi), float(xi)],
                [y_row / 1.015, y_row * 1.015],
                color=color,
                lw=1.3,
                clip_on=True,
            )
        ax_mae.text(
            0.5 * (x0 + x1),
            y_row * 1.08,
            group["name"],
            color=color,
            fontsize=fs_mae_group,
            ha="center",
            va="bottom",
            clip_on=True,
            bbox=dict(facecolor="none", edgecolor="none", pad=1.5),
        )

        group_vals = mae_std_arr[valid_pos_arr]
        finite_mask = np.isfinite(group_vals)
        if np.any(finite_mask):
            best_pos = int(valid_pos_arr[finite_mask][np.argmin(group_vals[finite_mask])])
            best_markers_by_pos.setdefault(best_pos, []).append((color, float(mae_std_arr[best_pos])))

    for best_pos, entries in best_markers_by_pos.items():
        n = len(entries)
        for j, (color, y_val) in enumerate(entries):
            s_ring = 130 + 55 * (n - 1 - j)
            ax_mae.scatter(
                [x[best_pos]],
                [y_val],
                s=s_ring,
                facecolors="none",
                edgecolors=color,
                linewidths=1.8,
                zorder=5 + 0.01 * j,
                label="best in group" if not best_in_group_labeled else None,
            )
            best_in_group_labeled = True

    for b in MAE_SECTION_BOUNDARIES_1BASED:
        b0 = b - 1
        if 0 <= b0 < len(x) - 1:
            ax_mae.axvline(float(b0) + 0.5, color="0.45", lw=0.9, ls=":", alpha=0.8, zorder=1)

    ax_mae.legend(
        fontsize=fs_mae_legend,
        loc="upper left",
        ncol=4,
        framealpha=0.85,
        handletextpad=0.5,
        columnspacing=0.9,
        borderpad=0.28,
        borderaxespad=0.2,
    )

    x_offset = -0.10
    for i, jobid in enumerate(jobids_in_order):
        label_text = _format_job_diff_text(
            jobid,
            model_meta.get(jobid, {}),
            baseline_hyperparams,
            show_jobid=show_jobid,
        )
        label_text = _wrap_label_text(label_text, width=34)
        fw = "bold" if jobid == BASELINE_JOBID else "normal"
        fs = "italic" if jobid == BASELINE_JOBID else "normal"
        ax_mae.text(
            x[i] + x_offset,
            0.012,
            label_text,
            transform=text_transform,
            fontsize=fs_mae_text,
            ha="center",
            va="bottom",
            rotation=90,
            fontweight=fw,
            fontstyle=fs,
            alpha=0.88,
            zorder=30,
        )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])

    if savename:
        root, ext = os.path.splitext(savename)
        mae_savename = f"{root}_mae_trend{ext if ext else '.pdf'}"
        plt.savefig(mae_savename, bbox_inches="tight")
        print(f"Saved figure to {mae_savename}")
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot global signal: truth vs multiple model hyperparameter settings.")
    parser.add_argument("--real_h5", type=str, required=False, help="Real-data H5 filename. Relative path is resolved under $SCRATCH.", default="LEN128-DIM64-CUB16-Tvir4.699-zeta30-0812-104322.h5")
    parser.add_argument("--outputs_dir", type=str, default="../training/outputs", help="Directory containing generated .npy files.")
    parser.add_argument("--num_image", type=int, default=320)
    parser.add_argument("--num_redshift", type=int, default=1024)
    parser.add_argument("--HII_DIM", type=int, default=64)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--use_ema", type=int, default=0)
    parser.add_argument("--pt_fname", type=str, default="../utils/PowerTransformer_25600_z1.pkl")
    parser.add_argument("--z_idx", type=int, default=None)
    parser.add_argument("--show-jobid", action="store_true", help="Show jobid in plot labels/annotations.")
    parser.add_argument("--save", type=str, default="hparams.pdf")
    parser.add_argument("--save_pdf", type=str, default="hparams_pdf.pdf")
    args = parser.parse_args()
    target_pattern = derive_target_pattern_from_real_h5(args.real_h5)
    print(f"Using target_pattern={target_pattern} derived from real_h5={args.real_h5}")
    main_plot_jobids = _jobids_from_indices_1based(
        MAIN_PLOT_JOB_INDICES_1BASED,
        JOBID_HPARAMS,
        "MAIN_PLOT_JOB_INDICES_1BASED",
    )
    print(f"[MainPlot] Using job indices {MAIN_PLOT_JOB_INDICES_1BASED} -> jobIDs {main_plot_jobids}")

    # Load truth once at z_step=1, then adapt per job by slicing [..., ::z_step].
    x_true_full, _, los_full = load_h5_as_tensor(
        dir_name=args.real_h5,
        num_image=args.num_image,
        num_redshift=args.num_redshift,
        HII_DIM=args.HII_DIM,
        z_step=1,
        dim=args.dim,
    )

    x_ml_by_job: Dict[int, torch.Tensor] = {}
    x_true_by_job: Dict[int, torch.Tensor] = {}
    los_by_job: Dict[int, np.ndarray] = {}
    for jobid in JOBID_HPARAMS.keys():
        job_dim = _infer_job_dim(JOBID_HPARAMS.get(jobid, {}), fallback_dim=args.dim)
        job_z_step = int(JOBID_HPARAMS.get(jobid, {}).get("z_step", 1))
        if job_z_step < 1:
            raise ValueError(f"Invalid z_step={job_z_step} for jobid={jobid}. z_step must be >= 1.")
        job_transform = JOBID_HPARAMS.get(jobid, {}).get("transform", "pt_inv")

        x_true = x_true_full[..., ::job_z_step]
        los = los_full[:, ::job_z_step]

        x_ml = load_x_ml(
            target_pattern=target_pattern,
            jobid=jobid,
            num_image=args.num_image,
            ema=args.use_ema,
            outputs_dir=args.outputs_dir,
            pt_fname=args.pt_fname,
            transform=job_transform,
            apply_inverse=True,
        )
        print(
            "[GlobalSignal] Loaded "
            f"x_ml.shape={x_ml.shape}; "
            f"x_true.shape={x_true.shape}; "
            f"los.shape={los.shape}; "
            f"jobid={jobid}"
        )
        # Align LOS length if generated tensors use a different redshift-length.
        z_len = min(x_true.shape[-1], x_ml.shape[-1], los.shape[1])
        x_true = x_true[..., :z_len]
        x_ml = x_ml[..., :z_len]
        los = los[:, :z_len]

        n = min(len(x_true), len(x_ml))
        x_true_by_job[jobid] = x_true[:n]
        x_ml_by_job[jobid] = x_ml[:n]
        los_by_job[jobid] = los

    plot_global_signal_hyperparameters(
        x_true_by_job=x_true_by_job,
        x_ml_by_job=x_ml_by_job,
        los_by_job=los_by_job,
        model_meta=JOBID_HPARAMS,
        z_idx=args.z_idx,
        savename=args.save,
        show_jobid=args.show_jobid,
        main_plot_jobids=main_plot_jobids,
    )

    # PDF plotting: explicitly selected 3D jobs only.
    x_ml_raw_pdf_by_job: Dict[int, torch.Tensor] = {}
    x_true_pdf_by_job: Dict[int, torch.Tensor] = {}
    pdf_jobids = _jobids_from_indices_1based(
        PDF_JOB_INDICES_1BASED,
        JOBID_HPARAMS,
        "PDF_JOB_INDICES_1BASED",
    )
    pdf_jobids_dim3 = [
        jobid
        for jobid in pdf_jobids
        if _infer_job_dim(JOBID_HPARAMS.get(jobid, {}), fallback_dim=args.dim) == 3
    ]
    dropped_pdf_non_dim3 = [jobid for jobid in pdf_jobids if jobid not in pdf_jobids_dim3]
    if dropped_pdf_non_dim3:
        print(f"Skipped non-dim=3 PDF jobIDs: {dropped_pdf_non_dim3}")

    if pdf_jobids_dim3:
        print(f"[PDF] Using job indices {PDF_JOB_INDICES_1BASED} -> dim=3 jobIDs: {pdf_jobids_dim3}")
        x_true_full_dim3, _, _ = load_h5_as_tensor(
            dir_name=args.real_h5,
            num_image=args.num_image,
            num_redshift=args.num_redshift,
            HII_DIM=args.HII_DIM,
            z_step=1,
            dim=3,
        )
        for jobid in pdf_jobids_dim3:
            job_meta = JOBID_HPARAMS.get(jobid, {})
            job_z_step = int(job_meta.get("z_step", 1))
            job_transform = job_meta.get("transform", "pt_inv")
            x_true_dim3 = x_true_full_dim3[..., ::job_z_step]
            x_ml_raw = load_x_ml(
                target_pattern=target_pattern,
                jobid=jobid,
                num_image=args.num_image,
                ema=args.use_ema,
                outputs_dir=args.outputs_dir,
                pt_fname=args.pt_fname,
                transform=job_transform,
                apply_inverse=False,
            )
            z_len_pdf = min(x_true_dim3.shape[-1], x_ml_raw.shape[-1])
            n_pdf = min(len(x_true_dim3), len(x_ml_raw))
            x_true_pdf_by_job[jobid] = x_true_dim3[:n_pdf, ..., :z_len_pdf]
            x_ml_raw_pdf_by_job[jobid] = x_ml_raw[:n_pdf, ..., :z_len_pdf]
            print(
                "[PDF] Loaded "
                f"x_ml_raw.shape={x_ml_raw_pdf_by_job[jobid].shape}; "
                f"x_true_dim3.shape={x_true_pdf_by_job[jobid].shape}; "
                f"jobid={jobid}; transform={job_transform}"
            )

    plot_pixel_pdf_by_job_transform(
        x_true_by_job=x_true_pdf_by_job,
        x_ml_raw_by_job=x_ml_raw_pdf_by_job,
        model_meta=JOBID_HPARAMS,
        pt_fname=args.pt_fname,
        savename=args.save_pdf,
        show_jobid=args.show_jobid,
    )


if __name__ == "__main__":
    main()
