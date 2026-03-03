import os
import sys
import argparse
import re
import textwrap
from typing import Dict, Any

import h5py
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.load_h5 import Dataset4h5, ranges_dict


# Manually specify the experiment registry here.
# key: jobID, value: primary hyperparameters for legend labels.
JOBID_HPARAMS: Dict[int, Dict[str, Any]] = {
    46941305: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 2,
        "epochs": 120,
        "z_step": "1",
        "transform": "arcsinh",
    },
    49299747: {
    # 46941303: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 2,
        "epochs": 120,
        "z_step": "1",
        "transform": "min_max",
    },
    46941293: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 2,
        "epochs": 120,
        "z_step": "1",
        "transform": "z_score",
    },
    46941286: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 2,
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    49299542: {
    # 47032706: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 3,
        "epochs": 120,
        "z_step": "2",
        "transform": "min_max",
    },
    47032672: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 3,
        "epochs": 120,
        "z_step": "2",
        "transform": "z_score",
    },
    47032656: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 3,
        "epochs": 120,
        "z_step": "2",
        "transform": "pt_inv",
    },
    48820329: {
        "num_res_blocks": 1,
    # 48057253: {
    #     "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 3,
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    48902106: {
        "num_res_blocks": 1,
    # 48057168: {
    #     "num_res_blocks": 3,
        "squish": "0.5,0",
        "dim": 3,
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    49325389: {
        "num_res_blocks": 1,
        "squish": "0.01,0",
        "dim": 3,
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    48057143: {
        "num_res_blocks": 3,
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    48580330: {
        "num_res_blocks": 2,
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    48820652: {
    # 48436662: {
        "num_res_blocks": 1,
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 240,
        "z_step": "1",
        "transform": "pt_inv",
    },
    48436662: {
        "num_res_blocks": 1, # baseline
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    # 48902183: {
    #     "num_res_blocks": 1,
    #     "squish": "0.1,1",
    #     # "squish": "0.1,0",
    #     "dim": 3,
    #     "epochs": 120,
    #     "z_step": "1",
    #     "transform": "pt_inv",
    # },
    48820480: {
        "num_res_blocks": 1,
    # 47908550: {
    #     "num_res_blocks": 3,
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 60,
        "z_step": "1",
        "transform": "pt_inv",
    },
    48820424: {
    # 47356556: {
        # "num_res_blocks": 3,
        "num_res_blocks": 1,
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 30,
        "z_step": "1",
        "transform": "pt_inv",
    },
}
BASELINE_JOBID = 48436662
# PDF_JOBIDS = [49299542, 47032672, 47032656,]
PDF_JOBIDS = [49299747, 46941293, 46941286, 46941305]
MAIN_PLOT_EXCLUDED_JOBIDS = {
    46941305,
    49299747,
    46941293,
    46941286,
    48820480,
    48820424,
    48820652,
}


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


def _format_model_label(jobid: int, hyperparams: Dict[str, Any], baseline_hyperparams: Dict[str, Any]) -> str:
    if not hyperparams:
        return f"job={jobid}"

    if jobid == BASELINE_JOBID:
        hp_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])
        return f"job={jobid} | {hp_str}"

    diff_items = []
    for k, v in hyperparams.items():
        if k not in baseline_hyperparams or str(v) != str(baseline_hyperparams[k]):
            diff_items.append((k, v))

    if not diff_items:
        return f"job={jobid} | same as baseline"

    hp_str = ", ".join([f"{k}={v}" for k, v in diff_items])
    return f"job={jobid} | {hp_str}"


def _format_job_diff_text(jobid: int, hyperparams: Dict[str, Any], baseline_hyperparams: Dict[str, Any]) -> str:
    if jobid == BASELINE_JOBID:
        if not hyperparams:
            return f"{jobid}"
        return f"{jobid}: " + ", ".join([f"{k}={v}" for k, v in hyperparams.items()])
    if not hyperparams:
        return f"{jobid}"

    diff_items = []
    for k, v in hyperparams.items():
        if k not in baseline_hyperparams or str(v) != str(baseline_hyperparams[k]):
            diff_items.append(f"{k}={v}")

    if not diff_items:
        return f"{jobid} (same)"
    return f"{jobid}: " + ", ".join(diff_items)


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
    savename: str = "global_signal_hparams_pdf.png",
):
    if not x_ml_raw_by_job:
        print("No dim=2 jobs found for PDF plotting; skipped.")
        return

    jobids = list(x_ml_raw_by_job.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), dpi=220)
    ax_l, ax_r = axes
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

        job_handles.append(Line2D([0], [0], color=color, lw=1.8, linestyle="-", label=f"job={jobid}, {transform}"))

        # Right subplot: inverse-transformed sampled vs raw testing-set space.
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
        ax_l.hist(t_by_job[jobid], bins=bins_l, density=True, histtype="step", linewidth=1.5, linestyle="--", color=color)
        ax_l.hist(m_by_job[jobid], bins=bins_l, density=True, histtype="step", linewidth=1.2, linestyle="-", color=color)

    raw_lows = [np.percentile(x_true_ref, 0.1)]
    raw_highs = [np.percentile(x_true_ref, 99.9)]
    for jobid in jobids:
        raw_lows.append(np.percentile(m_inv_by_job[jobid], 0.1))
        raw_highs.append(np.percentile(m_inv_by_job[jobid], 99.9))
    bins_r = _make_symlog_bins(min(raw_lows), max(raw_highs), n_bins=180, linthresh=linthresh_raw)

    for i, jobid in enumerate(jobids):
        color = f"C{i}"
        ax_r.hist(m_inv_by_job[jobid], bins=bins_r, density=True, histtype="step", linewidth=1.2, linestyle="-", color=color)

    # Raw testing-set reference (same space as inverse-transformed sampled).
    ax_r.hist(
        x_true_ref,
        bins=bins_r,
        density=True,
        histtype="step",
        linewidth=1.8,
        linestyle="--",
        color="black",
        label="testing set (raw)",
    )

    ax_l.set_yscale("log")
    ax_l.set_ylim(bottom=1e-3)
    ax_l.set_xscale("symlog", linthresh=linthresh_tf)
    ax_l.grid(alpha=0.35)
    ax_l.set_title("Transform Space: test(transformed) vs sampled")
    ax_l.set_xlabel("pixel value")
    ax_l.set_ylabel("PDF")

    ax_r.set_yscale("log")
    ax_r.set_ylim(bottom=1e-3)
    ax_r.set_xscale("symlog", linthresh=linthresh_raw)
    ax_r.grid(alpha=0.35)
    ax_r.set_title("Raw Space: sampled (inverse) vs testing set")
    ax_r.set_xlabel("pixel value")
    ax_r.set_ylabel("PDF")

    legend_jobs = ax_l.legend(handles=job_handles, fontsize=7, loc="upper right", title="Job / Transform")
    ax_l.add_artist(legend_jobs)
    style_handles = [
        Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="testing set (transformed)"),
        Line2D([0], [0], color="black", lw=1.2, linestyle="-", label="sampled data"),
    ]
    ax_l.legend(handles=style_handles, fontsize=8, loc="upper left", title="Line Style")
    style_handles_right = [
        Line2D([0], [0], color="black", lw=1.8, linestyle="--", label="testing set (raw)"),
        Line2D([0], [0], color="black", lw=1.2, linestyle="-", label="sampled data (inverse)"),
    ]
    ax_r.legend(handles=style_handles_right, fontsize=8, loc="upper right", title="Line Style")

    plt.tight_layout()
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
):
    low = (100 - sigma_level) / 2
    high = 100 - low

    fig, ax_left = plt.subplots(1, 1, figsize=(11, 6), dpi=220)

    jobids_in_order = []
    mae_rel_by_job = []
    mae_std_by_job = []
    mae_sigma_by_job = []

    handles_for_legend = [Line2D([0], [0], color="black", linestyle=":", lw=1.7, label="21cmFAST median")]
    baseline_hyperparams = model_meta.get(BASELINE_JOBID, {})

    z_other_band = 1
    z_other_line = 2
    z_baseline_band = 5
    z_baseline_line = 6
    z_true_band = 7
    z_true_line = 8
    skipped_in_main_plot = []
    delta_plot_records = []
    delta_ref_x_axis = None
    delta_ref_low = None
    delta_ref_high = None

    for idx, (jobid, x_ml) in enumerate(x_ml_by_job.items()):
        color = f"C{idx}"
        x_true = x_true_by_job[jobid]
        los = los_by_job[jobid]

        tb_true = x2Tb(x_true)
        y_true = np.median(tb_true, axis=0)
        perc_true = np.percentile(tb_true, [low, high], axis=0)
        sigma_true = 0.5 * (perc_true[1] - perc_true[0])
        interval = max(1, y_true.shape[0] // 100)
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
                lw=1.7,
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

        if jobid not in MAIN_PLOT_EXCLUDED_JOBIDS:
            # Use translucent uncertainty bands instead of dense error bars
            # to reduce severe overlap across many jobs.
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
                linewidth=lw,
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

        if jobid not in MAIN_PLOT_EXCLUDED_JOBIDS:
            model_label = _format_model_label(jobid, model_meta.get(jobid, {}), baseline_hyperparams)
            handles_for_legend.append(Line2D([0], [0], color=color, lw=1.5, label=model_label))

    if z_idx is not None:
        # z_idx is interpreted relative to each job's LOS grid; use first job for marker.
        first_jobid = next(iter(los_by_job.keys()))
        los = los_by_job[first_jobid]
        if z_idx < len(los[1]):
            x_mark = los[1][z_idx]
            ax_left.axvline(x=x_mark, color="red", linestyle="--", linewidth=1.5)

    ax_left.set_ylabel(r"$\langle T_b \rangle$ [mK]")
    ax_left.set_xlabel("distance [Gpc]")
    ax_left.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x / 1000:.1f}"))
    ax_left.set_title("Global Signal Comparison")
    ax_left.grid()

    ax_twin = ax_left.secondary_xaxis("top")
    ax_twin.set_xlim(ax_left.get_xlim())
    ax_twin.set_xlabel("redshift")
    first_jobid = next(iter(los_by_job.keys()))
    los = los_by_job[first_jobid]
    z_ticks_x = ax_twin.get_xticks()
    z_ticks = np.interp(z_ticks_x, los[1], los[0])
    ax_twin.set_xticks(z_ticks_x)
    ax_twin.set_xticklabels([f"{zv:.1f}" for zv in z_ticks])

    legend = ax_left.legend(handles=handles_for_legend, fontsize=8, loc="best")
    for txt in legend.get_texts():
        if txt.get_text().startswith(f"job={BASELINE_JOBID}"):
            txt.set_fontweight("bold")
            txt.set_fontstyle("italic")
    if skipped_in_main_plot:
        print(f"Skipped jobIDs in global_signal_hparams.png: {sorted(set(skipped_in_main_plot))}")

    plt.tight_layout()

    if savename:
        plt.savefig(savename, bbox_inches="tight")
        print(f"Saved figure to {savename}")
        plt.close()
    else:
        plt.show()

    # Additional figure: residual global signal vs 21cmFAST.
    fig_delta, ax_delta = plt.subplots(1, 1, figsize=(11, 6), dpi=220)
    delta_handles_for_legend = [
        Line2D([0], [0], color="black", linestyle=":", lw=1.7, label=r"21cmFAST reference ($\Delta T_b=0$)")
    ]
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
            linewidth=lw,
            zorder=z_baseline_line if jobid == BASELINE_JOBID else z_other_line,
        )

        model_label = _format_model_label(jobid, model_meta.get(jobid, {}), baseline_hyperparams)
        delta_handles_for_legend.append(Line2D([0], [0], color=color, lw=1.5, label=model_label))

    if delta_ref_x_axis is not None:
        ax_delta.fill_between(
            delta_ref_x_axis,
            delta_ref_low,
            delta_ref_high,
            alpha=alpha_true,
            facecolor="black",
            edgecolor="black",
            label="21cmFAST CI (relative to median)",
            zorder=z_true_band,
        )
        ax_delta.plot(
            delta_ref_x_axis,
            np.zeros_like(delta_ref_x_axis),
            linestyle=":",
            c="black",
            lw=1.7,
            zorder=z_true_line,
        )
    ax_delta.axhline(y=0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)

    ax_delta.set_ylabel(r"$\Delta\langle T_b \rangle$ [mK] (model - 21cmFAST)")
    ax_delta.set_yscale("symlog", linthresh=10.0)
    ax_delta.set_xlabel("distance [Gpc]")
    ax_delta.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x / 1000:.1f}"))
    ax_delta.set_title("Global Signal Residuals vs 21cmFAST")
    ax_delta.grid()

    ax_delta_twin = ax_delta.secondary_xaxis("top")
    ax_delta_twin.set_xlim(ax_delta.get_xlim())
    ax_delta_twin.set_xlabel("redshift")
    first_jobid = next(iter(los_by_job.keys()))
    los = los_by_job[first_jobid]
    z_ticks_x_delta = ax_delta_twin.get_xticks()
    z_ticks_delta = np.interp(z_ticks_x_delta, los[1], los[0])
    ax_delta_twin.set_xticks(z_ticks_x_delta)
    ax_delta_twin.set_xticklabels([f"{zv:.1f}" for zv in z_ticks_delta])

    legend_delta = ax_delta.legend(handles=delta_handles_for_legend, fontsize=8, loc="best")
    for txt in legend_delta.get_texts():
        if txt.get_text().startswith(f"job={BASELINE_JOBID}"):
            txt.set_fontweight("bold")
            txt.set_fontstyle("italic")

    plt.tight_layout()
    if savename:
        root, ext = os.path.splitext(savename)
        delta_savename = f"{root}_deltaTb{ext if ext else '.png'}"
        plt.savefig(delta_savename, bbox_inches="tight")
        print(f"Saved figure to {delta_savename}")
        plt.close()
    else:
        plt.show()

    # Separate figure: per-job MAE trend.
    fig_mae, ax_mae = plt.subplots(
        1,
        1,
        figsize=(max(6.8, 0.45 * len(jobids_in_order) + 1.6), 4.8),
        dpi=220,
    )
    x = np.arange(len(jobids_in_order))
    ax_mae.plot(x, mae_rel_by_job, marker="o", lw=1.5, label=r"$\mathrm{MAE}_{rel}$")
    ax_mae.plot(x, mae_std_by_job, marker="s", lw=1.5, label=r"$\mathrm{MAE}_{std}$")
    ax_mae.plot(x, mae_sigma_by_job, marker="^", lw=1.5, label=r"$\mathrm{MAE}_{\sigma}$")
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels([str(i + 1) for i in range(len(x))])
    ax_mae.set_xlabel("job index")
    ax_mae.set_ylabel("MAE")
    ax_mae.set_title("Per-Job Error Trend")
    ax_mae.set_yscale("log")
    ax_mae.grid()
    ax_mae.legend()

    # Annotate each job near MAE_rel points using jobID + diffs to baseline.
    text_transform = ax_mae.get_xaxis_transform()
    for i, jobid in enumerate(jobids_in_order):
        label_text = _format_job_diff_text(jobid, model_meta.get(jobid, {}), baseline_hyperparams)
        label_text = _wrap_label_text(label_text, width=50)
        fw = "bold" if jobid == BASELINE_JOBID else "normal"
        fs = "italic" if jobid == BASELINE_JOBID else "normal"
        ax_mae.text(
            x[i],
            0.02,
            label_text,
            transform=text_transform,
            fontsize=7,
            ha="left",
            va="bottom",
            rotation=90,
            fontweight=fw,
            fontstyle=fs,
        )
    plt.tight_layout()

    if savename:
        root, ext = os.path.splitext(savename)
        mae_savename = f"{root}_mae_trend{ext if ext else '.png'}"
        plt.savefig(mae_savename, bbox_inches="tight")
        print(f"Saved figure to {mae_savename}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot global signal: truth vs multiple model hyperparameter settings.")
    parser.add_argument("--real_h5", type=str, required=True, help="Real-data H5 filename. Relative path is resolved under $SCRATCH.")
    parser.add_argument("--outputs_dir", type=str, default="../training/outputs", help="Directory containing generated .npy files.")
    parser.add_argument("--num_image", type=int, default=256)
    parser.add_argument("--num_redshift", type=int, default=1024)
    parser.add_argument("--HII_DIM", type=int, default=64)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--use_ema", type=int, default=0)
    parser.add_argument("--pt_fname", type=str, default="../utils/PowerTransformer_25600_z1.pkl")
    parser.add_argument("--z_idx", type=int, default=None)
    parser.add_argument("--save", type=str, default="global_signal_hyperparams.png")
    parser.add_argument("--save_pdf", type=str, default="global_signal_hparams_pdf.png")
    args = parser.parse_args()
    target_pattern = derive_target_pattern_from_real_h5(args.real_h5)
    print(f"Using target_pattern={target_pattern} derived from real_h5={args.real_h5}")

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
    )

    # PDF plotting: explicitly selected jobs only.
    x_ml_raw_dim2_by_job: Dict[int, torch.Tensor] = {}
    x_true_dim2_by_job: Dict[int, torch.Tensor] = {}
    pdf_jobids = [jobid for jobid in PDF_JOBIDS if jobid in JOBID_HPARAMS]
    missing_pdf_jobids = [jobid for jobid in PDF_JOBIDS if jobid not in JOBID_HPARAMS]
    if missing_pdf_jobids:
        print(f"Skipped PDF jobIDs not found in JOBID_HPARAMS: {missing_pdf_jobids}")

    if pdf_jobids:
        print(f"[PDF] Using explicit jobIDs: {pdf_jobids}")
        x_true_full_dim2, _, _ = load_h5_as_tensor(
            dir_name=args.real_h5,
            num_image=args.num_image,
            num_redshift=args.num_redshift,
            HII_DIM=args.HII_DIM,
            z_step=1,
            dim=2,
        )
        for jobid in pdf_jobids:
            job_meta = JOBID_HPARAMS.get(jobid, {})
            job_z_step = int(job_meta.get("z_step", 1))
            job_transform = job_meta.get("transform", "pt_inv")
            x_true_dim2 = x_true_full_dim2[..., ::job_z_step]
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
            z_len_pdf = min(x_true_dim2.shape[-1], x_ml_raw.shape[-1])
            n_pdf = min(len(x_true_dim2), len(x_ml_raw))
            x_true_dim2_by_job[jobid] = x_true_dim2[:n_pdf, ..., :z_len_pdf]
            x_ml_raw_dim2_by_job[jobid] = x_ml_raw[:n_pdf, ..., :z_len_pdf]
            print(
                "[PDF] Loaded "
                f"x_ml_raw.shape={x_ml_raw_dim2_by_job[jobid].shape}; "
                f"x_true_dim2.shape={x_true_dim2_by_job[jobid].shape}; "
                f"jobid={jobid}; transform={job_transform}"
            )

    plot_pixel_pdf_by_job_transform(
        x_true_by_job=x_true_dim2_by_job,
        x_ml_raw_by_job=x_ml_raw_dim2_by_job,
        model_meta=JOBID_HPARAMS,
        pt_fname=args.pt_fname,
        savename=args.save_pdf,
    )


if __name__ == "__main__":
    main()
