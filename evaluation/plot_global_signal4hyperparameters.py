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
    46941303: {
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
    47032706: {
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
    48436662: {
        "num_res_blocks": 1, # baseline
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
    48057143: {
        "num_res_blocks": 3,
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    48057168: {
        "num_res_blocks": 3,
        "squish": "0.5,0",
        "dim": 3,
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    48057253: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "dim": 3,
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    47908550: {
        "num_res_blocks": 3,
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 60,
        "z_step": "1",
        "transform": "pt_inv",
    },
    47356556: {
        "num_res_blocks": 3,
        "squish": "0.1,0",
        "dim": 3,
        "epochs": 30,
        "z_step": "1",
        "transform": "pt_inv",
    },
}
BASELINE_JOBID = 48436662


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

    if transform == "pt_inv" and pt_fname is not None:
        pt = joblib.load(pt_fname)
        x_ml = pt.inverse_transform(x_ml.reshape(-1, 1))
    elif transform == "min_max":
        min_val, max_val = ranges_dict[transform]
        x_ml = x_ml * (max_val - min_val) + min_val
    elif transform == "z_score":
        mean_val, std_val = ranges_dict[transform]
        x_ml = x_ml * std_val + mean_val
    elif transform == "arcsinh":
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


def plot_global_signal_hyperparameters(
    x_true_by_job: Dict[int, torch.Tensor],
    x_ml_by_job: Dict[int, torch.Tensor],
    los_by_job: Dict[int, np.ndarray],
    model_meta: Dict[int, Dict[str, Any]],
    sigma_level: float = 68.27,
    y_eps: float = 0.0,
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

    color_cycle = _generate_distinct_colors(len(x_ml_by_job))

    for idx, (jobid, x_ml) in enumerate(x_ml_by_job.items()):
        color = color_cycle[idx % len(color_cycle)]
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
            )
            ax_left.plot(x_axis, y_true, linestyle=":", c="black", lw=1.7, label="21cmFAST median")

        tb_ml = x2Tb(x_ml)
        y_ml = np.median(tb_ml, axis=0)
        perc_ml = np.percentile(tb_ml, [low, high], axis=0)
        sigma_ml = 0.5 * (perc_ml[1] - perc_ml[0])

        # Use translucent uncertainty bands instead of dense error bars
        # to reduce severe overlap across many jobs.
        ax_left.fill_between(
            x_axis,
            perc_ml[0],
            perc_ml[1],
            color=color,
            alpha=0.10,
            linewidth=0,
        )
        ax_left.plot(
            x_axis,
            y_ml,
            linestyle="-",
            c=color,
            linewidth=lw,
        )

        mask_rel = np.abs(y_true) > y_eps
        mask_std = sigma_true > y_eps

        eps_rel = ((y_ml - y_true) / np.abs(y_true))[mask_rel]
        eps_std = ((y_ml - y_true) / sigma_true)[mask_std]
        eps_sigma = (sigma_ml / sigma_true - 1)[mask_std]

        jobids_in_order.append(jobid)
        mae_rel_by_job.append(np.abs(eps_rel).mean() if eps_rel.size > 0 else np.nan)
        mae_std_by_job.append(np.abs(eps_std).mean() if eps_std.size > 0 else np.nan)
        mae_sigma_by_job.append(np.abs(eps_sigma).mean() if eps_sigma.size > 0 else np.nan)

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

    plt.tight_layout()

    if savename:
        plt.savefig(savename, bbox_inches="tight")
        print(f"Saved figure to {savename}")
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
        )
        print(
            "⛳️ Loaded "
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


if __name__ == "__main__":
    main()
