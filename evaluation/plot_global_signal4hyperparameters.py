import os
import sys
import argparse
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
    48436662: {
        "num_res_blocks": 1,
        "squish": "0.1,0",
        "stride": "2-2-4",
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    48580330: {
        "num_res_blocks": 2,
        "squish": "0.1,0",
        "stride": "2-2-4",
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    48057143: {
        "num_res_blocks": 3,
        "squish": "0.1,0",
        "stride": "2-2-4",
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    47908550: {
        "num_res_blocks": 3,
        "squish": "0.1,0",
        "stride": "2-2-4",
        "epochs": 60,
        "z_step": "1",
        "transform": "pt_inv",
    },
    47356556: {
        "num_res_blocks": 3,
        "squish": "0.1,0",
        "stride": "2-2-4",
        "epochs": 30,
        "z_step": "1",
        "transform": "pt_inv",
    },
    48057168: {
        "num_res_blocks": 3,
        "squish": "0.5,0",
        "stride": "2-2-4",
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    48057253: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "stride": "2-2-4",
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    },
    47032656: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "stride": "2-2-4",
        "epochs": 120,
        "z_step": "2",
        "transform": "pt_inv",
    },
    47032706: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "stride": "2-2-4",
        "epochs": 120,
        "z_step": "2",
        "transform": "min_max",
    },
    47032672: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "stride": "2-2-4",
        "epochs": 120,
        "z_step": "2",
        "transform": "z_score",
    },
    46941293: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "stride": "2-4",
        "epochs": 120,
        "z_step": "1",
        "transform": "z_score",
    },
    46941305: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "stride": "2-4",
        "epochs": 120,
        "z_step": "1",
        "transform": "arcsinh",
    },
    46941303: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "stride": "2-4",
        "epochs": 120,
        "z_step": "1",
        "transform": "min_max",
    },
    46941286: {
        "num_res_blocks": 3,
        "squish": "1,0",
        "stride": "2-4",
        "epochs": 120,
        "z_step": "1",
        "transform": "pt_inv",
    }
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


def _format_model_label(jobid: int, hyperparams: Dict[str, Any]) -> str:
    if not hyperparams:
        return f"job={jobid}"
    hp_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])
    return f"job={jobid} | {hp_str}"


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

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(11, 7), dpi=220, gridspec_kw={"height_ratios": [3, 0.5, 0.5, 0.5]})

    eps_rel_all = []
    eps_std_all = []
    eps_sigma_all = []

    handles_for_legend = [
        Line2D([0], [0], color="black", linestyle=":", lw=1.7, label="21cmFAST median")
    ]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(10)])

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
            ax[0].fill_between(
                x_axis,
                perc_true[0],
                perc_true[1],
                alpha=alpha_true,
                facecolor="black",
                edgecolor="black",
                label="21cmFAST CI",
            )
            ax[0].plot(x_axis, y_true, linestyle=":", c="black", lw=1.7, label="21cmFAST median")

        tb_ml = x2Tb(x_ml)
        y_ml = np.median(tb_ml, axis=0)
        perc_ml = np.percentile(tb_ml, [low, high], axis=0)
        sigma_ml = 0.5 * (perc_ml[1] - perc_ml[0])

        yerr_lower = y_ml - perc_ml[0]
        yerr_upper = perc_ml[1] - y_ml

        ax[0].errorbar(
            x_axis[::interval],
            y_ml[::interval],
            yerr=[yerr_lower[::interval], yerr_upper[::interval]],
            linestyle="-",
            c=color,
            marker="|",
            markersize=1.5,
            linewidth=lw,
        )

        mask_rel = np.abs(y_true) > y_eps
        mask_std = sigma_true > y_eps

        eps_rel = ((y_ml - y_true) / np.abs(y_true))[mask_rel]
        eps_std = ((y_ml - y_true) / sigma_true)[mask_std]
        eps_sigma = (sigma_ml / sigma_true - 1)[mask_std]

        eps_rel_all.append(eps_rel)
        eps_std_all.append(eps_std)
        eps_sigma_all.append(eps_sigma)

        model_label = _format_model_label(jobid, model_meta.get(jobid, {}))
        ax[1].plot(x_axis[mask_rel], eps_rel, c=color, lw=lw, label=model_label)
        ax[2].plot(x_axis[mask_std], eps_std, c=color, lw=lw)
        ax[3].plot(x_axis[mask_std], eps_sigma, c=color, lw=lw)

        handles_for_legend.append(Line2D([0], [0], color=color, lw=1.5, label=model_label))

    if z_idx is not None:
        # z_idx is interpreted relative to each job's LOS grid; use first job for marker.
        first_jobid = next(iter(los_by_job.keys()))
        los = los_by_job[first_jobid]
        if z_idx < len(los[1]):
            x_mark = los[1][z_idx]
            for ax_i in ax:
                ax_i.axvline(x=x_mark, color="red", linestyle="--", linewidth=1.5)

    ax[0].set_ylabel(r"$\langle T_b \rangle$ [mK]")
    ax[0].grid()

    ax[1].set_ylabel(r"$\epsilon_{rel}$")
    ax[2].set_ylabel(r"$\epsilon_{std}$")
    ax[3].set_ylabel(r"$\epsilon_{\sigma}$")

    mae_rel = np.abs(np.concatenate(eps_rel_all)).mean() if eps_rel_all else np.nan
    mae_std = np.abs(np.concatenate(eps_std_all)).mean() if eps_std_all else np.nan
    mae_sigma = np.abs(np.concatenate(eps_sigma_all)).mean() if eps_sigma_all else np.nan

    ax[1].text(0.01, 0.12, rf"$\overline{{|\epsilon_{{rel}}|}}={mae_rel:.3f}$", transform=ax[1].transAxes)
    ax[2].text(0.01, 0.12, rf"$\overline{{|\epsilon_{{std}}|}}={mae_std:.3f}$", transform=ax[2].transAxes)
    ax[3].text(0.01, 0.12, rf"$\overline{{|\epsilon_{{\sigma}}|}}={mae_sigma:.3f}$", transform=ax[3].transAxes)

    for i in [1, 2, 3]:
        ax[i].set_ylim(-1.99, 1.99)
        ax[i].grid()

    ax[3].set_xlabel("distance [Gpc]")
    ax[3].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x / 1000:.1f}"))

    ax_twin = ax[0].secondary_xaxis("top")
    ax_twin.set_xlim(ax[0].get_xlim())
    ax_twin.set_xlabel("redshift")
    first_jobid = next(iter(los_by_job.keys()))
    los = los_by_job[first_jobid]
    z_ticks_x = ax_twin.get_xticks()
    z_ticks = np.interp(z_ticks_x, los[1], los[0])
    ax_twin.set_xticks(z_ticks_x)
    ax_twin.set_xticklabels([f"{zv:.1f}" for zv in z_ticks])

    ax[0].legend(handles=handles_for_legend, fontsize=8, loc="best")
    plt.subplots_adjust(hspace=0)

    if savename:
        plt.savefig(savename, bbox_inches="tight")
        print(f"Saved figure to {savename}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot global signal: truth vs multiple model hyperparameter settings.")
    parser.add_argument("--real_h5", type=str, required=True, help="Real-data H5 filename. Relative path is resolved under $SCRATCH.")
    parser.add_argument("--target_pattern", type=str, required=True, help="Substring used to match generated files, e.g. 'Tvir4.400-zeta131.341'.")
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
            target_pattern=args.target_pattern,
            jobid=jobid,
            num_image=args.num_image,
            ema=args.use_ema,
            outputs_dir=args.outputs_dir,
            pt_fname=args.pt_fname,
            transform=job_transform,
        )
        print(f"⛳️ Loaded {x_ml.shape=}; {x_true.shape=}; {los.shape=}; {jobid=}")
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
