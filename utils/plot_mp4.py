import re
from pathlib import Path
from PIL import Image
import subprocess
import math

def plot_mp4(jobid, seconds=4, transform=None):
    outdir = Path("panels")
    outdir.mkdir(exist_ok=True)
    
    # 获取 redshift
    z_values = sorted({
        int(re.search(r"_z(\d+)", p.name).group(1))
        for p in Path(".").glob(f"Tbs_grid_{jobid}_*_z*.png")
    })

    fps = len(z_values) // seconds
    if fps == 0:
        fps = 1
        
    panel_frames = []
    
    for z in z_values:
        tbs = Image.open(next(Path(".").glob(f"Tbs_grid_{jobid}_*_z{z}.png")))
        top2 = Image.open(next(Path(".").glob(f"global_Tb_{jobid}_*_z{z}.png")))
        ps   = Image.open(next(Path(".").glob(f"power_spectrum_{jobid}_*_z{z}.png")))
        sc   = Image.open(next(Path(".").glob(f"scattering_coefficients_{jobid}_*_z{z}.png")))
    
        # ------------------------
        # 第二排高度 = 两张图保持相同高度
        # ------------------------
        H_mid = max(top2.height, ps.height)
        scale_top2 = H_mid / top2.height
        scale_ps   = H_mid / ps.height
        top2_scaled = top2.resize((int(top2.width * scale_top2), H_mid), Image.LANCZOS)
        ps_scaled   = ps.resize((int(ps.width * scale_ps), H_mid), Image.LANCZOS)
    
        # 第二排总宽度
        W_mid = top2_scaled.width + ps_scaled.width
    
        # ------------------------
        # 第一排和第三排宽度 = 第二排总宽度，高度按比例缩放
        # ------------------------
        scale_tbs = W_mid / tbs.width
        tbs_scaled = tbs.resize((W_mid, int(tbs.height * scale_tbs)), Image.LANCZOS)
        
        scale_sc = W_mid / sc.width
        sc_scaled = sc.resize((W_mid, int(sc.height * scale_sc)), Image.LANCZOS)
    
        # ------------------------
        # 画布总尺寸
        # ------------------------
        H_total = tbs_scaled.height + H_mid + sc_scaled.height
        if H_total % 2 == 1:
            H_total += 1
        canvas = Image.new("RGB", (W_mid, H_total), "white")
    
        # ------------------------
        # 粘贴图像
        # ------------------------
        y = 0
        canvas.paste(tbs_scaled, (0, y))
        y += tbs_scaled.height
        canvas.paste(top2_scaled, (0, y))
        canvas.paste(ps_scaled, (top2_scaled.width, y))
        y += H_mid
        canvas.paste(sc_scaled, (0, y))
    
        # ------------------------
        # 保存
        # ------------------------
        out = outdir / f"panel_z{z:04d}.png"
        canvas.save(out)
        panel_frames.append(out)
    
    print(f"Generated {len(panel_frames)} scaled panel frames in {outdir}/")
    
    # ------------------------
    # 用 ffmpeg 合成 MP4
    # ------------------------
    with open("panel_list_scaled.txt", "w") as f:
        for p in panel_frames:
            f.write(f"file '{p.resolve()}'\n")
    
    mp4_out = f"{transform}_{jobid}.mp4"
    subprocess.run([
        "ffmpeg",
        "-y",
        "-r", str(fps),
        "-f", "concat",
        "-safe", "0",
        "-i", "panel_list_scaled.txt",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        mp4_out
    ], check=True)
    
    print(f"Created MP4: {mp4_out}")
