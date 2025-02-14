import numpy as np
import matplotlib
from vedo import Volume, Plotter

# 定义自定义 colormap
def get_eor_cmap(vmin=-150, vmax=30):
    name = f"EoR-{vmin}-{vmax}"
    negative_segments = 4
    positive_segments = 2
    neg_frac = abs(vmin) / (vmax - vmin)
    neg_seg_size = neg_frac / negative_segments
    pos_frac = abs(vmax) / (vmax - vmin)
    pos_seg_size = pos_frac / positive_segments

    EoR_colour = matplotlib.colors.LinearSegmentedColormap.from_list(
        name,
        [
            (0, 'white'),
            (0.33, 'yellow'),
            (0.5, 'orange'),
            (0.68, 'red'),
            (0.83333333, 'black'),
            (0.9, 'blue'),
            (1, 'cyan')
        ])

    try:
        matplotlib.colormaps.register(cmap=EoR_colour)
    except ValueError:
        matplotlib.colormaps.unregister(name)
        matplotlib.colormaps.register(cmap=EoR_colour)

    return EoR_colour

# 加载数据
data = np.load("Tb.npy")  # 你的实际数据
print(f"data.shape = {data.shape}, data.min() = {data.min()}, data.max() = {data.max()}")

# 获取自定义 colormap
cmap = get_eor_cmap(vmin=-150, vmax=30)

# 将 matplotlib colormap 转换为 vedo 识别的格式
def matplotlib_to_vedo_cmap(mpl_cmap, n_colors=256):
    # 从 matplotlib colormap 中提取颜色并生成颜色样本
    colors = mpl_cmap(np.linspace(0, 1, n_colors))
    return colors

# 转换 colormap
vedo_cmap = matplotlib_to_vedo_cmap(cmap)

# 创建 Volume 对象并应用自定义 colormap
vol = Volume(data).cmap(vedo_cmap).mode(1)  # mode(1) 使用 MIP 渲染模式

# 创建 Plotter 并显示
plotter = Plotter(bg='black', size=(800, 600))  # 设置黑色背景，限制窗口大小
plotter.renderer.SetUseDepthPeeling(True)  # 启用 Depth Peeling
plotter.show(vol, axes=1, interactive=True)
plotter.camera.Zoom(5)  # 放大 5 倍

