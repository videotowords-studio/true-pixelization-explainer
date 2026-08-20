# True Pixelizer

True Pixelizer 是一套可在本地运行、后续可直接接入服务器的静态图片真像素化程序。它把源图重建为用户指定的逻辑像素尺寸，限制可见颜色数量，并对透明度、调色板和导出结果做硬性校验。

这里的“真像素”指最终的 `sprite.png` 本身就是例如 `32x32` 的逻辑网格，每个格子只有一个确定的 RGBA 像素值；它不是在一张大图上叠加马赛克滤镜。放大预览只采用最近邻整数倍放大，不会产生模糊过渡色。

## 处理流程

```text
读取静态图片
  -> 根据 Alpha、蒙版或背景键色确定可见区域
  -> 裁切并适配目标画布
  -> 使用预乘 Alpha 面积采样重建逻辑像素网格
  -> 二值化或保留透明度
  -> 自动限色或映射到固定调色板
  -> 写入 PNG 后重新解码复检
  -> 导出精灵图、最近邻预览、调色板和质量报告
```

“预乘 Alpha 面积采样”是一种在缩小时同时考虑颜色和透明度的采样方式，目的是避免透明像素中隐藏的颜色污染角色边缘。

## 当前能力

- 指定最终尺寸，范围为 `1x1` 到 `1024x1024`，总像素不超过 1,048,576。预览还需满足 `宽 x 高 x 放大倍数² <= 16,777,216`；例如 `1024x1024` 最多生成 4 倍预览。
- 指定最大可见颜色数，范围为 1 到 256 色。
- 自动调色板、固定调色板和自动调色板中的锁定颜色。
- 二值透明度或保留采样后的半透明层级。
- `contain`、`cover`、`stretch` 三种画布适配方式。
- 居中或底部居中对齐，可指定内边距。
- 透明源图、显式蒙版和背景键色三种主体处理方式。
- 可选 4x4 Bayer 有序抖色。
- 可重复的 PNG、JSON 报告和 SHA-256 哈希记录。

首版只处理静态单图。动画和 Sprite Sheet 后续应逐帧复用同一个核心接口，并在整组帧之间共享调色板和锚点规则。

## 直接运行

当前机器已具备兼容版本的 Python、Pillow 和 NumPy，可以直接运行：

```bash
cd true-pixelizer
./pixelize pixelize /绝对路径/input.png \
  --output-dir /绝对路径/output-32x32 \
  --size 32x32 \
  --max-colors 8 \
  --alpha binary \
  --preview-scale 8 \
  --json
```

在其他电脑或服务器首次安装时，建议创建独立 Python 环境：

```bash
cd true-pixelizer
python3 -m venv .venv
.venv/bin/python -m pip install .
./pixelize --version
```

这里使用普通安装，以兼容 Python 3.9 新环境自带的旧版 pip。开发人员确实需要可编辑安装时，应先升级 pip，再使用 `pip install -e .`。

程序要求 Python 3.9 或更高版本、Pillow 11 到 12、NumPy 2。限制 Pillow 和 NumPy 的大版本，是为了减少不同机器之间的量化结果漂移。

## 让 Codex 调用

后续可以直接向 Codex 提供源图路径并说明目标参数，例如：

```text
请调用当前项目 true-pixelizer/pixelize，把 /绝对路径/character.png
转换为 32x32、最多 12 色、二值透明的真像素精灵图，结果放到
/绝对路径/character-pixel-32，读取 JSON 结果并告诉我质量检查是否通过。
```

Codex 实际执行的命令应类似：

```bash
./pixelize pixelize /绝对路径/character.png \
  --output-dir /绝对路径/character-pixel-32 \
  --size 32x32 \
  --max-colors 12 \
  --alpha binary \
  --json
```

`--json` 会返回稳定的成功状态、产物绝对路径、哈希、实际颜色数、警告和质量检查结果，便于 Codex 判断任务是否真正完成。

## 常用参数

| 参数 | 作用 | 默认值 |
|---|---|---|
| `--size 32x32` | 最终精灵图的真实逻辑尺寸 | 必填 |
| `--max-colors 8` | 所有可见像素最多使用多少种 RGB 颜色 | `16` |
| `--alpha binary` | `binary` 只保留全透明/全不透明；`preserve` 保留半透明 | `binary` |
| `--alpha-threshold 0.5` | 二值透明的判定阈值，范围 0 到 1 | `0.5` |
| `--fit contain` | 保持比例完整放入、裁切铺满或拉伸铺满 | `contain` |
| `--anchor bottom-center` | 主体居中或贴近底部居中 | `center` |
| `--padding 1` | 目标画布四周预留逻辑像素 | `0` |
| `--trim alpha` | 按可见区域裁切，或保留原始画布 | `alpha` |
| `--preview-scale 8` | 最近邻预览的整数放大倍数 | `8` |
| `--dither bayer4` | 用规则点阵模拟有限颜色之间的层次 | `none` |
| `--mask mask.png` | 使用与源图同尺寸的灰度蒙版控制透明度 | 无 |
| `--key-color "#FF00FF"` | 将指定背景颜色变为透明 | 无 |
| `--key-tolerance 12` | 背景键色的 RGB 容差 | `0` |
| `--max-input-pixels 16000000` | 解码后源图允许的最大像素数，硬上限 2500 万 | `16000000` |
| `--max-input-bytes 100000000` | 读取前允许的最大源文件或蒙版字节数 | `100000000` |
| `--force` | 仅替换输出目录中的同名产物，保留其他文件 | 关闭 |

`--preview-scale` 的可选范围是 1 到 64，但仍受预览总像素上限约束；`--key-tolerance` 的范围是 0 到约 441.7。

固定调色板可以直接传入。十六进制颜色在命令行中应加引号，避免 `#` 被终端当作注释：

```bash
./pixelize pixelize input.png \
  --output-dir output-fixed \
  --size 24x24 \
  --max-colors 4 \
  --palette "#1A1C2C,#5D275D,#B13E53,#F4F4F4" \
  --json
```

也可以使用 JSON 文件：

```json
{
  "colors": ["#1A1C2C", "#5D275D", "#B13E53", "#F4F4F4"]
}
```

然后传入 `--palette-file palette.json`。`--locked-color "#RRGGBB"` 用于要求自动调色板保留某个颜色，可重复使用。

## 固定产物

每次成功处理会生成：

| 文件 | 用途 |
|---|---|
| `sprite.png` | 指定逻辑尺寸的 1 倍真像素精灵图，供游戏直接使用 |
| `preview-8x.png` | 最近邻整数倍预览，倍数随参数变化 |
| `palette.png` | 调色板色块图 |
| `palette.json` | 调色板颜色、使用像素数和最大颜色数 |
| `report.json` | 尺寸、颜色、Alpha、透明隐藏色和 PNG 重读等检查 |
| `manifest.json` | 参数、算法版本、运行环境、输入与输出哈希 |

输出目录默认必须为空，防止误覆盖。使用 `--force` 时只会替换上述同名产物，不会删除目录中的其他文件；源图、蒙版或调色板文件如果与任一目标产物同路径，程序会直接拒绝执行。

## 独立校验已有精灵图

```bash
./pixelize validate output-32x32/sprite.png \
  --size 32x32 \
  --max-colors 8 \
  --alpha binary \
  --json
```

退出码约定：

- `0`：成功，且硬性质量检查通过。
- `2`：参数、输入文件或输出目录不符合要求。
- `3`：处理过程发生异常。
- `4`：图片已读取，但硬性质量检查未通过。

## 服务器接入

命令行和服务器可以共用同一个字节接口：

```python
from true_pixelizer import PixelizeConfig, pixelize_bytes

config = PixelizeConfig(width=32, height=32, max_colors=8)
result = pixelize_bytes(uploaded_image_bytes, config)
sprite_png = result.sprite_png
quality_report = result.report
```

服务器层只需要负责上传鉴权、任务排队、对象存储、超时和资源配额；真像素化规则继续由 `pixelize_bytes()` 负责。上传层仍应在形成完整 `bytes` 之前流式限制请求大小，核心接口中的字节上限是第二道保护。这样本地和线上不会形成两套不同算法。

## 已知边界

- 普通不透明背景会被视为图片内容。程序不会猜测并自动删除背景；应提供透明 PNG、显式蒙版，或用 `--key-color` 指定背景色。
- 目标尺寸过小时，角色神态、装备和轮廓细节可能无法同时保留。机械规则合格不等于美术质量一定合格。
- 面积采样会在重建网格时产生平均色，随后再执行限色；这是把连续图像压入离散像素网格的必要步骤，不是模糊放大。
- `preserve` 表示保留重采样后得到的半透明层级，不保证与源图 Alpha 字节逐一相同。
- 同一输入、参数和依赖版本应得到一致结果；不同 Pillow 或 NumPy 版本可以保证规则合规，但不承诺像素哈希完全一致。
- 首版不处理动画文件、多帧 GIF 或 Sprite Sheet 的自动切帧。

## 开发验证

```bash
cd true-pixelizer
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

当前父仓库的 `.gitignore` 会忽略整个首页原型目录，因此本工具目前适合本机直接使用，但不会自动进入 Git 提交。需要随仓库交付时，应由仓库维护者显式解除 `true-pixelizer/` 的忽略规则，或将该目录迁移到受 Git 跟踪的位置。
