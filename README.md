# 真像素化工具与原理说明

页面包含两个 TAB：

- 「工具使用」：上传图片并设置参数，调用项目中的 `true-pixelizer` 完成真像素化。
- 「查看原理」：用六阶段动画展示真像素化的处理过程。

在线访问：

- Cloudflare Pages：<https://true-pixelization-explainer.pages.dev/>
- GitHub Pages：<https://videotowords-studio.github.io/true-pixelization-explainer/>

在线页面通过 Pyodide 在浏览器内运行同一套 Python 核心，图片不会上传到第三方服务。浏览器运行组件与页面同域托管，首次使用需要下载约 10 MB 的压缩资源，后续可使用浏览器缓存。

## 启动页面

在项目目录运行：

```bash
python3 server.py
```

然后访问 <http://127.0.0.1:8765/>。

本地启动时优先使用 `server.py` 提供的处理接口；静态托管时自动切换为浏览器内处理。不要使用 `file://` 直接打开 `index.html`，否则浏览器无法载入 Worker 和 Python 源文件。

## 下载产物

处理完成后，页面可下载以下文件：

- `sprite.png`：指定逻辑尺寸的真像素精灵图。
- `preview-{倍数}x.png`：最近邻整数倍预览图。
- `palette.png`：调色板色块图。
- `report.json`：质量校验报告。
- `manifest.json`：处理参数、版本和文件哈希记录。
- `true-pixelizer-result.zip`：全部处理产物，其中还包含 `palette.json`。
