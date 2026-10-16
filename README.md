# VisualMap

本机视频识别与三维空间工作台。通过普通单目视频查看相机轨迹、稀疏三维点、物体观测与空间证据。

A local-first monocular video workspace for sparse reconstruction, object evidence, and reviewable spatial visualization.

> 研究与学习用途：距离单位为未标定的 `map_units`，体积是估计。跟踪覆盖率、模型分数与多帧一致性均不是识别准确率；不能作为碰撞安全、米制测量或品牌识别认证。

## 功能

- 本机视频库：导入、预览、搜索、处理进度、多次结果、回收站与显式永久删除。
- OpenCV 特征跟踪、三角化、PnP、重定位与独立子地图；没有配准的空间段不混用坐标。
- YOLOX 目标检测、逐帧证据、物体轨迹及浏览器三维查看；缺几何证据时保留二维观测，不编造位置。
- 可选 GroundingDINO / CLIP 接口，代码保留；模型获取与授权由使用者单独核实。
- 可选手动三图大模型复核：明确发送后才调用用户配置的视觉接口；名称确认、撤销及重开持久化。
- 托管视频的内容相同副本整理，不重新压缩原视频；JSON/HTML 无损压缩。

## 快速开始

**Windows 安装使用**：从 [Release 下载公开测试版安装包](https://github.com/Hyperdimensionalreference/VisualMap/releases/tag/v0.1.0)，无需另装 Python，已含 YOLOX 离线模型。完整步骤见 [安装与使用说明](docs/WINDOWS_GUIDE.md)。本版未代码签名，不内置 GroundingDINO / CLIP 扩展权重；源码使用方法如下。

推荐 Python 3.13、Windows 10/11 x64。当前回归主要在 Windows 验证；其他平台及 Python 版本未逐一验收。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe scripts/download_yolox.py
.venv\Scripts\python.exe -m visualmap.web_app
```

浏览器打开终端提示的本机地址（默认 `http://127.0.0.1:8765/`），导入自己的视频后手动开始识别。
模型下载是独立、明确的命令，启动程序不会偷偷下载权重。下载约 36 MB，核对固定 SHA-256 后才保存。
源码版默认数据在本项目目录；可用 `--data-dir <独立目录>` 改变保存位置。关闭网页不会退出服务，可在任务结束后运行同一命令加 `--stop`。

不下载模型的合成素材冒烟：

```powershell
.venv\Scripts\python.exe scripts/make_sample_video.py
.venv\Scripts\python.exe -m visualmap.app --config configs/default.yaml --video sample_data/sample.avi --run-name synthetic_demo
```

这里的 `demo` 检测器仅识别合成颜色区域，用于处理链路验证，不代表自然物体识别能力。

## 开放词汇与大模型

公开默认配置使用 YOLOX；`configs/open_vocabulary.example.yaml` 保留完整扩展配置，但没有权重、分词器或文本嵌入缓存。
先阅读 [模型说明](docs/MODELS.md)，用自己有权使用的资产补齐配置，再明确启用。此源码发布没有分发历史量化 ONNX，也没有声称该 ONNX 获得发布者单独授权。
大模型复核需要支持多图的 OpenAI 兼容视觉 API。调用可能收费，选区、问题和源帧时间会发送给配置的服务方；详见 [隐私说明](PRIVACY.md)。

## 开发与验证

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
node --test tests/frontend/*.cjs
.venv\Scripts\python.exe scripts/audit_public_tree.py
```

完整回归包含本机回环接口、真实合成视频处理、存储与复核安全边界。测试不需要作者视频、真实 API Key 或付费服务。
CI 使用固定提交的 GitHub Actions、只读仓库权限和无凭据 checkout，不运行来自 PR 的特权发布脚本。

## 许可与分发

VisualMap 自有代码采用 [MIT](LICENSE)。字体及参考接口保留各自归属和原许可，见 [第三方说明](THIRD_PARTY_NOTICES.md) 与 `LICENSES/`。
依赖、模型、视频、音乐及数据集不是因为本项目 MIT 就自动变成 MIT；不得擅自转发他人素材。
Git 源码不含作者个人视频、结果、密钥、电脑路径、日志、临时诊断 EXE 或模型二进制。Windows 公开测试安装包单独放在 Release，按公开配置构建，不上传历史三模型安装器；内含第三方许可和必要源码。[构建说明](packaging/BUILDING.md) 和 [发布检查](docs/RELEASE_CHECKLIST.md) 分别记录实际载荷与后续复核要求。

发现漏洞请按 [安全政策](SECURITY.md) 私下报告；普通问题请先移除密钥、视频画面、个人路径及其他身份信息。贡献约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。
