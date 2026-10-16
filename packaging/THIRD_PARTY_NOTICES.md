# VisualMap 0.1.0 — 第三方组件、模型与分发说明

核对日期：2026-10-16。适用范围：本版 Windows x64 安装包。VisualMap 自有程序代码采用 MIT 许可证；第三方软件、模型和字体保留各自的许可证，不能把整个安装包一概改称 MIT。版权人、免责条款、NOTICE 和完整许可证随安装保存在 `licenses/`；精确文件大小、SHA-256、来源及分发方式见 `license_manifest.json`。

## 运行时软件

| 组件 | 实际版本与用途 | 许可及随包材料 |
|---|---|---|
| CPython | 3.13.7，Python 解释器和所需标准库 | PSF-2.0 及历史许可证；`licenses/CPython/LICENSE.txt`、`Doc-license.rst`。解释器本体未修改；本版由 PyInstaller 只收集运行所需模块并冻结为字节码，不附带 Python 开发工具。|
| NumPy | 2.5.3，矩阵和图像数值处理 | BSD-3-Clause；`licenses/numpy/` 原样保留 wheel 所附全部嵌套许可，包括 OpenBLAS/LAPACK 和 GCC 运行库例外。GCC 运行库采用 GPL-3.0-or-later WITH GCC-exception-3.1，符合例外的独立应用无需因此改用 GPL。|
| OpenCV / opencv-python | 5.0.0 / 5.0.0.93，图像、视频、特征和 DNN | OpenCV Apache-2.0，Python 包装脚本 MIT；完整 wheel 的 `LICENSE.txt`、`LICENSE-3RD-PARTY.txt` 原样保存在 `licenses/opencv-python/`，包含其静态原生组件和 Intel IPP 通知。|
| FFmpeg | 7.1，OpenCV 动态视频插件 | LGPL-2.1-or-later；本包 DLL 为 `opencv_videoio_ffmpeg500_64.dll`。许可在 `licenses/ffmpeg/`，对应源版本、构建脚本及相关依赖源码在 `licenses/ffmpeg-build/` 和 `licenses/sources/`。|
| ONNX Runtime CPU | 1.31.0，可选 ONNX 接口的 CPU 运行库，本公开配置不启用扩展模型 | MIT；`licenses/onnxruntime/LICENSE.txt`、完整 `ThirdPartyNotices.txt`，不能只保留微软一行版权。|
| Hugging Face tokenizers | Python 0.23.3，原生 Rust tokenizers 0.23.2；可选接口运行库，本公开包不含 BERT 分词器资产 | Apache-2.0；`licenses/tokenizers/LICENSE.txt`、wheel 自带 SBOM 及 `rust-crates/` 中按 130 个组件条目收集的许可证与版权。SBOM 含所有目标/构建依赖，因而是本机实际二进制的归属上界，不能据此声称每个条目都在运行。|
| PyYAML / LibYAML | 6.0.3 / 0.2.5，配置解析 | MIT；`licenses/PyYAML/`。|
| OpenSSL | 3.0.16，CPython HTTPS | Apache-2.0；`licenses/CPython/OpenSSL-LICENSE.txt`。CPython 另含 libffi、bzip2、Expat、zlib、mpdecimal 和 public-domain liblzma 等独立组件，所需版权材料在同目录及 CPython 原始许可文档中。|
| Microsoft Visual C++ / Universal CRT 运行库 | 未修改的 MSVC release DLL、Universal CRT/API sets，支持 CPython、NumPy、OpenCV、ONNX Runtime | 微软专有许可，**不是开源或 MIT**。`licenses/Microsoft/VC-Runtime-license.txt` 和 Visual Studio Community 2026 的分发条款分别说明接收者使用权及开发者分发权。CPython 原始 LICENSE 同样明确说明其 Windows 解释器所附微软 Distributable Code 的继续分发条件；构建机已核实安装 Visual Studio Community 2026，附有其官方 REDIST 指向；不随包分发开发工具、debug_nonredist 或预览运行库。|
| Geist 字体 | 本地 Geist-Variable.woff2 | SIL OFL-1.1；Copyright 2024 The Geist Project Authors；`licenses/Geist/OFL.txt`。字体本身保持 OFL；未改名、未修改字形，也不作为单独字体产品销售。|

FFmpeg 还包含 libvpx v1.16.0 和 AOM v3.14.1；各自的许可/专利文本保存在 `licenses/libvpx/`、`licenses/aom/`，完整源码随包。OpenH264 仅引用 Cisco v2.5.0 的公共接口头文件和动态加载包装层；本包没有 OpenH264 编解码 DLL，也未把 Cisco 二进制专利条款当成任意编码器授权。本包编译配置没有启用 FFmpeg GPL 或 nonfree 组件。视频格式所涉专利的适用性与开源版权许可是不同问题，本文不提供专利许可。

本版最低要求 Windows 10，使用 Windows 自带的 Universal CRT；不再分发无效的本地 `ucrtbase.dll` 副本。微软文档明确 Windows 10/11 始终使用系统目录中的 UCRT，即使应用附带更新的副本也一样。应用专用的 MSVC release DLL 和兼容 API set 转发文件仍随包保留，归属及条款不变。依据：[微软 UCRT 部署说明](https://learn.microsoft.com/en-us/cpp/windows/universal-crt-deployment?view=msvc-170)。

## 随包识别模型与接口归属

公开安装包只内置 OpenCV Model Zoo 的 YOLOX-S（Apache-2.0），保留原模型文件、官方模型目录许可证和来源 SHA-256。
GroundingDINO / CLIP 只保留接口适配代码及其 Apache / MIT 归属；不含它们的权重、分词器、嵌入缓存、GPL 导出脚本或历史 fork 源码。历史量化 ONNX 的独立分发授权尚未明确，因此未在公开包中分发。

## FFmpeg 的接收者权利与源码

本包保留了完整 LGPL 许可、原始版权、FFmpeg 7.1 源码，以及与 DLL 精确匹配的 OpenCV 构建提交 `06dc20cad65dc7fcf784f70c95d46750520889a7` 的脚本/包装层。DLL 的 MD5 `a821a1135251859655090c795af05789` 与 OpenCV 5.0.0 官方 ffmpeg.cmake 指定值一致；SHA-256 记于清单。libvpx、aom 和 OpenH264 公共头文件的源码与上游脚本指定版本一致。

接收者可依 LGPL 修改并重新构建视频库，退出程序后替换安装目录中的该 DLL；软件不对该库做哈希锁定，不限制为调试 LGPL 库修改而进行的逆向工程。`licenses/sources/README_SOURCES.md` 提供文件位置和构建入口。源码离线随包，不以一个可能失效的外部链接或没有履行安排的“按需提供”承诺代替源码。未在本次打包中重新编译/逐位复现 FFmpeg DLL。

## 打包工具与源代码归属

- PyInstaller 6.22.3 与 hooks-contrib 2026.8：GPL-2.0-or-later 加 bootloader exception；运行时 hooks 为 Apache-2.0。未修改 bootloader。该例外允许把生成的可执行程序按本项目及各依赖原许可分发，不要求因此把应用改为 GPL。完整原文见 `licenses/PyInstaller/` 与 `licenses/build-tools/`。
- NSIS 3.13：主体 zlib/libpng 许可；LZMA 模块为 CPL-1.0 加官方链接例外，未修改。应用不因使用安装器而继承 CPL；该模块的原文和源码随包见 `licenses/NSIS/` 与 `licenses/sources/nsis-3.13.tar.gz`。
- 构建期用到的 setuptools、packaging、altgraph、pefile、pywin32-ctypes 等按实际构建环境保留了版权文本；构建工具不等同于全部在接收者机器运行。以最终安装目录清单为准。
- 本地适配器参考的 YOLOX 网格/步长解码归属 Megvii/OpenCV Zoo；GroundingDINO 句段注意力接口归属 IDEA-Research；CLIP 预处理约定归属 OpenAI。相关上游许可均随包。前端使用项目自身的 HTML/CSS/JavaScript；没有打包第三方 CDN 框架、Three.js 或外部图标库；本版安装图标为项目自行生成。

## 本版未分发的项目内容

GroundingDINO / CLIP 权重与词向量、BERT 分词器、YOLO-World 实验 ONNX、Open3D、PyTorch、训练工具及演示用 Mixkit 音乐均未打入本包。COCO/TUM 测试数据及其图片/视频、个人视频、识别结果、模型 API 密钥、日志和本机环境也不属于本版分发内容。不要把整个开发目录作为安装包一起分享；新增任何这类文件都要重新检查授权与隐私。

## 使用与继续分享

可以把完整、未篡改的 VisualMap 安装包分享给他人。请保留 `LICENSE`、本说明、`license_manifest.json`、`licenses/` 和其中源码。自有代码的 MIT 不覆盖第三方作品；不得去掉其版权、把第三方名称用作背书，或把微软运行库作为独立产品转售/分发。微软组件受各自许可保护，开源组件按各自许可享有的修改、再分发与调试权利不受本文额外限制。

本记录落实已确认的版权归属和分发材料，不等于任何权利人背书，也不是“零侵权”保证。更换二进制、模型或字体后，应更新哈希与授权记录。
