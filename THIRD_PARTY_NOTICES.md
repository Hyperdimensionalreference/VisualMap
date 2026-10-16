# 第三方组件与版权范围

本记录适用于本次**源码仓库**。VisualMap 自有代码的 MIT 不覆盖第三方作品，也不表示任何第三方对本项目背书。

## 仓库内材料

| 材料 | 来源与版权 | 许可与保留方式 |
|---|---|---|
| Geist 字体原文件 | Copyright 2024 The Geist Project Authors；https://github.com/vercel/geist-font | SIL OFL-1.1，原文位于字体旁 `src/visualmap/visualization/fonts/OFL.txt` 和 `LICENSES/Geist-OFL-1.1.txt`；字体未修改，不能仅以 MIT 转发字体。|
| YOLOX 解码/输入输出接口适配 | Megvii YOLOX、OpenCV Model Zoo；https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox | Apache-2.0；网格/步长解码及预处理约定保留此归属，原文 `LICENSES/YOLOX-Apache-2.0.txt`。没有把上游模型或实现冒称为原创。|
| GroundingDINO 句段注意力/输入输出接口适配 | Copyright IDEA Research；https://github.com/IDEA-Research/GroundingDINO | Apache-2.0；`LICENSES/GroundingDINO-Apache-2.0.txt`。本仓库不包含历史量化 ONNX、X-AnyLabeling GUI 或其 GPL 导出脚本。|
| CLIP 图像预处理约定 | Copyright 2021 OpenAI；https://github.com/openai/CLIP | MIT；`LICENSES/CLIP-MIT.txt`。没有分发模型二进制或预计算的私人词向量。|

适配器及项目的修改继续按根 LICENSE 提供；上游各自许可和版权通知同时保留，不能删除。
前端 HTML/CSS/JavaScript 和动效逻辑为本项目代码，未捆绑第三方 CDN 框架、音乐、照片、图标库或示例视频。

## 运行依赖（通过用户自己的包管理器安装）

Python/CPython（PSF 与历史许可）、NumPy（BSD 及 wheel 内嵌组件）、OpenCV（Apache-2.0）、opencv-python 包装（MIT）、PyYAML/LibYAML（MIT）；可选 ONNX Runtime（MIT 与其 ThirdPartyNotices）、tokenizers（Apache-2.0 与 Rust 依赖）、Open3D（MIT）。
本仓库没有复制这些软件的原生运行二进制。用户安装的 wheel 可能包含 FFmpeg、OpenBLAS、微软运行库等，不能仅依据顶层包名判断全部许可。
如果后续将这些依赖打进 EXE/安装器，应按实际版本和实际载荷重新收集全部许可、NOTICE、LGPL 对应源码/可替换库权利及微软分发条款，不能拿本源码说明替代二进制合规材料。

## 模型与数据

本仓库不含权重。可显式下载的 YOLOX-S 文件在 OpenCV Model Zoo 对应目录明确按 Apache-2.0 提供，见 `models/README.md`；运行前核对固定 SHA-256。
GroundingDINO 历史量化 ONNX 的发布者没有提供单独明确的资产许可声明。上游 Apache-2.0、历史转换路径及“工具输出通常继承输入作品版权”的解释，不能替代该发布者单独授权。本仓库保留可选适配代码，但不转发该资产，也不提供自动下载该资产的脚本。
CLIP、其他自行选择的模型、训练数据、图像、视频与音乐分别遵循各自条款；请确认自己的用途和分发权。不附带 COCO/TUM/Mixkit 素材或作者私人视频。
开源许可证不能替代对专利、训练数据权利、肖像/隐私或商标使用的核查。本记录不提供零侵权或专利清权保证。
