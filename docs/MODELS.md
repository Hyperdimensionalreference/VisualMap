# 模型获取与可选功能

默认工作台使用 YOLOX-S（COCO 80 类）；公开配置未启用开放词汇补充。所有适配代码和配置参数保留，默认差异是公开发行的许可边界，不是精度升级。

## 默认 YOLOX

明确执行 `python scripts/download_yolox.py` 才联网。脚本只从 OpenCV 官方分发域下载，SHA-256 必须为
`c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`，大小 35,858,002 字节。
官方说明与许可：https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox 。该目录声明 Apache-2.0。
单独运行下载命令后，可完全离线执行目标检测；COCO 没有独立 tablet 类，手机/平板等候选需要人工核对。

## 可选 GroundingDINO + CLIP

`configs/open_vocabulary.example.yaml` 描述所需模型、分词器、提示词、来源哈希清单及 CLIP 词向量文件，接口实现位于 `src/visualmap/detection/`。
需要 `pip install -e ".[recognition]"` 并由使用者提供有权使用的资产。编辑工作台的 `configs/spatial_consistent.yaml` 才会影响工作台，CLI 可显式传 `--config`。
请用自己的合法权重及对应哈希/模型卡补齐路径，不能简单把网上任意 ONNX 当成同一接口。
上游：https://github.com/IDEA-Research/GroundingDINO 、https://github.com/openai/CLIP 。
历史 X-AnyLabeling 发布量化 ONNX 的单独资产授权尚未澄清，因此本仓库不分发它，也不自动取得它。

模型权重、分词器和嵌入缓存默认都被 Git 排除。新增公开资产前重新核查作者、版本、完整许可证、NOTICE、来源及 SHA-256，不能靠修改文件名解决授权问题。
