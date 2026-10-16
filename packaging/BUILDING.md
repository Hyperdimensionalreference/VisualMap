# Windows 公开测试安装包构建

本配方只打包公开源码、YOLOX 离线模型和明确列出的运行资源。不能把原开发目录、私人数据或历史三模型安装包直接上传。

使用 Windows x64、CPython 3.13.7 和 NSIS 3.13；在独立虚拟环境安装 `requirements-build.txt`。此构建含微软运行库，构建者还应核对自己具备适用的微软分发权；开源说明本身不授予微软许可。

从同一 [Release](https://github.com/Hyperdimensionalreference/VisualMap/releases/tag/v0.1.0) 下载 `VisualMap-0.1.0-ThirdParty.zip`，SHA-256：

```text
b6683402dc3585501c91c008435379913f8480cfab8dd5e5addd2cf98a7c220a
```

校验后解压到 `work/windows/legal`，其中应直接有 `licenses/` 和 `legal_sources.json`。普通使用者不需要此压缩包；同样的许可和必要源码已经包含在安装器内。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r packaging/requirements-build.txt
.venv\Scripts\python.exe scripts/download_yolox.py
.venv\Scripts\python.exe packaging/build_windows.py --legal-dir work/windows/legal --makensis D:\tools\nsis-3.13\Bin\makensis.exe
```

`--makensis` 应替换为自己从 NSIS 官方分发取得的编译器位置。脚本不会安装全局工具，也不会自动下载扩展模型。产物位于 `release/`，构建与检查中间数据在被 Git 忽略的 `work/windows/` 和 `outputs/windows/`。

运行时固定在安装目录，避免每次启动重新解压模型。不包含 CUDA、Torch、Open3D、训练工具或模型转换器；未用 UPX。公开默认配置禁用开放词汇/CLIP，并清理未携带资产的路径；其他识别、建图和工作台源代码沿用公开版本。

`prepare_bundle.py` 以白名单拷贝一个 YOLOX ONNX，`scripts/download_yolox.py` 固定模型哈希。字体保留 OFL 原文。`license_manifest.json` 记录实际载荷、第三方来源及随包许可/源码的哈希，`bundle_manifest.json` 只记录程序自己拥有的文件，不含它自身。安装器卸载时逐文件移除自己的载荷，保留额外文件和个人数据；不会递归删除用户选择的安装目录。

可以先使用 `--freeze-only` 生成目录程序，完成验证后使用 `--installer-only` 生成安装器。重建时使用新的构建目录，防止残留旧文件污染载荷。默认安装与数据位置、退出、升级和卸载说明见 [使用指南](../docs/WINDOWS_GUIDE.md)。

第三方压缩包含完整运行时许可、FFmpeg 7.1 及相关依赖源码、OpenCV 对应构建脚本、NSIS/LZMA 源码及条款。以其中的 `legal_sources.json` 为来源记录，构建时重新生成实际文件清单。没有宣称逐位重建 FFmpeg 或历史模型，也没有以自有 MIT 覆盖第三方许可。

发布前按 [检查清单](../docs/RELEASE_CHECKLIST.md) 复核功能、载荷隐私、第三方义务、安全检查与最终哈希。构建工具或原生库更新后必须重新核对许可，不能沿用旧清单冒称新版本已审查。此配方不自动签名或上传，不声称二进制可逐位复现。
