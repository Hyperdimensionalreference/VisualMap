VisualMap 0.1.0 Windows x64 公开测试版
开发者：Hyperdimensionalreference

从开始菜单打开 VisualMap，浏览器将显示本机工作台。
导入视频 → 预览 → 保存并识别 → 等待实际处理进度 → 打开三维空间。
运行环境和 YOLOX 离线模型已包含，不需要另装 Python 或下载模型。
公开版不内置 GroundingDINO / CLIP 扩展权重，不应称为三模型版。

完整使用说明随安装保存为 WINDOWS_GUIDE.md；也可在线查看：
https://github.com/Hyperdimensionalreference/VisualMap/blob/main/docs/WINDOWS_GUIDE.md

大模型设置是可选项。只有手动发送三张选图才会调用自己配置的视觉 API，
选图、问题及相关帧/时间信息会发送给服务方，可能收费。
不要在公开截图、issue 或配置中泄露 API Key、私人画面或电脑路径。

个人视频、结果和设置默认保存在 %LOCALAPPDATA%\VisualMap。
关闭浏览器不等于退出服务。等任务结束后，从开始菜单“退出 VisualMap”。
升级和卸载默认保留个人数据；重要内容请另行备份。

距离单位为未标定 map_units；体积为估计，不能认证真实尺寸或碰撞安全。
安装包未进行代码签名，也没有独立安全认证。请只从项目 Release 获取，核对 SHA256。
安全软件若报警，请停止运行并保留检测信息，不要关闭防护、加白名单或改名绕过。

请保留 THIRD_PARTY_NOTICES.txt、license_manifest.json、licenses 及必要源码。
自有代码 MIT 不覆盖第三方；微软运行库单独遵循微软条款。
