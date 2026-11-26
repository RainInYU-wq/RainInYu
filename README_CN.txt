BE Item Transform Inspector 0.1.2 - BTG 原版持物测量原型
======================================================
这是 C++ 源码工程，不是已编译 DLL，也不是可直接导入的 mcmod。
目标：Windows x64，LeviLamina 客户端 SDK 26.60.9。已根据用户截图确认游戏 1.26.60.05 GDK、LeviLamina 26.60.9。
沿用 0.1.1 的 v26.60.9 三个 Hook 声明、MatrixStackRef、Matrix 和 ItemContextFlags 以及 xmake 版本依赖。没有在 Windows 或 Minecraft 内实测，符号解析、编译及运行仍待验证。

【编译，二选一】
A. GitHub Actions（本机无需安装 C++ 工具链）
1. 新建自己的空白 GitHub 仓库，将本压缩包解压后的全部内容提交到 main，
   注意保留隐藏目录 .github/workflows/build.yml。
2. 仓库 Actions -> Build Windows client DLL -> Run workflow（推送 main 也会触发；如已自动运行，无需再点击）。
3. 若构建成功，下载该次运行的 Artifacts：BEItemTransformInspector-Windows-client。
   里面应包含 DLL 和 manifest.json。失败则下载/复制完整报错日志。
B. 本地：安装 Git、Visual Studio 2022（桌面 C++、Windows SDK）、LLVM 22、
   xmake 3.x，确保 xmake 和 clang-cl 在 PATH；运行 Build.cmd。
   成功时查看 bin/BEItemTransformInspector，失败时上传 build.log。
   不要通过更改 SDK 版本掩盖符号错误；先核对实际 LeviLamina 版本。

【安装】
完全退出游戏。使用独立测试实例，确认已启用对应版本客户端 LeviLamina。
将成功生成的 DLL 与 manifest.json 放在同一个 BEItemTransformInspector 文件夹，
将整个文件夹复制到该实例 mods 目录。仅启用这个测量模组，先关闭 BTG 与其他
原生模组。启动游戏。不需要输入脚本命令、不需要修改存档。

【采集】
1. 新建临时创造世界，使用默认皮肤，关闭其他资源包。不要使用正式开发地图。
2. 准备普通未附魔的铁镐、石镐，最好加铁剑作对照。不要放置掉落物或盔甲架。
3. 第一人称：手持铁镐，静止 4 秒，再挥镐 4 秒。
4. 按 F5 切换第三人称背视，静止 4 秒，再挥镐 4 秒。
5. 按 F5 切换第三人称正视，静止 4 秒。
6. 用石镐重复以上步骤。可选铁剑、盾牌、三叉戟。
7. 退出并关闭游戏。将 mods/BEItemTransformInspector/ItemTransformDump.txt 上传。
   文件每次启动以新 session 追加，重测时可在关闭游戏后将旧文件改名。
   若只看到 HOOK_STATUS 没有 SAMPLE，或加载失败，发送该文件及加载器日志。
   暂时不要启用 BTG 同时采集，先获得干净的原版数据。之后可另开会话，仅启用对应 BTG BP/RP，记录同一物品对照；请分开保存两份 TXT。
可以继续切换其他物品，无需修改源码或重新编译；不是自动扫描整个物品注册表。

【输出说明】
每 250ms 至多记录一次同类别，单类别最多 32 个，整个会话最多 50000 个；达到上限会记录提示。
取消物品白名单，保留完整物品 ID，排除空手和 UI。OBSERVED_RENDER_ITEM 表示进入 renderItem，不代表有矩阵样本。特殊渲染路径可能没有 SAMPLE。
BEFORE_DEFAULT / AFTER_DEFAULT：默认物品变换函数调用前后的 4x4 矩阵。
CANDIDATE_POSTMULTIPLY_DELTA = inverse(before) * after。
CANDIDATE_PREMULTIPLY_DELTA = after * inverse(before)。两者不是已确认的握姿。
flags 按 26.60.9 头文件解释；WORLD_HELD_CANDIDATE 为世界渲染候选，
UNKNOWN 表示未进入上下文钩子，不应猜为第三人称。记录不含玩家账号或世界数据。
仅观察默认变换函数；不能保证覆盖全部原版物品渲染分支。
没有实现网格顶点、UV、原版手骨骼矩阵或 GPU 最终绘制矩阵读取。
若默认函数未被调用或矩阵不足以校准，还需要下一轮定向测量。

【安全与卸载】
钩子会调用原函数，不主动修改矩阵、物品或世界，也不发送任何网络数据。
原生模组仍可能因游戏版本、ABI 或其它钩子冲突而崩溃。退出后移出整个模组文件夹即可。
本项目为未编译诊断原型，不应用于唯一正式存档。

【参考】
https://github.com/LiteLDev/LeviLamina/tree/v26.60.9
https://github.com/LiteLDev/levilamina-mod-template
https://levimc.org/faq/develop/levilamina/build/
