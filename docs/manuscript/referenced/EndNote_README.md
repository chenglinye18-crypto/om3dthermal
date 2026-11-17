# EndNote 可编辑引用版

2026-11-17。使用本机 EndNote 21 与 Word Cite While You Write 生成。

- `DAC2027V1_EndNote.docx`：37 处 EndNote 引用域、28 条自动参考文献，按首次出现顺序编号。
- `DAC2027V1_EndNote.pdf`：7 页，正文 6 页、参考文献 1 页。
- `IOM3D_EndNote_package.zip`：解压后包含 `IOM3D_DAC2027.enl`、同名 `.Data` 目录和 `IEEE IOM3D DAC2027.ens`。文献库与 `.Data` 必须保存在同一目录。
- `references_audit.md`：文献支持边界和原稿问题清单，仍然适用。

## 后续使用

1. 解压压缩包，打开 `.enl` 文献库。
2. 双击 `.ens`，在 EndNote 中保存该样式；本机已安装。
3. 打开 Word 文件，在 EndNote 21 中选择 `IEEE IOM3D DAC2027`。使用 Insert Citation 插入新引用，Update Citations and Bibliography 更新编号和文献表。不要转换为纯文本，否则会失去动态引用域。
4. 原稿含数学大括号，插件更新时偶尔会将公式误识别为临时引用；如果匹配窗口显示公式而非作者或文献，点击 Ignore，不插入替代条目。

样式基于 EndNote 自带 IEEE，会议论文模板补充页码与 DOI。参考文献采用 Times New Roman 9 pt、单倍行距、18 pt 悬挂缩进；清除了原稿 Normal 样式的中文首行字符缩进继承。正文、图注和图片格式未调整。

## 完整性核验

- 与 `DAC2027V1(1).docx` 比较：110 个正文段落除引用标记外文字一致。
- 原稿 21 个 `word/media` 文件逐字节一致。
- 两个原生 Office Math 公式内容一致；其余公式图片亦由媒体一致性覆盖。
- 37 个 `ADDIN EN.CITE` 域、1 个 `ADDIN EN.REFLIST` 域，28 条文献均有正文引用。
- PDF 全 7 页已逐页检查。
- 证据不足的 3 处 `[X]` 按原稿保留，详见审计报告；没有为了凑够 30–35 篇强加不相关引用。

之前的 `DAC2027V1_referenced.docx` 是手工编号版。继续维护引用时请使用本次 `_EndNote.docx`，不要混用两个版本。
