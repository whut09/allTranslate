# allTranslate

allTranslate 基于 [guaguastandup/zotero-pdf2zh](https://github.com/guaguastandup/zotero-pdf2zh) 项目继续优化，重点改进了论文 PDF 中表格区域的识别、保留与翻译流程，使复杂版式下的译文质量和排版稳定性达到商用收费软件水平。

项目面向本地命令行和本地 WebUI 使用，不再提供容器相关启动方式。翻译器的 key、url、模型等信息被拆分到单独配置文件中：提交到 GitHub 的 `translator_config.json` 使用 `xx` 隐藏敏感值，本地运行时自动读取未提交的 `translator_config.local.json`。

## 原理简述

allTranslate 不做整页 OCR，而是直接解析 PDF 内部文字、字体、坐标和图形结构。程序先用版面分析模型识别正文、标题、图片、表格等区域，再按段落或表格单元组织待翻译文本，调用配置的翻译服务生成译文，最后把译文写回原始版面，输出单语译文 PDF 和双语对照 PDF。

表格优化的核心思路是减少“把表格当普通正文”的误判：在布局检测阶段保留表格边界与单元结构，在翻译阶段按更细粒度处理表格内容，在回写阶段尽量维持原表格的列宽、行高和文字位置，从而减少错列、串行、遮挡和表格整体漂移。

## 运行流程

1. 安装 Python 3.11 或 3.12。
2. 在项目根目录安装依赖：

```bash
pip install -e .
```

3. 按需编辑公开配置 `config.json` 和 `translator_config.json`。真实 key、url 请写入 `translator_config.local.json`，该文件已加入 `.gitignore`，不会提交到 GitHub。
4. 使用命令行翻译 PDF：

```bash
allTranslate input.pdf -s qwen-mt -o output
```

5. 启动本地 WebUI：

```bash
allTranslate -i
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `-s, --service` | 指定翻译服务，例如 `qwen-mt`、`openailiked`、`openai` |
| `-li, --lang-in` | 源语言，默认 `en` |
| `-lo, --lang-out` | 目标语言，默认 `zh` |
| `-p, --pages` | 指定页码，例如 `1,3-5` |
| `-o, --output` | 输出目录 |
| `-t, --thread` | 翻译并发数 |
| `--config` | 指定主配置文件 |
| `--ignore-cache` | 忽略缓存，强制重新翻译 |

## 配置说明

`config.json` 保存通用运行配置，例如默认语言、启用的服务、WebUI 显示策略等。

`translator_config.json` 保存可提交的翻译器配置模板，敏感字段统一写成 `xx`：

```json
{
    "translators": [
        {
            "name": "qwen-mt",
            "envs": {
                "ALI_DOMAINS": "scientific",
                "ALI_API_KEY": "xx",
                "ALI_MODEL": "qwen-mt-turbo"
            }
        }
    ]
}
```

`translator_config.local.json` 保存本机真实 key 和 url。程序启动时会先读取公开配置，再用本地配置覆盖同名翻译器的字段，因此本地可直接使用，提交后敏感信息仍保持隐藏。

## 致谢

感谢 [guaguastandup/zotero-pdf2zh](https://github.com/guaguastandup/zotero-pdf2zh) 提供的开源基础与 Zotero/PDF 翻译实践，本项目在其思路上继续改进表格翻译与本地运行体验。
