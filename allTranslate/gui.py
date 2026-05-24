import asyncio
import cgi
import os
import shutil
import socket
import uuid
from asyncio import CancelledError
from pathlib import Path
import typing as T

import gradio as gr
import requests
import tqdm
from gradio_pdf import PDF
from string import Template
import logging

from allTranslate import __version__
from allTranslate.high_level import translate
from allTranslate.doclayout import ModelInstance
from allTranslate.config import ConfigManager
from allTranslate.translator import (
    AnythingLLMTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DifyTranslator,
    ArgosTranslator,
    GeminiTranslator,
    GoogleTranslator,
    MiniMaxTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAITranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
    GrokTranslator,
    GroqTranslator,
    DeepseekTranslator,
    OpenAIlikedTranslator,
    QwenMtTranslator,
    X302AITranslator,
)
from babeldoc.docvision.doclayout import OnnxModel
from babeldoc import __version__ as babeldoc_version

logger = logging.getLogger(__name__)


class _LazyModel:
    """Defers model loading until first access so the GUI starts instantly."""

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model = OnnxModel.load_available()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self._ensure_loaded()
        return getattr(self._model, name)


BABELDOC_MODEL = _LazyModel()
# The following variables associate strings with translators
service_map: dict[str, BaseTranslator] = {
    "Google": GoogleTranslator,
    "Bing": BingTranslator,
    "DeepL": DeepLTranslator,
    "DeepLX": DeepLXTranslator,
    "Ollama": OllamaTranslator,
    "Xinference": XinferenceTranslator,
    "AzureOpenAI": AzureOpenAITranslator,
    "OpenAI": OpenAITranslator,
    "Zhipu": ZhipuTranslator,
    "ModelScope": ModelScopeTranslator,
    "Silicon": SiliconTranslator,
    "Gemini": GeminiTranslator,
    "Azure": AzureTranslator,
    "Tencent": TencentTranslator,
    "Dify": DifyTranslator,
    "AnythingLLM": AnythingLLMTranslator,
    "Argos Translate": ArgosTranslator,
    "Grok": GrokTranslator,
    "Groq": GroqTranslator,
    "DeepSeek": DeepseekTranslator,
    "MiniMax": MiniMaxTranslator,
    "OpenAI-liked": OpenAIlikedTranslator,
    "Ali Qwen-Translation": QwenMtTranslator,
    "302.AI": X302AITranslator,
}

# The following variables associate strings with specific languages
lang_map = {
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
}

# The following variable associate strings with page ranges
page_map = {
    "All": None,
    "First": [0],
    "First 5 pages": list(range(0, 5)),
    "Others": None,
}

# Check if this is a public demo, which has resource limits
flag_demo = False

# Limit resources
if ConfigManager.get("ALLTRANSLATE_DEMO"):
    flag_demo = True
    service_map = {
        "Google": GoogleTranslator,
    }
    page_map = {
        "First": [0],
        "First 20 pages": list(range(0, 20)),
    }
    client_key = ConfigManager.get("ALLTRANSLATE_CLIENT_KEY")
    server_key = ConfigManager.get("ALLTRANSLATE_SERVER_KEY")


# Limit Enabled Services
enabled_services: T.Optional[T.List[str]] = ConfigManager.get("ENABLED_SERVICES")
if isinstance(enabled_services, list):
    enabled_services_names = [str(_).lower().strip() for _ in enabled_services]
    enabled_services = [
        k
        for k in service_map.keys()
        if str(k).lower().strip() in enabled_services_names
    ]
    if len(enabled_services) == 0:
        raise RuntimeError("No services available.")
    # Always include Google and Bing
    for svc in ["Google", "Bing"]:
        if svc not in enabled_services:
            enabled_services.append(svc)
    # Reorder: configured services first, then Google and Bing
    configured = [k for k in enabled_services if str(k).lower().strip() in enabled_services_names]
    others = [k for k in enabled_services if k not in configured]
    enabled_services = configured + others
else:
    enabled_services = list(service_map.keys())


# Configure about Gradio show keys
hidden_gradio_details: bool = bool(ConfigManager.get("HIDDEN_GRADIO_DETAILS"))


# Public demo control
def verify_recaptcha(response):
    """
    This function verifies the reCAPTCHA response.
    """
    recaptcha_url = "https://www.google.com/recaptcha/api/siteverify"
    data = {"secret": server_key, "response": response}
    result = requests.post(recaptcha_url, data=data).json()
    return result.get("success")


def download_with_limit(url: str, save_path: str, size_limit: int) -> str:
    """
    This function downloads a file from a URL and saves it to a specified path.

    Inputs:
        - url: The URL to download the file from
        - save_path: The path to save the file to
        - size_limit: The maximum size of the file to download

    Returns:
        - The path of the downloaded file
    """
    chunk_size = 1024
    total_size = 0
    with requests.get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        content = response.headers.get("Content-Disposition")
        try:  # filename from header
            _, params = cgi.parse_header(content)
            filename = params["filename"]
        except Exception:  # filename from url
            filename = os.path.basename(url)
        filename = os.path.splitext(os.path.basename(filename))[0] + ".pdf"
        with open(save_path / filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                total_size += len(chunk)
                if size_limit and total_size > size_limit:
                    raise gr.Error("Exceeds file size limit")
                file.write(chunk)
    return save_path / filename


def stop_translate_file(state: dict) -> None:
    """
    This function stops the translation process.

    Inputs:
        - state: The state of the translation process

    Returns:- None
    """
    session_id = state["session_id"]
    if session_id is None:
        return
    if session_id in cancellation_event_map:
        logger.info(f"Stopping translation for session {session_id}")
        cancellation_event_map[session_id].set()
        # 清理取消事件，允许下一次翻译
        del cancellation_event_map[session_id]
        state["session_id"] = None


def translate_file(
    file_type,
    file_input,
    link_input,
    service,
    lang_from,
    lang_to,
    page_range,
    page_input,
    prompt,
    threads,
    skip_subset_fonts,
    ignore_cache,
    vfont,
    mode_choice,
    skip_classes,
    recaptcha_response,
    state,
    progress=gr.Progress(),
    *envs,
):
    """
    This function translates a PDF file from one language to another.

    Inputs:
        - file_type: The type of file to translate
        - file_input: The file to translate
        - link_input: The link to the file to translate
        - service: The translation service to use
        - lang_from: The language to translate from
        - lang_to: The language to translate to
        - page_range: The range of pages to translate
        - page_input: The input for the page range
        - prompt: The custom prompt for the llm
        - threads: The number of threads to use
        - recaptcha_response: The reCAPTCHA response
        - state: The state of the translation process
        - progress: The progress bar
        - envs: The environment variables

    Returns:
        - The translated file
        - The translated file
        - The translated file
        - The progress bar
        - The progress bar
        - The progress bar
    """
    session_id = uuid.uuid4()
    state["session_id"] = session_id
    cancellation_event_map[session_id] = asyncio.Event()
    # Translate PDF content using selected service.
    if flag_demo and not verify_recaptcha(recaptcha_response):
        raise gr.Error("reCAPTCHA fail")

    progress(0, desc="Starting translation...")

    output = Path("ALLTRANSLATE_files")
    output.mkdir(parents=True, exist_ok=True)

    if file_type == "File":
        if not file_input:
            raise gr.Error("No input")
        file_path = shutil.copy(file_input, output)
    else:
        if not link_input:
            raise gr.Error("No input")
        file_path = download_with_limit(
            link_input,
            output,
            5 * 1024 * 1024 if flag_demo else None,
        )

    filename = os.path.splitext(os.path.basename(file_path))[0]
    file_raw = output / f"{filename}.pdf"
    file_mono = output / f"{filename}-mono.pdf"
    file_dual = output / f"{filename}-dual.pdf"

    translator = service_map[service]
    if page_range != "Others":
        selected_page = page_map[page_range]
    else:
        selected_page = []
        for p in page_input.split(","):
            if "-" in p:
                start, end = p.split("-")
                selected_page.extend(range(int(start) - 1, int(end)))
            else:
                selected_page.append(int(p) - 1)
    lang_from = lang_map[lang_from]
    lang_to = lang_map[lang_to]

    _envs = {}
    for i, env in enumerate(translator.envs.items()):
        # envs[i] 是 gr.update 对象，需要提取其中的 value
        _envs[env[0]] = envs[i].value if hasattr(envs[i], 'value') else envs[i]
    for k, v in _envs.items():
        if str(k).upper().endswith("API_KEY") and str(v) == "***":
            # Load Real API_KEYs from local configure file (read-only)
            translator_config = ConfigManager.get_translator_by_name(translator.name)
            if translator_config and k in translator_config and translator_config[k]:
                _envs[k] = translator_config[k]
            # else: keep as None or empty, will cause auth error but won't modify config

    print(f"Files before translation: {os.listdir(output)}")

    def progress_bar(t: tqdm.tqdm):
        desc = getattr(t, "desc", "Translating...")
        if desc == "":
            desc = "Translating..."
        progress(t.n / t.total, desc=desc)

    try:
        threads = int(threads)
    except ValueError:
        threads = 1

    param = {
        "files": [str(file_raw)],
        "pages": selected_page,
        "lang_in": lang_from,
        "lang_out": lang_to,
        "service": f"{translator.name}",
        "output": output,
        "thread": int(threads),
        "callback": progress_bar,
        "cancellation_event": cancellation_event_map[session_id],
        "envs": _envs,
        "prompt": Template(prompt) if prompt else None,
        "skip_subset_fonts": skip_subset_fonts,
        "ignore_cache": ignore_cache,
        "vfont": vfont,  # 添加自定义公式字体正则表达式
        "model": ModelInstance.value,
    }

    try:
        from allTranslate.kernel import KernelRegistry
        from allTranslate.kernel.protocol import TranslateRequest

        KernelRegistry.switch(mode_choice)
        kernel = KernelRegistry.get()
        request = TranslateRequest(
            files=[str(file_raw)],
            output=str(output),
            pages=selected_page,
            lang_in=lang_from,
            lang_out=lang_to,
            service=f"{translator.name}",
            thread=int(threads),
            envs=_envs,
            prompt=str(prompt) if prompt else None,
            skip_subset_fonts=skip_subset_fonts,
            ignore_cache=ignore_cache,
            vfont=vfont,
            skip_classes=skip_classes,
        )
        kernel.translate(
            request,
            callback=progress_bar,
            cancellation_event=cancellation_event_map[session_id],
        )
    except CancelledError:
        del cancellation_event_map[session_id]
        raise gr.Error("Translation cancelled")
    print(f"Files after translation: {os.listdir(output)}")

    if not file_mono.exists() or not file_dual.exists():
        raise gr.Error("No output")

    progress(1.0, desc="Translation complete!")

    return (
        str(file_mono),
        str(file_mono),
        str(file_dual),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


def babeldoc_translate_file(**kwargs):
    from babeldoc.high_level import init as babeldoc_init

    babeldoc_init()
    from babeldoc.high_level import async_translate as babeldoc_translate
    from babeldoc.translation_config import TranslationConfig as YadtConfig

    for translator in [
        GoogleTranslator,
        BingTranslator,
        DeepLTranslator,
        DeepLXTranslator,
        OllamaTranslator,
        XinferenceTranslator,
        AzureOpenAITranslator,
        OpenAITranslator,
        ZhipuTranslator,
        ModelScopeTranslator,
        SiliconTranslator,
        GeminiTranslator,
        AzureTranslator,
        TencentTranslator,
        DifyTranslator,
        AnythingLLMTranslator,
        ArgosTranslator,
        GrokTranslator,
        GroqTranslator,
        DeepseekTranslator,
        OpenAIlikedTranslator,
        QwenMtTranslator,
        X302AITranslator,
    ]:
        if kwargs["service"] == translator.name:
            translator = translator(
                kwargs["lang_in"],
                kwargs["lang_out"],
                "",
                envs=kwargs["envs"],
                prompt=kwargs["prompt"],
                ignore_cache=kwargs["ignore_cache"],
            )
            break
    else:
        raise ValueError("Unsupported translation service")
    import asyncio
    from babeldoc.main import create_progress_handler

    for file in kwargs["files"]:
        file = file.strip("\"'")
        yadt_config = YadtConfig(
            input_file=file,
            font=None,
            pages=",".join((str(x) for x in getattr(kwargs, "raw_pages", []))),
            output_dir=kwargs["output"],
            doc_layout_model=BABELDOC_MODEL,
            translator=translator,
            debug=False,
            lang_in=kwargs["lang_in"],
            lang_out=kwargs["lang_out"],
            no_dual=False,
            no_mono=False,
            qps=kwargs["thread"],
            use_rich_pbar=False,
            disable_rich_text_translate=not isinstance(translator, OpenAITranslator),
            skip_clean=kwargs["skip_subset_fonts"],
            report_interval=0.5,
        )

        async def yadt_translate_coro(yadt_config):
            progress_context, progress_handler = create_progress_handler(yadt_config)

            # 开始翻译
            with progress_context:
                async for event in babeldoc_translate(yadt_config):
                    progress_handler(event)
                    if yadt_config.debug:
                        logger.debug(event)
                    kwargs["callback"](progress_context)
                    if kwargs["cancellation_event"].is_set():
                        yadt_config.cancel_translation()
                        raise CancelledError
                    if event["type"] == "finish":
                        result = event["translate_result"]
                        logger.info("Translation Result:")
                        logger.info(f"  Original PDF: {result.original_pdf_path}")
                        logger.info(f"  Time Cost: {result.total_seconds:.2f}s")
                        logger.info(f"  Mono PDF: {result.mono_pdf_path or 'None'}")
                        logger.info(f"  Dual PDF: {result.dual_pdf_path or 'None'}")
                        file_mono = result.mono_pdf_path
                        file_dual = result.dual_pdf_path
                        break
            import gc

            gc.collect()
            return (
                str(file_mono),
                str(file_mono),
                str(file_dual),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        return asyncio.run(yadt_translate_coro(yadt_config))


# Global setup
custom_blue = gr.themes.Color(
    c50="#E8F3FF",
    c100="#BEDAFF",
    c200="#94BFFF",
    c300="#6AA1FF",
    c400="#4080FF",
    c500="#165DFF",  # Primary color
    c600="#0E42D2",
    c700="#0A2BA6",
    c800="#061D79",
    c900="#03114D",
    c950="#020B33",
)

custom_css = """
    .secondary-text {color: #999 !important;}
    footer {visibility: hidden}
    .env-warning {color: #dd5500 !important;}
    .env-success {color: #559900 !important;}

    /* Add dashed border to input-file class */
    .input-file {
        border: 1.2px dashed #165DFF !important;
        border-radius: 6px !important;
    }

    .progress-bar-wrap {
        border-radius: 8px !important;
    }

    .progress-bar {
        border-radius: 8px !important;
    }

    .pdf-canvas canvas {
        width: 100%;
    }

    /* Logo image style - no border/background */
    .logo-image-no-border img {
        object-fit: contain;
        max-height: 40px !important;
        width: auto !important;
        margin: 0;
        padding: 0;
    }

    .logo-image-no-border {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Header row - left aligned */
    .logo-header {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        padding: 0;
        margin: 0;
    }
    """

demo_recaptcha = """
    <script src="https://www.google.com/recaptcha/api.js?render=explicit" async defer></script>
    <script type="text/javascript">
        var onVerify = function(token) {
            el=document.getElementById('verify').getElementsByTagName('textarea')[0];
            el.value=token;
            el.dispatchEvent(new Event('input'));
        };
    </script>
    """

tech_details_string = f"""
                    <summary>技术细节</summary>
                    - GitHub: <a href="https://github.com/Byaidu/allTranslate">Byaidu/allTranslate</a><br>
                    - BabelDOC: <a href="https://github.com/funstory-ai/BabelDOC">funstory-ai/BabelDOC</a><br>
                    - GUI作者: <a href="https://github.com/reycn">Rongxin</a><br>
                    - allTranslate版本: {__version__} <br>
                    - BabelDOC版本: {babeldoc_version}
                """
cancellation_event_map = {}


# The following code creates the GUI
with gr.Blocks(
    title="allTranslate - 保留格式的PDF翻译",
    theme=gr.themes.Default(
        primary_hue=custom_blue, spacing_size="md", radius_size="lg"
    ),
    css=custom_css,
    head=demo_recaptcha if flag_demo else "",
) as demo:
    with gr.Row(elem_classes=["logo-header"]):
        gr.Image(
            value="gaodeLOGO.svg",
            type="filepath",
            show_label=False,
            show_download_button=False,
            height=40,
            width=None,
            interactive=False,
            elem_classes=["logo-image-no-border"]
        )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 文件 | < 5 MB" if flag_demo else "## 文件")
            file_type = gr.Radio(
                choices=[("文件", "File"), ("链接", "Link")],
                label="类型",
                value="File",
            )
            file_input = gr.File(
                label="文件",
                file_count="single",
                file_types=[".pdf", ".doc", ".docx"],
                type="filepath",
                elem_classes=["input-file"],
            )
            link_input = gr.Textbox(
                label="链接",
                visible=False,
                interactive=True,
            )
            gr.Markdown("## 选项")
            service = gr.Dropdown(
                label="服务",
                choices=[("阿里千问翻译", "Ali Qwen-Translation")],
                value="Ali Qwen-Translation",
            )
            # 根据最大环境变量数量（OpenAI有6个）创建足够多的文本框
            envs = []
            for i in range(6):
                envs.append(
                    gr.Textbox(
                        visible=False,
                        interactive=True,
                    )
                )
            with gr.Row():
                lang_from = gr.Dropdown(
                    label="翻译来源",
                    choices=[("英语", "English"), ("简体中文", "Simplified Chinese"), ("繁体中文", "Traditional Chinese"),
                            ("法语", "French"), ("德语", "German"), ("日语", "Japanese"), ("韩语", "Korean"),
                            ("俄语", "Russian"), ("西班牙语", "Spanish"), ("意大利语", "Italian")],
                    value=ConfigManager.get("ALLTRANSLATE_LANG_FROM", "英语"),
                )
                lang_to = gr.Dropdown(
                    label="翻译目标",
                    choices=[("简体中文", "Simplified Chinese"), ("繁体中文", "Traditional Chinese"),
                            ("英语", "English"), ("法语", "French"), ("德语", "German"), ("日语", "Japanese"),
                            ("韩语", "Korean"), ("俄语", "Russian"), ("西班牙语", "Spanish"), ("意大利语", "Italian")],
                    value=ConfigManager.get("ALLTRANSLATE_LANG_TO", "简体中文"),
                )
            page_range = gr.Radio(
                choices=[("全部", "All"), ("第一页", "First"), ("前5页", "First 5 pages"), ("其他", "Others")],
                label="页面",
                value="All",
            )

            page_input = gr.Textbox(
                label="页面范围",
                visible=False,
                interactive=True,
            )

            # 内容类型选择：用户可以选择跳过哪些类型的元素不翻译
            skip_types = gr.CheckboxGroup(
                label="跳过以下元素翻译",
                choices=[
                    ("图", "figure"),
                    ("表", "table"),
                ],
                value=["figure", "table"],  # 默认跳过图和表，不翻译图像和表格
                interactive=True,
            )

            with gr.Accordion("展开更多实验选项！", open=False, visible=False):
                gr.Markdown("#### 实验性")
                threads = gr.Textbox(
                    label="线程数", interactive=True, value="4"
                )
                skip_subset_fonts = gr.Checkbox(
                    label="跳过字体子集化", interactive=True, value=False
                )
                ignore_cache = gr.Checkbox(
                    label="忽略缓存", interactive=True, value=False
                )
                vfont = gr.Textbox(
                    label="自定义公式字体正则表达式 (vfont)",
                    interactive=True,
                    value=ConfigManager.get("ALLTRANSLATE_VFONT", ""),
                )
                prompt = gr.Textbox(
                    label="LLM自定义提示", interactive=True, visible=False
                )
                mode_choice = gr.Dropdown(
                    label="翻译模式",
                    choices=[("快速", "fast"), ("精确", "precise")],
                    value="fast",
                    interactive=True,
                )

            def on_select_service(service, evt: gr.EventData = None):
                translator = service_map[service]
                _envs = []
                for i in range(6):
                    _envs.append(gr.update(visible=False, value=""))
                # 只从config.json读取，不写入
                translator_config = ConfigManager.get_translator_by_name(translator.name)
                for i, env in enumerate(translator.envs.items()):
                    label = env[0]
                    default_value = env[1]
                    # 只读取，不自动保存
                    if translator_config and label in translator_config and translator_config[label]:
                        value = translator_config[label]
                    else:
                        value = default_value
                    visible = True
                    if hidden_gradio_details:
                        # 判断当前索引是否在translator.envs范围内
                        env_items = list(translator.envs.items())
                        if i < len(env_items):
                            # 有对应的环境变量
                            label = env_items[i][0]
                            # 特殊处理：Ali Qwen-Translation 隐藏所有环境变量
                            if service == "Ali Qwen-Translation":
                                visible = False
                            else:
                                # 其他服务：隐藏有值的非MODEL参数
                                if value and "MODEL" not in str(label).upper():
                                    visible = False
                        else:
                            # 没有对应的环境变量，强制隐藏
                            visible = False
                        # Hidden Keys From Gradio
                        if "API_KEY" in label.upper():
                            value = "***"  # We use "***" Present Real API_KEY
                    _envs[i] = gr.update(
                        visible=visible,
                        label=label,
                        value=value,
                    )
                # 最后一个Textbox用于CustomPrompt，但只在有对应环境变量时才显示
                if 5 < len(list(translator.envs.items())):
                    _envs[-1] = gr.update(visible=translator.CustomPrompt)
                else:
                    _envs[-1] = gr.update(visible=False)
                return _envs

            def on_select_filetype(file_type):
                return (
                    gr.update(visible=file_type == "File"),
                    gr.update(visible=file_type == "Link"),
                )

            def on_select_page(choice):
                if choice == "Others":
                    return gr.update(visible=True)
                else:
                    return gr.update(visible=False)

            def on_vfont_change(value):
                return value

            output_title = gr.Markdown("## 已翻译", visible=False)
            output_file_mono = gr.File(
                label="下载单语翻译", visible=False
            )
            output_file_dual = gr.File(
                label="下载双语翻译", visible=False
            )
            recaptcha_response = gr.Textbox(
                label="reCAPTCHA响应", elem_id="verify", visible=False
            )
            recaptcha_box = gr.HTML('<div id="recaptcha-box"></div>')
            with gr.Row():
                translate_btn = gr.Button("翻译", variant="primary")
                cancellation_btn = gr.Button("取消", variant="secondary")
            gr.Markdown("""
### ⚠️ 使用说明：
- 默认不翻译图像和表格，如需翻译请取消勾选，表格翻译耗时较久，请耐心等待
- 如果程序运行时间过长或出现错误，请点击 "取消" 按钮，按 F5 刷新页面重新运行
- 如遇到 API 频率限制，系统会自动重试（最多5次，间隔递增）
- 翻译完成可预览和下载
""")
            page_range.select(on_select_page, page_range, page_input)
            service.change(
                on_select_service,
                service,
                envs,
            )
            demo.load(
                fn=lambda: on_select_service(service.value, None),
                inputs=None,
                outputs=envs,
            )
            vfont.change(on_vfont_change, inputs=vfont, outputs=None)
            file_type.select(
                on_select_filetype,
                file_type,
                [file_input, link_input],
                js=(
                    f"""
                    (a,b)=>{{
                        try{{
                            grecaptcha.render('recaptcha-box',{{
                                'sitekey':'{client_key}',
                                'callback':'onVerify'
                            }});
                        }}catch(error){{}}
                        return [a];
                    }}
                    """
                    if flag_demo
                    else ""
                ),
            )

        with gr.Column(scale=2):
            gr.Markdown("## 预览")
            preview = PDF(label="文档预览", visible=True, height=2000)

    # Event handlers
    file_input.upload(
        lambda x: x,
        inputs=file_input,
        outputs=preview,
        js=(
            f"""
            (a,b)=>{{
                try{{
                    grecaptcha.render('recaptcha-box',{{
                        'sitekey':'{client_key}',
                        'callback':'onVerify'
                    }});
                }}catch(error){{}}
                return [a];
            }}
            """
            if flag_demo
            else ""
        ),
    )

    state = gr.State({"session_id": None})

    translate_btn.click(
        translate_file,
        inputs=[
            file_type,
            file_input,
            link_input,
            service,
            lang_from,
            lang_to,
            page_range,
            page_input,
            prompt,
            threads,
            skip_subset_fonts,
            ignore_cache,
            vfont,
            mode_choice,
            skip_types,
            recaptcha_response,
            state,
            *envs,
        ],
        outputs=[
            output_file_mono,
            preview,
            output_file_dual,
            output_file_mono,
            output_file_dual,
            output_title,
        ],
    ).then(lambda: None, js="()=>{grecaptcha.reset()}" if flag_demo else "")

    cancellation_btn.click(
        stop_translate_file,
        inputs=[state],
    )


def parse_user_passwd(file_path: str) -> tuple:
    """
    Parse the user name and password from the file.

    Inputs:
        - file_path: The file path to read.
    Outputs:
        - tuple_list: The list of tuples of user name and password.
        - content: The content of the file
    """
    tuple_list = []
    content = ""
    if not file_path:
        return tuple_list, content
    if len(file_path) == 2:
        try:
            with open(file_path[1], "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path[1]}' not found.")
    try:
        with open(file_path[0], "r", encoding="utf-8") as file:
            tuple_list = [
                tuple(line.strip().split(",")) for line in file if line.strip()
            ]
    except FileNotFoundError:
        print(f"Error: File '{file_path[0]}' not found.")
    return tuple_list, content


def _has_ipv6() -> bool:
    """Check whether the system can bind an IPv6 socket."""
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.close()
        return True
    except OSError:
        return False


def setup_gui(
    share: bool = False, auth_file: list = ["", ""], server_port=7860
) -> None:
    """
    Setup the GUI with the given parameters.

    Inputs:
        - share: Whether to share the GUI.
        - auth_file: The file path to read the user name and password.

    Outputs:
        - None
    """
    user_list, html = parse_user_passwd(auth_file)

    auth_kwargs = {}
    if len(user_list) > 0:
        auth_kwargs = {"auth": user_list, "auth_message": html}

    if flag_demo:
        demo.launch(server_name="0.0.0.0", max_file_size="5mb", inbrowser=True)
        return

    # Try binding addresses in order: "0.0.0.0" for IPv4, fallback to loopback
    bind_addresses = ["0.0.0.0", "127.0.0.1"]

    for addr in bind_addresses:
        try:
            demo.launch(
                server_name=addr,
                debug=True,
                inbrowser=True,
                share=share,
                server_port=server_port,
                **auth_kwargs,
            )
            return
        except Exception:
            print(
                f"Error launching GUI using {addr}.\n"
                "This may be caused by global mode of proxy software."
            )

    # Last resort: let Gradio create a share link
    demo.launch(
        debug=True,
        inbrowser=True,
        share=True,
        server_port=server_port,
        **auth_kwargs,
    )


# For auto-reloading while developing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setup_gui()
