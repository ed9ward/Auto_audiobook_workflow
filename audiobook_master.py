#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
АУДИОКНИГА: озвучка книги через Qwen3-TTS (локально, GPU) или Gemini TTS (облако).

Положи рядом со скриптом книгу (1.txt) и запусти:   python audiobook_master.py
Дальше всё через меню: скрипт сам создаст окружение .venv, покажет найденные главы,
предложит докачать нужные пакеты и начнёт озвучку.

Облако (Gemini) работает по API-ключу из Google AI Studio: ключ вставляется в меню «Настройки»
(там же модель и голос) и хранится в settings.json рядом со скриптом. Больше ничего настраивать не нужно.

Всё живёт в ЭТОЙ папке и нигде больше:
    .venv/             окружение Python (все пакеты только сюда, системный Python не трогается)
    models/            скачанные модели
    .cache/            временные файлы и кэши (кэш pip удаляется сам после установки)
    <Название книги>/  всё по одной книге: готовые MP3 глав, chapters/ (тексты глав)
                       и temp_chunks/ (куски озвучки для продолжения после сбоя)

Если скрипт упал или ты его остановил: просто запусти снова — озвучатся только недостающие
куски, а в конце глава склеится в один MP3 сама.
"""
import argparse
import base64
import copy
import gc
import hashlib
import html
import importlib.util
import io
import json
import os
import platform
import posixpath
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, fields, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from functools import lru_cache
from html.parser import HTMLParser
from importlib import metadata
from pathlib import Path
from xml.etree import ElementTree

try:                    # при первом запуске в свежем .venv их ещё нет: startup() доустановит их и импортирует
    import numpy as np
    import soundfile as sf
except ImportError:
    np = sf = None

# ============================================================
# НАСТРОЙКИ — меняй здесь
# ============================================================

DEFAULT_ENGINE = "qwen"            # qwen | gemini
INPUT_BOOK_FILENAME = "1.txt"      # книга по умолчанию (лежит рядом со скриптом)

# --- Голос для Qwen (клонирование) ---
REF_AUDIO_FILENAME = "ref.wav"     # образец голоса, лежит рядом со скриптом

# --- Gemini (облако, по API-ключу из Google AI Studio) ---
# Ключ, модель, голос, размер куска и тариф меняются в меню «Настройки» и хранятся в settings.json;
# значения по умолчанию описаны в классе Settings.
GEMINI_MODELS = [
    ("gemini-3.8-flash-tts", "максимум качества, Google рекомендует для аудиокниг"),
    ("gemini-3.8-flash-lite-tts", "быстрее и дешевле, для больших объёмов"),
    ("gemini-3.1-flash-tts-preview", "старая версия, Google заменяет её на 3.8"),
    ("gemini-2.5-pro-preview-tts", "старая preview-версия"),
    ("gemini-2.5-flash-preview-tts", "старая preview-версия"),
]
GEMINI_MODEL_IDS = [name for name, _ in GEMINI_MODELS]
GEMINI_VOICES = [
    ("Zephyr", "Bright"), ("Puck", "Upbeat"), ("Charon", "Informative"), ("Kore", "Firm"),
    ("Fenrir", "Excitable"), ("Leda", "Youthful"), ("Orus", "Firm"), ("Aoede", "Breezy"),
    ("Callirrhoe", "Easy-going"), ("Autonoe", "Bright"), ("Enceladus", "Breathy"), ("Iapetus", "Clear"),
    ("Umbriel", "Easy-going"), ("Algieba", "Smooth"), ("Despina", "Smooth"), ("Erinome", "Clear"),
    ("Algenib", "Gravelly"), ("Rasalgethi", "Informative"), ("Laomedeia", "Upbeat"), ("Achernar", "Soft"),
    ("Alnilam", "Firm"), ("Schedar", "Even"), ("Gacrux", "Mature"), ("Pulcherrima", "Forward"),
    ("Achird", "Friendly"), ("Zubenelgenubi", "Casual"), ("Vindemiatrix", "Gentle"), ("Sadachbia", "Lively"),
    ("Sadaltager", "Knowledgeable"), ("Sulafat", "Warm"),
]

# --- Нарезка: размер куска текста, пауза между кусками (сек), битрейт MP3 ---
ENGINE_SETTINGS = {
    "qwen":   {"max_chars": 1800, "pause": 0.4, "bitrate": "128k"},
    "gemini": {"max_chars": 1200, "pause": 0.0, "bitrate": "192k"},
    "dummy":  {"max_chars": 1800, "pause": 0.4, "bitrate": "64k"},   # тест без голоса
}
SPLIT_PART_SIZE = 30000            # для разбивки «равными частями» (символов)
MAX_SPLIT_SIZE = 100_000_000       # больше — это уже не размер части, а ошибка в данных
SAMPLE_RATE = 24000

# --- Qwen ---
QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
QWEN_LANGUAGES = ["Russian", "English", "Chinese", "Japanese", "Korean", "German", "French", "Portuguese",
                  "Spanish", "Italian"]
QWEN_MAX_SEQ_LEN = 4096
QWEN_PATCH_RANGE = (0, 24)

# Версии для установки в .venv — ровно те, что у тебя уже работают
REQUIRED_PYTHON = (3, 12)                      # минимальная версия
PREFERRED_PYTHONS = ((3, 14), (3, 13), (3, 12))  # какие версии берём для .venv, по порядку; готовые .venv на 3.12 и 3.13 работают
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
QWEN_TORCH_PKGS = ["torch==2.11.0+cu128", "torchaudio==2.11.0+cu128"]
LEGACY_TORCH_INDEX = "https://download.pytorch.org/whl/cu126"         # сборка для видеокарт старше Turing
LEGACY_TORCH_PKGS = ["torch==2.11.0+cu126", "torchaudio==2.11.0+cu126"]
MODERN_GPU = (7, 5)              # Turing и новее: сборка cu128 и ускорение Triton; старше — сборка cu126 без Triton
GPU_GENERATIONS = [((10, 0), "Blackwell"), ((9, 0), "Hopper"), ((8, 9), "Ada Lovelace"), ((8, 0), "Ampere"),
                   ((7, 5), "Turing"), ((7, 0), "Volta"), ((6, 0), "Pascal"), ((5, 0), "Maxwell")]
# numpy и soundfile скрипт использует сам, поэтому они ставятся сразу нужной версии: Windows не даёт заменить
# библиотеку, пока её держит запущенный скрипт.
REQUESTS_PIN = "requests==2.34.2"
IMAGEIO_FFMPEG_PIN = "imageio-ffmpeg==0.6.0"
NVML_PIN = "nvidia-ml-py==13.615.71"
GUI_PACKAGES = (("tkinterdnd2", "tkinterdnd2==0.6.4"), ("sv_ttk", "sv-ttk==2.6.1"))      # (модуль, что ставить)
NUMPY_PIN = "numpy==2.5.3"
SOUNDFILE_PIN = "soundfile==0.14.0"
QWEN_CORE_PKGS = [
    "faster-qwen3-tts==0.5.3", "qwen-tts-hf==0.1.1.post1", "transformers==5.15.1",
    "huggingface_hub==1.33.0", "safetensors==0.8.0", NUMPY_PIN, SOUNDFILE_PIN,
    "librosa==1.0.0", "scipy==1.18.1", "onnxruntime==1.30.0",
]
QWEN_ACCELERATE = "accelerate==1.15.0"           # qwen-tts-hf просит 1.12.0, но у тебя работает 1.15.0
QWEN_TRITON_WIN = "triton-windows==3.8.0.post28"
QWEN_TRITON_PKG = "qwen3-tts-triton==0.3.0"      # ставится с --no-deps (его зависимости конфликтуют)

# Gemini
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
TEST_PHRASE = "Это проба голоса. Так будет звучать ваша книга."
GEMINI_KEY_URL = "https://aistudio.google.com/apikey"
GEMINI_RETRIES = 5
GEMINI_RETRY_DELAY = 10      # базовая пауза между повторами (сек)
GEMINI_FREE_INTERVAL = 20    # бесплатный тариф: не чаще запроса в 20 с (лимит модели около 3 запросов в минуту)
GEMINI_PAID_INTERVAL = 0.5   # платный тариф: пауза между началами запросов (сек)
GEMINI_PARALLEL_PAID = 3     # платный тариф: столько кусков озвучивается одновременно
QWEN_CHUNK_LIMITS = (600, 3000)             # допустимый размер куска Qwen (символов)
GEMINI_CHUNK_LIMITS = (300, 8000, 6000)   # размер куска: минимум, максимум, предел «безопасного» (символов)

# --- Расчёт расходов Gemini ---
# Цены в $ за 1 млн токенов (вход, выход) взяты из открытых источников и могут меняться: сверяй со страницей цен.
GEMINI_PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
GEMINI_PRICES = {      # модель: [(действует с даты, вход, выход), ...]
    "gemini-3.8-flash-tts": [(date(2026, 1, 1), 0.50, 9.00), (date(2027, 1, 1), 1.00, 18.00)],
    "gemini-3.8-flash-lite-tts": [(date(2026, 1, 1), 0.50, 6.00), (date(2027, 1, 1), 1.00, 12.00)],
    "gemini-3.1-flash-tts-preview": [(date(2026, 1, 1), 1.00, 20.00)],
}
AUDIO_TOKENS_PER_SEC = 25     # сколько токенов Google считает на секунду звука
SPEECH_CHARS_PER_SEC = 15     # сколько символов русского текста в среднем звучит за секунду
TEXT_CHARS_PER_TOKEN = 3      # грубая оценка входных токенов

# ============================================================
# ПАПКИ И ОКРУЖЕНИЕ (только стандартная библиотека)
# ============================================================


LANGUAGES = ("ru", "en")
_language = "ru"                 # текущий язык интерфейса; тексты в коде написаны по-русски, перевод ищется в словаре EN


def system_language():
    """Язык системы: русский для русскоязычной Windows (или другой системы), иначе английский."""
    try:
        if os.name == "nt":
            import ctypes
            return "ru" if ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF == 0x19 else "en"
        import locale
        name = locale.getlocale()[0] or os.environ.get("LANG", "")
        return "ru" if name.lower().startswith("ru") else "en"
    except Exception:
        return "en"


def resolve_language(choice):
    """Язык по настройке: 'ru' или 'en'; всё остальное (auto) следует за системой. Переменная AUDIOBOOK_LANG главнее
    (нужна тестам и тем, кто хочет зафиксировать язык)."""
    forced = os.environ.get("AUDIOBOOK_LANG")
    if forced in LANGUAGES:
        return forced
    return choice if choice in LANGUAGES else system_language()


def set_language(language):
    global _language
    _language = language if language in LANGUAGES else "ru"


def tr(text):
    """Текст на языке интерфейса: русский как в коде или английский из словаря EN (если перевода нет, остаётся русский)."""
    return EN.get(text, text) if _language == "en" else text


class Paths:
    """Все папки и файлы проекта считаются от одной корневой папки (переменная AUDIOBOOK_HOME нужна тестам)."""

    def __init__(self, base):
        self.base = Path(base).resolve()
        self.venv = self.base / ".venv"
        self.models = self.base / "models"
        self.cache = self.base / ".cache"
        self.scratch = self.cache / "tmp"            # временные файлы вместо системной папки TEMP
        self.settings = self.base / "settings.json"
        self.usage = self.base / "usage.json"        # журнал запросов, которые сделал этот скрипт


PATHS = Paths(os.environ.get("AUDIOBOOK_HOME") or Path(__file__).resolve().parent)


def project_environment(paths):
    """Всё, что скачивают и кэшируют pip, HuggingFace, Torch, Triton и CUDA, уходит в папку проекта, а не в профиль."""
    hf = paths.models / "huggingface"
    return {
        "PIP_REQUIRE_VIRTUALENV": "1",          # pip физически не поставит ничего вне .venv
        "PIP_CACHE_DIR": str(paths.cache / "pip"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "HF_HOME": str(hf),
        # более конкретные переменные HuggingFace главнее HF_HOME, поэтому задаём и их: пользовательские значения
        # не должны уводить модели из папки проекта
        "HF_HUB_CACHE": str(hf / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(hf / "hub"),
        "TRANSFORMERS_CACHE": str(hf / "hub"),
        "HF_ASSETS_CACHE": str(hf / "assets"),
        "HF_XET_CACHE": str(hf / "xet"),
        "HF_DATASETS_CACHE": str(hf / "datasets"),
        "TORCH_HOME": str(paths.models / "torch"),
        "TRITON_HOME": str(paths.cache / "triton_home"),
        "TRITON_CACHE_DIR": str(paths.cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(paths.cache / "inductor"),
        "TORCH_EXTENSIONS_DIR": str(paths.cache / "torch_extensions"),
        "CUDA_CACHE_PATH": str(paths.cache / "cuda"),
        "XDG_CACHE_HOME": str(paths.cache / "xdg"),
        "MPLCONFIGDIR": str(paths.cache / "matplotlib"),
        "TMP": str(paths.scratch),
        "TEMP": str(paths.scratch),
        "TMPDIR": str(paths.scratch),
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",                 # библиотека Hugging Face не шлёт даже обезличенную статистику
        "PYTHONIOENCODING": "utf-8",
    }


def configure_environment():
    os.environ.update(project_environment(PATHS))
    PATHS.scratch.mkdir(parents=True, exist_ok=True)


def log(msg=""):
    print(msg, flush=True)


def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        raise SystemExit(0) from None


def ask_yes(question, default=True):
    ans = ask(f"{question} ({tr("Д/н") if default else tr("д/Н")}): ").lower()
    return default if not ans else ans in ("y", "yes", "д", "да", "1")


WINDOWLESS = False                       # запуск без консоли (из FoxyAudioBookMaker.exe): запоминается до подмены вывода ниже


def ensure_output_streams():
    """Без консоли sys.stdout и sys.stderr пустые (None), и библиотеки, которые пишут туда (например, полоса загрузки
    модели Qwen), падают. Тогда их вывод уходит в «пустоту», а фатальные ошибки показываются окном."""
    global WINDOWLESS
    if sys.stdout is None or sys.stderr is None or sys.stdin is None:
        WINDOWLESS = True
    sink = None
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            sink = sink or open(os.devnull, "w", encoding="utf-8")
            setattr(sys, name, sink)


ensure_output_streams()


def windowless():
    """Запуск без консоли (из FoxyAudioBookMaker.exe): print и input здесь никуда не ведут."""
    return WINDOWLESS or sys.stdout is None or sys.stdin is None


def show_fatal(text):
    """Последнее сообщение, когда консоли нет: обычное окно Windows с ошибкой."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, str(text)[-3000:], "Foxy AudioBook Maker", 0x10)
    except Exception:
        pass


def fail(msg):
    """Сообщение об ошибке и пауза, чтобы окно не закрылось при запуске двойным щелчком."""
    if windowless():
        show_fatal(msg.strip())
        sys.exit(1)
    log(msg)
    try:
        input(tr("\nНажми Enter, чтобы закрыть..."))
    except EOFError:
        pass
    sys.exit(1)


def in_project_venv():
    try:
        return Path(sys.prefix).resolve() == PATHS.venv.resolve()
    except OSError:
        return False


def venv_python():
    return PATHS.venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def find_base_python(windows=None):
    """Python для создания .venv: 3.14, а если его нет, 3.13 или 3.12 (на Windows через запуск py); иначе тот, из которого
    запущена программа, если он не старше 3.12."""
    if (os.name == "nt" if windows is None else windows) and shutil.which("py"):
        for version in PREFERRED_PYTHONS:
            r = subprocess.run(["py", "-%d.%d" % version, "-c", "import sys; print(sys.executable)"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
    if sys.version_info[:2] >= REQUIRED_PYTHON:
        return sys.executable
    return None


def folder_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def fmt_size(size):
    for unit in (tr("Б"), tr("КБ"), tr("МБ")):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return tr("{0:.1f} ГБ").format(size)


def purge_install_cache():
    """Удаляет кэш pip и временные файлы установки (они лежат только в папке проекта). Возвращает освобождённые байты."""
    freed = folder_size(PATHS.cache / "pip") + folder_size(PATHS.scratch)
    for folder in (PATHS.cache / "pip", PATHS.scratch):
        shutil.rmtree(folder, ignore_errors=True)
    PATHS.scratch.mkdir(parents=True, exist_ok=True)
    return freed


def pip_command(args):
    return [console_python(), "-m", "pip", *args]


def stream_process(command, events, token=None):
    """Запускает команду и строка за строкой отправляет её вывод в журнал. «Стоп» завершает процесс; True — успех."""
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in proc.stdout:
            if line.strip():
                events.log(line.rstrip())
            if token and token.wait(0):
                proc.terminate()
                break
    finally:
        proc.stdout.close()
        proc.wait()
    return proc.returncode == 0 and not (token and token.wait(0))


def pip(*args, events=None, token=None):
    """Единственное место, где что-то ставится. Работает только внутри .venv проекта.
    Без events вывод идёт прямо в консоль; с events (окно) — строками в журнал, и «Стоп» прерывает установку.
    После успешной установки кэш pip удаляется; после неудачной остаётся, чтобы повтор не качал всё заново."""
    if not in_project_venv():
        raise RuntimeError(tr("Отказ: установка пакетов разрешена только в .venv рядом со скриптом."))
    (events.log if events else log)("   $ pip " + " ".join(args))
    if events:
        ok = stream_process(pip_command(args), events, token)
    else:
        ok = subprocess.run(pip_command(args)).returncode == 0
    if ok and args[0] == "install":
        purge_install_cache()
        importlib.invalidate_caches()           # только что установленные пакеты сразу видны в этом же процессе
    return ok


def bootstrap():
    """Если запущены не из .venv проекта — создаём его (при необходимости) и перезапускаемся из него."""
    if in_project_venv():
        return
    vpy = venv_python()
    if not vpy.exists():
        setup_lock = acquire_lock(PATHS.base / ".venv-setup.lock", wait=180)      # две копии не создают окружение сразу
        if setup_lock is None:
            fail(tr("❌ Окружение .venv сейчас создаёт другая копия программы. Подождите и запустите снова."))
        try:
            if not vpy.exists():                                                  # другая копия могла успеть первой
                log(tr("Окружение .venv в папке {0} не найдено.").format(PATHS.base))
                base = find_base_python()
                if not base:
                    fail(tr("❌ Нужен Python %d.%d. Поставь его с python.org и запусти скрипт снова.") % REQUIRED_PYTHON)
                if not ask_yes(tr("Создать .venv здесь (Python: {0})?").format(base)):
                    sys.exit(0)
                subprocess.check_call([base, "-m", "venv", str(PATHS.venv)])
                log(tr("✅ Окружение создано."))
        finally:
            setup_lock.release()
    result = subprocess.run([str(vpy), str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.exit(result.returncode)


def pkg_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def remove_pip_leftovers():
    """pip называет «~имя» папки, которые не смог удалить (файл был занят). Когда скрипт стартует заново, они свободны."""
    import sysconfig
    for leftover in Path(sysconfig.get_paths()["purelib"]).glob("~*"):
        if leftover.is_dir():
            shutil.rmtree(leftover, ignore_errors=True)
        else:
            leftover.unlink(missing_ok=True)


def pin_parts(pin):
    """«имя==версия» → (имя, версия без локальной метки вроде +cu128)."""
    name, _, version = pin.partition("==")
    return name, version.split("+")[0]


def installed_older_than_pin(pin):
    """Версия, которая стоит в .venv, если пакет есть и старше закреплённой в этой версии программы; иначе None.
    Новее закреплённой не считается расхождением (пользователь мог обновить сам), нет пакета — отдельный случай."""
    name, wanted = pin_parts(pin)
    have = pkg_version(name)
    if have and version_tuple(have.split("+")[0]) < version_tuple(wanted):
        return have
    return None


def light_pins():
    """Лёгкие пакеты, которые программа обновляет сама при запуске. Тяжёлые (torch и остальной Qwen) сюда не входят."""
    return ([NUMPY_PIN, SOUNDFILE_PIN, NUM2WORDS_PIN, REQUESTS_PIN, IMAGEIO_FFMPEG_PIN, NVML_PIN]
            + [pkg for _, pkg in GUI_PACKAGES])


def sync_light_packages():
    """После обновления программы закреплённые версии могли смениться: лёгкие пакеты, которые стоят старее, ставятся
    заново (только они, а не всё окружение). Ошибка не мешает работе: остаётся то, что уже установлено."""
    for pin in light_pins():
        have = installed_older_than_pin(pin)
        if have:
            log(tr("📦 Обновляю библиотеку до версии этой программы: {0} (стояла {1})").format(pin, have))
            if not pip("install", pin):
                log(tr("ℹ️ Не удалось обновить {0}, работаю с установленной версией.").format(pin_parts(pin)[0]))


def ensure_base_packages():
    """numpy и soundfile нужны самому скрипту (на Windows ещё tzdata для часовых поясов): ставим и импортируем."""
    global np, sf
    packages = [("numpy", NUMPY_PIN), ("soundfile", SOUNDFILE_PIN)]
    if os.name == "nt":
        packages.append(("tzdata", "tzdata"))
    for module, package in packages:
        if not has_module(module):
            log(tr("📦 Ставлю базовый пакет в .venv: {0}").format(package))
            if not pip("install", package):
                fail(tr("❌ Не удалось установить ") + package)
    if not has_module("num2words"):                         # числа словами; без него числа читаются как есть
        log(tr("📦 Ставлю базовый пакет в .venv: {0}").format(NUM2WORDS_PIN))
        if not pip("install", NUM2WORDS_PIN):
            log(tr("ℹ️ Не удалось установить num2words: числа будут читаться цифрами, как есть."))
    sync_light_packages()                                   # до import: у загруженной библиотеки Windows не даёт заменить файлы
    import numpy
    import soundfile
    np, sf = numpy, soundfile


def startup():
    """Подготовка при запуске как программы. При импорте модуля (например, в тестах) ничего подобного не происходит."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_environment()
    set_language(resolve_language(RecordStore(PATHS.settings, Settings, only_changed=True).load().language))
    bootstrap()
    global APP_LOCK
    lock = acquire_app_lock(wait=5.0 if "--after-update" in sys.argv[1:] else 0.0)    # до чистки и установок; дубликат — сразу
    if lock is None:
        fail(tr("❌ Программа уже запущена из этой папки. Закройте другое окно или переключитесь на него."))
    APP_LOCK = lock
    remove_pip_leftovers()
    ensure_base_packages()


# ============================================================
# НАСТРОЙКИ, СОСТОЯНИЕ КНИГИ И ЖУРНАЛ РАСХОДОВ
# ============================================================


SCHEMA_VERSION = 1            # версия формата settings.json и split.json; растёт, когда формат меняется несовместимо


@dataclass
class Settings:
    """Настройки пользователя. Значения по умолчанию описаны здесь; в settings.json пишутся только изменённые."""
    ref_text: str = ""                   # старый текст образца для ref.wav: из него один раз создаётся первый голос
    qwen_voices: list = field(default_factory=list)   # голоса Qwen: [{"name", "path", "text"}]
    qwen_voice: str = ""                 # выбранный голос Qwen (по названию)
    qwen_language: str = ""              # язык озвучки Qwen; пусто — по языку интерфейса (русский или английский)
    gemini_api_key: str = ""             # лучше вводить в меню или в переменную окружения GEMINI_API_KEY
    gemini_model: str = "gemini-3.8-flash-tts"
    gemini_voice: str = "Kore"
    gemini_chunk_chars: int = 1200       # размер куска Gemini (символов)
    qwen_chunk_chars: int = 1800         # размер куска Qwen (символов): влияет на скорость и на качество интонации
    gemini_paid: bool = False            # платный тариф (включён биллинг) или бесплатный
    gemini_daily_limit: int = 0          # лимит запросов в сутки на модель для бесплатного тарифа (0 — не учитывать)
    normalize_text: bool = True          # читать числа и сокращения словами
    replacements: str = ""               # свои замены, по одной на строку: «что = как читать»
    theme: str = "auto"                  # оформление окна: auto (как в системе) | light | dark
    language: str = "auto"               # язык интерфейса: auto (как в системе) | ru | en
    qwen_language_auto: bool = True      # язык книги «Авто»: модели Qwen передаётся «Auto», она сама определяет язык
    qwen_no_triton_gpu: str = ""         # «видеокарта|версия», где Triton не заработал: до новой версии программы без него
    window_geometry: str = ""            # размер и положение окна при закрытии, например 860x820+120+80
    book_titles: dict = field(default_factory=dict)   # названия для файлов 1.txt: путь → {title, sha}

    def normalize(self):
        """Вложенные значения проверяются глубже, чем тип поля: негодные записи отбрасываются, а не роняют программу."""
        self.book_titles = {key: value for key, value in self.book_titles.items()
                            if isinstance(key, str) and isinstance(value, dict) and isinstance(value.get("title"), str)
                            and value["title"].strip() and isinstance(value.get("sha", ""), str)}
        self.qwen_voices = [voice for voice in self.qwen_voices
                            if isinstance(voice, dict) and all(isinstance(voice.get(k), str) for k in ("name", "path", "text"))]


@dataclass
class BookState:
    """Состояние одной книги (лежит в <книга>/chapters/split.json)."""
    profile: str = "auto"
    params: dict = field(default_factory=dict)
    confirmed: bool = False
    engine: str = ""
    voice: str = ""                      # голос Qwen, которым озвучивалась книга: при открытии выбирается он же
    language: str = ""                   # язык книги, выбранный вручную; пусто — определённый по тексту
    gemini_model: str = ""               # модель и голос Gemini этой книги; пусто — как в «Настройках»
    gemini_voice: str = ""
    author: str = ""                     # сведения о книге для тегов MP3 (из EPUB/FB2 или введённые вручную)
    series: str = ""
    series_index: str = ""
    description: str = ""
    cover: str = ""                      # своя обложка (путь к картинке); пусто — из книги; «-» — без обложки
    info_loaded: bool = False            # сведения из файла книги уже взяты (дальше правки пользователя главнее)
    cuts: list = field(default_factory=list)      # ручное разбиение: строки текста, с которых начинаются главы
    titles: dict = field(default_factory=dict)    # свои названия глав: {строка начала главы: название}
    cuts_digest: str = ""                # отпечаток текста, к которому относятся правки (текст изменился — правки сброшены)

    def normalize(self):
        """Допустимость значений проверяется глубже, чем тип поля (split.json могли править руками или создать более новой
        версией программы): негодное заменяется значением по умолчанию, в журнал идёт предупреждение."""
        fixed = []
        if self.profile != "auto" and self.profile not in PROFILE_SHORT and self.profile not in PROFILE_ALIASES:
            self.profile = "auto"
            fixed.append("profile")
        params = dict(self.params)
        if "size" in params:
            size = int(params["size"]) if isinstance(params["size"], str) and params["size"].isdigit() else params["size"]
            if isinstance(size, int) and not isinstance(size, bool) and 1 <= size <= MAX_SPLIT_SIZE:
                params["size"] = size
            else:
                del params["size"]
                fixed.append("params.size")
        if "regex" in params and not isinstance(params["regex"], str):
            del params["regex"]
            fixed.append("params.regex")
        self.params = params
        if self.language not in ("", *QWEN_LANGUAGES):
            self.language = ""
            fixed.append("language")
        if self.engine not in ("", *ENGINE_SETTINGS):
            self.engine = ""
            fixed.append("engine")
        if fixed:
            log(tr("⚠️ В сохранённом состоянии книги были недопустимые значения, они заменены стандартными: {0}.")
                .format(", ".join(fixed)))


SECRET_PREFIX = "dpapi:"
DPAPI_AVAILABLE = os.name == "nt"          # отдельная константа, чтобы тест мог имитировать Windows


class SecretProtectionError(Exception):
    """Секрет не удалось зашифровать: в файл его писать нельзя."""


def _dpapi(data, protect):
    """Шифрование средствами Windows (DPAPI): расшифровать может только тот же пользователь Windows на этом компьютере."""
    import ctypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_char))]

    buffer = ctypes.create_string_buffer(data, len(data))
    source, result = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), Blob()
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    function = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    if not function(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(result)):
        raise OSError("DPAPI")
    try:
        return ctypes.string_at(result.pbData, result.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(result.pbData, ctypes.c_void_p))


def protect_secret(text):
    """Секрет для записи в файл: на Windows зашифрованный (dpapi:...), на других системах как есть. Если Windows
    зашифровать не смогла — SecretProtectionError, открытым текстом секрет не записывается никогда."""
    if not DPAPI_AVAILABLE or not text:
        return text
    try:
        return SECRET_PREFIX + base64.b64encode(_dpapi(text.encode("utf-8"), True)).decode("ascii")
    except Exception as e:
        raise SecretProtectionError(str(e)) from e


def unprotect_secret(value):
    """Обратное к protect_secret. Если расшифровать нельзя (папку перенесли на другой компьютер), вернёт пустую строку:
    ключ придётся ввести заново."""
    if not value.startswith(SECRET_PREFIX):
        return value
    if not DPAPI_AVAILABLE:
        return ""
    try:
        return _dpapi(base64.b64decode(value[len(SECRET_PREFIX):]), False).decode("utf-8")
    except Exception:
        return ""


class RecordStore:
    """Хранилище одной записи-dataclass в JSON-файле (паттерн Repository). Чужие и повреждённые значения пропускаются.
    Поля из secrets на диске хранятся зашифрованными (на Windows)."""

    def __init__(self, path, cls, only_changed=False, secrets=()):
        self.path, self.cls, self.only_changed, self.secrets = Path(path), cls, only_changed, secrets
        self.secret_errors = []                                # секреты, которые при последнем save() не удалось зашифровать
        self.newer_version = 0                                 # версия формата файла, если он создан более новой программой

    def _keep_copy(self, suffix, message):
        target = self.path.with_name(self.path.name + suffix)
        try:
            shutil.copy2(self.path, target)
            log(message.format(self.path.name, target.name))
        except OSError:
            pass

    def load(self):
        record = self.cls()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("not an object")
        except FileNotFoundError:
            return record
        except Exception:                                      # повреждён: до перезаписи оставляем копию рядом
            self._keep_copy(".bad", tr("⚠️ Файл {0} повреждён и не прочитан, взяты значения по умолчанию. "
                                       "Копия сохранена как {1}."))
            return record
        version = data.get("schema_version", 1)
        if isinstance(version, int) and version > SCHEMA_VERSION:
            self.newer_version = version
        if isinstance(data, dict):
            for name in self.secrets:
                if isinstance(data.get(name), str):
                    data[name] = unprotect_secret(data[name])
            for f in fields(self.cls):
                if f.name in data and isinstance(data[f.name], type(getattr(record, f.name))):
                    setattr(record, f.name, data[f.name])
        if hasattr(record, "normalize"):
            record.normalize()
        return record

    def save(self, record):
        defaults = self.cls()
        data = {f.name: getattr(record, f.name) for f in fields(record)
                if not self.only_changed or getattr(record, f.name) != getattr(defaults, f.name)}
        self.secret_errors = []
        data = {"schema_version": SCHEMA_VERSION, **data}
        if self.newer_version:                                  # файл от более новой программы: сначала копия
            self._keep_copy(f".v{self.newer_version}", tr("ℹ️ Файл {0} создан более новой версией программы. "
                                                          "Прежняя копия сохранена как {1}."))
            self.newer_version = 0
        for name in self.secrets:
            if isinstance(data.get(name), str):
                try:
                    data[name] = protect_secret(data[name])
                except SecretProtectionError:
                    del data[name]                             # лучше потерять ключ, чем записать его открытым текстом
                    self.secret_errors.append(name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        if self.secrets and os.name != "nt":
            try:
                os.chmod(tmp, 0o600)                           # вне Windows файл с ключом читает только владелец
            except OSError:
                pass
        os.replace(tmp, self.path)


class UsageLog:
    """Журнал запросов к Gemini по датам (тихоокеанское время) и моделям."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()                          # запросы идут из нескольких потоков

    def load(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def record(self, model, in_tokens, out_tokens, seconds):
        with self._lock:
            data = self.load()
            row = data.setdefault(str(pacific_today()), {}).setdefault(
                model, {"requests": 0, "in_tokens": 0, "out_tokens": 0, "seconds": 0.0})
            row["requests"] += 1
            row["in_tokens"] += int(in_tokens)
            row["out_tokens"] += int(out_tokens)
            row["seconds"] = round(row["seconds"] + seconds, 1)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)

    def requests_today(self, model):
        return self.load().get(str(pacific_today()), {}).get(model, {}).get("requests", 0)

    def summarize(self, prefix=""):
        """Итоги по дням, начинающимся с prefix (например, «2026-11»); пустой prefix — за всё время."""
        total = {"requests": 0, "seconds": 0.0, "cost": 0.0, "unknown": False}
        for day, models in self.load().items():
            if not day.startswith(prefix):
                continue
            for model, row in models.items():
                total["requests"] += row["requests"]
                total["seconds"] += row["seconds"]
                cost = gemini_cost(model, row["in_tokens"], row["out_tokens"], date.fromisoformat(day))
                if cost is None:
                    total["unknown"] = True
                else:
                    total["cost"] += cost
        return total


class AppContext:
    """Всё, что нужно программе в работе: пути, настройки (с сохранением) и журнал расходов. Передаётся явно."""

    def __init__(self, paths=None):
        self.paths = paths or PATHS
        self.store = RecordStore(self.paths.settings, Settings, only_changed=True, secrets=("gemini_api_key",))
        self.settings = self.store.load()
        self.usage = UsageLog(self.paths.usage)
        set_language(resolve_language(self.settings.language))
        migrate_reference(self)

    def update(self, **changes):
        """Меняет настройки и сразу сохраняет их."""
        for key, value in changes.items():
            if not hasattr(self.settings, key):
                raise KeyError(tr("неизвестная настройка: {0}").format(key))
            setattr(self.settings, key, value)
        self.store.save(self.settings)
        if "gemini_api_key" in changes and self.store.secret_errors:
            log(tr("⚠️ Не удалось зашифровать API-ключ, поэтому он не записан в файл и действует только до закрытия "
                   "программы. Введи ключ снова или задай переменную окружения GEMINI_API_KEY."))


LANGUAGE_NAMES = {"Russian": "Русский", "English": "Английский", "Chinese": "Китайский", "Japanese": "Японский",
                  "Korean": "Корейский", "German": "Немецкий", "French": "Французский", "Portuguese": "Португальский",
                  "Spanish": "Испанский", "Italian": "Итальянский"}
LATIN_HINTS = {          # частые короткие слова: по ним различаем языки с латиницей
    "English": "the and of to in is that it was he for with as his on be at by you had not but".split(),
    "German": "der die und den von zu das mit sich des auf für ist im dem nicht ein eine als auch".split(),
    "French": "le la les et des un une du est que qui dans pour pas au sur ne se plus avec il elle".split(),
    "Spanish": "el los del que y en las por una con no es para lo al se su como más pero".split(),
    "Italian": "il di che la e un per non una con del le si da sono gli nel della alla anche".split(),
    "Portuguese": "de que o a e do da em um para com não uma os no se na por mais como foi ao".split(),
}


def detect_language(text):
    """Язык книги по тексту (название из QWEN_LANGUAGES): алфавит, а для латиницы — частые короткие слова. Смотрит
    только начало и середину текста, поэтому быстро и для большой книги. Пусто, если понять не удалось."""
    middle = len(text) // 2
    sample = text[:20000] + "\n" + text[middle:middle + 20000]
    count = {name: len(re.findall(pattern, sample)) for name, pattern in (
        ("cyrillic", "[а-яА-ЯёЁ]"), ("latin", "[a-zA-Z]"), ("kana", "[\u3040-\u30ff]"), ("hangul", "[\uac00-\ud7af]"),
        ("han", "[\u4e00-\u9fff]"))}
    if not any(count.values()):
        return ""
    if count["kana"] and count["kana"] >= 0.05 * (count["kana"] + count["han"]):
        return "Japanese"
    top = max(count, key=count.get)
    if top != "latin":
        return {"cyrillic": "Russian", "hangul": "Korean", "han": "Chinese", "kana": "Japanese"}[top]
    words = re.findall(r"[a-zà-ÿ]+", sample.lower())
    hints = {language: set(words_) for language, words_ in LATIN_HINTS.items()}
    scores = {language: sum(word in common for word in words) for language, common in hints.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "English"


def language_label(language):
    return tr(LANGUAGE_NAMES.get(language, language)) if language else tr("не определён")


def qwen_model_language(settings):
    """Язык для модели Qwen: «Auto», если у книги выбран «Авто» (в нашем списке не все языки, модель определит сама),
    иначе выбранный язык."""
    return "Auto" if settings.qwen_language_auto else effective_qwen_language(settings)


def effective_qwen_language(settings):
    """Язык озвучки Qwen: выбранный в настройках, а если не выбран — по языку интерфейса (английский интерфейс: книга
    скорее английская)."""
    return settings.qwen_language or ("English" if _language == "en" else "Russian")


def engine_options(settings, engine_name):
    """Размер куска, пауза и битрейт для движка; размер куска Gemini задаёт пользователь."""
    options = dict(ENGINE_SETTINGS[engine_name])
    if engine_name == "gemini":
        options["max_chars"] = settings.gemini_chunk_chars
    elif engine_name == "qwen":
        options["max_chars"] = settings.qwen_chunk_chars
    return options


# ============================================================
# СОБЫТИЯ И ОТМЕНА
# ============================================================


class Cancelled(Exception):
    """Озвучку остановил пользователь («Стоп» в окне или Ctrl+C)."""


class CancellationToken:
    """Сигнал «Стоп»: движки и сервис проверяют его между кусками, а ожидания (паузы Gemini) прерываются сразу."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    def wait(self, seconds):
        """Ждёт seconds секунд; True, если за это время нажали «Стоп»."""
        return self._event.wait(seconds)

    def cancelled(self):
        """Уже ли нажали «Стоп» (без ожидания)."""
        return self._event.is_set()

    def raise_if_cancelled(self):
        if self._event.is_set():
            raise Cancelled()


class EventBus:
    """Наблюдатель (Observer): ядро сообщает о ходе работы, консоль и окно подписываются на события.
    Типы событий: log, error, chapter_started, chunk_started, chunk_done, chapter_done, run_finished.
    У каждого события есть поле text (готовая строка для журнала) и, если нужно, данные для индикаторов."""

    def __init__(self):
        self._listeners = []

    def subscribe(self, listener):
        """listener(kind, data) вызывается для каждого события."""
        self._listeners.append(listener)

    def emit(self, kind, **data):
        for listener in list(self._listeners):
            listener(kind, data)

    def log(self, text):
        self.emit("log", text=text)


# ============================================================
# РАЗБИВКА КНИГИ НА ГЛАВЫ
# ============================================================

PROFILE_ORDER = ["auto", "headings", "markdown", "numbers", "custom", "size", "single"]
PROFILE_TITLES = {
    "auto": "Автоопределение (рекомендуется)",
    "headings": "Заголовки «Глава 5», «Глава пятая», «Chapter 3», «Пролог» и т. п. (разные языки)",
    "markdown": "Заголовки Markdown (# Название)",
    "numbers": "Только номера отдельными строками: «1», «II.», «3. Название»",
    "custom": "Своё регулярное выражение для строки-заголовка",
    "size": "Равные части по размеру (если в книге нет заголовков)",
    "single": "Весь файл как одна глава",
}
PROFILE_SHORT = {"headings": "по заголовкам", "markdown": "по заголовкам Markdown", "numbers": "по номерам",
                 "custom": "по своему правилу", "size": "равные части", "single": "одна глава", "manual": "вручную"}
MANUAL_TITLE = "Вручную (главы изменены)"
PROFILE_ALIASES = {"words": "headings"}        # старое имя способа из прежних версий
AUTO_ORDER = ["headings", "markdown", "numbers"]

HEADING_MAX_LEN = 100        # строка длиннее — точно не заголовок
PART_MERGE_CHARS = 800       # «Часть I» с коротким эпиграфом перед главой присоединяется к этой главе
PREFACE_MIN_CHARS = 300      # текст до первой главы: длиннее — отдельная глава «Начало», короче — в начало 1-й главы
TINY_SECTION_CHARS = 40      # «глава» короче — это строка оглавления, а не глава
AUTO_MIN_AVG_CHARS = 800     # автоопределение отвергает способ, при котором главы в среднем короче
NUMBERS_AUTO_LIMITS = (120, 3000)    # для «только номера»: не больше глав и не короче в среднем (иначе это номера страниц)

_CHAPTER_WORDS = ("глава|розділ|chapter|kapitel|chapitre|capítulo|capitulo|capitolo|rozdział|rozdzial|"
                  "kapitola|hoofdstuk")
_PART_WORDS = "часть|книга|том|раздел|part|book|volume|section|teil|partie|parte|część|czesc|deel"
_SPECIAL_WORDS = ("пролог|эпилог|предисловие|послесловие|введение|вступление|заключение|от автора|prologue|"
                  "epilogue|preface|foreword|afterword|introduction|conclusion|prolog|epilog|vorwort|nachwort|"
                  "préface|prefacio|prólogo|epílogo")
_SEP = r"[.:!?—–-]"
_WORD = r"[^\W\d_]{2,20}"


def _numbered(words):
    """«Глава 5», «Глава V. Название», «Глава пятая.», «Chapter Twenty-One: Title»."""
    digits = (rf"^(?:{words})\s+(?P<num>\d{{1,4}}|(?-i:[IVXLCDM]{{1,8}}))(?=[\s.:!?—–-]|$)"
              rf"\s*{_SEP}*\s*(?P<title>\S.*)?$")
    spelled = rf"^(?:{words})\s+(?P<num>{_WORD}(?:[ -]{_WORD}){{0,2}})\s*(?:{_SEP}+\s*(?P<title>\S.*)?)?$"
    return [re.compile(digits, re.IGNORECASE), re.compile(spelled, re.IGNORECASE)]


_SPECIAL_RE = re.compile(rf"^(?:{_SPECIAL_WORDS})\s*(?:{_SEP}+\s*(?P<title>\S.*)?)?$", re.IGNORECASE)
_CJK_RE = re.compile(r"^第\s*[〇零一二三四五六七八九十百千两0-9]+\s*[章节節回](?P<title>.*)$")
_HEADING_PATTERNS = [(2, _numbered(_CHAPTER_WORDS) + [_SPECIAL_RE, _CJK_RE]),   # глава
                     (1, _numbered(_PART_WORDS))]                               # часть, том, книга
_NUMBER_LINE_RE = re.compile(r"^(?P<n>\d{1,3}|(?-i:[IVXLCDM]{1,7}))[.)]?(?:\s+(?P<title>\S.{0,60}))?$", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def slugify(title, limit=50):
    s = re.sub(r"[^\w]+", "_", title, flags=re.UNICODE).strip("_")
    return s[:limit].rstrip("_")


def roman_to_int(text):
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    text = text.upper()
    return sum(-values[c] if i + 1 < len(text) and values[text[i + 1]] > values[c] else values[c]
               for i, c in enumerate(text))


def _match_heading(text):
    """(уровень, название из той же строки) или None. Уровень 2 — глава, 1 — часть/том."""
    for level, patterns in _HEADING_PATTERNS:
        for pattern in patterns:
            m = pattern.match(text)
            if m:
                return level, (m.groupdict().get("title") or "").strip()
    return None


def _title_after(lines, i):
    """Название главы, если оно стоит отдельной короткой строкой сразу после заголовка."""
    for line in lines[i + 1:i + 4]:
        text = line.strip()
        if text:
            return "" if len(text) > 80 or _match_heading(text) else text
    return ""


def find_headings(lines, strict=True):
    """Строки-заголовки. strict: заголовок обязан стоять после пустой строки."""
    found = []
    for i, line in enumerate(lines):
        text = line.strip()
        if not text or len(text) > HEADING_MAX_LEN or (strict and i > 0 and lines[i - 1].strip()):
            continue
        match = _match_heading(text)
        if match:
            found.append({"line": i, "level": match[0], "heading": text, "title": match[1]})
    return found


def _marks_headings(lines):
    """Границы глав по заголовкам. «Часть I» с коротким эпиграфом присоединяется к следующей главе."""
    for strict in (True, False):
        found = find_headings(lines, strict)
        chapters = [h for h in found if h["level"] == 2]
        parts = [h for h in found if h["level"] == 1]
        if len(chapters) >= 2:
            marks, dividers = chapters, parts
        elif len(parts) >= 2:
            marks, dividers = parts, []
        else:
            continue
        result, previous = [], -1
        for h in marks:
            start = h["line"]
            for d in dividers:
                if previous < d["line"] < h["line"] and sum(len(x) for x in lines[d["line"] + 1:h["line"]]) <= PART_MERGE_CHARS:
                    start = min(start, d["line"])
            result.append({"line": h["line"], "start": start, "heading": h["heading"],
                           "title": h["title"] or _title_after(lines, h["line"])})
            previous = h["line"]
        return result
    return []


def _marks_markdown(lines):
    found = [(i, len(m.group(1)), m.group(2)) for i, line in enumerate(lines) if (m := _MARKDOWN_RE.match(line))]
    levels = [lv for lv in sorted({lv for _, lv, _ in found}) if sum(1 for _, x, _ in found if x == lv) >= 2]
    if not levels:
        return []
    return [{"line": i, "start": i, "heading": lines[i].strip(), "title": text}
            for i, lv, text in found if lv == levels[0]]


def _marks_numbers(lines):
    """Строки из одного номера; берётся только ряд 1, 2, 3, … подряд (случайные числа вроде страниц пропускаются)."""
    marks, expected = [], 1
    for i, line in enumerate(lines):
        text = line.strip()
        if not text or len(text) > 70 or (i > 0 and lines[i - 1].strip()):
            continue
        m = _NUMBER_LINE_RE.match(text)
        if m:
            n = m.group("n")
            if (int(n) if n.isdigit() else roman_to_int(n)) == expected:
                marks.append({"line": i, "start": i, "heading": text,
                              "title": (m.group("title") or "").strip() or _title_after(lines, i)})
                expected += 1
    return marks if len(marks) >= 3 else []


def _marks_custom(lines, regex):
    if not regex.strip():
        return []
    try:
        pattern = re.compile(regex, re.IGNORECASE)
    except re.error:
        return []
    return [{"line": i, "start": i, "heading": line.strip(), "title": line.strip()}
            for i, line in enumerate(lines) if line.strip() and pattern.search(line.strip())]


def _drop_tiny(lines, marks):
    """Убирает «главы» из одной строки (оглавление): настоящая глава содержит текст.
    Возвращает (оставшиеся границы, строка, с которой начинается оглавление или None)."""
    kept = []
    for k, mark in enumerate(marks):
        end = marks[k + 1]["start"] if k + 1 < len(marks) else len(lines)
        if k + 1 == len(marks) or len("".join(lines[mark["line"] + 1:end]).strip()) >= TINY_SECTION_CHARS:
            kept.append(mark)
    toc_start = None
    leading = [m for m in marks if kept and m not in kept and m["line"] < kept[0]["start"]]
    if len(leading) >= 3:                         # три и больше «глав»-строк подряд в начале — это оглавление
        toc_start = leading[0]["line"]
        before = toc_start - 1
        while before >= 0 and not lines[before].strip():
            before -= 1
        if before >= 0 and len(lines[before].strip()) <= 30:      # слово «Оглавление» над списком
            toc_start = before
    return kept, toc_start


def _chapters_from_marks(lines, marks, toc_start=None):
    chapters = []
    for k, mark in enumerate(marks):
        end = marks[k + 1]["start"] if k + 1 < len(marks) else len(lines)
        chapters.append({"heading": mark["heading"], "title": mark["title"],
                         "text": "\n".join(lines[mark["start"]:end]).strip() + "\n"})
    front = "\n".join(lines[:marks[0]["start"] if toc_start is None else toc_start]).strip()   # до первой главы
    if front:
        if len(front) >= PREFACE_MIN_CHARS:
            chapters.insert(0, {"heading": "Начало", "title": "Начало", "text": front + "\n"})
        else:
            chapters[0]["text"] = front + "\n\n" + chapters[0]["text"]
    return chapters


def _split_size(text, size):
    pieces = split_text_into_chunks(text, size)
    return [{"heading": f"Часть {k}", "title": f"Часть {k}", "text": piece.strip() + "\n"}
            for k, piece in enumerate(pieces, 1)]


def split_book(text, profile="auto", params=None, book_title="Книга"):
    """Возвращает (главы, использованный способ разбивки)."""
    params = params or {}
    profile = PROFILE_ALIASES.get(profile, profile)
    lines = text.split("\n")
    single = [{"heading": book_title, "title": book_title, "text": text.strip() + "\n"}]

    def run(name):
        if name == "size":
            return _split_size(text, int(params.get("size", SPLIT_PART_SIZE)))
        finders = {"headings": _marks_headings, "markdown": _marks_markdown, "numbers": _marks_numbers,
                   "custom": lambda ls: _marks_custom(ls, params.get("regex", ""))}
        if name not in finders:
            return single
        marks, toc_start = _drop_tiny(lines, finders[name](lines))
        return _chapters_from_marks(lines, marks, toc_start) if marks else []

    if profile == "auto":
        for name in AUTO_ORDER:
            chapters = run(name)
            if len(chapters) >= 2 and len(text) / len(chapters) >= AUTO_MIN_AVG_CHARS and (
                    name != "numbers" or (len(chapters) <= NUMBERS_AUTO_LIMITS[0]
                                          and len(text) / len(chapters) >= NUMBERS_AUTO_LIMITS[1])):
                profile = name
                break
        else:
            profile = "size" if len(text) > 2 * SPLIT_PART_SIZE else "single"
            chapters = run(profile)
    else:
        chapters = run(profile)
    if not chapters:
        chapters, profile = single, "single"

    return number_chapters(chapters), profile


def number_chapters(chapters):
    """Номера и имена файлов глав по порядку."""
    width = max(2, len(str(len(chapters))))
    for k, ch in enumerate(chapters, 1):
        ch["id"] = k
        ch["name"] = f"{k:0{width}d}_{slugify(ch['title'] or ch['heading']) or 'chapter'}"
    return chapters


def chapter_start_lines(lines, chapters):
    """Строка текста, с которой начинается каждая глава (по её первой строке). None — найти не удалось (например,
    равные части режут текст посреди строки): тогда главы вручную не правятся."""
    starts, cursor = [], 0
    for ch in chapters:
        first = next((line.strip() for line in ch["text"].split("\n") if line.strip()), "")
        found = next((k for k in range(cursor, len(lines)) if lines[k].strip() == first), None)
        if found is None:
            return None
        starts.append(found)
        cursor = found + 1
    return starts


def chapters_from_cuts(lines, cuts, titles, known=None):
    """Главы по ручным разрезам: каждая — от своей строки до начала следующей. known — заголовки и названия, которые
    у глав с той же строкой начала были при автоматическом разбиении."""
    known = known or {}
    chapters = []
    for k, start in enumerate(cuts):
        end = cuts[k + 1] if k + 1 < len(cuts) else len(lines)
        text = "\n".join(lines[start:end]).strip()
        if not text:
            continue
        first = text.split("\n", 1)[0].strip()
        heading, title = known.get(start) or ((first, first) if len(first) <= 80 else (first[:60].rstrip() + "…",) * 2)
        if str(start) in titles:
            heading = title = titles[str(start)]
        chapters.append({"heading": heading, "title": title, "text": text + "\n", "line": start})
    return chapters


def chapter_warnings(chapters, profile):
    warnings = []
    if len(chapters) > 1:
        sizes = sorted(len(c["text"]) for c in chapters)
        median = sizes[len(sizes) // 2]
        for c in chapters:
            n = len(c["text"])
            if n < 300:
                warnings.append(tr("Глава {0} очень короткая ({1} симв.) — возможно, ложный заголовок.").format(c['id'], n))
            elif median and n > 3 * median:
                warnings.append(tr("Глава {0} в {1:.1f} раза длиннее обычной — возможно, пропущен заголовок.").format(c['id'],
                                                                                                                      n / median))
        titles = [c["title"] for c in chapters if c["title"]]
        repeated = {t for t in titles if titles.count(t) > 1}
        if repeated:
            warnings.append(tr("Повторяются названия глав: ") + ", ".join(sorted(repeated))[:120])
    if profile in ("headings", "numbers"):
        found = [re.search(r"\d+", c["heading"]) for c in chapters if c["heading"] != "Начало"]
        if found and all(found):
            values = [int(m.group()) for m in found]
            if values != list(range(values[0], values[0] + len(values))):
                warnings.append(tr("Номера глав идут не подряд: ") + ", ".join(map(str, values[:15]))
                                + ("…" if len(values) > 15 else ""))
    return warnings


# ============================================================
# НАРЕЗКА ГЛАВЫ НА КУСКИ ДЛЯ ОЗВУЧКИ
# ============================================================


def _hard_cut(part, max_chars):
    """Часть длиннее лимита (URL, текст без пробелов и знаков препинания) режется по символам: иначе кусок вышел бы за лимит."""
    return [part[i:i + max_chars] for i in range(0, len(part), max_chars)] if len(part) > max_chars else [part]


def _pack(parts, max_chars, sep):
    """Склеивает части в куски не длиннее max_chars (слишком длинную часть предварительно режет по символам)."""
    out, buf = [], ""
    for part in (piece for p in parts for piece in _hard_cut(p, max_chars)):
        if buf and len(buf) + len(sep) + len(part) > max_chars:
            out.append(buf)
            buf = part
        else:
            buf = f"{buf}{sep}{part}" if buf else part
    if buf:
        out.append(buf)
    return out


def _split_long_paragraph(par, max_chars):
    pieces = []
    for s in re.split(r"(?<=[.!?…»])\s+|(?<=[。！？])\s*", par):      # у китайского и японского нет пробелов после точки
        if not s:
            continue
        if len(s) <= max_chars:
            pieces.append(s)
        else:
            pieces.extend(_pack(s.split(), max_chars, " "))
    return _pack(pieces, max_chars, " ")


def split_text_into_chunks(text, max_chars):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) == 1 and "\n" in text:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    pieces = []
    for p in paras:
        pieces.extend(_split_long_paragraph(p, max_chars) if len(p) > max_chars else [p])
    return _pack(pieces, max_chars, "\n\n")


# ============================================================
# ДВИЖКИ ОЗВУЧКИ
# ============================================================


class Engine:
    """Базовый класс движков озвучки (паттерн Strategy): сервис работает с любым движком одинаково.
    Движок получает настройки, шину событий и токен отмены; synth(text) возвращает звук (float32, моно)."""
    sample_rate = SAMPLE_RATE
    max_parallel = 1             # сколько кусков движок может озвучивать одновременно

    def __init__(self, ctx, events=None, token=None):
        self.ctx = ctx
        self.events = events or EventBus()
        self.token = token or CancellationToken()
        self._local = threading.local()                       # пояснение к куску у каждого потока своё
        self.last_note = ""

    @property
    def last_note(self):
        """Пояснение к последнему куску этого потока (например, цена), выводится в журнале."""
        return getattr(self._local, "note", "")

    @last_note.setter
    def last_note(self, value):
        self._local.note = value

    def synth(self, text):
        raise NotImplementedError

    def close(self):
        """Освобождает ресурсы движка (модель, память видеокарты)."""


class DummyEngine(Engine):
    """Тестовый движок: вместо голоса тихий тон."""

    def synth(self, text):
        n = max(int(len(text) / 15 * self.sample_rate), self.sample_rate // 2)
        t = np.arange(n) / self.sample_rate
        return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


IDLE_RELEASE_SECONDS = 30                           # столько секунд без работы — и память отдаётся системе
PREVIEW_UNLOAD_SECONDS = 30                         # модель Qwen для проб выгружается после стольких секунд без проб
QWEN_MIN_VRAM_GB = {"float16": 4.5, "float32": 9}   # меньше — модель не поместится в видеопамять
QWEN_LOAD_NOTICE_SECONDS = 20                       # пока модель грузится, журнал раз в столько секунд сообщает, что жив


def qwen_dtype_name(capability, vram_gb):
    """Точность модели: fp16 для Volta и новее; на Pascal и Maxwell fp16 очень медленный, поэтому fp32, если хватает
    видеопамяти (модели в fp32 нужно около 10 ГБ)."""
    return "float32" if capability < (7, 0) and vram_gb >= 10 else "float16"


def is_triton_failure(error):
    """Сбой ускоренных ядер Triton (сборка или запуск), а не какая-то другая ошибка."""
    return "triton" in "".join(traceback.format_exception(error)).lower()


class QwenEngine(Engine):
    """Локальный Qwen3-TTS с клонированием голоса из ref.wav. Ускорение Triton включается, если видеокарта его
    поддерживает; если ядра не собрались или упали, модель перезагружается без него, и озвучка продолжается."""

    def __init__(self, ctx, events=None, token=None):
        super().__init__(ctx, events, token)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(tr("CUDA недоступна. Проверь драйвер NVIDIA и установку torch (меню "
                                  "«Окружение»)."))
        voice = current_voice(ctx.settings)
        if voice is None:
            raise RuntimeError(tr("Нет голоса для Qwen: добавьте запись голоса в «Настройках»."))
        self.ref_audio, self.ref_text = voice_file(voice), voice["text"]
        if not self.ref_audio.exists():
            raise FileNotFoundError(tr("Нет образца голоса: {0}").format(self.ref_audio))
        self.torch = torch
        self.gpu = torch.cuda.get_device_name(0)
        capability = tuple(torch.cuda.get_device_capability(0))
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        self.dtype = qwen_dtype_name(capability, vram_gb)
        need = QWEN_MIN_VRAM_GB[self.dtype]
        if vram_gb < need:
            raise RuntimeError(tr("У видеокарты {0} {1:.1f} ГБ памяти, а модели Qwen нужно не меньше {2} ГБ: на ней Qwen не "
                                  "поместится. Используйте Gemini.").format(self.gpu, vram_gb, need))
        refused = ctx.settings.qwen_no_triton_gpu == f"{self.gpu}|{APP_VERSION}"   # после обновления пробуем снова
        use_triton = capability >= MODERN_GPU and not refused
        self.events.log(tr("🚀 Загружаю Qwen3-TTS (Faster + Triton). При первом запуске модель "
                           "скачается в папку models/ ..."))
        if not use_triton:
            self.events.log(tr("   Видеокарта {0} ({1}): работаю без ускорения Triton, озвучка будет медленнее.").format(
                self.gpu, gpu_generation(capability)))
        self._load(use_triton)
        if self.triton:
            self.events.log(tr("   Ускорение Triton: включено."))
        elif refused:
            self.events.log(tr("   Ускорение Triton: выключено — в этой версии программы оно на этой видеокарте уже не "
                               "заработало. После обновления программы будет новая попытка."))
        self.events.log(tr("🎙️ Голос из {0} будет считан при первом куске и дальше берётся из кэша "
                           "библиотеки.").format(self.ref_audio.name))

    def _load(self, use_triton):
        from faster_qwen3_tts import FasterQwen3TTS
        self.model, self.triton = None, False
        gc.collect()
        self.torch.cuda.empty_cache()
        cached = (PATHS.models / "huggingface" / "hub" / ("models--" + QWEN_MODEL_ID.replace("/", "--"))).exists()
        self.events.log(tr("   Загружаю модель в видеопамять…") if cached
                        else tr("   Скачиваю модель Qwen (первый раз, несколько ГБ — это долго)…"))
        started, done = time.monotonic(), threading.Event()

        def still_loading():                       # без этого долгая загрузка выглядит как зависание
            while not done.wait(QWEN_LOAD_NOTICE_SECONDS):
                self.events.log(tr("   …ещё загружаю модель, прошло {0} с").format(int(time.monotonic() - started)))
        threading.Thread(target=still_loading, daemon=True).start()
        try:
            self.model = FasterQwen3TTS.from_pretrained(
                QWEN_MODEL_ID, device="cuda", dtype=getattr(self.torch, self.dtype),
                attn_implementation="sdpa", backend="torch", max_seq_len=QWEN_MAX_SEQ_LEN,
            )
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            hint = (tr(" Не хватило видеопамяти: закройте игры и другие программы, которые её занимают, или используйте "
                       "Gemini.") if "out of memory" in reason.lower() else
                    tr(" Windows не хватило памяти для загрузки модели: закройте лишние программы (браузер, игры) или "
                       "включите «Автоматически выбирать объём файла подкачки»: Параметры → Система → О системе → "
                       "Дополнительные параметры системы → Быстродействие: Параметры → Дополнительно → Виртуальная "
                       "память, затем перезагрузите компьютер.")
                    if "1455" in reason or "paging file" in reason.lower() else "")
            raise RuntimeError(tr("Модель Qwen не загрузилась: {0}.").format(reason[:300]) + hint) from e
        finally:
            done.set()
        self.load_seconds = time.monotonic() - started
        self.events.log(tr("   Модель загружена за {0} с.").format(int(self.load_seconds)))
        if use_triton:
            try:
                from qwen3_tts_triton.models.patching import apply_triton_kernels, find_patchable_model
                apply_triton_kernels(find_patchable_model(self.model.model),
                                     patch_range=QWEN_PATCH_RANGE, enable_fused_norm=True)
                self.triton = True
            except Exception as e:
                self._without_triton(e)

    def warm_up(self):
        """Короткая фраза сразу после загрузки: ускоренные ядра собираются здесь, а не в первом куске книги, и RTF кусков
        честный. Если Triton тут сломается, запасной режим включится до озвучки."""
        started = time.monotonic()
        self.synth(tr("Проверка голоса."))
        warm = time.monotonic() - started
        self.startup_note = tr(" · загрузка модели {0} с, прогрев {1} с").format(int(getattr(self, "load_seconds", 0)),
                                                                                  int(warm))
        self.events.log(tr("   Прогрев модели: {0} с (в RTF кусков не входит).").format(int(warm)))

    def _without_triton(self, error):
        """Ускорение не заработало: запоминаем видеокарту и перезагружаем модель без него."""
        reason = " ".join(f"{type(error).__name__}: {error}".split())[:160]
        self.events.log(tr("   ⚠️ Ускорение Triton на этой видеокарте не заработало ({0}). Продолжаю без него, озвучка "
                           "будет медленнее.").format(reason))
        self.ctx.update(qwen_no_triton_gpu=f"{self.gpu}|{APP_VERSION}")
        self._load(False)

    def _generate(self, text):
        # библиотека сама считывает ref.wav один раз и запоминает его по паре (файл, текст образца)
        cfg = self.ctx.settings
        waves, sample_rate = self.model.generate_voice_clone(
            text=text, language=qwen_model_language(cfg), ref_audio=str(self.ref_audio), ref_text=self.ref_text,
            non_streaming_mode=True,
        )
        self.sample_rate = int(sample_rate)
        audio = np.concatenate([np.asarray(w).reshape(-1) for w in waves]).astype(np.float32, copy=False)
        self.torch.cuda.synchronize()
        gc.collect()
        self.torch.cuda.empty_cache()
        return audio

    def synth(self, text):
        try:
            return self._generate(text)
        except (Cancelled, KeyboardInterrupt):
            raise
        except Exception as e:
            if not (self.triton and is_triton_failure(e)):
                raise
            self._without_triton(e)
            return self._generate(text)

    def close(self):
        self.model = None
        gc.collect()
        self.torch.cuda.empty_cache()


def gemini_api_key(settings):
    """Ключ из переменной окружения (главнее) или из настроек."""
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or settings.gemini_api_key or "").strip()


def mask_key(key):
    return (key[:4] + "…" + key[-4:]) if len(key) > 10 else (tr("задан") if key else tr("не задан"))


def decode_audio_bytes(raw, target_sr=SAMPLE_RATE):
    """Ответ Gemini → моно float32. Модели 3.8 отдают WAV с заголовком, старые — сырой PCM 16 бит."""
    if raw[:4] == b"RIFF":
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            n = int(len(data) * target_sr / sr)
            data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
        return data.astype(np.float32, copy=False)
    pcm = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


class GeminiClient:
    """HTTP-запросы к Gemini API. Отдельный класс, чтобы в тестах подменять сеть заглушкой."""

    def __init__(self, key):
        import requests
        self.requests, self.key = requests, key

    def _headers(self):
        return {"x-goog-api-key": self.key, "Content-Type": "application/json"}

    def generate(self, model, payload):
        return self.requests.post(f"{GEMINI_API_BASE}/models/{model}:generateContent", json=payload,
                                  headers=self._headers(), timeout=300)

    def list_models(self):
        return self.requests.get(f"{GEMINI_API_BASE}/models", params={"pageSize": 1000},
                                 headers=self._headers(), timeout=30)


class RateLimiter:
    """Не пускает запросы чаще, чем раз в interval секунд; один на все потоки. Ожидание прерывается «Стопом»."""

    def __init__(self, token):
        self.token, self._lock, self._next_at = token, threading.Lock(), 0.0

    def wait(self, interval):
        if interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_at)
            self._next_at = start + interval
        if start > now and self.token.wait(start - now):
            raise Cancelled()


class GeminiEngine(Engine):
    """Облачный Gemini TTS по API-ключу: повторы, темп запросов по тарифу, учёт расходов. На платном тарифе несколько
    кусков озвучиваются одновременно."""

    def __init__(self, ctx, events=None, token=None, client=None):
        super().__init__(ctx, events, token)
        key = gemini_api_key(ctx.settings)
        if not key:
            raise RuntimeError(tr("Нет API-ключа Gemini. Меню «Настройки» → ключ (бесплатно: {0}).").format(GEMINI_KEY_URL))
        self.client = client or GeminiClient(key)
        self.model, self.voice = ctx.settings.gemini_model, ctx.settings.gemini_voice
        paid = ctx.settings.gemini_paid
        self.max_parallel = GEMINI_PARALLEL_PAID if paid else 1
        self.interval = GEMINI_PAID_INTERVAL if paid else GEMINI_FREE_INTERVAL
        self.limiter = RateLimiter(self.token)
        self.events.log(tr("☁️ Gemini API · модель {0} · голос {1}").format(self.model, self.voice))

    def payload(self, text):
        # у моделей 3.8 новая схема выбора голоса, у старых — prebuiltVoiceConfig
        voice = {"voice": self.voice} if self.model.startswith("gemini-3.8") \
            else {"prebuiltVoiceConfig": {"voiceName": self.voice}}
        return {"contents": [{"role": "user", "parts": [{"text": text}]}],
                "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {"voiceConfig": voice}}}

    @staticmethod
    def fatal_error(r):
        """Ошибки, при которых повторять бессмысленно: понятное сообщение или None."""
        t = r.text or ""
        low = t.lower()
        bad_key = ("api key not valid", "api_key_invalid", "api key expired")
        if r.status_code in (400, 401, 403) and any(b in low for b in bad_key):
            return tr("Google не принял API-ключ. Проверь его в меню «Настройки».")
        if r.status_code == 403:
            return tr("Доступ запрещён (403): у ключа нет прав на эту модель, регион не "
                      "поддерживается или нужна оплата. {0}").format(t[:200])
        if r.status_code == 404:
            return tr("Модель не найдена для этого ключа. Выбери другую в меню «Настройки» → "
                      "модель.")
        if r.status_code == 429 and ("perday" in low or "per day" in low or "daily" in low):
            return (tr("Дневной лимит бесплатного тарифа исчерпан. Запусти скрипт завтра — он "
                       "продолжит с этого же места, либо подключи оплату в Google AI Studio."))
        if r.status_code == 400:
            return tr("Запрос отклонён (400): {0}").format(t[:300])
        return None

    @staticmethod
    def retry_delay(r):
        """Сколько Google просит подождать при 429 (секунды) или None."""
        try:
            for d in r.json().get("error", {}).get("details", []):
                v = d.get("retryDelay")
                if v:
                    return min(float(str(v).rstrip("s")), 180.0)
        except Exception:
            pass
        m = re.search(r"retry in ([\d.]+)s", r.text or "", re.IGNORECASE)
        return min(float(m.group(1)), 180.0) if m else None

    @staticmethod
    def extract(r, text):
        """(звук, (входных токенов, выходных токенов), причина неудачи). Ничего не хранит в самом движке: он работает из
        нескольких потоков."""
        try:
            data = r.json()
            parts = data["candidates"][0]["content"]["parts"]
            raw = base64.b64decode(next(p["inlineData"]["data"] for p in parts if "inlineData" in p))
        except Exception:
            try:
                reason = data["candidates"][0].get("finishReason") or data.get("promptFeedback")
            except Exception:
                reason = "?"
            return None, (0, 0), tr("в ответе нет аудио (finishReason: {0})").format(reason)
        audio = decode_audio_bytes(raw)
        meta = data.get("usageMetadata") or {}
        usage = (meta.get("promptTokenCount") or len(text) / TEXT_CHARS_PER_TOKEN,
                 meta.get("candidatesTokenCount") or len(audio) / SAMPLE_RATE * AUDIO_TOKENS_PER_SEC)
        return audio, usage, ""

    def finish(self, audio, usage):
        """Записывает расход в журнал и готовит строку про стоимость куска."""
        in_tokens, out_tokens = usage
        self.ctx.usage.record(self.model, in_tokens, out_tokens, len(audio) / SAMPLE_RATE)
        cost = gemini_cost(self.model, in_tokens, out_tokens, pacific_today())
        price = f"≈ ${cost:.3f}" if cost is not None else tr("цена модели неизвестна")
        self.last_note = price + ("" if self.ctx.settings.gemini_paid else tr(" по платному тарифу (сейчас бесплатно)"))

    def synth(self, text):
        cfg = self.ctx.settings
        if not cfg.gemini_paid and cfg.gemini_daily_limit and self.ctx.usage.requests_today(self.model) >= cfg.gemini_daily_limit:
            raise RuntimeError(tr("Дневной лимит запросов ({0}) исчерпан по твоим настройкам. Продолжи "
                                  "завтра, после полуночи по тихоокеанскому времени, или включи оплату в "
                                  "AI Studio и отметь это в «Настройках».").format(cfg.gemini_daily_limit))
        payload = self.payload(text)
        last = ""
        for attempt in range(1, GEMINI_RETRIES + 1):
            self.token.raise_if_cancelled()
            self.limiter.wait(self.interval)
            wait = GEMINI_RETRY_DELAY * attempt
            try:
                r = self.client.generate(self.model, payload)
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if r.status_code == 200:
                    audio, usage, error = self.extract(r, text)
                    if audio is not None and len(audio):
                        self.finish(audio, usage)
                        return audio
                    last = error
                else:
                    last = f"HTTP {r.status_code}: {(r.text or '')[:200]}"
                    fatal = self.fatal_error(r)
                    if fatal:
                        raise RuntimeError(fatal)
                    if r.status_code == 429:
                        wait = self.retry_delay(r) or min(30 * attempt, 120)
            if attempt < GEMINI_RETRIES:
                self.events.log(tr("\n   ⚠️ {0}\n   🔄 повтор {1}/{2} через {3:.0f} с...").format(last, attempt + 1,
                                                                                                 GEMINI_RETRIES, wait))
                if self.token.wait(wait):
                    raise Cancelled()
        raise RuntimeError(tr("Gemini не дал результат после {0} попыток: {1}").format(GEMINI_RETRIES, last))


ENGINES = {"qwen": QwenEngine, "gemini": GeminiEngine, "dummy": DummyEngine}


def make_engine(ctx, name, events=None, token=None):
    """Фабрика (Factory): создаёт движок по имени; неизвестное имя даёт тестовый движок."""
    return ENGINES.get(name, DummyEngine)(ctx, events, token)


# ============================================================
# ПРОВЕРКА И УСТАНОВКА ОКРУЖЕНИЯ
# ============================================================

AUTOFIX_MODULES = {"plotly": "plotly", "streamlit": "streamlit", "pynvml": NVML_PIN}


def console_python():
    """Python для служебных процессов. Из FoxyAudioBookMaker.exe программа работает в pythonw.exe, у которого нет обычного
    вывода: его ответы пропадают. Поэтому проверки и pip запускаются через python.exe из той же папки."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return sys.executable


def torch_cuda_status():
    """(версия torch, есть ли CUDA). Проверяется в отдельном процессе. Если проверка не дала ответа, судим по сборке:
    сборка с CUDA («+cu…») считается рабочей — иначе программа предлагала бы зря скачивать torch заново."""
    tv = pkg_version("torch")
    if not tv:
        return None, False
    r = subprocess.run([console_python(), "-c", "import torch; print(int(torch.cuda.is_available()))"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    answer = r.stdout.strip().splitlines()[-1:] if r.stdout.strip() else []
    if answer in (["0"], ["1"]):
        return tv, answer == ["1"]
    return tv, "+cu" in tv


def gpu_info():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        name, mem, drv = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "vram_gb": round(int(mem) / 1024, 1), "driver": drv}
    except Exception:
        return None


def gpu_compute_capability():
    """Поколение видеокарты как (старшая, младшая) версия архитектуры по nvidia-smi; None — карты нет или драйвер старый."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=20)
        major, minor = r.stdout.strip().splitlines()[0].strip().split(".")
        return int(major), int(minor)
    except Exception:
        return None


def gpu_generation(capability):
    return next((name for start, name in GPU_GENERATIONS if capability >= start), "?") if capability else "?"


def legacy_gpu(capability=None):
    """Видеокарта старше Turing (GTX 10xx, 9xx, Titan V): нужна сборка PyTorch для старых карт и работа без Triton."""
    capability = capability or gpu_compute_capability()
    return bool(capability) and capability < MODERN_GPU


def qwen_warnings():
    """Не блокирующие предупреждения: железо и путь к папке."""
    w, gpu = [], gpu_info()
    if gpu:
        try:
            if legacy_gpu():
                if int(gpu["driver"].split(".")[0]) < 560:
                    w.append(tr("драйвер NVIDIA {0} старый: сборка torch для старых карт (cu126) просит 560 или "
                                "новее — обнови драйвер").format(gpu['driver']))
            elif int(gpu["driver"].split(".")[0]) < 570:
                w.append(tr("драйвер NVIDIA {0} старый: сборка torch cu128 просит 570 или новее — "
                            "обнови драйвер").format(gpu['driver']))
        except ValueError:
            pass
        if gpu["vram_gb"] < 7.5:
            w.append(tr("видеопамяти {0} ГБ: для Qwen рекомендуется от 8 ГБ, на меньшем объёме "
                        "возможны вылеты").format(gpu['vram_gb']))
    p = str(PATHS.base)
    if os.name == "nt" and len(p) > 70:
        w.append(tr("путь к папке длинный ({0} симв.): у Windows лимит около 260, установка "
                    "torch может упасть. Лучше короткая папка, например C:\\audiobook").format(len(p)))
    if not p.isascii():
        w.append(tr("в пути к папке есть не латинские символы: Triton может не "
                    "скомпилировать ядра. Лучше путь вроде C:\\audiobook"))
    return w


def engine_problems(ctx, name):
    """(что можно починить установкой, что нужно сделать руками)."""
    fix, manual = [], []
    if name == "qwen":
        if sys.version_info[:2] < REQUIRED_PYTHON:
            manual.append(tr("Python в .venv ниже %d.%d — qwen3-tts-triton не заработает. Пересоздай "
                             ".venv на нужной версии.") % REQUIRED_PYTHON)
        tv, cuda = torch_cuda_status()
        legacy = legacy_gpu()
        if not tv:
            fix.append(tr("не установлен torch"))
        elif "+cu" not in tv or not cuda:
            fix.append(tr("torch {0} без поддержки CUDA").format(tv))
        elif legacy and "+cu128" in tv:
            fix.append(tr("torch {0} не поддерживает видеокарту поколения {1}: нужна сборка для старых карт").format(
                tv, gpu_generation(gpu_compute_capability())))
        packages = ("faster-qwen3-tts",) if legacy else (
            "faster-qwen3-tts", "qwen3-tts-triton", "triton-windows" if os.name == "nt" else "triton")
        for pkg in packages:
            if not pkg_version(pkg):
                fix.append(tr("нет пакета {0}").format(pkg))
        triton = [] if legacy else [QWEN_TRITON_PKG, QWEN_TRITON_WIN if os.name == "nt" else ""]
        wanted = QWEN_CORE_PKGS + [QWEN_ACCELERATE] + triton
        for pin in wanted:                                  # torch не сверяем: его замена — это гигабайты и сборка под видеокарту
            have = installed_older_than_pin(pin) if "==" in pin else None
            if have:
                fix.append(tr("пакет {0} {1} старее нужного ({2})").format(pin_parts(pin)[0], have, pin_parts(pin)[1]))
        if not gpu_info():
            manual.append(tr("видеокарта NVIDIA или её драйвер не найдены (nvidia-smi) — Qwen "
                             "работает только на NVIDIA, используй Gemini"))
        voice = current_voice(ctx.settings)
        if voice is None:
            manual.append(tr("нет голоса для Qwen: добавьте запись голоса в «Настройках»"))
        elif not voice_file(voice).exists():
            manual.append(tr("не найден файл голоса «{0}»: {1}").format(voice["name"], voice_file(voice)))
        elif not voice["text"]:
            manual.append(tr("у голоса «{0}» нет текста записи: впишите его в «Настройках»").format(voice["name"]))
    elif name == "gemini":
        if not has_module("requests"):
            fix.append(tr("нет пакета requests"))
        if not gemini_api_key(ctx.settings):
            manual.append(tr("нет API-ключа Gemini: меню «Настройки» → ключ (бесплатно: {0})").format(GEMINI_KEY_URL))
    return fix, manual


def loaded_lib_versions():
    """Версии библиотек, которые скрипт уже загрузил в память (проверяется свежим процессом)."""
    code = "import importlib.metadata as m; print(' '.join(m.version(n) for n in ('numpy', 'soundfile', 'cffi')))"
    return subprocess.run([console_python(), "-c", code], capture_output=True, text=True).stdout.strip()


class RestartRequired(Exception):
    """После установки сменились версии библиотек, которые программа уже загрузила: нужен перезапуск."""


def install_engine(name, events=None, token=None):
    """Ставит в .venv то, что нужно для qwen, gemini или ffmpeg; True — получилось. Вопросов не задаёт: предупреждения
    и подтверждения — дело вызывающего (консоль или окно). Если сменились версии уже загруженных библиотек, поднимает
    RestartRequired."""
    say = events.log if events else log

    def run_pip(*args):
        return pip(*args, events=events, token=token)

    say(tr("\n📦 Установка для {0} — только в {1}\n").format(name, PATHS.venv))
    if name == "gemini":
        return run_pip("install", REQUESTS_PIN)
    if name == "ffmpeg":
        return run_pip("install", IMAGEIO_FFMPEG_PIN) and ensure_ffmpeg()

    versions_before = loaded_lib_versions()
    capability = gpu_compute_capability()
    legacy = legacy_gpu(capability)
    index, torch_pkgs = (LEGACY_TORCH_INDEX, LEGACY_TORCH_PKGS) if legacy else (TORCH_INDEX, QWEN_TORCH_PKGS)
    tv, cuda = torch_cuda_status()
    if not tv or "+cu" not in tv or not cuda or (legacy and "+cu128" in tv):
        say(tr("1/5 PyTorch с CUDA (скачивается около 3 ГБ, это долго)..."))
        if legacy:
            say(tr("   Видеокарта поколения {0}: ставлю сборку PyTorch для старых карт, без ускорения Triton.").format(
                gpu_generation(capability)))
        if not run_pip("install", *torch_pkgs, "--index-url", index):
            return False
    if not legacy:
        say("2/5 Triton...")
        if not run_pip("install", QWEN_TRITON_WIN if os.name == "nt" else "triton"):
            return False
    say(tr("3/5 Qwen-TTS и его зависимости..."))
    if not run_pip("install", *QWEN_CORE_PKGS):
        return False
    say(tr("4/5 Версия accelerate, с которой у тебя всё работало..."))
    run_pip("install", QWEN_ACCELERATE)            # конфликт с pin у qwen-tts-hf — это ожидаемо, не критично
    if not legacy:
        say(tr("5/5 qwen3-tts-triton (без его конфликтующих зависимостей)..."))
        if not run_pip("install", QWEN_TRITON_PKG, "--no-deps"):
            return False
    run_pip("install", NVML_PIN)

    code = "import torch, faster_qwen3_tts; " + (
        "" if legacy else "from qwen3_tts_triton.models.patching import apply_triton_kernels, find_patchable_model; ")
    code += "print('OK')"
    for _ in range(4):
        r = subprocess.run([console_python(), "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if "OK" in r.stdout:
            say(tr("✅ Qwen установлен, импорт проходит."))
            if loaded_lib_versions() != versions_before:
                say(tr("⚠️ При установке сменились версии библиотек (numpy и др.), которые "
                       "программа уже использует. Закройте её и запустите заново: она продолжит "
                       "с того же места."))
                raise RestartRequired()
            return True
        m = re.search(r"No module named '([\w.]+)'", r.stderr)
        mod = m.group(1).split(".")[0] if m else None
        if mod in AUTOFIX_MODULES:
            say(tr("   не хватает {0}, докачиваю...").format(mod))
            run_pip("install", AUTOFIX_MODULES[mod])
            continue
        say(tr("❌ Проверка импорта не прошла:\n") + r.stderr[-1200:])
        return False
    return False


FFMPEG = None


def ensure_ffmpeg():
    """Есть ли ffmpeg: в системе или из пакета imageio-ffmpeg (его ставит установка движка)."""
    global FFMPEG
    if FFMPEG:
        return True
    FFMPEG = shutil.which("ffmpeg")
    if not FFMPEG and has_module("imageio_ffmpeg"):
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return bool(FFMPEG)


def check_key_models(key, model, client=None):
    """Проверяет ключ запросом списка моделей (бесплатно). Возвращает (получилось ли, текст для пользователя)."""
    if not key:
        return False, tr("❌ Ключ не задан.")
    try:
        client = client or GeminiClient(key)
    except ImportError:
        return False, tr("❌ Нужен пакет requests: установите его в меню «Окружение» (или кнопкой "
                         "установки в окне).")
    try:
        r = client.list_models()
    except Exception as e:
        return False, tr("❌ Нет связи с Google: {0}: {1}").format(type(e).__name__, e)
    if r.status_code != 200:
        return False, tr("❌ Ключ не принят (HTTP {0}): {1}").format(r.status_code, r.text[:200])
    names = sorted({m.get("name", "").split("/")[-1] for m in r.json().get("models", [])})
    tts = [n for n in names if "tts" in n]
    text = tr("✅ Ключ работает ({0}). TTS-модели, доступные этому ключу: {1}").format(mask_key(key),
                                                                                      ', '.join(tts) or tr("не найдено"))
    if tts and model not in tts:
        text += tr("\n⚠️ Выбранная модель {0} в этом списке не найдена: выберите другую в "
                   "«Настройках».").format(model)
    return True, text


def tariff_label(ctx):
    if ctx.settings.gemini_paid:
        return tr("платный")
    limit = ctx.settings.gemini_daily_limit
    return tr("бесплатный, лимит {0} запросов в сутки").format(limit) if limit else tr("бесплатный, лимит не задан")


def preview_sentences(text, count=2, limit=400):
    """Первые предложения текста главы (без строки заголовка) для пробы голоса."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    body = " ".join(lines[1:] if len(lines) > 1 else lines)
    sentences = re.split(r"(?<=[.!?…])\s+", body)
    return " ".join(sentences[:count])[:limit].strip()


def make_voice_preview(ctx, engine_name, text, events=None, token=None, engine=None):
    """Проба голоса: озвучивает text выбранным движком и голосом, пишет .cache/preview.wav. Возвращает (путь, движок):
    движок Qwen можно передать снова, чтобы не загружать модель повторно."""
    if engine is None or engine.name != engine_name:
        engine = make_engine(ctx, engine_name, events, token)
        engine.name = engine_name
    if engine_name == "qwen":
        voice = current_voice(ctx.settings)
        engine.ref_audio, engine.ref_text = voice_file(voice), voice["text"]
    audio = engine.synth(speech_text(ctx, text))
    out = PATHS.cache / "preview.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, engine.sample_rate, subtype="PCM_16")
    return out, engine


def diagnostics_lines(ctx):
    """Сведения для отчёта об ошибке: версии, пути, размеры, видеокарта, готовность движков. Секретов в них нет."""
    def mark(ok):
        return "+" if ok else "−"

    try:
        import tkinter
        tk_version = str(tkinter.TkVersion)
    except ImportError:
        tk_version = tr("нет")
    packages = ("numpy", "soundfile", "requests", "num2words", "torch", "faster-qwen3-tts", "qwen3-tts-triton", "tkinterdnd2",
                "sv-ttk")
    started = tr("запуск из {0}").format(Path(exe_path()).name) if exe_path() else tr("запуск скриптом")
    lines = [f"{APP_TITLE} {APP_VERSION} · {started}",
             f"Python {sys.version.split()[0]} · {platform.platform()} · Tk {tk_version}",
             tr("Пакеты: ") + ", ".join(f"{name} {pkg_version(name) or '—'}" for name in packages),
             tr("Окружение: {0}").format(PATHS.venv), tr("Модели: {0}").format(PATHS.models),
             tr("Занимает: .venv {0} · models {1} · .cache {2}").format(
                 fmt_size(folder_size(PATHS.venv)), fmt_size(folder_size(PATHS.models)), fmt_size(folder_size(PATHS.cache))),
             f"ffmpeg: {mark(bool(shutil.which('ffmpeg') or has_module('imageio_ffmpeg')))}"]
    for name, title in (("qwen", "Qwen"), ("gemini", "Gemini")):
        fix, manual = engine_problems(ctx, name)
        lines += [f"{title}: {mark(not fix and not manual)}"] + [f"  − {p}" for p in fix + manual]
    model_dir = PATHS.models / "huggingface" / "hub" / ("models--" + QWEN_MODEL_ID.replace("/", "--"))
    model_state = tr("скачана") if model_dir.exists() else tr("скачается при первом запуске (несколько ГБ)")
    lines.append(tr("Модель Qwen: {0}").format(model_state))
    gpu = gpu_info()
    capability = gpu_compute_capability()
    if capability:
        mode = (tr("сборка для старых карт, без Triton") if legacy_gpu(capability)
                else tr("без Triton (на этой карте оно не заработало)")
                if ctx.settings.qwen_no_triton_gpu.endswith(f"|{APP_VERSION}")
                else tr("с ускорением Triton"))
        lines.append(tr("Поколение видеокарты: {0} ({1}.{2}) · Qwen: {3}").format(gpu_generation(capability), *capability, mode))
    lines.append(tr("Видеокарта: {0}, {1} ГБ, драйвер {2}").format(gpu['name'], gpu['vram_gb'], gpu['driver']) if gpu
                 else tr("Видеокарта NVIDIA: не найдена"))
    lines += [f"⚠️ {w}" for w in qwen_warnings()]
    cfg = ctx.settings
    lines.append(tr("Gemini: ключ {0} · модель {1} · голос {2} · кусок {3} · {4}").format(mask_key(gemini_api_key(cfg)),
                                                                                          cfg.gemini_model, cfg.gemini_voice,
                                                                                          cfg.gemini_chunk_chars,
                                                                                          tariff_label(ctx)))
    return lines


RELEASES_URL = "https://api.github.com/repos/ed9ward/Foxy_AudioBook_Maker/releases/latest"
EXE_ASSET = "FoxyAudioBookMaker.exe"
SCRIPT_ASSET = "audiobook_master.py"
SUMS_ASSET = "SHA256SUMS"                       # файл выпуска со строками «sha256  имя»: его кладёт workflow выпуска
SCRIPT_PATH = Path(__file__).resolve()          # какой файл заменяет обновление (тесты подставляют копию)


def http_get(url, timeout=20):
    import urllib.request
    request = urllib.request.Request(url, headers={"User-Agent": APP_TITLE})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_checksums(text):
    """{имя файла: sha256} из SHA256SUMS (строки «хеш  имя»)."""
    sums = {}
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            sums[parts[1].strip().lstrip("*")] = parts[0].lower()
    return sums


def verify_checksum(data, name, sums):
    """ValueError, если у файла нет записи в контрольных суммах выпуска или хеш не совпал."""
    expected = sums.get(name)
    if not expected or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(tr("скачанный файл не прошёл проверку контрольной суммы, обновление отменено"))


def release_asset(release, name):
    return next((a for a in release.get("assets") or [] if a.get("name") == name), None)


def keep_previous(path):
    """Перед заменой кладёт рядом прежнюю версию (имя.prev): если новая не заработает, её можно вернуть переименованием."""
    try:
        shutil.copy2(path, path.with_name(path.name + ".prev"))
    except OSError:
        pass


def check_update(fetch=http_get):
    """Есть ли новая версия скрипта в последнем выпуске GitHub: (текст новой версии или None, если у вас уже она;
    описание; ошибка). Принимается только более новая версия, чей файл совпал с SHA256SUMS выпуска, компилируется
    и является этой программой."""
    try:
        release = json.loads(fetch(RELEASES_URL))
    except Exception as e:
        return None, "", tr("нет связи с GitHub: {0}").format(e)
    version = str(release.get("tag_name") or "").lstrip("v")
    if not version:
        return None, "", tr("на GitHub пока нет выпусков")
    if version_tuple(version) <= version_tuple(APP_VERSION):            # старую версию не ставим: это был бы откат
        return None, "", ""
    asset, sums_asset = release_asset(release, SCRIPT_ASSET), release_asset(release, SUMS_ASSET)
    if asset is None or sums_asset is None:
        return None, "", tr("в выпуске {0} нет скрипта или контрольных сумм, обновление отменено").format(version)
    try:
        data = fetch(asset["browser_download_url"], timeout=60)
        sums = parse_checksums(fetch(sums_asset["browser_download_url"]).decode("utf-8"))
    except Exception as e:
        return None, "", tr("нет связи с GitHub: {0}").format(e)
    try:
        verify_checksum(data, SCRIPT_ASSET, sums)
    except ValueError as e:
        return None, "", str(e)
    try:
        text = data.decode("utf-8")
        compile(text, "audiobook_master.py", "exec")
        if 'APP_TITLE = "Foxy AudioBook Maker"' not in text or f'APP_VERSION = "{version}"' not in text:
            raise ValueError("unexpected file")
    except Exception:
        return None, "", tr("скачанный файл повреждён, обновление отменено")
    notes = str(release.get("body") or "").strip().splitlines()
    return text, version + (f": {notes[0]}" if notes else ""), ""


def apply_update(text):
    """Заменяет файл программы новой версией целиком (сначала пишется рядом, потом подменяется одним шагом);
    прежняя версия остаётся рядом как audiobook_master.py.prev."""
    tmp = SCRIPT_PATH.with_name(SCRIPT_PATH.name + ".new")
    tmp.write_bytes(text.encode("utf-8"))                      # байт в байт, без перекодировки переводов строк
    keep_previous(SCRIPT_PATH)
    os.replace(tmp, SCRIPT_PATH)


def exe_path():
    """Путь к FoxyAudioBookMaker.exe, если программа запущена из него; иначе пустая строка (запуск скриптом)."""
    return os.environ.get("AUDIOBOOK_EXE", "")


def version_tuple(text):
    numbers = [int(n) for n in re.findall(r"\d+", text)[:3]]
    return tuple(numbers + [0] * (3 - len(numbers)))


def check_exe_update(fetch=http_get):
    """Новый FoxyAudioBookMaker.exe на GitHub: (ссылка или None, версия, что нового, ошибка)."""
    try:
        release = json.loads(fetch(RELEASES_URL))
    except Exception as e:
        return None, "", "", tr("нет связи с GitHub: {0}").format(e)
    version = str(release.get("tag_name") or "").lstrip("v")
    asset = next((a for a in release.get("assets") or [] if a.get("name") == EXE_ASSET), None)
    if not version or asset is None:
        return None, "", "", tr("на GitHub пока нет готового FoxyAudioBookMaker.exe")
    if version_tuple(version) <= version_tuple(APP_VERSION):
        return None, version, "", ""
    return asset["browser_download_url"], version, str(release.get("body") or "").strip(), ""


def drop_previous_versions():
    """Новая версия успешно открылась: запасные копии прежней (*.prev) больше не нужны."""
    for current in filter(None, (exe_path(), str(SCRIPT_PATH))):
        Path(current + ".prev").unlink(missing_ok=True)


def apply_exe_update(url, fetch=http_get):
    """Скачивает новый exe, сверяет его с SHA256SUMS того же выпуска и ставит на место старого (программа запущена не из
    самого exe, поэтому файл свободен). Прежний exe остаётся рядом как FoxyAudioBookMaker.exe.prev."""
    sums = parse_checksums(fetch(url.rsplit("/", 1)[0] + "/" + SUMS_ASSET).decode("utf-8"))
    data = fetch(url, timeout=180)
    if data[:2] != b"MZ" or len(data) < 50_000:
        raise ValueError(tr("скачанный файл не похож на программу"))
    verify_checksum(data, EXE_ASSET, sums)
    exe = Path(exe_path())
    tmp = exe.with_name(exe.name + ".new")
    tmp.write_bytes(data)
    keep_previous(exe)
    os.replace(tmp, exe)


def restart_exe():
    """Запускает обновлённый exe отдельно от текущей программы."""
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen([exe_path(), "--after-update"], cwd=str(PATHS.base), close_fds=True, creationflags=flags)


# ============================================================
# ТЕКСТ ДЛЯ ОЗВУЧКИ: числа, сокращения и свои замены словами
# ============================================================

NORMALIZER_VERSION = 1           # меняется, если правила изменились: начатые главы тогда озвучатся заново
NUM2WORDS_PIN = "num2words==0.5.14"

SENTENCE_END_ABBREVIATIONS = {"и так далее", "и тому подобное", "и другие", "и прочее", "et cetera"}
RU_ABBREVIATIONS = [             # (что найти, как читать) — применяются по порядку
    (r"\bи\s+т\.\s*д\.", "и так далее"), (r"\bи\s+т\.\s*п\.", "и тому подобное"), (r"\bи\s+др\.", "и другие"),
    (r"\bи\s+пр\.", "и прочее"), (r"\bт\.\s*е\.", "то есть"), (r"\bт\.\s*к\.", "так как"),
    (r"\bт\.\s*н\.", "так называемый"), (r"\bт\.\s*о\.", "таким образом"), (r"\bнапр\.", "например"),
    (r"\bсм\.", "смотри"), (r"\bул\.", "улица"), (r"\bим\.", "имени"), (r"\bпроф\.", "профессор"),
    (r"\bакад\.", "академик"), (r"\bгл\.", "глава"), (r"\bгг\.", "годы"), (r"\bг\.(?=\s*[А-ЯЁ])", "город"),
    (r"№\s*", "номер "),
]
EN_ABBREVIATIONS = [
    (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"), (r"\betc\.", "et cetera"), (r"\bMrs\.", "Missus"),
    (r"\bMr\.", "Mister"), (r"\bMs\.", "Miz"), (r"\bDr\.", "Doctor"), (r"\bvs\.", "versus"),
    (r"\bNo\.\s*(?=\d)", "number "), (r"№\s*", "number "),
]
RU_UNITS = [                     # единица после числа: (как найти, формы для 1 / 2–4 / 5+, род числа)
    (r"руб\.?|₽", ("рубль", "рубля", "рублей"), "m"), (r"коп\.?", ("копейка", "копейки", "копеек"), "f"),
    (r"тыс\.?", ("тысяча", "тысячи", "тысяч"), "f"), (r"млн\.?", ("миллион", "миллиона", "миллионов"), "m"),
    (r"млрд\.?", ("миллиард", "миллиарда", "миллиардов"), "m"), (r"км", ("километр", "километра", "километров"), "m"),
    (r"кг", ("килограмм", "килограмма", "килограммов"), "m"), (r"см", ("сантиметр", "сантиметра", "сантиметров"), "m"),
    (r"мин\.?", ("минута", "минуты", "минут"), "f"), (r"сек\.?", ("секунда", "секунды", "секунд"), "f"),
    (r"%", ("процент", "процента", "процентов"), "m"),
]
RU_ORDINAL_ENDINGS = {           # окончание после дефиса: (падеж, род, множественное ли)
    "й": ("nominative", "m", False), "ый": ("nominative", "m", False), "ий": ("nominative", "m", False),
    "я": ("nominative", "f", False), "ая": ("nominative", "f", False), "е": ("nominative", "n", False),
    "ое": ("nominative", "n", False), "го": ("genitive", "m", False), "ого": ("genitive", "m", False),
    "му": ("dative", "m", False), "ому": ("dative", "m", False), "м": ("prepositional", "m", False),
    "ом": ("prepositional", "m", False), "ю": ("accusative", "f", False), "ую": ("accusative", "f", False),
    "ой": ("genitive", "f", False), "х": ("genitive", "m", True), "ых": ("genitive", "m", True),
    "ми": ("instrumental", "m", True),
}
RU_DECADE_ENDINGS = {"е": "nominative", "х": "prepositional", "м": "dative", "ми": "instrumental"}
RU_YEAR_WORDS = {"году": "prepositional", "года": "genitive", "год": "nominative", "годом": "instrumental"}
RU_CENTURY_WORDS = {"в.": ("nominative", "век"), "век": ("nominative", "век"), "века": ("genitive", "века"),
                    "веке": ("prepositional", "веке"), "веку": ("dative", "веку"), "веком": ("instrumental", "веком")}
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


PREPOSITIONAL = ("в", "во", "на", "о", "об", "при")
GENITIVE = ("с", "со", "до", "от", "после", "около", "начала", "конца", "середины")
DATIVE = ("к", "по")


def _sub(pattern, replacement, text, keep_stop=True):
    """Как re.sub, но если найденное кончалось точкой в конце предложения, точка остаётся («…и т. д.» → «…и так далее.»)."""
    def repl(m):
        out = replacement(m) if callable(replacement) else replacement
        if keep_stop and m.group(0).endswith(".") and not out.endswith("."):
            rest = m.string[m.end():]
            if not rest.strip() or re.match(r"\s*(\n|[A-ZА-ЯЁ«\"“])", rest):
                out += "."
        return out
    return re.sub(pattern, repl, text)


@lru_cache(maxsize=256)
def _literal_hint(pattern):
    """Кусок текста, без которого правило не сработает (например, «д.» для «и т. д.»); "" — такой кусок не выделить.
    Правило пропускается, если этого куска в тексте нет: результат тот же, а большая книга открывается быстрее."""
    parts = re.split(r"\\b|\\s[*+?]|\(\?=[^)]*\)|\(\?<?!?[^)]*\)", pattern)
    literals = [part.replace("\\.", ".") for part in parts]
    literals = [part for part in literals if part and not re.search(r"[\\\[\](){}|*+?^$]", part)]
    return max(literals, key=len) if literals and len(literals) == len([part for part in parts if part]) else ""


def _apply_abbreviations(rules, text):
    for pattern, words in rules:
        hint = _literal_hint(pattern)
        if hint and hint not in text:
            continue
        text = _sub(pattern, words, text, keep_stop=words in SENTENCE_END_ABBREVIATIONS)
    return text


def _word_before(m):
    words = re.findall(r"[А-Яа-яЁёA-Za-z]+", m.string[max(0, m.start() - 20):m.start()])
    return words[-1].lower() if words else ""


@lru_cache(maxsize=1)
def has_num2words():
    return has_module("num2words")


def parse_replacements(text):
    """Свои замены из настроек: по одной на строку «что = как читать»; пустые строки и строки с # пропускаются."""
    rules = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            source, target = (part.strip() for part in line.split("=", 1))
            if source:
                rules.append((source, target))
    return rules


def bad_replacement_lines(text):
    """Номера строк, которые не похожи на «что = как читать»."""
    return [n for n, line in enumerate((text or "").splitlines(), 1)
            if line.strip() and not line.strip().startswith("#") and ("=" not in line or not line.split("=", 1)[0].strip())]


def _word_pattern(source):
    left = r"(?<!\w)" if re.match(r"\w", source) else ""
    right = r"(?!\w)" if re.search(r"\w$", source) else ""
    return left + re.escape(source) + right


def _roman(text):
    values = [ROMAN[c] for c in text]
    return sum(-v if i + 1 < len(values) and v < values[i + 1] else v for i, v in enumerate(values))


def _ru_words(number, gender="m", case="nominative"):
    from num2words import num2words
    if isinstance(number, str) and re.search(r"[.,]", number):
        return num2words(Decimal(number.replace(",", ".")), lang="ru")
    return num2words(int(number), lang="ru", gender=gender, case=case)


def _ru_ordinal(number, case="nominative", gender="m", plural=False):
    from num2words import num2words
    return num2words(int(number), lang="ru", to="ordinal", case=case, gender=gender, plural=plural)


def _ru_century(m):
    case, word = RU_CENTURY_WORDS[m.group(2)]
    if m.group(2) == "в.":
        before = _word_before(m)
        case = ("prepositional" if before in PREPOSITIONAL else "genitive" if before in GENITIVE
                else "dative" if before in DATIVE else "nominative")
        word = {"prepositional": "веке", "genitive": "века", "dative": "веку", "nominative": "век"}[case]
    return f"{_ru_ordinal(_roman(m.group(1)), case)} {word}"


SPEECH_SPLIT_CHARS = 2000      # глава длиннее: числа читаются по абзацам, и абзацы без цифр правила для чисел не трогают


def _by_paragraphs(text, numbers_rule, digitless_rule):
    """Правила для чисел только там, где есть цифры: короткий текст целиком, длинный — по абзацам."""
    if len(text) <= SPEECH_SPLIT_CHARS or "\n\n" not in text:
        return numbers_rule(text) if re.search(r"\d", text) else digitless_rule(text)
    return "\n\n".join(numbers_rule(part) if re.search(r"\d", part) else digitless_rule(part) for part in text.split("\n\n"))


def _ru_centuries(text):
    return _sub(r"\b([IVXLC]+)\s*(в\.|веком|веке|века|веку|век)(?![а-яё])", _ru_century, text)


def _normalize_ru(text, numbers):
    text = _apply_abbreviations(RU_ABBREVIATIONS, text)
    if not numbers:
        return text
    return _by_paragraphs(text, _ru_numbers, _ru_centuries)


def _ru_numbers(text):
    text = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}(?!\d))", "", text)               # 1 000 000 → 1000000
    text = re.sub(r"(?<![\d.])\d+(?:\.\d+){2,}(?![\d.]*\d)",                         # версия 1.2.3 → один точка два точка три
                  lambda m: " точка ".join(_ru_words(part) for part in m.group(0).split(".")), text)

    def decade(m):
        return _ru_ordinal(m.group(1), RU_DECADE_ENDINGS[m.group(2)], plural=True)
    text = re.sub(r"(?<!\d)(\d{3,4})-(е|х|ми|м)(?![а-яё])", decade, text)

    def year(m):
        preposition, number, word = (m.group(1) or "").lower(), m.group(2), m.group(3)
        if word in ("г.", "г"):
            case = ("prepositional" if preposition in ("в", "во") else "dative" if preposition in ("к", "по")
                    else "genitive" if preposition in ("с", "со", "до", "от", "после", "около") else "nominative")
            word = {"prepositional": "году", "dative": "году", "genitive": "года", "nominative": "год"}[case]
        else:
            case = "dative" if word == "году" and preposition in ("к", "по") else RU_YEAR_WORDS[word]
        return (m.group(1) + " " if m.group(1) else "") + f"{_ru_ordinal(number, case)} {word}"
    text = _sub(r"(?:\b(в|во|к|по|с|со|до|от|после|около|В|Во|К|По|С|Со|До|От|После|Около)\s+)?(?<!\d)(\d{3,4})\s*"
                r"(году|года|годом|год|г\.|г)(?![а-яё])", year, text)

    text = _sub(r"\b([IVXLC]+)\s*(в\.|веком|веке|века|веку|век)(?![а-яё])", _ru_century, text)

    def ordinal(m):
        case, gender, plural = RU_ORDINAL_ENDINGS[m.group(2)]
        if m.group(2) == "м" and _word_before(m) not in PREPOSITIONAL:
            case = "instrumental"                                  # «пришёл 2-м» → вторым, «во 2-м» → втором
        return _ru_ordinal(m.group(1), case, gender, plural)
    endings = "|".join(sorted(RU_ORDINAL_ENDINGS, key=len, reverse=True))
    text = re.sub(rf"(?<!\d)(\d+)-({endings})(?![а-яё])", ordinal, text)

    for pattern, forms, gender in RU_UNITS:
        def unit(m, forms=forms, gender=gender):
            number = m.group(1)
            if re.search(r"[.,]", number):
                return f"{_ru_words(number)} {forms[1]}"
            return f"{_ru_words(number, gender)} {ru_plural(int(number), *forms)}"
        text = _sub(rf"(?<![\d.,])(\d+(?:[.,]\d+)?)\s?(?:{pattern})(?![а-яё])", unit, text)

    text = re.sub(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])",
                  lambda m: f"{_ru_words(m.group(1))} {_ru_words(m.group(2)) if m.group(2) != '00' else 'ноль ноль'}", text)
    text = re.sub(r"(?<![\d.,])(\d+[.,]\d+)(?![\d.,]*\d)", lambda m: _ru_words(m.group(1)), text)
    return re.sub(r"\d+", lambda m: _ru_words(m.group(0)), text)


def _normalize_en(text, numbers):
    text = _apply_abbreviations(EN_ABBREVIATIONS, text)
    if not numbers:
        return text
    return _by_paragraphs(text, _en_numbers, lambda part: part)


def _en_numbers(text):
    from num2words import num2words

    def words(number):
        return num2words(Decimal(number) if "." in number else int(number), lang="en")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)                                # 1,000,000 → 1000000
    text = re.sub(r"(?<![\d.])\d+(?:\.\d+){2,}(?![\d.]*\d)",                        # version 1.2.3 → one point two point three
                  lambda m: " point ".join(words(p) for p in m.group(0).split(".")), text)
    text = re.sub(r"\$(\d+)\.(\d\d)\b", lambda m: f"{words(m.group(1))} {'dollar' if m.group(1) == '1' else 'dollars'} "
                  f"{words(str(int(m.group(2))))}", text)
    text = re.sub(r"\$(\d+(?:\.\d+)?)", lambda m: f"{words(m.group(1))} {'dollar' if m.group(1) == '1' else 'dollars'}", text)
    text = re.sub(r"€(\d+(?:\.\d+)?)", lambda m: f"{words(m.group(1))} {'euro' if m.group(1) == '1' else 'euros'}", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?%", lambda m: f"{words(m.group(1))} percent", text)
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: num2words(int(m.group(1)), lang="en", to="ordinal"), text)
    text = re.sub(r"(?<![\d.,])(1[1-9]\d\d|20\d\d)(?![\d.,]*\d)",                   # годы: nineteen eighty-four
                  lambda m: num2words(int(m.group(1)), lang="en", to="year"), text)
    text = re.sub(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])",
                  lambda m: f"{words(m.group(1))} {words(m.group(2)) if m.group(2) != '00' else 'o clock'}", text)
    text = re.sub(r"(?<![\d.])(\d+\.\d+)(?![\d.]*\d)", lambda m: words(m.group(1)), text)
    return re.sub(r"\d+", lambda m: words(m.group(0)), text)


def normalize_for_speech(text, replacements="", builtin=True, language=""):
    """Текст так, как его надо прочитать: свои замены, потом (если включено) сокращения и числа словами. Язык
    определяется по буквам текста: кириллица — русские правила, иначе английские."""
    for source, target in parse_replacements(replacements):
        text = re.sub(_word_pattern(source), lambda m, target=target: target, text)
    if not builtin:
        return text
    numbers = has_num2words()
    if language == "Russian":
        return _normalize_ru(text, numbers)
    if language == "English":
        return _normalize_en(text, numbers)
    if language:
        return text                                   # для других языков встроенных правил нет: только свои замены
    cyrillic = len(re.findall("[а-яА-ЯёЁ]", text))
    latin = len(re.findall("[a-zA-Z]", text))
    return _normalize_ru(text, numbers) if cyrillic >= latin else _normalize_en(text, numbers)


@lru_cache(maxsize=4096)
def _speech_cached(text, replacements, builtin, numbers, language):
    return normalize_for_speech(text, replacements, builtin, language)


def speech_text(ctx, text):
    """Текст главы так, как он будет прочитан: свои замены и (если включено) числа и сокращения словами."""
    settings = ctx.settings
    text = re.sub(r"(?m)^#{1,6}[ \t]+", "", text)                 # решётки заголовков Markdown не читаются вслух
    if not settings.normalize_text and not settings.replacements.strip():
        return text
    return _speech_cached(text, settings.replacements, settings.normalize_text, has_num2words(), settings.qwen_language)


def speech_signature(settings):
    """Часть отпечатка главы: правила чтения изменились — недоделанная глава озвучится заново. Пусто, если всё выключено,
    поэтому у тех, кто нормализацией не пользуется, отпечатки глав прежние."""
    if not settings.normalize_text and not settings.replacements.strip():
        return ""
    rules = hashlib.sha1(settings.replacements.encode("utf-8")).hexdigest()[:10]
    language = "" if settings.qwen_language in ("", "Russian", "English") else f":{settings.qwen_language}"   # ru/en: как раньше
    return f"|norm{NORMALIZER_VERSION}:{int(settings.normalize_text)}:{rules}:{int(has_num2words())}{language}"


# ============================================================
# ЧАНКИ ГЛАВЫ: сохранение, дозапуск, склейка
# ============================================================


VOICE_FORMATS = (".wav", ".flac", ".mp3", ".ogg")


def voice_file(voice):
    """Путь к записи голоса; относительный путь считается от папки программы."""
    path = Path(voice.get("path", ""))
    return path if path.is_absolute() else PATHS.base / path


def library_voices(settings):
    """Голоса Qwen из настроек: только корректные записи {name, path, text}, без повторов названий."""
    voices, seen = [], set()
    for item in settings.qwen_voices if isinstance(settings.qwen_voices, list) else []:
        if not isinstance(item, dict):
            continue
        name, path = str(item.get("name", "")).strip(), str(item.get("path", "")).strip()
        if name and path and name.lower() not in seen:
            seen.add(name.lower())
            voices.append({"name": name, "path": path, "text": str(item.get("text", "")).strip()})
    return voices


def current_voice(settings):
    """Выбранный голос Qwen, а если выбранного нет — первый из библиотеки; None, если библиотека пуста."""
    voices = library_voices(settings)
    return next((v for v in voices if v["name"] == settings.qwen_voice), voices[0] if voices else None)


def stored_voice_path(path):
    """Как записать путь к голосу: внутри папки программы — относительно неё (папку можно переносить), иначе полностью."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PATHS.base.resolve()))
    except ValueError:
        return str(path)


def voice_problem(voice, voices, editing=None):
    """Что не так с голосом (пустая строка — всё в порядке): название, запись и её точный текст."""
    name = voice["name"].strip()
    if not name:
        return tr("Укажите название голоса.")
    if any(v["name"].lower() == name.lower() and v["name"] != editing for v in voices):
        return tr("Голос «{0}» уже есть: выберите другое название.").format(name)
    path = voice_file(voice)
    if not voice["path"].strip() or not path.is_file():
        return tr("Файл записи не найден: {0}").format(voice["path"] or "—")
    if path.suffix.lower() not in VOICE_FORMATS:
        return tr("Нужна запись в формате WAV, FLAC, MP3 или OGG.")
    try:
        import soundfile
        duration = soundfile.info(str(path)).duration
    except Exception:
        return tr("Не удалось прочитать запись: {0}").format(path.name)
    if not 1 <= duration <= 120:
        return tr("Запись длится {0:.0f} с, а нужна от нескольких секунд до полуминуты.").format(duration)
    if not voice["text"].strip():
        return tr("Впишите точный текст, который звучит в записи.")
    return ""


def migrate_reference(ctx):
    """Раньше голос был один: ref.wav рядом с программой и его текст. Делаем из него первый голос библиотеки."""
    if library_voices(ctx.settings) or not (PATHS.base / REF_AUDIO_FILENAME).exists():
        return
    voice = {"name": tr("Мой голос"), "path": REF_AUDIO_FILENAME, "text": ctx.settings.ref_text}
    ctx.update(qwen_voices=[voice], qwen_voice=voice["name"])


@lru_cache(maxsize=32)
def _content_digest(path, _size, _mtime, _ctime):          # размер и время файла — часть ключа кэша
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def file_digest(path):
    """Хеш содержимого файла; пересчитывается, только если файл заменили или изменили. Пусто, если файла нет."""
    try:
        stat = os.stat(path)
    except OSError:
        return ""
    return _content_digest(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def voice_signature(ctx, engine_name):
    if engine_name == "qwen":                  # другая запись голоса, её текст, язык или модель — глава озвучится заново
        voice = current_voice(ctx.settings) or {"path": "", "text": ""}
        recording = file_digest(voice_file(voice)) if voice["path"] else ""
        return f"qwen|{effective_qwen_language(ctx.settings)}|{voice['text']}|{recording}|{QWEN_MODEL_ID}"
    if engine_name == "gemini":
        return f"gemini|{ctx.settings.gemini_model}|{ctx.settings.gemini_voice}"
    return "dummy"


@lru_cache(maxsize=8192)
def _digest(max_chars, signature, text):
    return hashlib.sha1(f"{max_chars}\n{signature}\n{text}".encode("utf-8")).hexdigest()


def chapter_digest(ctx, ch, engine_name):
    """Отпечаток главы: меняется, если изменился текст, размер куска или голос. Значение запоминается: таблица окна
    пересчитывается часто, а текст глав не меняется."""
    signature = voice_signature(ctx, engine_name) + speech_signature(ctx.settings)
    return _digest(engine_options(ctx.settings, engine_name)["max_chars"], signature, ch["text"])


def chapters_dir(book_dir):
    return book_dir / "chapters"


def work_dir(book_dir, engine_name, ch):
    return book_dir / "temp_chunks" / engine_name / f"{ch['id']:03d}"


def out_mp3(book_dir, ch):
    return book_dir / f"{ch['name']}.mp3"


def chunk_path(d, i):
    return d / f"{i:04d}.wav"


def chunk_done(path):
    """Быстрая проверка для таблицы окна (она вызывается очень часто): файл есть и не крошечный."""
    return path.exists() and path.stat().st_size > 1024


def chunk_valid(path, sample_rate=None):
    """Строгая проверка при возобновлении: WAV читается, один канал, нужная частота, не пустой и не обрезан
    (размер по заголовку RIFF не больше фактического)."""
    try:
        info = sf.info(str(path))
        with open(path, "rb") as f:
            head = f.read(12)
        declared = int.from_bytes(head[4:8], "little") + 8
        return (head[:4] == b"RIFF" and info.format == "WAV" and info.channels == 1 and info.frames > 0
                and (sample_rate is None or info.samplerate == sample_rate) and declared <= Path(path).stat().st_size)
    except Exception:
        return False


def load_manifest(d, digest):
    f = d / "chunks.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("digest") == digest:
            return data["chunks"]
    except Exception:
        pass
    return None


RESULTS_FILE = "book_manifest.json"
_results_lock = threading.Lock()


@lru_cache(maxsize=8192)
def _text_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def result_fingerprint(ctx, ch, engine_name):
    """Отпечаток готового MP3: текст главы, движок, модель и голос, правила чтения. Размер куска не входит:
    он на звучание не влияет, и главы не должны «устаревать» из-за него."""
    return hashlib.sha1(f"{voice_signature(ctx, engine_name)}{speech_signature(ctx.settings)}\n{_text_hash(ch['text'])}"
                        .encode("utf-8")).hexdigest()


@lru_cache(maxsize=16)
def _read_results(path, _mtime, _size):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        chapters = data.get("chapters", {})
        return chapters if isinstance(chapters, dict) else {}
    except Exception:
        return {}


def load_results(book_dir):
    """{имя MP3: {"fingerprint", "engine"}} из book_manifest.json. Нет файла или он битый — пусто."""
    f = book_dir / RESULTS_FILE
    try:
        st = f.stat()
    except OSError:
        return {}
    return _read_results(str(f), st.st_mtime_ns, st.st_size)


def _write_results(book_dir, mutate):
    with _results_lock:
        chapters = dict(load_results(book_dir))
        mutate(chapters)
        f = book_dir / RESULTS_FILE
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(json.dumps({"version": 1, "chapters": chapters}, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, f)


def record_result(ctx, book_dir, ch, engine_name):
    """Запоминает, из чего сделан только что собранный MP3 главы."""
    entry = {"fingerprint": result_fingerprint(ctx, ch, engine_name), "engine": engine_name}
    _write_results(book_dir, lambda chapters: chapters.__setitem__(f"{ch['name']}.mp3", entry))


def rename_result(book_dir, old_name, new_name):
    """Глава переименована: запись о её MP3 переходит на новое имя файла."""
    if (book_dir / RESULTS_FILE).exists():
        _write_results(book_dir, lambda chapters: chapters.__setitem__(new_name, chapters.pop(old_name))
                       if old_name in chapters else None)


def forget_result(book_dir, ch):
    if (book_dir / RESULTS_FILE).exists():
        _write_results(book_dir, lambda chapters: chapters.pop(f"{ch['name']}.mp3", None))


def result_is_stale(ctx, book_dir, ch):
    """True, если MP3 главы сделан из другого текста или другим голосом. Нет записи (книга озвучена до появления
    манифеста) — считаем актуальным: такому MP3 доверяем, как раньше."""
    entry = load_results(book_dir).get(f"{ch['name']}.mp3")
    if not isinstance(entry, dict) or entry.get("engine") not in ENGINE_SETTINGS:
        return False
    return entry.get("fingerprint") != result_fingerprint(ctx, ch, entry["engine"])


def prepare_chunks(ctx, ch, d, engine_name, events=None):
    """Список кусков главы фиксируется ОДИН раз (это и есть «временная копия главы»).
    Если изменились текст главы, размер куска или голос — старые готовые куски недействительны."""
    digest = chapter_digest(ctx, ch, engine_name)
    chunks = load_manifest(d, digest)
    if chunks is not None:
        return chunks
    if d.exists():
        if events:
            events.log(tr("   ⚠️ Изменились текст главы, размер куска или голос — готовые куски "
                          "этой главы сброшены."))
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    chunks = split_text_into_chunks(speech_text(ctx, ch["text"]), engine_options(ctx.settings, engine_name)["max_chars"])
    tmp = d / "chunks.json.tmp"
    tmp.write_text(json.dumps({"digest": digest, "chunks": chunks}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, d / "chunks.json")
    return chunks


def write_chunk_wav(path, audio, sample_rate):
    if audio is None or len(audio) == 0:
        raise RuntimeError(tr("движок вернул пустое аудио"))
    tmp = path.with_name(path.name + ".tmp")
    sf.write(str(tmp), audio, sample_rate, subtype="PCM_16", format="WAV")
    os.replace(tmp, path)                  # файл появляется целиком или не появляется вовсе


def synthesize_chapter(engine, d, chunks):
    """Озвучивает только недостающие куски (если движок позволяет — несколько одновременно). True — глава готова к
    склейке, False — ошибка (готовое сохранено). Остановка пользователем поднимает Cancelled (или KeyboardInterrupt),
    его обрабатывает сервис."""
    events, total = engine.events, len(chunks)
    for stale in d.glob("*.tmp"):
        stale.unlink()
    for i in range(1, total + 1):                    # повреждённый кусок удаляем: иначе он вечно считался бы готовым
        wav = chunk_path(d, i)
        if wav.exists() and not chunk_valid(wav, engine.sample_rate):
            events.log(tr("   ⚠️ Кусок {0} повреждён, озвучу его заново.").format(i))
            wav.unlink()
    todo = [(i, text) for i, text in enumerate(chunks, 1) if not chunk_done(chunk_path(d, i))]
    if total - len(todo):
        events.log(tr("   🔄 Уже готово кусков: {0}/{1}. Продолжаю с недостающих.").format(total - len(todo), total))
    spent, lock, abort = {"seconds": 0.0, "audio": 0.0, "done": total - len(todo)}, threading.Lock(), threading.Event()

    def one(i, text):
        """Озвучивает один кусок и записывает его на диск."""
        if abort.is_set():
            return
        engine.token.raise_if_cancelled()
        events.emit("chunk_started", index=i, total=total, text=tr("   ↳ кусок {0}/{1} ({2} симв.) ...").format(i, total,
                                                                                                                len(text)))
        t0 = time.perf_counter()
        audio = engine.synth(text)
        write_chunk_wav(chunk_path(d, i), audio, engine.sample_rate)
        dur, el, note = len(audio) / engine.sample_rate, time.perf_counter() - t0, engine.last_note
        with lock:
            spent["seconds"] += el
            spent["audio"] += dur
            spent["done"] += 1
            done = spent["done"]
        text = tr("     готово: {0:.1f} с, аудио {1:.1f} с, RTF {2:.2f}").format(el, dur, el / dur if dur else 0)
        events.emit("chunk_done", index=i, total=total, done=done, seconds=el, audio_seconds=dur, note=note,
                    text=text + (f" · {note}" if note else ""))

    def report_error(i, error):
        events.emit("error", text=tr("\n❌ Ошибка на куске {0}/{1}: {2}: {3}").format(i, total, type(error).__name__, error))
        events.log(tr("🛑 Остановлено. Готовые куски сохранены — при следующем запуске продолжу "
                      "с этого места."))

    if engine.max_parallel <= 1:
        for i, text in todo:
            try:
                one(i, text)
            except (Cancelled, KeyboardInterrupt):
                raise
            except Exception as e:
                report_error(i, e)
                return False
    else:
        failure, cancelled = None, False
        with ThreadPoolExecutor(max_workers=engine.max_parallel) as pool:
            futures = {pool.submit(one, i, text): i for i, text in todo}
            try:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Cancelled:
                        cancelled = True
                    except Exception as e:
                        if failure is None:
                            failure = (futures[future], e)
                        abort.set()
            except KeyboardInterrupt:
                abort.set()
                engine.token.cancel()
                raise
        if failure:
            report_error(*failure)
            return False
        if cancelled:
            raise Cancelled()
    if spent["audio"]:
        startup = engine.__dict__.pop("startup_note", "")          # загрузка и прогрев модели — в итоге первой главы
        events.log(tr("   ⏱ Итог по главе: озвучено за {0}, звука {1}, средний RTF {2:.2f}").format(
            fmt_duration(spent['seconds']), fmt_duration(spent['audio']), spent['seconds'] / spent['audio']) + startup)
    return True


def prune_empty(*folders):
    """Удаляет перечисленные служебные папки, если они опустели."""
    for folder in folders:
        try:
            folder.rmdir()
        except OSError:
            pass


def chapter_display_name(heading, title):
    """Название главы для людей: «Глава 3 · Старый маяк», без повтора, если название уже есть в заголовке."""
    heading, title = display_heading(heading), display_heading(title)
    return heading if not title or title in heading else f"{heading} · {title}"


def assemble_chapter(d, total, sample_rate, pause_sec, bitrate, out_path, tags=None, cover=None):
    missing = [i for i in range(1, total + 1) if not chunk_done(chunk_path(d, i))]
    if missing:
        raise RuntimeError(tr("не хватает кусков: {0}").format(missing[:10]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_wav = out_path.with_name(out_path.stem + ".tmp.wav")
    tmp_mp3 = out_path.with_name(out_path.stem + ".tmp.mp3")
    silence = np.zeros(int(sample_rate * pause_sec), dtype=np.float32)
    with sf.SoundFile(str(tmp_wav), "w", samplerate=sample_rate, channels=1, subtype="PCM_16", format="WAV") as out:
        for i in range(1, total + 1):
            data, sr = sf.read(str(chunk_path(d, i)), dtype="float32")
            if sr != sample_rate:
                raise RuntimeError(tr("кусок {0}: частота {1}, ожидалась {2}").format(i, sr, sample_rate))
            out.write(data)
            if i < total and len(silence):
                out.write(silence)
    proc = subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(tmp_wav), *_cover_args(cover),
         "-ac", "1", "-b:a", bitrate, *(arg for key, value in (tags or {}).items() for arg in ("-metadata", f"{key}={value}")),
         "-id3v2_version", "3", "-f", "mp3", str(tmp_mp3)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    tmp_wav.unlink(missing_ok=True)
    if proc.returncode != 0 or not tmp_mp3.exists():
        tmp_mp3.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg: {proc.stderr.strip()[:300]}")
    os.replace(tmp_mp3, out_path)
    shutil.rmtree(d, ignore_errors=True)


# ============================================================
# КНИГА, СОСТОЯНИЕ, ТАБЛИЦА
# ============================================================


RESERVED_FOLDERS = {PATHS.venv.name.lower(), PATHS.cache.name.lower(), PATHS.models.name.lower()}
WINDOWS_DEVICE_NAMES = {"con", "prn", "aux", "nul"} | {f"{p}{i}" for p in ("com", "lpt") for i in range(1, 10)}


def safe_folder_name(title):
    """Название книги → имя папки без символов, недопустимых в Windows, и без конфликтов с папками скрипта."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", title)
    name = re.sub(r"\s+", " ", name).strip(" .")[:80].rstrip(" .")
    if not name:
        return "Книга"
    if name.lower() in RESERVED_FOLDERS or name.lower().split(".")[0] in WINDOWS_DEVICE_NAMES:
        name += tr(" (книга)")
    return name


SHORT_SECTION_CHARS = 300        # раздел книги короче этого (титул части, посвящение) приклеивается к следующему


class BookFormatError(ValueError):
    """Файл EPUB или FB2 не удалось разобрать (повреждён или защищён DRM)."""


def book_suffix(path):
    name = Path(path).name.lower()
    return ".fb2.zip" if name.endswith(".fb2.zip") else Path(path).suffix.lower()


def book_stem(path):
    name = Path(path).name
    suffix = book_suffix(path)
    return name[:-len(suffix)] if suffix and name.lower().endswith(suffix) else Path(path).stem


class _HtmlText(HTMLParser):
    """Глава EPUB (XHTML) → абзацы. Заголовки h1 и h2 становятся строками «# …», сноски и служебное отбрасываются."""
    BLOCKS = {"p", "div", "br", "li", "tr", "section", "blockquote", "h3", "h4", "h5", "h6", "pre", "dd", "dt"}
    SKIPPED = {"script", "style", "head", "sup"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.current, self.skipping, self.heading = [], [], [], False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in self.SKIPPED or "noteref" in (attrs.get("epub:type") or ""):
            self.skipping.append(tag)
        elif tag in ("h1", "h2"):
            self._flush()
            self.heading = True
        elif tag in self.BLOCKS:
            self._flush()

    def handle_endtag(self, tag):
        if self.skipping:
            if tag == self.skipping[-1]:
                self.skipping.pop()
        elif tag in ("h1", "h2") or tag in self.BLOCKS:
            self._flush()

    def handle_data(self, data):
        if not self.skipping:
            self.current.append(data)

    def close(self):
        super().close()
        self._flush()

    def _flush(self):
        text = " ".join("".join(self.current).split())
        self.current = []
        if text:
            self.parts.append("# " + text if self.heading else text)
        self.heading = False


def _join_sections(paragraphs):
    """Абзацы и заголовки «# …» → текст книги. Короткий раздел с заголовком (титул части, посвящение) приклеивается к
    следующему: его заголовок входит в название следующей главы, а текст — в её начало."""
    sections = []
    for paragraph in paragraphs:
        if paragraph.startswith("# "):
            sections.append([paragraph[2:].strip(), []])
        elif sections:
            sections[-1][1].append(paragraph)
        else:
            sections.append([None, [paragraph]])
    merged, carried_titles, carried_text = [], [], []
    for index, (heading, body) in enumerate(sections):
        if heading and sum(map(len, body)) < SHORT_SECTION_CHARS and index + 1 < len(sections):
            carried_titles.append(heading)
            carried_text += body
            continue
        if heading and carried_titles:
            heading = ". ".join(carried_titles + [heading])
        merged.append((heading, carried_text + body))
        carried_titles, carried_text = [], []
    lines = []
    for heading, body in merged:
        lines += ([f"# {heading}"] if heading else []) + body
    return "\n\n".join(lines).strip() + "\n"


MAX_ARCHIVE_FILES = 20_000                  # файлов внутри EPUB / FB2.ZIP
MAX_XML_BYTES = 64 * 1024 * 1024            # один читаемый файл (OPF, XHTML, FB2)
MAX_ARCHIVE_READ = 256 * 1024 * 1024        # всего прочитанного из архива
MAX_BOOK_CHARS = 100_000_000                # текст книги в памяти (обычная книга — 1–5 млн символов)


def _too_big():
    return BookFormatError(tr("файл слишком велик для книги (возможно, это «бомба» в архиве), открытие отменено"))


class _BookZip:
    """ZIP книги с ограничениями: число файлов проверяется при открытии, размер каждого читаемого файла и общий
    объём — до чтения (zipfile отдаёт не больше объявленного размера, так что объявленному можно верить)."""

    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        if len(self.zip.infolist()) > MAX_ARCHIVE_FILES:
            self.zip.close()
            raise _too_big()
        self.read_total = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.zip.close()

    def namelist(self):
        return self.zip.namelist()

    def read(self, name):
        size = self.zip.getinfo(name).file_size
        self.read_total += size
        if size > MAX_XML_BYTES or self.read_total > MAX_ARCHIVE_READ:
            raise _too_big()
        return self.zip.read(name)


def _epub_toc(archive, items, base):
    """Оглавление EPUB: файл главы → её название (EPUB 3 nav или EPUB 2 toc.ncx)."""
    toc = {}
    for item in items.values():
        href = posixpath.normpath(posixpath.join(base, item.get("href") or ""))
        folder = posixpath.dirname(href)
        try:
            if "nav" in (item.get("properties") or ""):
                for link in ElementTree.fromstring(archive.read(href)).iter():
                    if link.tag.endswith("}a") and link.get("href"):
                        target = posixpath.normpath(posixpath.join(folder, link.get("href").split("#")[0]))
                        toc.setdefault(target, " ".join("".join(link.itertext()).split()))
            elif item.get("media-type") == "application/x-dtbncx+xml":
                for point in ElementTree.fromstring(archive.read(href)).iter():
                    label, content = point.find(".//{*}text"), point.find("{*}content")
                    if point.tag.endswith("navPoint") and label is not None and content is not None:
                        target = posixpath.normpath(posixpath.join(folder, content.get("src", "").split("#")[0]))
                        toc.setdefault(target, " ".join((label.text or "").split()))
        except (KeyError, ElementTree.ParseError):
            pass
    return toc


def _read_epub(path):
    with _BookZip(path) as archive:
        rootfile = ElementTree.fromstring(archive.read("META-INF/container.xml")).find(".//{*}rootfile").get("full-path")
        opf = ElementTree.fromstring(archive.read(rootfile))
        base = posixpath.dirname(rootfile)
        title = " ".join((opf.findtext(".//{*}title") or "").split())
        items = {item.get("id"): item for item in opf.iter() if item.tag.endswith("}item")}
        toc = _epub_toc(archive, items, base)
        paragraphs = []
        for ref in (node for node in opf.iter() if node.tag.endswith("}itemref")):
            item = items.get(ref.get("idref"))
            if (ref.get("linear") == "no" or item is None or "html" not in (item.get("media-type") or "")
                    or "nav" in (item.get("properties") or "")):
                continue
            href = posixpath.normpath(posixpath.join(base, item.get("href")))
            parser = _HtmlText()
            parser.feed(archive.read(href).decode("utf-8", "replace"))
            parser.close()
            if toc.get(href) and not any(part.startswith("# ") for part in parser.parts[:3]):   # и для пустого титула части
                parser.parts.insert(0, "# " + toc[href])
            paragraphs += parser.parts
    return _join_sections(paragraphs), title


def _read_fb2(path):
    data = _fb2_data(path)
    root = ElementTree.fromstring(data)

    def name(node):
        return node.tag.rsplit("}", 1)[-1]

    def text(node):
        for link in node.iter():
            if name(link) == "a" and any(key.endswith("type") and value == "note" for key, value in link.attrib.items()):
                link.text = ""                                   # ссылка на сноску — не читаем
        return " ".join("".join(node.itertext()).split())

    info = next((node for node in root.iter() if name(node) == "title-info"), None)
    title = text(next(node for node in info if name(node) == "book-title")) if info is not None and any(
        name(node) == "book-title" for node in info) else ""
    paragraphs = []

    def walk(node, headings):
        for child in node:
            kind = name(child)
            if kind == "title" and headings:
                heading = text(child)
                if heading:
                    paragraphs.append("# " + heading)
            elif kind == "section":
                walk(child, True)
            elif kind in ("poem", "stanza", "cite", "epigraph", "title"):
                walk(child, False)
            elif kind in ("p", "subtitle", "v", "text-author"):
                line = text(child)
                if line:
                    paragraphs.append(line)
    bodies = [node for node in root if name(node) == "body" and node.get("name") not in ("notes", "comments")]
    if bodies:
        walk(bodies[0], False)                           # заголовок самой книги в начале — не глава, а вступление
    return _join_sections(paragraphs), title


COVER_TYPES = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png"}


def _plain(text):
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(text or "")).split())


def _epub_metadata(path):
    with _BookZip(path) as archive:
        rootfile = ElementTree.fromstring(archive.read("META-INF/container.xml")).find(".//{*}rootfile").get("full-path")
        opf = ElementTree.fromstring(archive.read(rootfile))
        base = posixpath.dirname(rootfile)
        meta = {"author": _plain(opf.findtext(".//{*}creator")), "description": _plain(opf.findtext(".//{*}description"))}
        metas = [node for node in opf.iter() if node.tag.endswith("}meta")]
        named = {node.get("name"): node.get("content", "") for node in metas if node.get("name")}
        meta["series"] = named.get("calibre:series", "")
        meta["series_index"] = named.get("calibre:series_index", "").removesuffix(".0")
        for node in metas:                                    # EPUB 3: belongs-to-collection и group-position
            if node.get("property") == "belongs-to-collection" and not meta["series"]:
                meta["series"] = _plain(node.text)
            if node.get("property") == "group-position" and not meta["series_index"]:
                meta["series_index"] = _plain(node.text)
        items = [node for node in opf.iter() if node.tag.endswith("}item")]
        cover = next((i for i in items if "cover-image" in (i.get("properties") or "")), None)
        if cover is None and named.get("cover"):
            cover = next((i for i in items if i.get("id") == named["cover"]), None)
        if cover is not None and cover.get("media-type") in COVER_TYPES:
            data = archive.read(posixpath.normpath(posixpath.join(base, cover.get("href"))))
            meta["cover"] = (data, COVER_TYPES[cover.get("media-type")])
    return meta


def _fb2_data(path):
    """Байты FB2 (из .fb2 или .fb2.zip) с теми же ограничениями размера, что и при чтении текста."""
    if book_suffix(path) == ".fb2.zip":
        with _BookZip(path) as archive:
            return archive.read(next(name for name in archive.namelist() if name.lower().endswith(".fb2")))
    if Path(path).stat().st_size > MAX_XML_BYTES:
        raise _too_big()
    return Path(path).read_bytes()


def _xml_name(node):
    return node.tag.rsplit("}", 1)[-1]


def _xml_child(parent, tag, deep=False):
    """Первый узел с таким именем (без пространства имён) среди детей или, при deep, во всём поддереве."""
    if parent is None:
        return None
    return next((node for node in (parent.iter() if deep else parent) if _xml_name(node) == tag), None)


def _xml_text(node, separator=" "):
    return " ".join(separator.join(node.itertext()).split())


def _fb2_cover(root, info):
    """(байты, тип) обложки из <coverpage>, если она есть и в понятном формате."""
    image = _xml_child(_xml_child(info, "coverpage"), "image", deep=True)
    target = next((v for k, v in (image.attrib.items() if image is not None else []) if k.endswith("href")), "").lstrip("#")
    binary = next((node for node in root if _xml_name(node) == "binary" and node.get("id") == target), None) if target else None
    if binary is not None and binary.get("content-type") in COVER_TYPES:
        return base64.b64decode("".join((binary.text or "").split())), COVER_TYPES[binary.get("content-type")]
    return None


def _fb2_metadata(path):
    root = ElementTree.fromstring(_fb2_data(path))
    info = _xml_child(root, "title-info", deep=True)
    if info is None:
        return {}
    meta = {}
    author = _xml_child(info, "author")
    if author is not None:
        parts = {_xml_name(part): _xml_text(part, "") for part in author}
        meta["author"] = " ".join(p for p in (parts.get("first-name"), parts.get("middle-name"), parts.get("last-name")) if p)
    sequence = _xml_child(info, "sequence")
    if sequence is not None:
        meta["series"], meta["series_index"] = sequence.get("name", ""), sequence.get("number", "")
    annotation = _xml_child(info, "annotation")
    if annotation is not None:
        meta["description"] = _xml_text(annotation)
    cover = _fb2_cover(root, info)
    if cover:
        meta["cover"] = cover
    return meta


@lru_cache(maxsize=4)
def _book_metadata_cached(path, _mtime, _size, _ctime=0):
    suffix = book_suffix(path)
    try:
        if suffix == ".epub":
            return _epub_metadata(path)
        if suffix in (".fb2", ".fb2.zip"):
            return _fb2_metadata(path)
    except Exception:
        pass                                          # нет сведений — не беда: книга всё равно откроется
    return {}


def book_metadata(path):
    """Автор, серия, номер, описание и обложка ((данные, расширение)) из EPUB или FB2; у TXT — пусто."""
    try:
        stat = Path(path).stat()
    except OSError:
        return {}
    return _book_metadata_cached(str(path), stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns)


def book_cover(book):
    """Файл обложки для тегов: своя картинка, картинка из книги (кладётся в .cache) или None."""
    if book.state.cover == "-":
        return None
    if book.state.cover:
        path = Path(book.state.cover)
        return path if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png") else None
    cover = book_metadata(book.path).get("cover")
    if not cover:
        return None
    out = PATHS.cache / f"cover{cover[1]}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(cover[0])
    return out


def chapter_tags(book, chapter):
    """Теги MP3 главы: название главы и книги, номер, автор, серия, описание."""
    state = book.state
    series = f"{state.series} {state.series_index}".strip() if state.series else ""
    tags = {"title": chapter_display_name(chapter["heading"], chapter["title"]), "album": book.title,
            "track": f"{chapter['id']}/{len(book.chapters)}", "genre": "Audiobook", "artist": state.author,
            "album_artist": state.author, "grouping": series, "comment": state.description[:1000]}
    return {key: value for key, value in tags.items() if value}


def _cover_args(cover):
    return ["-i", str(cover), "-map", "0:a", "-map", "1:v", "-c:v", "copy", "-disposition:v", "attached_pic",
            "-metadata:s:v", "title=Album cover", "-metadata:s:v", "comment=Cover (front)"] if cover else ["-map", "0:a"]


def retag_mp3(path, tags, cover=None):
    """Заменяет теги и обложку готового MP3 без перекодирования звука."""
    tmp = path.with_name(f"{path.stem}.{os.getpid()}-{threading.get_ident()}.tags.mp3")   # у каждого запуска свой файл
    proc = subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(path), *_cover_args(cover), "-c:a", "copy",
         "-map_metadata", "-1", *(arg for key, value in tags.items() for arg in ("-metadata", f"{key}={value}")),
         "-id3v2_version", "3", "-f", "mp3", str(tmp)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg: {proc.stderr.strip()[:300]}")
    os.replace(tmp, path)


def _read_text_limited(path, block=1_000_000):
    """TXT читается кусками и не дальше лимита: огромный файл отклоняется, не попав целиком в память."""
    if Path(path).stat().st_size > MAX_BOOK_CHARS * 4:          # даже по 4 байта на символ это больше лимита
        raise _too_big()
    parts, count = [], 0
    with open(path, encoding="utf-8-sig", newline="") as source:
        while chunk := source.read(block):
            count += len(chunk)
            if count > MAX_BOOK_CHARS:
                raise _too_big()
            parts.append(chunk)
    return "".join(parts)


@lru_cache(maxsize=8)
def _read_book_cached(path, _mtime, _size, _ctime=0):    # время, размер и ctime файла — часть ключа кэша
    suffix = book_suffix(path)
    if suffix in (".epub", ".fb2", ".fb2.zip"):
        try:
            text, title = _read_epub(path) if suffix == ".epub" else _read_fb2(path)
        except (KeyError, StopIteration, AttributeError, zipfile.BadZipFile, ElementTree.ParseError) as e:
            raise BookFormatError(tr("файл {0} повреждён или защищён (DRM): {1}").format(Path(path).name, e)) from e
    else:
        text, title = _read_text_limited(path).replace("\r\n", "\n").replace("\r", "\n"), ""
    if len(text) > MAX_BOOK_CHARS:
        raise _too_big()
    return text, title


def read_book(path):
    """(текст книги, название из самой книги или пустая строка). TXT читается как есть, EPUB и FB2 превращаются в текст
    с заголовками глав «# …». Результат запоминается, пока файл не изменился."""
    stat = Path(path).stat()
    return _read_book_cached(str(path), stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns)


def read_book_text(path):
    return read_book(path)[0]


def text_digest(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


GENERIC_TITLES = {                      # «безымянные» слова: такое имя файла не годится в название книги
    "book", "text", "untitled", "new", "document", "doc", "file", "new text document", "new document",
    "книга", "текст", "новый", "документ", "файл", "безымянный", "без названия", "новый текстовый документ",
    "новый документ",
}


def is_placeholder_title(name):
    """Имя файла, которое не годится в название книги: одни цифры («1», «02») или шаблонное («Новый текстовый документ»).
    Цифры в настоящем названии (например, «1984») пользователь вводит сам: так мы уверены, что оно осознанное."""
    core = re.sub(r"[\W\d_]+", " ", name.lower()).strip()
    return not core or core in GENERIC_TITLES


def file_title(path):
    """Название книги по имени файла. Для безымянного файла (1.txt, одни цифры, «Новый текстовый документ») — None:
    название надо спросить у пользователя."""
    if path.name.lower() == INPUT_BOOK_FILENAME.lower() or is_placeholder_title(book_stem(path)):
        return None
    return safe_folder_name(book_stem(path))


def remembered_title(ctx, path, text):
    """(название, актуально ли): что вводили раньше для этого файла и менялось ли с тех пор его содержимое."""
    saved = ctx.settings.book_titles.get(str(Path(path).resolve()))
    if not saved:
        return None, False
    return saved["title"], saved.get("sha") == text_digest(text)


def remember_title(ctx, path, text, title):
    ctx.update(book_titles={**ctx.settings.book_titles, str(Path(path).resolve()): {"title": title, "sha": text_digest(text)}})


def apply_book_profile(ctx, book):
    """Открыли книгу: её язык, голос Qwen, модель и голос Gemini становятся текущими. Возвращает то, что поменялось."""
    state = book.state
    changes = {"qwen_language": book.language(), "qwen_language_auto": not state.language}
    if state.gemini_model in GEMINI_MODEL_IDS:
        changes["gemini_model"] = state.gemini_model
    if state.gemini_voice in (name for name, _ in GEMINI_VOICES):
        changes["gemini_voice"] = state.gemini_voice
    if state.voice and any(v["name"] == state.voice for v in library_voices(ctx.settings)):
        changes["qwen_voice"] = state.voice
    changes = {name: value for name, value in changes.items() if getattr(ctx.settings, name) != value}
    if changes:
        ctx.update(**changes)
    return changes


def remember_book_profile(ctx, book):
    """Что сейчас выбрано (голоса и модель), запоминается в книге; язык книги хранится в ней самой."""
    chosen = current_voice(ctx.settings)
    book.state.voice = chosen["name"] if chosen else book.state.voice
    book.state.gemini_model, book.state.gemini_voice = ctx.settings.gemini_model, ctx.settings.gemini_voice


class Book:
    def __init__(self, ctx, path, title):
        self.path = Path(path)
        self.text = read_book_text(self.path)
        self.title = title
        self.dir = PATHS.base / self.title
        self.store = RecordStore(chapters_dir(self.dir) / "split.json", BookState)
        self.state = self.store.load()
        self.detected_language = detect_language(self.text)    # быстро: смотрит только часть текста
        if not self.state.info_loaded:                          # первый раз: автор, серия, описание из файла книги
            meta = book_metadata(self.path)
            for name in ("author", "series", "series_index", "description"):
                setattr(self.state, name, meta.get(name, ""))
            self.state.info_loaded = True
        self.resplit()

    def language(self):
        """Язык книги: выбранный вручную или определённый по тексту (пусто, если не определился)."""
        return self.state.language or self.detected_language

    def save(self):
        self.store.save(self.state)

    def resplit(self):
        self.chapters, self.profile_used = self._auto_split()
        state, lines = self.state, self.text.split("\n")
        state.cuts = sorted({c for c in state.cuts if isinstance(c, int) and not isinstance(c, bool) and 0 <= c < len(lines)}) \
            if isinstance(state.cuts, list) else []                         # испорченные разрезы просто отбрасываем
        state.titles = {str(k): v.strip() for k, v in state.titles.items() if isinstance(v, str) and v.strip()} \
            if isinstance(state.titles, dict) else {}
        if (state.cuts or state.titles) and state.cuts_digest and state.cuts_digest != text_digest(self.text):
            state.cuts, state.titles, state.cuts_digest = [], {}, ""          # файл книги изменился: правки к нему не подходят
            self.edits_reset = True
        starts = chapter_start_lines(lines, self.chapters)
        if state.cuts:
            known = {start: (ch["heading"], ch["title"]) for ch, start in zip(self.chapters, starts or [], strict=False)}
            self.chapters = number_chapters(chapters_from_cuts(lines, state.cuts, state.titles, known))
            self.profile_used = "manual"
            return
        for ch, start in zip(self.chapters, starts or [], strict=False):
            ch["line"] = start
            if str(start) in state.titles:
                ch["title"] = ch["heading"] = state.titles[str(start)]
        if state.titles and starts:
            number_chapters(self.chapters)                                   # своё название — новое имя файла

    def _auto_split(self):
        if self.state.profile == "auto" and book_suffix(self.path) != ".txt":    # у EPUB и FB2 главы размечены в самой книге
            chapters, used = split_book(self.text, "markdown", self.state.params, self.title)
            if used == "markdown" and len(chapters) >= 2:
                return chapters, used
        return split_book(self.text, self.state.profile, self.state.params, self.title)

    # ---- ручная правка глав
    def editable(self):
        """Можно ли резать и склеивать главы: для каждой известна строка начала."""
        return all("line" in ch for ch in self.chapters)

    def profile_key(self):
        return "manual" if self.state.cuts else self.state.profile

    def _remember_edit(self):
        self.state.cuts_digest = text_digest(self.text)
        self.resplit()

    def chapter_lines(self, ch_id):
        """Строки главы, с которых может начаться новая глава: (номер строки, начало абзаца)."""
        index = next(k for k, ch in enumerate(self.chapters) if ch["id"] == ch_id)
        lines = self.text.split("\n")
        start = self.chapters[index]["line"]
        end = self.chapters[index + 1]["line"] if index + 1 < len(self.chapters) else len(lines)
        return [(k, " ".join(lines[k].split())[:90]) for k in range(start + 1, end) if lines[k].strip()]

    def rename_chapter(self, ch_id, title):
        ch = next(ch for ch in self.chapters if ch["id"] == ch_id)
        self.state.titles[str(ch["line"])] = title
        self._remember_edit()

    def split_chapter(self, line, title=""):
        self.state.cuts = sorted({ch["line"] for ch in self.chapters} | {line})
        if title:
            self.state.titles[str(line)] = title
        self._remember_edit()

    def merge_with_next(self, ch_id):
        index = next(k for k, ch in enumerate(self.chapters) if ch["id"] == ch_id)
        following = self.chapters[index + 1]["line"]
        self.state.cuts = [ch["line"] for ch in self.chapters if ch["line"] != following]
        self.state.titles.pop(str(following), None)
        self._remember_edit()

    def reset_edits(self):
        self.state.cuts, self.state.titles, self.state.cuts_digest = [], {}, ""
        self.resplit()

    def started(self):
        """Есть ли у книги озвученные главы или начатые куски (тогда резать и склеивать главы нельзя)."""
        files, work = scan_book(self)
        return any(name.endswith(".mp3") for name in files) or any(work.values())

    def export_chapters(self):
        d = chapters_dir(self.dir)
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.txt"):
            old.unlink()
        for ch in self.chapters:
            (d / f"{ch['name']}.txt").write_text(ch["text"], encoding="utf-8", newline="\n")


def scan_book(book):
    """Что уже лежит в папке книги: имена файлов и папки незаконченных глав по движкам. Один просмотр вместо тысяч
    проверок «есть ли файл»: таблица окна обновляется часто."""
    def names(folder):
        try:
            return set(os.listdir(folder))
        except OSError:
            return set()
    return names(book.dir), {eng: names(book.dir / "temp_chunks" / eng) for eng in ENGINE_SETTINGS}


def chapter_progress(ctx, book, ch, scan=None):
    """(done | stale | part | new, готово кусков, всего кусков, движок) для главы. stale — MP3 есть, но сделан из
    другого текста или другим голосом."""
    files, work = scan or scan_book(book)
    if f"{ch['name']}.mp3" in files:
        return ("stale" if result_is_stale(ctx, book.dir, ch) else "done"), 0, 0, ""
    best = None
    for eng in ENGINE_SETTINGS:
        if f"{ch['id']:03d}" not in work[eng]:
            continue
        d = work_dir(book.dir, eng, ch)
        chunks = load_manifest(d, chapter_digest(ctx, ch, eng))
        if chunks:
            n = sum(chunk_done(chunk_path(d, i)) for i in range(1, len(chunks) + 1))
            if n and (best is None or n > best[0]):
                best = (n, len(chunks), eng)
    if best:
        return "part", best[0], best[1], best[2]
    return "new", 0, 0, ""


def status_label(kind, done, total, engine):
    if kind == "done":
        return tr("✅ готова")
    if kind == "stale":
        return tr("⚠️ устарела")
    if kind == "part":
        return tr("⏸ {0}/{1} кусков ({2})").format(done, total, engine)
    return tr("· не начата")


def chapter_status(ctx, book, ch):
    """(done | part | new, подпись для консоли)."""
    kind, done, total, engine = chapter_progress(ctx, book, ch)
    return kind, status_label(kind, done, total, engine)


def summary(service, book):
    kinds = [chapter_status(service.ctx, book, ch)[0] for ch in book.chapters]
    return kinds.count("done") + kinds.count("stale"), kinds.count("part"), kinds.count("new")


# ============================================================
# ЛИМИТЫ И РАСХОДЫ GEMINI
# ============================================================

def pacific_today():
    """Дата по тихоокеанскому времени: по нему Google сбрасывает суточные лимиты."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).date()
    except Exception:
        return (datetime.now(timezone.utc) - timedelta(hours=8)).date()


def fmt_duration(seconds):
    minutes = int(round(seconds / 60))
    return tr("{0} ч {1:02d} мин").format(minutes // 60, minutes % 60) if minutes >= 60 else tr("{0} мин").format(minutes)


def ru_plural(n, one, few, many):
    """Слово по числу: 1 глава, 2 главы, 5 глав."""
    n = abs(int(n)) % 100
    if 11 <= n <= 14:
        return many
    last = n % 10
    return one if last == 1 else few if 2 <= last <= 4 else many


PLURAL_WORDS = {         # слово: (русские формы для 1, 2-4 и 5+; английские для 1 и остальных)
    "chapter": (("глава", "главы", "глав"), ("chapter", "chapters")),
    "chunk": (("кусок", "куска", "кусков"), ("chunk", "chunks")),
    "request": (("запрос", "запроса", "запросов"), ("request", "requests")),
}


def plural_word(n, noun):
    """Слово с правильной формой для числа на языке интерфейса: 1 глава, 2 главы, 5 глав / 1 chapter, 2 chapters."""
    russian, english = PLURAL_WORDS[noun]
    if _language == "ru":
        return ru_plural(n, *russian)
    return english[0] if abs(int(n)) == 1 else english[1]


def display_heading(text):
    """Названия глав, которые программа придумала сама («Начало», «Часть 3»), показываем на языке интерфейса, а в имена
    файлов они всегда попадают одинаковыми, чтобы прогресс не терялся при смене языка."""
    text = re.sub(r"^#+\s*", "", text)                           # заголовок Markdown показываем без решёток
    if text == "Начало":
        return tr("Начало")
    m = re.fullmatch(r"Часть (\d+)", text)
    return tr("Часть {0}").format(m.group(1)) if m else text


def fmt_int(number):
    """Число с разделителем тысяч: 8 283 (по-русски) или 8,283 (по-английски)."""
    return f"{number:,}".replace(",", " " if _language == "ru" else ",")


def eta_text(seconds):
    """Сколько осталось: «≈ 1 ч 20 мин» или «меньше минуты»."""
    return tr("меньше минуты") if seconds < 60 else f"≈ {fmt_duration(seconds)}"


def gemini_cost(model, in_tokens, out_tokens, on):
    """Стоимость по ценам платного тарифа в долларах (None, если цена модели неизвестна)."""
    price = None
    for since, p_in, p_out in GEMINI_PRICES.get(model, []):
        if since <= on:
            price = (p_in, p_out)
    return None if price is None else (in_tokens * price[0] + out_tokens * price[1]) / 1e6


def fmt_cost(total):
    return f"≈ ${total['cost']:.2f}" + (tr(" (без моделей с неизвестной ценой)") if total["unknown"] else "")


@lru_cache(maxsize=4096)
def _cached_chunks(text, max_chars):
    return tuple(split_text_into_chunks(text, max_chars))


def chunk_list(text, max_chars):
    """Куски главы; результат запоминается: окно пересчитывает таблицу часто, а текст глав не меняется."""
    return list(_cached_chunks(text, max_chars))


def pending_chunk_stats(ctx, book, chapters, engine_name):
    """(сколько кусков осталось озвучить, сколько в них символов) по указанным главам."""
    count = chars = 0
    started = scan_book(book)[1][engine_name]
    for ch in chapters:
        d = work_dir(book.dir, engine_name, ch)
        saved = load_manifest(d, chapter_digest(ctx, ch, engine_name)) if f"{ch['id']:03d}" in started else None
        chunks = saved or chunk_list(speech_text(ctx, ch["text"]), engine_options(ctx.settings, engine_name)["max_chars"])
        for i, chunk in enumerate(chunks, 1):
            if not (saved and chunk_done(chunk_path(d, i))):
                count += 1
                chars += len(chunk)
    return count, chars


# ============================================================
# СЕРВИС: единая точка входа для консоли и окна
# ============================================================


class BookLock:
    """Замок папки книги на время озвучки: вторая копия программы не начнёт работать с теми же кусками и не испортит
    их. Операционная система снимает замок сама, даже если программа упала."""

    def __init__(self, path):
        self.path, self.handle = Path(path), None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self.handle = handle
        return True

    def release(self):
        if not self.handle:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        self.handle.close()
        self.handle = None


@dataclass
class Estimate:
    """Прикидка расхода Gemini на выбранные главы."""
    model: str
    chapters: int
    requests: int
    chunk_chars: int
    seconds: float
    paid: bool
    cost: float | None                # по ценам платного тарифа; None, если цена модели неизвестна
    later_date: date | None           # дата, с которой цены меняются
    later_cost: float | None
    daily_limit: int
    used_today: int
    left_today: int
    days: int | None                  # сколько дней уйдёт на бесплатном тарифе при заданном суточном лимите


@dataclass
class ChapterRow:
    id: int
    heading: str
    title: str
    chars: int
    chunks: int
    kind: str                         # done | stale | part | new
    label: str                        # подпись статуса для консоли
    done_chunks: int = 0
    total_chunks: int = 0
    engine: str = ""                  # каким движком начата глава (для kind == part)


def status_chip(row):
    """Статус главы для таблицы окна: цветной кружок и короткий текст."""
    if row.kind == "done":
        return tr("● готова")
    if row.kind == "stale":
        return tr("⚠ устарела")
    if row.kind == "part":
        return f"◐ {row.done_chunks}/{row.total_chunks}" + (f" · {row.engine}" if row.engine else "")
    return tr("○ не начата")


@dataclass
class Preflight:
    """Что нужно знать перед запуском: какие главы будут озвучены, что начато другим движком, сколько это стоит."""
    pending: list
    conflicts: list                   # [(глава, подпись)]: уже начаты другим движком, текущий начнёт их с нуля
    estimate: Estimate | None         # расчёт Gemini (для других движков None)


def partial_conflicts(ctx, book, pending, engine_name):
    found = []
    for ch in pending:
        kind, label = chapter_status(ctx, book, ch)
        if kind == "part" and f"({engine_name})" not in label:
            found.append((ch, label))
    return found


class AudiobookService:
    """Фасад (Facade): всё, что умеет программа, одним набором методов. Сам ничего не печатает и не спрашивает:
    ход работы идёт событиями в EventBus, остановка — через CancellationToken."""

    def __init__(self, ctx, events=None):
        self.ctx = ctx
        self.events = events or EventBus()

    def open_book(self, path, title):
        book = Book(self.ctx, path, title)
        book.profile_changes = apply_book_profile(self.ctx, book)
        return book

    def pending(self, book, ids):
        """Главы из ids, у которых ещё нет готового MP3."""
        files = scan_book(book)[0]
        return [ch for ch in book.chapters if ch["id"] in ids and f"{ch['name']}.mp3" not in files]

    def chapter_rows(self, book, engine_name):
        chunk_chars = engine_options(self.ctx.settings, engine_name)["max_chars"]
        rows, scan = [], scan_book(book)
        for ch in book.chapters:
            kind, done, total, engine = chapter_progress(self.ctx, book, ch, scan)
            chunks = len(chunk_list(speech_text(self.ctx, ch["text"]), chunk_chars))
            rows.append(ChapterRow(ch["id"], ch["heading"], ch["title"], len(ch["text"]), chunks, kind,
                                   status_label(kind, done, total, engine), done, total, engine))
        return rows

    def retag(self, book, ids=None):
        """Сведения о книге поменялись: обновляет теги и обложку уже готовых MP3. Возвращает, сколько файлов обновлено."""
        if not ensure_ffmpeg():
            return 0
        cover, done = book_cover(book), 0
        for chapter in book.chapters:
            if ids is not None and chapter["id"] not in ids:
                continue
            path = out_mp3(book.dir, chapter)
            if path.exists():
                retag_mp3(path, chapter_tags(book, chapter), cover)
                done += 1
        return done

    def reset_chapters(self, book, ids):
        """Удаляет готовый MP3 и недоделанные куски выбранных глав (у всех движков), чтобы озвучить их заново."""
        folders = []
        for ch in book.chapters:
            if ch["id"] in ids:
                out_mp3(book.dir, ch).unlink(missing_ok=True)
                forget_result(book.dir, ch)
                for engine_name in ENGINE_SETTINGS:
                    shutil.rmtree(work_dir(book.dir, engine_name, ch), ignore_errors=True)
                    folders += [work_dir(book.dir, engine_name, ch).parent]
        prune_empty(*folders, book.dir / "temp_chunks")

    def readiness(self, engine_name):
        """(что можно починить установкой, что нужно сделать руками) для выбранного движка."""
        return engine_problems(self.ctx, engine_name)

    def estimate(self, book, chapters, engine_name="gemini"):
        cfg = self.ctx.settings
        requests_left, chars = pending_chunk_stats(self.ctx, book, chapters, engine_name)
        seconds = chars / SPEECH_CHARS_PER_SEC
        in_tokens, out_tokens = chars / TEXT_CHARS_PER_TOKEN, seconds * AUDIO_TOKENS_PER_SEC
        today = pacific_today()
        later = next((since for since, _, _ in GEMINI_PRICES.get(cfg.gemini_model, []) if since > today), None)
        used = self.ctx.usage.requests_today(cfg.gemini_model)
        limit = cfg.gemini_daily_limit
        left = max(0, limit - used) if limit else 0
        days = None
        if limit and not cfg.gemini_paid:
            days = 1 if requests_left <= left else 1 + -(-(requests_left - left) // limit)
        return Estimate(
            model=cfg.gemini_model, chapters=len(chapters), requests=requests_left,
            chunk_chars=engine_options(cfg, engine_name)["max_chars"], seconds=seconds, paid=cfg.gemini_paid,
            cost=gemini_cost(cfg.gemini_model, in_tokens, out_tokens, today), later_date=later,
            later_cost=gemini_cost(cfg.gemini_model, in_tokens, out_tokens, later) if later else None,
            daily_limit=limit, used_today=used, left_today=left, days=days)

    def preflight(self, book, engine_name, ids):
        pending = self.pending(book, ids)
        estimate = self.estimate(book, pending, engine_name) if engine_name == "gemini" and pending else None
        return Preflight(pending, partial_conflicts(self.ctx, book, pending, engine_name), estimate)

    def install_missing(self, engine_name, token=None):
        """Доустанавливает в .venv то, чего не хватает: ffmpeg и пакеты движка. Ход работы идёт событиями, в конце
        install_done (ok, restart). True — теперь всё есть."""
        events, ok, restart = self.events, False, False
        try:
            ok = ensure_ffmpeg() or install_engine("ffmpeg", events, token)
            if ok and self.readiness(engine_name)[0]:
                ok = install_engine(engine_name, events, token)
        except RestartRequired:
            ok, restart = True, True
        except Exception as e:
            events.emit("error", text=tr("❌ Установка не удалась: {0}: {1}").format(type(e).__name__, e))
            ok = False
        events.emit("install_done", ok=ok, restart=restart, text="")
        return ok

    def _assemble(self, book, chapter, engine_name, folder, count, sample_rate, failed):
        """Склейка главы в MP3 (в отдельном потоке: следующая глава озвучивается, пока кодируется эта)."""
        options = engine_options(self.ctx.settings, engine_name)
        events = self.events
        try:
            tags = chapter_tags(book, chapter)
            assemble_chapter(folder, count, sample_rate, options["pause"], options["bitrate"], out_mp3(book.dir, chapter),
                             tags=tags, cover=book_cover(book))
        except Exception as e:
            failed.set()
            events.emit("error", text=tr("❌ Склейка не удалась: {0}\n   Куски сохранены — запусти снова.").format(e))
            return False
        try:
            record_result(self.ctx, book.dir, chapter, engine_name)
        except Exception as e:                   # MP3 готов; без записи глава просто считается актуальной
            events.log(f"   ⚠️ book_manifest.json: {type(e).__name__}: {e}")
        events.emit("chapter_done", chapter=chapter["id"], path=str(out_mp3(book.dir, chapter)),
                    text=tr("✅ Готово: {0}").format(out_mp3(book.dir, chapter).name))
        return True

    def run(self, book, engine_name, ids, token=None):
        """Озвучивает главы ids: всё нужное должно быть готово заранее (см. readiness)."""
        return self._run(book, engine_name, ids, token)

    def _report_stopped(self):
        self.events.emit("run_finished", ok=False, stopped=True, text=tr("\n🛑 Остановлено. Готовые куски сохранены — при "
                                                                        "следующем запуске продолжу."))

    def _run(self, book, engine_name, ids, token=None):
        """Озвучивает выбранные главы по порядку; True, если все готовы. Ничего не спрашивает: всё нужное должно быть
        готово заранее (см. readiness). Ход работы идёт событиями, остановка — токеном; прогресс сохраняется всегда."""
        ctx, events = self.ctx, self.events
        token = token or CancellationToken()
        pending = self.pending(book, ids)
        if not pending:
            stale = [ch["id"] for ch in book.chapters if ch["id"] in ids and result_is_stale(ctx, book.dir, ch)]
            text = (tr("✅ Нечего озвучивать: выбранные главы уже готовы.") if not stale else
                    tr("⚠️ Готовые MP3 глав {0} сделаны из другого текста или другим голосом. "
                       "Чтобы озвучить их заново, выбери «Озвучить заново».").format(", ".join(map(str, stale[:10]))))
            events.emit("run_finished", ok=True, stopped=False, text=text)
            return True
        fix, manual = self.readiness(engine_name)
        problems = fix + manual
        if not ensure_ffmpeg():
            problems.append(tr("ffmpeg не найден (нужен для сборки MP3)"))
        if problems:
            for problem in problems:
                events.emit("error", text=f"❌ {problem}")
            return False

        if token.cancelled():                                  # «Стоп» нажат, пока проверялись движок и ffmpeg
            self._report_stopped()
            return False
        options = engine_options(ctx.settings, engine_name)
        book.state.engine = engine_name
        book.save()
        chapters_word = plural_word(len(pending), "chapter")
        events.log(tr("\nК озвучке: {0} {1} · движок {2} · кусок {3} симв. · пауза {4} с").format(len(pending), chapters_word,
                                                                                                  engine_name,
                                                                                                  options['max_chars'],
                                                                                                  options['pause']))
        lock = BookLock(book.dir / ".lock")
        if not lock.acquire():
            events.emit("error", text=tr("❌ Эта книга уже озвучивается другим запуском программы. Закройте его "
                                         "или дождитесь окончания."))
            return False
        try:
            engine = make_engine(ctx, engine_name, events, token)
            if hasattr(engine, "warm_up"):
                engine.warm_up()
        except Exception as e:
            lock.release()
            events.emit("error", text=tr("❌ Не удалось запустить движок {0}: {1}: {2}").format(engine_name, type(e).__name__, e))
            return False

        if token.cancelled():                                  # «Стоп» нажат во время загрузки модели: на куски не переходим
            engine.close()
            lock.release()
            self._report_stopped()
            return False
        stopped = broken = False
        assembler, assemblies, assembly_failed = ThreadPoolExecutor(max_workers=1), [], threading.Event()
        try:
            for index, ch in enumerate(pending, 1):
                if assembly_failed.is_set():
                    broken = True
                    break
                events.emit("chapter_started", chapter=ch["id"], index=index, total=len(pending),
                            text=tr("\n🎙️ Глава {0}/{1}: {2} {3}").format(ch['id'], len(book.chapters),
                                                                          display_heading(ch['heading']),
                                                                          display_heading(ch['title'])))
                d = work_dir(book.dir, engine_name, ch)
                chunks = prepare_chunks(ctx, ch, d, engine_name, events)
                if not synthesize_chapter(engine, d, chunks):
                    broken = True
                    break
                events.log(tr("🔗 Склеиваю главу {0} в MP3...").format(ch['id']))
                assemblies.append(assembler.submit(self._assemble, book, ch, engine_name, d, len(chunks),
                                                   engine.sample_rate, assembly_failed))
        except (Cancelled, KeyboardInterrupt):
            token.cancel()
            stopped = True
        finally:
            assembler.shutdown(wait=True)                      # склейка уже озвученных глав дозавершается всегда
            assembled = [future.result() for future in assemblies]
            engine.close()
            lock.release()
            prune_empty(*(book.dir / "temp_chunks" / name for name in ENGINE_SETTINGS), book.dir / "temp_chunks")
        if stopped:
            self._report_stopped()
            return False
        if broken or not all(assembled):
            return False
        events.emit("run_finished", ok=True, stopped=False, text=tr("\n🎉 Готово. MP3 лежат в {0}").format(book.dir))
        return True


def format_estimate(est):
    """Строки для консоли по результату AudiobookService.estimate."""
    lines = [tr("\nРасчёт для Gemini (прикидка) · модель {0}").format(est.model),
             tr("  Осталось: {0} {1}, {2} {3} (кусок {4} симв.), звука около {5}").format(est.chapters,
                                                                                          plural_word(est.chapters, "chapter"),
                                                                                          est.requests,
                                                                                          plural_word(est.requests, "request"),
                                                                                          est.chunk_chars,
                                                                                          fmt_duration(est.seconds))]
    if est.paid:
        if est.cost is None:
            lines.append(tr("  Тариф платный, но цена этой модели в скрипте не задана: смотри {0}").format(GEMINI_PRICING_URL))
        else:
            lines.append(tr("  Тариф платный: стоимость около ${0:.2f}").format(est.cost))
            if est.later_date and est.later_cost is not None:
                lines.append(tr("  С {0:%d.%m.%Y}, по открытым данным, цены вырастут: тогда около "
                                "${1:.2f}").format(est.later_date, est.later_cost))
    else:
        worth = tr(" (по платному тарифу это было бы около ${0:.2f})").format(est.cost) if est.cost is not None else ""
        lines.append(tr("  Тариф бесплатный: деньги не списываются{0}, но действует лимит "
                        "запросов.").format(worth))
        if est.daily_limit:
            lines.append(tr("  Лимит {0} запросов в сутки на модель: сегодня использовано {1}, "
                            "осталось {2}.").format(est.daily_limit, est.used_today, est.left_today))
            lines.append(tr("  Потребуется примерно дней: {0} (лимит обновляется в полночь по "
                            "тихоокеанскому времени).").format(est.days))
        else:
            lines.append(tr("  Суточный лимит не задан: укажи его в «Настройках» (цифру смотри в AI "
                            "Studio), и я посчитаю дни."))
    lines.append(tr("  Это оценка: точные счётчики и баланс показывает только Google AI "
                    "Studio / Cloud Billing."))
    return lines


def estimate_line(est):
    """Одна строка с расчётом Gemini для окна."""
    parts = [tr("Осталось: {0} {1}").format(est.chapters, plural_word(est.chapters, "chapter")),
             f"{est.requests} {plural_word(est.requests, "request")}",
             tr("около {0} звука").format(fmt_duration(est.seconds))]
    if est.paid and est.cost is not None:
        parts.append(tr("примерно ${0:.2f}").format(est.cost))
    elif not est.paid and est.days:
        parts.append(tr("при лимите {0} в сутки: дней {1}").format(est.daily_limit, est.days))
    return " · ".join(parts)


def usage_report_lines(ctx):
    """Строки отчёта о лимитах и расходах Gemini (для консоли и окна)."""
    cfg = ctx.settings
    lines = [tr("Gemini · тариф: {0} · модель: {1}").format(tr("платный") if cfg.gemini_paid else tr("бесплатный"),
                                                            cfg.gemini_model)]
    today = pacific_today()
    for title, prefix in ((tr("Сегодня ({0}, тихоокеанское время)").format(today), str(today)),
                          (tr("В этом месяце"), today.strftime("%Y-%m")), (tr("За всё время"), "")):
        total = ctx.usage.summarize(prefix)
        lines.append(tr("  {0}: запросов {1}, звука {2}, по ценам платного тарифа {3}").format(title, total['requests'],
                                                                                               fmt_duration(total['seconds']),
                                                                                               fmt_cost(total)))
    if cfg.gemini_daily_limit and not cfg.gemini_paid:
        used = ctx.usage.requests_today(cfg.gemini_model)
        lines.append(tr("  Лимит этой модели на сегодня: {0} из {1}, осталось {2}").format(used, cfg.gemini_daily_limit,
                                                                                           max(0, cfg.gemini_daily_limit - used)))
    lines.append(tr("  Считаются только запросы этого скрипта. Остаток по лимитам Google (с "
                    "учётом других программ) и баланс счёта"))
    lines.append(tr("  показывают только AI Studio и Google Cloud Billing: через API их "
                    "получить нельзя."))
    return lines


# ============================================================
# ЗАПУСК ОЗВУЧКИ
# ============================================================


# ============================================================
# МЕНЮ
# ============================================================


# ============================================================
# ОКНО (Tkinter): презентер без tkinter и окно, которое только рисует
# ============================================================


NEEDS_TITLE_HINT = "Введите название книги: по нему называется папка с результатом."
UNSUPPORTED_BOOK_FORMATS = {".pdf", ".doc", ".docx", ".rtf", ".mobi", ".azw3", ".odt", ".djvu"}
PALETTES = {           # цвета подписей и статусов; фон и основной текст берутся из темы sv-ttk
    "light": {"accent": "#005fb8", "muted": "#6b6b6b", "danger": "#c42b1c", "ok": "#0f7b0f", "warn": "#9d5d00",
              "plain": "#4a4a4a", "border": "#d0d0d0"},
    "dark": {"accent": "#60cdff", "muted": "#a6a6a6", "danger": "#ff7b72", "ok": "#6ccb5f", "warn": "#e8b04a",
             "plain": "#d8d8d8", "border": "#555555"},
}
APP_TITLE = "Foxy AudioBook Maker"
APP_VERSION = "1.0.17"                    # версия программы: совпадает с выпуском (тегом) на GitHub
APP_ICON_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAASFBMVEUPbMMAXrgAbbIAZeQAX7kAAP8AP78AX7lemtKnx+YA"
    "f3+BsNvO4PEAAH8A//8AXbf8/f4AAAAEYroAWLG81ewAW7Z7rNpCiMqt5UiSAAAAGHRSTlP+XhAJ0gEEnP//Av//AgH+/wD+"
    "Df8n///DPB7iAAAB5ElEQVR42u2Y25KDIAyGw8lTu0sJat//TTegVdkKovRqZ3Nh7Zh8pskA+QtmNcs7k2GKW7V+gzXcXYSw"
    "Bya493sDdMp88VpKfWyyZnxFwPJ6JhEAMwwAJCHsFlCZoQbU2YYgBwpaAJVhCPqUAbKJAD7/AVGfNCSCnQCduhDvCL4OBBBK"
    "QtwrzkZphSIAFSAev1x26+DKAEZZidH4Z9s+dfwxNwqs4Yn4B1mcQClw4KaGKKB3gD4OqAkgjLwMoDJSDXjiNx4BtAPYnedT"
    "97aASD+HfcDcvQ0g1s8IQI99C/TSBYAa2n7MzQD16OLCDNzN+P6qPcDs7d0XgEc+8I2QAty2gFseYK59OoOgHSEA59onARi0"
    "41cG0MBxBt5pDzB5jUeAMWjHBkCt9k4uhzhgccIdgF+7tHiTgJfTP+Af8IcBhYupeDmXbyif2NJKN9Xybf0zB0vx0VZ+uH7i"
    "eD81YIjCEee7ZMiiQbF0zLM0aLI4oHGAJj1okoZKjLrN/d4kS6CKhu2alJMb961MEeLjvradG/evCRanm9gsOOiDXZA8sEie"
    "SXTh+fgqkH3ynOx7ibZVeFrmdWeO2CK/2obC00tXy+oM5YtO+grzil/FtxJee/PhwIRwjqt+h42sr4TJsuBvgh/VTPlnA7u2"
    "5QAAAABJRU5ErkJggg=="
)
GAP = 12                                 # единый вертикальный отступ между блоками окна
DROP_HEIGHTS = (96, 44)                  # зона перетаскивания: пустая и сжатая (когда книга уже выбрана)
DEFAULT_WINDOW_SIZE = (800, 740)
MIN_WINDOW_SIZE = (780, 720)


def mix_colors(first, second, share):
    """Смесь двух цветов вида #rrggbb: share — доля второго (от 0 до 1)."""
    a = [int(first[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(second[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * share):02x}" for x, y in zip(a, b, strict=True))


def valid_color(value, default):
    return str(value) if re.fullmatch(r"#[0-9a-fA-F]{6}", str(value)) else default      # Tk отдаёт цвет объектом, нужна строка


def system_is_dark():
    """Включена ли тёмная тема Windows. На других системах считаем, что светлая."""
    if os.name != "nt":
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def resolve_theme(choice):
    """'light' или 'dark' по настройке: auto следует за системой."""
    return choice if choice in ("light", "dark") else ("dark" if system_is_dark() else "light")


def fit_geometry(saved, screen_width, screen_height):
    """Размер окна всегда стандартный (края окна не тянут), от сохранённого берётся только положение; пустая строка —
    данных нет или они негодные. Положение за краями экрана (например, был второй монитор) сдвигается на основной."""
    m = re.fullmatch(r"(\d+)x(\d+)(?:([+-]\d+)([+-]\d+))?", saved or "")
    if not m:
        return ""
    width = min(DEFAULT_WINDOW_SIZE[0], screen_width)
    height = min(DEFAULT_WINDOW_SIZE[1], screen_height)
    if m[3] is None:
        return f"{width}x{height}"
    x = max(0, min(int(m[3]), screen_width - width))
    y = max(0, min(int(m[4]), screen_height - height))
    return f"{width}x{height}+{x}+{y}"


def set_console_visible(visible):
    """Прячет или показывает окно консоли на Windows: при запуске двойным щелчком по start.bat оно не нужно за окном
    программы. Больше нигде ничего не делает."""
    if os.name != "nt":
        return
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 5 if visible else 0)
    except Exception:
        pass


def start_thread(work):
    threading.Thread(target=work, daemon=True).start()


class MainPresenter:
    """Презентер (MVP): вся логика окна без tkinter, поэтому её можно проверять тестами. Окно (view) только рисует и
    сообщает о действиях пользователя; события из фоновой работы попадают в очередь и переносятся в окно методом pump."""

    def __init__(self, service, view, runner=None, clock=None):
        self.service, self.ctx, self.view = service, service.ctx, view
        self.runner = runner or start_thread
        self.clock = clock or time.monotonic
        self.queue = queue.Queue()
        service.events.subscribe(lambda kind, data: self.queue.put((kind, data)))
        self.path = self.book = self.token = self.run_started = self.finish_info = None
        self.text = self.title_text = ""
        self.needs_input = self.running = self.resume_after_install = self.restart_needed = False
        self.engine = DEFAULT_ENGINE
        self.selected = set()                              # главы, выбранные в таблице; пусто — озвучиваются все
        self.chapter_index = self.chapter_total = self.chunks_done = self.chunks_total = self.current_chapter = 0
        self.previewing, self.preview_engine, self.retagging, self.preview_token = False, None, False, None
        self.preview_idle_since = 0.0
        self.idle_since, self.memory_released = time.monotonic(), True      # при запуске освобождать нечего
        self.hint = ("", False)                            # подсказка под названием: русский текст-ключ и «тревожная» ли
        self.last_progress = None                          # что показано в полосе прогресса: (доля, подпись, ожидание)
        self._show_voices()

    # ---- язык
    def change_theme(self, theme):
        """Переключатель ☀ | ☾: запоминает выбор и сразу перекрашивает окно."""
        self.ctx.update(theme=theme)
        self.view.apply_theme(theme)

    def change_language(self, language):
        """Переключатель RU | EN: запоминает выбор, окно пересобирается на новом языке."""
        if language == _language and self.ctx.settings.language == language:
            return
        self.ctx.update(language=language)
        set_language(resolve_language(language))
        self.view.rebuild()

    def restore_view(self):
        """Окно пересобрано (смена языка): возвращаем в него книгу, название, выбор и ход работы."""
        if self.book:
            self.view.set_book_name(self.path.name)
            self.view.set_title(self.title_text, tr(self.hint[0]), self.hint[1])
            self.view.set_engine(self.engine)
            self.view.set_profile(self.book.profile_key())
        self._refresh()
        self._refresh_controls()
        if self.last_progress:
            self._progress(*self.last_progress)

    def _progress(self, fraction, text, busy=False):
        self.last_progress = (fraction, text, busy)
        if busy:
            self.view.set_busy(text)
        else:
            self.view.set_progress(fraction, text)

    # ---- книга и название
    def open_file(self, path):
        if self.running:
            self.view.notify(tr("Сейчас идёт работа. Дождитесь окончания или остановите её, потом "
                                "откройте другую книгу."))
            return
        path = Path(path)
        if path.suffix.lower() in UNSUPPORTED_BOOK_FORMATS:
            self.view.show_error(tr("Формат {0} не поддерживается. Поддерживаются TXT, EPUB и FB2. "
                                    "Переведите книгу в EPUB или .txt "
                                    "(UTF-8), например в бесплатной программе Calibre.").format(path.suffix))
            return
        try:
            text = read_book_text(path)
        except BookFormatError as e:
            self.view.show_error(tr("Не удалось открыть книгу: {0}").format(e))
            return
        except (OSError, UnicodeDecodeError) as e:
            self.view.show_error(tr("Не удалось прочитать файл «{0}»: {1}\nНужен текст в кодировке UTF-8.").format(path.name, e))
            return
        self.path, self.text, self.selected = path, text, set()
        title, hint, needs_input = self._suggest_title()
        self.title_text, self.needs_input, self.hint = title, needs_input, (hint, needs_input)
        self._load_book(title or safe_folder_name(path.stem))
        self.view.set_book_name(path.name)
        self.view.set_title(title, tr(hint), needs_input)
        self._refresh()

    def _suggest_title(self):
        """(название, подсказка, нужно ли ввести вручную)."""
        saved, fresh = remembered_title(self.ctx, self.path, self.text)
        if saved and fresh:
            return saved, "Название запомнено для этого файла. Всё верно?", False
        if saved:
            return saved, "Файл изменился с прошлого раза. Название прежнее?", False
        inside = safe_folder_name(read_book(self.path)[1]) if read_book(self.path)[1] else ""
        if inside:
            return inside, "Название взято из самой книги. Всё верно?", False
        title = file_title(self.path)
        if title:
            return title, "Название взято из имени файла. Всё верно?", False
        return "", NEEDS_TITLE_HINT, True

    def _load_book(self, title):
        self.book = self.service.open_book(self.path, title)
        self.engine = self.book.state.engine or self.engine
        self.view.set_engine(self.engine)
        self.view.set_profile(self.book.profile_key())
        if getattr(self.book, "edits_reset", False):
            self.view.notify(tr("Файл книги изменился: ручные правки глав сброшены."))
        changes = getattr(self.book, "profile_changes", {})
        if "qwen_voice" in changes and self.engine == "qwen":     # книга озвучивалась этим голосом: продолжаем им же
            self.view.notify(tr("Выбран голос «{0}»: им уже озвучивалась эта книга.").format(changes["qwen_voice"]))

    def title_edited(self, text):
        self.title_text = text.strip()
        if self.needs_input:
            if self.title_text:
                self._set_hint("Название введено. Проверьте, что оно верное.", False)
            else:
                self._set_hint(NEEDS_TITLE_HINT, True)
        self._refresh_controls()

    def _set_hint(self, key, alarm):
        """Подсказка под названием: хранится по-русски (ключ перевода) и переводится при показе."""
        self.hint = (key, alarm)
        self.view.set_hint(tr(key), alarm)

    def title_committed(self):
        """Название закончили вводить: показываем главы и прогресс именно для этой папки."""
        title = safe_folder_name(self.title_text) if self.title_text else ""
        if title and self.book and title != self.book.title and not self.running:
            self._load_book(title)
            self._refresh()

    # ---- настройки запуска
    def set_engine(self, name):
        if name != self.engine:
            self._drop_preview_engine()
        self.engine = name
        if self.book:
            self._refresh()
        else:
            self._show_voices()

    # ---- голос
    def voice_choices(self):
        """Что показать в списке «Голос»: голоса Qwen из библиотеки и пункт «Добавить» или готовые голоса Gemini."""
        if self.engine == "gemini":
            names = [name for name, _ in GEMINI_VOICES]
            current = self.ctx.settings.gemini_voice
            return [f"{name} · {note}" for name, note in GEMINI_VOICES], names.index(current) if current in names else 0
        if self.engine != "qwen":
            return [], -1
        voices, chosen = library_voices(self.ctx.settings), current_voice(self.ctx.settings)
        return [v["name"] for v in voices] + [tr("＋ Добавить голос…")], voices.index(chosen) if chosen else len(voices)

    def choose_voice(self, index):
        if self.running:
            return
        if self.engine == "gemini":
            self.ctx.update(gemini_voice=GEMINI_VOICES[index][0])
        elif self.engine == "qwen":
            voices = library_voices(self.ctx.settings)
            if index >= len(voices):                       # «Добавить голос…»: выбор остаётся прежним
                self._show_voices()
                self.open_settings()
                return
            self.ctx.update(qwen_voice=voices[index]["name"])
        if self.book:
            remember_book_profile(self.ctx, self.book)
            self._refresh()
        else:
            self._show_voices()

    # ---- язык книги
    def language_choices(self):
        """Список «Язык книги»: первым «Авто (как определилось)», дальше языки Qwen."""
        detected = self.book.detected_language if self.book else ""
        items = [tr("Авто ({0})").format(language_label(detected))] + [tr(LANGUAGE_NAMES[name]) for name in QWEN_LANGUAGES]
        chosen = self.book.state.language if self.book else ""
        return items, (QWEN_LANGUAGES.index(chosen) + 1 if chosen in QWEN_LANGUAGES else 0)

    def choose_language(self, index):
        if self.running or not self.book:
            return
        self.book.state.language = QWEN_LANGUAGES[index - 1] if index > 0 else ""
        self.book.save()                                       # выбор не должен пропасть, если озвучку не начали
        self.ctx.update(qwen_language=self.book.language(), qwen_language_auto=index == 0)
        self._refresh()

    # ---- проба голоса
    def preview(self):
        """Кнопка «Прослушать»: первые два предложения выбранной главы текущим движком, голосом и языком. Пока проба
        озвучивается или звучит, на кнопке «Стоп»: нажатие её прерывает."""
        if self.previewing or self.view.sound_playing():
            self.stop_preview()
            return
        if self.running:
            return
        if not self._credentials_ready(self.engine):
            return
        fixable, manual = self.service.readiness(self.engine)
        if fixable or manual:
            self.view.show_error("\n".join([tr("Движок пока не готов:")] + [f"• {p}" for p in fixable + manual]))
            return
        text = tr(TEST_PHRASE)
        if self.book:
            chosen = [ch for ch in self.book.chapters if ch["id"] in self.selected] or self.book.chapters
            text = preview_sentences(chosen[0]["text"]) or text
        self.previewing = True
        self.preview_token = token = CancellationToken()
        self._refresh_controls()
        self.view.notify(tr("Озвучиваю пробу…"))
        engine_name, cached = self.engine, self.preview_engine

        def job():
            try:
                path, engine = make_voice_preview(self.ctx, engine_name, text, self.service.events, token, cached)
                self.queue.put(("preview_done", {"path": path, "engine": engine, "token": token}))
            except Exception as e:
                self.queue.put(("preview_done", {"error": f"{type(e).__name__}: {e}", "token": token}))
        self.runner(job)

    def stop_preview(self):
        """«Стоп»: прерывает озвучку пробы (её результат будет отброшен) и останавливает звук."""
        if self.previewing:
            self.preview_token.cancel()
            self.previewing = False
            self.view.notify(tr("Проба остановлена."))
        self.view.stop_sound()
        self._refresh_controls()

    def _release_when_idle(self):
        """Через IDLE_RELEASE_SECONDS без озвучки и проб программа отдаёт системе неиспользуемую память (один раз)."""
        busy = self.running or self.previewing or self.preview_engine is not None or self.view.sound_playing()
        if busy:
            self.memory_released = False
            self.idle_since = time.monotonic()
        elif not self.memory_released and time.monotonic() - self.idle_since > IDLE_RELEASE_SECONDS:
            self.memory_released = True
            release_memory()

    def _unload_idle_preview(self):
        """Модель Qwen, загруженная для проб, выгружается после PREVIEW_UNLOAD_SECONDS без проб и звука."""
        if (self.preview_engine is not None and not self.previewing and not self.view.sound_playing()
                and time.monotonic() - self.preview_idle_since > PREVIEW_UNLOAD_SECONDS):
            self._drop_preview_engine()
            self.view.notify(tr("Модель Qwen выгружена из видеопамяти: {0} с без проб.").format(PREVIEW_UNLOAD_SECONDS))

    def _drop_preview_engine(self):
        """Модель Qwen для проб освобождается перед озвучкой и при смене движка: видеопамяти нужно место."""
        if self.preview_engine is not None:
            self.preview_engine.close()
            self.preview_engine = None

    def _show_voices(self):
        items, index = self.voice_choices()
        self.view.set_voices(items, index)

    def set_profile(self, key):
        if not self.book or self.running:
            return
        if key == "manual":                                    # «Вручную» появляется само, когда главы правят
            if not self.book.state.cuts:
                self.view.notify(tr("Чтобы изменить главы, щёлкните по главе в таблице правой кнопкой мыши."))
            self.view.set_profile(self.book.profile_key())
            return
        rows = self.service.chapter_rows(self.book, self.engine)
        question = tr("У этой книги уже есть озвученные главы. Смена разбивки может сдвинуть "
                      "нумерацию.\nВсё равно изменить?")
        if any(r.kind != "new" for r in rows) and not self.view.ask_yes(tr("Смена разбивки"), question):
            self.view.set_profile(self.book.profile_key())
            return
        params = {}
        if key == "custom":
            regex = self.view.ask_text(tr("Свой способ разбивки"),
                                       tr("Регулярное выражение для строки-заголовка (например ^Глава \\d+):"))
            if not regex:
                self.view.set_profile(self.book.profile_key())
                return
            params["regex"] = regex
        elif key == "size":
            value = self.view.ask_text(tr("Равные части"), tr("Размер части в символах [{0}]:").format(SPLIT_PART_SIZE))
            params["size"] = int(value) if value and value.isdigit() else SPLIT_PART_SIZE
        self.book.state.profile, self.book.state.params = key, params
        self.book.state.cuts = []                              # другой способ разбиения заменяет ручные правки
        self.book.save()
        self.book.resplit()
        self.selected = set()                                  # нумерация глав могла измениться
        self._refresh()

    def open_settings(self):
        if self.running:
            self.view.notify(tr("Пока идёт работа, настройки менять нельзя."))
            return
        self.view.show_settings(lambda dialog: SettingsPresenter(
            self.ctx, dialog, on_saved=self.settings_changed, extra_report=self._estimate_report))

    def remember_geometry(self, geometry):
        """Окно закрывают: запоминаем его размер и положение до следующего запуска."""
        self.ctx.update(window_geometry=geometry)

    def settings_changed(self):
        """Настройки сохранены: размер куска, голос и тариф меняют расчёт и статусы глав."""
        if self.book:
            remember_book_profile(self.ctx, self.book)
            self._refresh()
        else:
            self._show_voices()

    def _estimate_report(self):
        if not self.book:
            return []
        pending = self.service.pending(self.book, {ch["id"] for ch in self.book.chapters})
        return format_estimate(self.service.estimate(self.book, pending)) if pending else []

    # ---- выбор глав в таблице
    def target_ids(self):
        """Какие главы озвучивать: выбранные в таблице, а если ничего не выбрано — все."""
        everything = {ch["id"] for ch in self.book.chapters}
        return (self.selected & everything) or everything

    def select(self, ids):
        """В таблице выбрали главы (Ctrl или Shift + клик): озвучиваться будут только они."""
        if self.running or not self.book:
            return
        self.selected = set(ids)
        self.view.set_estimate(self._estimate_text())
        self._refresh_controls()

    def row_actions(self, ids):
        """Что доступно в меню строки: озвучить выбранные, открыть MP3 (одна готовая глава), озвучить заново."""
        if self.running or not self.book:
            return {"start": False, "open": False, "redo": False}
        kinds = [chapter_progress(self.ctx, self.book, ch)[0] for ch in self.book.chapters if ch["id"] in ids]
        return {"start": self.can_start(), "open": kinds in (["done"], ["stale"]), "redo": any(kind != "new" for kind in kinds)}

    def open_chapter(self, ids):
        """Двойной щелчок или меню: открывает MP3 готовой главы."""
        chapters = [ch for ch in self.book.chapters if ch["id"] in ids] if self.book else []
        if len(chapters) != 1 or self.running:
            return
        path = out_mp3(self.book.dir, chapters[0])
        if path.exists():
            self.view.open_path(path)
        else:
            self.view.notify(tr("Эта глава ещё не озвучена: файла MP3 пока нет."))

    # ---- ручная правка глав (правый щелчок по таблице)
    def _chapter(self, ids):
        if not self.book or self.running or not ids:
            return None
        return next((ch for ch in self.book.chapters if ch["id"] == min(ids)), None)

    def _layout_changed(self):
        self.book.save()
        self.selected = set()
        self.view.set_profile(self.book.profile_key())
        self._refresh()

    def _may_cut(self):
        if not self.book.editable():
            self.view.notify(tr("При разбиении на равные части главы вручную не правятся: выберите другой способ."))
            return False
        if self.book.started():
            self.view.notify(tr("Разделять и объединять главы можно, пока книга не начата. Переименовать можно всегда."))
            return False
        return True

    def rename_chapter(self, ids):
        ch = self._chapter(ids)
        if ch is None or "line" not in ch:
            return
        title = self.view.ask_text(tr("Переименовать главу"), tr("Название главы:"), initial=ch["title"])
        if not title or not title.strip() or title.strip() == ch["title"]:
            return
        book, old = self.book, out_mp3(self.book.dir, ch)
        book.rename_chapter(ch["id"], title.strip())
        new = out_mp3(book.dir, next(c for c in book.chapters if c["id"] == ch["id"]))
        if old.exists() and old != new:                       # готовый MP3 переименовывается, озвучивать заново не нужно
            os.replace(old, new)
            rename_result(book.dir, old.name, new.name)
            self.retagging = True

            def job():
                try:
                    self.queue.put(("info_saved", {"count": self.service.retag(book, {ch["id"]})}))
                except Exception as e:
                    self.queue.put(("info_saved", {"error": f"{type(e).__name__}: {e}"}))
            self.runner(job)
        self._layout_changed()

    def split_chapter(self, ids):
        ch = self._chapter(ids)
        if ch is None or not self._may_cut():
            return
        choices = self.book.chapter_lines(ch["id"])
        if not choices:
            self.view.notify(tr("В этой главе одна строка: делить нечего."))
            return
        answer = self.view.ask_split(ch["title"], choices)
        if answer:
            self.book.split_chapter(*answer)
            self._layout_changed()

    def merge_chapter(self, ids):
        ch = self._chapter(ids)
        if ch is None or not self._may_cut():
            return
        if ch["id"] == len(self.book.chapters):
            self.view.notify(tr("Это последняя глава: объединять не с чем."))
            return
        self.book.merge_with_next(ch["id"])
        self._layout_changed()

    def reset_split(self):
        if not self.book or self.running or not (self.book.state.cuts or self.book.state.titles):
            return
        if self.book.state.cuts and self.book.started():
            self.view.notify(tr("Разделять и объединять главы можно, пока книга не начата. Переименовать можно всегда."))
            return
        self.book.reset_edits()
        self._layout_changed()

    def redo(self, ids):
        """Удаляет готовый MP3 и недоделанные куски выбранных глав, чтобы озвучить их заново."""
        chapters = [ch for ch in self.book.chapters if ch["id"] in ids] if self.book else []
        if not chapters or self.running:
            return
        names = ", ".join(str(ch["id"]) for ch in chapters[:8]) + ("…" if len(chapters) > 8 else "")
        question = tr("Удалить готовый MP3 и незаконченные куски глав {0}, чтобы озвучить их заново?").format(names)
        if self.view.ask_yes(tr("Озвучить заново"), question):
            self.service.reset_chapters(self.book, {ch["id"] for ch in chapters})
            self._refresh()

    # ---- запуск, установка и остановка
    def can_start(self):
        return bool(self.book and self.title_text.strip() and not self.running)

    def start(self):
        if self.running or not self.book:
            return
        if self.retagging:                              # пока обновляются теги, MP3 этой книги не трогаем
            self.view.notify(tr("Теги ещё обновляются, подождите несколько секунд."))
            return
        title = safe_folder_name(self.title_text) if self.title_text else ""
        if not title:
            self.view.show_error(tr(NEEDS_TITLE_HINT))
            return
        if title != self.book.title:
            self._load_book(title)
        book, engine = self.book, self.engine
        ids = self.target_ids()
        pre = self.service.preflight(book, engine, ids)
        if not pre.pending:
            self.view.notify(tr("Все главы уже озвучены."))
            return
        if not self._credentials_ready(engine) or not self._engine_ready(engine):
            return
        if pre.conflicts:
            chapter, label = pre.conflicts[0]
            if not self.view.ask_yes(tr("Глава уже начата другим движком"),
                                     tr("Глава {0}: {1}.\nТекущий движок ({2}) начнёт её с нуля, голоса не "
                                        "смешиваются.\nПродолжить?").format(chapter['id'], label, engine)):
                return
        if pre.estimate and not self.view.ask_yes(tr("Расчёт для Gemini"),
                                                  "\n".join(format_estimate(pre.estimate)).strip() + tr("\n\nНачать озвучку?")):
            return
        if file_title(self.path) is None:
            remember_title(self.ctx, self.path, self.text, title)
        book.state.profile, book.state.confirmed, book.state.engine = book.profile_used, True, engine
        remember_book_profile(self.ctx, book)
        book.save()
        book.export_chapters()
        self._drop_preview_engine()
        self.chunks_total, _ = pending_chunk_stats(self.ctx, book, pre.pending, engine)
        self.chunks_done, self.run_started, self.finish_info = 0, None, None
        self._progress(0, tr("Подготовка…"))
        self._launch("run", lambda token: self.service.run(book, engine, ids, token))

    def _credentials_ready(self, engine):
        """Спрашивает то, без чего движок не запустится: ключ Gemini или текст образца Qwen."""
        cfg = self.ctx.settings
        if engine == "gemini" and not gemini_api_key(cfg):
            prompt = tr("Вставьте ключ (бесплатно: {0}):").format(GEMINI_KEY_URL)
            key = self.view.ask_text(tr("API-ключ Gemini"), prompt, secret=True)
            if not key:
                return False
            self.ctx.update(gemini_api_key=key.strip().strip("\"'"))
        if engine == "qwen" and current_voice(cfg) is None:
            if self.view.ask_yes(tr("Нужен голос"), tr(
                    "Для Qwen нужен голос: короткая запись, которую он будет повторять, и её точный текст.\n\n"
                    "Открыть настройки, чтобы добавить первый голос?")):
                self.open_settings()
            return False
        return True

    def _engine_ready(self, engine):
        """True, если можно озвучивать. Недостающие пакеты предлагаем установить прямо из окна."""
        fixable, manual = self.service.readiness(engine)
        need_ffmpeg = not ensure_ffmpeg()
        if manual:
            lines = [tr("Движок пока не готов:")] + [f"• {problem}" for problem in fixable + manual]
            self.view.show_error("\n".join(lines))
            return False
        if fixable or need_ffmpeg:
            self._offer_install(engine, fixable, need_ffmpeg)
            return False
        return True

    def _offer_install(self, engine, fixable, need_ffmpeg):
        lines = ([tr("ffmpeg (нужен для сборки MP3, небольшой)")] if need_ffmpeg else []) + fixable
        text = tr("Не хватает:\n") + "\n".join(f"• {line}" for line in lines)
        warnings = qwen_warnings() if engine == "qwen" and fixable else []
        if warnings:
            text += tr("\n\nОбратите внимание:\n") + "\n".join(f"• {w}" for w in warnings)
        text += tr("\n\nУстановить в папку программы (.venv) сейчас?")
        if engine == "qwen" and fixable:
            text += tr(" Qwen: около 3 ГБ, это долго; ход установки виден в журнале.")
        if self.view.ask_yes(tr("Нужна установка"), text):
            self.resume_after_install = True
            self._progress(0, tr("Установка… (подробности в журнале)"), busy=True)
            self._launch("install", lambda token: self.service.install_missing(engine, token))

    def _launch(self, name, job):
        """Запускает job(token) в фоновом потоке; по окончании окно получает worker_exit."""
        self.token, self.running = CancellationToken(), True
        self._refresh_controls()
        token = self.token
        self.runner(lambda: self._work(name, job, token))

    def _work(self, name, job, token):
        ok = False
        try:
            ok = job(token)
        except Exception as e:
            self.service.events.emit("error", text=tr("❌ Непредвиденная ошибка: {0}: {1}").format(type(e).__name__, e))
        finally:
            self.service.events.emit("worker_exit", job=name, ok=ok)

    def stop(self):
        if self.running and self.token and not self.token.cancelled():       # повторный «Стоп» ничего не добавляет
            self.token.cancel()
            self.view.append_log(tr("Останавливаю…"))
            self.view.set_stopping()

    def edit_info(self):
        """Кнопка «Сведения…»: автор, серия, номер, описание и обложка книги для тегов MP3."""
        if not self.book or self.running:
            return
        if self.retagging:
            self.view.notify(tr("Теги ещё обновляются, подождите несколько секунд."))
            return
        state = self.book.state
        current = {"author": state.author, "series": state.series, "series_index": state.series_index,
                   "description": state.description, "cover": state.cover,
                   "book_cover": bool(book_metadata(self.book.path).get("cover"))}
        answer = self.view.ask_book_info(current)
        if not answer:
            return
        for name in ("author", "series", "series_index", "description", "cover"):
            setattr(state, name, str(answer[name]).strip())
        self.book.save()
        book = self.book
        self.retagging = True

        def job():
            try:
                self.queue.put(("info_saved", {"count": self.service.retag(book)}))
            except Exception as e:
                self.queue.put(("info_saved", {"error": f"{type(e).__name__}: {e}"}))
        self.runner(job)

    def open_folder(self):
        if self.book and self.book.dir.exists():
            self.view.open_path(self.book.dir)
        else:
            self.view.notify(tr("Папки с результатом ещё нет: она появится, когда начнётся озвучка."))

    def request_close(self):
        """True, если окно можно закрывать. Во время работы спрашиваем: готовые куски сохранятся, потом можно продолжить."""
        if not self.running:
            return True
        if self.view.ask_yes(tr("Идёт работа"), tr("Остановить и выйти? Готовые куски сохранятся, потом можно продолжить.")):
            self.token.cancel()
            return True
        return False

    # ---- события из фоновой работы
    def pump(self):
        """Окно вызывает это по таймеру: переносит события фоновой работы в окно."""
        self._unload_idle_preview()
        self._release_when_idle()
        while True:
            try:
                kind, data = self.queue.get_nowait()
            except queue.Empty:
                return
            self._handle(kind, data)

    def _handle(self, kind, data):
        if data.get("text"):
            self.view.append_log(data["text"])
        if kind == "chapter_started":
            self.chapter_index, self.chapter_total, self.current_chapter = data["index"], data["total"], data["chapter"]
            if self.run_started is None:
                self.run_started = self.clock()
            self._progress(self._fraction(), tr("Глава {0} · {1} из {2}").format(
                data['chapter'], self.chapter_index, self.chapter_total))
        elif kind == "chunk_done":
            self.chunks_done += 1
            self.view.update_row(self.current_chapter, data.get("done", data["index"]), data["total"], self.engine)
            eta = self._eta()
            text = tr("Глава {0} из {1} · кусок {2} из {3}").format(
                self.chapter_index, self.chapter_total, data['index'], data['total'])
            if eta:
                text += tr(" · осталось {0}").format(eta)
            self._progress(self._fraction((data["index"], data["total"])), text)
        elif kind == "chapter_done":
            self._refresh()
        elif kind == "install_done":
            self.restart_needed = data["restart"]
        elif kind == "run_finished":
            self.finish_info = data
        elif kind == "info_saved":
            self.retagging = False
            if data.get("error"):
                self.view.notify(tr("❌ Теги не обновились: {0}").format(data["error"]))
            else:
                self.view.notify(tr("Сведения о книге сохранены, теги обновлены в файлах: {0}").format(data["count"]))
        elif kind == "preview_done":
            self.preview_idle_since = time.monotonic()        # отсчёт до выгрузки модели — с конца пробы
            if data.get("engine") is not None and self.engine == "qwen":
                self.preview_engine = data["engine"]          # загруженная модель пригодится и после «Стоп»
            if data.get("token") is not self.preview_token or data["token"].cancelled():
                return                                        # пробу остановили: результат не нужен
            self.previewing = False
            if data.get("error"):
                self.view.notify(tr("❌ Проба не удалась: {0}").format(data["error"]))
            else:
                self.preview_engine = data["engine"] if self.engine == "qwen" else None
                self.view.play(data["path"])
            self._refresh_controls()
        elif kind == "worker_exit":
            self.running, self.token = False, None
            if data["job"] == "install":
                self._after_install(data["ok"])
            else:
                info, self.finish_info = self.finish_info or {}, None
                stopped = bool(info.get("stopped"))
                if data["ok"]:
                    self._progress(1.0, tr("Готово"))
                self.view.notify_done(data["ok"], stopped)
            self._refresh()

    def _fraction(self, chunk=None):
        """Доля выполненного: по кускам всей работы, а если их число неизвестно — по главам."""
        if self.chunks_total:
            return min(self.chunks_done / self.chunks_total, 1.0)
        return (self.chapter_index - 1 + (chunk[0] / chunk[1] if chunk else 0)) / self.chapter_total

    def _eta(self):
        """Сколько осталось, по средней скорости этого запуска; пусто, пока измерений мало."""
        left = self.chunks_total - self.chunks_done
        if self.chunks_done < 2 or left <= 0 or self.run_started is None:
            return ""
        return eta_text(left * (self.clock() - self.run_started) / self.chunks_done)

    def _after_install(self, ok):
        self._progress(0, "")
        resume, self.resume_after_install = self.resume_after_install, False
        if not ok:
            self.view.show_error(tr("Установка не удалась или остановлена. Подробности в журнале."))
        elif self.restart_needed:
            self.view.show_info(tr("Установка завершена, но версии библиотек изменились. Закройте программу "
                                   "и запустите её заново: она продолжит с того же места."))
        elif resume:
            self._refresh_controls()
            self.start()

    # ---- что показываем
    def _refresh(self):
        if not self.book:
            return
        rows = self.service.chapter_rows(self.book, self.engine)
        counts = [sum(r.kind == kind for r in rows) for kind in ("done", "part", "new")]
        stale = sum(r.kind == "stale" for r in rows)
        summary = tr("Глав: {0} · готово {1} · начато {2} · осталось {3} · {4}").format(
            len(rows), counts[0] + stale, counts[1], counts[2], tr(PROFILE_SHORT[self.book.profile_used]))
        if stale:
            summary += tr(" · устарело {0}").format(stale)
        if len(rows) > 1:
            summary += tr(" · Ctrl/Shift+клик: выбрать главы")
        self.view.show_rows(rows, summary, chapter_warnings(self.book.chapters, self.book.profile_used), self.selected)
        self.view.set_estimate(self._estimate_text())
        self._refresh_controls()

    def _estimate_text(self):
        pending = self.service.pending(self.book, self.target_ids())
        if not pending:
            return tr("Выбранные главы уже озвучены.") if self.selected else tr("Все главы озвучены.")
        if self.engine == "gemini":
            return estimate_line(self.service.estimate(self.book, pending))
        count, _ = pending_chunk_stats(self.ctx, self.book, pending, self.engine)
        return (tr("Осталось: {0} {1} · {2} {3}").format(len(pending), plural_word(len(pending), "chapter"), count,
                                                         plural_word(count, "chunk")))

    def _start_label(self):
        chosen = self.selected & {ch["id"] for ch in self.book.chapters} if self.book else set()
        return tr("Озвучить выбранные ({0})").format(len(chosen)) if chosen else tr("Озвучить")

    def _refresh_controls(self):
        self.view.set_state(self.can_start(), self.running, self._start_label())
        self._show_voices()
        items, index = self.language_choices()
        self.view.set_languages(items, index, bool(self.book) and not self.running)
        self.view.set_preview(not self.running and self.engine in ("qwen", "gemini", "dummy"), self.previewing)


SETTINGS_FIELDS = ("gemini_api_key", "gemini_model", "gemini_voice", "gemini_chunk_chars", "gemini_paid",
                   "gemini_daily_limit", "qwen_chunk_chars", "normalize_text",
                   "replacements")


class SettingsPresenter:
    """Презентер окна «Настройки»: проверка введённого, сохранение, проверка ключа и пробная фраза (в фоне)."""

    def __init__(self, ctx, view, on_saved=None, extra_report=None, runner=None):
        self.ctx, self.view = ctx, view
        self.on_saved = on_saved or (lambda: None)
        self.extra_report = extra_report or (lambda: [])      # строки расчёта для текущей книги
        self.runner = runner or start_thread
        self.queue = queue.Queue()
        self.busy = set()                                    # фоновые проверки, которые сейчас идут
        self.voices = library_voices(ctx.settings)           # голоса Qwen: правки применятся только по «Сохранить»
        self.voice = (current_voice(ctx.settings) or {}).get("name", "")

    def load(self):
        values = {name: getattr(self.ctx.settings, name) for name in SETTINGS_FIELDS}
        key_from_env = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        self.view.show_values(values, key_from_env)
        self.view.show_voices(self.voices, self.voice)

    def _clean(self, raw):
        """(очищенные значения, список ошибок)."""
        low, high, _ = GEMINI_CHUNK_LIMITS
        clean, errors = {}, []
        clean["gemini_api_key"] = str(raw["gemini_api_key"]).strip().strip("\"'")
        for name, title in (("gemini_model", tr("Модель")), ("gemini_voice", tr("Голос"))):
            clean[name] = str(raw[name]).split(" (")[0].strip()
            if not clean[name]:
                errors.append(tr("{0}: выберите значение из списка или впишите своё.").format(title))
        chunk = str(raw["gemini_chunk_chars"]).strip()
        if chunk.isdigit() and low <= int(chunk) <= high:
            clean["gemini_chunk_chars"] = int(chunk)
        else:
            errors.append(tr("Размер куска: число от {0} до {1}.").format(low, high))
        qwen_chunk = str(raw["qwen_chunk_chars"]).strip()
        if qwen_chunk.isdigit() and QWEN_CHUNK_LIMITS[0] <= int(qwen_chunk) <= QWEN_CHUNK_LIMITS[1]:
            clean["qwen_chunk_chars"] = int(qwen_chunk)
        else:
            errors.append(tr("Размер куска Qwen: число от {0} до {1}.").format(QWEN_CHUNK_LIMITS[0], QWEN_CHUNK_LIMITS[1]))
        limit = str(raw["gemini_daily_limit"]).strip() or "0"
        if limit.isdigit():
            clean["gemini_daily_limit"] = int(limit)
        else:
            errors.append(tr("Суточный лимит: целое число (0 — не учитывать)."))
        clean["gemini_paid"] = bool(raw["gemini_paid"])
        clean["normalize_text"] = bool(raw["normalize_text"])
        for line in bad_replacement_lines(raw["replacements"]):
            errors.append(tr("Свои замены, строка {0}: нужен вид «что = как читать».").format(line))
        clean["replacements"] = "\n".join(line.rstrip() for line in str(raw["replacements"]).strip().splitlines())
        return clean, errors

    def save(self, raw):
        """Сохраняет настройки; True — окно можно закрывать."""
        clean, errors = self._clean(raw)
        if errors:
            self.view.show_error("\n".join(errors))
            return False
        safe = GEMINI_CHUNK_LIMITS[2]
        question = tr("{0} больше {1}: запрос может упереться в предел длины ответа (около 11 "
                      "минут звука). Всё равно сохранить?").format(clean["gemini_chunk_chars"], safe)
        if clean["gemini_chunk_chars"] > safe and not self.view.ask_yes(tr("Большой кусок"), question):
            return False
        cfg = self.ctx.settings
        changes = {name: value for name, value in clean.items() if getattr(cfg, name) != value}
        if self.voices != library_voices(cfg):
            changes["qwen_voices"] = self.voices
        if self.voice != cfg.qwen_voice:
            changes["qwen_voice"] = self.voice
        if changes:
            self.ctx.update(**changes)
        self.on_saved()
        return True

    # ---- голоса Qwen
    def _ask_voice(self, voice, editing=None):
        """Окно голоса, пока данные не станут правильными или пользователь не отменит. None — отмена."""
        others = [v for v in self.voices if v["name"] != editing]
        while True:
            voice = self.view.ask_voice(voice)
            if voice is None:
                return None
            voice = {"name": voice["name"].strip(), "path": voice["path"].strip().strip("\"'"), "text": voice["text"].strip()}
            problem = voice_problem(voice, others)
            if not problem:
                voice["path"] = stored_voice_path(voice_file(voice))
                return voice
            self.view.show_error(problem)

    def add_voice(self):
        voice = self._ask_voice({"name": "", "path": "", "text": ""})
        if voice:
            self.voices.append(voice)
            self.voice = voice["name"]
            self.view.show_voices(self.voices, self.voice)
            self.view.set_status(tr("Голос «{0}» добавлен и выбран. Не забудьте «Сохранить».").format(voice["name"]), False)

    def edit_voice(self, index):
        if not 0 <= index < len(self.voices):
            return
        old = self.voices[index]
        voice = self._ask_voice(dict(old), editing=old["name"])
        if voice:
            self.voices[index] = voice
            if self.voice == old["name"]:
                self.voice = voice["name"]
            self.view.show_voices(self.voices, self.voice)

    def delete_voice(self, index):
        if not 0 <= index < len(self.voices):
            return
        name = self.voices[index]["name"]
        question = tr("Убрать голос «{0}» из списка? Сама запись на диске останется.").format(name)
        if self.view.ask_yes(tr("Удалить голос"), question):
            del self.voices[index]
            if self.voice == name:
                self.voice = self.voices[0]["name"] if self.voices else ""
            self.view.show_voices(self.voices, self.voice)

    def select_voice(self, index):
        if 0 <= index < len(self.voices):
            self.voice = self.voices[index]["name"]

    def play_voice(self, index):
        if 0 <= index < len(self.voices):
            path = voice_file(self.voices[index])
            if path.exists():
                self.view.play(path)
            else:
                self.view.show_error(tr("Файл записи не найден: {0}").format(path))

    def _trial_context(self, raw):
        """Копия контекста с введёнными, но ещё не сохранёнными значениями (для проверки и пробы)."""
        clean, _ = self._clean(raw)
        trial = copy.copy(self.ctx)
        trial_values = {k: v for k, v in clean.items() if k.startswith(("gemini_", "qwen_")) and k != "gemini_chunk_chars"}
        trial.settings = replace(self.ctx.settings, **trial_values)
        return trial

    def _once(self, name, status, job):
        """Фоновая проверка не запускается второй раз, пока не закончилась первая."""
        if name in self.busy:
            return
        self.busy.add(name)
        self.view.set_busy(name, True)
        self.view.set_status(status, False)
        self.runner(job)

    def _done(self, name):
        self.busy.discard(name)
        self.view.set_busy(name, False)

    def key_info(self, raw):
        """«Инфо о ключе»: работает ли ключ, какие TTS-модели ему доступны, тариф, запросы и расходы."""
        trial = self._trial_context(raw)
        key, extra = gemini_api_key(trial.settings), self.extra_report()

        def job():
            ok, text = check_key_models(key, trial.settings.gemini_model)
            self.queue.put(("key_info", (ok, "\n".join([text, ""] + usage_report_lines(trial) + [""] + extra).strip())))
        self._once("key", tr("Проверяю ключ…"), job)

    def check_update(self):
        if exe_path():
            self._once("update", tr("Проверяю обновления…"), lambda: self.queue.put(("exe_update_checked", check_exe_update())))
        else:
            self._once("update", tr("Проверяю обновления…"), lambda: self.queue.put(("update_checked", check_update())))

    def _download_exe_update(self, url):
        def job():
            try:
                apply_exe_update(url)
                self.queue.put(("exe_updated", ""))
            except Exception as e:
                self.queue.put(("exe_updated", f"{type(e).__name__}: {e}"))
        self.view.set_status(tr("Скачиваю новую версию…"), False)
        self.runner(job)

    def show_diagnostics(self):
        self._once("diagnostics", tr("Собираю сведения об окружении…"),
                   lambda: self.queue.put(("diagnostics", "\n".join(diagnostics_lines(self.ctx)))))

    def pump(self):
        """Окно вызывает это по таймеру: показывает результаты фоновых проверок."""
        while True:
            try:
                kind, result = self.queue.get_nowait()
            except queue.Empty:
                return
            if kind == "key_info":
                self._done("key")
                ok, text = result
                self.view.set_status("" if ok else tr("❌ Ключ не работает: подробности в окне."), not ok)
                self.view.show_text(tr("Ключ и лимиты Gemini"), text)
            elif kind == "update_checked":
                self._done("update")
                text, info, error = result
                if error:
                    self.view.set_status(f"❌ {error}", True)
                elif text is None:
                    self.view.set_status(tr("✅ У вас последняя версия."), False)
                elif self.view.ask_yes(tr("Обновление"), tr("Доступна новая версия ({0}). Обновить?").format(info or "?")):
                    try:
                        apply_update(text)
                    except OSError as e:
                        self.view.set_status(tr("❌ Не удалось обновить: {0}").format(e), True)
                    else:
                        self.view.set_status(tr("✅ Программа обновлена. Закройте её и откройте снова."), False)
                else:
                    self.view.set_status("", False)
            elif kind == "exe_update_checked":
                url, version, notes, error = result
                if error:
                    self._done("update")
                    self.view.set_status(f"❌ {error}", True)
                elif url is None:
                    self._done("update")
                    self.view.set_status(tr("✅ У вас последняя версия."), False)
                elif self.view.ask_yes(tr("Обновление"), tr(
                        "Доступна версия {0} (у вас {1}).\n\nЧто нового:\n{2}\n\nОбновить? Программа скачает новую "
                        "версию и сама перезапустится.").format(version, APP_VERSION, notes[:1500] or "—")):
                    self._download_exe_update(url)             # кнопка остаётся занятой до конца замены exe
                else:
                    self._done("update")
                    self.view.set_status("", False)
            elif kind == "exe_updated":
                self._done("update")
                if result:
                    self.view.set_status(tr("❌ Не удалось обновить: {0}").format(result), True)
                else:
                    self.view.restart()
            elif kind == "diagnostics":
                self._done("diagnostics")
                self.view.set_status("", False)
                self.view.show_text(tr("Диагностика (можно скопировать и отправить)"), result)


def release_memory():
    """Отдаёт неиспользуемую память: сборка мусора, кэш видеопамяти torch и, в Windows, просьба системе забрать у
    процесса незанятые страницы (если они понадобятся снова, Windows вернёт их сама)."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        except Exception:
            pass


def stop_sound():
    """Останавливает звук, который играет программа (Windows)."""
    if os.name == "nt":
        try:
            import winsound
            winsound.PlaySound(None, 0)
        except Exception:
            pass


def sound_seconds(path):
    try:
        import soundfile
        return float(soundfile.info(str(path)).duration)
    except Exception:
        return 0.0


class SoundButton:
    """Кнопка прослушивания: пока звук играет, на ней «■ Стоп», и нажатие его останавливает. Звучит всегда что-то одно:
    новая запись останавливает прежнюю и возвращает её кнопке обычную подпись."""
    current = None

    def __init__(self, button, idle_text, on_change=None):
        self.button, self.idle_text, self.on_change, self.busy, self.turn = button, idle_text, on_change, False, 0

    def play(self, path):
        if SoundButton.current not in (None, self):
            SoundButton.current.stop()
        play_sound(path)
        SoundButton.current, self.busy, self.turn = self, True, self.turn + 1
        turn = self.turn
        self.button.after(int(sound_seconds(path) * 1000) + 300, lambda: self._finished(turn))
        self._show()

    def stop(self):
        if self.busy:
            stop_sound()
        self._finished(self.turn)

    def _finished(self, turn):
        if turn != self.turn or not self.busy:
            return
        self.busy = False
        if SoundButton.current is self:
            SoundButton.current = None
        self._show()

    def _show(self):
        try:
            self.button.configure(text=tr("■ Стоп") if self.busy else self.idle_text)
        except Exception:
            pass                                          # окно уже закрыли
        if self.on_change:
            self.on_change()


def play_sound(path):
    """Проигрывает WAV прямо в программе (Windows); на других системах открывает его плеером по умолчанию."""
    if os.name == "nt":
        try:
            import winsound
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        except Exception:
            pass
    open_in_system(path)


def open_in_system(path):
    """Открывает файл или папку программой по умолчанию."""
    if os.name == "nt":
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


def enable_dpi_awareness():
    """На Windows без этого текст в окне выглядит размытым на экранах с масштабом больше 100%."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


class Tooltip:
    """Небольшая подсказка у элемента окна: появляется при наведении, исчезает, когда мышь уходит."""

    DELAY_MS = 600

    def __init__(self, widget, text, colors):
        self.widget, self.text, self.colors, self.tip, self.pending = widget, text, colors, None, None
        widget.bind("<Enter>", lambda e: self.schedule(), add="+")
        for event in ("<Leave>", "<ButtonPress>"):
            widget.bind(event, lambda e: self.hide(), add="+")

    def schedule(self):
        self.hide()
        self.pending = self.widget.after(self.DELAY_MS, self.show)

    def show(self):
        import tkinter as tk
        self.pending = None
        if self.tip is not None or not self.widget.winfo_exists():
            return
        pointer = self.widget.winfo_containing(*self.widget.winfo_pointerxy())
        if pointer is not self.widget:                         # мышь уже ушла — подсказка не нужна
            return
        x, y = self.widget.winfo_rootx(), self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", wraplength=320, background=self.colors["bg"],
                 foreground=self.colors["fg"], relief="solid", borderwidth=1, padx=8, pady=5).pack()
        self.tip.update_idletasks()                          # не вылезать за правый край экрана
        x = min(x, self.widget.winfo_screenwidth() - self.tip.winfo_width() - 8)
        self.tip.geometry(f"+{max(x, 0)}+{y}")

    def hide(self):
        if self.pending is not None:
            self.widget.after_cancel(self.pending)
            self.pending = None
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def play_notification(window, success):
    """Звук окончания: системный сигнал Windows (радостный или тревожный), на других системах обычный звонок Tk."""
    try:
        if os.name == "nt":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK if success else winsound.MB_ICONHAND)
        else:
            window.bell()
    except Exception:
        pass


def tame_palette(window):
    """Tk 9 (Python 3.14 в Windows): функция палитры sv-ttk (tk_setPalette) перекрашивает все существующие виджеты, в том
    числе ttk-подписи, и стирает заданные вручную цвета (серые подсказки, акцент у переключателей). Подставляем вариант,
    который лишь задаёт цвета по умолчанию для новых виджетов; свои цвета программа меняет сама. В Tk 8.6 ничего не меняется."""
    try:
        if int(str(window.tk.call("info", "patchlevel")).split(".")[0]) < 9:
            return False
        window.tk.eval("proc foxy_palette {args} {foreach {name value} $args {option add *$name $value widgetDefault}}")
        body = str(window.tk.call("info", "body", "configure_colors"))
        if "tk_setPalette" in body:
            window.tk.call("proc", "configure_colors", "", body.replace("tk_setPalette", "foxy_palette"))
        return True
    except Exception:
        return False


def apply_theme_colors(window):
    """Цвета фона и текста темы sv-ttk применяются сразу. Сама библиотека делает это позже, по событию смены темы, и на
    Windows оно может потеряться: тогда поля тёмные, а фон и подписи серые."""
    try:
        window.tk.call("configure_colors")
    except Exception:
        pass


def set_title_bar_theme(window, dark, recolor=False):
    """Тёмная рамка заголовка окна в тёмной теме (Windows 10 и 11); на других системах ничего не делает. После неё цвета
    темы применяются заново, а рамка перерисовывается сразу."""
    if os.name != "nt":
        return
    try:
        import ctypes
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        value = ctypes.c_int(1 if dark else 0)
        for attribute in (20, 19):                   # DWMWA_USE_IMMERSIVE_DARK_MODE: новые и старые сборки Windows 10
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)) == 0:
                break
        ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0027)    # перерисовать рамку: без сдвига и смены размера
    except Exception:
        pass
    if recolor:              # только для главного окна: перекраска касается всех окон программы, от неё всё мигает
        apply_theme_colors(window)


def patient_tkdnd(TkinterDnD):
    """Библиотека перетаскивания (tkdnd) изредка не загружается с первого раза (замечено в CI на Python 3.14 в Windows):
    пробуем ещё дважды с паузой, прежде чем сдаться."""
    if getattr(TkinterDnD._require, "patient", False):
        return
    original = TkinterDnD._require

    def require(tkroot):
        for attempt in range(3):
            try:
                return original(tkroot)
            except RuntimeError:
                if attempt == 2:
                    raise
                time.sleep(0.3)
    require.patient = True
    TkinterDnD._require = require


def center_over(window, owner):
    """Ставит окно по центру над главным окном программы (а не в угол экрана)."""
    main = owner.winfo_toplevel()
    while getattr(main, "master", None) is not None:
        main = main.master.winfo_toplevel()
    window.update_idletasks()
    x = main.winfo_rootx() + (main.winfo_width() - window.winfo_reqwidth()) // 2
    y = main.winfo_rooty() + (main.winfo_height() - window.winfo_reqheight()) // 2
    window.geometry(f"+{max(x, 0)}+{max(y, 0)}")


def tidy_selection(root):
    """После выбора в списке или щелчка стрелками поля выделение текста сразу снимается (иначе оно так и висит)."""
    fields = ("TCombobox", "TSpinbox", "TEntry", "Entry", "Spinbox")

    def clear(widget, leave=False):
        try:
            if widget.winfo_class() in fields:
                widget.selection_clear()
                if leave:                                        # без фокуса подсветка выбранного значения исчезает
                    widget.winfo_toplevel().focus_set()
        except Exception:
            pass
    root.bind_all("<<ComboboxSelected>>", lambda e: e.widget.after_idle(lambda: clear(e.widget, True)), add="+")
    for name in ("<<Increment>>", "<<Decrement>>"):
        root.bind_all(name, lambda e: e.widget.after_idle(lambda: clear(e.widget)), add="+")
    root.bind_all("<FocusOut>", lambda e: clear(e.widget), add="+")


def flash_taskbar(window):
    """Мигает кнопкой окна на панели задач Windows, пока окно не вернут в фокус."""
    if os.name != "nt":
        return
    try:
        import ctypes

        class FlashInfo(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("hwnd", ctypes.c_void_p), ("dwFlags", ctypes.c_uint),
                        ("uCount", ctypes.c_uint), ("dwTimeout", ctypes.c_uint)]
        info = FlashInfo(ctypes.sizeof(FlashInfo), int(window.wm_frame(), 16), 0x0E, 3, 0)    # трей, до возврата в фокус
        ctypes.windll.user32.FlashWindowEx(ctypes.byref(info))
    except Exception:
        pass


class MainWindow:
    """Окно (View): рисует интерфейс, перенаправляет действия пользователя презентеру и показывает то, что он скажет."""

    def __init__(self, make_presenter, theme="light", geometry=""):
        import tkinter as tk
        from tkinter import ttk
        from tkinterdnd2 import DND_FILES, TkinterDnD
        self.tk, self.ttk = tk, ttk
        enable_dpi_awareness()
        patient_tkdnd(TkinterDnD)
        self.root = TkinterDnD.Tk()
        self.style = ttk.Style(self.root)
        self._use_theme(theme)
        dark = self.dark
        self.root.withdraw()                                 # главное окно тоже показываем уже собранным
        self.root.title(APP_TITLE)
        try:
            self.icon = tk.PhotoImage(data=APP_ICON_PNG)
            self.root.iconphoto(True, self.icon)
        except tk.TclError:
            pass
        default = "x".join(str(n) for n in DEFAULT_WINDOW_SIZE)
        self.root.geometry(fit_geometry(geometry, self.root.winfo_screenwidth(), self.root.winfo_screenheight()) or default)
        self.root.minsize(*MIN_WINDOW_SIZE)
        self.book_file, self.row_items, self.log_open, self.running, self.stopping = "", {}, False, False, False
        self.dnd_files, self.frame, self.menu = DND_FILES, None, None
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self._on_drop)
        self._build()
        self._build_language_switch()
        self._build_theme_switch()
        set_title_bar_theme(self.root, dark, recolor=True)
        tidy_selection(self.root)
        self.root.deiconify()
        self.presenter = make_presenter(self)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<FocusIn>", lambda e: self._reset_title())
        self.root.after(100, self._tick)

    def run(self):
        self.root.mainloop()

    def _use_theme(self, theme):
        """Включает тему sv-ttk для нашего окна и пересчитывает цвета, которыми рисуются зона перетаскивания, журнал и т. п."""
        import sv_ttk
        sv_ttk.set_theme(theme, root=self.root)               # явно наше окно, а не «окно по умолчанию» Tk
        tame = tame_palette(self.root)
        apply_theme_colors(self.root)
        self.dark = dark = theme == "dark"
        colors = dict(PALETTES[theme],
                      bg=valid_color(self.style.lookup("TFrame", "background"), "#1c1c1c" if dark else "#fafafa"),
                      fg=valid_color(self.style.lookup("TLabel", "foreground"), "#fafafa" if dark else "#1c1c1c"))
        if not hasattr(self, "colors"):
            self.colors = {}
        self.colors.clear()
        self.colors.update(colors)
        if tame:                                              # палитра больше не перекрашивает окно за нас
            self.root.configure(background=colors["bg"])

    COLOR_OPTIONS = ("foreground", "background", "insertbackground", "highlightbackground", "highlightcolor",
                     "selectbackground")

    def _painted(self):
        """Все заданные цвета элементов окна: (элемент, параметр, цвет)."""
        found, pending = [], list(self.root.winfo_children())
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            for option in self.COLOR_OPTIONS:
                try:
                    found.append((widget, option, str(widget.cget(option)).lower()))
                except Exception:
                    pass
        return found

    def _recolor(self, old, new, painted):
        """Новая тема без пересборки окна: каждый цвет, заданный вручную, меняется на такой же из новой палитры."""
        mapping = {str(old[key]).lower(): new[key] for key in new if key in old}
        for widget, option, value in painted:
            if value in mapping:
                try:
                    widget.configure(**{option: mapping[value]})
                except Exception:
                    pass
        for tag, key in (("done", "ok"), ("stale", "warn"), ("part", "warn"), ("new", "plain")):
            self.table.tag_configure(tag, foreground=new[key])
        self.style.configure("Stop.TButton", foreground=new["danger"])
        self._draw_drop()
        self._update_language_switch()
        self._update_theme_switch()

    def apply_theme(self, choice):
        """Оформление сменили в «Настройках»: окно перекрашивается сразу, без перезапуска; книга, выделение, журнал
        и прогресс остаются на месте."""
        theme = resolve_theme(choice)
        if theme == ("dark" if self.dark else "light"):
            return
        old, painted = dict(self.colors), self._painted()   # цвета запоминаем до смены: библиотека часть перекрасит сама
        self._use_theme(theme)                               # элементы sv-ttk перекрашивает сама библиотека
        self._recolor(old, self.colors, painted)             # свои цвета меняем на месте — окно не пересобирается
        set_title_bar_theme(self.root, self.dark, recolor=True)

    # ---- построение окна
    def _build_language_switch(self):
        """Переключатель языка в правом верхнем углу: RU | EN. Живёт отдельно от остального окна и не пересобирается."""
        ttk, c = self.ttk, self.colors
        self.language_box = box = ttk.Frame(self.root)
        box.place(relx=1.0, x=-18, y=2, anchor="ne")
        self.language_labels = {}
        for column, language in enumerate(LANGUAGES):
            if column:
                ttk.Label(box, text="|", foreground=c["muted"]).grid(row=0, column=2 * column - 1, padx=5)
            label = ttk.Label(box, text=language.upper(), cursor="hand2")
            label.grid(row=0, column=2 * column)
            label.bind("<Button-1>", lambda e, lang=language: self.presenter.change_language(lang))
            self.language_labels[language] = label
        self._update_language_switch()

    def _build_theme_switch(self):
        """Переключатель темы в левом верхнем углу: ☀ | ☾ — так же, как RU | EN справа."""
        ttk, c = self.ttk, self.colors
        self.theme_box = box = ttk.Frame(self.root)
        box.place(x=18, y=2, anchor="nw")
        self.theme_labels = {}
        for column, (theme, sign, hint) in enumerate((("light", "☀", tr("Светлая тема")), ("dark", "☾", tr("Тёмная тема")))):
            if column:
                ttk.Label(box, text="|", foreground=c["muted"]).grid(row=0, column=1, padx=5)
            label = ttk.Label(box, text=sign, cursor="hand2")
            label.grid(row=0, column=2 * column)
            label.bind("<Button-1>", lambda e, theme=theme: self.presenter.change_theme(theme))
            Tooltip(label, hint, c)
            self.theme_labels[theme] = label
        self._update_theme_switch()

    def _update_theme_switch(self):
        for theme, label in self.theme_labels.items():
            active = theme == ("dark" if self.dark else "light")
            label.configure(foreground=self.colors["accent"] if active else self.colors["muted"],
                            font=("Segoe UI Symbol", 11, "bold") if active else ("Segoe UI Symbol", 11))

    def _update_language_switch(self):
        for language, label in self.language_labels.items():
            active = language == _language
            label.configure(foreground=self.colors["accent"] if active else self.colors["muted"],
                            font=("Segoe UI", 9, "bold") if active else ("Segoe UI", 9))

    def rebuild(self):
        """Язык сменили: окно собирается заново на новом языке, а презентер возвращает в него всё показанное."""
        log_text, log_open = self.log_text.get("1.0", "end").rstrip("\n"), self.log_open
        self.frame.destroy()
        self.row_items, self.log_open = {}, False
        self._build()
        if log_text:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", log_text + "\n")
            self.log_text.configure(state="disabled")
            self.status_line.configure(text=log_text.splitlines()[-1].strip()[:140])
        self._toggle_log(log_open)
        if self.book_file:
            self.drop.configure(height=DROP_HEIGHTS[1])
        for switch in (self.language_box, self.theme_box):         # новое содержимое окна не должно их закрывать
            switch.lift()
        self._update_language_switch()
        self._update_theme_switch()
        self.presenter.restore_view()

    def _build(self):
        tk, ttk, root, c, DND_FILES = self.tk, self.ttk, self.root, self.colors, self.dnd_files
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.frame = frame = ttk.Frame(root, padding=(16, 24, 16, 16))
        frame.grid(sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)

        self.drop = tk.Canvas(frame, height=DROP_HEIGHTS[0], highlightthickness=0, bd=0, background=c["bg"], cursor="hand2")
        self.drop.grid(row=0, column=0, sticky="ew")
        self.drop.bind("<Configure>", lambda e: self._schedule_drop())
        self.drop.bind("<Button-1>", lambda e: self._choose_file())
        self.drop.drop_target_register(DND_FILES)
        self.drop.dnd_bind("<<Drop>>", self._on_drop)
        self.drop.dnd_bind("<<DropEnter>>", lambda e: self._draw_drop(True))
        self.drop.dnd_bind("<<DropLeave>>", lambda e: self._draw_drop(False))

        # подписи слева в одной сетке: колонка подписей одной ширины, все поля начинаются на одной линии
        form = ttk.Frame(frame)
        form.grid(row=1, column=0, sticky="ew", pady=(GAP, 0))
        form.columnconfigure(1, weight=1)

        def label(row, text):
            ttk.Label(form, text=text).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=(0, GAP // 2))

        label(0, tr("Название книги:"))
        title = ttk.Frame(form)
        title.grid(row=0, column=1, sticky="ew", pady=(0, GAP // 2))
        title.columnconfigure(0, weight=1)
        self.title_var = tk.StringVar()
        self.title_var.trace_add("write", lambda *a: self.presenter.title_edited(self.title_var.get()))
        self.entry = ttk.Entry(title, textvariable=self.title_var)
        self.entry.grid(row=0, column=0, sticky="ew")
        self.info_button = ttk.Button(title, text=tr("Сведения"), command=lambda: self.presenter.edit_info())
        self.info_button.grid(row=0, column=1, padx=(8, 0))
        self.entry.bind("<FocusOut>", lambda e: self.presenter.title_committed())
        self.entry.bind("<Return>", lambda e: self.presenter.title_committed())
        self.hint = ttk.Label(form, text="", foreground=c["muted"])        # пока подсказки нет, строка места не занимает

        label(2, tr("Главы:"))
        halves = ttk.Frame(form)                                           # «Главы» и «Язык книги» — по половине строки
        halves.grid(row=2, column=1, sticky="ew", pady=(0, GAP // 2))
        halves.columnconfigure(0, weight=1, uniform="half")
        halves.columnconfigure(1, weight=1, uniform="half")
        self.profile_keys = list(PROFILE_ORDER) + ["manual"]
        self.profile_box = ttk.Combobox(halves, state="readonly",
                                        values=[tr(PROFILE_TITLES.get(key, MANUAL_TITLE)) for key in self.profile_keys])
        self.profile_box.grid(row=0, column=0, sticky="ew")
        self.profile_box.current(self.profile_keys.index("auto"))
        self.profile_box.bind("<<ComboboxSelected>>",
                              lambda e: self.presenter.set_profile(self.profile_keys[self.profile_box.current()]))
        language = ttk.Frame(halves)
        language.grid(row=0, column=1, sticky="ew", padx=(16, 0))
        language.columnconfigure(1, weight=1)
        ttk.Label(language, text=tr("Язык книги:")).grid(row=0, column=0, padx=(0, 12))
        self.book_language_box = ttk.Combobox(language, state="disabled")
        self.book_language_box.grid(row=0, column=1, sticky="ew")
        self.book_language_box.bind("<<ComboboxSelected>>",
                                    lambda e: self.presenter.choose_language(self.book_language_box.current()))

        label(3, tr("Движок:"))
        options = ttk.Frame(form)
        options.grid(row=3, column=1, sticky="w", pady=(0, GAP // 2))
        self.engine_var = tk.StringVar(value=DEFAULT_ENGINE)
        self.radios = []
        for column, (key, text) in enumerate((("qwen", tr("Qwen (локально)")), ("gemini", tr("Gemini (облако)")))):
            radio = ttk.Radiobutton(options, text=text, value=key, variable=self.engine_var,
                                    command=lambda: self.presenter.set_engine(self.engine_var.get()))
            radio.grid(row=0, column=column, padx=(0, 14))
            self.radios.append(radio)

        label(4, tr("Голос:"))
        voice = ttk.Frame(form)                                            # список голосов — по ширине левой половины
        voice.grid(row=4, column=1, sticky="ew", pady=(0, GAP // 2))
        voice.columnconfigure(0, weight=1, uniform="half")
        voice.columnconfigure(1, weight=1, uniform="half")
        self.voice_box = ttk.Combobox(voice, state="readonly")
        self.voice_box.grid(row=0, column=0, sticky="ew")
        self.voice_box.bind("<<ComboboxSelected>>", lambda e: self.presenter.choose_voice(self.voice_box.current()))
        self.preview_button = ttk.Button(voice, text=tr("▶ Прослушать"), command=lambda: self.presenter.preview())
        self.preview_sound = SoundButton(self.preview_button, tr("▶ Прослушать"))
        self.preview_button.grid(row=0, column=1, sticky="w", padx=(16, 0))
        Tooltip(self.preview_button, tr("Озвучивает первые два предложения выбранной главы текущим голосом и языком.\n"
                                        "Каждая проба через Gemini тратит один запрос из суточного лимита."), c)

        self.summary = ttk.Label(frame, text="", foreground=c["muted"])
        self.summary.grid(row=2, column=0, sticky="w", pady=(GAP // 2, 4))

        table = ttk.Frame(frame)
        table.grid(row=3, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        self.table = ttk.Treeview(table, columns=("n", "heading", "chars", "status"), show="headings", height=5,
                                  selectmode="extended")
        columns = (("n", "№", 44, "e", False), ("heading", tr("Глава"), 360, "w", True),
                   ("chars", tr("Символов"), 90, "e", False), ("status", tr("Статус"), 200, "w", False))
        for key, text, width, anchor, stretch in columns:
            self.table.heading(key, text=text)
            self.table.column(key, width=width, anchor=anchor, stretch=stretch)
        for tag, color in (("done", c["ok"]), ("stale", c["warn"]), ("part", c["warn"]), ("new", c["plain"])):
            self.table.tag_configure(tag, foreground=color)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.bind("<<TreeviewSelect>>", lambda e: self.presenter.select(self.selected_ids()))
        self.table.bind("<Double-1>", self._on_double_click)
        for button in ("<Button-3>", "<Button-2>"):
            self.table.bind(button, self._on_context)
        if self.menu is not None:
            self.menu.destroy()
        self.menu = tk.Menu(root, tearoff=0)
        self.menu.add_command(label=tr("Озвучить выбранные"), command=lambda: self.presenter.start())
        self.menu.add_separator()
        self.menu.add_command(label=tr("Открыть MP3"), command=lambda: self.presenter.open_chapter(self.selected_ids()))
        self.menu.add_command(label=tr("Озвучить заново…"), command=lambda: self.presenter.redo(self.selected_ids()))
        self.menu.add_separator()
        self.menu.add_command(label=tr("Переименовать…"), command=lambda: self.presenter.rename_chapter(self.selected_ids()))
        self.menu.add_command(label=tr("Разделить…"), command=lambda: self.presenter.split_chapter(self.selected_ids()))
        self.menu.add_command(label=tr("Объединить со следующей"),
                              command=lambda: self.presenter.merge_chapter(self.selected_ids()))
        self.menu.add_command(label=tr("Вернуть автоматическое разбиение"), command=lambda: self.presenter.reset_split())

        # предупреждения, расчёт и подпись прогресса занимают место, только когда в них есть текст
        self.warnings = ttk.Label(frame, text="", foreground=c["warn"], wraplength=800, justify="left")
        self.warnings.grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.estimate = ttk.Label(frame, text="", foreground=c["muted"])
        self.estimate.grid(row=5, column=0, sticky="w", pady=(4, 0))
        progress = ttk.Frame(frame)
        progress.grid(row=6, column=0, sticky="ew", pady=(GAP // 2, 0))
        progress.columnconfigure(0, weight=1)
        self.progress_text = ttk.Label(progress, text="", foreground=c["muted"])
        self.progress_text.grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.bar = ttk.Progressbar(progress, maximum=1000)
        self.bar.grid(row=1, column=0, sticky="ew")
        for optional in (self.warnings, self.estimate, self.progress_text):
            optional.grid_remove()

        buttons = ttk.Frame(frame)
        buttons.grid(row=7, column=0, sticky="ew", pady=(GAP, 0))
        buttons.columnconfigure(3, weight=1)
        self.style.configure("Stop.TButton", foreground=c["danger"])    # «Стоп»: обычная кнопка с красной подписью
        self.start_button = ttk.Button(buttons, text=tr("Озвучить"), style="Accent.TButton", padding=(20, 7),
                                       command=lambda: self.presenter.stop() if self.running else self.presenter.start())
        folder_button = ttk.Button(buttons, text=tr("Открыть папку"), command=lambda: self.presenter.open_folder())
        self.settings_button = ttk.Button(buttons, text=tr("Настройки"), command=lambda: self.presenter.open_settings())
        self.settings_button.grid(row=0, column=0, padx=(0, 8))
        folder_button.grid(row=0, column=1, padx=(0, 8))
        self.log_toggle = ttk.Button(buttons, text=tr("Подробности ▸"), command=lambda: self._toggle_log())
        self.log_toggle.grid(row=0, column=2)
        self.start_button.grid(row=0, column=4)                  # одна кнопка: «Озвучить» ↔ «Стоп»

        status = ttk.Frame(frame)
        status.grid(row=8, column=0, sticky="ew", pady=(GAP, 0))
        status.columnconfigure(0, weight=1)
        self.status_line = ttk.Label(status, text="", foreground=c["muted"])
        self.status_line.grid(row=0, column=0, sticky="w")

        self.log_frame = ttk.Frame(frame)
        self.log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(self.log_frame, height=8, wrap="word", state="disabled", font="TkFixedFont", relief="flat",
                                borderwidth=0, highlightthickness=1, highlightbackground=c["border"], highlightcolor=c["border"],
                                background=c["bg"], foreground=c["fg"], insertbackground=c["fg"])
        log_scroll = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, sticky="ew")
        # журнал только для чтения, но текст выделяется мышью и копируется (Ctrl+C, Ctrl+A, меню по правому щелчку)
        self.log_text.bind("<Button-1>", lambda e: self.log_text.focus_set(), add="+")
        self.log_text.bind("<Control-Key>", self._log_keys)
        log_menu = tk.Menu(self.log_text, tearoff=0)
        log_menu.add_command(label=tr("Копировать"), command=self._copy_log)
        log_menu.add_command(label=tr("Выделить всё"), command=lambda: self.log_text.tag_add("sel", "1.0", "end-1c"))
        for button in ("<Button-3>", "<Button-2>"):
            self.log_text.bind(button, lambda e: log_menu.tk_popup(e.x_root, e.y_root))
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.set_state(False, False, tr("Озвучить"))

    # ---- зона перетаскивания
    def _rounded(self, x1, y1, x2, y2, radius, **options):
        points = [x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius, x2, y2 - radius, x2, y2, x2 - radius, y2,
                  x1 + radius, y2, x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1]
        return self.drop.create_polygon(points, smooth=True, **options)

    def _schedule_drop(self):
        """При изменении размера зона перетаскивания перерисовывается один раз, когда размер устоялся."""
        if not getattr(self, "_drop_pending", False):
            self._drop_pending = True

            def draw():
                self._drop_pending = False
                self._draw_drop()
            self.root.after_idle(draw)

    def _draw_drop(self, active=False):
        canvas, c = self.drop, self.colors
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 40:
            return
        line = c["accent"] if active else c["muted"]
        self._rounded(2, 2, width - 2, height - 2, 10, outline=line, dash=(5, 4), width=1.5,
                      fill=mix_colors(c["bg"], c["accent"], 0.10) if active else "")
        if self.book_file:
            canvas.create_text(18, height // 2, anchor="w", fill=c["fg"], font=("Segoe UI", 10, "bold"), text=self.book_file)
            canvas.create_text(width - 18, height // 2, anchor="e", fill=c["muted"], font=("Segoe UI", 9),
                               text=tr("перетащите другую книгу, чтобы заменить"))
            return
        x = width // 2
        for points in ((x, 14, x, 36), (x - 7, 29, x, 36, x + 7, 29), (x - 13, 38, x - 13, 44, x + 13, 44, x + 13, 38)):
            canvas.create_line(*points, fill=line, width=2, capstyle="round", joinstyle="round")
        canvas.create_text(x, 64, fill=c["fg"], font=("Segoe UI", 12, "bold"), text=tr("Перетащите книгу сюда"))
        canvas.create_text(x, 85, fill=c["muted"], font=("Segoe UI", 9), text=tr("TXT, EPUB, FB2 · или нажмите, чтобы выбрать"))

    def set_book_name(self, name):
        """Книга выбрана: зона перетаскивания сжимается в строку с именем файла."""
        self.book_file = name
        self.drop.configure(height=DROP_HEIGHTS[1])
        self._draw_drop()

    # ---- действия пользователя
    def _choose_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(parent=self.root, title=tr("Выберите книгу"),
                                          filetypes=[(tr("Книги"), "*.txt *.epub *.fb2 *.zip"), (tr("Все файлы"), "*.*")])
        if path:
            self.presenter.open_file(path)

    def _on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        self._draw_drop(False)
        if paths:
            self.presenter.open_file(paths[0])
        return event.action

    def selected_ids(self):
        return [int(self.table.set(item, "n")) for item in self.table.selection()]

    def _on_double_click(self, event):
        item = self.table.identify_row(event.y)
        if item:
            self.presenter.open_chapter([int(self.table.set(item, "n"))])

    def _on_context(self, event):
        item = self.table.identify_row(event.y)
        if not item:
            return
        if item not in self.table.selection():
            self.table.selection_set(item)
        actions = self.presenter.row_actions(self.selected_ids())
        for index, key in ((0, "start"), (2, "open"), (3, "redo")):
            self.menu.entryconfigure(index, state="normal" if actions[key] else "disabled")
        self.menu.tk_popup(event.x_root, event.y_root)

    def _tick(self):
        self.presenter.pump()
        self.root.after(100, self._tick)

    def _on_close(self):
        if self.presenter.request_close():
            if self.root.state() != "zoomed":                # развёрнутое на весь экран положение не запоминаем
                self.presenter.remember_geometry(self.root.geometry())
            self.root.destroy()

    def _copy_log(self):
        """Копирует выделенное, а если ничего не выделено, весь журнал."""
        try:
            text = self.log_text.get("sel.first", "sel.last")
        except self.tk.TclError:
            text = self.log_text.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _log_keys(self, event):
        """Ctrl+C и Ctrl+A в журнале при любой раскладке (по коду клавиши: при русской Tk не узнаёт букву)."""
        letter = event.keysym.lower()
        if letter == "c" or (os.name == "nt" and event.keycode == 67):
            self._copy_log()
            return "break"
        if letter == "a" or (os.name == "nt" and event.keycode == 65):
            self.log_text.tag_add("sel", "1.0", "end-1c")
            return "break"
        return None

    def _toggle_log(self, show=None):
        show = (not self.log_open) if show is None else show
        if show == self.log_open:
            return
        self.log_open = show
        if show:
            self.log_frame.grid(row=9, column=0, sticky="ew", pady=(GAP, 0))
            self.root.update_idletasks()
            needed = self.root.winfo_reqheight()
            if self.root.winfo_height() < needed:                 # окно подрастает, чтобы журнал поместился
                self.root.geometry(f"{self.root.winfo_width()}x{needed}")
        else:
            self.log_frame.grid_remove()
        self.log_toggle.configure(text=tr("Подробности ▾") if show else tr("Подробности ▸"))

    def _set_window_title(self, prefix=""):
        self.root.title(f"{prefix} · {APP_TITLE}" if prefix else APP_TITLE)

    def _reset_title(self):
        if not self.running and self.root.title() != APP_TITLE:
            self._set_window_title()

    # ---- то, что показывает презентер
    def show_rows(self, rows, summary, warnings, selected=()):
        values = [(row.id, self._chapter_name(row), fmt_int(row.chars), status_chip(row)) for row in rows]
        if [row.id for row in rows] == list(self.row_items):        # те же главы: обновляем строки на месте
            for row, value in zip(rows, values, strict=True):
                self.table.item(self.row_items[row.id], values=value, tags=(row.kind,))
        else:
            self.table.delete(*self.table.get_children())
            self.row_items = {row.id: self.table.insert("", "end", values=value, tags=(row.kind,))
                              for row, value in zip(rows, values, strict=True)}
        self.table.selection_set([self.row_items[i] for i in selected if i in self.row_items])
        self.summary.configure(text=summary)
        self._show_optional(self.warnings, "\n".join(f"⚠ {w}" for w in warnings[:3]))

    @staticmethod
    def _chapter_name(row):
        return chapter_display_name(row.heading, row.title)

    def update_row(self, chapter_id, done, total, engine):
        """Живой статус главы во время озвучки: «◐ 5/12 · qwen», не дожидаясь конца главы."""
        item = self.row_items.get(chapter_id)
        if item:
            row = ChapterRow(chapter_id, "", "", 0, 0, "part", "", done, total, engine)
            self.table.set(item, "status", status_chip(row))
            self.table.item(item, tags=("part",))

    def set_title(self, text, hint, needs_input):
        self.title_var.set(text)
        self.set_hint(hint, needs_input)
        if needs_input:
            self.entry.focus_set()

    def set_hint(self, text, alarm):
        self.hint.configure(text=text, foreground=self.colors["danger"] if alarm else self.colors["muted"])
        if text:
            self.hint.grid(row=1, column=1, sticky="w", pady=(0, GAP // 2))
        else:
            self.hint.grid_remove()

    def set_engine(self, name):
        self.engine_var.set(name)

    def ask_book_info(self, info):
        """Окно сведений о книге: автор, серия, номер, описание и обложка. None — отмена."""
        tk, ttk, c = self.tk, self.ttk, self.colors
        top = tk.Toplevel(self.root)
        top.withdraw()
        top.title(tr("Сведения о книге"))
        top.transient(self.root)
        frame = ttk.Frame(top, padding=16)
        frame.grid(sticky="nsew")
        frame.columnconfigure(1, weight=1)
        fields = {}
        labels = (("author", tr("Автор")), ("series", tr("Серия")), ("series_index", tr("Номер в серии")))
        for row, (key, label) in enumerate(labels):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
            fields[key] = tk.StringVar(value=info[key])
            ttk.Entry(frame, textvariable=fields[key], width=44 if key != "series_index" else 8).grid(
                row=row, column=1, sticky="ew" if key != "series_index" else "w", pady=4)
        ttk.Label(frame, text=tr("Описание")).grid(row=3, column=0, sticky="nw", padx=(0, 12), pady=4)
        description = tk.Text(frame, height=5, width=54, wrap="word", relief="flat", highlightthickness=1, font="TkDefaultFont",
                              highlightbackground=c["border"], highlightcolor=c["accent"], background=c["bg"],
                              foreground=c["fg"], insertbackground=c["fg"])
        description.insert("1.0", info["description"])
        description.grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text=tr("Обложка")).grid(row=4, column=0, sticky="w", padx=(0, 12), pady=4)
        cover_row = ttk.Frame(frame)
        cover_row.grid(row=4, column=1, sticky="ew", pady=4)
        cover = {"value": info["cover"]}
        cover_label = ttk.Label(cover_row, text="", foreground=c["muted"])
        cover_label.grid(row=0, column=0, sticky="w")

        def show_cover():
            value = cover["value"]
            text = (tr("нет") if value == "-" else Path(value).name if value
                    else tr("из книги") if info["book_cover"] else tr("нет (в книге её нет)"))
            cover_label.configure(text=text)

        def choose():
            from tkinter import filedialog
            path = filedialog.askopenfilename(parent=top, title=tr("Выберите обложку"),
                                              filetypes=[(tr("Картинка"), "*.jpg *.jpeg *.png"), (tr("Все файлы"), "*.*")])
            if path:
                cover["value"] = path
                show_cover()
        buttons_row = ttk.Frame(cover_row)
        buttons_row.grid(row=0, column=1, sticky="e", padx=(12, 0))
        cover_row.columnconfigure(0, weight=1)
        ttk.Button(buttons_row, text=tr("Выбрать…"), command=choose).grid(row=0, column=0, padx=(0, 6))
        for column, (text, value) in enumerate(((tr("Из книги"), ""), (tr("Без обложки"), "-")), 1):
            ttk.Button(buttons_row, text=text, command=lambda value=value: (cover.update(value=value), show_cover())
                       ).grid(row=0, column=column, padx=(0, 6))
        show_cover()
        ttk.Label(frame, text=tr("Записывается в MP3: автор, серия, описание и обложка видны в плеерах."),
                  foreground=c["muted"]).grid(row=5, column=1, sticky="w")
        result = {}
        actions = ttk.Frame(frame)
        actions.grid(row=6, column=1, sticky="e", pady=(12, 0))

        def save():
            result.update({key: var.get() for key, var in fields.items()}, description=description.get("1.0", "end"),
                          cover=cover["value"])
            top.destroy()
        ttk.Button(actions, text=tr("Сохранить"), style="Accent.TButton", command=save).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text=tr("Отмена"), command=top.destroy).grid(row=0, column=1)
        center_over(top, self.root)
        set_title_bar_theme(top, self.dark)
        top.deiconify()
        try:
            top.grab_set()
        except tk.TclError:
            pass
        top.wait_window()
        return dict(result) if result else None

    def set_languages(self, items, index, enabled):
        self.book_language_box.configure(values=items, state="readonly" if enabled else "disabled")
        if 0 <= index < len(items):
            self.book_language_box.current(index)

    def set_preview(self, enabled, busy=False):
        self.preview_button.state(["!disabled"] if enabled or busy else ["disabled"])
        self.preview_button.configure(text=tr("■ Стоп") if busy or self.preview_sound.busy else tr("▶ Прослушать"))

    def play(self, path):
        self.preview_sound.play(path)

    def stop_sound(self):
        self.preview_sound.stop()

    def sound_playing(self):
        return self.preview_sound.busy

    def set_voices(self, items, index):
        self.voice_box.configure(values=items)
        self.voice_box.configure(state="readonly" if items and not self.running else "disabled")
        if 0 <= index < len(items):
            self.voice_box.current(index)
        else:
            self.voice_box.set("")

    def set_profile(self, key):
        self.profile_box.current(self.profile_keys.index(key) if key in self.profile_keys else 0)

    @staticmethod
    def _show_optional(label, text):
        """Строка с текстом видна, пустая — не занимает места."""
        label.configure(text=text)
        if text:
            label.grid()
        else:
            label.grid_remove()

    def set_estimate(self, text):
        self._show_optional(self.estimate, text)

    def set_progress(self, fraction, text):
        self.bar.stop()
        self.bar.configure(mode="determinate", value=int(fraction * 1000))
        self._show_optional(self.progress_text, text or "")
        self._set_window_title(f"{int(fraction * 100)}%" if 0 < fraction < 1 else "")

    def set_busy(self, text):
        """Работа без известной длительности (установка): бегущая полоса и подпись; журнал открывается только при ошибках."""
        self.bar.configure(mode="indeterminate")
        self.bar.start(12)
        self._show_optional(self.progress_text, text)

    def set_state(self, can_start, running, start_label=None):
        start_label = start_label or tr("Озвучить")             # перевод на момент вызова, а не загрузки программы
        def enable(widget, on):
            widget.state(["!disabled"] if on else ["disabled"])
        self.running = running
        if not running:
            self.stopping = False
        if running:                                           # та же кнопка работает как «Стоп»
            self.start_button.configure(text=tr("Останавливаю…") if self.stopping else tr("Стоп"), style="Stop.TButton")
            enable(self.start_button, not self.stopping)
        else:
            self.start_button.configure(text=start_label, style="Accent.TButton")
            enable(self.start_button, can_start)
        for widget in self.radios:
            enable(widget, not running)
        enable(self.entry, not running)
        enable(self.settings_button, not running)
        enable(self.info_button, not running and bool(self.book_file))
        self.profile_box.configure(state="disabled" if running else "readonly")
        self.voice_box.configure(state="disabled" if running or not self.voice_box.cget("values") else "readonly")
        if running:
            self.book_language_box.configure(state="disabled")
        self.table.configure(selectmode="none" if running else "extended")

    def set_stopping(self):
        """«Стоп» нажат: кнопка ждёт конца текущего куска и повторно не нажимается."""
        self.stopping = True
        self.start_button.configure(text=tr("Останавливаю…"))
        self.start_button.state(["disabled"])

    def notify_done(self, ok, stopped):
        """Работа закончилась: звук, значок на панели задач и заголовок окна (если не сам пользователь остановил)."""
        if stopped:
            self._set_window_title()
            return
        self._set_window_title(tr("✓ Готово") if ok else tr("⚠ Остановлено"))
        play_notification(self.root, ok)
        flash_taskbar(self.root)

    def append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        if int(self.log_text.index("end-1c").split(".")[0]) > 2000:
            self.log_text.delete("1.0", "500.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        lines = text.strip().splitlines()
        if lines:
            self._last_log_line = lines[-1].strip()[:140]
            if not getattr(self, "_notice", None):
                self.status_line.configure(text=self._last_log_line, foreground=self.colors["muted"])
        if text.lstrip().startswith(("❌", "🛑")):
            self._toggle_log(True)                                 # ошибки сразу видны

    def ask_yes(self, title, text):
        from tkinter import messagebox
        return messagebox.askyesno(title, text, parent=self.root)

    def ask_text(self, title, prompt, secret=False, initial=""):
        """Окно с одной строкой ввода в стиле приложения: строка или None, если нажали «Отмена»."""
        tk, ttk = self.tk, self.ttk
        from tkinter import font as tkfont
        top = tk.Toplevel(self.root)
        top.withdraw()
        top.title(title)
        top.transient(self.root)
        top.resizable(False, False)
        width = max(460, tkfont.nametofont("TkDefaultFont").measure(title) + 170)    # заголовок окна не должен обрезаться
        frame = ttk.Frame(top, padding=16)
        frame.grid(sticky="nsew")
        frame.columnconfigure(0, weight=1, minsize=width - 32)
        ttk.Label(frame, text=prompt, wraplength=width - 32, justify="left").grid(row=0, column=0, sticky="w")
        value = tk.StringVar(value=initial)
        entry = ttk.Entry(frame, textvariable=value, show="•" if secret else "")
        entry.grid(row=1, column=0, sticky="ew", pady=(8, 14))
        result = {}

        def accept(_event=None):
            result["value"] = value.get()
            top.destroy()
        actions = ttk.Frame(frame)
        actions.grid(row=2, column=0, sticky="e")
        ttk.Button(actions, text="OK", style="Accent.TButton", command=accept).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text=tr("Отмена"), command=top.destroy).grid(row=0, column=1)
        top.bind("<Return>", accept)
        top.bind("<Escape>", lambda e: top.destroy())
        center_over(top, self.root)
        set_title_bar_theme(top, self.dark)
        top.deiconify()
        try:
            top.grab_set()
        except tk.TclError:
            pass
        entry.focus_set()
        entry.select_range(0, "end")
        top.wait_window()
        return result.get("value")

    def ask_split(self, chapter_title, choices):
        """Окно «Разделить главу»: с какой строки начать новую главу и как её назвать. (строка, название) или None."""
        tk, ttk, c = self.tk, self.ttk, self.colors
        top = tk.Toplevel(self.root)
        top.withdraw()
        top.title(tr("Разделить главу"))
        top.transient(self.root)
        frame = ttk.Frame(top, padding=16)
        frame.grid(sticky="nsew")
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=tr("«{0}»: выберите строку, с которой начнётся новая глава.").format(chapter_title[:60]),
                  wraplength=520).grid(row=0, column=0, columnspan=2, sticky="w")
        box = tk.Listbox(frame, height=14, width=72, activestyle="none", relief="flat", highlightthickness=1,
                         highlightbackground=c["border"], background=c["bg"], foreground=c["fg"],
                         selectbackground=c["accent"], selectforeground="#ffffff", exportselection=False)
        for _, preview in choices:
            box.insert("end", preview)
        box.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        scroll = ttk.Scrollbar(frame, orient="vertical", command=box.yview)
        scroll.grid(row=1, column=1, sticky="ns", pady=(8, 8))
        box.configure(yscrollcommand=scroll.set)
        ttk.Label(frame, text=tr("Название новой главы:")).grid(row=2, column=0, sticky="w")
        name = tk.StringVar()
        ttk.Entry(frame, textvariable=name).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        def picked(_event=None):
            if box.curselection():
                name.set(choices[box.curselection()[0]][1][:80])
        box.bind("<<ListboxSelect>>", picked)
        result = {}
        actions = ttk.Frame(frame)
        actions.grid(row=4, column=0, columnspan=2, sticky="e", pady=(12, 0))

        def save():
            if box.curselection():
                result["value"] = (choices[box.curselection()[0]][0], name.get().strip())
                top.destroy()
        ttk.Button(actions, text=tr("Разделить"), style="Accent.TButton", command=save).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text=tr("Отмена"), command=top.destroy).grid(row=0, column=1)
        box.bind("<Double-1>", lambda e: save())
        center_over(top, self.root)
        set_title_bar_theme(top, self.dark)
        top.deiconify()
        try:
            top.grab_set()
        except tk.TclError:
            pass
        top.wait_window()
        return result.get("value")

    def show_error(self, text):
        from tkinter import messagebox
        messagebox.showerror(APP_TITLE, text, parent=self.root)

    def show_info(self, text):
        from tkinter import messagebox
        messagebox.showinfo(APP_TITLE, text, parent=self.root)

    def notify(self, text):
        """Сообщение без окна: на несколько секунд в строке внизу, акцентным цветом, потом там снова журнал."""
        self.status_line.configure(text=f"ℹ {text}"[:140], foreground=self.colors["accent"])
        if getattr(self, "_notice", None):
            self.root.after_cancel(self._notice)
        self._notice = self.root.after(8000, self._end_notice)

    def _end_notice(self):
        self._notice = None
        self.status_line.configure(text=getattr(self, "_last_log_line", ""), foreground=self.colors["muted"])

    def open_path(self, path):
        open_in_system(path)

    def show_settings(self, make_presenter):
        return SettingsDialog(self.root, make_presenter, self.colors, self.dark)


class SettingsDialog:
    """Окно «Настройки» (View): поля и переключатели; проверки и сохранение делает SettingsPresenter."""

    def __init__(self, parent, make_presenter, colors, dark=False):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.parent, self.colors, self.dark = tk, ttk, parent, colors, dark
        self.top = tk.Toplevel(parent)
        self.top.withdraw()                                  # показываем, только когда окно собрано и стоит на месте
        self.top.title(tr("Настройки"))
        self.top.transient(parent)
        self.closed = False
        self._build()
        self.presenter = make_presenter(self)
        self.presenter.load()
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.top.resizable(False, False)
        center_over(self.top, parent)
        set_title_bar_theme(self.top, dark)
        self.top.deiconify()
        try:
            self.top.wait_visibility()
            self.top.grab_set()
        except tk.TclError:
            pass
        self.top.after(100, self._tick)

    def _build(self):
        tk, ttk, top, c = self.tk, self.ttk, self.top, self.colors
        top.columnconfigure(0, weight=1)
        frame = ttk.Frame(top, padding=14)
        frame.grid(sticky="nsew")
        frame.columnconfigure(0, weight=1)

        gemini = ttk.LabelFrame(frame, text=tr("Gemini (облако)"), padding=10)
        gemini.grid(row=0, column=0, sticky="ew")
        gemini.columnconfigure(0, minsize=130)
        gemini.columnconfigure(1, weight=1)
        self.key_var, self.model_var, self.voice_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.chunk_var, self.limit_var, self.paid_var = tk.StringVar(), tk.StringVar(), tk.IntVar()
        ttk.Label(gemini, text=tr("API-ключ")).grid(row=0, column=0, sticky="w", pady=3, padx=(0, 12))
        key_row = ttk.Frame(gemini)
        key_row.grid(row=0, column=1, columnspan=2, sticky="w", pady=3)
        ttk.Style(top).configure("Key.TEntry", padding=(8, 5, 30, 5))       # справа место под «глазик»
        self.key_entry = ttk.Entry(key_row, textvariable=self.key_var, show="•", width=38, style="Key.TEntry")
        self.key_entry.grid(row=0, column=0)
        self.eye = ttk.Label(key_row, text="👁", foreground=c["muted"], cursor="hand2")
        self.eye.place(in_=self.key_entry, relx=1.0, x=-8, rely=0.5, anchor="e")
        self.eye.bind("<Button-1>", lambda e: self._toggle_key())
        Tooltip(self.eye, tr("Показать или скрыть ключ"), c)
        self.key_button = ttk.Button(key_row, text=tr("Инфо о ключе"),
                                     command=lambda: self.presenter.key_info(self.read_values()))
        self.key_button.grid(row=0, column=1, padx=(8, 0))
        self.key_note = ttk.Label(gemini, text="", foreground=c["warn"])       # видна, только когда есть что сказать
        ttk.Label(gemini, text=tr("Модель")).grid(row=2, column=0, sticky="w", pady=3)
        ttk.Combobox(gemini, textvariable=self.model_var, values=[name for name, _ in GEMINI_MODELS]
                     ).grid(row=2, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Label(gemini, text=tr("Голос")).grid(row=3, column=0, sticky="w", pady=3)
        ttk.Combobox(gemini, textvariable=self.voice_var, values=[f"{name} ({note})" for name, note in GEMINI_VOICES]
                     ).grid(row=3, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Label(gemini, text=tr("Размер куска")).grid(row=4, column=0, sticky="w", pady=3)
        chunk = ttk.Frame(gemini)
        chunk.grid(row=4, column=1, columnspan=2, sticky="w", pady=3)
        ttk.Spinbox(chunk, textvariable=self.chunk_var, from_=GEMINI_CHUNK_LIMITS[0], to=GEMINI_CHUNK_LIMITS[1],
                    increment=100, width=7).grid(row=0, column=0)
        ttk.Label(chunk, text=tr("символов: больше — меньше запросов, но выше риск сбоя"),
                  foreground=c["muted"]).grid(row=0, column=1, padx=(8, 0))
        ttk.Label(gemini, text=tr("Тариф")).grid(row=5, column=0, sticky="w", pady=3)
        tariff = ttk.Frame(gemini)
        tariff.grid(row=5, column=1, columnspan=2, sticky="w", pady=3)
        ttk.Radiobutton(tariff, text=tr("бесплатный"), value=0, variable=self.paid_var).grid(row=0, column=0)
        ttk.Radiobutton(tariff, text=tr("платный"), value=1, variable=self.paid_var).grid(row=0, column=1, padx=(12, 0))
        ttk.Label(tariff, text=tr("лимит запросов в сутки (0 — не учитывать)")).grid(row=0, column=2, padx=(16, 6))
        ttk.Spinbox(tariff, textvariable=self.limit_var, from_=0, to=100000, width=6).grid(row=0, column=3)

        qwen = ttk.LabelFrame(frame, text=tr("Qwen (локально)"), padding=10)
        qwen.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        qwen.columnconfigure(0, minsize=130)
        qwen.columnconfigure(1, weight=1)
        ttk.Label(qwen, text=tr("Размер куска")).grid(row=0, column=0, sticky="w", pady=3, padx=(0, 12))
        qwen_chunk = ttk.Frame(qwen)
        qwen_chunk.grid(row=0, column=1, sticky="w")
        self.qwen_chunk_var = tk.StringVar()
        ttk.Spinbox(qwen_chunk, textvariable=self.qwen_chunk_var, from_=QWEN_CHUNK_LIMITS[0], to=QWEN_CHUNK_LIMITS[1],
                    increment=100, width=7).grid(row=0, column=0)
        ttk.Label(qwen_chunk, text=tr("символов: влияет на скорость и интонацию (RTF в журнале)"),
                  foreground=c["muted"]).grid(row=0, column=1, padx=(8, 0))
        ttk.Label(qwen, text=tr("Голоса")).grid(row=1, column=0, sticky="nw", pady=3)
        voices = ttk.Frame(qwen)
        voices.grid(row=1, column=1, sticky="ew", pady=3)
        voices.columnconfigure(0, weight=1)
        self.voice_list = ttk.Treeview(voices, columns=("name", "file"), show="headings", height=3, selectmode="browse")
        self.voice_list.heading("name", text=tr("Название"))
        self.voice_list.heading("file", text=tr("Запись"))
        self.voice_list.column("name", width=150, stretch=False)
        self.voice_list.column("file", width=200, stretch=True)
        self.voice_list.grid(row=0, column=0, sticky="nsew")
        self.voice_list.bind("<<TreeviewSelect>>", lambda e: self.presenter.select_voice(self._voice_index()))
        self.voice_list.bind("<Double-1>", lambda e: self.presenter.edit_voice(self._voice_index()))
        actions = ttk.Frame(voices)
        actions.grid(row=0, column=1, sticky="n", padx=(8, 0))
        for row, (text, action) in enumerate(((tr("Добавить…"), lambda: self.presenter.add_voice()),
                                              (tr("Изменить…"), lambda: self.presenter.edit_voice(self._voice_index())),
                                              (tr("Удалить"), lambda: self.presenter.delete_voice(self._voice_index())),
                                              (tr("Прослушать"), self._listen_voice))):
            button = ttk.Button(actions, text=text, command=action)
            button.grid(row=row, column=0, sticky="ew", pady=(0, 4))
        self.voice_sound = SoundButton(button, tr("Прослушать"))
        ttk.Label(qwen, text=tr("Выделенный голос — тот, которым будет озвучиваться книга."),
                  foreground=c["muted"]).grid(row=2, column=1, sticky="w")

        speech = ttk.LabelFrame(frame, text=tr("Текст для озвучки"), padding=10)
        speech.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        speech.columnconfigure(0, minsize=130)
        speech.columnconfigure(1, weight=1)
        self.normalize_var = tk.BooleanVar()
        ttk.Checkbutton(speech, text=tr("Читать числа и сокращения словами (1984 году, 15%, т. е.)"),
                        variable=self.normalize_var).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(speech, text=tr("Свои замены")).grid(row=1, column=0, sticky="nw", pady=(8, 0), padx=(0, 12))
        self.rules_text = tk.Text(speech, height=2, width=50, wrap="none", relief="flat", highlightthickness=1,
                                  highlightbackground=c["border"], highlightcolor=c["accent"], background=c["bg"],
                                  foreground=c["fg"], insertbackground=c["fg"])
        self.rules_text.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        ttk.Label(speech, text=tr("по одной на строку: что = как читать, например «ИИ = искусственный интеллект»"),
                  foreground=c["muted"]).grid(row=2, column=1, sticky="w")

        for section in (gemini, qwen, speech):
            for label in section.grid_slaves(column=0):
                label.grid_configure(padx=(0, 12))
        self.status = ttk.Label(frame, text="", wraplength=560, justify="left")
        self.status.grid(row=4, column=0, sticky="w", pady=(12, 0))
        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        buttons.columnconfigure(5, weight=1)
        self.diag_button = ttk.Button(buttons, text=tr("Диагностика…"), command=lambda: self.presenter.show_diagnostics())
        self.diag_button.grid(row=0, column=0, padx=(0, 8))
        self.update_button = ttk.Button(buttons, text=tr("Обновить…"), command=lambda: self.presenter.check_update())
        self.update_button.grid(row=0, column=1)
        ttk.Button(buttons, text=tr("Сохранить"), style="Accent.TButton", command=self._save).grid(row=0, column=6, padx=(0, 8))
        ttk.Button(buttons, text=tr("Отмена"), command=self.close).grid(row=0, column=7)

    # ---- то, что показывает презентер
    def show_values(self, values, key_from_env):
        self.key_var.set(values["gemini_api_key"])
        self.model_var.set(values["gemini_model"])
        self.voice_var.set(values["gemini_voice"])
        self.chunk_var.set(str(values["gemini_chunk_chars"]))
        self.limit_var.set(str(values["gemini_daily_limit"]))
        self.paid_var.set(1 if values["gemini_paid"] else 0)
        self.qwen_chunk_var.set(str(values["qwen_chunk_chars"]))
        self.normalize_var.set(values["normalize_text"])
        self.rules_text.delete("1.0", "end")
        self.rules_text.insert("1.0", values["replacements"])
        self.key_note.configure(text=tr("Ключ задан ещё и переменной окружения: она главнее этого поля.") if key_from_env else "")
        if key_from_env:
            self.key_note.grid(row=1, column=1, columnspan=2, sticky="w")
        else:
            self.key_note.grid_remove()

    def show_voices(self, voices, chosen):
        self.voice_list.delete(*self.voice_list.get_children())
        for voice in voices:
            item = self.voice_list.insert("", "end", values=(voice["name"], voice["path"]))
            if voice["name"] == chosen:
                self.voice_list.selection_set(item)

    def _listen_voice(self):
        if self.voice_sound.busy:
            self.voice_sound.stop()
        else:
            self.presenter.play_voice(self._voice_index())

    def _voice_index(self):
        selected = self.voice_list.selection()
        return self.voice_list.index(selected[0]) if selected else -1

    def ask_voice(self, voice):
        """Окно одного голоса: название, файл записи (с прослушиванием) и точный текст. None — отмена."""
        tk, ttk, c = self.tk, self.ttk, self.colors
        top = tk.Toplevel(self.top)
        top.withdraw()
        top.title(tr("Голос"))
        top.transient(self.top)
        frame = ttk.Frame(top, padding=16)
        frame.grid(sticky="nsew")
        frame.columnconfigure(1, weight=1)
        name_var, path_var, result = tk.StringVar(value=voice["name"]), tk.StringVar(value=voice["path"]), {}
        ttk.Label(frame, text=tr("Название")).grid(row=0, column=0, sticky="w", padx=(0, 12), pady=3)
        ttk.Entry(frame, textvariable=name_var, width=40).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(frame, text=tr("Запись")).grid(row=1, column=0, sticky="w", padx=(0, 12), pady=3)
        file_row = ttk.Frame(frame)
        file_row.grid(row=1, column=1, sticky="ew", pady=3)
        file_row.columnconfigure(0, weight=1)
        ttk.Entry(file_row, textvariable=path_var).grid(row=0, column=0, sticky="ew")

        def choose():
            kinds = [(tr("Звук"), "*.wav *.flac *.mp3 *.ogg"), (tr("Все файлы"), "*.*")]
            path = self.ask_open_file(tr("Выберите запись голоса"), kinds)
            if path:
                path_var.set(path)
                if not name_var.get().strip():
                    name_var.set(Path(path).stem)

        def listen():
            path = voice_file({"path": path_var.get().strip().strip("\"'")})
            if sound.busy:
                sound.stop()
            elif path.is_file():
                sound.play(path)
        ttk.Button(file_row, text=tr("Выбрать…"), command=choose).grid(row=0, column=1, padx=(8, 0))
        listen_button = ttk.Button(file_row, text=tr("Прослушать"), command=listen)
        listen_button.grid(row=0, column=2, padx=(8, 0))
        sound = SoundButton(listen_button, tr("Прослушать"))
        ttk.Label(frame, text=tr("Текст записи")).grid(row=2, column=0, sticky="nw", padx=(0, 12), pady=3)
        text = tk.Text(frame, height=4, width=54, wrap="word", relief="flat", highlightthickness=1, font="TkDefaultFont",
                       highlightbackground=c["border"], highlightcolor=c["accent"], background=c["bg"],
                       foreground=c["fg"], insertbackground=c["fg"])
        text.insert("1.0", voice["text"])
        text.grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Label(frame, text=tr("Ровно то, что звучит в записи, слово в слово: иначе голос получится хуже."),
                  foreground=c["muted"]).grid(row=3, column=1, sticky="w")
        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=1, sticky="e", pady=(12, 0))

        def save():
            result.update(name=name_var.get(), path=path_var.get(), text=text.get("1.0", "end"))
            top.destroy()
        ttk.Button(buttons, text=tr("Сохранить"), style="Accent.TButton", command=save).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text=tr("Отмена"), command=top.destroy).grid(row=0, column=1)
        center_over(top, self.top)
        set_title_bar_theme(top, self.dark)
        top.deiconify()
        try:
            top.grab_set()
        except tk.TclError:
            pass
        top.wait_window()
        return dict(result) if result else None

    def _toggle_key(self):
        hidden = self.key_entry.cget("show") != ""
        self.key_entry.configure(show="" if hidden else "•")
        self.eye.configure(foreground=self.colors["accent"] if hidden else self.colors["muted"])

    def set_busy(self, name, busy):
        """Кнопка фоновой проверки недоступна, пока проверка не закончилась (повторные нажатия не копятся)."""
        button = {"key": self.key_button, "diagnostics": self.diag_button, "update": self.update_button}[name]
        button.state(["disabled"] if busy else ["!disabled"])

    def read_values(self):
        return {"gemini_api_key": self.key_var.get(), "gemini_model": self.model_var.get(), "gemini_voice": self.voice_var.get(),
                "gemini_chunk_chars": self.chunk_var.get(), "gemini_paid": bool(self.paid_var.get()),
                "gemini_daily_limit": self.limit_var.get(),
                "qwen_chunk_chars": self.qwen_chunk_var.get(), "normalize_text": self.normalize_var.get(),
                "replacements": self.rules_text.get("1.0", "end")}

    def set_status(self, text, alarm):
        self.status.configure(text=text, foreground=self.colors["danger"] if alarm else self.colors["muted"])

    def ask_yes(self, title, text):
        from tkinter import messagebox
        return messagebox.askyesno(title, text, parent=self.top)

    def ask_open_file(self, title, filetypes):
        from tkinter import filedialog
        return filedialog.askopenfilename(parent=self.top, title=title, filetypes=filetypes)

    def show_error(self, text):
        from tkinter import messagebox
        messagebox.showerror(tr("Настройки"), text, parent=self.top)

    def open_path(self, path):
        open_in_system(path)

    def play(self, path):
        self.voice_sound.play(path)

    def restart(self):
        """Обновление скачано: запускаем новый exe и закрываем программу."""
        restart_exe()
        self.parent.after(200, self.parent.destroy)

    def show_text(self, title, text):
        tk, ttk = self.tk, self.ttk
        popup = tk.Toplevel(self.top)
        popup.withdraw()
        popup.title(title)
        popup.transient(self.top)
        view = tk.Text(popup, width=100, height=min(24, text.count("\n") + 2), wrap="word", relief="flat", font="TkFixedFont",
                       background=self.colors["bg"], foreground=self.colors["fg"])
        view.insert("1.0", text)
        view.configure(state="disabled")
        view.grid(row=0, column=0, padx=12, pady=12)
        actions = ttk.Frame(popup)
        actions.grid(row=1, column=0, pady=(0, 12))
        ttk.Button(actions, text=tr("Скопировать"), command=lambda: self._copy(text)).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text=tr("Закрыть"), command=popup.destroy).grid(row=0, column=1)
        center_over(popup, self.top)
        set_title_bar_theme(popup, self.dark)
        popup.deiconify()

    def _copy(self, text):
        self.top.clipboard_clear()
        self.top.clipboard_append(text)

    def _save(self):
        if self.presenter.save(self.read_values()):
            self.close()

    def _tick(self):
        if not self.closed:
            self.presenter.pump()
            self.top.after(100, self._tick)

    def close(self):
        self.closed = True
        if SoundButton.current is not None:                # окно закрывают — звук из него тоже
            SoundButton.current.stop()
        self.top.destroy()


GUI_PROBLEM = ""              # почему окно открыть нельзя (для сообщения об ошибке)


def gui_problem(text):
    """Запоминает и выводит причину, по которой окно открыть нельзя."""
    global GUI_PROBLEM
    GUI_PROBLEM = text
    log(text)


def ensure_gui_packages():
    """Библиотеки для окна ставятся в .venv сами (они маленькие). False — окно открыть нельзя (причина в GUI_PROBLEM)."""
    try:
        import tkinter
        has_tk = bool(tkinter.TkVersion)
    except ImportError:
        has_tk = False
    if not has_tk:
        gui_problem(tr("❌ В этом Python нет Tk (графической библиотеки окна), без неё программа не работает.\n"
               "   Tk входит в установку Python, через pip его не поставить. Чтобы появилось окно:\n"
               "   1) Параметры → Приложения → Python 3.14 → Изменить → Modify → отметьте «tcl/tk and IDLE» → Install;\n"
               "   2) запустите программу снова.\n"
               "   Если Python из Microsoft Store или Anaconda, поставьте Python 3.14 с python.org и удалите папку .venv.\n"
               "   Сейчас используется Python из {0}").format(sys.base_prefix))
        return False
    for module, package in GUI_PACKAGES:
        if not has_module(module):
            log(tr("📦 Ставлю библиотеку для окна в .venv: {0}").format(package))
            if not pip("install", package):
                gui_problem(tr("❌ Не удалось установить {0}: без неё окно не откроется.").format(package))
                return False
    return True


CREATE_NO_WINDOW = 0x08000000


def no_window_flags(flags):
    """Флаги запуска служебной программы без своего окна консоли. Если окно запрошено явно (новая консоль) или процесс
    отсоединён, флаги не меняются."""
    return flags if flags & (0x00000010 | 0x00000008) else flags | CREATE_NO_WINDOW


def hide_child_consoles():
    """В оконном режиме на Windows служебные программы (nvidia-smi, ffmpeg, pip, проверка torch) запускаются без
    мелькающих чёрных окон: их вывод и так собирается программой."""
    if os.name != "nt" or getattr(subprocess.Popen, "hides_console", False):
        return
    base = subprocess.Popen

    class HiddenPopen(base):
        hides_console = True

        def __init__(self, *args, **kwargs):
            kwargs["creationflags"] = no_window_flags(kwargs.get("creationflags", 0))
            super().__init__(*args, **kwargs)
    subprocess.Popen = HiddenPopen


def acquire_lock(path, wait):
    """Замок на файл (его снимает сама система, даже если программа упала). Ждёт до wait секунд; None — занят."""
    lock = BookLock(path)
    deadline = time.monotonic() + wait
    while not lock.acquire():
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)
    return lock


def acquire_app_lock(wait=5.0):
    """Одна работающая копия программы на папку: общие файлы (настройки, журнал расходов, обложка, проба голоса) не
    рассчитаны на запись из двух процессов сразу. Ждёт до wait секунд: после обновления новое окно запускается, пока
    прежнее ещё закрывается. None — копия уже работает."""
    return acquire_lock(PATHS.cache / "app.lock", wait)


APP_LOCK = None                      # замок этой копии: держится до конца работы процесса


def run_gui(ctx, book_path=None):
    """Открывает окно и работает, пока его не закроют. False — окно открыть не получилось (причина в GUI_PROBLEM)."""
    if not ensure_gui_packages():
        return False
    hide_child_consoles()
    try:
        window = MainWindow(lambda view: MainPresenter(AudiobookService(ctx, EventBus()), view),
                            theme=resolve_theme(ctx.settings.theme), geometry=ctx.settings.window_geometry)
    except Exception as e:
        gui_problem(tr("❌ Окно открыть не удалось: {0}: {1}").format(type(e).__name__, e))
        return False
    drop_previous_versions()
    if book_path:
        window.presenter.open_file(book_path)
    if os.environ.get("AUDIOBOOK_HIDE_CONSOLE") == "1":          # start.bat: чёрное окно за программой не нужно
        set_console_visible(False)
    window.run()
    return True


def main():
    ap = argparse.ArgumentParser(description=tr("Озвучка книги: qwen / gemini"))
    ap.add_argument("--book", type=Path)
    ap.add_argument("--lang", choices=LANGUAGES, help=tr("язык интерфейса (запоминается в настройках)"))
    ap.add_argument("--after-update", action="store_true", help=argparse.SUPPRESS)     # перезапуск после обновления: прежнее окно ещё закрывается
    ap.add_argument("--prepare", action="store_true", help=tr("только поставить библиотеки окна и звука (для FoxyAudioBookMaker.exe)"))
    args = ap.parse_args()
    if args.prepare:                               # базовые пакеты уже поставлены при запуске, осталось окно
        if not ensure_gui_packages():
            return 2
        (PATHS.venv / "audiobook-ready.txt").write_text(APP_VERSION, encoding="utf-8")
        log(tr("✅ Готово."))
        return 0
    ctx = AppContext()
    if args.lang:
        ctx.update(language=args.lang)
        set_language(args.lang)
    if not run_gui(ctx, args.book):
        fail(GUI_PROBLEM or tr("❌ Окно открыть не удалось."))
    return 0


# ============================================================
# ПЕРЕВОД ИНТЕРФЕЙСА НА АНГЛИЙСКИЙ
# ============================================================

EN = {
    '❌ В этом Python нет Tk (графической библиотеки окна), без неё программа не работает.\n   Tk входит в установку Python, через pip его не поставить. Чтобы появилось окно:\n   1) Параметры → Приложения → Python 3.14 → Изменить → Modify → отметьте «tcl/tk and IDLE» → Install;\n   2) запустите программу снова.\n   Если Python из Microsoft Store или Anaconda, поставьте Python 3.14 с python.org и удалите папку .venv.\n   Сейчас используется Python из {0}': '❌ This Python has no Tk (the window graphics library), the program cannot work without it.\n   Tk is part of the Python installation and cannot be installed with pip. To get the window:\n   1) Settings → Apps → Python 3.14 → Modify → Modify → tick "tcl/tk and IDLE" → Install;\n   2) start the program again.\n   If Python comes from the Microsoft Store or Anaconda, install Python 3.14 from python.org and delete the .venv folder.\n   Python in use: {0}',
    '❌ Не удалось установить {0}: без неё окно не откроется.': '❌ Could not install {0}: the window cannot open without it.',
    '❌ Окно открыть не удалось.': '❌ The window could not be opened.',
    '❌ Окно открыть не удалось: {0}: {1}': '❌ The window could not be opened: {0}: {1}',
    "\n\nНачать озвучку?": "\n\nStart narration?",
    "\n\nОбратите внимание:\n": "\n\nPlease note:\n",
    "\n\nУстановить в папку программы (.venv) сейчас?": "\n\nInstall into the program folder (.venv) now?",
    "\n   ⚠️ {0}\n   🔄 повтор {1}/{2} через {3:.0f} с...": "\n   ⚠️ {0}\n   🔄 retry {1}/{2} in {3:.0f} s...",
    "\nВыход.": "\nExiting.",
    "\nК озвучке: {0} {1} · движок {2} · кусок {3} симв. · пауза {4} с": (
        "\nTo narrate: {0} {1} · engine {2} · chunk {3} chars · pause {4} s"),
    "\nНажми Enter, чтобы закрыть...": "\nPress Enter to close...",
    "\nРасчёт для Gemini (прикидка) · модель {0}": "\nGemini estimate (rough) · model {0}",
    "\n⚠️ Выбранная модель {0} в этом списке не найдена: выберите другую в «Настройках».": (
        "\n⚠️ The selected model {0} is not in this list: choose another one in Settings."),
    "\n❌ Непредвиденная ошибка (текст выше). Скопируй его и покажи.": (
        "\n❌ Unexpected error (text above). Copy it and report it."),
    "\n❌ Ошибка на куске {0}/{1}: {2}: {3}": "\n❌ Error on chunk {0}/{1}: {2}: {3}",
    "\n🎉 Готово. MP3 лежат в {0}": "\n🎉 Done. The MP3 files are in {0}",
    "\n🎙️ Глава {0}/{1}: {2} {3}": "\n🎙️ Chapter {0}/{1}: {2} {3}",
    "\n📦 Установка для {0} — только в {1}\n": "\n📦 Installing for {0} — only into {1}\n",
    "\n🛑 Остановлено. Готовые куски сохранены — при следующем запуске продолжу.": (
        "\n🛑 Stopped. Finished chunks are saved; the next run will continue."),
    "     готово: {0:.1f} с, аудио {1:.1f} с, RTF {2:.2f}": "     done: {0:.1f} s, audio {1:.1f} s, RTF {2:.2f}",
    "   Видеокарта {0} ({1}): работаю без ускорения Triton, озвучка будет медленнее.": (
        "   Graphics card {0} ({1}): working without Triton acceleration, narration will be "
        "slower."),
    ("   Видеокарта поколения {0}: ставлю сборку PyTorch для старых карт, без ускорения "
     "Triton."): (
        "   A {0} graphics card: installing the PyTorch build for older cards, without Triton "
        "acceleration."),
    "   Загружаю модель в видеопамять…": "   Loading the model into video memory…",
    "   Модель загружена за {0} с.": "   Model loaded in {0} s.",
    "   Прогрев модели: {0} с (в RTF кусков не входит).": "   Model warm-up: {0} s (not counted in chunk RTF).",
    "   Скачиваю модель Qwen (первый раз, несколько ГБ — это долго)…": (
        "   Downloading the Qwen model (first time, several GB, this takes a while)…"),
    "   Ускорение Triton: включено.": "   Triton acceleration: on.",
    ("   Ускорение Triton: выключено — в этой версии программы оно на этой видеокарте уже не "
     "заработало. После обновления программы будет новая попытка."): (
        "   Triton acceleration: off, it already failed on this graphics card in this version of "
        "the program. It will be tried again after an update."),
    "   не хватает {0}, докачиваю...": "   {0} is missing, installing it...",
    "   …ещё загружаю модель, прошло {0} с": "   …still loading the model, {0} s so far",
    "   ↳ кусок {0}/{1} ({2} симв.) ...": "   ↳ chunk {0}/{1} ({2} chars) ...",
    "   ⏱ Итог по главе: озвучено за {0}, звука {1}, средний RTF {2:.2f}": (
        "   ⏱ Chapter summary: narrated in {0}, {1} of audio, average RTF {2:.2f}"),
    ("   ⚠️ Изменились текст главы, размер куска или голос — готовые куски этой главы "
     "сброшены."): (
        "   ⚠️ The chapter text, chunk size or voice changed, so the finished chunks of this "
        "chapter were reset."),
    "   ⚠️ Кусок {0} повреждён, озвучу его заново.": "   ⚠️ Chunk {0} is damaged, narrating it again.",
    ("   ⚠️ Ускорение Triton на этой видеокарте не заработало ({0}). Продолжаю без него, "
     "озвучка будет медленнее."): (
        "   ⚠️ Triton acceleration did not work on this graphics card ({0}). Continuing without "
        "it, narration will be slower."),
    "   🔄 Уже готово кусков: {0}/{1}. Продолжаю с недостающих.": (
        "   🔄 Chunks already done: {0}/{1}. Continuing with the missing ones."),
    "  {0}: запросов {1}, звука {2}, по ценам платного тарифа {3}": (
        "  {0}: {1} requests, {2} of audio, at paid-tier prices {3}"),
    "  Лимит {0} запросов в сутки на модель: сегодня использовано {1}, осталось {2}.": (
        "  Limit {0} requests per day per model: used today {1}, left {2}."),
    "  Лимит этой модели на сегодня: {0} из {1}, осталось {2}": "  This model's limit for today: {0} of {1}, left {2}",
    "  Осталось: {0} {1}, {2} {3} (кусок {4} симв.), звука около {5}": (
        "  Remaining: {0} {1}, {2} {3} (chunk {4} chars), about {5} of audio"),
    ("  Потребуется примерно дней: {0} (лимит обновляется в полночь по тихоокеанскому "
     "времени)."): (
        "  Days needed: about {0} (the limit resets at midnight Pacific time)."),
    "  С {0:%d.%m.%Y}, по открытым данным, цены вырастут: тогда около ${1:.2f}": (
        "  From {0:%Y-%m-%d}, according to public data, prices will rise: then about ${1:.2f}"),
    ("  Суточный лимит не задан: укажи его в «Настройках» (цифру смотри в AI Studio), и я "
     "посчитаю дни."): (
        "  The daily limit is not set: enter it in Settings (see the number in AI Studio) and I "
        "will count the days."),
    ("  Считаются только запросы этого скрипта. Остаток по лимитам Google (с учётом других "
     "программ) и баланс счёта"): (
        "  Only requests made by this program are counted. The remaining Google quota (including "
        "other programs) and the account balance"),
    "  Тариф бесплатный: деньги не списываются{0}, но действует лимит запросов.": (
        "  Free tier: no money is charged{0}, but the request limit applies."),
    "  Тариф платный, но цена этой модели в скрипте не задана: смотри {0}": (
        "  Paid tier, but the price of this model is not set in the program: see {0}"),
    "  Тариф платный: стоимость около ${0:.2f}": "  Paid tier: cost about ${0:.2f}",
    ("  Это оценка: точные счётчики и баланс показывает только Google AI Studio / Cloud "
     "Billing."): (
        "  This is an estimate: exact counters and balance are shown only by Google AI Studio / "
        "Cloud Billing."),
    "  показывают только AI Studio и Google Cloud Billing: через API их получить нельзя.": (
        "  are shown only by AI Studio and Google Cloud Billing: they cannot be read through the "
        "API."),
    " (без моделей с неизвестной ценой)": " (without models of unknown price)",
    " (книга)": " (book)",
    " (по платному тарифу это было бы около ${0:.2f})": " (at paid-tier prices it would be about ${0:.2f})",
    " Qwen: около 3 ГБ, это долго; ход установки виден в журнале.": (
        " Qwen: about 3 GB, this takes a long time; progress is shown in the log."),
    (" Windows не хватило памяти для загрузки модели: закройте лишние программы (браузер, "
     "игры) или включите «Автоматически выбирать объём файла подкачки»: Параметры → Система → "
     "О системе → Дополнительные параметры системы → Быстродействие: Параметры → "
     "Дополнительно → Виртуальная память, затем перезагрузите компьютер."): (
        " Windows ran out of memory while loading the model: close unneeded programs (browser, "
        "games) or turn on \"Automatically manage paging file size\": Settings → System → About → "
        "Advanced system settings → Performance: Settings → Advanced → Virtual memory, then "
        "restart the computer."),
    " · Ctrl/Shift+клик: выбрать главы": " · Ctrl/Shift+click: select chapters",
    " · загрузка модели {0} с, прогрев {1} с": " · model loading {0} s, warm-up {1} s",
    " · осталось {0}": " · {0} left",
    " · устарело {0}": " · outdated {0}",
    (" Не хватило видеопамяти: закройте игры и другие программы, которые её занимают, или "
     "используйте Gemini."): (
        " Not enough video memory: close games and other programs that use it, or use Gemini."),
    " по платному тарифу (сейчас бесплатно)": " at paid-tier prices (free for now)",
    "1/5 PyTorch с CUDA (скачивается около 3 ГБ, это долго)...": (
        "1/5 PyTorch with CUDA (about 3 GB to download, this takes a while)..."),
    "3/5 Qwen-TTS и его зависимости...": "3/5 Qwen-TTS and its dependencies...",
    "4/5 Версия accelerate, с которой у тебя всё работало...": "4/5 The accelerate version that is known to work...",
    "5/5 qwen3-tts-triton (без его конфликтующих зависимостей)...": (
        "5/5 qwen3-tts-triton (without its conflicting dependencies)..."),
    "API-ключ": "API key",
    "API-ключ Gemini": "Gemini API key",
    "CUDA недоступна. Проверь драйвер NVIDIA и установку torch (меню «Окружение»).": (
        "CUDA is not available. Check the NVIDIA driver and the torch installation (Environment "
        "menu)."),
    "Gemini (облако)": "Gemini (cloud)",
    "Gemini · тариф: {0} · модель: {1}": "Gemini · tier: {0} · model: {1}",
    "Gemini не дал результат после {0} попыток: {1}": "Gemini gave no result after {0} attempts: {1}",
    "Gemini: ключ {0} · модель {1} · голос {2} · кусок {3} · {4}": "Gemini: key {0} · model {1} · voice {2} · chunk {3} · {4}",
    "Google не принял API-ключ. Проверь его в меню «Настройки».": "Google did not accept the API key. Check it in Settings.",
    ("Python в .venv ниже %d.%d — qwen3-tts-triton не заработает. Пересоздай .venv на нужной "
     "версии."): (
        "Python in .venv is older than %d.%d, so qwen3-tts-triton will not work. Recreate .venv "
        "with the right version."),
    "Qwen (локально)": "Qwen (local)",
    "TXT, EPUB, FB2 · или нажмите, чтобы выбрать": "TXT, EPUB, FB2 · or click to choose",
    "ffmpeg (нужен для сборки MP3, небольшой)": "ffmpeg (needed to assemble MP3, small)",
    "ffmpeg не найден (нужен для сборки MP3)": "ffmpeg not found (needed to assemble MP3)",
    "torch {0} без поддержки CUDA": "torch {0} without CUDA support",
    "torch {0} не поддерживает видеокарту поколения {1}: нужна сборка для старых карт": (
        "torch {0} does not support a {1} graphics card: the build for older cards is needed"),
    "{0:.1f} ГБ": "{0:.1f} GB",
    ("{0} больше {1}: запрос может упереться в предел длины ответа (около 11 минут звука). "
     "Всё равно сохранить?"): (
        "{0} is above {1}: a request may hit the response length limit (about 11 minutes of "
        "audio). Save anyway?"),
    "{0} мин": "{0} min",
    "{0} ч {1:02d} мин": "{0} h {1:02d} min",
    "{0}: выберите значение из списка или впишите своё.": "{0}: choose a value from the list or type your own.",
    "«{0}»: выберите строку, с которой начнётся новая глава.": "\"{0}\": choose the line where the new chapter starts.",
    "· не начата": "· not started",
    "Авто ({0})": "Auto ({0})",
    "Автоопределение (рекомендуется)": "Auto-detect (recommended)",
    "Автор": "Author",
    "Английский": "English",
    "Б": "B",
    "Без обложки": "No cover",
    "Большой кусок": "Large chunk",
    "В этой главе одна строка: делить нечего.": "This chapter has a single line: nothing to split.",
    "В этом месяце": "This month",
    "Введите название книги: по нему называется папка с результатом.": (
        "Enter the book title: the result folder is named after it."),
    "Вернуть автоматическое разбиение": "Restore automatic split",
    "Весь файл как одна глава": "The whole file as one chapter",
    "Видеокарта NVIDIA: не найдена": "NVIDIA graphics card: not found",
    "Видеокарта: {0}, {1} ГБ, драйвер {2}": "Graphics card: {0}, {1} GB, driver {2}",
    "Впишите точный текст, который звучит в записи.": "Enter the exact text spoken in the recording.",
    "Вручную (главы изменены)": "Manual (chapters edited)",
    "Все главы озвучены.": "All chapters are narrated.",
    "Все главы уже озвучены.": "All chapters are already narrated.",
    "Все файлы": "All files",
    "Вставьте ключ (бесплатно: {0}):": "Paste the key (free: {0}):",
    "Выберите запись голоса": "Choose a voice recording",
    "Выберите книгу": "Choose a book",
    "Выберите обложку": "Choose a cover",
    "Выбран голос «{0}»: им уже озвучивалась эта книга.": "Voice \"{0}\" selected: this book was already narrated with it.",
    "Выбранные главы уже озвучены.": "The selected chapters are already narrated.",
    "Выбрать…": "Choose…",
    "Выделенный голос — тот, которым будет озвучиваться книга.": (
        "The highlighted voice is the one the book will be narrated with."),
    "Выделить всё": "Select all",
    "Глав: {0} · готово {1} · начато {2} · осталось {3} · {4}": "Chapters: {0} · done {1} · started {2} · left {3} · {4}",
    "Глава": "Chapter",
    "Глава {0} · {1} из {2}": "Chapter {0} · {1} of {2}",
    "Глава {0} в {1:.1f} раза длиннее обычной — возможно, пропущен заголовок.": (
        "Chapter {0} is {1:.1f} times longer than usual: a heading may be missing."),
    "Глава {0} из {1} · кусок {2} из {3}": "Chapter {0} of {1} · chunk {2} of {3}",
    "Глава {0} очень короткая ({1} симв.) — возможно, ложный заголовок.": (
        "Chapter {0} is very short ({1} chars): it may be a false heading."),
    ("Глава {0}: {1}.\nТекущий движок ({2}) начнёт её с нуля, голоса не смешиваются.\n"
     "Продолжить?"): (
        "Chapter {0}: {1}.\nThe current engine ({2}) will start it from scratch, voices are never "
        "mixed.\nContinue?"),
    "Глава уже начата другим движком": "Chapter already started by another engine",
    "Главы:": "Chapters:",
    "Голос": "Voice",
    "Голос «{0}» добавлен и выбран. Не забудьте «Сохранить».": "Voice \"{0}\" is added and selected. Remember to press Save.",
    "Голос «{0}» уже есть: выберите другое название.": "Voice \"{0}\" already exists: choose another name.",
    "Голос:": "Voice:",
    "Голоса": "Voices",
    "Готово": "Done",
    "Д/н": "Y/n",
    "Движок пока не готов:": "The engine is not ready yet:",
    "Движок:": "Engine:",
    "Диагностика (можно скопировать и отправить)": "Diagnostics (you can copy and send this)",
    "Диагностика…": "Diagnostics…",
    ("Для Qwen нужен голос: короткая запись, которую он будет повторять, и её точный текст.\n\n"
     "Открыть настройки, чтобы добавить первый голос?"): (
        "Qwen needs a voice: a short recording it will imitate, and its exact text.\n\nOpen "
        "Settings to add your first voice?"),
    ("Дневной лимит бесплатного тарифа исчерпан. Запусти скрипт завтра — он продолжит с этого "
     "же места, либо подключи оплату в Google AI Studio."): (
        "The daily limit of the free tier is used up. Run the program tomorrow and it will "
        "continue from the same place, or enable billing in Google AI Studio."),
    ("Дневной лимит запросов ({0}) исчерпан по твоим настройкам. Продолжи завтра, после "
     "полуночи по тихоокеанскому времени, или включи оплату в AI Studio и отметь это в "
     "«Настройках»."): (
        "The daily request limit ({0}) from your settings is used up. Continue tomorrow, after "
        "midnight Pacific time, or enable billing in AI Studio and mark it in Settings."),
    "Добавить…": "Add…",
    ("Доступ запрещён (403): у ключа нет прав на эту модель, регион не поддерживается или "
     "нужна оплата. {0}"): (
        "Access denied (403): the key has no rights to this model, the region is not supported "
        "or payment is required. {0}"),
    ("Доступна версия {0} (у вас {1}).\n\nЧто нового:\n{2}\n\nОбновить? Программа скачает новую "
     "версию и сама перезапустится."): (
        "Version {0} is available (you have {1}).\n\nWhat is new:\n{2}\n\nUpdate? The program will "
        "download the new version and restart by itself."),
    "Доступна новая версия ({0}). Обновить?": "A new version is available ({0}). Update?",
    "За всё время": "All time",
    "Заголовки Markdown (# Название)": "Markdown headings (# Title)",
    "Заголовки «Глава 5», «Глава пятая», «Chapter 3», «Пролог» и т. п. (разные языки)": (
        "Headings like \"Chapter 5\", \"Chapter Five\", \"Kapitel 2\", \"Prologue\" (many languages)"),
    "Закрыть": "Close",
    "Занимает: .venv {0} · models {1} · .cache {2}": "Size: .venv {0} · models {1} · .cache {2}",
    "Записывается в MP3: автор, серия, описание и обложка видны в плеерах.": (
        "Written into the MP3 files: players show the author, series, description and cover."),
    "Запись": "Recording",
    "Запись длится {0:.0f} с, а нужна от нескольких секунд до полуминуты.": (
        "The recording lasts {0:.0f} s, but it should be from a few seconds to half a minute."),
    "Запрос отклонён (400): {0}": "Request rejected (400): {0}",
    "Звук": "Audio",
    "Идёт работа": "Work in progress",
    "Из книги": "From the book",
    "Изменить…": "Edit…",
    "Инфо о ключе": "Key info",
    "Испанский": "Spanish",
    "Итальянский": "Italian",
    "КБ": "KB",
    "Картинка": "Image",
    "Китайский": "Chinese",
    "Ключ задан ещё и переменной окружения: она главнее этого поля.": (
        "The key is also set by an environment variable, which takes priority over this field."),
    "Ключ и лимиты Gemini": "Gemini key and limits",
    "Книги": "Books",
    "Копировать": "Copy",
    "Корейский": "Korean",
    "МБ": "MB",
    "Модели: {0}": "Models: {0}",
    "Модель": "Model",
    "Модель Qwen выгружена из видеопамяти: {0} с без проб.": (
        "The Qwen model was unloaded from video memory: {0} s without tests."),
    "Модель Qwen не загрузилась: {0}.": "The Qwen model did not load: {0}.",
    "Модель Qwen: {0}": "Qwen model: {0}",
    "Модель не найдена для этого ключа. Выбери другую в меню «Настройки» → модель.": (
        "The model was not found for this key. Choose another one in Settings → model."),
    "Мой голос": "My voice",
    "Название": "Title",
    "Название введено. Проверьте, что оно верное.": "Title entered. Check that it is right.",
    "Название взято из имени файла. Всё верно?": "The title is taken from the file name. Is it right?",
    "Название взято из самой книги. Всё верно?": "The title is taken from the book itself. Is it right?",
    "Название главы:": "Chapter title:",
    "Название запомнено для этого файла. Всё верно?": "The title is remembered for this file. Is it right?",
    "Название книги:": "Book title:",
    "Название новой главы:": "New chapter title:",
    "Настройки": "Settings",
    "Начало": "Beginning",
    "Не удалось открыть книгу: {0}": "Could not open the book: {0}",
    "Не удалось прочитать запись: {0}": "Could not read the recording: {0}",
    "Не удалось прочитать файл «{0}»: {1}\nНужен текст в кодировке UTF-8.": (
        "Could not read the file \"{0}\": {1}\nThe text must be UTF-8 encoded."),
    "Не хватает:\n": "Missing:\n",
    "Немецкий": "German",
    "Нет API-ключа Gemini. Меню «Настройки» → ключ (бесплатно: {0}).": "No Gemini API key. Settings → key (free: {0}).",
    "Нет голоса для Qwen: добавьте запись голоса в «Настройках».": "No voice for Qwen: add a voice recording in Settings.",
    "Нет образца голоса: {0}": "No voice sample: {0}",
    "Номер в серии": "Number in series",
    "Номера глав идут не подряд: ": "Chapter numbers are not consecutive: ",
    "Нужен голос": "A voice is needed",
    "Нужна запись в формате WAV, FLAC, MP3 или OGG.": "The recording must be WAV, FLAC, MP3 or OGG.",
    "Нужна установка": "Installation needed",
    "Обложка": "Cover",
    "Обновить…": "Update…",
    "Обновление": "Update",
    "Объединить со следующей": "Merge with the next one",
    ("Озвучивает первые два предложения выбранной главы текущим голосом и языком.\nКаждая "
     "проба через Gemini тратит один запрос из суточного лимита."): (
        "Narrates the first two sentences of the selected chapter with the current voice and "
        "language.\nEach Gemini test uses one request from the daily limit."),
    "Озвучиваю пробу…": "Narrating the test…",
    "Озвучить": "Narrate",
    "Озвучить выбранные": "Narrate selected",
    "Озвучить выбранные ({0})": "Narrate selected ({0})",
    "Озвучить заново": "Narrate again",
    "Озвучить заново…": "Narrate again…",
    "Озвучка книги: qwen / gemini": "Book narration: qwen / gemini",
    "Окружение .venv в папке {0} не найдено.": "The .venv environment was not found in {0}.",
    "Окружение: {0}": "Environment: {0}",
    "Описание": "Description",
    "Осталось: {0} {1}": "Remaining: {0} {1}",
    "Осталось: {0} {1} · {2} {3}": "Remaining: {0} {1} · {2} {3}",
    "Останавливаю…": "Stopping…",
    "Остановить и выйти? Готовые куски сохранятся, потом можно продолжить.": (
        "Stop and exit? Finished chunks are saved, so you can continue later."),
    "Отказ: установка пакетов разрешена только в .venv рядом со скриптом.": (
        "Refused: packages may only be installed into the .venv next to the program."),
    "Открыть MP3": "Open MP3",
    "Открыть папку": "Open folder",
    "Отмена": "Cancel",
    "Пакеты: ": "Packages: ",
    "Папки с результатом ещё нет: она появится, когда начнётся озвучка.": (
        "The result folder does not exist yet: it appears when narration starts."),
    "Переименовать главу": "Rename chapter",
    "Переименовать…": "Rename…",
    "Перетащите книгу сюда": "Drop a book here",
    "Повторяются названия глав: ": "Chapter titles repeat: ",
    "Подготовка…": "Preparing…",
    "Подробности ▸": "Details ▸",
    "Подробности ▾": "Details ▾",
    "Пока идёт работа, настройки менять нельзя.": "Settings cannot be changed while work is in progress.",
    "Показать или скрыть ключ": "Show or hide the key",
    "Поколение видеокарты: {0} ({1}.{2}) · Qwen: {3}": "Graphics card generation: {0} ({1}.{2}) · Qwen: {3}",
    "Португальский": "Portuguese",
    "При разбиении на равные части главы вручную не правятся: выберите другой способ.": (
        "Chapters split into equal parts cannot be edited by hand: choose another method."),
    "Проба остановлена.": "The test was stopped.",
    "Проверка голоса.": "Voice check.",
    "Проверяю ключ…": "Checking the key…",
    "Проверяю обновления…": "Checking for updates…",
    "Прослушать": "Listen",
    "Равные части": "Equal parts",
    "Равные части по размеру (если в книге нет заголовков)": "Equal parts by size (when the book has no headings)",
    "Разделить": "Split",
    "Разделить главу": "Split chapter",
    "Разделить…": "Split…",
    "Разделять и объединять главы можно, пока книга не начата. Переименовать можно всегда.": (
        "Chapters can be split and merged only before narration starts. Renaming is always "
        "possible."),
    "Размер куска": "Chunk size",
    "Размер куска Qwen: число от {0} до {1}.": "Qwen chunk size: a number from {0} to {1}.",
    "Размер куска: число от {0} до {1}.": "Chunk size: a number from {0} to {1}.",
    "Размер части в символах [{0}]:": "Part size in characters [{0}]:",
    "Расчёт для Gemini": "Gemini estimate",
    "Регулярное выражение для строки-заголовка (например ^Глава \\d+):": (
        "Regular expression for the heading line (for example ^Chapter \\d+):"),
    "Ровно то, что звучит в записи, слово в слово: иначе голос получится хуже.": (
        "Exactly what is said in the recording, word for word, otherwise the voice will come out "
        "worse."),
    "Русский": "Russian",
    "Сведения": "Details",
    "Сведения о книге": "Book details",
    "Сведения о книге сохранены, теги обновлены в файлах: {0}": "Book details saved, tags updated in files: {0}",
    "Светлая тема": "Light theme",
    "Свои замены": "Your replacements",
    "Свои замены, строка {0}: нужен вид «что = как читать».": (
        "Your replacements, line {0}: use the form \"what = how to read\"."),
    "Свой способ разбивки": "Custom split method",
    "Своё регулярное выражение для строки-заголовка": "Your own regular expression for the heading line",
    "Сегодня ({0}, тихоокеанское время)": "Today ({0}, Pacific time)",
    "Сейчас идёт работа. Дождитесь окончания или остановите её, потом откройте другую книгу.": (
        "Work is in progress. Wait for it to finish or stop it, then open another book."),
    "Серия": "Series",
    "Символов": "Characters",
    "Скачиваю новую версию…": "Downloading the new version…",
    "Скопировать": "Copy",
    "Смена разбивки": "Change split",
    "Собираю сведения об окружении…": "Collecting environment details…",
    "Создать .venv здесь (Python: {0})?": "Create .venv here (Python: {0})?",
    "Сохранить": "Save",
    "Статус": "Status",
    "Стоп": "Stop",
    "Суточный лимит: целое число (0 — не учитывать).": "Daily limit: a whole number (0 to ignore).",
    "Тариф": "Tier",
    "Теги ещё обновляются, подождите несколько секунд.": "Tags are still being updated, please wait a few seconds.",
    "Текст для озвучки": "Text for narration",
    "Текст записи": "Recording text",
    "Только номера отдельными строками: «1», «II.», «3. Название»": (
        "Bare numbers on their own lines: \"1\", \"II.\", \"3. Title\""),
    "Тёмная тема": "Dark theme",
    ("У видеокарты {0} {1:.1f} ГБ памяти, а модели Qwen нужно не меньше {2} ГБ: на ней Qwen "
     "не поместится. Используйте Gemini."): (
        "Graphics card {0} has {1:.1f} GB of memory, but the Qwen model needs at least {2} GB: "
        "Qwen will not fit on it. Use Gemini."),
    ("У этой книги уже есть озвученные главы. Смена разбивки может сдвинуть нумерацию.\nВсё "
     "равно изменить?"): (
        "This book already has narrated chapters. Changing the split may shift the numbering.\n"
        "Change it anyway?"),
    "Убрать голос «{0}» из списка? Сама запись на диске останется.": (
        "Remove voice \"{0}\" from the list? The recording itself stays on disk."),
    "Удалить": "Delete",
    "Удалить голос": "Delete voice",
    "Удалить готовый MP3 и незаконченные куски глав {0}, чтобы озвучить их заново?": (
        "Delete the finished MP3 and unfinished chunks of chapters {0} to narrate them again?"),
    "Укажите название голоса.": "Enter a name for the voice.",
    ("Установка завершена, но версии библиотек изменились. Закройте программу и запустите её "
     "заново: она продолжит с того же места."): (
        "Installation finished, but library versions changed. Close the program and start it "
        "again: it will continue from the same place."),
    "Установка не удалась или остановлена. Подробности в журнале.": (
        "Installation failed or was stopped. See the log for details."),
    "Установка… (подробности в журнале)": "Installing… (details in the log)",
    "Файл записи не найден: {0}": "Recording file not found: {0}",
    "Файл изменился с прошлого раза. Название прежнее?": "The file has changed since last time. Is the title the same?",
    "Файл книги изменился: ручные правки глав сброшены.": "The book file has changed: manual chapter edits were reset.",
    ("Формат {0} не поддерживается. Поддерживаются TXT, EPUB и FB2. Переведите книгу в EPUB "
     "или .txt (UTF-8), например в бесплатной программе Calibre."): (
        "The {0} format is not supported. Supported formats: TXT, EPUB and FB2. Convert the book "
        "to EPUB or .txt (UTF-8), for example with the free program Calibre."),
    "Французский": "French",
    "Часть {0}": "Part {0}",
    "Читать числа и сокращения словами (1984 году, 15%, т. е.)": "Read numbers and abbreviations as words (1984, 15%, e.g.)",
    "Чтобы изменить главы, щёлкните по главе в таблице правой кнопкой мыши.": (
        "To edit chapters, right-click a chapter in the table."),
    "Эта глава ещё не озвучена: файла MP3 пока нет.": "This chapter is not narrated yet: there is no MP3 file.",
    "Это последняя глава: объединять не с чем.": "This is the last chapter: nothing to merge with.",
    "Это проба голоса. Так будет звучать ваша книга.": "This is a voice test. This is how your book will sound.",
    "Язык книги:": "Book language:",
    "Японский": "Japanese",
    "без Triton (на этой карте оно не заработало)": "without Triton (it did not work on this card)",
    "бесплатный": "free",
    "бесплатный, лимит {0} запросов в сутки": "free, limit {0} requests per day",
    "бесплатный, лимит не задан": "free, limit not set",
    "быстрее и дешевле, для больших объёмов": "faster and cheaper, for large volumes",
    "в выпуске {0} нет скрипта или контрольных сумм, обновление отменено": (
        "release {0} has no script or checksums, update cancelled"),
    "в ответе нет аудио (finishReason: {0})": "no audio in the response (finishReason: {0})",
    ("в пути к папке есть не латинские символы: Triton может не скомпилировать ядра. Лучше "
     "путь вроде C:\\audiobook"): (
        "the folder path has non-Latin characters: Triton may fail to compile kernels. A path "
        "like C:\\audiobook is better"),
    ("видеокарта NVIDIA или её драйвер не найдены (nvidia-smi) — Qwen работает только на "
     "NVIDIA, используй Gemini"): (
        "NVIDIA graphics card or its driver not found (nvidia-smi): Qwen works only on NVIDIA, "
        "use Gemini"),
    "видеопамяти {0} ГБ: для Qwen рекомендуется от 8 ГБ, на меньшем объёме возможны вылеты": (
        "{0} GB of video memory: Qwen recommends 8 GB or more, with less it may crash"),
    "вручную": "manual",
    "д/Н": "y/N",
    "движок вернул пустое аудио": "the engine returned empty audio",
    "драйвер NVIDIA {0} старый: сборка torch cu128 просит 570 или новее — обнови драйвер": (
        "NVIDIA driver {0} is old: the torch cu128 build needs 570 or newer, update the driver"),
    ("драйвер NVIDIA {0} старый: сборка torch для старых карт (cu126) просит 560 или новее — "
     "обнови драйвер"): (
        "NVIDIA driver {0} is old: the torch build for older cards (cu126) needs 560 or newer, "
        "update the driver"),
    "задан": "set",
    "запуск из {0}": "started from {0}",
    "запуск скриптом": "started as a script",
    "из книги": "from the book",
    "кусков": "chunks",
    "кусок {0}: частота {1}, ожидалась {2}": "chunk {0}: sample rate {1}, expected {2}",
    "лимит запросов в сутки (0 — не учитывать)": "requests per day (0 to ignore)",
    "максимум качества, Google рекомендует для аудиокниг": "best quality, recommended by Google for audiobooks",
    "меньше минуты": "under a minute",
    "на GitHub пока нет выпусков": "there are no releases on GitHub yet",
    "на GitHub пока нет готового FoxyAudioBookMaker.exe": "there is no FoxyAudioBookMaker.exe on GitHub yet",
    "не задан": "not set",
    "не найден файл голоса «{0}»: {1}": "the file of voice \"{0}\" was not found: {1}",
    "не найдено": "none found",
    "не определён": "not detected",
    "не установлен torch": "torch is not installed",
    "не хватает кусков: {0}": "chunks are missing: {0}",
    "неизвестная настройка: {0}": "unknown setting: {0}",
    "нет": "none",
    "нет (в книге её нет)": "none (the book has none)",
    "нет API-ключа Gemini: меню «Настройки» → ключ (бесплатно: {0})": "no Gemini API key: Settings → key (free: {0})",
    "нет голоса для Qwen: добавьте запись голоса в «Настройках»": "no voice for Qwen: add a voice recording in Settings",
    "нет пакета requests": "the requests package is missing",
    "нет пакета {0}": "the {0} package is missing",
    "нет связи с GitHub: {0}": "no connection to GitHub: {0}",
    "одна глава": "one chapter",
    "около {0} звука": "about {0} of audio",
    "пакет {0} {1} старее нужного ({2})": "package {0} {1} is older than needed ({2})",
    "перетащите другую книгу, чтобы заменить": "drop another book to replace it",
    "платный": "paid",
    "по заголовкам": "by headings",
    "по заголовкам Markdown": "by Markdown headings",
    "по номерам": "by numbers",
    "по одной на строку: что = как читать, например «ИИ = искусственный интеллект»": (
        "one per line: what = how to read, for example \"AI = artificial intelligence\""),
    "по своему правилу": "by your rule",
    "при лимите {0} в сутки: дней {1}": "at a limit of {0} per day: {1} days",
    "примерно ${0:.2f}": "about ${0:.2f}",
    ("путь к папке длинный ({0} симв.): у Windows лимит около 260, установка torch может "
     "упасть. Лучше короткая папка, например C:\\audiobook"): (
        "the folder path is long ({0} chars): Windows has a limit of about 260, the torch "
        "installation may fail. A short folder such as C:\\audiobook is better"),
    "равные части": "equal parts",
    "с ускорением Triton": "with Triton acceleration",
    "сборка для старых карт, без Triton": "build for older cards, without Triton",
    "символов: больше — меньше запросов, но выше риск сбоя": "characters: bigger means fewer requests but more risk of failure",
    "символов: влияет на скорость и интонацию (RTF в журнале)": "characters: affects speed and intonation (RTF in the log)",
    "скачается при первом запуске (несколько ГБ)": "will be downloaded on first start (several GB)",
    "скачана": "downloaded",
    "скачанный файл не похож на программу": "the downloaded file does not look like the program",
    "скачанный файл не прошёл проверку контрольной суммы, обновление отменено": (
        "the downloaded file failed the checksum check, update cancelled"),
    "скачанный файл повреждён, обновление отменено": "the downloaded file is damaged, update cancelled",
    "старая preview-версия": "old preview version",
    "старая версия, Google заменяет её на 3.8": "old version, Google is replacing it with 3.8",
    "только поставить библиотеки окна и звука (для FoxyAudioBookMaker.exe)": (
        "only install the window and audio libraries (for FoxyAudioBookMaker.exe)"),
    "у голоса «{0}» нет текста записи: впишите его в «Настройках»": "voice \"{0}\" has no recording text: enter it in Settings",
    "файл {0} повреждён или защищён (DRM): {1}": "the file {0} is damaged or protected (DRM): {1}",
    "файл слишком велик для книги (возможно, это «бомба» в архиве), открытие отменено": (
        "the file is too large for a book (possibly an archive bomb), opening cancelled"),
    "цена модели неизвестна": "model price unknown",
    "язык интерфейса (запоминается в настройках)": "interface language (saved in the settings)",
    "ℹ️ Не удалось обновить {0}, работаю с установленной версией.": (
        "ℹ️ Could not update {0}, working with the installed version."),
    "ℹ️ Не удалось установить num2words: числа будут читаться цифрами, как есть.": (
        "ℹ️ Could not install num2words: numbers will be read as digits, as they are."),
    "ℹ️ Файл {0} создан более новой версией программы. Прежняя копия сохранена как {1}.": (
        "ℹ️ The file {0} was created by a newer version of the program. The previous copy was "
        "saved as {1}."),
    "⏸ {0}/{1} кусков ({2})": "⏸ {0}/{1} chunks ({2})",
    "■ Стоп": "■ Stop",
    "▶ Прослушать": "▶ Listen",
    "○ не начата": "○ not started",
    "● готова": "● done",
    "☁️ Gemini API · модель {0} · голос {1}": "☁️ Gemini API · model {0} · voice {1}",
    "⚠ Остановлено": "⚠ Stopped",
    "⚠ устарела": "⚠ outdated",
    ("⚠️ В сохранённом состоянии книги были недопустимые значения, они заменены стандартными: "
     "{0}."): (
        "⚠️ The saved state of the book had invalid values, they were replaced with the "
        "defaults: {0}."),
    ("⚠️ Готовые MP3 глав {0} сделаны из другого текста или другим голосом. Чтобы озвучить их "
     "заново, выбери «Озвучить заново»."): (
        "⚠️ The finished MP3 of chapters {0} were made from a different text or voice. To "
        "narrate them again, choose “Narrate again”."),
    ("⚠️ Не удалось зашифровать API-ключ, поэтому он не записан в файл и действует только до "
     "закрытия программы. Введи ключ снова или задай переменную окружения GEMINI_API_KEY."): (
        "⚠️ The API key could not be encrypted, so it was not written to the file and is kept "
        "only until the program is closed. Enter the key again or set the GEMINI_API_KEY "
        "environment variable."),
    ("⚠️ При установке сменились версии библиотек (numpy и др.), которые программа уже "
     "использует. Закройте её и запустите заново: она продолжит с того же места."): (
        "⚠️ The installation changed versions of libraries (numpy and others) that the program "
        "already uses. Close it and start it again: it will continue from the same place."),
    ("⚠️ Файл {0} повреждён и не прочитан, взяты значения по умолчанию. Копия сохранена как "
     "{1}."): (
        "⚠️ The file {0} is damaged and could not be read, defaults are used. A copy was saved "
        "as {1}."),
    "⚠️ устарела": "⚠️ outdated",
    "✅ Qwen установлен, импорт проходит.": "✅ Qwen is installed, the import works.",
    "✅ Готово.": "✅ Done.",
    "✅ Готово: {0}": "✅ Done: {0}",
    "✅ Ключ работает ({0}). TTS-модели, доступные этому ключу: {1}": (
        "✅ The key works ({0}). TTS models available to this key: {1}"),
    "✅ Нечего озвучивать: выбранные главы уже готовы.": "✅ Nothing to narrate: the selected chapters are already done.",
    "✅ Окружение создано.": "✅ Environment created.",
    "✅ Программа обновлена. Закройте её и откройте снова.": "✅ The program is updated. Close it and open it again.",
    "✅ У вас последняя версия.": "✅ You have the latest version.",
    "✅ готова": "✅ done",
    "✓ Готово": "✓ Done",
    "❌ Ключ не задан.": "❌ No key set.",
    "❌ Ключ не принят (HTTP {0}): {1}": "❌ The key was not accepted (HTTP {0}): {1}",
    "❌ Ключ не работает: подробности в окне.": "❌ The key does not work: details are in the window.",
    "❌ Не удалось запустить движок {0}: {1}: {2}": "❌ Could not start the {0} engine: {1}: {2}",
    "❌ Не удалось обновить: {0}": "❌ Could not update: {0}",
    "❌ Не удалось установить ": "❌ Could not install ",
    "❌ Непредвиденная ошибка:\n\n": "❌ Unexpected error:\n\n",
    "❌ Непредвиденная ошибка: {0}: {1}": "❌ Unexpected error: {0}: {1}",
    "❌ Нет связи с Google: {0}: {1}": "❌ No connection to Google: {0}: {1}",
    "❌ Нужен Python %d.%d. Поставь его с python.org и запусти скрипт снова.": (
        "❌ Python %d.%d is required. Install it from python.org and run the program again."),
    ("❌ Нужен пакет requests: установите его в меню «Окружение» (или кнопкой установки в "
     "окне)."): (
        "❌ The requests package is needed: install it in the Environment menu (or with the "
        "install prompt in the window)."),
    "❌ Окружение .venv сейчас создаёт другая копия программы. Подождите и запустите снова.": (
        "❌ Another copy of the program is creating the .venv environment right now. Wait and "
        "start again."),
    "❌ Проба не удалась: {0}": "❌ The test failed: {0}",
    "❌ Проверка импорта не прошла:\n": "❌ The import check failed:\n",
    "❌ Программа уже запущена из этой папки. Закройте другое окно или переключитесь на него.": (
        "❌ The program is already running from this folder. Close the other window or switch to "
        "it."),
    "❌ Склейка не удалась: {0}\n   Куски сохранены — запусти снова.": (
        "❌ Assembly failed: {0}\n   The chunks are saved, run again."),
    "❌ Теги не обновились: {0}": "❌ Tags were not updated: {0}",
    "❌ Установка не удалась: {0}: {1}": "❌ Installation failed: {0}: {1}",
    ("❌ Эта книга уже озвучивается другим запуском программы. Закройте его или дождитесь "
     "окончания."): (
        "❌ This book is already being narrated by another run of the program. Close it or wait "
        "for it to finish."),
    "＋ Добавить голос…": "＋ Add voice…",
    "🎙️ Голос из {0} будет считан при первом куске и дальше берётся из кэша библиотеки.": (
        "🎙️ The voice from {0} is read at the first chunk and then taken from the library cache."),
    "📦 Обновляю библиотеку до версии этой программы: {0} (стояла {1})": (
        "📦 Updating the library to the version this program needs: {0} (was {1})"),
    "📦 Ставлю базовый пакет в .venv: {0}": "📦 Installing a base package into .venv: {0}",
    "📦 Ставлю библиотеку для окна в .venv: {0}": "📦 Installing a window library into .venv: {0}",
    "🔗 Склеиваю главу {0} в MP3...": "🔗 Assembling chapter {0} into MP3...",
    ("🚀 Загружаю Qwen3-TTS (Faster + Triton). При первом запуске модель скачается в папку "
     "models/ ..."): (
        "🚀 Loading Qwen3-TTS (Faster + Triton). On the first start the model is downloaded into "
        "the models/ folder ..."),
    "🛑 Остановлено. Готовые куски сохранены — при следующем запуске продолжу с этого места.": (
        "🛑 Stopped. Finished chunks are saved; the next run will continue from this place."),
}


if __name__ == "__main__":
    try:
        startup()
        sys.exit(main())
    except KeyboardInterrupt:
        log(tr("\nВыход."))
    except Exception:
        if windowless():
            show_fatal(tr("❌ Непредвиденная ошибка:\n\n") + traceback.format_exc())
            sys.exit(1)
        set_console_visible(True)
        traceback.print_exc()
        fail(tr("\n❌ Непредвиденная ошибка (текст выше). Скопируй его и покажи."))
