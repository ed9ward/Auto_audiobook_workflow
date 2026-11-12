"""Тесты ядра audiobook_master.py. Запуск из корня репозитория: python -m pytest tests -q
Нужны: pytest, numpy, soundfile, requests. Тесты с MP3 пропускаются, если в системе нет ffmpeg.
Папка tests не попадает в «Download ZIP» (см. .gitattributes)."""
import base64
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import threading
import time
import types
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPT = Path(__file__).resolve().parents[1] / "audiobook_master.py"
PARA = "Это абзац обычного текста, он достаточно длинный, чтобы выглядеть как настоящая проза книги. "
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нет ffmpeg")


def body(paragraphs=6, size=4):
    return "\n\n".join(PARA * size for _ in range(paragraphs))


@pytest.fixture(autouse=True)
def no_leftover_tk_windows():
    """После каждого теста закрываем оставшееся окно Tk: иначе на Windows следующее окно может получить чужую тему
    или не загрузить tkdnd."""
    yield
    tkinter = sys.modules.get("tkinter")
    root = getattr(tkinter, "_default_root", None) if tkinter else None
    if root is not None:
        try:
            root.destroy()
        except Exception:
            pass
        tkinter._default_root = None


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Свежая копия модуля, у которой «папка проекта» — временная."""
    monkeypatch.setenv("AUDIOBOOK_HOME", str(tmp_path))
    monkeypatch.setenv("AUDIOBOOK_LANG", "ru")                       # тексты проверяем по-русски, как их задумали
    name = f"audiobook_master_{abs(hash(str(tmp_path)))}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.home = tmp_path
    module.real_wait = module.CancellationToken.wait
    module.waits = []
    monkeypatch.setattr(module.CancellationToken, "wait", lambda self, seconds: module.waits.append(seconds) or False)
    yield module
    sys.modules.pop(name, None)


def answers(monkeypatch, *replies):
    it = iter(replies)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


# ---------------------------------------------------------------- импорт и окружение

def test_import_has_no_side_effects(app):
    assert list(app.home.iterdir()) == []


def test_configure_environment_overrides_user_variables(app, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/elsewhere/hub")
    monkeypatch.setenv("HF_HOME", "/elsewhere/home")
    app.configure_environment()
    import os
    assert os.environ["HF_HUB_CACHE"].startswith(str(app.home))
    assert os.environ["HF_HOME"].startswith(str(app.home))
    assert (app.home / ".cache" / "tmp").is_dir()


def test_pip_refuses_to_install_outside_project_venv(app):
    with pytest.raises(RuntimeError):
        app.pip("install", "anything")


def test_remove_pip_leftovers(app, tmp_path, monkeypatch):
    import sysconfig
    site = tmp_path / "site"
    (site / "~umpy").mkdir(parents=True)
    (site / "numpy").mkdir()
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(site)})
    app.remove_pip_leftovers()
    assert not (site / "~umpy").exists() and (site / "numpy").exists()


# ---------------------------------------------------------------- настройки и состояние

def test_settings_save_only_changed_values_and_reload(app):
    ctx = app.AppContext()
    ctx.update(gemini_voice="Puck", gemini_daily_limit=10)
    stored = json.loads(app.PATHS.settings.read_text(encoding="utf-8"))
    assert stored.pop("schema_version") == app.SCHEMA_VERSION         # версия формата всегда, остальное только изменённое
    assert stored == {"gemini_voice": "Puck", "gemini_daily_limit": 10}
    again = app.AppContext()
    assert again.settings.gemini_voice == "Puck" and again.settings.gemini_daily_limit == 10
    assert again.settings.gemini_model == "gemini-3.8-flash-tts"


def test_settings_corrupt_or_wrong_types_fall_back_to_defaults(app):
    app.PATHS.settings.write_text("{not json", encoding="utf-8")
    assert app.AppContext().settings == app.Settings()
    app.PATHS.settings.write_text('{"gemini_daily_limit": "много", "gemini_voice": "Puck"}', encoding="utf-8")
    settings = app.AppContext().settings
    assert settings.gemini_daily_limit == 0 and settings.gemini_voice == "Puck"


def test_update_rejects_unknown_setting(app):
    with pytest.raises(KeyError):
        app.AppContext().update(no_such_setting=1)


def test_book_state_roundtrip(app, tmp_path):
    store = app.RecordStore(tmp_path / "x" / "split.json", app.BookState)
    state = store.load()
    assert state.profile == "auto" and state.confirmed is False
    state.profile, state.confirmed, state.engine = "headings", True, "gemini"
    store.save(state)
    assert store.load() == state


# ---------------------------------------------------------------- разбивка на главы

def test_split_russian_parts_epigraph_and_insert_inside_chapter(app):
    text = "Дневники\n\nАвтор\n\nПролог.\n\n" + body(2)
    for part, names in (("первая", ["первая", "вторая"]), ("вторая", ["третья"])):
        text += f"\n\nЧасть {part}.\n\nЭпиграф части. Короткий.\n"
        for name in names:
            text += f"\n\nГлава {name}.\n\nНазвание {name}.\n\n" + body(4)
            if name == "вторая":
                text += "\n\nВставка.\n\nЧасть первая.\n\n" + body(5)
    chapters, profile = app.split_book(text)
    assert profile == "headings" and len(chapters) == 4
    assert chapters[1]["text"].startswith("Часть первая.")
    assert "Вставка" in chapters[2]["text"]


def test_split_english_chapters_and_prologue(app):
    text = "THE BOOK\nby Someone\n\nPrologue\n\n" + body(2)
    text += "".join(f"\n\nCHAPTER {w}\n\n" + body(3) for w in ("ONE", "TWO"))
    text += "\n\nChapter Three: The Storm\n\n" + body(3) + "\n\nChapter Twenty-One\n\n" + body(3) + "\n\nEpilogue\n\n" + body(2)
    chapters, profile = app.split_book(text)
    assert profile == "headings" and len(chapters) == 6


def test_split_markdown_headings(app):
    text = "# Книга\n\n" + "".join(f"## Глава {i}\n\n" + body(3) + "\n\n" for i in range(1, 4))
    chapters, profile = app.split_book(text)
    assert profile == "markdown" and len(chapters) == 3


def test_bare_numbers_are_chapters_but_page_numbers_are_not(app):
    big = "\n\n".join(PARA * 12 for _ in range(3))
    chapters, profile = app.split_book("".join(f"\n\n{n}\n\n" + big for n in range(1, 6)))
    assert profile == "numbers" and len(chapters) == 5
    pages = "".join(f"\n\n{n}\n\n" + body(3) for n in range(1, 40))
    _, profile = app.split_book(pages)
    assert profile != "numbers"


def test_table_of_contents_is_not_voiced(app):
    toc = "Оглавление\n\n" + "".join(f"Глава {i}. Название {i}\n\n" for i in range(1, 5))
    text = toc + "".join(f"Глава {i}. Название {i}\n\n" + body(3) + "\n\n" for i in range(1, 5))
    chapters, _ = app.split_book(text)
    assert len(chapters) == 4 and "Оглавление" not in chapters[0]["text"]


def test_no_headings_short_text_is_single_and_long_text_is_split_by_size(app):
    chapters, profile = app.split_book(body(8))
    assert profile == "single" and len(chapters) == 1
    chapters, profile = app.split_book(body(1200))
    assert profile == "size" and len(chapters) > 5


def test_custom_regex_empty_or_broken_does_not_crash(app):
    for params in ({"regex": ""}, {"regex": "(["}):
        chapters, profile = app.split_book(body(4), "custom", params)
        assert len(chapters) == 1 and profile == "single"


def test_prose_lines_with_chapter_words_are_not_headings(app):
    text = ("Глава 1. Начало\n\n" + body(3) + "\n\nГлава партии долго не появлялась на людях, и все ждали заявления.\n\n"
            + "Часть работы была сделана раньше срока, но этого оказалось недостаточно для победы.\n\n" + body(2)
            + "\n\nГлава 2. Конец\n\n" + body(3))
    chapters, _ = app.split_book(text)
    assert len(chapters) == 2


def test_text_before_first_chapter_is_kept(app):
    long_front, short_front = body(3), "Название книги\nАвтор"
    chapters, _ = app.split_book(long_front + "\n\nГлава 1\n\n" + body(3) + "\n\nГлава 2\n\n" + body(3))
    assert len(chapters) == 3 and chapters[0]["heading"] == "Начало"
    chapters, _ = app.split_book(short_front + "\n\nГлава 1\n\n" + body(3) + "\n\nГлава 2\n\n" + body(3))
    assert len(chapters) == 2 and chapters[0]["text"].startswith("Название книги")


# ---------------------------------------------------------------- куски, имена, названия книг

def test_chunks_keep_all_text_and_respect_the_limit(app):
    text = body(30, 6)
    chunks = app.split_text_into_chunks(text, 1200)
    assert all(len(c) <= 1200 for c in chunks)
    assert " ".join(" ".join(chunks).split()) == " ".join(text.split())


@pytest.mark.parametrize("title, expected", [
    ('Дневники: "голодной" акулы?', "Дневники голодной акулы"),
    ("  ..Книга...  ", "Книга"),
    ("a/b\\c|d", "a b c d"),
    ("", "Книга"),
    ("models", "models (книга)"),
    ("CON", "CON (книга)"),
])
def test_safe_folder_name(app, title, expected):
    assert app.safe_folder_name(title) == expected
    assert len(app.safe_folder_name("x" * 200)) == 80


def test_file_title_and_remembered_title_helpers(app):
    ctx = app.AppContext()
    assert app.file_title(app.home / "Мой роман.txt") == "Мой роман"
    assert app.file_title(app.home / "1.txt") is None
    plain = app.home / "1.txt"
    assert app.remembered_title(ctx, plain, "текст") == (None, False)
    app.remember_title(ctx, plain, "текст", "Дневники")
    assert app.remembered_title(ctx, plain, "текст") == ("Дневники", True)
    assert app.remembered_title(ctx, plain, "другой текст") == ("Дневники", False)     # содержимое изменилось


# ---------------------------------------------------------------- Gemini

class FakeResponse:
    def __init__(self, status=200, body=None, text=None):
        self.status_code, self._body = status, body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("не JSON")
        return self._body


def audio_response(seconds=0.5, usage=True):
    buf = io.BytesIO()
    sf.write(buf, (0.1 * np.sin(np.arange(int(24000 * seconds)) / 9)).astype("float32"), 24000, subtype="PCM_16", format="WAV")
    body_ = {"candidates": [{"content": {"parts": [{"inlineData": {"data": base64.b64encode(buf.getvalue()).decode()}}]}}]}
    if usage:
        body_["usageMetadata"] = {"promptTokenCount": 100, "candidatesTokenCount": int(seconds * 25)}
    return FakeResponse(200, body_)


class FakeClient:
    """Заглушка GeminiClient: отвечает заранее заданными ответами, запоминает запросы."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def generate(self, model, payload):
        self.calls.append({"model": model, "payload": payload})
        response = self.responses.pop(0) if self.responses else audio_response()
        if isinstance(response, Exception):
            raise response
        return response

    def list_models(self):
        return self.responses.pop(0)


@pytest.fixture
def gemini(app, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    ctx = app.AppContext()
    ctx.update(gemini_api_key="AIzaTESTKEY0000001")

    def make(*responses):
        client = FakeClient(*responses)
        return ctx, app.GeminiEngine(ctx, client=client)
    return make


def test_gemini_payload_uses_new_schema_for_38_and_old_for_legacy_models(app, gemini):
    ctx, engine = gemini()
    assert engine.payload("привет")["generationConfig"]["speechConfig"]["voiceConfig"] == {"voice": "Kore"}
    ctx.update(gemini_model="gemini-3.1-flash-tts-preview")
    legacy = app.GeminiEngine(ctx, client=FakeClient())
    assert legacy.payload("привет")["generationConfig"]["speechConfig"]["voiceConfig"] == {
        "prebuiltVoiceConfig": {"voiceName": "Kore"}}


def test_decode_audio_accepts_wav_and_raw_pcm(app):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype="float32"), 16000, subtype="PCM_16", format="WAV")
    assert len(app.decode_audio_bytes(buf.getvalue())) == 24000          # 16 кГц приведён к 24 кГц
    pcm = (np.ones(2400, dtype="<i2") * 1000).tobytes()
    assert app.decode_audio_bytes(pcm).shape == (2400,)


def test_gemini_waits_as_google_asks_and_retries(app, gemini):
    retry = {"error": {"details": [{"retryDelay": "23s"}]}}
    _, engine = gemini(FakeResponse(429, retry), FakeResponse(503, text="overloaded"), audio_response())
    audio = engine.synth("текст")
    assert len(audio) > 0 and 23.0 in app.waits and len(engine.client.calls) == 3


@pytest.mark.parametrize("response, phrase", [
    (FakeResponse(400, text='{"error": {"message": "API key not valid"}}'), "API-ключ"),
    (FakeResponse(404, text="models/x is not found"), "не найдена"),
    (FakeResponse(429, text="Quota exceeded: GenerateRequestsPerDay"), "Дневной лимит"),
])
def test_gemini_fatal_errors_stop_without_retry(gemini, response, phrase):
    _, engine = gemini(response)
    with pytest.raises(RuntimeError, match=phrase):
        engine.synth("текст")
    assert len(engine.client.calls) == 1


def test_gemini_daily_limit_is_enforced_and_usage_is_recorded(app, gemini, monkeypatch):
    monkeypatch.setattr(app, "pacific_today", lambda: date(2026, 11, 11))
    ctx, engine = gemini()
    ctx.update(gemini_daily_limit=2)
    engine.synth("раз")
    engine.synth("два")
    with pytest.raises(RuntimeError, match="Дневной лимит"):
        engine.synth("три")
    assert ctx.usage.requests_today("gemini-3.8-flash-tts") == 2
    monkeypatch.setattr(app, "pacific_today", lambda: date(2026, 11, 12))
    assert len(engine.synth("завтра")) > 0                                # новый день — лимит снова свободен


def test_gemini_cost_and_price_change(app):
    now = app.gemini_cost("gemini-3.8-flash-tts", 0, 1_000_000, date(2026, 11, 11))
    later = app.gemini_cost("gemini-3.8-flash-tts", 0, 1_000_000, date(2027, 1, 2))
    assert now == pytest.approx(9.0) and later == pytest.approx(18.0)
    assert app.gemini_cost("gemini-2.5-pro-preview-tts", 1, 1, date(2026, 11, 11)) is None


def test_usage_summary_by_day_and_month(app, monkeypatch):
    ctx = app.AppContext()
    for day in (date(2026, 11, 11), date(2026, 12, 1)):
        monkeypatch.setattr(app, "pacific_today", lambda day=day: day)
        ctx.usage.record("gemini-3.8-flash-tts", 100, 25_000, 1000.0)
    assert ctx.usage.summarize("2026-11")["requests"] == 1
    total = ctx.usage.summarize()
    assert total["requests"] == 2 and total["cost"] > 0 and total["unknown"] is False


# ---------------------------------------------------------------- конвейер: куски, дозапуск, склейка

def make_book(app, ctx, name="story.txt", chapters=2, paragraphs=3):
    path = app.home / name
    path.write_text("".join(f"Глава {i}. Название {i}\n\n" + body(paragraphs) + "\n\n" for i in range(1, chapters + 1)),
                    encoding="utf-8")
    book = app.Book(ctx, path, app.file_title(path))
    book.state.profile, book.state.confirmed = book.profile_used, True
    book.save()
    book.export_chapters()
    app.FFMPEG = shutil.which("ffmpeg")
    return book


@needs_ffmpeg
def test_pipeline_creates_one_folder_per_book(app):
    ctx = app.AppContext()
    book = make_book(app, ctx)
    assert app.AudiobookService(ctx).run(book, "dummy", {1, 2})
    folder = app.home / "story"
    assert sorted(p.name for p in folder.glob("*.mp3")) == ["01_Название_1.mp3", "02_Название_2.mp3"]
    assert (folder / "chapters" / "split.json").exists() and not (folder / "temp_chunks").exists()


@needs_ffmpeg
def test_pipeline_resumes_after_crash_and_result_is_complete(app, monkeypatch):
    frames = []
    real_assemble = app.assemble_chapter

    def spy(d, total, sample_rate, pause_sec, bitrate, out_path, **kwargs):
        frames.append(sum(sf.info(str(app.chunk_path(d, i))).frames for i in range(1, total + 1)))
        return real_assemble(d, total, sample_rate, pause_sec, bitrate, out_path)
    monkeypatch.setattr(app, "assemble_chapter", spy)

    ctx = app.AppContext()
    service = app.AudiobookService(ctx)
    book = make_book(app, ctx, "clean.txt", chapters=1, paragraphs=12)
    service.run(book, "dummy", {1})
    clean_frames = frames[0]

    calls = {"n": 0}
    real_synth = app.DummyEngine.synth

    def flaky(self, text):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("падение")
        return real_synth(self, text)
    monkeypatch.setattr(app.DummyEngine, "synth", flaky)
    book = make_book(app, ctx, "crash.txt", chapters=1, paragraphs=12)
    assert service.run(book, "dummy", {1}) is False
    done = [p for p in (app.home / "crash" / "temp_chunks" / "dummy" / "001").glob("*.wav")]
    assert len(done) == 2 and app.chapter_status(ctx, book, book.chapters[0])[0] == "part"

    assert service.run(book, "dummy", {1})
    assert frames[1] == clean_frames


@needs_ffmpeg
def test_changing_the_voice_resets_a_half_done_chapter(app, monkeypatch):
    ctx = app.AppContext()
    ctx.update(gemini_api_key="AIzaTESTKEY0000001")
    book = make_book(app, ctx, chapters=1)
    engine = app.DummyEngine(ctx)
    d = app.work_dir(book.dir, "gemini", book.chapters[0])
    chunks = app.prepare_chunks(ctx, book.chapters[0], d, "gemini")
    assert app.synthesize_chapter(engine, d, chunks[:1] + chunks[1:])       # озвучены все куски, но не склеены
    assert app.chapter_status(ctx, book, book.chapters[0])[0] == "part"
    ctx.update(gemini_voice="Puck")
    assert app.chapter_status(ctx, book, book.chapters[0])[0] == "new"


def test_estimate_counts_requests_for_chunk_size_and_formats_for_console(app):
    ctx = app.AppContext()
    service = app.AudiobookService(ctx)
    book = make_book(app, ctx, chapters=2, paragraphs=12)
    small = service.estimate(book, book.chapters)
    ctx.update(gemini_chunk_chars=4000)
    large = service.estimate(book, book.chapters)
    assert large.requests < small.requests and large.chunk_chars == 4000
    text = "\n".join(app.format_estimate(large))
    assert "Расчёт для Gemini" in text and "Тариф бесплатный" in text
    ctx.update(gemini_daily_limit=3)
    assert service.estimate(book, book.chapters).days >= 2


# ---------------------------------------------------------------- сервис: события, «Стоп», готовность, фабрика

def test_event_bus_delivers_events_to_every_subscriber(app):
    bus, first, second = app.EventBus(), [], []
    bus.subscribe(lambda kind, data: first.append((kind, data)))
    bus.subscribe(lambda kind, data: second.append(kind))
    bus.log("привет")
    bus.emit("chunk_done", index=1)
    assert first == [("log", {"text": "привет"}), ("chunk_done", {"index": 1})] and second == ["log", "chunk_done"]


def test_cancellation_token_wait_returns_at_once_after_cancel(app, monkeypatch):
    monkeypatch.setattr(app.CancellationToken, "wait", app.real_wait)
    token = app.CancellationToken()
    assert token.wait(0.01) is False
    token.cancel()
    assert token.wait(10) is True                                  # не ждёт десять секунд
    with pytest.raises(app.Cancelled):
        token.raise_if_cancelled()


@needs_ffmpeg
def test_service_run_reports_progress_through_events(app):
    ctx = app.AppContext()
    bus, events = app.EventBus(), []
    bus.subscribe(lambda kind, data: events.append((kind, data)))
    book = make_book(app, ctx, chapters=2, paragraphs=12)
    assert app.AudiobookService(ctx, bus).run(book, "dummy", {1, 2})
    kinds = [kind for kind, _ in events]
    for expected in ("chapter_started", "chunk_started", "chunk_done", "chapter_done", "run_finished"):
        assert expected in kinds
    chunk = next(data for kind, data in events if kind == "chunk_done")
    assert chunk["index"] == 1 and chunk["total"] >= 2 and "готово" in chunk["text"]
    assert events[-1][1]["ok"] is True and kinds.count("chapter_done") == 2


@needs_ffmpeg
def test_stop_button_stops_after_current_chunk_and_run_continues_later(app):
    ctx = app.AppContext()
    token, bus, events = app.CancellationToken(), app.EventBus(), []
    bus.subscribe(lambda kind, data: events.append((kind, data)))
    bus.subscribe(lambda kind, data: token.cancel() if kind == "chunk_done" and data["index"] == 1 else None)
    service = app.AudiobookService(ctx, bus)
    book = make_book(app, ctx, chapters=1, paragraphs=12)
    assert service.run(book, "dummy", {1}, token) is False
    assert events[-1][0] == "run_finished" and events[-1][1]["stopped"] is True
    assert len(list((app.home / "story" / "temp_chunks" / "dummy" / "001").glob("*.wav"))) == 1
    assert service.run(book, "dummy", {1}) is True                  # новый запуск достраивает главу
    assert len(list((app.home / "story").glob("*.mp3"))) == 1


def test_run_refuses_to_start_when_the_engine_is_not_ready(app, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    ctx = app.AppContext()
    bus, errors = app.EventBus(), []
    bus.subscribe(lambda kind, data: errors.append(data["text"]) if kind == "error" else None)
    book = make_book(app, ctx)
    assert app.AudiobookService(ctx, bus).run(book, "gemini", {1}) is False
    assert any("API-ключа" in text for text in errors)
    fixable, manual = app.AudiobookService(ctx).readiness("gemini")
    assert manual and not (app.home / "story" / "temp_chunks").exists()


def test_chapter_rows_and_pending(app):
    ctx = app.AppContext()
    service = app.AudiobookService(ctx)
    book = make_book(app, ctx, chapters=2)
    rows = service.chapter_rows(book, "gemini")
    assert [r.kind for r in rows] == ["new", "new"] and rows[0].chunks >= 1 and rows[0].chars > 1000
    (book.dir / book.chapters[0]["name"]).with_suffix(".mp3").write_bytes(b"x")
    assert [ch["id"] for ch in service.pending(book, {1, 2})] == [2]
    assert service.chapter_rows(book, "gemini")[0].kind == "done"


def test_engine_factory_and_base_class(app):
    ctx = app.AppContext()
    assert isinstance(app.make_engine(ctx, "dummy"), app.DummyEngine)
    assert isinstance(app.make_engine(ctx, "неизвестный"), app.DummyEngine)
    assert set(app.ENGINES) == {"qwen", "gemini", "dummy"}
    with pytest.raises(NotImplementedError):
        app.Engine(ctx).synth("текст")


# ---------------------------------------------------------------- контракт с библиотекой Qwen

def fake_torch(capability, vram_gb, name="NVIDIA GeForce RTX 3060"):
    return types.SimpleNamespace(
        float16="float16", float32="float32",
        cuda=types.SimpleNamespace(
            is_available=lambda: True, synchronize=lambda: None, empty_cache=lambda: None,
            get_device_name=lambda index: name, get_device_capability=lambda index: capability,
            get_device_properties=lambda index: types.SimpleNamespace(total_memory=vram_gb * 1024 ** 3)))


def test_qwen_engine_calls_the_library_with_expected_arguments(app, monkeypatch):
    calls = []

    class FakeModel:
        model = object()

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return cls()

        def generate_voice_clone(self, **kwargs):
            calls.append(kwargs)
            return [np.ones(12000, dtype="float32") * 0.1], 24000

    torch = fake_torch((8, 6), 8)
    patching = types.SimpleNamespace(find_patchable_model=lambda m: m, apply_triton_kernels=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "faster_qwen3_tts", types.SimpleNamespace(FasterQwen3TTS=FakeModel))
    monkeypatch.setitem(sys.modules, "qwen3_tts_triton", types.ModuleType("qwen3_tts_triton"))
    monkeypatch.setitem(sys.modules, "qwen3_tts_triton.models", types.ModuleType("models"))
    monkeypatch.setitem(sys.modules, "qwen3_tts_triton.models.patching", patching)

    voice_dir = app.home / "голоса"
    voice_dir.mkdir()
    sf.write(str(voice_dir / "anna.wav"), np.zeros(24000, dtype="float32"), 24000)
    ctx = app.AppContext()
    ctx.update(qwen_language="English", qwen_language_auto=False, qwen_voice="Анна",
               qwen_voices=[{"name": "Другой", "path": "x.wav", "text": "Не тот."},
                            {"name": "Анна", "path": "голоса/anna.wav", "text": "Точный текст."}])
    audio = app.QwenEngine(ctx).synth("Привет.")
    assert audio.shape == (12000,) and audio.dtype == np.float32
    assert calls[0]["language"] == "English" and calls[0]["ref_text"] == "Точный текст."
    assert Path(calls[0]["ref_audio"]) == app.home / "голоса" / "anna.wav" and calls[0]["non_streaming_mode"] is True


# ---------------------------------------------------------------- правило названия книги

@pytest.mark.parametrize("name, placeholder", [
    ("1", True), ("02", True), ("2024", True), ("New Text Document (2)", True), ("Новый текстовый документ", True),
    ("book 3", True), ("untitled", True), ("Дневники голодной акулы", False), ("1984 Orwell", False), ("Война и мир", False),
])
def test_placeholder_names_are_not_used_as_book_titles(app, name, placeholder):
    assert app.is_placeholder_title(name) is placeholder
    assert (app.file_title(app.home / f"{name}.txt") is None) is placeholder


# ---------------------------------------------------------------- окно: презентер без Tk

class FakeView:
    """Заглушка окна: запоминает, что презентер просил показать, и отвечает на вопросы заранее заданными ответами."""

    def __init__(self):
        self.rows = self.summary = self.title = self.hint = self.needs_input = self.engine = self.profile = None
        self.hint_alarm = None
        self.estimate, self.progress, self.can_start, self.running = "", (0, ""), False, False
        self.start_label, self.book_name, self.selected, self.notifications, self.rebuilds = "Озвучить", None, set(), [], 0
        self.row_updates = []
        self.log, self.errors, self.infos, self.opened, self.questions = [], [], [], [], []
        self.yes, self.texts = True, []

    def show_rows(self, rows, summary, warnings, selected=()):
        self.rows, self.summary, self.warnings, self.selected = rows, summary, warnings, set(selected)

    def set_book_name(self, name):
        self.book_name = name

    def rebuild(self):
        self.rebuilds += 1

    def set_voices(self, items, index):
        self.voice_items, self.voice_index = items, index

    def set_languages(self, items, index, enabled):
        self.language_items, self.language_index, self.language_enabled = items, index, enabled

    def set_preview(self, enabled, busy=False):
        self.preview_enabled, self.preview_busy = enabled, busy

    def stop_sound(self):
        self.sound_stopped = getattr(self, "sound_stopped", 0) + 1

    def sound_playing(self):
        return getattr(self, "playing_now", False)

    def apply_theme(self, choice):
        self.themes = getattr(self, "themes", []) + [choice]

    def ask_book_info(self, info):
        self.info_asked = dict(info)
        return getattr(self, "info_answer", None)

    def play(self, path):
        self.played = getattr(self, "played", []) + [path]

    def update_row(self, chapter_id, done, total, engine):
        self.row_updates.append((chapter_id, done, total, engine))

    def notify_done(self, ok, stopped):
        self.notifications.append((ok, stopped))

    def set_title(self, text, hint, needs_input):
        self.title, self.hint, self.needs_input = text, hint, needs_input

    def set_hint(self, text, alarm):
        self.hint, self.hint_alarm = text, alarm

    def set_engine(self, name):
        self.engine = name

    def set_profile(self, key):
        self.profile = key

    def set_estimate(self, text):
        self.estimate = text

    def set_progress(self, fraction, text):
        self.progress = (fraction, text)

    def set_state(self, can_start, running, start_label="Озвучить"):
        self.can_start, self.running, self.start_label = can_start, running, start_label

    def set_stopping(self):
        self.stopping = True

    def append_log(self, text):
        self.log.append(text)

    def ask_yes(self, title, text):
        self.questions.append((title, text))
        return self.yes

    def ask_text(self, title, prompt, secret=False, initial=""):
        self.questions.append((title, prompt))
        self.initial_text = initial
        return self.texts.pop(0) if self.texts else None

    def ask_split(self, chapter_title, choices):
        self.split_choices = choices
        return self.split_answer(choices) if callable(getattr(self, "split_answer", None)) else None

    def show_error(self, text):
        self.errors.append(text)

    def show_info(self, text):
        self.infos.append(text)

    def open_path(self, path):
        self.opened.append(path)

    def notify(self, text):
        self.infos.append(text)

    def set_busy(self, text):
        self.busy = text

    def show_settings(self, make_presenter):
        self.settings_factory = make_presenter


@pytest.fixture
def window(app, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    ctx = app.AppContext()
    view = FakeView()
    presenter = app.MainPresenter(app.AudiobookService(ctx, app.EventBus()), view, runner=lambda work: work())
    return ctx, view, presenter


def write_book(app, name, chapters=2, paragraphs=3):
    path = app.home / name
    path.write_text("".join(f"Глава {i}. Название {i}\n\n" + body(paragraphs) + "\n\n" for i in range(1, chapters + 1)),
                    encoding="utf-8")
    return path


def test_opening_a_named_file_suggests_its_name_and_shows_the_chapters(app, window):
    _, view, presenter = window
    presenter.open_file(write_book(app, "Дневники.txt"))
    assert view.title == "Дневники" and view.needs_input is False and "имени файла" in view.hint
    assert [r.id for r in view.rows] == [1, 2] and view.can_start is True and "Глав: 2" in view.summary
    assert "по заголовкам" in view.summary and view.estimate.startswith("Осталось: 2 главы")


def test_nameless_file_must_get_a_typed_title_and_digits_are_allowed(app, window):
    _, view, presenter = window
    presenter.open_file(write_book(app, "1.txt"))
    assert view.title == "" and view.needs_input is True and view.can_start is False
    presenter.title_edited("1984")
    assert view.can_start is True and view.hint_alarm is False
    presenter.title_edited("")
    assert view.can_start is False and view.hint_alarm is True


def test_unreadable_file_shows_an_error(app, window):
    _, view, presenter = window
    broken = app.home / "broken.txt"
    broken.write_bytes(b"\xff\xfe\x00\x80\x81")
    presenter.open_file(broken)
    assert view.errors and "UTF-8" in view.errors[0] and view.rows is None


@needs_ffmpeg
def test_full_run_updates_progress_statuses_and_remembers_the_title(app, window):
    ctx, view, presenter = window
    path = write_book(app, "1.txt", paragraphs=12)
    presenter.open_file(path)
    presenter.title_edited("Моя книга")
    presenter.set_engine("dummy")
    presenter.start()
    presenter.pump()
    assert view.progress == (1.0, "Готово") and view.running is False
    assert [r.kind for r in view.rows] == ["done", "done"] and any("🎉" in line for line in view.log)
    assert (app.home / "Моя книга" / "chapters" / "split.json").exists()
    again_view = FakeView()
    again = app.MainPresenter(app.AudiobookService(app.AppContext(), app.EventBus()), again_view, runner=lambda work: work())
    again.open_file(path)
    assert again_view.title == "Моя книга" and "запомнено" in again_view.hint


def test_start_asks_for_a_gemini_key_and_stops_when_it_is_not_given(app, window, monkeypatch):
    ctx, view, presenter = window
    monkeypatch.setattr(app, "ensure_ffmpeg", lambda interactive=True: True)       # на раннере Windows ffmpeg нет
    presenter.open_file(write_book(app, "Рассказ.txt"))
    presenter.set_engine("gemini")
    presenter.start()                                          # ключ не ввели
    assert view.questions[0][0] == "API-ключ Gemini" and ctx.settings.gemini_api_key == "" and not view.running
    view.texts, view.yes = ['"AIzaTESTKEY0000001"'], False
    presenter.start()                                          # ключ ввели, но от расчёта отказались
    assert ctx.settings.gemini_api_key == "AIzaTESTKEY0000001"
    assert view.questions[-1][0] == "Расчёт для Gemini" and "Начать озвучку?" in view.questions[-1][1]
    assert not view.running and not (app.home / "Рассказ" / "chapters" / "split.json").exists()


def test_start_explains_what_is_missing_for_qwen(app, window, monkeypatch):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt"))
    presenter.set_engine("qwen")
    opened = []
    monkeypatch.setattr(presenter, "open_settings", lambda: opened.append(True))
    presenter.start()                                                            # голоса нет: предлагаем добавить
    assert view.questions[-1][0] == "Нужен голос" and opened == [True] and not view.running
    sf.write(str(app.home / "a.wav"), np.zeros(48000, dtype="float32"), 24000)
    ctx.update(qwen_voices=[{"name": "Анна", "path": "a.wav", "text": "Текст."}])
    monkeypatch.setattr(app, "ensure_ffmpeg", lambda interactive=True: True)
    monkeypatch.setattr(app.AudiobookService, "readiness", lambda self, engine: ([], ["нет видеокарты"]))
    presenter.start()                                                            # голос есть, но движок не готов
    assert view.errors and "Движок пока не готов" in view.errors[-1] and not view.running


def test_chapter_started_by_another_engine_is_confirmed_before_starting(app, window, monkeypatch):
    ctx, view, presenter = window
    monkeypatch.setattr(app, "ensure_ffmpeg", lambda interactive=True: True)       # на раннере Windows ffmpeg нет
    presenter.open_file(write_book(app, "Рассказ.txt", paragraphs=12))
    chapter = presenter.book.chapters[0]
    folder = app.work_dir(presenter.book.dir, "qwen", chapter)
    chunks = app.prepare_chunks(ctx, chapter, folder, "qwen")
    sf.write(str(app.chunk_path(folder, 1)), np.zeros(24000, dtype="float32"), 24000)
    assert len(chunks) > 1
    presenter.set_engine("dummy")
    view.yes = False
    presenter.start()
    assert view.questions[-1][0] == "Глава уже начата другим движком" and not view.running


def test_stop_and_close_while_running(app, window):
    _, view, presenter = window
    presenter.running, presenter.token = True, app.CancellationToken()
    presenter.stop()
    with pytest.raises(app.Cancelled):
        presenter.token.raise_if_cancelled()
    presenter.token = app.CancellationToken()
    view.yes = False
    assert presenter.request_close() is False
    view.yes = True
    assert presenter.request_close() is True
    with pytest.raises(app.Cancelled):
        presenter.token.raise_if_cancelled()


def test_changing_the_split_to_a_custom_rule_reloads_the_table(app, window):
    _, view, presenter = window
    path = app.home / "Эпизоды.txt"
    path.write_text("".join(f"\n\nЭПИЗОД {i}\n\n" + body(3) for i in (1, 2, 3)), encoding="utf-8")
    presenter.open_file(path)
    assert len(view.rows) == 1
    view.texts = [r"^ЭПИЗОД \d+"]
    presenter.set_profile("custom")
    assert len(view.rows) == 3 and "по своему правилу" in view.summary


def test_progress_follows_events_from_the_worker(app, window):
    _, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt"))
    events = presenter.service.events
    events.emit("chapter_started", chapter=2, index=2, total=4, text="Глава 2")
    events.emit("chunk_done", index=1, total=2, text="кусок")
    presenter.pump()
    assert view.progress[0] == pytest.approx((1 + 0.5) / 4) and "кусок 1 из 2" in view.progress[1]
    assert view.log == ["Глава 2", "кусок"]


def test_open_folder_only_when_it_exists(app, window):
    _, view, presenter = window
    presenter.open_folder()                                    # книги нет
    presenter.open_file(write_book(app, "Рассказ.txt"))
    presenter.open_folder()
    assert len(view.infos) == 2 and view.opened == []
    presenter.book.export_chapters()
    presenter.open_folder()
    assert view.opened == [presenter.book.dir]


def test_estimate_line_for_paid_and_free_tariffs(app):
    ctx = app.AppContext()
    book = make_book(app, ctx, chapters=2, paragraphs=12)
    service = app.AudiobookService(ctx)
    free = app.estimate_line(service.estimate(book, book.chapters))
    assert free.startswith("Осталось: 2 главы") and "$" not in free
    ctx.update(gemini_daily_limit=3)
    assert "дней" in app.estimate_line(service.estimate(book, book.chapters))
    ctx.update(gemini_paid=True)
    assert "примерно $" in app.estimate_line(service.estimate(book, book.chapters))


# ---------------------------------------------------------------- настоящее окно (если есть Tk и экран)

def test_real_window_opens_loads_a_dropped_book_and_closes(app):
    pytest.importorskip("tkinterdnd2")
    pytest.importorskip("sv_ttk")
    tk = pytest.importorskip("tkinter")
    try:
        service = app.AudiobookService(app.AppContext(), app.EventBus())
        window = app.MainWindow(lambda view: app.MainPresenter(service, view))
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")
    try:
        book = write_book(app, "Мой роман.txt", chapters=3)
        dropped = types.SimpleNamespace(data="{" + str(book) + "}", action="copy")      # так приходит путь при перетаскивании
        window._on_drop(dropped)
        window.root.update()
        assert window.title_var.get() == "Мой роман" and len(window.table.get_children()) == 3
        assert "Всё верно" in window.hint.cget("text")
        assert str(window.start_button.cget("style")) == "Accent.TButton"
        assert window.presenter.request_close() is True
    finally:
        window.root.destroy()


# ---------------------------------------------------------------- установка: вывод pip в журнал, «Стоп», доустановка

def test_pip_streams_lines_to_events_and_stop_ends_the_process(app, monkeypatch):
    monkeypatch.setattr(app, "in_project_venv", lambda: True)
    monkeypatch.setattr(app, "pip_command",
                        lambda args: [sys.executable, "-c", "print('first', flush=True); print('second', flush=True)"])
    bus, lines = app.EventBus(), []
    bus.subscribe(lambda kind, data: lines.append(data["text"]))
    assert app.pip("check", events=bus) is True
    assert "first" in lines and "second" in lines and lines[0] == "   $ pip check"

    monkeypatch.setattr(app, "pip_command",
                        lambda args: [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(30)"])
    monkeypatch.setattr(app.CancellationToken, "wait", app.real_wait)
    token = app.CancellationToken()
    bus.subscribe(lambda kind, data: token.cancel() if data.get("text") == "start" else None)
    started = time.time()
    assert app.pip("check", events=bus, token=token) is False
    assert time.time() - started < 15                              # не ждали 30 секунд


def test_install_engine_for_gemini_and_ffmpeg_and_failures(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "pip", lambda *a, **k: calls.append(a) or True)
    assert app.install_engine("gemini") is True and calls[-1] == ("install", app.REQUESTS_PIN)
    monkeypatch.setattr(app, "ensure_ffmpeg", lambda interactive=True: True)
    assert app.install_engine("ffmpeg") is True and calls[-1] == ("install", app.IMAGEIO_FFMPEG_PIN)
    monkeypatch.setattr(app, "pip", lambda *a, **k: False)
    assert app.install_engine("gemini") is False


def test_install_qwen_runs_every_step_and_asks_for_restart_when_loaded_libraries_change(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "pip", lambda *a, **k: calls.append(a) or True)
    monkeypatch.setattr(app, "torch_cuda_status", lambda: (None, False))
    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="OK", stderr="", returncode=0))
    versions = iter(["2.5.4 0.14.0 2.1.1", "2.5.3 0.14.0 2.1.1"])
    monkeypatch.setattr(app, "loaded_lib_versions", lambda: next(versions))
    with pytest.raises(app.RestartRequired):
        app.install_engine("qwen")
    flat = [" ".join(call) for call in calls]
    assert any("torch" in line and "--index-url" in line for line in flat)
    assert any(app.QWEN_TRITON_PKG in line and "--no-deps" in line for line in flat) and len(flat) >= 6
    versions = iter(["то же", "то же"])
    assert app.install_engine("qwen") is True


def test_install_missing_installs_ffmpeg_then_engine_and_reports_through_events(app, monkeypatch):
    bus, events = app.EventBus(), []
    bus.subscribe(lambda kind, data: events.append((kind, data)))
    service = app.AudiobookService(app.AppContext(), bus)
    installed, ffmpeg = [], {"ready": False}
    monkeypatch.setattr(app, "ensure_ffmpeg", lambda interactive=True: ffmpeg["ready"])
    monkeypatch.setattr(app, "install_engine",
                        lambda name, events=None, token=None: installed.append(name) or ffmpeg.update(ready=True) or True)
    monkeypatch.setattr(service, "readiness", lambda name: (["нет пакета"], []))
    assert service.install_missing("qwen") is True
    assert installed == ["ffmpeg", "qwen"] and events[-1] == ("install_done", {"ok": True, "restart": False, "text": ""})

    monkeypatch.setattr(app, "install_engine", lambda *a, **k: False)
    assert service.install_missing("qwen") is False and events[-1][1]["ok"] is False

    def needs_restart(*a, **k):
        raise app.RestartRequired()
    monkeypatch.setattr(app, "install_engine", needs_restart)
    assert service.install_missing("qwen") is True and events[-1][1]["restart"] is True


def test_missing_packages_are_offered_for_installation_and_the_run_starts_afterwards(app, window, monkeypatch):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt"))
    presenter.set_engine("dummy")
    state, runs, installs = {"installed": False}, [], []
    service = presenter.service
    monkeypatch.setattr(service, "readiness", lambda name: ([] if state["installed"] else ["нет пакета X"], []))
    monkeypatch.setattr(app, "ensure_ffmpeg", lambda interactive=True: True)

    def fake_install(engine, token=None):
        installs.append(engine)
        state["installed"] = True
        service.events.emit("install_done", ok=True, restart=False, text="")
        return True
    monkeypatch.setattr(service, "install_missing", fake_install)
    monkeypatch.setattr(service, "run", lambda *a, **k: runs.append(a) or True)

    view.yes = False
    presenter.start()                                              # отказались от установки
    assert view.questions[-1][0] == "Нужна установка" and installs == [] and runs == []
    view.yes = True
    presenter.start()
    presenter.pump()
    assert installs == ["dummy"] and len(runs) == 1                # поставили и сразу продолжили озвучку
    assert "Установка…" in view.busy and presenter.running is False


def test_failed_or_restart_installation_does_not_resume_the_run(app, window, monkeypatch):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt"))
    presenter.set_engine("dummy")
    service, runs = presenter.service, []
    monkeypatch.setattr(service, "readiness", lambda name: (["нет пакета X"], []))
    monkeypatch.setattr(app, "ensure_ffmpeg", lambda interactive=True: True)
    monkeypatch.setattr(service, "run", lambda *a, **k: runs.append(a) or True)
    monkeypatch.setattr(service, "install_missing", lambda engine, token=None: False)
    presenter.start()
    presenter.pump()
    assert view.errors and "Установка не удалась" in view.errors[-1] and runs == []

    def restart(engine, token=None):
        service.events.emit("install_done", ok=True, restart=True, text="")
        return True
    monkeypatch.setattr(service, "install_missing", restart)
    presenter.start()
    presenter.pump()
    assert "заново" in view.infos[-1] and runs == []


# ---------------------------------------------------------------- окно «Настройки»: презентер без Tk

class FakeSettingsView:
    def __init__(self):
        self.values = self.ref_present = self.key_from_env = None
        self.status, self.errors, self.questions, self.texts, self.opened = ("", False), [], [], [], []
        self.played, self.voice_asked, self.voice_answers = [], [], []
        self.yes, self.file, self.ref_status = True, None, None

    def show_values(self, values, key_from_env):
        self.values, self.key_from_env = values, key_from_env

    def show_voices(self, voices, chosen):
        self.voices, self.chosen_voice = [dict(v) for v in voices], chosen

    def set_busy(self, name, busy):
        self.busy = getattr(self, "busy", {})
        self.busy[name] = busy

    def ask_voice(self, voice):
        self.voice_asked.append(dict(voice))
        return self.voice_answers.pop(0) if self.voice_answers else None

    def set_status(self, text, alarm):
        self.status = (text, alarm)

    def ask_yes(self, title, text):
        self.questions.append((title, text))
        return self.yes

    def ask_open_file(self, title, filetypes):
        return self.file

    def show_error(self, text):
        self.errors.append(text)

    def open_path(self, path):
        self.opened.append(path)

    def play(self, path):
        self.played.append(path)

    def restart(self):
        self.restarted = True

    def show_text(self, title, text):
        self.texts.append((title, text))


@pytest.fixture
def settings_window(app, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    ctx, view, saved = app.AppContext(), FakeSettingsView(), []
    presenter = app.SettingsPresenter(ctx, view, on_saved=lambda: saved.append(True),
                                      extra_report=lambda: ["Осталось: 2 главы"], runner=lambda work: work())
    presenter.load()
    return ctx, view, presenter, saved


def raw_values(view, **changes):
    values = dict(view.values)
    values.update(changes)
    return values


def test_settings_load_current_values_and_notes_about_key_and_reference(app, settings_window, monkeypatch):
    ctx, view, presenter, _ = settings_window
    assert view.values["gemini_voice"] == "Kore" and view.values["gemini_chunk_chars"] == 1200
    assert view.voices == [] and view.chosen_voice == "" and view.key_from_env is False
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFROMENV000000001")
    presenter.load()
    assert view.key_from_env is True


def test_settings_save_only_changed_values_clean_the_key_and_notify(app, settings_window):
    ctx, view, presenter, saved = settings_window
    raw = raw_values(view, gemini_api_key=' "AIzaTESTKEY0000001" ', gemini_voice="Puck (Upbeat)", gemini_chunk_chars="4000",
                     gemini_daily_limit="10")
    assert presenter.save(raw) is True and saved == [True]
    stored = json.loads(app.PATHS.settings.read_text(encoding="utf-8"))
    stored["gemini_api_key"] = app.unprotect_secret(stored["gemini_api_key"])       # на Windows ключ в файле зашифрован
    assert stored.pop("schema_version") == app.SCHEMA_VERSION
    assert stored == {"gemini_api_key": "AIzaTESTKEY0000001", "gemini_voice": "Puck", "gemini_chunk_chars": 4000,
                      "gemini_daily_limit": 10}


@pytest.mark.parametrize("field, value, phrase", [
    ("gemini_chunk_chars", "50", "Размер куска"), ("gemini_chunk_chars", "много", "Размер куска"),
    ("gemini_daily_limit", "-1", "Суточный лимит"), ("gemini_model", "  ", "Модель"), ("gemini_voice", "", "Голос"),
])
def test_settings_reject_wrong_values_and_save_nothing(app, settings_window, field, value, phrase):
    ctx, view, presenter, saved = settings_window
    assert presenter.save(raw_values(view, **{field: value})) is False
    assert phrase in view.errors[0] and saved == [] and not app.PATHS.settings.exists()


def test_settings_ask_before_saving_a_very_large_chunk(app, settings_window):
    ctx, view, presenter, _ = settings_window
    view.yes = False
    assert presenter.save(raw_values(view, gemini_chunk_chars="7000")) is False
    assert view.questions[0][0] == "Большой кусок" and ctx.settings.gemini_chunk_chars == 1200
    view.yes = True
    assert presenter.save(raw_values(view, gemini_chunk_chars="7000")) is True and ctx.settings.gemini_chunk_chars == 7000


def test_key_info_checks_the_key_and_shows_limits_and_costs_in_one_window(app, settings_window, monkeypatch):
    ctx, view, presenter, _ = settings_window
    ctx.usage.record("gemini-3.8-flash-tts", 100, 25000, 1000.0)
    models = FakeResponse(200, {"models": [{"name": "models/gemini-3.8-flash-tts"}]})
    monkeypatch.setattr(app, "GeminiClient", lambda key: FakeClient(models))
    presenter.key_info(raw_values(view, gemini_api_key="AIzaTESTKEY0000001"))
    assert view.busy == {"key": True}
    presenter.key_info(raw_values(view, gemini_api_key="AIzaTESTKEY0000001"))      # второе нажатие, пока идёт проверка
    presenter.pump()
    assert len(view.texts) == 1 and view.busy == {"key": False}
    title, text = view.texts[0]
    assert title == "Ключ и лимиты Gemini" and "Ключ работает" in text and "gemini-3.8-flash-tts" in text
    assert "Gemini · тариф: бесплатный" in text and "Осталось: 2 главы" in text and view.status[1] is False
    monkeypatch.setattr(app, "GeminiClient", lambda key: FakeClient(FakeResponse(400, text="API key not valid")))
    presenter.key_info(raw_values(view, gemini_api_key="AIzaPLOHOY0000000001"))
    presenter.pump()
    assert view.status[1] is True and "Ключ не принят" in view.texts[-1][1]
    presenter.key_info(raw_values(view, gemini_api_key=""))
    presenter.pump()
    assert "Ключ не задан" in view.texts[-1][1]


def test_diagnostics_button_cannot_be_pressed_twice_while_collecting(app, settings_window):
    ctx, view, presenter, runs = settings_window
    presenter.runner = lambda job: runs.append(job)
    presenter.show_diagnostics()
    presenter.show_diagnostics()
    assert len(runs) == 1 and view.busy == {"diagnostics": True}
    runs[0]()
    presenter.pump()
    lines = view.texts[-1][1].splitlines()
    assert view.busy == {"diagnostics": False}
    assert any(line in ("ffmpeg: +", "ffmpeg: −") for line in lines) and not any("✅" in line or "❌" in line for line in lines)


def test_main_window_opens_settings_only_when_idle_and_refreshes_after_saving(app, window):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt", chapters=2, paragraphs=12))
    presenter.running = True
    presenter.open_settings()
    assert view.infos and not hasattr(view, "settings_factory")
    presenter.running = False
    presenter.open_settings()
    dialog_view = FakeSettingsView()
    settings_presenter = view.settings_factory(dialog_view)
    settings_presenter.load()
    presenter.set_engine("gemini")
    before = view.estimate
    settings_presenter.save(raw_values(dialog_view, gemini_chunk_chars="4000"))
    assert view.estimate != before                                 # расчёт пересчитан под новый размер куска


# ---------------------------------------------------------------- настоящее окно настроек

def test_real_settings_dialog_opens_with_current_values_and_saves(app):
    pytest.importorskip("tkinterdnd2")
    pytest.importorskip("sv_ttk")
    tk = pytest.importorskip("tkinter")
    ctx = app.AppContext()
    service = app.AudiobookService(ctx, app.EventBus())
    try:
        window = app.MainWindow(lambda view: app.MainPresenter(service, view))
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")
    try:
        dialog = window.show_settings(lambda view: app.SettingsPresenter(ctx, view, runner=lambda work: work()))
        window.root.update()
        assert dialog.model_var.get() == "gemini-3.8-flash-tts" and dialog.chunk_var.get() == "1200"
        dialog.limit_var.set("10")
        dialog.paid_var.set(1)
        dialog._save()
        assert ctx.settings.gemini_daily_limit == 10 and ctx.settings.gemini_paid is True and dialog.closed is True
    finally:
        window.root.destroy()


# ---------------------------------------------------------------- доводка: размер окна, тема, форматы, start.bat

@pytest.mark.parametrize("saved, expected", [
    ("", ""), ("не размер", ""),
    ("900x760", "800x740"),                                # размер всегда стандартный: края окна не тянут
    ("900x760+120+80", "800x740+120+80"),                  # от сохранённого берётся положение
    ("300x200+10+10", "800x740+10+10"),
    ("5000x4000+0+0", "800x740+0+0"),
    ("900x760+3000+2000", "800x740+1120+340"),             # ушло за край экрана: возвращается на экран
    ("900x760-1500+50", "800x740+0+50"),                   # слева за краем (был второй монитор)
])
def test_saved_window_geometry_is_fitted_to_the_screen(app, saved, expected):
    assert app.fit_geometry(saved, 1920, 1080) == expected


def test_theme_follows_the_system_only_in_auto_mode(app, monkeypatch):
    monkeypatch.setattr(app, "system_is_dark", lambda: True)
    assert app.resolve_theme("auto") == "dark" and app.resolve_theme("") == "dark"
    assert app.resolve_theme("light") == "light" and app.resolve_theme("dark") == "dark"
    monkeypatch.setattr(app, "system_is_dark", lambda: False)
    assert app.resolve_theme("auto") == "light"
    assert isinstance(app.system_is_dark(), bool)


@pytest.mark.skipif(os.name == "nt", reason="на Windows это прятало бы консоль самого раннера")
def test_hiding_the_console_does_nothing_outside_windows(app):
    app.set_console_visible(False)
    app.set_console_visible(True)


@pytest.mark.parametrize("name", ["книга.mobi", "книга.DJVU", "scan.pdf", "роман.docx"])
def test_books_in_other_formats_get_a_clear_message(app, window, name):
    _, view, presenter = window
    path = app.home / name
    path.write_bytes(b"PK\x03\x04 not a text")
    presenter.open_file(path)
    assert view.errors and ".txt" in view.errors[0] and "Calibre" in view.errors[0] and view.rows is None


def test_window_position_is_remembered_between_runs(app, window):
    ctx, _, presenter = window
    presenter.remember_geometry("900x760+10+20")
    assert app.AppContext().settings.window_geometry == "900x760+10+20"


def test_sun_and_moon_switch_remembers_the_theme(app, window):
    ctx, view, presenter = window
    presenter.change_theme("dark")
    presenter.change_theme("light")
    assert view.themes == ["dark", "light"] and ctx.settings.theme == "light"
    assert "theme" not in view_values_of_settings(app)


def view_values_of_settings(app):
    view = FakeSettingsView()
    app.SettingsPresenter(app.AppContext(), view, runner=lambda work: work()).load()
    return view.values


def test_start_bat_is_a_windows_script_that_hides_the_console_and_pauses_only_on_errors():
    data = (SCRIPT.parent / "start.bat").read_bytes()
    assert b"\n" not in data.replace(b"\r\n", b"")                    # только переводы строк Windows
    text = data.decode("ascii")
    assert "set AUDIOBOOK_HIDE_CONSOLE=1" in text and "py -%PYVER% audiobook_master.py %*" in text
    assert text.index("set PYVER=3.14") < text.index("set PYVER=3.13") < text.index("set PYVER=3.12")   # сначала 3.14, потом прежние
    assert text.rstrip().endswith("if errorlevel 1 pause")


def test_real_window_in_dark_theme_remembers_its_geometry_and_settings_has_a_theme_choice(app):
    pytest.importorskip("tkinterdnd2")
    pytest.importorskip("sv_ttk")
    tk = pytest.importorskip("tkinter")
    ctx = app.AppContext()
    service = app.AudiobookService(ctx, app.EventBus())
    try:
        window = app.MainWindow(lambda view: app.MainPresenter(service, view), theme="dark", geometry="900x760+50+40")
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")
    try:
        window.root.update()
        assert window.colors["accent"] == app.PALETTES["dark"]["accent"] and window.colors["bg"] != window.colors["fg"]
        assert str(window.log_text.cget("background")) == window.colors["bg"]
        dialog = window.show_settings(lambda view: app.SettingsPresenter(ctx, view, runner=lambda work: work()))
        window.root.update()
        dialog.close()
        window._on_close()
        assert re.fullmatch(r"\d+x\d+\+\d+\+\d+", ctx.settings.window_geometry)
    finally:
        try:
            window.root.destroy()
        except tk.TclError:
            pass                                                      # окно уже закрыто


# ---------------------------------------------------------------- пакет «понятность и визуал»

def test_status_chips_and_color_mixing(app):
    row = app.ChapterRow(1, "Глава 1", "Название", 100, 3, "part", "⏸", 5, 12, "qwen")
    assert app.status_chip(row) == "◐ 5/12 · qwen"
    assert app.status_chip(app.ChapterRow(1, "", "", 0, 0, "done", "")) == "● готова"
    assert app.status_chip(app.ChapterRow(1, "", "", 0, 0, "new", "")) == "○ не начата"
    assert app.mix_colors("#000000", "#ffffff", 0.5) == "#808080" and app.mix_colors("#102030", "#102030", 0.7) == "#102030"
    assert app.valid_color("#abc123", "#000000") == "#abc123" and app.valid_color("SystemWindow", "#fafafa") == "#fafafa"


def test_app_icon_is_a_small_valid_png(app):
    import base64
    data = base64.b64decode(app.APP_ICON_PNG)
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) < 3000


@pytest.mark.parametrize("seconds, expected", [(20, "меньше минуты"), (240, "≈ 4 мин"), (4800, "≈ 1 ч 20 мин")])
def test_remaining_time_text(app, seconds, expected):
    assert app.eta_text(seconds) == expected


@needs_ffmpeg
def test_only_the_selected_chapters_are_voiced_and_the_button_and_estimate_say_so(app, window):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt", chapters=3, paragraphs=12))
    presenter.set_engine("dummy")
    assert view.start_label == "Озвучить" and "Ctrl/Shift" in view.summary and "3 главы" in view.estimate
    presenter.select([2])
    assert view.start_label == "Озвучить выбранные (1)" and view.estimate.startswith("Осталось: 1 глава")
    presenter.start()
    presenter.pump()
    assert [r.kind for r in view.rows] == ["new", "done", "new"] and view.selected == {2}
    assert view.notifications == [(True, False)]
    presenter.select([2])
    assert "уже озвучены" in view.estimate                              # выбранная глава готова
    presenter.select([])
    assert view.start_label == "Озвучить" and "2 главы" in view.estimate


def test_selection_is_ignored_while_running_and_reset_when_the_split_changes(app, window):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt", chapters=3))
    presenter.running = True
    presenter.select([1])
    assert presenter.selected == set()
    presenter.running = False
    presenter.select([1, 3])
    view.texts = [r"^Глава \d+"]
    presenter.set_profile("custom")
    assert presenter.selected == set()


def test_time_left_appears_after_two_chunks_and_uses_the_measured_speed(app, window):
    ctx, view, presenter = window
    presenter.chunks_total, presenter.chapter_index, presenter.chapter_total, presenter.run_started = 10, 1, 2, 0
    presenter.clock = lambda: 60
    events = presenter.service.events
    events.emit("chunk_done", index=1, total=5, text="")
    presenter.pump()
    assert "осталось" not in view.progress[1]                           # один замер: рано судить
    events.emit("chunk_done", index=2, total=5, text="")
    presenter.pump()
    assert view.progress[0] == pytest.approx(0.2) and "осталось ≈ 4 мин" in view.progress[1]   # 8 кусков по 30 с


def test_finishing_stopping_and_failing_give_different_signals(app, window):
    ctx, view, presenter = window
    events = presenter.service.events
    for finished, expected in (({"ok": True, "stopped": False}, (True, False)), ({"ok": False, "stopped": True}, (False, True)),
                               (None, (False, False))):
        presenter.running = True
        if finished:
            events.emit("run_finished", text="", **finished)
        events.emit("worker_exit", job="run", ok=finished["ok"] if finished else False)
        presenter.pump()
        assert view.notifications[-1] == expected


def test_finished_chapter_opens_on_double_click_and_can_be_voiced_again(app, window):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt", chapters=2, paragraphs=12))
    book, chapter = presenter.book, presenter.book.chapters[0]
    presenter.open_chapter([1])
    assert view.infos[-1].startswith("Эта глава ещё не озвучена") and view.opened == []
    assert presenter.row_actions([1]) == {"start": True, "open": False, "redo": False}
    mp3 = app.out_mp3(book.dir, chapter)
    mp3.parent.mkdir(parents=True, exist_ok=True)
    mp3.write_bytes(b"mp3")
    folder = app.work_dir(book.dir, "gemini", presenter.book.chapters[1])        # у второй главы есть недоделанные куски
    app.prepare_chunks(ctx, presenter.book.chapters[1], folder, "gemini")
    sf.write(str(app.chunk_path(folder, 1)), np.zeros(2400, dtype="float32"), 24000)
    presenter.open_chapter([1])
    assert view.opened == [mp3]
    assert presenter.row_actions([1]) == {"start": True, "open": True, "redo": True}
    assert presenter.row_actions([1, 2])["open"] is False and presenter.row_actions([2])["redo"] is True
    view.yes = False
    presenter.redo([1, 2])
    assert mp3.exists()                                                  # отказались: ничего не удалено
    view.yes = True
    presenter.redo([1, 2])
    assert not mp3.exists() and not folder.exists() and [r.kind for r in view.rows] == ["new", "new"]
    assert not (book.dir / "temp_chunks").exists()                       # пустые служебные папки убраны


def test_actions_on_chapters_are_blocked_while_running(app, window):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt"))
    mp3 = app.out_mp3(presenter.book.dir, presenter.book.chapters[0])
    mp3.parent.mkdir(parents=True, exist_ok=True)
    mp3.write_bytes(b"mp3")
    presenter.running = True
    assert presenter.row_actions([1]) == {"start": False, "open": False, "redo": False}
    presenter.open_chapter([1])
    presenter.redo([1])
    assert view.opened == [] and mp3.exists() and view.questions == []


def test_dropping_another_book_while_running_explains_why_nothing_happens(app, window):
    ctx, view, presenter = window
    presenter.running = True
    presenter.open_file(write_book(app, "Рассказ.txt"))
    assert view.infos and "идёт работа" in view.infos[0] and view.rows is None


def test_opening_a_book_tells_the_window_its_file_name(app, window):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt"))
    assert view.book_name == "Рассказ.txt"


def test_diagnostics_have_versions_and_paths_but_no_secrets_and_are_collected_in_the_background(app, settings_window):
    ctx, view, presenter, _ = settings_window
    ctx.update(gemini_api_key="AIzaSECRETKEY123456789", ref_text="Особый текст образца.")
    lines = app.diagnostics_lines(ctx)
    text = "\n".join(lines)
    assert lines[0].startswith(f"Foxy AudioBook Maker {app.APP_VERSION}") and sys.version.split()[0] in lines[1]
    assert str(app.home) in text and "numpy" in text and "ffmpeg:" in text
    assert "AIzaSECRETKEY123456789" not in text and "AIza…" in text
    presenter.show_diagnostics()
    presenter.pump()
    title, shown = view.texts[0]
    assert title.startswith("Диагностика") and "Python" in shown and view.status == ("", False)


def test_service_resets_chapters_of_every_engine(app):
    ctx = app.AppContext()
    service = app.AudiobookService(ctx)
    book = make_book(app, ctx, chapters=2, paragraphs=12)
    for engine_name in ("qwen", "gemini"):
        folder = app.work_dir(book.dir, engine_name, book.chapters[0])
        app.prepare_chunks(ctx, book.chapters[0], folder, engine_name)
    mp3 = app.out_mp3(book.dir, book.chapters[1])
    mp3.parent.mkdir(parents=True, exist_ok=True)
    mp3.write_bytes(b"mp3")
    service.reset_chapters(book, {1, 2})
    assert not mp3.exists() and not (book.dir / "temp_chunks").exists()
    assert [row.kind for row in service.chapter_rows(book, "qwen")] == ["new", "new"]


def test_chapter_rows_carry_progress_data_for_the_window(app):
    ctx = app.AppContext()
    book = make_book(app, ctx, chapters=1, paragraphs=12)
    folder = app.work_dir(book.dir, "gemini", book.chapters[0])
    chunks = app.prepare_chunks(ctx, book.chapters[0], folder, "gemini")
    sf.write(str(app.chunk_path(folder, 1)), np.zeros(2400, dtype="float32"), 24000)
    row = app.AudiobookService(ctx).chapter_rows(book, "gemini")[0]
    assert (row.kind, row.done_chunks, row.total_chunks, row.engine) == ("part", 1, len(chunks), "gemini")
    assert row.label == f"⏸ 1/{len(chunks)} кусков (gemini)" and app.status_chip(row).startswith("◐ 1/")


def test_real_window_selection_menu_log_icon_and_compact_drop_zone(app):
    pytest.importorskip("tkinterdnd2")
    pytest.importorskip("sv_ttk")
    tk = pytest.importorskip("tkinter")
    ctx = app.AppContext()
    service = app.AudiobookService(ctx, app.EventBus())
    try:
        window = app.MainWindow(lambda view: app.MainPresenter(service, view))
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")
    try:
        window.root.update()
        assert int(window.drop.cget("height")) == app.DROP_HEIGHTS[0] and window.log_open is False
        window.presenter.open_file(write_book(app, "Мой роман.txt", chapters=3))
        window.root.update()
        assert int(window.drop.cget("height")) == app.DROP_HEIGHTS[1] and window.book_file == "Мой роман.txt"
        items = window.table.get_children()
        assert [window.table.set(i, "status") for i in items] == ["○ не начата"] * 3
        window.table.selection_set(items[1])
        window.root.update()
        assert window.presenter.selected == {2} and window.start_button.cget("text") == "Озвучить выбранные (1)"
        window.show_rows(window.presenter.service.chapter_rows(window.presenter.book, "qwen"), "сводка", [], {2})
        assert window.selected_ids() == [2]                              # выбор сохранился после обновления таблицы
        window.append_log("   $ pip install что-то")
        assert window.log_open is False and "pip install" in window.status_line.cget("text")
        window.append_log("❌ Ошибка на куске 1")
        assert window.log_open is True and "Ошибка" in window.status_line.cget("text")
        window.set_progress(0.4, "Глава 1")
        assert window.root.title().startswith("40%")
        window.notify_done(True, False)
        assert window.root.title().startswith("✓ Готово")
        window._toggle_log(False)
        assert window.log_open is False and window.log_toggle.cget("text").endswith("▸")
    finally:
        window.root.destroy()


@pytest.mark.parametrize("n, word", [(1, "глава"), (2, "главы"), (4, "главы"), (5, "глав"), (11, "глав"), (14, "глав"),
                                     (21, "глава"), (22, "главы"), (25, "глав"), (101, "глава"), (111, "глав"), (0, "глав")])
def test_russian_plural_forms(app, n, word):
    assert app.ru_plural(n, "глава", "главы", "глав") == word


# ---------------------------------------------------------------- скорость: темп и параллельность Gemini, склейка в фоне

def test_rate_limiter_spaces_requests_and_skips_the_first_wait(app):
    limiter = app.RateLimiter(app.CancellationToken())
    limiter.wait(10)
    assert app.waits == []                                           # первый запрос идёт сразу
    limiter.wait(10)
    assert len(app.waits) == 1 and 9.0 < app.waits[0] <= 10.0
    app.waits.clear()
    limiter.wait(0)
    assert app.waits == []


def test_rate_limiter_wait_can_be_stopped(app, monkeypatch):
    monkeypatch.setattr(app.CancellationToken, "wait", lambda self, seconds: True)
    limiter = app.RateLimiter(app.CancellationToken())
    limiter.wait(10)
    with pytest.raises(app.Cancelled):
        limiter.wait(10)


def test_gemini_pace_and_parallelism_follow_the_tariff(app, gemini):
    ctx, free = gemini()
    assert free.max_parallel == 1 and free.interval == app.GEMINI_FREE_INTERVAL
    ctx.update(gemini_paid=True)
    paid = app.GeminiEngine(ctx, client=FakeClient())
    assert paid.max_parallel == app.GEMINI_PARALLEL_PAID and paid.interval == app.GEMINI_PAID_INTERVAL


class SlowEngine:
    """Движок-заглушка: каждый кусок «озвучивается» заданное время; можно сломать один из кусков."""

    def __init__(self, app, parallel, delay=0.15, fail_on=None):
        self.events, self.token = app.EventBus(), app.CancellationToken()
        self.max_parallel, self.delay, self.fail_on, self.sample_rate = parallel, delay, fail_on, 24000
        self.last_note, self.log, self.calls = "", [], 0
        self.active, self.peak, self._lock = 0, 0, threading.Lock()                   # сколько кусков шло одновременно
        self.events.subscribe(lambda kind, data: self.log.append((kind, data)))
        self._np = np

    def synth(self, text):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            if self.fail_on and self.fail_on in text:
                raise RuntimeError("сбой на этом куске")
            return self._np.zeros(12000, dtype="float32")
        finally:
            with self._lock:
                self.active -= 1


def chunk_dir(app, tmp_name="parallel"):
    d = app.home / tmp_name
    d.mkdir(exist_ok=True)
    return d


def test_chunks_are_voiced_in_parallel_when_the_engine_allows_it(app):
    chunks = [f"кусок номер {i}" for i in range(1, 7)]
    d = chunk_dir(app)
    sequential, parallel = SlowEngine(app, 1), SlowEngine(app, 3)
    assert app.synthesize_chapter(sequential, d, chunks)
    for path in d.glob("*.wav"):
        path.unlink()
    assert app.synthesize_chapter(parallel, d, chunks)
    assert sorted(p.name for p in d.glob("*.wav")) == [f"{i:04d}.wav" for i in range(1, 7)]
    assert sequential.peak == 1 and parallel.peak > 1                       # куски шли одновременно (не по секундомеру)
    done = [data for kind, data in parallel.log if kind == "chunk_done"]
    assert sorted(item["index"] for item in done) == [1, 2, 3, 4, 5, 6]
    assert any("Итог по главе" in data.get("text", "") for kind, data in parallel.log)


def test_a_failed_chunk_stops_the_parallel_run_and_keeps_the_finished_ones(app):
    chunks = ["первый кусок", "второй кусок", "ПЛОХОЙ кусок", "четвёртый", "пятый", "шестой"]
    d = chunk_dir(app)
    engine = SlowEngine(app, 2, fail_on="ПЛОХОЙ")
    assert app.synthesize_chapter(engine, d, chunks) is False
    errors = [data["text"] for kind, data in engine.log if kind == "error"]
    assert len(errors) == 1 and "сбой на этом куске" in errors[0] and "3/6" in errors[0]
    assert engine.calls < 6                                                 # остальные куски не стали озвучивать
    assert (d / "0001.wav").exists()


def test_stop_during_a_parallel_run_raises_cancelled(app):
    d = chunk_dir(app)
    engine = SlowEngine(app, 3, delay=0.2)
    engine.events.subscribe(lambda kind, data: engine.token.cancel() if kind == "chunk_done" else None)
    with pytest.raises(app.Cancelled):
        app.synthesize_chapter(engine, d, [f"кусок {i}" for i in range(1, 10)])
    assert engine.calls < 9


def test_usage_log_does_not_lose_requests_recorded_from_many_threads(app):
    ctx = app.AppContext()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: ctx.usage.record("gemini-3.8-flash-tts", 10, 20, 1.0), range(80)))
    assert ctx.usage.summarize()["requests"] == 80


@needs_ffmpeg
def test_chapters_are_assembled_in_the_background_and_everything_is_finished_at_the_end(app):
    ctx = app.AppContext()
    bus, events = app.EventBus(), []
    bus.subscribe(lambda kind, data: events.append((kind, data)))
    book = make_book(app, ctx, chapters=3, paragraphs=12)
    assert app.AudiobookService(ctx, bus).run(book, "dummy", {1, 2, 3})
    kinds = [kind for kind, _ in events]
    assert kinds.count("chapter_done") == 3 and kinds[-1] == "run_finished"
    assert len(list((app.home / "story").glob("*.mp3"))) == 3 and not (app.home / "story" / "temp_chunks").exists()


@needs_ffmpeg
def test_a_failed_assembly_is_reported_and_the_run_does_not_pretend_to_succeed(app, monkeypatch):
    ctx = app.AppContext()
    bus, errors = app.EventBus(), []
    bus.subscribe(lambda kind, data: errors.append(data["text"]) if kind == "error" else None)
    book = make_book(app, ctx, chapters=2, paragraphs=12)

    def broken(*args, **kwargs):
        raise RuntimeError("ffmpeg сломался")
    monkeypatch.setattr(app, "assemble_chapter", broken)
    assert app.AudiobookService(ctx, bus).run(book, "dummy", {1, 2}) is False
    assert any("Склейка не удалась" in text and "ffmpeg сломался" in text for text in errors)
    assert list((app.home / "story").glob("*.mp3")) == [] and (app.home / "story" / "temp_chunks").exists()


def test_chunk_cache_returns_the_same_chunks_and_is_reused(app):
    text = body(30, 6)
    app._cached_chunks.cache_clear()
    assert app.chunk_list(text, 1200) == app.split_text_into_chunks(text, 1200)
    hits = app._cached_chunks.cache_info().hits
    app.chunk_list(text, 1200)
    assert app._cached_chunks.cache_info().hits == hits + 1
    assert app.chunk_list(text, 600) != app.chunk_list(text, 1200)


def test_qwen_chunk_size_is_a_setting_and_changes_the_chapter_fingerprint(app, settings_window):
    ctx, view, presenter, _ = settings_window
    assert app.engine_options(ctx.settings, "qwen")["max_chars"] == 1800
    chapter = {"text": body(10)}
    before = app.chapter_digest(ctx, chapter, "qwen")
    assert presenter.save(raw_values(view, qwen_chunk_chars="2400")) is True
    assert app.engine_options(ctx.settings, "qwen")["max_chars"] == 2400
    assert app.chapter_digest(ctx, chapter, "qwen") != before               # половина озвученной главы начнётся заново
    for bad in ("100", "9000", "много"):
        assert presenter.save(raw_values(view, qwen_chunk_chars=bad)) is False and "Размер куска Qwen" in view.errors[-1]


# ---------------------------------------------------------------- надёжность: замок папки книги и защита ключа

def test_book_lock_keeps_a_second_holder_out_until_it_is_released(app):
    path = app.home / "книга" / ".lock"
    first, second = app.BookLock(path), app.BookLock(path)
    assert first.acquire() is True and second.acquire() is False
    first.release()
    assert second.acquire() is True
    second.release()
    second.release()                                                       # повторное снятие безопасно


@needs_ffmpeg
def test_a_book_that_is_already_being_voiced_is_not_started_twice(app):
    ctx = app.AppContext()
    bus, errors = app.EventBus(), []
    bus.subscribe(lambda kind, data: errors.append(data["text"]) if kind == "error" else None)
    book = make_book(app, ctx, chapters=1, paragraphs=12)
    other = app.BookLock(book.dir / ".lock")
    assert other.acquire()
    service = app.AudiobookService(ctx, bus)
    assert service.run(book, "dummy", {1}) is False
    assert any("уже озвучивается" in text for text in errors) and list(book.dir.glob("*.mp3")) == []
    other.release()
    assert service.run(book, "dummy", {1}) is True and len(list(book.dir.glob("*.mp3"))) == 1


@needs_ffmpeg
def test_the_lock_is_released_after_a_run_and_after_an_engine_failure(app, monkeypatch):
    ctx = app.AppContext()
    service = app.AudiobookService(ctx)
    book = make_book(app, ctx, chapters=1, paragraphs=12)
    real = app.make_engine

    def broken(*args, **kwargs):
        raise RuntimeError("движок не запустился")
    monkeypatch.setattr(app, "make_engine", broken)
    assert service.run(book, "dummy", {1}) is False
    probe = app.BookLock(book.dir / ".lock")
    assert probe.acquire() is True                                         # после сбоя запуска замок свободен
    probe.release()
    monkeypatch.setattr(app, "make_engine", real)
    assert service.run(book, "dummy", {1}) is True
    probe = app.BookLock(book.dir / ".lock")
    assert probe.acquire() is True
    probe.release()


def test_secrets_are_stored_through_the_store_and_read_back(app):
    store = app.RecordStore(app.home / "s.json", app.Settings, only_changed=True, secrets=("gemini_api_key",))
    settings = store.load()
    settings.gemini_api_key = "AIzaSECRET0123456789"
    store.save(settings)
    assert store.load().gemini_api_key == "AIzaSECRET0123456789"
    assert app.protect_secret("") == "" and app.unprotect_secret("обычный текст") == "обычный текст"


@pytest.mark.skipif(os.name == "nt", reason="на Windows ключ шифруется")
def test_off_windows_the_key_stays_readable_and_foreign_encrypted_values_are_dropped(app):
    assert app.protect_secret("AIzaSECRET") == "AIzaSECRET"
    assert app.unprotect_secret("dpapi:AAAA") == ""                        # зашифрованное на Windows здесь не прочесть


@pytest.mark.skipif(os.name != "nt", reason="DPAPI есть только в Windows")
def test_on_windows_the_api_key_is_encrypted_in_the_settings_file(app):
    ctx = app.AppContext()
    ctx.update(gemini_api_key="AIzaSECRET0123456789")
    raw = app.PATHS.settings.read_text(encoding="utf-8")
    assert "AIzaSECRET0123456789" not in raw and app.SECRET_PREFIX in raw
    assert app.AppContext().settings.gemini_api_key == "AIzaSECRET0123456789"
    assert app.unprotect_secret(app.SECRET_PREFIX + "AAAA") == ""          # испорченное значение не роняет программу


# ---------------------------------------------------------------- английский интерфейс и переключатель языка

CYRILLIC = re.compile("[а-яА-ЯёЁ]")


def english_book(app, name="My Book.txt", chapters=3):
    paragraph = "This is an ordinary paragraph of text. " * 12
    text = "".join(f"Chapter {i}. Title {i}\n\n" + "\n\n".join([paragraph] * 4) + "\n\n" for i in range(1, chapters + 1))
    path = app.home / name
    path.write_text(text, encoding="utf-8")
    return path


def test_tr_translates_only_in_english_and_falls_back_to_the_original(app):
    assert app.tr("Стоп") == "Стоп"
    app.set_language("en")
    assert app.tr("Стоп") == "Stop" and app.tr("текст без перевода") == "текст без перевода"
    app.set_language("неведомый")
    assert app.tr("Стоп") == "Стоп"                                        # незнакомый язык: остаётся русский


def test_language_comes_from_the_environment_then_the_setting_then_the_system(app, monkeypatch):
    monkeypatch.setenv("AUDIOBOOK_LANG", "en")
    assert app.resolve_language("ru") == "en"
    monkeypatch.delenv("AUDIOBOOK_LANG")
    assert app.resolve_language("ru") == "ru" and app.resolve_language("en") == "en"
    monkeypatch.setattr(app, "system_language", lambda: "ru")
    assert app.resolve_language("auto") == "ru" and app.resolve_language("что-то") == "ru"
    monkeypatch.setattr(app, "system_language", lambda: "en")
    assert app.resolve_language("auto") == "en"


@pytest.mark.skipif(os.name == "nt", reason="на Windows язык берётся из системного API")
def test_system_language_follows_the_locale_off_windows(app, monkeypatch):
    import locale
    monkeypatch.setattr(locale, "getlocale", lambda *a: ("ru_RU", "UTF-8"))
    assert app.system_language() == "ru"
    monkeypatch.setattr(locale, "getlocale", lambda *a: (None, None))
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    assert app.system_language() == "en"
    monkeypatch.setenv("LANG", "ru_RU.UTF-8")
    assert app.system_language() == "ru"


def test_the_english_dictionary_is_complete_consistent_and_has_nothing_stale(app):
    import ast
    tree = ast.parse(Path(app.__file__).read_text(encoding="utf-8"))
    en_value = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                    and getattr(node.targets[0], "id", "") == "EN")
    skip = {id(node) for node in ast.walk(en_value)}
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in skip}
    calls = [call for call in ast.walk(tree) if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "tr"]
    wrapped = {call.args[0].value for call in calls if call.args and isinstance(call.args[0], ast.Constant)}
    assert not (wrapped - set(app.EN)), sorted(wrapped - set(app.EN))[:5]           # у каждого tr("...") есть перевод
    assert not (set(app.EN) - literals), sorted(set(app.EN) - literals)[:5]         # в словаре нет лишнего

    def holders(text):
        return sorted(re.findall(r"\{\d+[^}]*\}|%d|%s", text))

    for ru, en in app.EN.items():
        assert not CYRILLIC.search(en), (ru, en)                                   # перевод не содержит русского
        if "%d.%m.%Y" not in ru:
            assert holders(ru) == holders(en), (ru, en)                              # те же подстановки {0}, {1:.1f}, %d
    constants = [*app.PROFILE_TITLES.values(), *app.PROFILE_SHORT.values(),
                 *(note for _, note in app.GEMINI_MODELS), app.NEEDS_TITLE_HINT, app.TEST_PHRASE, "Начало", "Часть {0}"]
    assert [text for text in constants if text not in app.EN] == []                  # тексты констант переведены


@pytest.mark.parametrize("n, ru, en", [(1, "глава", "chapter"), (2, "главы", "chapters"), (5, "глав", "chapters"),
                                       (21, "глава", "chapters"), (0, "глав", "chapters")])
def test_words_with_a_number_follow_the_interface_language(app, n, ru, en):
    assert app.plural_word(n, "chapter") == ru
    app.set_language("en")
    assert app.plural_word(n, "chapter") == en


def test_numbers_dates_and_durations_follow_the_interface_language(app):
    assert app.fmt_int(8283) == "8 283" and app.fmt_duration(4800) == "1 ч 20 мин" and app.fmt_size(3_000_000_000) == "2.8 ГБ"
    app.set_language("en")
    assert app.fmt_int(8283) == "8,283" and app.fmt_duration(4800) == "1 h 20 min" and app.fmt_size(3_000_000_000) == "2.8 GB"
    assert app.eta_text(20) == "under a minute" and app.eta_text(240) == "≈ 4 min"


def test_generated_chapter_names_are_shown_translated_but_stay_the_same_inside(app):
    assert app.display_heading("Начало") == "Начало" and app.display_heading("Часть 3") == "Часть 3"
    app.set_language("en")
    assert app.display_heading("Начало") == "Beginning" and app.display_heading("Часть 3") == "Part 3"
    assert app.display_heading("Chapter 4") == "Chapter 4"
    chapters = app.split_book("Просто текст. " * 400, "size", {"size": 1000})[0]
    assert chapters[0]["heading"] == "Часть 1"                              # имена файлов не зависят от языка


def strings_in(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in list(value.keys()) + list(value.values()):
            yield from strings_in(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from strings_in(item)
    elif hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            yield from strings_in(getattr(value, name))


@needs_ffmpeg
def test_nothing_in_the_presenter_flow_stays_in_russian_when_the_language_is_english(app, window):
    ctx, view, presenter = window
    app.set_language("en")
    seen = []
    presenter.service.events.subscribe(lambda kind, data: seen.append(data.get("text", "")))
    presenter.open_file(english_book(app))
    presenter.set_engine("dummy")
    presenter.select([2])
    presenter.start()
    presenter.pump()
    presenter.open_chapter([1])
    presenter.redo([2])
    (app.home / "Book.pdf").write_bytes(b"%PDF")
    presenter.open_file(app.home / "Book.pdf")
    presenter.title_edited("")
    shown = [*seen, *view.log, *view.infos, *view.errors, view.summary, view.estimate, view.start_label,
             *(row.label for row in view.rows), *(app.status_chip(row) for row in view.rows), view.progress[1],
             *(question[1] for question in view.questions), *(question[0] for question in view.questions)]
    assert [text for text in shown if CYRILLIC.search(text)] == [], [t for t in shown if CYRILLIC.search(t)][:3]
    assert "Narrate selected" in " ".join(shown) or view.start_label.startswith("Narrate")


def test_changing_the_language_is_saved_rebuilds_the_window_and_is_restored_on_the_next_start(app, window, monkeypatch):
    ctx, view, presenter = window
    monkeypatch.delenv("AUDIOBOOK_LANG")
    presenter.change_language("en")
    assert ctx.settings.language == "en" and app._language == "en" and view.rebuilds == 1
    presenter.change_language("en")
    assert view.rebuilds == 1                                                 # тот же язык: ничего не делаем
    app.set_language("ru")
    app.AppContext()
    assert app._language == "en"                                              # при следующем запуске язык восстановлен
    presenter.change_language("ru")
    assert app.AppContext().settings.language == "ru" and app._language == "ru"


def test_restoring_the_view_after_a_language_change_brings_back_everything_shown(app, window):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt", chapters=3))
    presenter.select([2])
    presenter._progress(0.4, "Глава 1 из 3")
    app.set_language("en")
    view.hint = view.rows = view.summary = view.start_label = view.book_name = None
    presenter.restore_view()
    assert view.book_name == "Рассказ.txt" and len(view.rows) == 3 and view.selected == {2}
    assert view.start_label == "Narrate selected (1)" and view.summary.startswith("Chapters: 3")
    assert view.hint == "The title is taken from the file name. Is it right?" and view.progress == (0.4, "Глава 1 из 3")


def test_the_command_line_can_set_the_language(app, monkeypatch, capsys):
    path = english_book(app)
    monkeypatch.setattr(sys, "argv", ["audiobook_master.py", "--lang", "en", "--book", str(path)])
    monkeypatch.delenv("AUDIOBOOK_LANG")
    opened = []
    monkeypatch.setattr(app, "run_gui", lambda ctx, book_path=None: opened.append((app._language, book_path)) or True)
    assert app.main() == 0
    assert opened == [("en", path)] and app.AppContext().settings.language == "en"


def widget_texts(root):
    """Все надписи окна: подписи, кнопки, заголовки таблицы, пункты меню, значения списков, текст на холсте."""
    texts = []

    def walk(widget):
        kind = widget.winfo_class()
        try:
            value = widget.cget("text")
            if isinstance(value, str) and value:
                texts.append(value)
        except Exception:
            pass
        if kind == "Treeview":
            texts.extend(widget.heading(column, "text") for column in widget["columns"])
        elif kind == "Menu" and widget.index("end") is not None:
            texts.extend(widget.entrycget(i, "label") for i in range(widget.index("end") + 1)
                         if widget.type(i) == "command")
        elif kind == "TCombobox":
            texts.extend(widget.tk.splitlist(widget.cget("values")))
        elif kind == "Canvas":
            texts.extend(widget.itemcget(item, "text") for item in widget.find_all() if widget.type(item) == "text")
        for child in widget.winfo_children():
            walk(child)
    walk(root)
    return texts


def test_the_real_window_and_settings_have_no_russian_in_english_and_switch_without_losing_the_book(app, monkeypatch):
    pytest.importorskip("tkinterdnd2")
    pytest.importorskip("sv_ttk")
    tk = pytest.importorskip("tkinter")
    ctx = app.AppContext()
    app.set_language("en")
    try:
        window = app.MainWindow(lambda view: app.MainPresenter(app.AudiobookService(ctx, app.EventBus()), view))
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")
    try:
        window.presenter.open_file(english_book(app))
        window.root.update()
        items = window.table.get_children()
        window.table.selection_set(items[1])
        window.root.update()
        dialog = window.show_settings(lambda view: app.SettingsPresenter(ctx, view, runner=lambda work: work()))
        window.root.update()
        english = widget_texts(window.root)
        assert [text for text in english if CYRILLIC.search(text)] == []
        assert {"Narrate selected (1)", "Settings", "Open folder", "Book title:", "Status"} <= set(english)
        assert {"API key", "Key info", "Diagnostics…", "▶ Listen"} <= set(english)
        dialog.close()
        monkeypatch.delenv("AUDIOBOOK_LANG")
        window.presenter.change_language("ru")
        window.root.update()
        russian = widget_texts(window.root)
        assert {"Озвучить выбранные (1)", "Настройки", "Открыть папку", "Название книги:", "Статус"} <= set(russian)
        assert len(window.table.get_children()) == 3 and window.selected_ids() == [2]
        assert window.title_var.get() == "My Book" and window.book_file == "My Book.txt"
        assert ctx.settings.language == "ru" and str(window.language_labels["ru"].cget("foreground")) == window.colors["accent"]
        window.presenter.change_language("en")
        window.root.update()
        assert "Narrate selected (1)" in widget_texts(window.root) and window.selected_ids() == [2]
    finally:
        try:
            window.root.destroy()
        except tk.TclError:
            pass


def test_the_qwen_book_language_follows_the_interface_language_until_it_is_chosen(app, monkeypatch, settings_window):
    assert app.effective_qwen_language(app.AppContext().settings) == "Russian"        # русский интерфейс: русская книга
    monkeypatch.setenv("AUDIOBOOK_LANG", "en")
    ctx = app.AppContext()
    assert ctx.settings.qwen_language == "" and app.effective_qwen_language(ctx.settings) == "English"
    ctx.update(qwen_language="Russian")                                             # явный выбор главнее умолчания
    assert app.effective_qwen_language(app.AppContext().settings) == "Russian"
    ctx.update(qwen_language="German")
    assert app.effective_qwen_language(app.AppContext().settings) == "German"


def test_without_tk_the_console_explains_how_to_get_the_window(app, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "tkinter", None)
    assert app.ensure_gui_packages() is False
    out = capsys.readouterr().out
    assert "tcl/tk and IDLE" in out and "pip" in out and sys.base_prefix in out


def test_test_phrase_plays_in_place_or_opens_with_the_default_player(app, monkeypatch):
    opened = []
    monkeypatch.setattr(app, "open_in_system", opened.append)
    app.play_sound(app.home / "voice_test.wav")
    if os.name != "nt":
        assert opened == [app.home / "voice_test.wav"]


@needs_ffmpeg
def test_chapter_rows_update_live_while_chunks_are_voiced(app, window):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt", chapters=2, paragraphs=12))
    presenter.set_engine("dummy")
    presenter.select([1])
    presenter.start()
    presenter.pump()
    done = [update[1] for update in view.row_updates]
    assert view.row_updates and all(update[0] == 1 and update[3] == "dummy" for update in view.row_updates)
    assert done == sorted(done) and done[-1] == view.row_updates[-1][2]


@needs_ffmpeg
def test_mp3_files_carry_book_and_chapter_tags(app):
    ctx = app.AppContext()
    book = make_book(app, ctx, chapters=2, paragraphs=12)
    assert app.AudiobookService(ctx).run(book, "dummy", {1, 2})
    data = sorted(book.dir.glob("*.mp3"))[1].read_bytes()[:4096]
    assert data[:3] == b"ID3"
    for text in (book.title, "2/2", "Audiobook"):
        assert text.encode("latin-1", "ignore") in data or text.encode("utf-16-le") in data


def test_the_real_window_shows_live_chapter_progress(app):
    pytest.importorskip("tkinterdnd2")
    pytest.importorskip("sv_ttk")
    tk = pytest.importorskip("tkinter")
    ctx = app.AppContext()
    try:
        window = app.MainWindow(lambda view: app.MainPresenter(app.AudiobookService(ctx, app.EventBus()), view))
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")
    try:
        window.presenter.open_file(write_book(app, "Рассказ.txt", chapters=2))
        window.update_row(2, 3, 8, "qwen")
        item = window.row_items[2]
        assert window.table.set(item, "status") == "◐ 3/8 · qwen" and window.table.item(item, "tags") == ("part",)
        window.update_row(99, 1, 2, "qwen")                                      # неизвестная глава: просто ничего
    finally:
        window.root.destroy()


# ---------------------------------------------------------------- обновление программы

DL = "https://example.org/dl/"


def sums_for(**files):
    return "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in files.items()).encode()


def update_network(script, sums=None, version="9.9.9", with_script=True, with_sums=True, body="Новая функция\n\nподробности",
                   fail=False):
    """Подделка GitHub: последний выпуск, его файлы и SHA256SUMS."""
    assets = ([{"name": "audiobook_master.py", "browser_download_url": DL + "audiobook_master.py"}] if with_script else []) + \
             ([{"name": "SHA256SUMS", "browser_download_url": DL + "SHA256SUMS"}] if with_sums else [])
    files = {"https://api.github.com/repos/ed9ward/Foxy_AudioBook_Maker/releases/latest":
             json.dumps({"tag_name": f"v{version}", "body": body, "assets": assets}).encode(),
             DL + "audiobook_master.py": script,
             DL + "SHA256SUMS": sums if sums is not None else sums_for(**{"audiobook_master.py": script})}

    def fetch(url, **k):
        if fail or url not in files:
            raise OSError("нет сети")
        return files[url]
    return fetch


def test_update_is_offered_only_for_a_newer_verified_release_of_this_program(app, monkeypatch, tmp_path):
    script = tmp_path / "audiobook_master.py"
    current = Path(app.__file__).read_bytes()
    script.write_bytes(current)
    monkeypatch.setattr(app, "SCRIPT_PATH", script)
    version_line = f'APP_VERSION = "{app.APP_VERSION}"'.encode()
    assert current.count(version_line) == 1
    newer = current.replace(version_line, b'APP_VERSION = "9.9.9"')
    text, info, error = app.check_update(update_network(newer))
    assert text == newer.decode() and info == "9.9.9: Новая функция" and error == ""
    assert app.check_update(update_network(current, version=app.APP_VERSION)) == (None, "", "")          # уже последняя
    assert app.check_update(update_network(current, version="0.0.1")) == (None, "", "")                  # старую не ставим
    assert app.check_update(update_network(newer + b"# tampered\n", sums=sums_for(**{"audiobook_master.py": newer})))[2] \
        .startswith("скачанный файл не прошёл проверку")                                                  # хеш не совпал
    assert app.check_update(update_network(newer, sums=sums_for(**{"other.py": newer})))[0] is None        # нет записи о файле
    assert "нет скрипта или контрольных сумм" in app.check_update(update_network(newer, with_sums=False))[2]
    assert "нет скрипта или контрольных сумм" in app.check_update(update_network(newer, with_script=False))[2]
    broken = b"def broken(:\n"
    assert app.check_update(update_network(broken))[2].startswith("скачанный файл повреждён")             # хеш верный, но не код
    assert app.check_update(update_network("print('чужая программа')\n".encode()))[2].startswith("скачанный файл повреждён")
    assert app.check_update(update_network(current.replace(version_line, b'APP_VERSION = "9.9.8"'), version="9.9.9"))[2] \
        .startswith("скачанный файл повреждён")
    assert app.check_update(update_network(newer, fail=True))[2].startswith("нет связи с GitHub")
    app.apply_update(text)
    assert script.read_bytes() == newer and not (tmp_path / "audiobook_master.py.new").exists()
    assert (tmp_path / "audiobook_master.py.prev").read_bytes() == current                                # откат возможен


def test_without_a_console_fatal_errors_are_shown_in_a_window(app, monkeypatch):
    shown = []
    monkeypatch.setattr(app, "show_fatal", shown.append)
    monkeypatch.setattr(sys, "stdout", None)
    with pytest.raises(SystemExit):
        app.fail("❌ Что-то сломалось")
    assert shown == ["❌ Что-то сломалось"]


def release(version="1.0.0", asset=True):
    assets = [{"name": "FoxyAudioBookMaker.exe", "browser_download_url": "https://example.org/FoxyAudioBookMaker.exe"}] if asset else []
    return json.dumps({"tag_name": f"v{version}", "body": "• Новая кнопка\n• Исправлен баг", "assets": assets}).encode()


def test_exe_updates_are_offered_only_for_newer_releases_with_the_exe_attached(app):
    assert app.version_tuple("v1.2") == (1, 2, 0) and app.version_tuple("0.9.0") < app.version_tuple("1.0.0")
    url, version, notes, error = app.check_exe_update(lambda url, **k: release("9.0.0"))
    assert url.endswith("FoxyAudioBookMaker.exe") and version == "9.0.0" and "Новая кнопка" in notes and error == ""
    assert app.check_exe_update(lambda url, **k: release(app.APP_VERSION))[0] is None              # уже последняя
    assert "FoxyAudioBookMaker.exe" in app.check_exe_update(lambda url, **k: release("2.0.0", asset=False))[3]

    def offline(url, **k):
        raise OSError("нет сети")
    assert app.check_exe_update(offline)[3].startswith("нет связи с GitHub")


def test_a_downloaded_exe_is_verified_and_replaces_the_old_one(app, monkeypatch):
    exe = app.home / "FoxyAudioBookMaker.exe"
    exe.write_bytes(b"MZ old")
    monkeypatch.setenv("AUDIOBOOK_EXE", str(exe))
    url = "https://example.org/dl/FoxyAudioBookMaker.exe"
    new = b"MZ" + b"\0" * 60_000

    def network(data, sums):
        return lambda u, **k: sums if u.endswith("SHA256SUMS") else data
    app.apply_exe_update(url, network(new, sums_for(**{"FoxyAudioBookMaker.exe": new})))
    assert exe.read_bytes() == new and not (app.home / "FoxyAudioBookMaker.exe.new").exists()
    assert (app.home / "FoxyAudioBookMaker.exe.prev").read_bytes() == b"MZ old"                      # откат возможен
    exe.write_bytes(b"MZ old")
    tampered = b"MZ" + b"\1" * 60_000
    for fetch in (network(tampered, sums_for(**{"FoxyAudioBookMaker.exe": new})),                     # хеш не совпал
                  network(new, sums_for(**{"other.exe": new})),                                  # нет записи об exe
                  network(b"<html>not found</html>", sums_for(**{"FoxyAudioBookMaker.exe": new}))):   # не программа
        with pytest.raises(ValueError):
            app.apply_exe_update(url, fetch)
        assert exe.read_bytes() == b"MZ old"                                                     # старый exe не тронут


def test_settings_show_whats_new_then_update_the_exe_and_restart(app, settings_window, monkeypatch):
    ctx, view, presenter, _ = settings_window
    monkeypatch.setenv("AUDIOBOOK_EXE", str(app.home / "FoxyAudioBookMaker.exe"))
    applied = []
    monkeypatch.setattr(app, "check_exe_update", lambda: ("https://example.org/FoxyAudioBookMaker.exe", "1.0.0", "• Новая кнопка", ""))
    monkeypatch.setattr(app, "apply_exe_update", applied.append)
    view.yes = False
    presenter.check_update()
    presenter.pump()
    assert applied == [] and "• Новая кнопка" in view.questions[-1][1] and "перезапустится" in view.questions[-1][1]
    view.yes = True
    presenter.check_update()
    presenter.pump()
    presenter.pump()
    assert applied == ["https://example.org/FoxyAudioBookMaker.exe"] and getattr(view, "restarted", False)

    def broken(url):
        raise OSError("диск занят")
    monkeypatch.setattr(app, "apply_exe_update", broken)
    view.restarted = False
    presenter.check_update()
    presenter.pump()
    presenter.pump()
    assert not view.restarted and "диск занят" in view.status[0] and view.status[1] is True


def test_diagnostics_say_how_the_program_was_started(app, monkeypatch):
    ctx = app.AppContext()
    assert "запуск скриптом" in app.diagnostics_lines(ctx)[0]
    monkeypatch.setenv("AUDIOBOOK_EXE", str(app.home / "FoxyAudioBookMaker.exe"))
    assert "запуск из FoxyAudioBookMaker.exe" in app.diagnostics_lines(ctx)[0]


def test_replacing_the_voice_recording_with_another_file_of_the_same_size_restarts_a_half_done_chapter(app):
    ctx = app.AppContext()
    ref = app.PATHS.base / "ref.wav"
    sf.write(str(ref), np.full(24000, 0.1, dtype="float32"), 24000)
    ctx.update(qwen_voices=[{"name": "Мой голос", "path": "ref.wav", "text": "Текст образца."}])
    book = make_book(app, ctx, chapters=1, paragraphs=12)
    chapter = book.chapters[0]
    folder = app.work_dir(book.dir, "qwen", chapter)
    app.prepare_chunks(ctx, chapter, folder, "qwen")
    sf.write(str(app.chunk_path(folder, 1)), np.zeros(2400, dtype="float32"), 24000)
    assert app.chapter_status(ctx, book, chapter)[0] == "part"
    size = ref.stat().st_size
    ref.unlink()
    sf.write(str(ref), np.full(24000, -0.1, dtype="float32"), 24000)                  # другая запись того же размера
    assert ref.stat().st_size == size
    assert app.chapter_status(ctx, book, chapter)[0] == "new"                         # старые куски больше не засчитываются
    log = []
    bus = app.EventBus()
    bus.subscribe(lambda kind, data: log.append(data.get("text", "")))
    app.prepare_chunks(ctx, chapter, folder, "qwen", bus)
    assert not app.chunk_done(app.chunk_path(folder, 1)) and any("сброшены" in line for line in log)


@needs_ffmpeg
def test_finished_mp3_goes_stale_when_text_or_voice_changes(app):
    ctx = app.AppContext()
    service = app.AudiobookService(ctx)
    book = make_book(app, ctx)
    assert service.run(book, "dummy", {1, 2})
    assert (book.dir / "book_manifest.json").exists()
    assert [r.kind for r in service.chapter_rows(book, "dummy")] == ["done", "done"]
    book.chapters[0]["text"] += " Новая фраза."                      # текст главы изменился, имя файла прежнее
    assert [r.kind for r in service.chapter_rows(book, "dummy")] == ["stale", "done"]
    assert service.pending(book, {1, 2}) == []                       # устаревшую главу молча не перезаписываем
    service.reset_chapters(book, {1})                                # «Озвучить заново»
    assert "01_Название_1.mp3" not in app.load_results(book.dir)
    assert [r.kind for r in service.chapter_rows(book, "dummy")] == ["new", "done"]


def test_mp3_without_manifest_is_trusted(app):
    ctx = app.AppContext()
    service = app.AudiobookService(ctx)
    book = make_book(app, ctx)
    (book.dir / f"{book.chapters[0]['name']}.mp3").write_bytes(b"x")
    assert service.chapter_rows(book, "dummy")[0].kind == "done"     # книга озвучена до появления манифеста


@pytest.mark.parametrize("text", [
    "а" * 5000,                                                         # длинная строка без пробелов
    "Смотри тут: https://example.com/" + "a1b2/" * 600 + " и дальше обычный текст.",
    "word " * 50 + "x" * 3000 + " " + "word " * 50,
    "Привет, мир! " * 200 + "ы" * 2500,                                 # кириллица
    "你好，世界" * 1000,                                                  # китайский без пробелов и без ./!/?
    "这是第一句。这是第二句！" * 300,
    "😀" * 4000 + "👨‍👩‍👧" * 500,                                          # эмодзи, в том числе составные
    "a" * 1200,                                                         # ровно лимит
], ids=["no-spaces", "url", "long-token", "cyrillic", "chinese-no-punct", "chinese-sentences", "emoji", "exact-limit"])
@pytest.mark.parametrize("limit", [50, 300, 1200])
def test_chunks_never_exceed_limit_and_lose_nothing(app, text, limit):
    chunks = app.split_text_into_chunks(text, limit)
    assert all(0 < len(chunk) <= limit for chunk in chunks)

    def strip(value):
        return re.sub(r"\s+", "", value)

    assert strip("".join(chunks)) == strip(text)                       # ничего не потеряно и не добавлено


def test_chinese_text_is_split_at_sentence_ends(app):
    text = "这是第一句。" * 40
    chunks = app.split_text_into_chunks(text, 60)
    assert len(chunks) > 1 and all(chunk.rstrip().endswith("。") for chunk in chunks)


def test_failed_dpapi_never_writes_the_key_in_plain_text(app, monkeypatch, capsys):
    def broken(data, protect):
        raise OSError("DPAPI")
    monkeypatch.setattr(app, "DPAPI_AVAILABLE", True)                    # имитируем Windows, где шифрование сломалось
    monkeypatch.setattr(app, "_dpapi", broken)
    with pytest.raises(app.SecretProtectionError):
        app.protect_secret("AIzaSECRET0123456789")
    ctx = app.AppContext()
    ctx.update(gemini_api_key="AIzaSECRET0123456789", gemini_voice="Puck")
    raw = app.PATHS.settings.read_text(encoding="utf-8")
    assert "AIzaSECRET0123456789" not in raw and "gemini_api_key" not in raw
    assert "Puck" in raw                                                 # остальные настройки сохранены
    assert ctx.settings.gemini_api_key == "AIzaSECRET0123456789"         # до закрытия программы ключ работает
    assert ctx.store.secret_errors == ["gemini_api_key"]
    assert "GEMINI_API_KEY" in capsys.readouterr().out                   # пользователь видит, что ключ не сохранён


@pytest.mark.skipif(os.name == "nt", reason="права доступа POSIX")
def test_settings_file_with_the_key_is_owner_only(app):
    app.AppContext().update(gemini_api_key="AIzaSECRET0123456789")
    assert app.PATHS.settings.stat().st_mode & 0o077 == 0


# ---------------------------------------------------------------- лимиты EPUB / FB2 и проверка WAV

def make_epub_file(path, files=None, chapter="<html><body><h1>Глава 1</h1><p>Текст главы.</p></body></html>", extra=0):
    import zipfile
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("META-INF/container.xml", '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="content.opf"/></rootfiles></container>')
        z.writestr("content.opf", '<package xmlns="http://www.idpf.org/2007/opf">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Тест</dc:title></metadata>'
                   '<manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
                   '</manifest><spine><itemref idref="c1"/></spine></package>')
        z.writestr("c1.xhtml", chapter)
        for i in range(extra):
            z.writestr(f"img/{i}.bin", b"")


def test_a_normal_epub_opens_and_oversized_ones_are_refused(app, monkeypatch, tmp_path):
    good = tmp_path / "good.epub"
    make_epub_file(good)
    assert "Текст главы" in app.read_book(str(good))[0]
    many = tmp_path / "many.epub"
    make_epub_file(many, extra=20)
    monkeypatch.setattr(app, "MAX_ARCHIVE_FILES", 10)                                  # слишком много файлов
    with pytest.raises(app.BookFormatError, match="слишком велик"):
        app._read_epub(str(many))
    monkeypatch.setattr(app, "MAX_ARCHIVE_FILES", 20_000)
    big = tmp_path / "big.epub"
    make_epub_file(big, chapter="<html><body><p>" + "а" * 5000 + "</p></body></html>")
    monkeypatch.setattr(app, "MAX_XML_BYTES", 1000)                                    # один файл слишком большой
    with pytest.raises(app.BookFormatError, match="слишком велик"):
        app._read_epub(str(big))
    monkeypatch.setattr(app, "MAX_XML_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(app, "MAX_ARCHIVE_READ", 200)                                  # слишком много суммарно
    with pytest.raises(app.BookFormatError, match="слишком велик"):
        app._read_epub(str(good))


def test_oversized_fb2_and_too_long_text_are_refused(app, monkeypatch, tmp_path):
    fb2 = tmp_path / "b.fb2"
    fb2.write_text('<?xml version="1.0" encoding="utf-8"?><FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
                   '<body><section><title><p>Глава 1</p></title><p>' + "а" * 2000 + '</p></section></body></FictionBook>',
                   encoding="utf-8")
    assert "ааа" in app.read_book(str(fb2))[0]
    monkeypatch.setattr(app, "MAX_XML_BYTES", 500)
    with pytest.raises(app.BookFormatError, match="слишком велик"):
        app._read_fb2(str(fb2))
    monkeypatch.setattr(app, "MAX_XML_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(app, "MAX_BOOK_CHARS", 1000)                                   # текст в памяти слишком длинный
    with pytest.raises(app.BookFormatError, match="слишком велик"):
        app._read_book_cached(str(fb2), 1, 1)


def test_chunk_valid_rejects_damaged_wav_files(app, tmp_path):
    good = tmp_path / "0001.wav"
    app.write_chunk_wav(good, np.sin(np.linspace(0, 400, 24_000)).astype("float32") * 0.3, 24_000)
    assert app.chunk_valid(good, 24_000) and app.chunk_done(good)
    assert not app.chunk_valid(good, 16_000)                                           # другая частота
    raw = good.read_bytes()
    cut = tmp_path / "cut.wav"
    cut.write_bytes(raw[:len(raw) // 2])                                               # обрезан
    assert app.chunk_done(cut) and not app.chunk_valid(cut, 24_000)
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"\x01" * 5000)                                                   # не WAV
    assert app.chunk_done(junk) and not app.chunk_valid(junk)
    stereo = tmp_path / "stereo.wav"
    sf.write(str(stereo), np.zeros((2000, 2), dtype="float32"), 24_000, subtype="PCM_16", format="WAV")
    assert not app.chunk_valid(stereo, 24_000)                                         # два канала
    empty = tmp_path / "empty.wav"
    sf.write(str(empty), np.zeros(0, dtype="float32"), 24_000, subtype="PCM_16", format="WAV")
    assert not app.chunk_valid(empty, 24_000) and not app.chunk_valid(tmp_path / "missing.wav")


@needs_ffmpeg
def test_a_damaged_chunk_is_regenerated_on_resume_instead_of_blocking_the_chapter(app):
    ctx = app.AppContext()
    book = make_book(app, ctx, chapters=1, paragraphs=6)
    engine = app.DummyEngine(ctx)
    d = app.work_dir(book.dir, "dummy", book.chapters[0])
    chunks = app.prepare_chunks(ctx, book.chapters[0], d, "dummy")
    assert app.synthesize_chapter(engine, d, chunks)
    victim = app.chunk_path(d, 1)
    victim.write_bytes(victim.read_bytes()[:3000])                                     # обрезан, но «больше 1 КБ»
    assert app.chunk_done(victim) and not app.chunk_valid(victim, engine.sample_rate)
    assert app.synthesize_chapter(engine, d, chunks)                                   # возобновление чинит кусок
    assert app.chunk_valid(victim, engine.sample_rate)


# ---------------------------------------------------------------- возвращённые тесты (были удалены по ошибке)

def fake_fetch(main=None, info=None, fail=False):
    def fetch(url):
        if fail:
            raise OSError("нет сети")
        if url == UPDATE_INFO(url):
            return json.dumps(info or [{"commit": {"committer": {"date": "2026-11-12T10:00:00Z"},
                                                   "message": "Новая функция\n\nподробности"}}]).encode()
        return main
    return fetch


def UPDATE_INFO(url):
    return url if "api.github.com" in url else None


def test_settings_update_button_asks_before_replacing_the_program(app, settings_window, monkeypatch):
    ctx, view, presenter, _ = settings_window
    applied = []
    monkeypatch.setattr(app, "apply_update", applied.append)
    monkeypatch.setattr(app, "check_update", lambda: ("новый текст", "2026-11-12: Новая функция", ""))
    view.yes = False
    presenter.check_update()
    presenter.pump()
    assert applied == [] and "2026-11-12: Новая функция" in view.questions[-1][1]
    view.yes = True
    presenter.check_update()
    presenter.pump()
    assert applied == ["новый текст"] and "обновлена" in view.status[0] and view.status[1] is False
    monkeypatch.setattr(app, "check_update", lambda: (None, "", ""))
    presenter.check_update()
    presenter.pump()
    assert "последняя версия" in view.status[0]
    monkeypatch.setattr(app, "check_update", lambda: (None, "", "нет связи с GitHub: таймаут"))
    presenter.check_update()
    presenter.pump()
    assert view.status == ("❌ нет связи с GitHub: таймаут", True)


def test_notices_do_not_block_and_fade_back_to_the_log(app):
    pytest.importorskip("tkinterdnd2")
    pytest.importorskip("sv_ttk")
    tk = pytest.importorskip("tkinter")
    ctx = app.AppContext()
    try:
        window = app.MainWindow(lambda view: app.MainPresenter(app.AudiobookService(ctx, app.EventBus()), view), theme="dark")
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")
    try:
        window.append_log("строка журнала")
        window.presenter.open_folder()                              # папки ещё нет: уведомление, а не окно
        window.presenter.open_file(write_book(app, "Рассказ.txt"))
        window.presenter.open_chapter([1])
        assert window.status_line.cget("text").startswith("ℹ Эта глава ещё не озвучена")
        assert str(window.status_line.cget("foreground")) == window.colors["accent"]
        window.append_log("новая строка")                            # уведомление не перебивается журналом
        assert window.status_line.cget("text").startswith("ℹ")
        window._end_notice()
        assert window.status_line.cget("text") == "новая строка"
        app.set_title_bar_theme(window.root, True)                    # вне Windows ничего не делает и не падает
    finally:
        window.root.destroy()


needs_num2words = pytest.mark.skipif(importlib.util.find_spec("num2words") is None, reason="нет num2words")


@needs_num2words
@pytest.mark.parametrize("text, spoken", [
    ("В 1984 году он уехал, т. е. навсегда.", "В тысяча девятьсот восемьдесят четвёртом году он уехал, то есть навсегда."),
    ("в 1990-х годах, в XIX веке и в XX в.", "в тысяча девятьсот девяностых годах, в девятнадцатом веке и в двадцатом веке."),
    ("Он пришёл 2-м, к 5-му числу.", "Он пришёл вторым, к пятому числу."),
    ("Цена 1 500 руб., скидка 15%, рост 3,5%.",
     "Цена одна тысяча пятьсот рублей, скидка пятнадцать процентов, рост три целых пять десятых процента."),
    ("Всего 21 тыс. и 2 млн, 1 коп.", "Всего двадцать одна тысяча и два миллиона, одна копейка."),
    ("С 1941 г. до 1945 г. шла война.",
     "С тысяча девятьсот сорок первого года до тысяча девятьсот сорок пятого года шла война."),
    ("Дом № 7, г. Москва, и т. д.", "Дом номер семь, город Москва, и так далее."),
    ("Встреча в 10:30.", "Встреча в десять тридцать."),
    ("Версия 1.2.3", "Версия один точка два точка три"),
    ("In 1984 he left, i.e. for good.", "In nineteen eighty-four he left, that is for good."),
    ("It costs $5 or $1.50, about 15% more.", "It costs five dollars or one dollar fifty, about fifteen percent more."),
    ("He came 2nd. Mr. Smith met Dr. Who at 10:30, etc.",
     "He came second. Mister Smith met Doctor Who at ten thirty, et cetera."),
])
def test_numbers_and_abbreviations_are_read_as_words(app, text, spoken):
    assert app.normalize_for_speech(text) == spoken


def test_own_replacements_match_whole_words_and_work_without_the_builtin_rules(app):
    rules = "ИИ = искусственный интеллект\n# комментарий\n\nNASA = наса"
    spoken = app.normalize_for_speech("ИИ и NASA, ИИшка, 5 руб.", rules, builtin=False)
    assert spoken == "искусственный интеллект и наса, ИИшка, 5 руб."
    assert app.parse_replacements(rules) == [("ИИ", "искусственный интеллект"), ("NASA", "наса")]
    assert app.bad_replacement_lines("ок = да\nбез знака равно\n = пусто\n# можно") == [2, 3]


def test_without_num2words_numbers_stay_digits_but_abbreviations_are_still_read(app, monkeypatch):
    monkeypatch.setattr(app, "has_num2words", lambda: False)
    assert app.normalize_for_speech("В 1984 году, т. е. давно.") == "В 1984 году, то есть давно."


@needs_num2words
def test_chunks_use_the_spoken_text_and_rules_take_part_in_the_chapter_fingerprint(app):
    ctx = app.AppContext()
    chapter = {"id": 1, "text": "В 1984 году. " + body(5)}
    spoken = app.speech_text(ctx, chapter["text"])
    assert spoken.startswith("В тысяча девятьсот восемьдесят четвёртом году.")
    folder = app.home / "parts"
    assert app.prepare_chunks(ctx, chapter, folder, "gemini")[0].startswith("В тысяча девятьсот")
    first = app.chapter_digest(ctx, chapter, "gemini")
    ctx.update(replacements="году = годе")
    assert app.chapter_digest(ctx, chapter, "gemini") != first
    ctx.update(normalize_text=False, replacements="")
    assert app.speech_text(ctx, chapter["text"]) == chapter["text"] and app.speech_signature(ctx.settings) == ""


def test_settings_save_the_reading_rules_and_reject_malformed_lines(app, settings_window):
    ctx, view, presenter, _ = settings_window
    assert view.values["normalize_text"] is True and view.values["replacements"] == ""
    assert presenter.save(raw_values(view, normalize_text=False, replacements="ИИ = искусственный интеллект\n\n")) is True
    assert ctx.settings.normalize_text is False and ctx.settings.replacements == "ИИ = искусственный интеллект"
    assert presenter.save(raw_values(view, replacements="просто слово")) is False
    assert "строка 1" in view.errors[-1]


PROSE = "Это обычный абзац текста книги, достаточно длинный, чтобы походить на прозу. " * 6


def make_epub(path, version=3):
    import zipfile as zf
    chapters = [("ch1.xhtml", "Глава первая. Начало пути"), ("part2.xhtml", "Часть вторая"),
                ("ch2.xhtml", "Глава вторая. Буря")]
    with zf.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:'
                   'container" version="1.0"><rootfiles><rootfile full-path="OEBPS/content.opf" '
                   'media-type="application/oebps-package+xml"/></rootfiles></container>')
        items = "".join(f'<item id="i{n}" href="{href}" media-type="application/xhtml+xml"/>'
                        for n, (href, _) in enumerate(chapters))
        items += '<item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>'
        if version == 3:
            items += '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
        else:
            items += '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        spine = '<itemref idref="cover" linear="no"/>' + "".join(f'<itemref idref="i{n}"/>' for n in range(len(chapters)))
        z.writestr("OEBPS/content.opf",
                   f'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="{version}.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Морская книга</dc:title></metadata>'
                   f'<manifest>{items}</manifest><spine>{spine}</spine></package>')
        z.writestr("OEBPS/cover.xhtml", "<html><body><p>ОБЛОЖКА</p></body></html>")
        for href, title in chapters:
            heading = f"<h1>{title}</h1>" if version == 3 else f'<p class="title">{title}</p>'
            body = "" if href.startswith("part") else (f"<p>{PROSE}<sup><a epub:type=\"noteref\" href=\"#n1\">1</a></sup></p>"
                                                       f"<p>Второй абзац.&nbsp;{PROSE}</p>")
            if version == 2:
                heading = ""                                  # в EPUB 2 названия берутся из оглавления toc.ncx
            z.writestr(f"OEBPS/{href}", f'<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml" '
                       f'xmlns:epub="http://www.idpf.org/2007/ops"><head><title>x</title><style>p{{}}</style></head>'
                       f"<body>{heading}{body}</body></html>")
        if version == 3:
            links = "".join(f'<li><a href="{href}">{title}</a></li>' for href, title in chapters)
            z.writestr("OEBPS/nav.xhtml",
                       f'<html xmlns="http://www.w3.org/1999/xhtml"><body><nav><ol>{links}</ol></nav></body></html>')
        else:
            points = "".join(f'<navPoint id="p{n}"><navLabel><text>{title}</text></navLabel><content src="{href}"/></navPoint>'
                             for n, (href, title) in enumerate(chapters))
            z.writestr("OEBPS/toc.ncx", f'<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>{points}</navMap></ncx>')
    return path


def make_fb2(path, zipped=False):
    sections = "".join(
        f"<section><title><p>Часть {part}</p></title>"
        + "".join(f"<section><title><p>Глава {part}.{n}</p></title><epigraph><p>Эпиграф</p></epigraph>"
                  f"<p>{PROSE}<a l:href=\"#n1\" type=\"note\">[1]</a></p><poem><title><p>Песня</p></title>"
                  f"<stanza><v>Строка стиха</v></stanza></poem><p>{PROSE}</p></section>" for n in (1, 2))
        + "</section>" for part in (1, 2))
    xml = (f'<?xml version="1.0" encoding="windows-1251"?><FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0" '
           f'xmlns:l="http://www.w3.org/1999/xlink"><description><title-info><book-title>Северная повесть</book-title>'
           f'</title-info></description><body><title><p>Северная повесть</p></title>{sections}</body>'
           f'<body name="notes"><section id="n1"><p>Текст сноски</p></section></body></FictionBook>')
    data = xml.encode("cp1251")
    if zipped:
        import zipfile as zf
        with zf.ZipFile(path, "w") as z:
            z.writestr("book.fb2", data)
    else:
        path.write_bytes(data)
    return path


@pytest.mark.parametrize("version", [2, 3])
def test_epub_books_open_with_real_chapters_title_and_without_service_pages(app, version):
    path = make_epub(app.home / f"sea{version}.epub", version)
    text, title = app.read_book(path)
    assert title == "Морская книга"
    assert "ОБЛОЖКА" not in text and "\n1\n" not in text and "&nbsp" not in text            # обложка и сноски не читаются
    ctx = app.AppContext()
    book = app.AudiobookService(ctx).open_book(path, "Морская книга")
    assert book.profile_used == "markdown" and len(book.chapters) == 2
    assert app.display_heading(book.chapters[1]["heading"]) == "Часть вторая. Глава вторая. Буря"   # титул части в главе
    assert book.chapters[0]["text"].startswith("# Глава первая. Начало пути\n\nЭто обычный абзац")
    assert app.speech_text(ctx, book.chapters[0]["text"]).startswith("Глава первая. Начало пути\n\nЭто обычный абзац")


@pytest.mark.parametrize("zipped", [False, True])
def test_fb2_books_open_with_sections_as_chapters(app, zipped):
    path = make_fb2(app.home / ("north.fb2.zip" if zipped else "north.fb2"), zipped)
    text, title = app.read_book(path)
    assert title == "Северная повесть" and "Текст сноски" not in text and "[1]" not in text
    assert "# Песня" not in text and "Песня" in text and "Строка стиха" in text                # стих — не глава
    book = app.AudiobookService(app.AppContext()).open_book(path, "Северная повесть")
    headings = [app.display_heading(ch["heading"]) for ch in book.chapters]
    assert book.profile_used == "markdown"
    assert headings == ["Часть 1. Глава 1.1", "Глава 1.2", "Часть 2. Глава 2.1", "Глава 2.2"]
    assert app.book_stem(path) == "north" and app.file_title(path) == "north"


def test_damaged_books_give_a_clear_error_and_the_window_suggests_the_title_from_inside(app, window):
    ctx, view, presenter = window
    broken = app.home / "broken.epub"
    broken.write_bytes(b"not a zip")
    presenter.open_file(broken)
    assert view.errors and "повреждён или защищён" in view.errors[-1]
    presenter.open_file(make_fb2(app.home / "1.fb2"))
    assert view.title == "Северная повесть" and view.hint == "Название взято из самой книги. Всё верно?"
    assert len(view.rows) == 4


def test_python_for_the_venv_prefers_3_14_then_3_13_then_3_12_then_the_running_one(app, monkeypatch):
    asked = []

    def installed_only(*versions):
        def fake_run(command, **kwargs):
            asked.append(command[1])
            path = rf"C:\Python{command[1][1:].replace('.', '')}\python.exe" if command[1][1:] in versions else ""
            return type("Result", (), {"returncode": 0 if path else 103, "stdout": path + "\n" if path else ""})()
        return fake_run
    monkeypatch.setattr(app.shutil, "which", lambda name: "py")
    monkeypatch.setattr(app.subprocess, "run", installed_only("3.14", "3.13", "3.12"))
    assert app.find_base_python(windows=True) == r"C:\Python314\python.exe" and asked == ["-3.14"]   # 3.14 есть: берём его
    asked.clear()
    monkeypatch.setattr(app.subprocess, "run", installed_only("3.13", "3.12"))
    assert app.find_base_python(windows=True) == r"C:\Python313\python.exe" and asked == ["-3.14", "-3.13"]
    asked.clear()
    monkeypatch.setattr(app.subprocess, "run", installed_only("3.12"))
    assert app.find_base_python(windows=True) == r"C:\Python312\python.exe" and asked == ["-3.14", "-3.13", "-3.12"]
    monkeypatch.setattr(app.subprocess, "run", lambda command, **kw: type("R", (), {"returncode": 103, "stdout": ""})())
    assert app.find_base_python(windows=True) == sys.executable                       # py ничего не нашёл: текущий Python


def test_theme_colors_are_applied_at_once_and_the_language_switch_stays_on_top_after_switching(app, monkeypatch):
    pytest.importorskip("tkinterdnd2")
    pytest.importorskip("sv_ttk")
    tk = pytest.importorskip("tkinter")
    from tkinter import ttk
    ctx = app.AppContext()
    for theme, background in (("dark", "#1c1c1c"), ("light", "#fafafa")):
        try:
            window = app.MainWindow(lambda view: app.MainPresenter(app.AudiobookService(ctx, app.EventBus()), view), theme=theme)
        except tk.TclError:
            pytest.skip("нет экрана для окна Tk")
        try:
            style = ttk.Style(window.root)
            assert str(style.lookup("TFrame", "background")) == background == window.colors["bg"]   # без ожидания событий
            assert str(style.lookup("TLabel", "foreground")) == window.colors["fg"] != background
            window._draw_drop(True)                                               # подсветка смешивает цвета темы
            if theme == "dark":
                monkeypatch.delenv("AUDIOBOOK_LANG")
                for language in ("en", "ru", "en"):
                    window.presenter.change_language(language)
                    window.root.update()
                    label = window.language_labels["ru"]
                    hit = window.root.winfo_containing(label.winfo_rootx() + 3, label.winfo_rooty() + 3)
                    assert hit is label                                           # переключатель видно и по нему можно нажать
                assert str(style.lookup("TFrame", "background")) == background
        finally:
            window.root.destroy()


def fake_qwen(app, monkeypatch, capability=(7, 5), vram_gb=8, patched_error=None, patch_error=None):
    """Заглушки torch, faster-qwen3-tts и qwen3-tts-triton. patched_error — ошибка озвучки с ускорением,
    patch_error — ошибка при включении ускорения."""
    state = {"loads": [], "patches": 0}

    class FakeModel:
        def __init__(self):
            self.model, self.patched = types.SimpleNamespace(), False

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            state["loads"].append(kwargs["dtype"])
            model = cls()
            state["last"] = model
            return model

        def generate_voice_clone(self, **kwargs):
            if self.patched and patched_error:
                raise patched_error
            return [np.ones(2400, dtype="float32") * 0.1], 24000

    def apply(inner, **kwargs):
        state["patches"] += 1
        if patch_error:
            raise patch_error
        state["last"].patched = True
    monkeypatch.setitem(sys.modules, "torch", fake_torch(capability, vram_gb, "NVIDIA GeForce RTX 2070 Max-Q"))
    monkeypatch.setitem(sys.modules, "faster_qwen3_tts", types.SimpleNamespace(FasterQwen3TTS=FakeModel))
    monkeypatch.setitem(sys.modules, "qwen3_tts_triton", types.ModuleType("qwen3_tts_triton"))
    monkeypatch.setitem(sys.modules, "qwen3_tts_triton.models", types.ModuleType("models"))
    monkeypatch.setitem(sys.modules, "qwen3_tts_triton.models.patching",
                        types.SimpleNamespace(find_patchable_model=lambda m: m, apply_triton_kernels=apply))
    sf.write(str(app.home / "ref.wav"), np.zeros(24000, dtype="float32"), 24000)
    ctx = app.AppContext()
    ctx.update(qwen_voices=[{"name": "Мой голос", "path": "ref.wav", "text": "Точный текст."}])
    bus, log = app.EventBus(), []
    bus.subscribe(lambda kind, data: log.append(data.get("text", "")))
    return ctx, bus, log, state


def test_modern_cards_get_triton_acceleration_in_fp16(app, monkeypatch):
    ctx, bus, log, state = fake_qwen(app, monkeypatch, capability=(8, 6))
    engine = app.QwenEngine(ctx, bus)
    assert engine.triton is True and state["loads"] == ["float16"] and len(engine.synth("Привет.")) == 2400


def test_when_triton_kernels_fail_during_narration_the_engine_continues_without_them_and_remembers_it(app, monkeypatch):
    ctx, bus, log, state = fake_qwen(app, monkeypatch, patched_error=RuntimeError("triton compile failed: ptxas error"))
    engine = app.QwenEngine(ctx, bus)
    assert engine.triton is True
    assert len(engine.synth("Привет.")) == 2400                                    # тот же кусок озвучен без ускорения
    assert engine.triton is False and len(state["loads"]) == 2 and state["patches"] == 1
    assert ctx.settings.qwen_no_triton_gpu == f"NVIDIA GeForce RTX 2070 Max-Q|{app.APP_VERSION}"
    assert any("Triton" in line and "не заработало" in line for line in log)
    again = app.QwenEngine(ctx, bus)                                              # в следующий раз сразу без него
    assert again.triton is False and state["patches"] == 1


def test_when_triton_cannot_even_be_switched_on_the_model_is_reloaded_without_it(app, monkeypatch):
    ctx, bus, log, state = fake_qwen(app, monkeypatch, patch_error=ImportError("DLL load failed while importing triton"))
    engine = app.QwenEngine(ctx, bus)
    assert engine.triton is False and len(state["loads"]) == 2 and len(engine.synth("Привет.")) == 2400


def test_other_errors_are_not_mistaken_for_triton_failures(app, monkeypatch):
    ctx, bus, log, state = fake_qwen(app, monkeypatch, patched_error=ValueError("пустой текст"))
    engine = app.QwenEngine(ctx, bus)
    with pytest.raises(ValueError):
        engine.synth("Привет.")
    assert engine.triton is True and ctx.settings.qwen_no_triton_gpu == ""


@pytest.mark.parametrize("capability, vram_gb, dtype", [
    ((6, 1), 11, "float32"), ((6, 1), 8, "float16"), ((5, 2), 12, "float32"), ((7, 0), 16, "float16")])
def test_older_cards_run_without_triton_and_use_fp32_when_memory_allows(app, monkeypatch, capability, vram_gb, dtype):
    ctx, bus, log, state = fake_qwen(app, monkeypatch, capability=capability, vram_gb=vram_gb)
    engine = app.QwenEngine(ctx, bus)
    assert engine.triton is False and state["patches"] == 0 and state["loads"] == [dtype]
    assert any("без ускорения Triton" in line for line in log)


def test_gpu_generation_is_read_from_nvidia_smi(app, monkeypatch):
    monkeypatch.setattr(app.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="6.1\n", returncode=0))
    assert app.gpu_compute_capability() == (6, 1) and app.legacy_gpu() is True
    old_driver = types.SimpleNamespace(stdout='Field "compute_cap" is not a valid field', returncode=2)
    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: old_driver)
    assert app.gpu_compute_capability() is None and app.legacy_gpu() is False          # старый драйвер: считаем новой картой
    assert [app.gpu_generation(c) for c in ((12, 0), (8, 9), (8, 6), (7, 5), (7, 0), (6, 1), (5, 2))] == [
        "Blackwell", "Ada Lovelace", "Ampere", "Turing", "Volta", "Pascal", "Maxwell"]


def test_installing_qwen_for_an_older_card_uses_the_legacy_pytorch_build_and_skips_triton(app, monkeypatch):
    calls, checks = [], []
    monkeypatch.setattr(app, "pip", lambda *a, **k: calls.append(" ".join(a)) or True)
    monkeypatch.setattr(app, "gpu_compute_capability", lambda: (6, 1))
    monkeypatch.setattr(app, "torch_cuda_status", lambda: ("2.11.0+cu128", True))       # стояла сборка, которая карту не знает
    monkeypatch.setattr(app.subprocess, "run", lambda command, **k: checks.append(command[-1]) or types.SimpleNamespace(
        stdout="OK", stderr="", returncode=0))
    monkeypatch.setattr(app, "loaded_lib_versions", lambda: "то же")
    assert app.install_engine("qwen") is True
    assert any("torch==2.11.0+cu126" in line and app.LEGACY_TORCH_INDEX in line for line in calls)
    assert not any("triton" in line for line in calls) and "qwen3_tts_triton" not in checks[-1]


def test_readiness_for_an_older_card_does_not_ask_for_triton_but_spots_the_wrong_pytorch_build(app, monkeypatch):
    ctx = app.AppContext()
    monkeypatch.setattr(app, "gpu_compute_capability", lambda: (6, 1))
    monkeypatch.setattr(app, "gpu_info", lambda: {"name": "GTX 1080", "vram_gb": 8.0, "driver": "566.36"})
    monkeypatch.setattr(app, "pkg_version", lambda name: "0.5.3" if name == "faster-qwen3-tts" else None)
    monkeypatch.setattr(app, "torch_cuda_status", lambda: ("2.11.0+cu128", True))
    fix, _ = app.engine_problems(ctx, "qwen")
    assert any("Pascal" in item and "старых карт" in item for item in fix) and not any("triton" in item for item in fix)
    monkeypatch.setattr(app, "torch_cuda_status", lambda: ("2.11.0+cu126", True))
    assert app.engine_problems(ctx, "qwen")[0] == []
    assert app.qwen_warnings() == [] or not any("570" in w for w in app.qwen_warnings())


def test_diagnostics_show_the_card_generation_and_the_qwen_mode(app, monkeypatch):
    ctx = app.AppContext()
    monkeypatch.setattr(app, "gpu_info", lambda: {"name": "RTX 2070 Max-Q", "vram_gb": 8.0, "driver": "576.02"})
    monkeypatch.setattr(app, "gpu_compute_capability", lambda: (7, 5))
    assert any("Turing (7.5)" in line and "с ускорением Triton" in line for line in app.diagnostics_lines(ctx))
    ctx.update(qwen_no_triton_gpu=f"RTX 2070 Max-Q|{app.APP_VERSION}")
    assert any("без Triton (на этой карте" in line for line in app.diagnostics_lines(ctx))


def test_prepare_mode_installs_the_window_libraries_and_marks_the_venv_ready(app, monkeypatch):
    monkeypatch.setattr(app, "ensure_gui_packages", lambda: True)
    monkeypatch.setattr(sys, "argv", ["audiobook_master.py", "--prepare"])
    app.PATHS.venv.mkdir(parents=True, exist_ok=True)
    assert app.main() == 0 and (app.PATHS.venv / "audiobook-ready.txt").read_text(encoding="utf-8") == app.APP_VERSION
    (app.PATHS.venv / "audiobook-ready.txt").unlink()
    monkeypatch.setattr(app, "ensure_gui_packages", lambda: False)
    assert app.main() == 2 and not (app.PATHS.venv / "audiobook-ready.txt").exists()     # нет Tk: запускатель предложит консоль


def write_voice(app, name="anna.wav", seconds=2.0):
    path = app.home / name
    sf.write(str(path), np.full(int(24000 * seconds), 0.05, dtype="float32"), 24000)
    return path


def test_the_old_ref_wav_becomes_the_first_voice_of_the_library(app):
    ctx = app.AppContext()
    ctx.update(ref_text="Старый текст образца.")
    assert app.library_voices(ctx.settings) == []
    write_voice(app, "ref.wav")
    ctx = app.AppContext()
    assert app.library_voices(ctx.settings) == [{"name": "Мой голос", "path": "ref.wav", "text": "Старый текст образца."}]
    assert app.current_voice(ctx.settings)["name"] == "Мой голос"
    app.AppContext()                                                              # второй раз ничего не дублируется
    assert len(app.library_voices(app.AppContext().settings)) == 1


def test_voice_problems_are_explained(app):
    write_voice(app, "anna.wav")
    others = [{"name": "Анна", "path": "anna.wav", "text": "Т."}]
    good = {"name": "Борис", "path": "anna.wav", "text": "Привет."}
    assert app.voice_problem(good, others) == ""
    assert "уже есть" in app.voice_problem(dict(good, name="анна"), others)
    assert app.voice_problem(dict(good, name="Анна"), others, editing="Анна") == ""        # своё название при правке
    assert "название" in app.voice_problem(dict(good, name=" "), others)
    assert "не найден" in app.voice_problem(dict(good, path="нет.wav"), others)
    (app.home / "текст.txt").write_text("не звук", encoding="utf-8")
    assert "формате" in app.voice_problem(dict(good, path="текст.txt"), others)
    assert "текст" in app.voice_problem(dict(good, text=" "), others)
    write_voice(app, "long.wav", seconds=200)
    assert "длится" in app.voice_problem(dict(good, path="long.wav"), others)


def test_settings_manage_voices_and_apply_them_only_on_save(app, settings_window):
    ctx, view, presenter, _ = settings_window
    write_voice(app, "anna.wav")
    view.voice_answers = [{"name": "", "path": str(app.home / "anna.wav"), "text": "Привет."},          # нет названия
                          {"name": "Анна", "path": str(app.home / "anna.wav"), "text": "Привет."}]
    presenter.add_voice()
    assert "название" in view.errors[-1] and view.voice_asked[-1]["path"].endswith("anna.wav")         # спросили снова
    assert view.voices == [{"name": "Анна", "path": "anna.wav", "text": "Привет."}] and view.chosen_voice == "Анна"
    assert app.library_voices(ctx.settings) == []                                                      # ещё не сохранено
    assert presenter.save(raw_values(view)) is True
    assert app.current_voice(ctx.settings) == {"name": "Анна", "path": "anna.wav", "text": "Привет."}
    view.voice_answers = [{"name": "Анна Иванова", "path": "anna.wav", "text": "Привет, мир."}]
    presenter.edit_voice(0)
    presenter.play_voice(0)
    assert view.chosen_voice == "Анна Иванова" and view.played == [app.home / "anna.wav"]
    view.yes = False
    presenter.delete_voice(0)
    assert len(view.voices) == 1
    view.yes = True
    presenter.delete_voice(0)
    assert view.voices == [] and view.chosen_voice == ""
    presenter.save(raw_values(view))
    assert app.library_voices(ctx.settings) == [] and app.current_voice(ctx.settings) is None


def test_main_window_lists_qwen_voices_with_an_add_item_and_gemini_voices(app, window, monkeypatch):
    ctx, view, presenter = window
    presenter.set_engine("qwen")
    assert view.voice_items == ["＋ Добавить голос…"] and view.voice_index == 0
    opened = []
    monkeypatch.setattr(presenter, "open_settings", lambda: opened.append(True))
    presenter.choose_voice(0)
    assert opened == [True]
    write_voice(app, "a.wav")
    write_voice(app, "b.wav")
    ctx.update(qwen_voices=[{"name": "Анна", "path": "a.wav", "text": "А."}, {"name": "Борис", "path": "b.wav", "text": "Б."}])
    presenter.settings_changed()
    assert view.voice_items == ["Анна", "Борис", "＋ Добавить голос…"] and view.voice_index == 0
    presenter.choose_voice(1)
    assert ctx.settings.qwen_voice == "Борис" and view.voice_index == 1
    presenter.set_engine("gemini")
    assert view.voice_items[view.voice_index].startswith("Kore")
    presenter.choose_voice(1)
    assert ctx.settings.gemini_voice == "Puck"


@needs_ffmpeg
def test_a_book_remembers_its_voice_and_switching_voice_restarts_half_done_chapters(app, window, monkeypatch):
    ctx, view, presenter = window
    write_voice(app, "a.wav")
    write_voice(app, "b.wav", seconds=3)
    ctx.update(qwen_voices=[{"name": "Анна", "path": "a.wav", "text": "А."}, {"name": "Борис", "path": "b.wav", "text": "Б."}],
               qwen_voice="Борис")
    book_path = write_book(app, "Рассказ.txt", chapters=2)
    presenter.open_file(book_path)
    presenter.book.state.voice = "Анна"
    presenter.book.save()
    presenter.open_file(book_path)
    assert ctx.settings.qwen_voice == "Анна" and any("Анна" in text for text in view.infos)
    chapter = presenter.book.chapters[0]
    before = app.chapter_digest(ctx, chapter, "qwen")
    presenter.choose_voice(1)
    assert app.chapter_digest(ctx, chapter, "qwen") != before                    # другой голос — другой отпечаток


# ---------------------------------------------------------------- версия формата, копии, проверка вложенных значений

def test_state_files_carry_a_schema_version(app):
    ctx = app.AppContext()
    ctx.update(gemini_voice="Puck")
    assert json.loads(app.PATHS.settings.read_text(encoding="utf-8"))["schema_version"] == app.SCHEMA_VERSION
    book = make_book(app, ctx, chapters=1)
    assert json.loads((app.chapters_dir(book.dir) / "split.json").read_text(encoding="utf-8"))["schema_version"] \
        == app.SCHEMA_VERSION
    assert app.AppContext().settings.gemini_voice == "Puck"                     # лишний ключ чтению не мешает


def test_a_damaged_settings_file_is_kept_as_a_copy_before_being_overwritten(app, capsys):
    app.PATHS.settings.write_text('{"gemini_voice": "Puck", оборван', encoding="utf-8")
    ctx = app.AppContext()
    assert ctx.settings.gemini_voice == "Kore"                                   # значения по умолчанию
    bad = app.PATHS.settings.with_name("settings.json.bad")
    assert bad.read_text(encoding="utf-8").startswith('{"gemini_voice": "Puck"')
    assert "settings.json.bad" in capsys.readouterr().out
    ctx.update(gemini_voice="Aoede")                                             # обычное сохранение поверх
    assert bad.exists() and app.AppContext().settings.gemini_voice == "Aoede"
    app.PATHS.settings.write_text("[1, 2, 3]", encoding="utf-8")                 # JSON, но не объект
    assert app.AppContext().settings.gemini_voice == "Kore"
    assert bad.read_text(encoding="utf-8") == "[1, 2, 3]"


def test_a_missing_settings_file_is_not_an_error_and_makes_no_copy(app):
    app.PATHS.settings.unlink(missing_ok=True)
    assert app.AppContext().settings.gemini_voice == "Kore"
    assert not list(app.PATHS.settings.parent.glob("settings.json.*"))


def test_a_file_from_a_newer_program_is_copied_before_it_is_overwritten(app):
    app.PATHS.settings.write_text(json.dumps({"schema_version": 99, "gemini_voice": "Puck", "новое_поле": 1}), encoding="utf-8")
    ctx = app.AppContext()
    assert ctx.settings.gemini_voice == "Puck"
    ctx.update(gemini_voice="Aoede")
    old = json.loads(app.PATHS.settings.with_name("settings.json.v99").read_text(encoding="utf-8"))
    assert old["новое_поле"] == 1
    assert json.loads(app.PATHS.settings.read_text(encoding="utf-8"))["schema_version"] == app.SCHEMA_VERSION


def test_nested_settings_values_are_validated(app, tmp_path):
    book = tmp_path / "1.txt"
    book.write_text("Глава 1\n\nТекст.", encoding="utf-8")
    good = {"title": "Книга", "sha": "abc"}
    app.PATHS.settings.write_text(json.dumps({
        "book_titles": {str(book): good, "строка": "не словарь", "без названия": {"sha": "x"}, "пустое": {"title": " "},
                        "число": {"title": 5}},
        "qwen_voices": [{"name": "А", "path": "a.wav", "text": "т"}, "мусор", {"name": "Б"}, {"name": 1, "path": "", "text": ""}],
    }), encoding="utf-8")
    ctx = app.AppContext()
    assert ctx.settings.book_titles == {str(book): good}
    assert ctx.settings.qwen_voices == [{"name": "А", "path": "a.wav", "text": "т"}]
    assert app.remembered_title(ctx, book, "другой текст")[0] == "Книга"          # не падает на любой записи


# ---------------------------------------------------------------- язык и настройки книги, проба голоса

@pytest.mark.parametrize("text, language", [
    ("Это обычный русский текст книги про лес и реку. " * 40, "Russian"),
    ("This is an ordinary English text about the forest and the river, and it was quiet. " * 40, "English"),
    ("Der alte Mann ging mit dem Hund in den Wald und die Sonne war nicht zu sehen. " * 40, "German"),
    ("Le vieil homme marchait dans la forêt avec son chien et il ne voyait pas le soleil. " * 40, "French"),
    ("El viejo caminaba por el bosque con su perro y no se veía el sol en el cielo. " * 40, "Spanish"),
    ("Il vecchio camminava nel bosco con il suo cane e non si vedeva il sole del giorno. " * 40, "Italian"),
    ("O velho caminhava na floresta com o seu cão e não se via o sol no céu da manhã. " * 40, "Portuguese"),
    ("森の中を老人が犬と一緒に歩いていました。太陽は見えませんでした。" * 40, "Japanese"),
    ("老人和他的狗在森林里散步，看不见太阳。" * 40, "Chinese"),
    ("노인은 개와 함께 숲 속을 걸었고 해는 보이지 않았다. " * 40, "Korean"),
    ("12345 67890 ...", ""),
], ids=["ru", "en", "de", "fr", "es", "it", "pt", "ja", "zh", "ko", "none"])
def test_book_language_is_detected_from_the_text(app, text, language):
    assert app.detect_language(text) == language


def test_detecting_the_language_of_a_big_book_is_fast(app):
    text = "Это обычный текст книги. " * 400_000                                # около 10 млн символов
    started = time.perf_counter()
    assert app.detect_language(text) == "Russian" and time.perf_counter() - started < 0.5


def test_windows_line_breaks_in_txt_books_are_removed(app):
    path = app.home / "crlf.txt"
    path.write_bytes("Глава 1\r\n\r\nТекст.\r\n".encode("utf-8"))
    assert app.read_book_text(path) == "Глава 1\n\nТекст.\n"


def test_each_book_keeps_its_language_and_gemini_choice_and_brings_them_back(app, window):
    ctx, view, presenter = window
    english = app.home / "Novel.txt"
    english.write_text("Chapter 1. One\n\n" + "This is an ordinary text and it was quiet. " * 60 + "\n\n"
                       "Chapter 2. Two\n\n" + "The river was cold and the forest was dark. " * 60, encoding="utf-8")
    presenter.open_file(english)
    assert ctx.settings.qwen_language == "English" and view.language_items[0] == "Авто (Английский)" and view.language_index == 0
    presenter.choose_language(app.QWEN_LANGUAGES.index("German") + 1)              # выбрали вручную
    assert ctx.settings.qwen_language == "German" and presenter.book.state.language == "German"
    presenter.set_engine("gemini")
    presenter.choose_voice(1)                                                       # Puck
    app.remember_book_profile(ctx, presenter.book)
    presenter.book.save()
    presenter.open_file(write_book(app, "Рассказ.txt"))                             # другая книга: русский по тексту
    assert ctx.settings.qwen_language == "Russian"
    ctx.update(gemini_voice="Kore")
    presenter.open_file(english)                                                    # вернулись: всё как было у этой книги
    assert ctx.settings.qwen_language == "German" and ctx.settings.gemini_voice == "Puck"
    assert view.language_index == app.QWEN_LANGUAGES.index("German") + 1
    presenter.choose_language(0)                                                    # снова «Авто»
    assert presenter.book.state.language == "" and ctx.settings.qwen_language == "English"


def test_reading_rules_follow_the_book_language(app):
    assert app.normalize_for_speech("В 1984 году, т. е. давно.", language="Russian").startswith("В тысяча")
    assert app.normalize_for_speech("In 1984, i.e. long ago.", language="English").startswith("In nineteen")
    assert app.normalize_for_speech("Im Jahr 1984, d. h.", language="German") == "Im Jahr 1984, d. h."     # правил нет


def test_preview_text_is_the_first_sentences_without_the_heading(app):
    text = "Глава 1. Начало\n\nПервое предложение. Второе! Третье? Четвёртое."
    assert app.preview_sentences(text) == "Первое предложение. Второе!"


@needs_ffmpeg
def test_listen_voices_the_first_sentences_of_the_selected_chapter_with_the_current_engine(app, window, monkeypatch):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt", chapters=2))
    presenter.set_engine("dummy")
    presenter.select([2])
    spoken = []
    real = app.make_voice_preview
    monkeypatch.setattr(app, "make_voice_preview", lambda *a, **k: (spoken.append(a[2]), real(*a, **k))[1])
    presenter.preview()
    assert view.preview_busy is True                                               # пока идёт проба, на кнопке «Стоп»
    presenter.pump()
    assert view.played == [app.PATHS.cache / "preview.wav"] and (app.PATHS.cache / "preview.wav").exists()
    assert spoken and spoken[0] in presenter.book.chapters[1]["text"] and view.preview_busy is False


def test_listen_without_a_gemini_key_asks_for_it_and_shows_errors(app, window, monkeypatch):
    ctx, view, presenter = window
    presenter.set_engine("gemini")
    view.texts = [None]                                                             # ключ не ввели
    presenter.preview()
    assert not presenter.previewing
    monkeypatch.setattr(app, "make_voice_preview", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("нет сети")))
    monkeypatch.setattr(app.AudiobookService, "readiness", lambda self, engine: ([], []))
    ctx.update(gemini_api_key="AIzaTESTKEY0000001")
    presenter.preview()
    presenter.pump()
    assert any("Проба не удалась" in text and "нет сети" in text for text in view.infos) and not presenter.previewing


# ---------------------------------------------------------------- сведения о книге и обложка

def png_bytes(app):
    import base64
    return base64.b64decode(app.APP_ICON_PNG)


def epub_with_metadata(app, path):
    import zipfile as zf
    with zf.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:'
                   'container" version="1.0"><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>')
        z.writestr("OEBPS/content.opf", '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Морская книга</dc:title>'
                   '<dc:creator>Анна Петрова</dc:creator><dc:description>&lt;p&gt;Книга о &lt;b&gt;море&lt;/b&gt;.&lt;/p&gt;'
                   '</dc:description><meta name="calibre:series" content="Морской цикл"/>'
                   '<meta name="calibre:series_index" content="3.0"/></metadata>'
                   '<manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
                   '<item id="img" href="img/cover.png" media-type="image/png" properties="cover-image"/></manifest>'
                   '<spine><itemref idref="c1"/></spine></package>')
        z.writestr("OEBPS/c1.xhtml", f'<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Глава 1</h1><p>{PROSE}</p>'
                   f'<h1>Глава 2</h1><p>{PROSE}</p></body></html>')
        z.writestr("OEBPS/img/cover.png", png_bytes(app))
    return path


def fb2_with_metadata(app, path):
    import base64
    cover = base64.b64encode(png_bytes(app)).decode()
    xml = ('<?xml version="1.0" encoding="utf-8"?><FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0" '
           'xmlns:l="http://www.w3.org/1999/xlink"><description><title-info><author><first-name>Лев</first-name>'
           '<last-name>Толстой</last-name></author><book-title>Повесть</book-title><annotation><p>О жизни.</p></annotation>'
           '<coverpage><image l:href="#cover.png"/></coverpage><sequence name="Ранние повести" number="2"/></title-info>'
           f'</description><body><section><title><p>Глава 1</p></title><p>{PROSE}</p></section>'
           f'<section><title><p>Глава 2</p></title><p>{PROSE}</p></section></body>'
           f'<binary id="cover.png" content-type="image/png">{cover}</binary></FictionBook>')
    path.write_text(xml, encoding="utf-8")
    return path


def test_details_and_cover_are_read_from_epub_and_fb2(app):
    epub = app.book_metadata(epub_with_metadata(app, app.home / "sea.epub"))
    assert (epub["author"], epub["series"], epub["series_index"], epub["description"]) == (
        "Анна Петрова", "Морской цикл", "3", "Книга о море .")
    assert epub["cover"] == (png_bytes(app), ".png")
    fb2 = app.book_metadata(fb2_with_metadata(app, app.home / "tale.fb2"))
    assert (fb2["author"], fb2["series"], fb2["series_index"]) == ("Лев Толстой", "Ранние повести", "2")
    assert fb2["description"] == "О жизни."
    assert fb2["cover"] == (png_bytes(app), ".png")
    assert app.book_metadata(write_book(app, "Рассказ.txt")) == {}


@needs_ffmpeg
def test_mp3_files_get_author_series_description_and_the_cover(app):
    ctx = app.AppContext()
    service = app.AudiobookService(ctx)
    book = service.open_book(fb2_with_metadata(app, app.home / "tale.fb2"), "Повесть")
    assert book.state.author == "Лев Толстой" and book.state.series == "Ранние повести"
    assert service.run(book, "dummy", {1})
    data = app.out_mp3(book.dir, book.chapters[0]).read_bytes()[:20000]
    for text in ("Лев Толстой", "Ранние повести 2", "О жизни."):
        assert text.encode("utf-16-le") in data
    assert b"APIC" in data and png_bytes(app)[:16] in data


@needs_ffmpeg
def test_editing_the_details_retags_finished_mp3_files_without_redoing_them(app, window):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt", chapters=2, paragraphs=12))
    presenter.set_engine("dummy")
    presenter.start()
    presenter.pump()
    mp3 = app.out_mp3(presenter.book.dir, presenter.book.chapters[0])
    assert b"APIC" not in mp3.read_bytes()[:20000]                                  # у TXT обложки нет
    cover = app.home / "cover.png"
    cover.write_bytes(png_bytes(app))
    view.info_answer = {"author": "Иван Бунин", "series": "", "series_index": "", "description": "Рассказ.", "cover": str(cover)}
    presenter.edit_info()
    presenter.pump()
    data = mp3.read_bytes()[:20000]
    assert view.info_asked["book_cover"] is False and "Иван Бунин".encode("utf-16-le") in data and b"APIC" in data
    assert any("теги обновлены в файлах: 2" in text for text in view.infos)
    assert [row.kind for row in presenter.service.chapter_rows(presenter.book, "dummy")] == ["done", "done"]
    view.info_answer = dict(view.info_answer, cover="-")
    presenter.edit_info()
    presenter.pump()
    assert b"APIC" not in mp3.read_bytes()[:20000]


def test_russian_and_english_books_keep_their_fingerprints_other_languages_get_their_own(app):
    ctx = app.AppContext()
    signatures = {}
    for language in ("", "Russian", "English", "German"):
        ctx.update(qwen_language=language)
        signatures[language] = app.speech_signature(ctx.settings)
    assert signatures[""] == signatures["Russian"] == signatures["English"] != signatures["German"]


def test_real_window_has_one_button_that_narrates_and_stops(app):
    tk = pytest.importorskip("tkinter")
    pytest.importorskip("tkinterdnd2")
    pytest.importorskip("sv_ttk")
    service = app.AudiobookService(app.AppContext(), app.EventBus())
    try:
        window = app.MainWindow(lambda view: app.MainPresenter(service, view))
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")
    try:
        window.root.update()
        calls = []
        window.presenter.start = lambda: calls.append("start")
        window.presenter.stop = lambda: calls.append("stop")
        button = window.start_button
        assert not hasattr(window, "stop_button")                                          # кнопка одна
        assert window.settings_button.grid_info()["column"] < window.start_button.grid_info()["column"]   # озвучка справа
        window.set_state(True, False, "Озвучить")
        assert button.cget("text") == "Озвучить" and str(button.cget("style")) == "Accent.TButton"
        assert button.instate(["!disabled"])
        button.invoke()
        window.set_state(True, True, "Озвучить")                                           # пошла озвучка
        assert button.cget("text") == "Стоп" and str(button.cget("style")) == "Stop.TButton" and button.instate(["!disabled"])
        button.invoke()
        window.set_stopping()                                                              # «Стоп» нажат
        assert button.cget("text") == "Останавливаю…" and button.instate(["disabled"])
        window.set_state(True, True, "Озвучить")              # обновление статуса не сбрасывает ожидание
        assert button.cget("text") == "Останавливаю…" and button.instate(["disabled"])
        window.set_state(True, False, "Озвучить")                                          # работа закончилась
        assert button.cget("text") == "Озвучить" and button.instate(["!disabled"])
        assert calls == ["start", "stop"]
        assert window.hint.winfo_manager() == ""                                           # стартовой подсказки и отступа нет
        window.set_hint("Введите название книги", True)
        assert window.hint.winfo_manager() == "grid"
        window.set_hint("", False)
        assert window.hint.winfo_manager() == ""
        window.set_busy("Установка…")                                                      # журнал сам не открывается
        assert window.log_open is False
        window.append_log("❌ Ошибка")
        assert window.log_open is True
    finally:
        window.root.destroy()


def test_stop_is_sent_once_and_the_button_switches_to_stopping(app, window):
    _, view, presenter = window
    presenter.running, presenter.token = True, app.CancellationToken()
    presenter.stop()
    presenter.stop()
    assert view.stopping is True and view.log.count("Останавливаю…") == 1


def test_tooltip_appears_only_while_the_pointer_stays_and_goes_away_when_it_leaves(app, monkeypatch):
    pytest.importorskip("tkinterdnd2")
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")
    try:
        button = tk.Button(root, text="▶")
        button.pack()
        root.update()
        tip = app.Tooltip(button, "Подсказка", {"bg": "#ffffff", "fg": "#000000"})
        tip.schedule()
        tip.hide()                                                       # мышь ушла раньше задержки
        root.after(app.Tooltip.DELAY_MS + 100, root.quit)
        root.mainloop()
        assert tip.tip is None and tip.pending is None
        monkeypatch.setattr(button, "winfo_containing", lambda x, y: button)  # мышь стоит над кнопкой
        tip.show()
        assert tip.tip is not None
        tip.hide()
        assert tip.tip is None
        monkeypatch.setattr(button, "winfo_containing", lambda x, y: root)    # мышь уже не над кнопкой
        tip.show()
        assert tip.tip is None
    finally:
        root.destroy()


# ---------------------------------------------------------------- версии библиотек после обновления программы

def fake_versions(app, monkeypatch, versions):
    monkeypatch.setattr(app, "pkg_version", lambda name: versions.get(name))


def test_only_installed_packages_older_than_the_pin_count_as_outdated(app, monkeypatch):
    fake_versions(app, monkeypatch, {"requests": "2.30.0", "numpy": "2.5.3", "soundfile": "9.9.9", "sv-ttk": "2.6.1"})
    assert app.installed_older_than_pin(app.REQUESTS_PIN) == "2.30.0"                    # старее
    assert app.installed_older_than_pin(app.NUMPY_PIN) is None                           # то же самое
    assert app.installed_older_than_pin(app.SOUNDFILE_PIN) is None                       # новее: не трогаем
    assert app.installed_older_than_pin("tkinterdnd2==0.6.4") is None                    # не стоит: это отдельный случай
    fake_versions(app, monkeypatch, {"torch": "2.10.0+cu128"})
    assert app.installed_older_than_pin("torch==2.11.0+cu128") == "2.10.0+cu128"         # метка сборки не мешает сравнению
    assert app.pin_parts("torch==2.11.0+cu128") == ("torch", "2.11.0")


def test_start_reinstalls_only_the_light_packages_that_are_older(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "pip", lambda *a, **k: calls.append(a) or True)
    versions = {"numpy": "2.5.3", "soundfile": "0.14.0", "num2words": "0.5.14", "requests": "2.30.0",
                "tkinterdnd2": "0.6.4", "sv-ttk": "2.5.0"}
    fake_versions(app, monkeypatch, versions)
    app.sync_light_packages()
    assert calls == [("install", app.REQUESTS_PIN), ("install", "sv-ttk==2.6.1")]
    calls.clear()
    fake_versions(app, monkeypatch, {name: "99.0.0" for name in versions})
    app.sync_light_packages()
    assert calls == []                                                                   # всё актуально: pip не запускается


def test_a_failed_light_update_does_not_stop_the_program(app, monkeypatch, capsys):
    monkeypatch.setattr(app, "pip", lambda *a, **k: False)
    fake_versions(app, monkeypatch, {"requests": "2.30.0"})
    app.sync_light_packages()
    assert "Не удалось обновить requests" in capsys.readouterr().out


def test_readiness_reports_older_qwen_components_but_never_asks_to_replace_torch(app, monkeypatch):
    ctx = app.AppContext()
    monkeypatch.setattr(app, "gpu_compute_capability", lambda: (8, 6))
    monkeypatch.setattr(app, "gpu_info", lambda: {"name": "RTX", "vram_gb": 8.0, "driver": "566.36"})
    monkeypatch.setattr(app, "torch_cuda_status", lambda: ("2.9.0+cu128", True))            # torch старее закреплённого
    current = {pin_name: version for pin_name, version in (app.pin_parts(p) for p in app.QWEN_CORE_PKGS if "==" in p)}
    current.update({"accelerate": "1.15.0", "qwen3-tts-triton": "0.3.0", "triton-windows": "3.8.0", "triton": "3.8.0"})
    fake_versions(app, monkeypatch, dict(current))
    assert not any("старее" in item for item in app.engine_problems(ctx, "qwen")[0])      # torch сам по себе не повод
    old = dict(current, **{"transformers": "5.0.0"})
    fake_versions(app, monkeypatch, old)
    fix = app.engine_problems(ctx, "qwen")[0]
    assert any("transformers 5.0.0 старее нужного" in item for item in fix)


def test_choosing_from_a_list_or_spinning_a_number_leaves_no_selected_text(app):
    tk = pytest.importorskip("tkinter")
    from tkinter import ttk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")
    try:
        app.tidy_selection(root)
        box = ttk.Combobox(root, values=["Анна", "Борис"])
        spin = ttk.Spinbox(root, from_=0, to=10)
        box.pack()
        spin.pack()
        box.set("Анна")
        box.selection_range(0, "end")
        box.event_generate("<<ComboboxSelected>>")
        spin.set("3")
        spin.selection_range(0, "end")
        spin.event_generate("<<Increment>>")
        root.update()
        assert not box.selection_present() and not spin.selection_present()
    finally:
        root.destroy()


def test_theme_changes_at_once_and_the_book_stays_in_the_window(app, monkeypatch):
    pytest.importorskip("tkinterdnd2")
    pytest.importorskip("sv_ttk")
    tk = pytest.importorskip("tkinter")
    from tkinter import ttk
    ctx = app.AppContext()
    try:
        window = app.MainWindow(lambda view: app.MainPresenter(app.AudiobookService(ctx, app.EventBus()), view), theme="light")
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")
    try:
        window.presenter.open_file(write_book(app, "Рассказ.txt", chapters=3))
        window.table.selection_set(window.table.get_children()[1])
        window.root.update()
        window.theme_labels["dark"].event_generate("<Button-1>")                 # ☾ в левом верхнем углу
        window.root.update()
        assert window.dark is True and window.colors["bg"] == "#1c1c1c" and ctx.settings.theme == "dark"
        assert str(window.theme_labels["dark"].cget("foreground")) == window.colors["accent"]
        assert str(ttk.Style(window.root).lookup("TFrame", "background")) == "#1c1c1c"
        assert len(window.table.get_children()) == 3 and window.selected_ids() == [2] and window.title_var.get() == "Рассказ"
        assert str(window.summary.cget("foreground")) == window.colors["muted"] == app.PALETTES["dark"]["muted"]
        assert str(window.log_text.cget("background")) == "#1c1c1c" and str(window.drop.cget("background")) == "#1c1c1c"
        assert str(window.table.tag_configure("done", "foreground")) == app.PALETTES["dark"]["ok"]
        before = window.frame
        window.apply_theme("light")                                              # обратно — тоже без пересборки окна
        assert window.frame is before and str(window.summary.cget("foreground")) == app.PALETTES["light"]["muted"]
        assert str(window.log_text.cget("background")) == "#fafafa"
    finally:
        window.root.destroy()


def test_helper_programs_start_without_a_console_window_on_windows(app, monkeypatch):
    assert app.no_window_flags(0) == app.CREATE_NO_WINDOW
    assert app.no_window_flags(0x00000008) == 0x00000008                    # отсоединённый процесс (перезапуск exe)
    assert app.no_window_flags(0x00000010) == 0x00000010                    # окно консоли попросили явно
    before = app.subprocess.Popen
    monkeypatch.setattr(app.os, "name", "posix")
    app.hide_child_consoles()
    assert app.subprocess.Popen is before                                    # не Windows — ничего не меняется


# ---------------------------------------------------------------- выбор языка и разбивки сохраняется сразу; одна копия программы

def saved_state(app, book):
    return app.RecordStore(app.chapters_dir(book.dir) / "split.json", app.BookState).load()


def test_language_and_split_choices_are_saved_at_once(app, window):
    _, view, presenter = window
    path = app.home / "Эпизоды.txt"
    path.write_text("".join(f"\n\nЭПИЗОД {i}\n\n" + body(3) for i in (1, 2, 3)), encoding="utf-8")
    presenter.open_file(path)
    book = presenter.book
    assert saved_state(app, book).language == "" and saved_state(app, book).profile == "auto"
    presenter.choose_language(app.QWEN_LANGUAGES.index("German") + 1)           # озвучку не начинали и программу «закрыли»
    assert saved_state(app, book).language == "German"
    presenter.choose_language(0)
    assert saved_state(app, book).language == ""
    view.texts = [r"^ЭПИЗОД \d+"]
    presenter.set_profile("custom")
    state = saved_state(app, book)
    assert state.profile == "custom" and state.params == {"regex": r"^ЭПИЗОД \d+"}
    view.texts = ["5000"]
    presenter.set_profile("size")
    assert saved_state(app, book).profile == "size" and saved_state(app, book).params == {"size": 5000}


def startup_probe(app, monkeypatch, argv):
    """Подменяет тяжёлые шаги startup() и записывает порядок вызовов."""
    calls = []
    monkeypatch.setattr(sys, "argv", ["audiobook_master.py", *argv])
    monkeypatch.setattr(app, "configure_environment", lambda: calls.append("environment"))
    monkeypatch.setattr(app, "bootstrap", lambda: calls.append("bootstrap"))
    monkeypatch.setattr(app, "remove_pip_leftovers", lambda: calls.append("leftovers"))
    monkeypatch.setattr(app, "ensure_base_packages", lambda read_only=False: calls.append(f"packages(read_only={read_only})"))
    monkeypatch.setattr(app, "APP_LOCK", None)
    real_acquire = app.acquire_app_lock
    monkeypatch.setattr(app, "acquire_app_lock", lambda wait=0: real_acquire(wait=0))       # не ждём 5 секунд
    refusals = []

    def fake_fail(message):
        refusals.append(message)
        raise SystemExit(1)
    monkeypatch.setattr(app, "fail", fake_fail)
    return calls, refusals


def test_the_app_lock_is_taken_after_bootstrap_and_before_installing_anything(app, monkeypatch):
    calls, refusals = startup_probe(app, monkeypatch, [])
    app.startup()
    assert calls == ["environment", "bootstrap", "leftovers", "packages(read_only=False)"] and app.APP_LOCK is not None
    calls.clear()
    with pytest.raises(SystemExit):                                                          # вторая копия: до установок не дошла
        app.startup()
    assert "уже запущена" in refusals[0] and "leftovers" not in calls and not any(c.startswith("packages") for c in calls)
    app.APP_LOCK.release()
    calls.clear()
    app.startup()                                                                            # после выхода первой копии замок свободен
    assert "packages(read_only=False)" in calls
    app.APP_LOCK.release()


def test_only_one_copy_creates_the_venv(app, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "in_project_venv", lambda: False)
    vpy = tmp_path / "venv" / "python.exe"
    monkeypatch.setattr(app, "venv_python", lambda: vpy)
    monkeypatch.setattr(app.PATHS, "base", tmp_path)
    created, runs = [], []
    monkeypatch.setattr(app, "find_base_python", lambda: "python")
    monkeypatch.setattr(app, "ask_yes", lambda text: True)
    monkeypatch.setattr(app.subprocess, "check_call", lambda command, **kw: created.append(command) or vpy.parent.mkdir(exist_ok=True) or vpy.write_text("x"))
    monkeypatch.setattr(app.subprocess, "run", lambda command, **kw: runs.append(command) or type("R", (), {"returncode": 0})())
    refusals = []
    monkeypatch.setattr(app, "fail", lambda message: refusals.append(message) or (_ for _ in ()).throw(SystemExit(1)))
    real_acquire = app.acquire_lock
    other = real_acquire(tmp_path / ".venv-setup.lock", wait=0)                              # окружение сейчас создаёт другая копия
    monkeypatch.setattr(app, "acquire_lock", lambda path, wait: real_acquire(path, wait=0))
    with pytest.raises(SystemExit):
        app.bootstrap()
    assert "создаёт другая копия" in refusals[0] and created == []
    other.release()
    monkeypatch.setattr(app, "acquire_lock", lambda path, wait: (vpy.parent.mkdir(exist_ok=True), vpy.write_text("x"), real_acquire(path, wait=0))[2])
    with pytest.raises(SystemExit):                                                          # другая копия успела первой
        app.bootstrap()
    assert created == [] and len(runs) == 1                                                  # не создавали, а сразу запустили из него


def test_a_huge_txt_is_refused_before_it_is_read_into_memory(app, monkeypatch):
    path = app.home / "big.txt"
    path.write_text("а" * 5000, encoding="utf-8")
    monkeypatch.setattr(app, "MAX_BOOK_CHARS", 1000)
    opened = []
    real_open = open
    monkeypatch.setattr("builtins.open", lambda *a, **k: (opened.append(a[0]), real_open(*a, **k))[1])
    with pytest.raises(app.BookFormatError):
        app._read_text_limited(path)                                    # 10 000 байт > 4 × 1000: даже не открываем
    assert str(path) not in [str(p) for p in opened]
    path.write_bytes(b"a" * 1500)
    with pytest.raises(app.BookFormatError):
        app._read_text_limited(path, block=100)                         # 1 500 символов > 1 000: обрываем на ходу
    path.write_bytes(b"a" * 900)
    assert app._read_text_limited(path, block=100) == "a" * 900


@needs_ffmpeg
def test_tags_are_updated_by_one_job_at_a_time_and_narration_waits_for_it(app, window, monkeypatch):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt", chapters=2, paragraphs=12))
    presenter.set_engine("dummy")
    jobs = []
    presenter.runner = jobs.append                                        # фоновые задачи ждут, пока мы их не запустим
    view.info_answer = {"author": "Иван Бунин", "series": "", "series_index": "", "description": "", "cover": "-"}
    presenter.edit_info()
    presenter.edit_info()                                                 # второй раз, пока первая не закончилась
    presenter.start()                                                     # и озвучка тоже ждёт
    assert len(jobs) == 1 and not presenter.running
    assert sum("Теги ещё обновляются" in text for text in view.infos) == 2
    jobs[0]()
    presenter.pump()
    assert not presenter.retagging


def test_exe_update_stays_busy_until_the_new_exe_is_in_place(app, settings_window, monkeypatch):
    ctx, view, presenter, _ = settings_window
    monkeypatch.setenv("AUDIOBOOK_EXE", str(app.home / "FoxyAudioBookMaker.exe"))
    jobs = []
    presenter.runner = jobs.append
    monkeypatch.setattr(app, "check_exe_update", lambda: ("https://example.org/FoxyAudioBookMaker.exe", "9.9.9", "• Новое", ""))
    monkeypatch.setattr(app, "apply_exe_update", lambda url: None)
    view.yes = True
    presenter.check_update()
    jobs.pop()()
    presenter.pump()                                                      # проверили и согласились: пошла загрузка
    assert view.busy == {"update": True} and len(jobs) == 1
    presenter.check_update()                                              # повторное нажатие во время загрузки
    assert len(jobs) == 1
    jobs.pop()()
    presenter.pump()
    assert view.busy == {"update": False} and getattr(view, "restarted", False)


def test_old_copies_are_removed_once_the_new_version_has_started(app, monkeypatch):
    exe = app.home / "FoxyAudioBookMaker.exe"
    exe.write_bytes(b"MZ new")
    (app.home / "FoxyAudioBookMaker.exe.prev").write_bytes(b"MZ old")
    script_prev = Path(str(app.SCRIPT_PATH) + ".prev")
    monkeypatch.setenv("AUDIOBOOK_EXE", str(exe))
    created = not script_prev.exists()
    if created:
        script_prev.write_text("# старая версия", encoding="utf-8")
    try:
        app.drop_previous_versions()
        assert not (app.home / "FoxyAudioBookMaker.exe.prev").exists() and not script_prev.exists() and exe.exists()
        app.drop_previous_versions()                                       # копий уже нет — ничего не падает
    finally:
        if created:
            script_prev.unlink(missing_ok=True)


# ---------------------------------------------------------------- проверка состояния книги и ключ кэша исходного файла

def test_invalid_values_in_the_saved_book_state_are_replaced_with_defaults(app, capsys, tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"profile": "unknown", "params": {"size": "not-a-number", "regex": 5, "other": 1},
                                "language": "Klingon", "engine": "magic", "author": "Автор", "confirmed": True}), encoding="utf-8")
    state = app.RecordStore(path, app.BookState).load()
    assert state.profile == "auto" and state.params == {"other": 1} and state.language == "" and state.engine == ""
    assert state.author == "Автор" and state.confirmed is True                       # годное сохранено
    assert "недопустимые значения" in capsys.readouterr().out
    path.write_text(json.dumps({"profile": "words", "params": {"size": "5000"}, "language": "German", "engine": "gemini"}),
                    encoding="utf-8")
    state = app.RecordStore(path, app.BookState).load()
    assert (state.profile, state.params, state.language, state.engine) == ("words", {"size": 5000}, "German", "gemini")
    for bad in (0, -5, True, 10 ** 12, 2.5, None):
        path.write_text(json.dumps({"profile": "size", "params": {"size": bad}}), encoding="utf-8")
        assert app.RecordStore(path, app.BookState).load().params == {}


def test_a_book_with_a_broken_state_file_still_opens(app):
    ctx = app.AppContext()
    service = app.AudiobookService(ctx, app.EventBus())
    book = make_book(app, ctx, chapters=2)
    (app.chapters_dir(book.dir) / "split.json").write_text(
        json.dumps({"profile": "unknown", "params": {"size": "not-a-number"}}), encoding="utf-8")
    reopened = service.open_book(book.path, book.title)
    assert reopened.state.profile == "auto" and app.PROFILE_SHORT[reopened.profile_used]
    assert [row.kind for row in service.chapter_rows(reopened, "dummy")] == ["new"] * len(reopened.chapters)


@pytest.mark.skipif(os.name == "nt", reason="в Windows ctime это время создания, а NTFS сохраняет его при замене файла")
def test_replacing_a_book_file_with_the_same_size_and_mtime_is_noticed(app, tmp_path):
    path = tmp_path / "Книга.txt"
    path.write_text("Первый текст книги.", encoding="utf-8")
    old = path.stat()
    assert app.read_book(str(path))[0].strip() == "Первый текст книги."
    time.sleep(0.05)
    path.write_text("Второй текст книги.", encoding="utf-8")                           # тот же размер
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))                             # и то же время изменения
    assert path.stat().st_size == old.st_size and path.stat().st_mtime_ns == old.st_mtime_ns
    assert app.read_book(str(path))[0].strip() == "Второй текст книги."


# ---------------------------------------------------------------- ручная правка глав

def edit_book(app, name="Рассказ.txt"):
    path = app.home / name
    body = "\n\n".join([PROSE] * 12)
    path.write_text("Глава 1. Начало\n\nПервый абзац.\nВторой абзац.\nТретий абзац.\n\n" + body + "\n\n"
                    "Глава 2. Середина\n\n" + body + "\n\nГлава 3. Конец\n\n" + body + "\n", encoding="utf-8")
    return path


def test_a_chapter_is_split_merged_renamed_and_the_automatic_split_comes_back(app):
    ctx = app.AppContext()
    book = app.Book(ctx, edit_book(app), "Рассказ")
    assert [ch["title"] for ch in book.chapters] == ["Начало", "Середина", "Конец"]
    assert book.editable() and book.profile_key() == "auto"
    line, preview = book.chapter_lines(1)[1]                                       # «Второй абзац.»
    assert preview == "Второй абзац."
    book.split_chapter(line, "Вставка")
    assert [ch["title"] for ch in book.chapters] == ["Начало", "Вставка", "Середина", "Конец"]
    assert book.chapters[1]["text"].startswith("Второй абзац.") and book.profile_key() == "manual"
    assert [ch["id"] for ch in book.chapters] == [1, 2, 3, 4] and book.chapters[1]["name"].startswith("02_")
    book.merge_with_next(3)                                                         # «Середина» + «Конец»
    assert [ch["title"] for ch in book.chapters] == ["Начало", "Вставка", "Середина"]
    assert "Глава 3. Конец" in book.chapters[2]["text"]
    book.rename_chapter(1, "Пролог")
    book.save()
    again = app.Book(ctx, book.path, "Рассказ")                                     # правки сохранились в split.json
    assert [ch["title"] for ch in again.chapters] == ["Пролог", "Вставка", "Середина"]
    again.reset_edits()
    assert [ch["title"] for ch in again.chapters] == ["Начало", "Середина", "Конец"]


def test_renaming_works_without_manual_cuts_and_edits_are_dropped_when_the_book_file_changes(app):
    ctx = app.AppContext()
    path = edit_book(app)
    book = app.Book(ctx, path, "Рассказ")
    book.rename_chapter(2, "Дорога")
    book.save()
    assert app.Book(ctx, path, "Рассказ").chapters[1]["title"] == "Дорога"
    assert app.Book(ctx, path, "Рассказ").profile_key() == "auto"                   # только название — разбиение прежнее
    path.write_text(path.read_text(encoding="utf-8") + "\nНовый абзац в конце.\n", encoding="utf-8")
    changed = app.Book(ctx, path, "Рассказ")
    assert changed.chapters[1]["title"] == "Середина" and getattr(changed, "edits_reset", False)


def test_equal_parts_cannot_be_edited_by_hand(app):
    ctx = app.AppContext()
    path = app.home / "Сплошной.txt"
    path.write_text(" ".join(["Слово."] * 40000), encoding="utf-8")
    book = app.Book(ctx, path, "Сплошной")
    book.state.profile = "size"
    book.resplit()
    assert len(book.chapters) > 1 and not book.editable()


def test_the_window_edits_chapters_and_refuses_to_cut_a_started_book(app, window):
    ctx, view, presenter = window
    presenter.open_file(edit_book(app))
    view.split_answer = lambda choices: (choices[0][0], "")                          # с «Первого абзаца», название само
    presenter.split_chapter({1})
    assert [ch["title"] for ch in presenter.book.chapters][:2] == ["Начало", "Первый абзац."]
    assert view.profile == "manual"
    presenter.merge_chapter({4})
    assert any("последняя глава" in text for text in view.infos)
    presenter.merge_chapter({1})
    assert len(presenter.book.chapters) == 3
    view.texts = ["Пролог"]
    presenter.rename_chapter({1})
    assert view.initial_text == "Начало" and presenter.book.chapters[0]["title"] == "Пролог"
    folder = app.work_dir(presenter.book.dir, "qwen", presenter.book.chapters[1])
    folder.mkdir(parents=True)                                                         # книга начата
    presenter.split_chapter({2})
    presenter.reset_split()
    assert sum("пока книга не начата" in text for text in view.infos) == 2 and len(presenter.book.chapters) == 3
    presenter.set_profile("manual")
    assert view.profile == "manual"


@needs_ffmpeg
def test_renaming_a_finished_chapter_renames_its_mp3_without_redoing_it(app, window):
    ctx, view, presenter = window
    presenter.open_file(edit_book(app))
    presenter.set_engine("dummy")
    presenter.start()
    presenter.pump()
    book = presenter.book
    old = app.out_mp3(book.dir, book.chapters[0])
    assert old.exists()
    view.texts = ["Пролог"]
    presenter.rename_chapter({1})
    presenter.pump()
    new = app.out_mp3(book.dir, book.chapters[0])
    assert new != old and new.exists() and not old.exists() and "Пролог".encode("utf-16-le") in new.read_bytes()[:20000]
    assert [row.kind for row in presenter.service.chapter_rows(book, "dummy")] == ["done", "done", "done"]


def test_broken_manual_edits_in_split_json_are_dropped_and_the_book_still_opens(app):
    ctx = app.AppContext()
    path = edit_book(app)
    book = app.Book(ctx, path, "Рассказ")
    book.state.cuts, book.state.titles = ["x", -3, 10 ** 9, True], {"5": 7, "0": "  "}
    book.state.cuts_digest = app.text_digest(book.text)
    book.resplit()
    assert book.state.cuts == [] and book.state.titles == {} and len(book.chapters) == 3



# ---------------------------------------------------------------- окна: ввод текста в стиле приложения, рамка, быстрый отказ дубликату

def real_window(app, tk):
    pytest.importorskip("tkinterdnd2")
    pytest.importorskip("sv_ttk")
    service = app.AudiobookService(app.AppContext(), app.EventBus())
    try:
        return app.MainWindow(lambda view: app.MainPresenter(service, view))
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")


def test_text_prompt_is_themed_wide_enough_and_returns_the_text(app):
    tk = pytest.importorskip("tkinter")
    window = real_window(app, tk)
    try:
        window.root.update()
        seen = {}

        def dialog_widgets():
            top = [w for w in window.root.winfo_children() if isinstance(w, tk.Toplevel)][-1]
            top.update_idletasks()
            widgets = []

            def walk(widget):
                for child in widget.winfo_children():
                    widgets.append(child)
                    walk(child)
            walk(top)
            return top, widgets

        def type_and_accept():
            top, widgets = dialog_widgets()
            seen["classes"] = {w.winfo_class() for w in widgets}
            seen["width"] = top.winfo_reqwidth()
            entry = next(w for w in widgets if w.winfo_class() == "TEntry")
            seen["initial"], seen["show"] = entry.get(), str(entry.cget("show"))
            entry.delete(0, "end")
            entry.insert(0, "Новое название")
            next(w for w in widgets if w.winfo_class() == "TButton" and w.cget("text") == "OK").invoke()

        def cancel():
            _, widgets = dialog_widgets()
            seen["secret"] = str(next(w for w in widgets if w.winfo_class() == "TEntry").cget("show"))
            next(w for w in widgets if w.winfo_class() == "TButton" and w.cget("text") == "Отмена").invoke()

        def failsafe():                                                    # если диалог завис, тест не должен висеть вместе с ним
            for w in window.root.winfo_children():
                if isinstance(w, tk.Toplevel):
                    w.destroy()
        window.root.after(300, type_and_accept)
        window.root.after(6000, failsafe)
        title = "Переименовать главу и ещё немного длинного текста в заголовке"
        assert window.ask_text(title, "Название главы:", initial="Старое") == "Новое название"
        assert seen["classes"] >= {"TEntry", "TButton", "TLabel"} and not ({"Entry", "Button"} & seen["classes"])   # без системных виджетов
        assert seen["initial"] == "Старое" and seen["show"] == "" and seen["width"] >= 460
        window.root.after(300, cancel)
        assert window.ask_text("Ключ", "Введите ключ:", secret=True) is None                # «Отмена»
        assert seen["secret"] == "•"
    finally:
        window.root.destroy()


@pytest.mark.skipif(os.name != "nt", reason="рамка окна управляется только в Windows")
def test_the_window_has_a_normal_border_so_windows_snap_can_maximize_it(app):
    import ctypes
    tk = pytest.importorskip("tkinter")
    window = real_window(app, tk)
    try:
        window.root.update()
        hwnd = ctypes.windll.user32.GetParent(window.root.winfo_id())
        style = ctypes.windll.user32.GetWindowLongW(hwnd, -16)
        assert style & 0x00040000 != 0 and style & 0x00010000 != 0           # рамка-растяжка (нужна для Snap) и «развернуть»
    finally:
        window.root.destroy()


def test_a_duplicate_is_refused_at_once_and_only_a_restart_after_an_update_waits(app, monkeypatch):
    waits = []
    monkeypatch.setattr(app, "configure_environment", lambda: None)
    monkeypatch.setattr(app, "bootstrap", lambda: None)
    monkeypatch.setattr(app, "remove_pip_leftovers", lambda: None)
    monkeypatch.setattr(app, "ensure_base_packages", lambda read_only=False: None)
    monkeypatch.setattr(app, "APP_LOCK", None)
    monkeypatch.setattr(app, "acquire_app_lock", lambda wait=5.0: waits.append(wait) or app.acquire_lock(app.PATHS.cache / "app.lock", 0))
    for argv in (["audiobook_master.py"], ["audiobook_master.py", "--after-update"]):
        monkeypatch.setattr(sys, "argv", argv)
        app.startup()
        app.APP_LOCK.release()
    assert waits == [0.0, 5.0]
    sent = []
    monkeypatch.setattr(app, "exe_path", lambda: r"C:\Книги\FoxyAudioBookMaker.exe")
    monkeypatch.setattr(app.subprocess, "Popen", lambda command, **kw: sent.append(command))
    app.restart_exe()
    assert sent == [[r"C:\Книги\FoxyAudioBookMaker.exe", "--after-update"]]                      # новый exe знает, что это перезапуск


def test_without_a_console_library_output_goes_nowhere_instead_of_crashing(app, monkeypatch):
    monkeypatch.setattr(app, "WINDOWLESS", False)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    app.ensure_output_streams()
    sys.stderr.write("Downloading model.safetensors: 42%\r")                       # как полоса загрузки модели
    print("строка журнала")
    assert app.windowless() is True


def test_the_log_text_can_be_selected_and_copied(app):
    pytest.importorskip("tkinterdnd2")
    tk = pytest.importorskip("tkinter")
    ctx = app.AppContext()
    try:
        window = app.MainWindow(lambda view: app.MainPresenter(app.AudiobookService(ctx, app.EventBus()), view), theme="light")
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")
    try:
        window.log_text.configure(state="normal")
        window.log_text.insert("end", "первая строка\nвторая строка")
        window.log_text.configure(state="disabled")
        window._copy_log()
        assert window.root.clipboard_get() == "первая строка\nвторая строка"          # ничего не выделено — весь журнал
        window.log_text.tag_add("sel", "2.0", "2.6")
        event = types.SimpleNamespace(keysym="Cyrillic_es", keycode=67 if os.name == "nt" else 0, state=4)
        if os.name != "nt":
            event.keysym = "c"
        assert window._log_keys(event) == "break" and window.root.clipboard_get() == "вторая"
    finally:
        window.root.destroy()


def test_stop_during_a_preview_drops_its_result_and_stops_the_sound(app, window, monkeypatch):
    ctx, view, presenter = window
    presenter.set_engine("dummy")
    jobs = []
    presenter.runner = jobs.append
    monkeypatch.setattr(app, "make_voice_preview", lambda *a, **k: (app.home / "p.wav", None))
    presenter.preview()
    assert presenter.previewing and view.preview_busy is True
    presenter.preview()                                                            # «Стоп», пока проба озвучивается
    assert not presenter.previewing and view.preview_busy is False and view.sound_stopped == 1
    jobs[0]()
    presenter.pump()
    assert getattr(view, "played", []) == []                                       # остановленную пробу не играем
    view.playing_now = True
    presenter.preview()                                                            # «Стоп», пока проба звучит
    assert view.sound_stopped == 2 and len(jobs) == 1


def test_sound_button_shows_stop_while_playing_and_only_one_sound_plays_at_a_time(app, monkeypatch):
    played, stopped = [], []
    monkeypatch.setattr(app, "play_sound", played.append)
    monkeypatch.setattr(app, "stop_sound", lambda: stopped.append(True))
    monkeypatch.setattr(app, "sound_seconds", lambda path: 1.0)

    class Button:
        def __init__(self):
            self.text, self.timers = "", []

        def configure(self, text):
            self.text = text

        def after(self, ms, callback):
            self.timers.append(callback)
    first, second = Button(), Button()
    a, b = app.SoundButton(first, "Прослушать"), app.SoundButton(second, "Прослушать")
    a.play("a.wav")
    assert a.busy and first.text == "■ Стоп"
    b.play("b.wav")                                                                # новая запись останавливает прежнюю
    assert not a.busy and first.text == "Прослушать" and b.busy and stopped == [True]
    first.timers[0]()                                                              # старый таймер ничего не ломает
    assert b.busy and second.text == "■ Стоп"
    second.timers[0]()                                                             # запись доиграла
    assert not b.busy and second.text == "Прослушать" and app.SoundButton.current is None


# ---------------------------------------------------------------- Qwen: «Авто», загрузка модели, выгрузка после проб

def test_auto_book_language_lets_qwen_detect_the_language_itself(app, window):
    ctx, view, presenter = window
    presenter.open_file(write_book(app, "Рассказ.txt"))
    assert ctx.settings.qwen_language_auto is True and app.qwen_model_language(ctx.settings) == "Auto"
    assert ctx.settings.qwen_language == "Russian"                                 # для правил чтения — определённый язык
    presenter.choose_language(app.QWEN_LANGUAGES.index("German") + 1)
    assert app.qwen_model_language(ctx.settings) == "German"
    presenter.choose_language(0)
    assert app.qwen_model_language(ctx.settings) == "Auto"


def test_a_card_with_too_little_video_memory_gets_a_clear_message(app, monkeypatch):
    ctx, bus, log, state = fake_qwen(app, monkeypatch, capability=(6, 1), vram_gb=2)
    with pytest.raises(RuntimeError, match="не поместится"):
        app.QwenEngine(ctx, bus)
    assert state["loads"] == []                                                   # даже не начинали грузить


def test_loading_the_model_is_visible_in_the_log_and_a_failure_is_explained(app, monkeypatch):
    ctx, bus, log, state = fake_qwen(app, monkeypatch, capability=(8, 6))
    monkeypatch.setattr(app, "QWEN_LOAD_NOTICE_SECONDS", 0.01)
    real = sys.modules["faster_qwen3_tts"].FasterQwen3TTS.from_pretrained
    monkeypatch.setattr(sys.modules["faster_qwen3_tts"].FasterQwen3TTS, "from_pretrained",
                        classmethod(lambda cls, *a, **k: (time.sleep(0.05), real(*a, **k))[1]))
    app.QwenEngine(ctx, bus)
    assert any("Скачиваю модель Qwen" in line for line in log) and any("ещё загружаю модель" in line for line in log)
    assert any("Модель загружена за" in line for line in log)

    def out_of_memory(cls, *a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
    monkeypatch.setattr(sys.modules["faster_qwen3_tts"].FasterQwen3TTS, "from_pretrained", classmethod(out_of_memory))
    with pytest.raises(RuntimeError, match="Не хватило видеопамяти"):
        app.QwenEngine(ctx, bus)


def test_the_preview_model_is_unloaded_after_thirty_idle_seconds(app, window, monkeypatch):
    ctx, view, presenter = window
    closed = []
    presenter.preview_engine = types.SimpleNamespace(close=lambda: closed.append(True))
    clock = [1000.0]
    monkeypatch.setattr(app.time, "monotonic", lambda: clock[0])
    presenter.preview_idle_since = 1000.0
    clock[0] = 1029.0
    presenter.pump()
    assert closed == [] and presenter.preview_engine is not None                  # 29 с — ещё держим
    view.playing_now = True
    clock[0] = 1100.0
    presenter.pump()
    assert closed == []                                                            # пока звучит проба — не выгружаем
    view.playing_now = False
    presenter.pump()
    assert closed == [True] and presenter.preview_engine is None
    assert any("выгружена из видеопамяти" in text for text in view.infos)


def test_triton_is_tried_again_after_an_update_and_its_state_is_logged(app, monkeypatch):
    ctx, bus, log, state = fake_qwen(app, monkeypatch, capability=(7, 5))
    ctx.update(qwen_no_triton_gpu="NVIDIA GeForce RTX 2070 Max-Q")                  # отметка старой версии
    engine = app.QwenEngine(ctx, bus)
    assert engine.triton is True and any("Ускорение Triton: включено" in line for line in log)
    ctx.update(qwen_no_triton_gpu=f"NVIDIA GeForce RTX 2070 Max-Q|{app.APP_VERSION}")
    log.clear()
    assert app.QwenEngine(ctx, bus).triton is False and any("После обновления программы" in line for line in log)


def test_warm_up_runs_before_the_book_and_its_time_appears_in_the_first_chapter_summary(app, monkeypatch):
    ctx, bus, log, state = fake_qwen(app, monkeypatch, capability=(8, 6))
    engine = app.QwenEngine(ctx, bus)
    spoken = []
    real = engine.synth
    monkeypatch.setattr(engine, "synth", lambda text: (spoken.append(text), real(text))[1])
    engine.warm_up()
    assert spoken == ["Проверка голоса."] and any("Прогрев модели" in line for line in log)
    assert "загрузка модели" in engine.startup_note and "прогрев" in engine.startup_note


def test_helper_processes_use_python_exe_when_the_program_runs_in_pythonw(app, monkeypatch, tmp_path):
    (tmp_path / "pythonw.exe").write_text("")
    (tmp_path / "python.exe").write_text("")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "pythonw.exe"))
    assert app.console_python() == str(tmp_path / "python.exe")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    assert app.console_python() == str(tmp_path / "python.exe")


def test_a_cuda_check_without_an_answer_does_not_claim_that_cuda_is_missing(app, monkeypatch):
    monkeypatch.setattr(app, "pkg_version", lambda name: "2.11.0+cu128")
    for output, expected in (("", True), ("1\n", True), ("0\n", False), ("предупреждение\n1\n", True)):
        monkeypatch.setattr(app.subprocess, "run", lambda *a, output=output, **k: types.SimpleNamespace(stdout=output))
        assert app.torch_cuda_status() == ("2.11.0+cu128", expected)
    monkeypatch.setattr(app, "pkg_version", lambda name: "2.11.0+cpu")
    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=""))
    assert app.torch_cuda_status() == ("2.11.0+cpu", False)


def test_a_too_small_paging_file_gets_advice_instead_of_a_bare_system_error(app, monkeypatch):
    ctx, bus, log, state = fake_qwen(app, monkeypatch, capability=(8, 6))

    def no_memory(cls, *a, **k):
        raise OSError("The paging file is too small for this operation to complete. (os error 1455)")
    monkeypatch.setattr(sys.modules["faster_qwen3_tts"].FasterQwen3TTS, "from_pretrained", classmethod(no_memory))
    with pytest.raises(RuntimeError, match="файла подкачки"):
        app.QwenEngine(ctx, bus)


def test_memory_is_given_back_once_after_thirty_idle_seconds(app, window, monkeypatch):
    ctx, view, presenter = window
    released = []
    monkeypatch.setattr(app, "release_memory", lambda: released.append(True))
    clock = [1000.0]
    monkeypatch.setattr(app.time, "monotonic", lambda: clock[0])
    presenter.running = True
    presenter.pump()                                                                # идёт озвучка
    presenter.running = False
    clock[0] = 1029.0
    presenter.pump()
    assert released == []                                                           # 29 с — рано
    clock[0] = 1031.0
    presenter.pump()
    presenter.pump()
    assert released == [True]                                                       # один раз, не на каждом шаге
    presenter.previewing = True
    presenter.pump()
    presenter.previewing = False
    clock[0] = 1062.0
    presenter.pump()
    assert released == [True, True]                                                 # после новой работы — снова


def test_release_memory_runs_everywhere_without_errors(app):
    app.release_memory()


def test_the_palette_fix_applies_only_to_tk_9(app):
    tk = pytest.importorskip("tkinter")
    sv_ttk = pytest.importorskip("sv_ttk")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("нет экрана для окна Tk")
    try:
        sv_ttk.set_theme("light", root=root)
        major = int(str(root.tk.call("info", "patchlevel")).split(".")[0])
        assert app.tame_palette(root) is (major >= 9)                        # в Tk 8.6 (Python 3.12) всё остаётся как было
        body = str(root.tk.call("info", "body", "configure_colors"))
        assert ("tk_setPalette" not in body) is (major >= 9)
    finally:
        root.destroy()


@needs_ffmpeg
def test_stop_pressed_before_the_model_loads_ends_the_run_at_once(app):
    ctx = app.AppContext()
    service = app.AudiobookService(ctx, app.EventBus())
    kinds = []
    service.events.subscribe(lambda kind, data: kinds.append(kind))
    book = make_book(app, ctx, chapters=2)
    token = app.CancellationToken()
    token.cancel()                                                                 # «Стоп» уже нажат
    assert service.run(book, "dummy", {1, 2}, token) is False
    assert "run_finished" in kinds and "chapter_started" not in kinds              # на куски не переходили
    assert not list(book.dir.glob("*.mp3"))


SPEECH_SAMPLES = [
    "Пришли гости и т. д. и т.п., т. е. все, и др. и пр., т. к. так надо, т.н. эксперт, т. о. всё ясно.",
    "Напр. см. ул. Ленина, им. Пушкина, проф. Иванов и акад. Петров, гл. 5, в 1990-е гг. В г. Москве № 7.",
    "В 1984 году, к 2001 г., XIX в. и во 2-м ряду, 3,5 км и 12:30, версия 1.2.3, 1 000 000 рублей.",
    "Обычный текст без цифр и сокращений, только слова. А ещё XX век и в XVII в. жили люди.",
    "Mr. Smith and Mrs. Brown, Dr. Who vs. Ms. Jones, e.g. this, i.e. that, etc. No. 5 costs $3.50 or 20%.",
    "The 21st of May 1984 at 7:05, version 2.0.1, plain words only here.",
    "\n\n".join(["Глава 3. Начало, т. е. пролог."] + ["Абзац без чисел, но о XX веке и т. д. " * 8] * 6
                  + ["В 1812 году, к 2-му числу, 15 км и 10:45.", "Ещё абзац без цифр, см. выше."] * 3),
    "\n\n".join(["Chapter 2. Mr. Smith, e.g. a man."] + ["Plain paragraph without numbers. " * 10] * 6
                  + ["In 1984 he paid $3.50, 20% more, on the 21st.", "More words, etc."] * 3),
]


@pytest.mark.parametrize("text", SPEECH_SAMPLES)
def test_skipping_rules_that_cannot_match_does_not_change_the_result(app, monkeypatch, text):
    fast = app.normalize_for_speech(text)
    monkeypatch.setattr(app, "_literal_hint", lambda pattern: "")                  # все правила подряд, как раньше
    monkeypatch.setattr(app, "SPEECH_SPLIT_CHARS", 10 ** 9)                        # и весь текст целиком
    real_search = re.search
    monkeypatch.setattr(app.re, "search", lambda pattern, string, flags=0: True if pattern == r"\d" else real_search(pattern, string, flags))
    assert app.normalize_for_speech(text) == fast


def test_tkdnd_loading_is_retried_before_giving_up(app, monkeypatch):
    calls = []

    def flaky(tkroot):
        calls.append(tkroot)
        if len(calls) < 3:
            raise RuntimeError("Unable to load tkdnd library.")
        return "2.9"
    module = types.SimpleNamespace(_require=flaky)
    monkeypatch.setattr(app.time, "sleep", lambda seconds: None)
    app.patient_tkdnd(module)
    assert module._require("root") == "2.9" and len(calls) == 3                  # третья попытка удалась
    app.patient_tkdnd(module)
    assert module._require.patient is True                                        # повторно не оборачивается
    calls.clear()
    module2 = types.SimpleNamespace(_require=lambda tkroot: calls.append(1) or (_ for _ in ()).throw(RuntimeError("no tkdnd")))
    app.patient_tkdnd(module2)
    with pytest.raises(RuntimeError):
        module2._require("root")
    assert len(calls) == 3
