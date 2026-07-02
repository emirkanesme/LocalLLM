"""Foundry Local singleton — model lifecycle and chat completions."""

import threading

from config import FOUNDRY_CHAT_MODEL, FOUNDRY_MAX_TOKENS, FOUNDRY_TEMPERATURE

_lock = threading.Lock()
_manager = None
_model = None
_init_error = None


def _get_model():
    return _model


def is_ready():
    model = _model
    return model is not None and model.is_loaded


def get_init_error():
    return _init_error


def get_model_name():
    return FOUNDRY_CHAT_MODEL


def initialize(on_status=None):
    """
    Initialize Foundry Local and load the chat model.
    on_status(phase, percent, message) — optional progress callback.
      phase: "init" | "download" | "load"
    """
    global _manager, _model, _init_error

    with _lock:
        if is_ready():
            return True

        try:
            from foundry_local_sdk import Configuration, FoundryLocalManager

            if on_status:
                on_status("init", 0, "Foundry Local başlatılıyor...")

            config = Configuration(app_name="local_rag", log_level="error")
            FoundryLocalManager.initialize(config)
            _manager = FoundryLocalManager.instance

            _model = _manager.catalog.get_model(FOUNDRY_CHAT_MODEL)

            if not _model.is_cached:
                if on_status:
                    on_status(
                        "download",
                        0,
                        f"Model indiriliyor: {FOUNDRY_CHAT_MODEL} (ilk seferde birkaç dakika sürebilir)",
                    )

                def download_progress(percent):
                    if on_status:
                        on_status("download", percent, None)

                _model.download(progress_callback=download_progress)

            if not _model.is_loaded:
                if on_status:
                    on_status("load", 0, "Model belleğe yükleniyor...")
                _model.load()

            _init_error = None
            if on_status:
                on_status("ready", 100, f"Model hazır: {FOUNDRY_CHAT_MODEL}")
            return True

        except Exception as exc:
            _init_error = str(exc)
            if on_status:
                on_status("error", 0, str(exc))
            return False


def complete_chat(system_prompt, user_message):
    """Send a chat completion request to the loaded Foundry model."""
    if not is_ready():
        if not initialize():
            raise RuntimeError(
                _init_error or "Foundry Local modeli yüklenemedi."
            )

    client = _model.get_chat_client()
    client.settings.temperature = FOUNDRY_TEMPERATURE
    client.settings.max_tokens = FOUNDRY_MAX_TOKENS

    # Küçük modeller tek user mesajında daha iyi takip eder
    combined = f"{system_prompt}\n\n{user_message}"
    response = client.complete_chat([
        {"role": "user", "content": combined},
    ])
    content = response.choices[0].message.content
    return content.strip() if content else ""


def shutdown():
    """Unload model from memory."""
    global _model
    with _lock:
        if _model is not None and _model.is_loaded:
            _model.unload()
