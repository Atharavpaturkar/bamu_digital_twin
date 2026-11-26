from types import SimpleNamespace

from fastapi.testclient import TestClient

import app as chat_app


def interaction(answer: str, file_names: list[str] | None = None) -> SimpleNamespace:
    annotations = [
        {"type": "file_citation", "file_name": file_name}
        for file_name in (file_names or [])
    ]
    return SimpleNamespace(
        output_text=answer,
        steps=[
            SimpleNamespace(
                type="model_output",
                content=[SimpleNamespace(annotations=annotations)],
            )
        ],
    )


class FakeInteractions:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return self.result


def install_fake_client(monkeypatch, result: SimpleNamespace) -> FakeInteractions:
    interactions = FakeInteractions(result)
    monkeypatch.setattr(chat_app, "_read_store_name", lambda: "fileSearchStores/test")
    monkeypatch.setattr(
        chat_app,
        "_get_gemini_client",
        lambda: SimpleNamespace(interactions=interactions),
    )
    return interactions


def test_chat_uses_file_search_and_returns_sources(monkeypatch) -> None:
    client = install_fake_client(
        monkeypatch,
        interaction("The program covers anatomy and physiology.", ["MU 2022.pdf", "MU 2022.pdf"]),
    )
    response = TestClient(chat_app.app).post(
        "/chat",
        json={
            "message": "What subjects are covered?",
            "history": [{"role": "user", "content": "I am reviewing my notes."}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The program covers anatomy and physiology.",
        "sources": ["MU 2022.pdf"],
    }
    assert client.kwargs["tools"] == [
        {"type": "file_search", "file_search_store_names": ["fileSearchStores/test"]}
    ]
    assert client.kwargs["store"] is False
    assert "I am reviewing my notes." in str(client.kwargs["input"])
    assert client.kwargs["system_instruction"] == chat_app.SYSTEM_INSTRUCTION


def test_uncited_response_returns_exact_fallback(monkeypatch) -> None:
    install_fake_client(monkeypatch, interaction("A plausible but uncited answer."))
    response = TestClient(chat_app.app).post("/chat", json={"message": "Unrelated question"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "I don't have that information.",
        "sources": [],
    }


def test_rejects_overlong_message() -> None:
    response = TestClient(chat_app.app).post("/chat", json={"message": "x" * 2001})

    assert response.status_code == 422


def test_chat_without_store_returns_setup_message(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FILE_SEARCH_STORE_NAME", raising=False)
    monkeypatch.setattr(chat_app, "STORE_NAME_FILE", tmp_path / "missing-store.txt")
    response = TestClient(chat_app.app).post("/chat", json={"message": "Question"})

    assert response.status_code == 503
    assert "Run `python ingest.py` first" in response.json()["detail"]


def test_chat_without_key_returns_setup_message(monkeypatch) -> None:
    monkeypatch.setenv("FILE_SEARCH_STORE_NAME", "fileSearchStores/test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = TestClient(chat_app.app).post("/chat", json={"message": "Question"})

    assert response.status_code == 503
    assert "missing GEMINI_API_KEY" in response.json()["detail"]


def test_health_endpoint_reports_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("FILE_SEARCH_STORE_NAME", raising=False)
    response = TestClient(chat_app.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "configured": False}


def test_health_endpoint_reports_configured(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "configured-for-test")
    monkeypatch.setenv("FILE_SEARCH_STORE_NAME", "fileSearchStores/test")
    response = TestClient(chat_app.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "configured": True}


def test_widget_script_is_served() -> None:
    response = TestClient(chat_app.app).get("/widget.js")

    assert response.status_code == 200
    assert "document-chat-widget-button" in response.text
    assert 'frame.allow = "microphone"' in response.text


def test_index_includes_mic_button() -> None:
    response = TestClient(chat_app.app).get("/")

    assert response.status_code == 200
    assert 'id="mic-button"' in response.text
    assert 'fetch("/transcribe"' in response.text


class FakeModels:
    def __init__(self, text: str) -> None:
        self.text = text
        self.kwargs: dict[str, object] = {}

    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(text=self.text)


def test_transcribe_returns_text(monkeypatch) -> None:
    import base64

    models = FakeModels("What are the key topics in the diagnostic report?")
    monkeypatch.setattr(
        chat_app,
        "_get_gemini_client",
        lambda: SimpleNamespace(models=models),
    )
    audio_b64 = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEfmt ").decode("ascii")
    response = TestClient(chat_app.app).post(
        "/transcribe",
        json={"audio_base64": audio_b64, "mime_type": "audio/webm;codecs=opus"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "What are the key topics in the diagnostic report?"}
    assert models.kwargs["model"] == chat_app.MODEL_NAME
    contents = models.kwargs["contents"]
    assert len(contents) == 2
    assert contents[0].inline_data.mime_type == "audio/webm"


def test_transcribe_rejects_invalid_mime_type(monkeypatch) -> None:
    import base64

    audio_b64 = base64.b64encode(b"hello").decode("ascii")
    response = TestClient(chat_app.app).post(
        "/transcribe",
        json={"audio_base64": audio_b64, "mime_type": "text/plain"},
    )

    assert response.status_code == 422


def test_transcribe_rejects_invalid_base64(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "configured-for-test")
    response = TestClient(chat_app.app).post(
        "/transcribe",
        json={"audio_base64": "not-valid-base64!!!", "mime_type": "audio/webm"},
    )

    assert response.status_code == 422


def test_transcribe_without_key_returns_setup_message(monkeypatch) -> None:
    import base64

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    audio_b64 = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVE").decode("ascii")
    response = TestClient(chat_app.app).post(
        "/transcribe",
        json={"audio_base64": audio_b64, "mime_type": "audio/wav"},
    )

    assert response.status_code == 503
    assert "missing GEMINI_API_KEY" in response.json()["detail"]