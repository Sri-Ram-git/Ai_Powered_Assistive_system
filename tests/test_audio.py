"""Unit tests for the speech module (no audio output in tests)."""

import pytest

import src.audio.tts as tts_module
from src.audio.tts import SpeechOutput
from src.utils.exceptions import SpeechError


class _FakeEngine:
    def __init__(self):
        self.properties = {}
        self.said = []
        self.ran = 0
        self.stopped = False
        self.loop_started = False
        self.loop_ended = False

    def setProperty(self, name, value):
        self.properties[name] = value

    def getProperty(self, name):
        return self.properties.get(name)

    def say(self, text):
        self.said.append(text)

    def runAndWait(self):
        self.ran += 1

    def startLoop(self, use_driver_loop=True):
        self.loop_started = True

    def endLoop(self):
        self.loop_ended = True

    def iterate(self):
        pass

    def isBusy(self):
        return False

    def stop(self):
        self.stopped = True


@pytest.fixture
def fake_engine(monkeypatch):
    engine = _FakeEngine()

    class _FakePytTTSX3:
        def init(self, *a, **k):
            return engine

    fake = _FakePytTTSX3()
    monkeypatch.setattr(tts_module, "pyttsx3", fake)
    return engine


class TestSpeechOutput:
    def test_init_sets_properties(self, fake_engine):
        tts = SpeechOutput(rate=150, volume=0.5)
        assert fake_engine.properties["rate"] == 150
        assert fake_engine.properties["volume"] == 0.5
        tts.shutdown()

    def test_speak_enqueues_and_worker_speaks(self, fake_engine):
        tts = SpeechOutput()
        tts.speak("hello")
        tts.shutdown()
        assert "hello" in fake_engine.said

    def test_say_now_speaks_synchronously(self, fake_engine):
        tts = SpeechOutput()
        tts.say_now("sync message")
        assert "sync message" in fake_engine.said
        tts.shutdown()

    def test_empty_speak_ignored(self, fake_engine):
        tts = SpeechOutput()
        tts.speak("")
        tts.speak("   ")
        tts.shutdown()
        assert fake_engine.said == []

    def test_set_rate(self, fake_engine):
        tts = SpeechOutput()
        tts.set_rate(200)
        assert fake_engine.properties["rate"] == 200
        tts.shutdown()

    def test_shutdown_stops_engine(self, fake_engine):
        tts = SpeechOutput()
        tts.shutdown()
        assert fake_engine.loop_ended


class TestErrors:
    def test_init_failure_raises(self, monkeypatch):
        class _Broken:
            def init(self, *a, **k):
                raise RuntimeError("no speech engine")

        monkeypatch.setattr(tts_module, "pyttsx3", _Broken())
        with pytest.raises(SpeechError):
            SpeechOutput()
