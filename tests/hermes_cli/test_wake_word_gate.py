"""Tests for the wake-word feature gate and mic-contention state machine
added to ``hermes_cli.voice`` for the "Hey Jarvis" wake-word feature.

Covers:
  * ``wake_word_config`` -- the shape-safe config gate.
  * ``start_wake_word`` / ``rearm_wake_word_after_turn`` no-op when the
    feature is unconfigured.
  * ``_set_voice_state`` / the IDLE<->ACTIVE mic-ownership guard, including
    duplicate-detect no-ops and a concurrent-thread race.

The real ``WakeWordListener`` (which needs openwakeword/sounddevice) is
stubbed out here -- ``test_wakeword_listener.py`` covers its own detection
logic in isolation.
"""

import threading

import pytest

import hermes_cli.voice as voice


@pytest.fixture(autouse=True)
def _reset_wake_state(monkeypatch):
    """``_voice_state`` and ``_wake_listener`` are module-level globals.
    Cross-*file* isolation is guaranteed by the test runner (fresh
    interpreter per file), but tests within this file share state -- reset
    before every test regardless of ordering."""
    monkeypatch.setattr(voice, "_voice_state", "IDLE")
    monkeypatch.setattr(voice, "_wake_listener", None)
    monkeypatch.setattr(voice, "_play_beep", lambda *_a, **_k: None)


class TestWakeWordConfigGate:
    """``wake_word_config`` -- unset/malformed shapes must all resolve to
    the off state ``(None, 0.5)``; only a configured, non-blank string
    wake word turns the feature on."""

    def test_empty_config_returns_none(self):
        assert voice.wake_word_config({}) == (None, 0.5)

    def test_non_dict_voice_section_returns_none(self):
        for bad_voice in (None, True, "jarvis", 1, ["jarvis"]):
            assert voice.wake_word_config({"voice": bad_voice}) == (None, 0.5), bad_voice

    def test_missing_wake_word_key_returns_none(self):
        assert voice.wake_word_config({"voice": {"beep_enabled": True}}) == (None, 0.5)

    def test_non_string_or_blank_wake_word_returns_none(self):
        for bad_word in (None, 1, True, "   ", [], {}):
            assert voice.wake_word_config({"voice": {"wake_word": bad_word}}) == (None, 0.5), bad_word

    def test_configured_wake_word_uses_default_threshold(self):
        assert voice.wake_word_config({"voice": {"wake_word": "hey_jarvis"}}) == ("hey_jarvis", 0.5)

    def test_configured_wake_word_and_custom_threshold(self):
        cfg = {"voice": {"wake_word": "hey_jarvis", "wake_threshold": 0.7}}
        assert voice.wake_word_config(cfg) == ("hey_jarvis", 0.7)

    def test_wake_word_is_stripped(self):
        assert voice.wake_word_config({"voice": {"wake_word": "  hey_jarvis  "}}) == ("hey_jarvis", 0.5)

    def test_malformed_threshold_falls_back_to_default(self):
        cfg = {"voice": {"wake_word": "hey_jarvis", "wake_threshold": "loud"}}
        assert voice.wake_word_config(cfg) == ("hey_jarvis", 0.5)

    def test_bool_threshold_rejected(self):
        """``bool`` is an ``int`` subclass in Python -- must not sneak
        through the ``isinstance(threshold, (int, float))`` check."""
        cfg = {"voice": {"wake_word": "hey_jarvis", "wake_threshold": True}}
        assert voice.wake_word_config(cfg) == ("hey_jarvis", 0.5)


class TestGateOffEntryPoints:
    """When ``voice.wake_word`` is unset, every entry point must no-op
    rather than touch the mic."""

    def test_start_wake_word_returns_false_when_unset(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})

        assert voice.start_wake_word(lambda: None) is False
        assert voice._wake_listener is None

    def test_start_wake_word_returns_false_on_config_load_failure(self, monkeypatch):
        def _boom():
            raise RuntimeError("disk on fire")

        monkeypatch.setattr("hermes_cli.config.load_config", _boom)

        assert voice.start_wake_word(lambda: None) is False

    def test_rearm_is_noop_from_the_idle_starting_state(self, monkeypatch):
        """Starting state is IDLE (see fixture) -- rearm is a true no-op:
        _set_voice_state("IDLE") itself already refuses the redundant
        transition, so start_wake_word is never even attempted."""
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})

        assert voice.rearm_wake_word_after_turn(lambda: None) is False

    def test_rearm_from_active_transitions_to_idle_but_stays_disarmed(self, monkeypatch):
        """Force the ACTIVE->IDLE half of rearm to actually execute
        (simulating the real post-turn call site), then verify the config
        gate still prevents the listener from arming."""
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
        monkeypatch.setattr(voice, "_voice_state", "ACTIVE")

        result = voice.rearm_wake_word_after_turn(lambda: None)

        assert result is False
        assert voice._voice_state == "IDLE"
        assert voice._wake_listener is None


class _FakeWakeListener:
    """Stand-in for ``hermes_cli.wakeword.WakeWordListener``."""

    instances = []

    def __init__(self, model_path, threshold, on_detect):
        self.model_path = model_path
        self.threshold = threshold
        self.on_detect = on_detect
        self._running = False
        _FakeWakeListener.instances.append(self)

    def start(self):
        self._running = True

    def stop(self, timeout=3.0):
        self._running = False

    def is_running(self):
        return self._running


@pytest.fixture
def armed_gate(monkeypatch):
    """Turn the wake-word gate on and stub out the real listener class +
    model-path lookup so ``start_wake_word`` arms a ``_FakeWakeListener``
    without touching openwakeword/sounddevice."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"voice": {"wake_word": "hey_jarvis"}},
    )
    monkeypatch.setattr(voice, "_resolve_wake_model_path", lambda _w: "/fake/hey_jarvis_v0.1.onnx")
    _FakeWakeListener.instances = []
    monkeypatch.setattr("hermes_cli.wakeword.WakeWordListener", _FakeWakeListener)
    return _FakeWakeListener


class TestMicContentionStateMachine:
    def test_set_voice_state_idle_to_active(self):
        assert voice._voice_state == "IDLE"
        assert voice._set_voice_state("ACTIVE") is True
        assert voice._voice_state == "ACTIVE"

    def test_set_voice_state_active_to_idle(self, monkeypatch):
        monkeypatch.setattr(voice, "_voice_state", "ACTIVE")

        assert voice._set_voice_state("IDLE") is True
        assert voice._voice_state == "IDLE"

    def test_set_voice_state_same_state_is_noop(self):
        assert voice._voice_state == "IDLE"

        assert voice._set_voice_state("IDLE") is False
        assert voice._voice_state == "IDLE"

    def test_detect_transitions_idle_to_active_and_releases_wake_listener(self, armed_gate):
        on_wake_calls = []
        assert voice.start_wake_word(lambda: on_wake_calls.append(1)) is True
        listener = armed_gate.instances[0]
        assert listener.is_running() is True
        assert voice._voice_state == "IDLE"

        # Simulate the hardware firing: invoke the internal on_detect
        # closure exactly as WakeWordListener._run would.
        listener.on_detect()

        assert voice._voice_state == "ACTIVE"
        assert on_wake_calls == [1]
        # Detection must release the wake listener -- it can't still own
        # the mic while a capture turn is starting.
        assert listener.is_running() is False
        assert voice._wake_listener is None

    def test_duplicate_detect_while_active_is_noop(self, armed_gate):
        on_wake_calls = []
        voice.start_wake_word(lambda: on_wake_calls.append(1))
        listener = armed_gate.instances[0]

        listener.on_detect()
        assert on_wake_calls == [1]

        # Stray/duplicate detection firing again while already ACTIVE.
        listener.on_detect()

        assert on_wake_calls == [1]                # no second capture started
        assert len(armed_gate.instances) == 1       # no second listener spawned
        assert voice._voice_state == "ACTIVE"

    def test_rearm_after_turn_returns_to_idle_and_rearms_a_fresh_listener(self, armed_gate):
        on_wake_calls = []
        voice.start_wake_word(lambda: on_wake_calls.append(1))
        listener = armed_gate.instances[0]
        listener.on_detect()
        assert voice._voice_state == "ACTIVE"

        result = voice.rearm_wake_word_after_turn(lambda: on_wake_calls.append(1))

        assert result is True
        assert voice._voice_state == "IDLE"
        assert len(armed_gate.instances) == 2  # a fresh listener re-armed
        assert armed_gate.instances[1].is_running() is True

    def test_lock_guards_concurrent_transition_race(self):
        """Many threads racing to flip IDLE->ACTIVE simultaneously: exactly
        one must win the transition, proving ``_voice_state_lock`` (not
        just the equality check) is what prevents two mic owners."""
        n_threads = 32
        results = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(n_threads)

        def _race():
            barrier.wait()
            won = voice._set_voice_state("ACTIVE")
            with results_lock:
                results.append(won)

        threads = [threading.Thread(target=_race) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert results.count(True) == 1
        assert results.count(False) == n_threads - 1
        assert voice._voice_state == "ACTIVE"
