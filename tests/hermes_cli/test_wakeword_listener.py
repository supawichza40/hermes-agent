"""Tests for ``hermes_cli.wakeword.WakeWordListener`` -- the "Hey Jarvis"
wake-word detection loop.

Uses the real bundled openWakeWord ONNX model (loading it is cheap and
network-free), but drives ``_run()`` synchronously with a monkeypatched
``predict()`` and a fake ``sounddevice`` module so the test needs no real
microphone and no timing-dependent background thread. ``sounddevice`` is
not installed in this environment (matching CI/boxes without an audio
backend), so the fake is injected directly into ``sys.modules`` -- exactly
the "sounddevice may be absent" case the module itself is written to
tolerate lazily (``_run`` imports it on first use, not at module import).
"""

import os
import sys
import types

import numpy as np
import pytest

import openwakeword

MODEL_PATH = os.path.join(
    os.path.dirname(openwakeword.__file__), "resources", "models", "hey_jarvis_v0.1.onnx"
)

pytestmark = pytest.mark.skipif(
    not os.path.isfile(MODEL_PATH),
    reason="bundled hey_jarvis_v0.1.onnx model not found in this environment",
)


def _install_fake_sounddevice(monkeypatch, frames):
    """Register a fake ``sounddevice`` module whose ``InputStream`` pushes
    ``frames`` into the listener's internal buffer synchronously as soon as
    ``.start()`` is called -- no real audio hardware, no background thread,
    no timing races."""

    fake_sd = types.ModuleType("sounddevice")

    class _FakeInputStream:
        def __init__(self, samplerate=None, channels=None, dtype=None,
                     blocksize=None, callback=None):
            self.callback = callback
            self.blocksize = blocksize
            self.stopped = False
            self.closed = False

        def start(self):
            for frame in frames:
                indata = frame.reshape(-1, 1)
                self.callback(indata, len(frame), None, None)

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    fake_sd.InputStream = _FakeInputStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)


def _make_listener(on_detect, threshold=0.5, cooldown_s=100.0):
    from hermes_cli.wakeword import WakeWordListener

    return WakeWordListener(
        model_path=MODEL_PATH,
        on_detect=on_detect,
        threshold=threshold,
        cooldown_s=cooldown_s,
    )


def _run_with_scores(monkeypatch, listener, scores):
    """Feed ``len(scores)`` silence frames through ``listener._run()``
    synchronously, one call to the (monkeypatched) ``predict()`` per frame
    returning the corresponding score, then stop the loop.

    Runs on the calling thread (not ``listener.start()``'s daemon thread) so
    the whole sequence is deterministic: no sleeps, no polling, no races.
    """
    frames = [np.zeros(listener.frame_size, dtype="int16") for _ in scores]
    _install_fake_sounddevice(monkeypatch, frames)

    calls = {"n": 0}

    def fake_predict(_chunk):
        i = calls["n"]
        calls["n"] += 1
        if calls["n"] >= len(scores):
            # Last frame processed -- stop the loop after this iteration.
            listener._stop_event.set()
        return {listener._model_key: scores[i]}

    monkeypatch.setattr(listener._model, "predict", fake_predict)
    listener._run()
    return calls["n"]


class TestSilenceDoesNotTrigger:
    def test_silence_frames_never_fire_on_detect(self, monkeypatch):
        detections = []
        listener = _make_listener(on_detect=lambda: detections.append(1))

        processed = _run_with_scores(monkeypatch, listener, [0.0] * 5)

        assert processed == 5
        assert detections == []


class TestDetectionAndCooldown:
    def test_score_above_threshold_fires_on_detect_exactly_once(self, monkeypatch):
        detections = []
        listener = _make_listener(on_detect=lambda: detections.append(1), threshold=0.5)

        # Silence, then one clear detection, then silence again.
        _run_with_scores(monkeypatch, listener, [0.0, 0.0, 0.9, 0.0, 0.0])

        assert detections == [1]

    def test_repeat_high_score_within_cooldown_does_not_refire(self, monkeypatch):
        """A second frame scoring above threshold immediately after the
        first (cooldown_s=100, well beyond the test's real wall-clock
        duration) must not trigger a second callback -- this is the
        debounce the design spec requires."""
        detections = []
        listener = _make_listener(
            on_detect=lambda: detections.append(1), threshold=0.5, cooldown_s=100.0
        )

        _run_with_scores(monkeypatch, listener, [0.9, 0.9, 0.9])

        assert detections == [1]

    def test_score_exactly_at_threshold_counts_as_detection(self, monkeypatch):
        detections = []
        listener = _make_listener(on_detect=lambda: detections.append(1), threshold=0.5)

        _run_with_scores(monkeypatch, listener, [0.5])

        assert detections == [1]

    def test_on_detect_exception_does_not_crash_the_loop(self, monkeypatch):
        """on_detect raising must be swallowed (logged) so a bad callback
        can't wedge the whole listener thread."""
        calls = []

        def _boom():
            calls.append(1)
            raise RuntimeError("callback exploded")

        listener = _make_listener(on_detect=_boom, threshold=0.5)

        # Should not raise despite on_detect() blowing up on the first hit.
        # Detection breaks out of the read loop (and closes the stream)
        # before on_detect() runs -- per the single-fire/"stops listening"
        # contract -- so the second (silence) frame is never reached.
        processed = _run_with_scores(monkeypatch, listener, [0.9, 0.0])

        assert processed == 1
        assert calls == [1]


class TestConstructorValidation:
    def test_non_callable_on_detect_raises(self):
        from hermes_cli.wakeword import WakeWordListener

        with pytest.raises(ValueError):
            WakeWordListener(model_path=MODEL_PATH, on_detect="not callable")

    def test_default_frame_size_is_1280_samples(self):
        listener = _make_listener(on_detect=lambda: None)

        # openWakeWord requires exactly 80ms @ 16kHz frames == 1280 samples.
        assert listener.frame_size == 1280
