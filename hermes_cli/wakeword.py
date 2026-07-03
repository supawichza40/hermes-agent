"""Wake-word front-gate for the JARVIS voice interface.

Owns the microphone while Hermes voice mode is otherwise idle, running
openWakeWord on live 16 kHz/16-bit PCM frames. On a detection above
threshold it fires a callback exactly once and stops listening -- the
caller (``hermes_cli.voice``) is responsible for starting the existing
Hermes capture and, once the turn is complete, re-arming a new
``WakeWordListener``.

Kept as an isolated module so the mic-contention state machine in
``hermes_cli/voice.py`` stays the single owner of "who has the mic right
now" -- this class only ever opens/closes its own ``sounddevice``
stream and never touches Hermes' recorder objects.

Dependencies: ``openwakeword`` (ONNX inference on macOS -- NOT tflite)
and ``sounddevice`` -- the same audio capture library already used by
``tools/voice_mode.py``. No new pip dependencies.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class WakeWordListener:
    """Background wake-word listener built on openWakeWord + sounddevice.

    Usage::

        listener = WakeWordListener(
            model_path="/path/to/hey_jarvis_v0.1.onnx",
            threshold=0.5,
            on_detect=my_callback,
        )
        listener.start()   # begins polling the mic on a daemon thread
        ...
        listener.stop()    # stops the thread and releases the mic

    Raises ``RuntimeError`` from ``__init__``/``start()`` if openwakeword
    or sounddevice are missing, or if the model fails to load -- callers
    should catch this and fall back to push-to-talk (per design spec).
    """

    def __init__(
        self,
        model_path: str,
        on_detect: Callable[[], None],
        threshold: float = 0.5,
        frame_ms: int = 80,
        sample_rate: int = 16000,
        cooldown_s: float = 2.0,
    ) -> None:
        if not callable(on_detect):
            raise ValueError("on_detect must be callable")

        self.model_path = model_path
        self.on_detect = on_detect
        self.threshold = float(threshold)
        self.sample_rate = int(sample_rate)
        # openWakeWord operates on exactly 80ms (1280 samples @ 16kHz)
        # frames; frame_ms/sample_rate are kept configurable for testing
        # but the product default must yield 1280.
        self.frame_size = int(self.sample_rate * frame_ms / 1000)
        self.cooldown_s = float(cooldown_s)

        try:
            import numpy as np  # noqa: F401
            from openwakeword.model import Model
        except ImportError as e:
            raise RuntimeError(
                "Wake word detection requires openwakeword and numpy.\n"
                f"Install with: pip install openwakeword"
            ) from e

        try:
            self._model = Model(
                wakeword_models=[model_path],
                inference_framework="onnx",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load wake word model at {model_path!r}: {e}"
            ) from e

        model_keys = list(self._model.models.keys())
        if not model_keys:
            raise RuntimeError(f"No models loaded from {model_path!r}")
        # Single-model listener -- use whichever key openWakeWord assigned
        # (derived from the model filename, e.g. "hey_jarvis_v0.1").
        self._model_key = model_keys[0]

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stream = None
        self._cooldown_until = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start polling the microphone on a daemon background thread.

        Idempotent -- calling while already running is a no-op.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="wakeword-listener", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the poll loop and release the microphone.

        Safe to call multiple times / when not running.
        """
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            self._thread = None
        # Guard against self-join: on_detect() (invoked from _run, after
        # this listener's own stream teardown) may call back into stop()
        # on this same thread -- joining ourselves would raise
        # "RuntimeError: cannot join current thread". The thread is about
        # to return on its own once on_detect() finishes, so skipping the
        # join here is safe.
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=timeout)
        self._close_stream()

    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # -- internals -----------------------------------------------------

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            logger.debug("wakeword: stream close raised %s: %s", type(e).__name__, e)

    def _run(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as e:
            logger.warning("wakeword: sounddevice/numpy unavailable: %s", e)
            return

        frame_size = self.frame_size
        # Callback-based capture, mirroring tools/voice_mode.py's
        # AudioRecorder pattern: a lightweight buffer append in the audio
        # thread, prediction work done on this polling thread.
        buffer: list = []
        buffer_lock = threading.Lock()

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                logger.debug("wakeword: sounddevice status: %s", status)
            with buffer_lock:
                buffer.append(indata[:, 0].copy())

        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=frame_size,
                callback=_callback,
            )
            stream.start()
        except Exception as e:
            logger.warning("wakeword: failed to open mic input stream: %s", e)
            return

        self._stream = stream
        logger.info(
            "Wake word listener armed (model=%s, threshold=%.2f)",
            self._model_key,
            self.threshold,
        )

        detected = False
        try:
            while not self._stop_event.is_set():
                with buffer_lock:
                    if not buffer:
                        chunk = None
                    else:
                        chunk = buffer.pop(0)
                if chunk is None:
                    time.sleep(0.01)
                    continue
                if len(chunk) != frame_size:
                    # Partial/oversized chunk (stream startup/shutdown) --
                    # skip rather than feed a malformed frame to predict().
                    continue

                try:
                    scores = self._model.predict(chunk)
                except Exception as e:
                    logger.warning("wakeword: predict() failed: %s", e)
                    continue

                score = scores.get(self._model_key, 0.0)
                now = time.monotonic()
                if score >= self.threshold and now >= self._cooldown_until:
                    self._cooldown_until = now + self.cooldown_s
                    logger.info("Wake word detected (score=%.3f)", score)
                    # Don't fire on_detect() here -- this InputStream is
                    # still open at this point, and on_detect() (via
                    # hermes_cli.voice._on_detect) synchronously starts
                    # Hermes' own recorder stream. Break out and let the
                    # `finally` below close this stream first, so the two
                    # streams are never open at the same time (single-fire
                    # is guaranteed by breaking out of the loop for good).
                    detected = True
                    break
        finally:
            self._close_stream()

        if detected:
            try:
                self.on_detect()
            except Exception as e:
                logger.error(
                    "wakeword: on_detect callback raised %s: %s",
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
