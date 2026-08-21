import atexit
import ctypes
import random
import re
from collections import deque
from ctypes import wintypes
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import hypot
from pathlib import Path
from queue import Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter, sleep

import cv2
import numpy as np
import pydirectinput
import torch
from mss import mss
from PIL import Image, ImageDraw, ImageFont
from rapidocr import RapidOCR
from ultralytics import YOLO

from capture_game_window import enable_dpi_awareness, get_client_region, list_visible_windows


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "maplestory_snake.pt"
CLASS_CONFIDENCE_THRESHOLDS = {
    "snake": 0.60,
    "slime": 0.8,
    "neso": 0.20,
    "player": 0.60,
    "platform": 0.25,
    "rope": 0.10,
}
IMAGE_SIZE = 960
# Use "cpu" for CPU inference; use 0 for the first NVIDIA GPU.
INFERENCE_DEVICE: str | int = 0
CPU_INFERENCE_THREADS = 24
RENDER_PREVIEW_WINDOW = True
SHOW_PLAYER_COORDINATES = True
TERMINAL_STATUS_INTERVAL = 1.0
OCR_INTERVAL = 0.25
OCR_CONFIDENCE = 0.60
OCR_FONT_PATH = Path("C:/Windows/Fonts/msjhbd.ttc")
MY_PLAYER_NAME = "阿罵的手機"
NAME_MATCH_THRESHOLD = 0.80

# Jump until the player's initially obscured name can be identified by OCR.
PLAYER_DISCOVERY_JUMP_ENABLED = True
PLAYER_DISCOVERY_JUMP_INITIAL_DELAY = 0.50
PLAYER_DISCOVERY_JUMP_INTERVAL = 1.00

MONSTER_CLASS_NAMES = frozenset(
    {"snake", "slime"}
)
COIN_CLASS_NAMES = frozenset({"neso"})
PLATFORM_CLASS_NAMES = frozenset({"platform"})
IGNORED_CLASS_NAMES = frozenset()

# A YOLO monster must demonstrate motion before it becomes a valid target.
MONSTER_TRACK_MATCH_DISTANCE = 80
MONSTER_TRACK_MAX_MISSING = 0.50
MONSTER_MOTION_WINDOW = 0.75
MONSTER_MIN_MOTION_PIXELS = 8
MONSTER_MOTION_MIN_OBSERVATIONS = 3
MONSTER_MOTION_CONFIRM_SAMPLES = 2

# Automatic pickup timing.
AUTOMATION_START_DELAY = 3.0
AUTO_PICKUP_KEY = "z"
AUTO_PICKUP_INTERVAL = 0.15

# Timed buffs: (key, recast interval in seconds). Due buffs press their key
# BUFF_TAP_COUNT times, spaced BUFF_TAP_INTERVAL seconds apart.
BUFF_SCHEDULES: list[tuple[str, float]] = [
    ("s", 275.0),
]
BUFF_TAP_INTERVAL = 0.20
BUFF_TAP_COUNT = 5
BUFF_ROPE_BLOCKING_STATES = frozenset(
    {
        "climb_prepare",
        "climb_up",
        "rope_escape_prepare",
        "rope_escape_jump",
    }
)

# Two-layer combat and navigation hyperparameters.
SLIME_ATTACK_KEY = "q"
SNAKE_ATTACK_KEY = "w"
ATTACK_KEY_BY_MONSTER = {
    "slime": SLIME_ATTACK_KEY,
    "snake": SNAKE_ATTACK_KEY,
}
ATTACK_INTERVAL = 0.50
ATTACK_RANGE_HORIZONTAL_PADDING = 200
ATTACK_RANGE_VERTICAL_PADDING = 10
SAME_LAYER_MONSTER_MAX_DISTANCE = 350
STUCK_ATTACK_DURATION = 3.0
UNSTICK_MOVE_DURATION = 2.0

LEFT_KEY = "left"
RIGHT_KEY = "right"
UP_KEY = "up"
DOWN_KEY = "down"
JUMP_KEY = "alt"

LAYER_SPLIT_Y = 500
UPPER_PLATFORM_SNAKE_THRESHOLD = 3
UPPER_CLEAR_CONFIRM_DURATION = 1.0
PLAYER_PLATFORM_HORIZONTAL_MARGIN = 30
PLAYER_PLATFORM_FEET_TOLERANCE = 45
PLATFORM_MONSTER_HORIZONTAL_MARGIN = 30
PLATFORM_MONSTER_FEET_TOLERANCE = 45
SAME_PLATFORM_FALLBACK_FEET_TOLERANCE = 50
ROPE_PLATFORM_HORIZONTAL_MARGIN = 35
ROPE_PLATFORM_VERTICAL_TOLERANCE = 45
LOWER_PATROL_DIRECTION_DURATION = 10.0
# The middle 50% of an upper platform is treated as its combat-ready area.
UPPER_PLATFORM_COMBAT_REGION_RATIO = 0.50
COIN_PICKUP_X_TOLERANCE = 25
COIN_SAME_LEVEL_FEET_TOLERANCE = 50
COIN_DETECTION_GRACE_DURATION = 0.75
UPPER_COIN_REVERSE_SEARCH_DURATION = 5.0
COIN_TARGET_LOST_GRACE_DURATION = 1.0
COIN_TARGET_BLIND_PURSUIT_MAX_DURATION = 3.0
COIN_DIRECTION_LOCK_DURATION = 1.0
COIN_PICKUP_PASS_DURATION = 0.40
COIN_TARGET_MATCH_DISTANCE = 120
COIN_REACQUIRE_COOLDOWN = 0.75
COIN_REACQUIRE_IGNORE_DISTANCE = 80

ROPE_FINE_ALIGNMENT_DISTANCE = 30
ROPE_CENTER_ENTER_TOLERANCE = 8
ROPE_CENTER_EXIT_TOLERANCE = 18
ROPE_ALIGNMENT_TARGET_X_TOLERANCE = 20
ROPE_ALIGN_TAP_INTERVAL = 0.10
ROPE_CENTER_CONFIRM_DURATION = 0.50
ROPE_SEARCH_DIRECTION_DURATION = 0.50
ROPE_HANG_CENTER_TOLERANCE = 12
ROPE_VERTICAL_MARGIN = 15
CLIMB_JUMP_LEAD_DURATION = 0.20
CLIMB_ALT_HOLD_DURATION = 1.00
CLIMB_UP_DURATION = 2.25
DROP_DOWN_HOLD_DURATION = 0.50
DROP_ALT_HOLD_DURATION = 0.50
ROPE_HANG_TIMEOUT = 5.0
ROPE_HANG_DETECTION_GRACE = 0.50
ROPE_ESCAPE_DIRECTION_HOLD_DURATION = 1.00
ROPE_ESCAPE_JUMP_DURATION = 0.50
PLAYER_STATIONARY_TIMEOUT = 10.0
PLAYER_STATIONARY_MOVEMENT_THRESHOLD = 10

# Resource automation hyperparameters.
RESOURCE_OCR_INTERVAL = 0.25
HP_POTION_KEY = "1"
CRITICAL_HP_THRESHOLD = 300
CRITICAL_HP_HEAL_KEY = "e"
CRITICAL_HP_HEAL_INTERVAL = 0.25
CRITICAL_HP_FALLBACK_DELAY = 1.00
CRITICAL_HP_FALLBACK_INTERVAL = 0.25
MP_DEFICIT_THRESHOLD = 1000
MP_POTION_KEY = "2"

HP_ROI_OFFSETS = (-183, -104, -35, -15)
MP_ROI_OFFSETS = (-73, 7, -35, -15)
RESOURCE_OCR_SCALE = 5
DIRECTINPUT_KEY_HOLD = 0.060

# Hunt/rest cycle hyperparameters. Mouse positions are absolute screen coordinates.
HUNT_DURATION_MINUTES = 30
REST_DURATION_MINUTES = 2
REST_MOUSE_SLOW_MOVE_DURATION = 0.80
REST_POST_RETURN_MOVE_DURATION = 2.0
REST_MOUSE_PARK_MOVE_DURATION = 2.0
REST_MOUSE_STEP_DELAY = 1.0
REST_MOUSE_DOUBLE_CLICK_INTERVAL = 0.10
REST_MENU_FIRST_POSITION = (1276, 916)
REST_MENU_SECOND_POSITION = (1269, 877)
REST_MENU_THIRD_POSITION = (1012, 591)
REST_RETURN_POSITION = (865, 613)
REST_POST_RETURN_CLICK_POSITION = (1349, 882)
REST_MOUSE_PARK_POSITION = (1605, 182)

# Scroll-template detection hyperparameters (detection only).
SCROLL_TEMPLATE_DIRECTORY = ROOT / "assests" / "scroll"
SCROLL_IDLE_SCAN_INTERVAL = 3.0
SCROLL_ACTIVE_SCAN_INTERVAL = 0.25
SCROLL_TEMPLATE_MATCH_THRESHOLD = 0.92
SCROLL_MAX_MEAN_COLOR_ERROR = 35.0
SCROLL_MAX_MATCHES_PER_TEMPLATE = 5
SCROLL_NMS_THRESHOLD = 0.30

user32 = ctypes.windll.user32
user32.GetForegroundWindow.argtypes = ()
user32.GetForegroundWindow.restype = wintypes.HWND


class DirectInputController:
    """Route DirectInput through independent key-group workers."""

    WORKER_KEYS = {
        "z": frozenset({AUTO_PICKUP_KEY}),
        "attack": frozenset({SLIME_ATTACK_KEY, SNAKE_ATTACK_KEY}),
        "horizontal": frozenset({LEFT_KEY, RIGHT_KEY}),
        "vertical": frozenset({UP_KEY, DOWN_KEY}),
        "alt": frozenset({JUMP_KEY}),
        "resource": frozenset(
            {HP_POTION_KEY, MP_POTION_KEY, CRITICAL_HP_HEAL_KEY}
        ),
        "buff": frozenset(key for key, _interval in BUFF_SCHEDULES),
    }

    def __init__(self, hwnd: int) -> None:
        self.hwnd = hwnd
        self.held_keys: set[str] = set()
        self._physical_held_keys: set[str] = set()
        self._pending_taps: set[str] = set()
        self._lock = Lock()
        self._stop_event = Event()
        self._closed = False
        self._paused = False
        self._key_to_worker = {
            key: worker_name
            for worker_name, keys in self.WORKER_KEYS.items()
            for key in keys
        }
        self._commands: dict[str, Queue[tuple[str, object]]] = {
            worker_name: Queue() for worker_name in self.WORKER_KEYS
        }
        self._workers = {
            worker_name: Thread(
                target=self._worker_loop,
                args=(worker_name,),
                name=f"directinput-{worker_name}",
                daemon=True,
            )
            for worker_name in self.WORKER_KEYS
        }
        for worker in self._workers.values():
            worker.start()

    def has_focus(self) -> bool:
        return user32.GetForegroundWindow() == self.hwnd

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def _command_queue_for(self, key: str) -> Queue[tuple[str, object]] | None:
        worker_name = self._key_to_worker.get(key)
        return None if worker_name is None else self._commands[worker_name]

    def key_down(self, key: str) -> bool:
        command_queue = self._command_queue_for(key)
        if command_queue is None or not self.has_focus():
            return False

        with self._lock:
            if self._closed or self._paused:
                return False
            if key in self.held_keys:
                return True
            if key in self._pending_taps:
                return False
            self.held_keys.add(key)
        command_queue.put(("down", key))
        return True

    def key_up(self, key: str) -> bool:
        command_queue = self._command_queue_for(key)
        if command_queue is None:
            return False

        with self._lock:
            if self._closed:
                return False
            if key not in self.held_keys:
                return True
            self.held_keys.discard(key)
        command_queue.put(("up", key))
        return True

    def tap(self, key: str) -> bool:
        command_queue = self._command_queue_for(key)
        if command_queue is None or not self.has_focus():
            return False

        with self._lock:
            if (
                self._closed
                or self._paused
                or key in self.held_keys
                or key in self._pending_taps
            ):
                return False
            self._pending_taps.add(key)
        command_queue.put(("tap", key))
        return True

    def tap_attack(self, direction: str, attack_key: str) -> bool:
        """Tap a facing direction, then tap the requested monster skill."""
        if (
            direction not in (LEFT_KEY, RIGHT_KEY)
            or attack_key not in ATTACK_KEY_BY_MONSTER.values()
            or not self.has_focus()
        ):
            return False

        requested_keys = (direction, attack_key)
        with self._lock:
            if self._closed or self._paused or any(
                key in self.held_keys or key in self._pending_taps
                for key in requested_keys
            ):
                return False
            self._pending_taps.update(requested_keys)

        direction_completed = Event()
        self._commands["horizontal"].put(
            ("tap_signal", (direction, direction_completed))
        )
        self._commands["attack"].put(
            ("tap_after", (attack_key, direction_completed))
        )
        return True

    def set_horizontal(self, direction: str | None) -> None:
        for key in (LEFT_KEY, RIGHT_KEY):
            if key != direction:
                self.key_up(key)
        if direction is not None:
            self.key_down(direction)

    def release_navigation(self) -> None:
        for key in (LEFT_KEY, RIGHT_KEY, UP_KEY, DOWN_KEY, JUMP_KEY):
            self.key_up(key)

    def release_all(self) -> None:
        with self._lock:
            keys = tuple(self.held_keys)
        for key in keys:
            self.key_up(key)

    def set_paused(self, paused: bool) -> None:
        """Pause every keyboard worker and release all physically held keys."""
        with self._lock:
            if self._closed or self._paused == paused:
                return
            self._paused = paused
            if not paused:
                return
            self.held_keys.clear()
            self._pending_taps.clear()

        for worker_name, keys in self.WORKER_KEYS.items():
            command_queue = self._commands[worker_name]
            for key in keys:
                command_queue.put(("up", key))

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.held_keys.clear()
        self._stop_event.set()
        for command_queue in self._commands.values():
            command_queue.put(("stop", None))
        for worker in self._workers.values():
            if worker.is_alive():
                worker.join(timeout=1.0)
        with self._lock:
            self._pending_taps.clear()

    def _worker_loop(self, worker_name: str) -> None:
        command_queue = self._commands[worker_name]
        try:
            while not self._stop_event.is_set():
                action, payload = command_queue.get()
                if action == "stop" or payload is None or self._stop_event.is_set():
                    break

                with self._lock:
                    paused = self._paused
                if paused and action != "up":
                    self._cancel_pending_action(action, payload)
                    continue

                if action == "down":
                    key = str(payload)
                    self._physical_key_down(key)
                elif action == "up":
                    key = str(payload)
                    self._physical_key_up(key)
                elif action == "tap":
                    key = str(payload)
                    try:
                        self._physical_tap(key)
                    finally:
                        with self._lock:
                            self._pending_taps.discard(key)
                elif action == "tap_signal":
                    key, completed = payload
                    try:
                        self._physical_tap(str(key))
                    finally:
                        with self._lock:
                            self._pending_taps.discard(str(key))
                        completed.set()
                elif action == "tap_after":
                    key, prerequisite = payload
                    try:
                        while not prerequisite.wait(0.01):
                            if self._stop_event.is_set():
                                break
                        if self._stop_event.is_set() or self.paused:
                            continue
                        self._physical_tap(str(key))
                    finally:
                        with self._lock:
                            self._pending_taps.discard(str(key))
        finally:
            with self._lock:
                worker_keys = tuple(
                    key
                    for key in self._physical_held_keys
                    if self._key_to_worker.get(key) == worker_name
                )
            for key in worker_keys:
                self._physical_key_up(key)

    def _cancel_pending_action(self, action: str, payload: object) -> None:
        if action in {"tap", "down"}:
            key = str(payload)
            with self._lock:
                self._pending_taps.discard(key)
                self.held_keys.discard(key)
            return

        if action == "tap_signal":
            key, completed = payload
            with self._lock:
                self._pending_taps.discard(str(key))
            completed.set()
            return

        if action == "tap_after":
            key, _prerequisite = payload
            with self._lock:
                self._pending_taps.discard(str(key))

    def _physical_key_down(self, key: str) -> bool:
        with self._lock:
            if self._paused:
                self.held_keys.discard(key)
                return False
            if key in self._physical_held_keys:
                return True
        if not self.has_focus():
            with self._lock:
                self.held_keys.discard(key)
            return False
        succeeded = bool(pydirectinput.keyDown(key))
        if succeeded:
            with self._lock:
                self._physical_held_keys.add(key)
        else:
            with self._lock:
                self.held_keys.discard(key)
        return succeeded

    def _physical_key_up(self, key: str) -> bool:
        with self._lock:
            if key not in self._physical_held_keys:
                return True
        try:
            return bool(pydirectinput.keyUp(key))
        finally:
            with self._lock:
                self._physical_held_keys.discard(key)

    def _physical_tap(self, key: str) -> bool:
        if self.paused or not self.has_focus():
            return False

        key_down_ok = pydirectinput.keyDown(key)
        try:
            sleep(DIRECTINPUT_KEY_HOLD)
        finally:
            key_up_ok = pydirectinput.keyUp(key)
        return bool(key_down_ok and key_up_ok)


class TimedBuffController:
    """Schedule non-blocking buff key bursts when gameplay is safe."""

    def __init__(self, inputs: DirectInputController, started_at: float) -> None:
        self.inputs = inputs
        self._intervals = dict(BUFF_SCHEDULES)
        if any(not key or interval <= 0 for key, interval in BUFF_SCHEDULES):
            raise ValueError("Every buff needs a key and a positive interval")
        if len(self._intervals) != len(BUFF_SCHEDULES):
            raise ValueError("Buff keys must be unique")

        # Every configured buff is immediately due on program startup.
        self._next_due_at = {
            key: started_at for key in self._intervals
        }
        self._completed_once: set[str] = set()
        self._active_key: str | None = None
        self._status = "BUFF: IDLE"
        self._lock = Lock()
        self._stop_event = Event()
        self._cancel_event = Event()
        self._closed = False
        self._worker: Thread | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active_key is not None

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def initial_buffs_completed(self) -> bool:
        with self._lock:
            return len(self._completed_once) == len(self._intervals)

    def update(self, now: float, blocked_reason: str | None) -> str | None:
        worker: Thread | None = None
        with self._lock:
            if self._closed:
                return None

            if self._active_key is not None:
                if blocked_reason is not None:
                    self._cancel_event.set()
                    self._status = (
                        f"BUFF {self._active_key.upper()}: DEFERRED "
                        f"({blocked_reason})"
                    )
                return self._status

            due_keys = [
                key
                for key, due_at in self._next_due_at.items()
                if now >= due_at
            ]
            if not due_keys:
                return None

            key = min(due_keys, key=self._next_due_at.__getitem__)
            if blocked_reason is not None:
                self._status = (
                    f"BUFF {key.upper()}: WAITING ({blocked_reason})"
                )
                return self._status
            if self.inputs.paused or not self.inputs.has_focus():
                self._status = f"BUFF {key.upper()}: WAITING (INPUT PAUSED)"
                return self._status

            self._cancel_event.clear()
            self._active_key = key
            self._status = f"BUFF {key.upper()}: STARTING"
            worker = Thread(
                target=self._run_buff,
                args=(key, self._intervals[key]),
                name=f"timed-buff-{key}",
                daemon=True,
            )
            self._worker = worker

        worker.start()
        return self.status

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop_event.set()
        self._cancel_event.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

    def _run_buff(self, key: str, interval: float) -> None:
        completed = False
        try:
            for index in range(BUFF_TAP_COUNT):
                if self._stop_event.is_set() or self._cancel_event.is_set():
                    return
                if not self.inputs.tap(key):
                    with self._lock:
                        self._status = (
                            f"BUFF {key.upper()}: INPUT FAILED; WILL RETRY"
                        )
                    return

                with self._lock:
                    self._status = (
                        f"BUFF {key.upper()}: "
                        f"{index + 1}/{BUFF_TAP_COUNT}"
                    )
                if (
                    index + 1 < BUFF_TAP_COUNT
                    and self._cancel_event.wait(BUFF_TAP_INTERVAL)
                ):
                    return
            completed = True
        finally:
            finished_at = perf_counter()
            with self._lock:
                if completed:
                    self._completed_once.add(key)
                    self._next_due_at[key] = finished_at + interval
                    self._status = (
                        f"BUFF {key.upper()}: COMPLETED; "
                        f"NEXT IN {interval:.0f}s"
                    )
                elif not self._closed:
                    self._status = f"BUFF {key.upper()}: DEFERRED; WILL RETRY"
                self._active_key = None
                self._worker = None


class RestCycleController:
    """Run the timed mouse-only rest sequence without blocking video processing."""

    def __init__(
        self,
        inputs: DirectInputController,
        hunting_started_at: float,
    ) -> None:
        self.inputs = inputs
        self._lock = Lock()
        self._stop_event = Event()
        self._resume_ready = Event()
        self._active = False
        self._closed = False
        self._status = "HUNTING"
        self._rest_ends_at: float | None = None
        self._next_rest_at = (
            hunting_started_at + HUNT_DURATION_MINUTES * 60.0
        )
        self._worker: Thread | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def resume_ready(self) -> bool:
        return self._resume_ready.is_set()

    def status(self, now: float) -> str:
        with self._lock:
            status = self._status
            rest_ends_at = self._rest_ends_at
        if rest_ends_at is not None:
            remaining = max(0.0, rest_ends_at - now)
            return f"RESTING: {remaining:.0f}s remaining"
        return status

    def update(self, now: float) -> None:
        if not self.inputs.has_focus():
            return
        with self._lock:
            if self._closed or self._active or now < self._next_rest_at:
                return
            self._active = True
            self._status = "REST: preparing mouse sequence"

        print("[REST] Hunt duration reached; keyboard input is now paused.", flush=True)
        self.inputs.set_paused(True)
        self._worker = Thread(
            target=self._run_mouse_sequence,
            name="rest-mouse-sequence",
            daemon=True,
        )
        self._worker.start()

    def resume_hunting(self, now: float) -> None:
        if not self._resume_ready.is_set():
            return
        with self._lock:
            if self._closed:
                return
            self._active = False
            self._status = "HUNTING"
            self._rest_ends_at = None
            self._next_rest_at = now + HUNT_DURATION_MINUTES * 60.0
        self._resume_ready.clear()
        self.inputs.set_paused(False)
        print("[REST] Rest complete; keyboard input and hunting resumed.", flush=True)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop_event.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

    def _wait(self, duration: float) -> bool:
        return not self._stop_event.wait(duration)

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def _move_and_click(
        self,
        position: tuple[int, int],
        status: str,
        clicks: int = 1,
    ) -> bool:
        if self._stop_event.is_set():
            return False
        self._set_status(status)
        pydirectinput.moveTo(
            position[0],
            position[1],
            duration=REST_MOUSE_SLOW_MOVE_DURATION,
        )
        if self._stop_event.is_set():
            return False
        pydirectinput.click(
            button="left",
            clicks=clicks,
            interval=REST_MOUSE_DOUBLE_CLICK_INTERVAL if clicks > 1 else 0.0,
        )
        return self._wait(REST_MOUSE_STEP_DELAY)

    def _run_mouse_sequence(self) -> None:
        try:
            if not self._move_and_click(
                REST_MENU_FIRST_POSITION,
                "REST: mouse step 1/5",
            ):
                return
            if not self._move_and_click(
                REST_MENU_SECOND_POSITION,
                "REST: mouse step 2/5",
            ):
                return
            if not self._move_and_click(
                REST_MENU_THIRD_POSITION,
                "REST: mouse step 3/5",
            ):
                return

            rest_duration = REST_DURATION_MINUTES * 60.0
            with self._lock:
                self._rest_ends_at = perf_counter() + rest_duration
                self._status = "RESTING"
            if not self._wait(rest_duration):
                return
            with self._lock:
                self._rest_ends_at = None

            if not self._move_and_click(
                REST_RETURN_POSITION,
                "REST: returning to game",
                clicks=2,
            ):
                return

            self._set_status("REST: post-return click")
            pydirectinput.moveTo(
                REST_POST_RETURN_CLICK_POSITION[0],
                REST_POST_RETURN_CLICK_POSITION[1],
                duration=REST_POST_RETURN_MOVE_DURATION,
            )
            if self._stop_event.is_set():
                return
            pydirectinput.click(button="left")

            self._set_status("REST: parking mouse")
            pydirectinput.moveTo(
                REST_MOUSE_PARK_POSITION[0],
                REST_MOUSE_PARK_POSITION[1],
                duration=REST_MOUSE_PARK_MOVE_DURATION,
            )
        except Exception as error:
            print(f"[REST] Mouse sequence failed: {error}", flush=True)
        finally:
            if not self._stop_event.is_set():
                with self._lock:
                    self._rest_ends_at = None
                    self._status = "REST: ready to resume"
                self._resume_ready.set()


@dataclass(frozen=True)
class ScrollTemplate:
    name: str
    image: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class ScrollMatch:
    name: str
    coordinates: tuple[int, int, int, int]
    score: float


def load_scroll_templates(directory: Path) -> list[ScrollTemplate]:
    templates = []
    for path in sorted(directory.glob("*.png")):
        source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if source is None or source.ndim != 3 or source.shape[2] != 4:
            raise ValueError(f"Scroll template must be an RGBA PNG: {path}")

        image = source[:, :, :3]
        alpha = source[:, :, 3]
        mask = np.where(alpha >= 128, 255, 0).astype(np.uint8)
        if cv2.countNonZero(mask) == 0:
            raise ValueError(f"Scroll template has no visible pixels: {path}")
        templates.append(ScrollTemplate(path.stem, image, mask))

    if len(templates) != 3:
        raise RuntimeError(
            f"Expected 3 scroll templates in {directory}, found {len(templates)}"
        )
    return templates


def find_scroll_matches(
    frame: np.ndarray,
    templates: list[ScrollTemplate],
) -> list[ScrollMatch]:
    candidates: list[ScrollMatch] = []
    for template in templates:
        template_height, template_width = template.image.shape[:2]
        frame_height, frame_width = frame.shape[:2]
        if template_width > frame_width or template_height > frame_height:
            continue

        result = cv2.matchTemplate(
            frame,
            template.image,
            cv2.TM_CCORR_NORMED,
            mask=template.mask,
        )
        result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
        candidate_map = result.copy()
        suppress_x = max(2, template_width // 2)
        suppress_y = max(2, template_height // 2)
        foreground = template.mask > 0

        for _ in range(SCROLL_MAX_MATCHES_PER_TEMPLATE):
            _min_score, max_score, _min_location, max_location = cv2.minMaxLoc(
                candidate_map
            )
            if max_score < SCROLL_TEMPLATE_MATCH_THRESHOLD:
                break

            x, y = max_location
            patch = frame[y : y + template_height, x : x + template_width]
            color_error = float(
                np.abs(
                    patch.astype(np.int16) - template.image.astype(np.int16)
                )[foreground].mean()
            )
            if color_error <= SCROLL_MAX_MEAN_COLOR_ERROR:
                candidates.append(
                    ScrollMatch(
                        template.name,
                        (x, y, x + template_width, y + template_height),
                        float(max_score),
                    )
                )

            candidate_map[
                max(0, y - suppress_y) : min(
                    candidate_map.shape[0], y + suppress_y + 1
                ),
                max(0, x - suppress_x) : min(
                    candidate_map.shape[1], x + suppress_x + 1
                ),
            ] = -1.0

    if not candidates:
        return []

    boxes = [
        [
            match.coordinates[0],
            match.coordinates[1],
            match.coordinates[2] - match.coordinates[0],
            match.coordinates[3] - match.coordinates[1],
        ]
        for match in candidates
    ]
    selected = cv2.dnn.NMSBoxes(
        boxes,
        [match.score for match in candidates],
        score_threshold=SCROLL_TEMPLATE_MATCH_THRESHOLD,
        nms_threshold=SCROLL_NMS_THRESHOLD,
    )
    selected_indices = np.asarray(selected).reshape(-1)
    return [candidates[int(index)] for index in selected_indices]


class ScrollDetector:
    def __init__(self, templates: list[ScrollTemplate]) -> None:
        self.templates = templates
        self._matches: list[ScrollMatch] = []
        self._active = False
        self._next_scan_at = 0.0
        self._scan_pending = False
        self._last_error: Exception | None = None
        self._lock = Lock()
        self._commands: Queue[tuple[str, object]] = Queue(maxsize=1)
        self._stop_event = Event()
        self._closed = False
        self._worker = Thread(
            target=self._worker_loop,
            name="scroll-template-worker",
            daemon=True,
        )
        self._worker.start()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def scan_interval(self) -> float:
        return (
            SCROLL_ACTIVE_SCAN_INTERVAL
            if self.active
            else SCROLL_IDLE_SCAN_INTERVAL
        )

    def update(self, frame: np.ndarray, now: float) -> list[ScrollMatch]:
        with self._lock:
            if self._last_error is not None:
                error = self._last_error
                self._last_error = None
                raise RuntimeError("Scroll template worker failed") from error

            matches = list(self._matches)
            should_scan = (
                not self._closed
                and not self._scan_pending
                and now >= self._next_scan_at
            )
            if should_scan:
                self._scan_pending = True

        if should_scan:
            self._commands.put(("scan", (frame.copy(), now)))
        return matches

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop_event.set()
        try:
            self._commands.put_nowait(("stop", None))
        except Full:
            pass
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            action, payload = self._commands.get()
            if action == "stop" or payload is None or self._stop_event.is_set():
                break

            frame, scan_started_at = payload
            try:
                matches = find_scroll_matches(frame, self.templates)
                interval = (
                    SCROLL_ACTIVE_SCAN_INTERVAL
                    if matches
                    else SCROLL_IDLE_SCAN_INTERVAL
                )
                with self._lock:
                    self._matches = matches
                    self._active = bool(matches)
                    self._next_scan_at = scan_started_at + interval
            except Exception as error:
                with self._lock:
                    self._last_error = error
                    self._next_scan_at = perf_counter() + SCROLL_IDLE_SCAN_INTERVAL
            finally:
                with self._lock:
                    self._scan_pending = False


def read_resource_stat(
    frame: np.ndarray,
    ocr: RapidOCR,
    roi_offsets: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    """Read a current/max HUD value from a bottom-center anchored region."""
    frame_height, frame_width = frame.shape[:2]
    center_x = frame_width // 2
    left, right, top, bottom = roi_offsets
    x1, x2 = center_x + left, center_x + right
    y1, y2 = frame_height + top, frame_height + bottom
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height or x1 >= x2 or y1 >= y2:
        return None

    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(
        crop,
        None,
        fx=RESOURCE_OCR_SCALE,
        fy=RESOURCE_OCR_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    result = ocr(crop, use_det=False, use_cls=False, use_rec=True)
    if not result.txts:
        return None

    text = result.txts[0].translate(
        str.maketrans({"S": "5", "s": "5", "O": "0", "I": "1", "l": "1"})
    )
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if not match:
        return None

    current, maximum = map(int, match.groups())
    if maximum <= 0 or current < 0 or current > maximum:
        return None
    return current, maximum


def choose_window() -> tuple[int, str]:
    windows = list_visible_windows()
    if not windows:
        raise RuntimeError("找不到可擷取的視窗。")

    print("可擷取的視窗：")
    for index, (_hwnd, title, width, height) in enumerate(windows, start=1):
        print(f"  {index:>2}. {title} [{width}x{height}]")

    while True:
        answer = input("請輸入視窗編號：").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(windows):
            hwnd, title, _width, _height = windows[int(answer) - 1]
            return hwnd, title
        print(f"請輸入 1 到 {len(windows)} 之間的編號。")


def crop_player_name(frame: np.ndarray, player_box) -> np.ndarray | None:
    """Crop the single-line character name immediately below a player box."""
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, player_box.xyxy[0])
    box_width = x2 - x1
    box_height = y2 - y1

    crop_x1 = max(0, int(x1 - 0.75 * box_width))
    crop_x2 = min(frame_width, int(x2 + 0.15 * box_width))
    crop_y1 = max(0, int(y2))
    crop_y2 = min(frame_height, int(y2 + 0.42 * box_height))
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)


def is_my_player_name(text: str) -> bool:
    recognized = "".join(character for character in text if character.isalnum())
    expected = "".join(character for character in MY_PLAYER_NAME if character.isalnum())
    if not recognized or not expected:
        return False
    return SequenceMatcher(None, recognized, expected).ratio() >= NAME_MATCH_THRESHOLD


def player_box_center(player_box) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, player_box.xyxy[0])
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def coordinates_center(
    coordinates: tuple[int, int, int, int],
) -> tuple[float, float]:
    x1, y1, x2, y2 = coordinates
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def build_attack_range(
    player: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Build the visible attack box around the tracked player."""
    x1, y1, x2, y2 = player
    frame_height, frame_width = frame_shape[:2]
    return (
        max(0, x1 - ATTACK_RANGE_HORIZONTAL_PADDING),
        max(0, y1 - ATTACK_RANGE_VERTICAL_PADDING),
        min(frame_width - 1, x2 + ATTACK_RANGE_HORIZONTAL_PADDING),
        min(frame_height - 1, y2 + ATTACK_RANGE_VERTICAL_PADDING),
    )


def box_center_is_inside(
    coordinates: tuple[int, int, int, int],
    area: tuple[int, int, int, int],
) -> bool:
    center_x, center_y = coordinates_center(coordinates)
    x1, y1, x2, y2 = area
    return x1 <= center_x <= x2 and y1 <= center_y <= y2


def player_layer(player: tuple[int, int, int, int]) -> str:
    """Classify the map layer using the player's center Y coordinate."""
    _player_x, player_y = coordinates_center(player)
    return "lower" if player_y >= LAYER_SPLIT_Y else "upper"


def object_is_on_player_layer(
    player: tuple[int, int, int, int],
    coordinates: tuple[int, int, int, int],
) -> bool:
    _object_x, object_y = coordinates_center(coordinates)
    object_layer = "lower" if object_y >= LAYER_SPLIT_Y else "upper"
    return object_layer == player_layer(player)


def coin_is_on_player_level(
    player: tuple[int, int, int, int],
    coin: tuple[int, int, int, int],
) -> bool:
    """Keep boundary-adjacent neso on the player's platform level."""
    return abs(player[3] - coin[3]) <= COIN_SAME_LEVEL_FEET_TOLERANCE


def coin_is_on_other_map_layer(
    player: tuple[int, int, int, int],
    coin: tuple[int, int, int, int],
) -> bool:
    """Ignore Y=500 boundary jitter when deciding whether travel is required."""
    return (
        not object_is_on_player_layer(player, coin)
        and not coin_is_on_player_level(player, coin)
    )


def find_player_platform(
    player: tuple[int, int, int, int],
    platforms: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Find the detected platform directly supporting the player's feet."""
    player_x, _player_y = coordinates_center(player)
    player_feet_y = player[3]
    candidates = [
        platform
        for platform in platforms
        if (
            platform[0] - PLAYER_PLATFORM_HORIZONTAL_MARGIN
            <= player_x
            <= platform[2] + PLAYER_PLATFORM_HORIZONTAL_MARGIN
            and abs(player_feet_y - platform[1])
            <= PLAYER_PLATFORM_FEET_TOLERANCE
        )
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda platform: abs(player_feet_y - platform[1]))


def horizontal_distance(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_x, _first_y = coordinates_center(first)
    second_x, _second_y = coordinates_center(second)
    return abs(second_x - first_x)


def horizontal_direction_to(
    player: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
) -> str | None:
    player_x, _player_y = coordinates_center(player)
    target_x, _target_y = coordinates_center(target)
    if target_x < player_x - 1:
        return LEFT_KEY
    if target_x > player_x + 1:
        return RIGHT_KEY
    return None


def nearest_target(
    player: tuple[int, int, int, int],
    targets: list[tuple[tuple[int, int, int, int], str]],
) -> tuple[tuple[int, int, int, int], str]:
    player_x, player_y = coordinates_center(player)
    return min(
        targets,
        key=lambda target: (
            (coordinates_center(target[0])[0] - player_x) ** 2
            + (coordinates_center(target[0])[1] - player_y) ** 2
        ),
    )


def same_platform_nearby_monsters(
    player: tuple[int, int, int, int],
    monsters: list[tuple[tuple[int, int, int, int], str]],
    platforms: list[tuple[int, int, int, int]],
) -> list[tuple[tuple[int, int, int, int], str]]:
    current_platform = find_player_platform(player, platforms)
    if current_platform is not None:
        platform_monsters = monsters_on_platform(current_platform, monsters)
    else:
        platform_monsters = [
            monster
            for monster in monsters
            if abs(player[3] - monster[0][3])
            <= SAME_PLATFORM_FALLBACK_FEET_TOLERANCE
        ]
    return [
        monster
        for monster in platform_monsters
        if horizontal_distance(player, monster[0])
        <= SAME_LAYER_MONSTER_MAX_DISTANCE
    ]


def build_same_layer_pursuit_range(
    player: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    _frame_height, frame_width = frame_shape[:2]
    player_x, _player_y = coordinates_center(player)
    _player_x1, player_y1, _player_x2, player_y2 = player
    return (
        max(0, int(round(player_x - SAME_LAYER_MONSTER_MAX_DISTANCE))),
        player_y1,
        min(
            frame_width - 1,
            int(round(player_x + SAME_LAYER_MONSTER_MAX_DISTANCE)),
        ),
        player_y2,
    )


def player_is_hanging_on_rope(
    player: tuple[int, int, int, int],
    ropes: list[tuple[int, int, int, int]],
) -> bool:
    player_x, player_y = coordinates_center(player)
    for rope_x1, rope_y1, rope_x2, rope_y2 in ropes:
        rope_x = (rope_x1 + rope_x2) / 2
        centered_on_rope = abs(player_x - rope_x) <= ROPE_HANG_CENTER_TOLERANCE
        inside_rope_height = (
            rope_y1 - ROPE_VERTICAL_MARGIN
            <= player_y
            <= rope_y2 + ROPE_VERTICAL_MARGIN
        )
        if centered_on_rope and inside_rope_height:
            return True
    return False


def nearest_upper_platform(
    player: tuple[int, int, int, int],
    platforms: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Choose the closest platform whose top belongs to the upper layer."""
    candidates = [platform for platform in platforms if platform[1] < LAYER_SPLIT_Y]
    if not candidates:
        return None

    player_x, player_y = coordinates_center(player)

    def distance_to_platform(platform: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = platform
        nearest_x = min(max(player_x, x1), x2)
        nearest_y = min(max(player_y, y1), y2)
        return hypot(player_x - nearest_x, player_y - nearest_y)

    return min(candidates, key=distance_to_platform)


def upper_platform_for_target(
    target: tuple[int, int, int, int],
    platforms: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Choose the upper platform geometrically closest to a target object."""
    candidates = [platform for platform in platforms if platform[1] < LAYER_SPLIT_Y]
    if not candidates:
        return None

    target_x, target_y = coordinates_center(target)

    def distance_to_platform(platform: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = platform
        nearest_x = min(max(target_x, x1), x2)
        nearest_y = min(max(target_y, y1), y2)
        return hypot(target_x - nearest_x, target_y - nearest_y)

    return min(candidates, key=distance_to_platform)


def monsters_on_platform(
    platform: tuple[int, int, int, int],
    monsters: list[tuple[tuple[int, int, int, int], str]],
    class_name: str | None = None,
) -> list[tuple[tuple[int, int, int, int], str]]:
    platform_x1, platform_y1, platform_x2, _platform_y2 = platform
    matched = []
    for monster in monsters:
        coordinates, detected_name = monster
        if class_name is not None and detected_name != class_name:
            continue
        monster_x, _monster_y = coordinates_center(coordinates)
        monster_feet_y = coordinates[3]
        if (
            platform_x1 - PLATFORM_MONSTER_HORIZONTAL_MARGIN
            <= monster_x
            <= platform_x2 + PLATFORM_MONSTER_HORIZONTAL_MARGIN
            and abs(monster_feet_y - platform_y1)
            <= PLATFORM_MONSTER_FEET_TOLERANCE
        ):
            matched.append(monster)
    return matched


def rope_connected_to_platform(
    player: tuple[int, int, int, int],
    platform: tuple[int, int, int, int],
    ropes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    platform_x1, platform_y1, platform_x2, platform_y2 = platform
    connected = []
    for rope in ropes:
        rope_x, _rope_y = coordinates_center(rope)
        vertical_connection = (
            rope[1] <= platform_y2 + ROPE_PLATFORM_VERTICAL_TOLERANCE
            and rope[3] >= platform_y1 - ROPE_PLATFORM_VERTICAL_TOLERANCE
        )
        horizontal_connection = (
            platform_x1 - ROPE_PLATFORM_HORIZONTAL_MARGIN
            <= rope_x
            <= platform_x2 + ROPE_PLATFORM_HORIZONTAL_MARGIN
        )
        if vertical_connection and horizontal_connection:
            connected.append(rope)

    if not connected:
        return None
    player_x, _player_y = coordinates_center(player)
    return min(connected, key=lambda rope: abs(coordinates_center(rope)[0] - player_x))


class MonsterMotionTrack:
    def __init__(
        self,
        track_id: int,
        class_name: str,
        coordinates: tuple[int, int, int, int],
        now: float,
    ) -> None:
        center_x, center_y = coordinates_center(coordinates)
        self.track_id = track_id
        self.class_name = class_name
        self.coordinates = coordinates
        self.last_seen_at = now
        self.history: deque[tuple[float, float, float]] = deque(
            [(now, center_x, center_y)]
        )
        self.confirmed_moving = False


class MovingMonsterFilter:
    """Only accept YOLO monster tracks after observing real displacement."""

    def __init__(self) -> None:
        self.tracks: dict[int, MonsterMotionTrack] = {}
        self.next_track_id = 1

    def update(
        self,
        detections: list[tuple[int, tuple[int, int, int, int], str]],
        now: float,
    ) -> set[int]:
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if now - track.last_seen_at <= MONSTER_TRACK_MAX_MISSING
        }

        candidate_matches = []
        for detection_position, (_result_index, coordinates, class_name) in enumerate(detections):
            center_x, center_y = coordinates_center(coordinates)
            for track_id, track in self.tracks.items():
                if track.class_name != class_name:
                    continue
                track_x, track_y = coordinates_center(track.coordinates)
                distance = hypot(center_x - track_x, center_y - track_y)
                if distance <= MONSTER_TRACK_MATCH_DISTANCE:
                    candidate_matches.append((distance, detection_position, track_id))

        detection_to_track: dict[int, int] = {}
        matched_track_ids = set()
        for _distance, detection_position, track_id in sorted(candidate_matches):
            if detection_position in detection_to_track or track_id in matched_track_ids:
                continue
            detection_to_track[detection_position] = track_id
            matched_track_ids.add(track_id)

        accepted_result_indices = set()
        for detection_position, (result_index, coordinates, class_name) in enumerate(detections):
            track_id = detection_to_track.get(detection_position)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = MonsterMotionTrack(
                    track_id,
                    class_name,
                    coordinates,
                    now,
                )
            else:
                self._update_track(self.tracks[track_id], coordinates, now)

            if self.tracks[track_id].confirmed_moving:
                accepted_result_indices.add(result_index)

        return accepted_result_indices

    @staticmethod
    def _update_track(
        track: MonsterMotionTrack,
        coordinates: tuple[int, int, int, int],
        now: float,
    ) -> None:
        center_x, center_y = coordinates_center(coordinates)
        track.coordinates = coordinates
        track.last_seen_at = now
        track.history.append((now, center_x, center_y))

        cutoff = now - MONSTER_MOTION_WINDOW
        while len(track.history) > 1 and track.history[0][0] < cutoff:
            track.history.popleft()

        if track.confirmed_moving or len(track.history) < MONSTER_MOTION_MIN_OBSERVATIONS:
            return

        _first_time, first_x, first_y = track.history[0]
        moved_samples = sum(
            hypot(sample_x - first_x, sample_y - first_y)
            >= MONSTER_MIN_MOTION_PIXELS
            for _sample_time, sample_x, sample_y in list(track.history)[1:]
        )
        if moved_samples >= MONSTER_MOTION_CONFIRM_SAMPLES:
            track.confirmed_moving = True


class TwoLayerCombatController:
    """Combat, coin pickup, and navigation state machine for the two-layer map."""

    TIMED_STATES = frozenset(
        {
            "unstick_move",
            "drop_prepare",
            "drop_jump",
            "climb_prepare",
            "climb_up",
            "rope_escape_prepare",
            "rope_escape_jump",
        }
    )
    ROPE_HANG_SUPPRESSED_STATES = frozenset(
        {
            "rope_search",
            "rope_align_coarse",
            "rope_align_fine",
            "rope_align_confirm",
            "climb_prepare",
            "climb_up",
        }
    )

    def __init__(self, inputs: DirectInputController, started_at: float) -> None:
        self.inputs = inputs
        self.started_at = started_at
        self.state = "idle"
        self.state_started_at = started_at
        self.last_attack_at = float("-inf")
        self.combat_started_at: float | None = None
        self.facing_direction = RIGHT_KEY
        self.unstick_direction: str | None = None
        self.patrol_direction: str | None = None
        self.patrol_started_at: float | None = None
        self.rope_search_direction: str | None = None
        self.rope_search_direction_started_at = 0.0
        self.rope_centered_at: float | None = None
        self.alignment_rope: tuple[int, int, int, int] | None = None
        self.last_rope_align_tap_at = float("-inf")
        self.climb_alt_started_at = 0.0
        self.climb_up_started_at = 0.0
        self.rope_hang_started_at: float | None = None
        self.rope_last_seen_at = float("-inf")
        self.rope_escape_direction = RIGHT_KEY
        self.target_platform: tuple[int, int, int, int] | None = None
        self.target_rope: tuple[int, int, int, int] | None = None
        self.current_player_platform: tuple[int, int, int, int] | None = None
        self.target_upper_snake_count = 0
        self.current_layer = "unknown"
        self.nearby_monster_count = 0
        self.attack_range_monster_count = 0
        self.upper_center_platform: tuple[int, int, int, int] | None = None
        self.upper_platform_had_monsters = False
        self.upper_center_return_required = False
        self.upper_clear_started_at: float | None = None
        self.player_motion_anchor: tuple[float, float] | None = None
        self.player_last_moved_at = started_at
        self.upper_coin_sweep_active = False
        self.upper_coin_sweep_direction: str | None = None
        self.upper_coin_last_seen_at = float("-inf")
        self.upper_coin_has_reversed = False
        self.upper_coin_reverse_started_at: float | None = None
        self.coin_target_coordinates: tuple[int, int, int, int] | None = None
        self.coin_target_direction: str | None = None
        self.coin_target_started_at: float | None = None
        self.coin_target_last_seen_at = float("-inf")
        self.coin_pickup_pass_started_at: float | None = None
        self.recently_passed_coin: tuple[int, int, int, int] | None = None
        self.coin_reacquire_block_until = float("-inf")

    def set_state(self, state: str, now: float) -> None:
        self.state = state
        self.state_started_at = now

    def reset_patrol(self) -> None:
        self.patrol_direction = None
        self.patrol_started_at = None

    def reset_rope_search(self) -> None:
        self.rope_search_direction = None
        self.rope_search_direction_started_at = 0.0

    def reset_rope_alignment(self) -> None:
        self.rope_centered_at = None
        self.alignment_rope = None

    def reset_rope_hang_tracking(self) -> None:
        self.rope_hang_started_at = None
        self.rope_last_seen_at = float("-inf")

    def reset_player_motion_tracking(self, now: float) -> None:
        self.player_motion_anchor = None
        self.player_last_moved_at = now

    def player_has_been_stationary(
        self,
        now: float,
        player: tuple[int, int, int, int],
    ) -> bool:
        player_center = coordinates_center(player)
        if self.player_motion_anchor is None:
            self.player_motion_anchor = player_center
            self.player_last_moved_at = now
            return False

        movement = hypot(
            player_center[0] - self.player_motion_anchor[0],
            player_center[1] - self.player_motion_anchor[1],
        )
        if movement >= PLAYER_STATIONARY_MOVEMENT_THRESHOLD:
            self.player_motion_anchor = player_center
            self.player_last_moved_at = now
            return False
        return now - self.player_last_moved_at >= PLAYER_STATIONARY_TIMEOUT

    def reset_free_navigation(self) -> None:
        self.reset_patrol()
        self.reset_rope_search()
        self.reset_rope_alignment()

    def reset_upper_coin_sweep(self) -> None:
        self.upper_coin_sweep_active = False
        self.upper_coin_sweep_direction = None
        self.upper_coin_last_seen_at = float("-inf")
        self.upper_coin_has_reversed = False
        self.upper_coin_reverse_started_at = None

    def reset_upper_clear_confirmation(self) -> None:
        self.upper_clear_started_at = None

    def confirm_upper_clear_and_drop(
        self,
        now: float,
        reason: str | None = None,
    ) -> str:
        self.set_walking_direction(now, None)
        if self.upper_clear_started_at is None:
            self.upper_clear_started_at = now

        elapsed = now - self.upper_clear_started_at
        if elapsed < UPPER_CLEAR_CONFIRM_DURATION:
            self.set_state("upper_clear_confirm", now)
            prefix = "" if reason is None else f"{reason}; "
            return (
                f"{prefix}UPPER: CLEAR CONFIRM "
                f"{elapsed:.1f}/{UPPER_CLEAR_CONFIRM_DURATION:.1f}s"
            )

        self.reset_upper_clear_confirmation()
        status = self.begin_drop(now)
        return status if reason is None else f"{reason}; {status}"

    def reset_coin_target(self, clear_recent: bool = False) -> None:
        self.coin_target_coordinates = None
        self.coin_target_direction = None
        self.coin_target_started_at = None
        self.coin_target_last_seen_at = float("-inf")
        self.coin_pickup_pass_started_at = None
        if clear_recent:
            self.recently_passed_coin = None
            self.coin_reacquire_block_until = float("-inf")

    def start_coin_target(
        self,
        now: float,
        coordinates: tuple[int, int, int, int],
        direction: str,
    ) -> None:
        self.coin_target_coordinates = coordinates
        self.coin_target_direction = direction
        self.coin_target_started_at = now
        self.coin_target_last_seen_at = now
        self.coin_pickup_pass_started_at = None

    def available_coin_targets(
        self,
        now: float,
        coins: list[tuple[tuple[int, int, int, int], str]],
    ) -> list[tuple[tuple[int, int, int, int], str]]:
        if now >= self.coin_reacquire_block_until:
            self.recently_passed_coin = None
            return coins
        if self.recently_passed_coin is None:
            return coins

        passed_x, passed_y = coordinates_center(self.recently_passed_coin)
        return [
            coin
            for coin in coins
            if hypot(
                coordinates_center(coin[0])[0] - passed_x,
                coordinates_center(coin[0])[1] - passed_y,
            )
            > COIN_REACQUIRE_IGNORE_DISTANCE
        ]

    def match_locked_coin(
        self,
        coins: list[tuple[tuple[int, int, int, int], str]],
    ) -> tuple[tuple[int, int, int, int], str] | None:
        if self.coin_target_coordinates is None or not coins:
            return None

        target_x, target_y = coordinates_center(self.coin_target_coordinates)
        matched = min(
            coins,
            key=lambda coin: hypot(
                coordinates_center(coin[0])[0] - target_x,
                coordinates_center(coin[0])[1] - target_y,
            ),
        )
        matched_x, matched_y = coordinates_center(matched[0])
        if hypot(matched_x - target_x, matched_y - target_y) > COIN_TARGET_MATCH_DISTANCE:
            return None
        return matched

    def continue_coin_pickup_pass(self, now: float) -> str | None:
        if (
            self.coin_pickup_pass_started_at is None
            or self.coin_target_direction is None
        ):
            return None

        elapsed = now - self.coin_pickup_pass_started_at
        direction = self.coin_target_direction
        if elapsed < COIN_PICKUP_PASS_DURATION:
            self.set_walking_direction(now, direction)
            self.facing_direction = direction
            self.set_state("coin_pickup_pass", now)
            return (
                f"NESO: PICKUP PASS {direction.upper()} "
                f"{elapsed:.1f}/{COIN_PICKUP_PASS_DURATION:.1f}s"
            )

        self.set_walking_direction(now, None)
        self.recently_passed_coin = self.coin_target_coordinates
        self.coin_reacquire_block_until = now + COIN_REACQUIRE_COOLDOWN
        self.reset_coin_target()
        self.set_state("coin_pickup_pass_complete", now)
        return "NESO: PICKUP PASS COMPLETED"

    def set_walking_direction(self, now: float, direction: str | None) -> bool:
        """Apply horizontal movement immediately on either map layer."""
        self.inputs.set_horizontal(direction)
        return True

    def tap_walking_direction(self, now: float, direction: str) -> bool:
        """Tap a horizontal direction without a reversal delay."""
        return self.inputs.tap(direction)

    def abort(self, now: float) -> None:
        self.inputs.release_navigation()
        self.set_state("idle", now)
        self.combat_started_at = None
        self.unstick_direction = None
        self.reset_free_navigation()
        self.reset_rope_hang_tracking()
        self.reset_player_motion_tracking(now)
        self.reset_upper_coin_sweep()
        self.reset_upper_clear_confirmation()
        self.reset_coin_target(clear_recent=True)

    def observe_rope_hang(self, now: float, on_or_near_rope: bool) -> bool:
        if self.state in {"rope_escape_prepare", "rope_escape_jump"}:
            self.reset_rope_hang_tracking()
            return False

        if on_or_near_rope:
            self.rope_last_seen_at = now
            if self.rope_hang_started_at is None:
                self.rope_hang_started_at = now
            return now - self.rope_hang_started_at >= ROPE_HANG_TIMEOUT

        if now - self.rope_last_seen_at > ROPE_HANG_DETECTION_GRACE:
            self.reset_rope_hang_tracking()
        return False

    def begin_unstick(self, now: float) -> str:
        self.inputs.release_navigation()
        self.reset_free_navigation()
        self.reset_coin_target()
        self.unstick_direction = random.choice((LEFT_KEY, RIGHT_KEY))
        self.inputs.key_down(self.unstick_direction)
        self.facing_direction = self.unstick_direction
        self.combat_started_at = None
        self.set_state("unstick_move", now)
        return f"UNSTICK: {self.unstick_direction.upper()} for {UNSTICK_MOVE_DURATION:.1f}s"

    def begin_drop(self, now: float) -> str:
        self.inputs.release_navigation()
        self.reset_free_navigation()
        self.reset_upper_coin_sweep()
        self.reset_upper_clear_confirmation()
        self.reset_coin_target()
        self.inputs.key_down(DOWN_KEY)
        self.set_state("drop_prepare", now)
        return "UPPER: DROP PREPARE (DOWN)"

    def begin_climb(self, now: float) -> str:
        self.inputs.release_navigation()
        self.reset_patrol()
        self.reset_rope_search()
        self.reset_rope_alignment()
        self.reset_coin_target()
        self.inputs.key_down(JUMP_KEY)
        self.climb_alt_started_at = now
        self.set_state("climb_prepare", now)
        return "LOWER: CLIMB PREPARE (ALT)"

    def begin_rope_escape(self, now: float) -> str:
        self.inputs.release_navigation()
        self.reset_free_navigation()
        self.reset_coin_target()
        self.reset_rope_hang_tracking()
        self.reset_player_motion_tracking(now)
        self.rope_escape_direction = random.choice((LEFT_KEY, RIGHT_KEY))
        self.inputs.key_down(self.rope_escape_direction)
        self.facing_direction = self.rope_escape_direction
        self.set_state("rope_escape_prepare", now)
        return f"ROPE ESCAPE: HOLD {self.rope_escape_direction.upper()}"

    def continue_timed_action(self, now: float) -> str | None:
        elapsed = now - self.state_started_at

        if self.state == "unstick_move":
            if elapsed >= UNSTICK_MOVE_DURATION:
                if self.unstick_direction is not None:
                    self.inputs.key_up(self.unstick_direction)
                self.unstick_direction = None
                self.set_state("idle", now)
                return "UNSTICK: COMPLETED"
            return f"UNSTICK: MOVING {self.unstick_direction.upper()}"

        if self.state == "drop_prepare":
            if elapsed >= DROP_DOWN_HOLD_DURATION:
                self.inputs.key_down(JUMP_KEY)
                self.set_state("drop_jump", now)
                return "UPPER: DROP JUMP (DOWN + ALT)"
            return "UPPER: DROP PREPARE (DOWN)"

        if self.state == "drop_jump":
            if elapsed >= DROP_ALT_HOLD_DURATION:
                self.inputs.key_up(JUMP_KEY)
                self.inputs.key_up(DOWN_KEY)
                self.set_state("idle", now)
                return "UPPER: DROP COMPLETED"
            return "UPPER: DROP JUMP (DOWN + ALT)"

        if self.state == "climb_prepare":
            if elapsed >= CLIMB_JUMP_LEAD_DURATION:
                self.inputs.key_down(UP_KEY)
                self.climb_up_started_at = now
                self.set_state("climb_up", now)
                return "LOWER: CLIMBING UP"
            return "LOWER: CLIMB PREPARE (ALT)"

        if self.state == "climb_up":
            if now - self.climb_alt_started_at >= CLIMB_ALT_HOLD_DURATION:
                self.inputs.key_up(JUMP_KEY)
            if now - self.climb_up_started_at >= CLIMB_UP_DURATION:
                self.inputs.key_up(UP_KEY)
                self.inputs.key_up(JUMP_KEY)
                self.set_state("idle", now)
                return "LOWER: CLIMB COMPLETED"
            return (
                "LOWER: CLIMBING UP "
                f"{now - self.climb_up_started_at:.1f}/{CLIMB_UP_DURATION:.1f}s"
            )

        if self.state == "rope_escape_prepare":
            if elapsed >= ROPE_ESCAPE_DIRECTION_HOLD_DURATION:
                self.inputs.key_down(JUMP_KEY)
                self.set_state("rope_escape_jump", now)
                return f"ROPE ESCAPE: {self.rope_escape_direction.upper()} + ALT"
            return f"ROPE ESCAPE: HOLD {self.rope_escape_direction.upper()}"

        if self.state == "rope_escape_jump":
            if elapsed >= ROPE_ESCAPE_JUMP_DURATION:
                self.inputs.key_up(JUMP_KEY)
                self.inputs.key_up(self.rope_escape_direction)
                self.set_state("idle", now)
                return "ROPE ESCAPE: COMPLETED"
            return f"ROPE ESCAPE: {self.rope_escape_direction.upper()} + ALT"

        return None

    def search_for_rope(self, now: float) -> str:
        self.reset_patrol()
        self.reset_rope_alignment()
        if self.rope_search_direction is None:
            self.rope_search_direction = random.choice((LEFT_KEY, RIGHT_KEY))
            self.rope_search_direction_started_at = now
        elif (
            now - self.rope_search_direction_started_at
            >= ROPE_SEARCH_DIRECTION_DURATION
        ):
            self.rope_search_direction = (
                RIGHT_KEY if self.rope_search_direction == LEFT_KEY else LEFT_KEY
            )
            self.rope_search_direction_started_at = now

        self.set_walking_direction(now, self.rope_search_direction)
        self.facing_direction = self.rope_search_direction
        self.set_state("rope_search", now)
        return f"LOWER: SEARCH ROPE {self.rope_search_direction.upper()}"

    def align_and_climb(
        self,
        now: float,
        player: tuple[int, int, int, int],
        rope: tuple[int, int, int, int],
    ) -> str:
        self.reset_patrol()
        self.reset_rope_search()
        previous_rope_x = (
            None
            if self.alignment_rope is None
            else coordinates_center(self.alignment_rope)[0]
        )
        rope_x, _rope_y = coordinates_center(rope)
        if (
            previous_rope_x is None
            or abs(rope_x - previous_rope_x)
            > ROPE_ALIGNMENT_TARGET_X_TOLERANCE
        ):
            self.reset_rope_alignment()
        self.alignment_rope = rope

        player_x, _player_y = coordinates_center(player)
        gap = rope_x - player_x
        absolute_gap = abs(gap)
        direction = LEFT_KEY if gap < 0 else RIGHT_KEY

        if absolute_gap > ROPE_FINE_ALIGNMENT_DISTANCE:
            self.rope_centered_at = None
            self.set_walking_direction(now, direction)
            self.facing_direction = direction
            self.set_state("rope_align_coarse", now)
            return f"LOWER: ALIGN ROPE HOLD {direction.upper()} ({absolute_gap:.0f}px)"

        self.set_walking_direction(now, None)
        if absolute_gap <= ROPE_CENTER_ENTER_TOLERANCE:
            if self.rope_centered_at is None:
                self.rope_centered_at = now
            centered_for = now - self.rope_centered_at
            if centered_for >= ROPE_CENTER_CONFIRM_DURATION:
                return self.begin_climb(now)
            self.set_state("rope_align_confirm", now)
            return (
                "LOWER: ROPE CENTER CONFIRM "
                f"{centered_for:.1f}/{ROPE_CENTER_CONFIRM_DURATION:.1f}s"
            )

        if (
            self.rope_centered_at is not None
            and absolute_gap <= ROPE_CENTER_EXIT_TOLERANCE
        ):
            centered_for = now - self.rope_centered_at
            if centered_for >= ROPE_CENTER_CONFIRM_DURATION:
                return self.begin_climb(now)
            self.set_state("rope_align_confirm", now)
            return (
                "LOWER: ROPE CENTER HOLD "
                f"{centered_for:.1f}/{ROPE_CENTER_CONFIRM_DURATION:.1f}s"
            )

        self.rope_centered_at = None
        if now - self.last_rope_align_tap_at >= ROPE_ALIGN_TAP_INTERVAL:
            if self.tap_walking_direction(now, direction):
                self.last_rope_align_tap_at = now
                self.facing_direction = direction
        self.set_state("rope_align_fine", now)
        return f"LOWER: ALIGN ROPE TAP {direction.upper()} ({absolute_gap:.0f}px)"

    def patrol_lower_layer(
        self,
        now: float,
        player: tuple[int, int, int, int],
        monsters: list[tuple[tuple[int, int, int, int], str]],
    ) -> str:
        self.reset_upper_coin_sweep()
        self.reset_rope_search()
        self.reset_rope_alignment()
        player_x, _player_y = coordinates_center(player)
        monsters_on_left = sum(
            coordinates_center(coordinates)[0] < player_x
            for coordinates, _class_name in monsters
        )
        monsters_on_right = sum(
            coordinates_center(coordinates)[0] > player_x
            for coordinates, _class_name in monsters
        )

        def choose_direction() -> str:
            if monsters_on_left > monsters_on_right:
                return LEFT_KEY
            if monsters_on_right > monsters_on_left:
                return RIGHT_KEY
            return random.choice((LEFT_KEY, RIGHT_KEY))

        if self.patrol_direction is None or self.patrol_started_at is None:
            self.patrol_direction = choose_direction()
            self.patrol_started_at = now
        elif now - self.patrol_started_at >= LOWER_PATROL_DIRECTION_DURATION:
            self.patrol_direction = choose_direction()
            self.patrol_started_at = now

        self.set_walking_direction(now, self.patrol_direction)
        self.facing_direction = self.patrol_direction
        self.set_state("lower_patrol", now)
        elapsed = max(0.0, now - self.patrol_started_at)
        return (
            f"LOWER: PATROL {self.patrol_direction.upper()} "
            f"{elapsed:.1f}/{LOWER_PATROL_DIRECTION_DURATION:.1f}s "
            f"(MONSTERS L{monsters_on_left}/R{monsters_on_right})"
        )

    def choose_upper_coin_target(
        self,
        now: float,
        player: tuple[int, int, int, int],
        coins: list[tuple[tuple[int, int, int, int], str]],
    ) -> tuple[tuple[int, int, int, int], str]:
        """Clear one side first, then explicitly reverse for remaining neso."""
        player_x, _player_y = coordinates_center(player)
        if not self.upper_coin_sweep_active:
            nearest_coordinates, _nearest_name = nearest_target(player, coins)
            initial_direction = horizontal_direction_to(player, nearest_coordinates)
            self.upper_coin_sweep_active = True
            self.upper_coin_sweep_direction = (
                self.facing_direction
                if initial_direction is None
                else initial_direction
            )
            self.upper_coin_has_reversed = False
            self.upper_coin_reverse_started_at = None

        self.upper_coin_last_seen_at = now
        direction = self.upper_coin_sweep_direction or self.facing_direction
        centered_coins = [
            coin
            for coin in coins
            if abs(coordinates_center(coin[0])[0] - player_x)
            <= COIN_PICKUP_X_TOLERANCE
        ]
        if centered_coins:
            target_coordinates, _target_name = nearest_target(player, centered_coins)
            return target_coordinates, direction

        coins_ahead = [
            coin
            for coin in coins
            if (
                coordinates_center(coin[0])[0] > player_x
                if direction == RIGHT_KEY
                else coordinates_center(coin[0])[0] < player_x
            )
        ]
        if coins_ahead:
            target_coordinates, _target_name = nearest_target(player, coins_ahead)
            return target_coordinates, direction

        direction = RIGHT_KEY if direction == LEFT_KEY else LEFT_KEY
        self.upper_coin_sweep_direction = direction
        self.upper_coin_has_reversed = True
        self.upper_coin_reverse_started_at = now
        target_coordinates, _target_name = nearest_target(player, coins)
        return target_coordinates, direction


    def continue_upper_coin_sweep(self, now: float) -> str | None:
        if not self.upper_coin_sweep_active or self.upper_coin_sweep_direction is None:
            return None

        time_without_coin = now - self.upper_coin_last_seen_at
        if time_without_coin <= COIN_DETECTION_GRACE_DURATION:
            self.set_walking_direction(now, self.upper_coin_sweep_direction)
            self.facing_direction = self.upper_coin_sweep_direction
            self.set_state("upper_coin_detection_grace", now)
            return (
                f"NESO: CONFIRM {self.upper_coin_sweep_direction.upper()} SIDE CLEAR"
            )

        if not self.upper_coin_has_reversed:
            self.upper_coin_sweep_direction = (
                RIGHT_KEY
                if self.upper_coin_sweep_direction == LEFT_KEY
                else LEFT_KEY
            )
            self.upper_coin_has_reversed = True
            self.upper_coin_reverse_started_at = now

        reverse_started_at = (
            now
            if self.upper_coin_reverse_started_at is None
            else self.upper_coin_reverse_started_at
        )
        reverse_elapsed = now - reverse_started_at
        if reverse_elapsed < UPPER_COIN_REVERSE_SEARCH_DURATION:
            self.set_walking_direction(now, self.upper_coin_sweep_direction)
            self.facing_direction = self.upper_coin_sweep_direction
            self.set_state("upper_coin_reverse_search", now)
            return (
                f"NESO: REVERSE SEARCH {self.upper_coin_sweep_direction.upper()} "
                f"{reverse_elapsed:.1f}/{UPPER_COIN_REVERSE_SEARCH_DURATION:.1f}s"
            )

        self.set_walking_direction(now, None)
        self.reset_upper_coin_sweep()
        return None

    def return_to_upper_platform_center(
        self,
        now: float,
        player: tuple[int, int, int, int],
        platform_monsters: list[
            tuple[tuple[int, int, int, int], str]
        ],
    ) -> str | None:
        if (
            not platform_monsters
            or not self.upper_center_return_required
            or self.current_player_platform is None
        ):
            return None

        player_x, _player_y = coordinates_center(player)
        platform_x1, _platform_y1, platform_x2, _platform_y2 = (
            self.current_player_platform
        )
        platform_x, _platform_y = coordinates_center(self.current_player_platform)
        platform_width = max(1, platform_x2 - platform_x1)
        combat_half_width = (
            platform_width * UPPER_PLATFORM_COMBAT_REGION_RATIO / 2
        )
        center_gap = platform_x - player_x
        if abs(center_gap) <= combat_half_width:
            self.upper_center_return_required = False
            return None

        direction = LEFT_KEY if center_gap < 0 else RIGHT_KEY
        self.reset_free_navigation()
        self.set_walking_direction(now, direction)
        self.facing_direction = direction
        self.set_state("upper_return_center", now)
        return (
            f"UPPER: RETURN CENTER {direction.upper()} "
            f"({abs(center_gap):.0f}px)"
        )

    def update(
        self,
        now: float,
        player: tuple[int, int, int, int] | None,
        monsters: list[tuple[tuple[int, int, int, int], str]],
        coins: list[tuple[tuple[int, int, int, int], str]],
        platforms: list[tuple[int, int, int, int]],
        ropes: list[tuple[int, int, int, int]],
        frame_shape: tuple[int, ...],
        critical_hp_active: bool = False,
    ) -> str:
        self.target_platform = None
        self.target_rope = None
        self.current_player_platform = None
        self.target_upper_snake_count = 0
        self.current_layer = "unknown" if player is None else player_layer(player)
        self.nearby_monster_count = 0
        self.attack_range_monster_count = 0

        if not self.inputs.has_focus():
            self.abort(now)
            return "PAUSED: GAME WINDOW NOT FOCUSED"

        self.current_player_platform = (
            None
            if player is None
            else find_player_platform(player, platforms)
        )
        rope_hang_detection_allowed = (
            self.state not in self.ROPE_HANG_SUPPRESSED_STATES
            and self.current_player_platform is None
        )
        hanging_on_rope = (
            player is not None
            and rope_hang_detection_allowed
            and player_is_hanging_on_rope(player, ropes)
        )
        rope_hang_timed_out = self.observe_rope_hang(now, hanging_on_rope)

        timed_status = self.continue_timed_action(now)
        if timed_status is not None:
            return timed_status

        if critical_hp_active:
            self.inputs.release_navigation()
            self.combat_started_at = None
            self.reset_coin_target()
            self.reset_upper_coin_sweep()
            self.reset_upper_clear_confirmation()
            self.reset_player_motion_tracking(now)
            self.set_state("critical_hp_healing", now)
            return "CRITICAL HP: ATTACK PAUSED; HEALING WITH E"

        if now - self.started_at < AUTOMATION_START_DELAY:
            self.inputs.release_navigation()
            remaining = AUTOMATION_START_DELAY - (now - self.started_at)
            return f"AUTOMATION STARTING IN {remaining:.1f}s"

        if player is None:
            self.combat_started_at = None
            self.reset_coin_target()
            self.reset_upper_coin_sweep()
            self.reset_upper_clear_confirmation()
            self.reset_free_navigation()
            self.reset_player_motion_tracking(now)
            self.set_walking_direction(now, RIGHT_KEY)
            self.facing_direction = RIGHT_KEY
            self.set_state("player_search_right", now)
            return "PLAYER NOT FOUND: MOVE RIGHT"

        if self.player_has_been_stationary(now, player):
            status = self.begin_rope_escape(now)
            return f"STATIONARY {PLAYER_STATIONARY_TIMEOUT:.0f}s; {status}"

        if rope_hang_timed_out:
            return self.begin_rope_escape(now)

        attack_range = build_attack_range(player, frame_shape)
        nearby_monsters = same_platform_nearby_monsters(
            player,
            monsters,
            platforms,
        )
        monsters_in_attack_range = [
            monster
            for monster in nearby_monsters
            if box_center_is_inside(monster[0], attack_range)
        ]
        self.nearby_monster_count = len(nearby_monsters)
        self.attack_range_monster_count = len(monsters_in_attack_range)
        previous_state = self.state

        if self.current_layer != "upper" or nearby_monsters:
            self.reset_upper_clear_confirmation()

        current_platform_monsters = (
            monsters_on_platform(
                self.current_player_platform,
                monsters,
            )
            if self.current_layer == "upper"
            and self.current_player_platform is not None
            else []
        )
        if self.current_layer == "upper" and self.current_player_platform is not None:
            platform_changed = (
                self.current_player_platform != self.upper_center_platform
            )
            has_platform_monsters = bool(current_platform_monsters)
            if has_platform_monsters and (
                platform_changed or not self.upper_platform_had_monsters
            ):
                self.upper_center_return_required = True
            if not has_platform_monsters:
                self.upper_center_return_required = False
            self.upper_center_platform = self.current_player_platform
            self.upper_platform_had_monsters = has_platform_monsters
        else:
            self.upper_center_platform = None
            self.upper_platform_had_monsters = False
            self.upper_center_return_required = False

        if current_platform_monsters:
            self.reset_coin_target()

        center_status = self.return_to_upper_platform_center(
            now,
            player,
            current_platform_monsters,
        )
        if center_status is not None:
            return center_status

        if monsters_in_attack_range:
            self.inputs.release_navigation()
            self.reset_free_navigation()
            self.reset_coin_target()
            target_coordinates, target_name = nearest_target(
                player,
                monsters_in_attack_range,
            )
            if self.combat_started_at is None:
                self.combat_started_at = now
            elif now - self.combat_started_at >= STUCK_ATTACK_DURATION:
                return self.begin_unstick(now)

            direction = horizontal_direction_to(player, target_coordinates)
            if direction is None:
                direction = self.facing_direction
            attack_key = ATTACK_KEY_BY_MONSTER[target_name]
            if now - self.last_attack_at >= ATTACK_INTERVAL:
                if self.inputs.tap_attack(direction, attack_key):
                    self.last_attack_at = now
                    self.facing_direction = direction
            self.set_state("combat", now)
            return (
                f"COMBAT: {target_name.upper()} WITH {attack_key.upper()} "
                f"FACING {direction.upper()}"
            )

        self.combat_started_at = None

        if current_platform_monsters:
            if previous_state == "combat":
                self.upper_center_return_required = True
            center_status = self.return_to_upper_platform_center(
                now,
                player,
                current_platform_monsters,
            )
            if center_status is not None:
                return center_status

        if nearby_monsters:
            self.reset_free_navigation()
            self.reset_coin_target()
            target_coordinates, target_name = nearest_target(player, nearby_monsters)
            direction = horizontal_direction_to(player, target_coordinates)
            if direction is None:
                direction = self.facing_direction
            self.set_walking_direction(now, direction)
            self.facing_direction = direction
            self.set_state("chase_monster", now)
            return (
                f"CHASE: {target_name.upper()} {direction.upper()} "
                f"({horizontal_distance(player, target_coordinates):.0f}px)"
            )

        pickup_pass_status = self.continue_coin_pickup_pass(now)
        if pickup_pass_status is not None:
            return pickup_pass_status

        detected_same_layer_coins = (
            monsters_on_platform(
                self.current_player_platform,
                coins,
            )
            if self.current_player_platform is not None
            else [
                coin
                for coin in coins
                if coin_is_on_player_level(player, coin[0])
            ]
        )
        if self.current_layer == "upper" and detected_same_layer_coins:
            self.reset_upper_clear_confirmation()
        same_layer_coins = self.available_coin_targets(
            now,
            detected_same_layer_coins,
        )

        locked_coin = self.match_locked_coin(same_layer_coins)
        if self.coin_target_direction is not None and locked_coin is None:
            missing_for = now - self.coin_target_last_seen_at
            direction = self.coin_target_direction
            target_started_at = (
                now
                if self.coin_target_started_at is None
                else self.coin_target_started_at
            )
            pursuit_elapsed = now - target_started_at
            reached_last_known_x = False
            if self.coin_target_coordinates is not None:
                player_x, _player_y = coordinates_center(player)
                target_x, _target_y = coordinates_center(
                    self.coin_target_coordinates
                )
                reached_last_known_x = (
                    player_x <= target_x + COIN_PICKUP_X_TOLERANCE
                    if direction == LEFT_KEY
                    else player_x >= target_x - COIN_PICKUP_X_TOLERANCE
                )

            if reached_last_known_x:
                self.coin_pickup_pass_started_at = now
                self.set_walking_direction(now, direction)
                self.facing_direction = direction
                self.set_state("coin_pickup_pass", now)
                return (
                    f"NESO: LAST POSITION REACHED; PICKUP PASS "
                    f"{direction.upper()} 0.0/{COIN_PICKUP_PASS_DURATION:.1f}s"
                )

            if pursuit_elapsed <= COIN_TARGET_BLIND_PURSUIT_MAX_DURATION:
                self.set_walking_direction(now, direction)
                self.facing_direction = direction
                if missing_for <= COIN_TARGET_LOST_GRACE_DURATION:
                    self.set_state("coin_target_lost_grace", now)
                    return (
                        f"NESO: TARGET TEMPORARILY LOST; KEEP "
                        f"{direction.upper()} {missing_for:.1f}/"
                        f"{COIN_TARGET_LOST_GRACE_DURATION:.1f}s"
                    )
                self.set_state("coin_target_blind_pursuit", now)
                return (
                    f"NESO: BLIND PURSUIT {direction.upper()} "
                    f"{pursuit_elapsed:.1f}/"
                    f"{COIN_TARGET_BLIND_PURSUIT_MAX_DURATION:.1f}s"
                )
            self.reset_coin_target()

        if locked_coin is not None or same_layer_coins:
            self.reset_free_navigation()
            if locked_coin is not None:
                target_coordinates, _target_name = locked_coin
                self.coin_target_coordinates = target_coordinates
                self.coin_target_last_seen_at = now
                direction = self.coin_target_direction or self.facing_direction
            else:
                if self.current_layer == "upper":
                    target_coordinates, direction = self.choose_upper_coin_target(
                        now,
                        player,
                        same_layer_coins,
                    )
                else:
                    self.reset_upper_coin_sweep()
                    target_coordinates, _target_name = nearest_target(
                        player,
                        same_layer_coins,
                    )
                    direction = horizontal_direction_to(player, target_coordinates)
                    if direction is None:
                        direction = self.facing_direction
                self.start_coin_target(now, target_coordinates, direction)

            distance = horizontal_distance(player, target_coordinates)
            desired_direction = horizontal_direction_to(player, target_coordinates)
            target_started_at = (
                now
                if self.coin_target_started_at is None
                else self.coin_target_started_at
            )
            if (
                desired_direction is not None
                and desired_direction != direction
                and now - target_started_at >= COIN_DIRECTION_LOCK_DURATION
            ):
                direction = desired_direction
                self.coin_target_direction = direction
                self.coin_target_started_at = now

            if distance <= COIN_PICKUP_X_TOLERANCE:
                if self.coin_pickup_pass_started_at is None:
                    self.coin_pickup_pass_started_at = now
                self.set_walking_direction(now, direction)
                self.facing_direction = direction
                self.set_state("coin_pickup_pass", now)
                return (
                    f"NESO: PICKUP PASS {direction.upper()} "
                    f"0.0/{COIN_PICKUP_PASS_DURATION:.1f}s"
                )
            self.set_walking_direction(now, direction)
            self.facing_direction = direction
            self.set_state("move_to_coin", now)
            return (
                f"NESO: MOVE {direction.upper()} ({distance:.0f}px)"
                if self.current_layer == "lower"
                else (
                    f"NESO: UPPER SWEEP {direction.upper()} "
                    f"({distance:.0f}px)"
                )
            )

        if self.current_layer == "upper":
            upper_sweep_status = self.continue_upper_coin_sweep(now)
            if upper_sweep_status is not None:
                return upper_sweep_status
        else:
            self.reset_upper_coin_sweep()

        cross_layer_coins = [
            coin for coin in coins if coin_is_on_other_map_layer(player, coin[0])
        ]
        if cross_layer_coins:
            target_coordinates, _target_name = nearest_target(
                player,
                cross_layer_coins,
            )
            if self.current_layer == "upper":
                return self.confirm_upper_clear_and_drop(
                    now,
                    "NESO: LOWER LAYER",
                )

            target_platform = upper_platform_for_target(
                target_coordinates,
                platforms,
            )
            self.target_platform = target_platform
            target_rope = None
            if target_platform is not None:
                self.target_upper_snake_count = len(
                    monsters_on_platform(
                        target_platform,
                        monsters,
                        class_name="snake",
                    )
                )
                target_rope = rope_connected_to_platform(
                    player,
                    target_platform,
                    ropes,
                )
            elif ropes:
                target_x, _target_y = coordinates_center(target_coordinates)
                target_rope = min(
                    ropes,
                    key=lambda rope: abs(coordinates_center(rope)[0] - target_x),
                )

            self.target_rope = target_rope
            if target_rope is None:
                return f"NESO: UPPER LAYER; {self.search_for_rope(now)}"
            return f"NESO: UPPER LAYER; {self.align_and_climb(now, player, target_rope)}"

        if self.current_layer == "upper":
            return self.confirm_upper_clear_and_drop(now)

        target_platform = nearest_upper_platform(player, platforms)
        self.target_platform = target_platform
        if target_platform is not None:
            upper_snakes = monsters_on_platform(
                target_platform,
                monsters,
                class_name="snake",
            )
            self.target_upper_snake_count = len(upper_snakes)
            if len(upper_snakes) >= UPPER_PLATFORM_SNAKE_THRESHOLD:
                target_rope = rope_connected_to_platform(
                    player,
                    target_platform,
                    ropes,
                )
                self.target_rope = target_rope
                if target_rope is None:
                    return self.search_for_rope(now)
                return self.align_and_climb(now, player, target_rope)

        return self.patrol_lower_layer(now, player, monsters)


def choose_nearest_scroll(
    player: tuple[int, int, int, int] | None,
    matches: list[ScrollMatch],
) -> ScrollMatch | None:
    if not matches:
        return None
    if player is None:
        return max(matches, key=lambda match: match.score)

    player_x, player_y = coordinates_center(player)
    return min(
        matches,
        key=lambda match: (
            (coordinates_center(match.coordinates)[0] - player_x) ** 2
            + (coordinates_center(match.coordinates)[1] - player_y) ** 2
        ),
    )


def draw_scroll_matches(
    frame: np.ndarray,
    matches: list[ScrollMatch],
    target: ScrollMatch | None,
) -> None:
    for match in matches:
        x1, y1, x2, y2 = match.coordinates
        is_target = match == target
        color = (0, 255, 255) if is_target else (255, 180, 0)
        thickness = 3 if is_target else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame,
            f"SCROLL {match.name} {match.score:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_navigation_debug(
    frame: np.ndarray,
    attack_range: tuple[int, int, int, int] | None,
    pursuit_range: tuple[int, int, int, int] | None,
    nearby_monsters: list[tuple[tuple[int, int, int, int], str]],
    monsters_in_attack_range: list[
        tuple[tuple[int, int, int, int], str]
    ],
    controller: TwoLayerCombatController,
) -> None:
    frame_height, frame_width = frame.shape[:2]
    if 0 <= LAYER_SPLIT_Y < frame_height:
        cv2.line(
            frame,
            (0, LAYER_SPLIT_Y),
            (frame_width - 1, LAYER_SPLIT_Y),
            (255, 255, 0),
            1,
        )
        cv2.putText(
            frame,
            f"LAYER Y={LAYER_SPLIT_Y}",
            (frame_width - 180, max(18, LAYER_SPLIT_Y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if pursuit_range is not None:
        x1, y1, x2, y2 = pursuit_range
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            frame,
            f"PURSUE {SAME_LAYER_MONSTER_MAX_DISTANCE}px",
            (x1, min(frame_height - 8, y1 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    if attack_range is not None:
        x1, y1, x2, y2 = attack_range
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            frame,
            "ATTACK RANGE",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    attack_coordinates = {
        coordinates for coordinates, _class_name in monsters_in_attack_range
    }
    for coordinates, _class_name in nearby_monsters:
        x1, y1, x2, y2 = coordinates
        color = (0, 255, 0) if coordinates in attack_coordinates else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    if controller.current_player_platform is not None:
        x1, y1, x2, y2 = controller.current_player_platform
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 3)
        cv2.putText(
            frame,
            "CURRENT PLATFORM",
            (x1, min(frame_height - 8, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 100, 0),
            2,
            cv2.LINE_AA,
        )

    if controller.target_platform is not None:
        x1, y1, x2, y2 = controller.target_platform
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
        cv2.putText(
            frame,
            f"TARGET PLATFORM: {controller.target_upper_snake_count} SNAKES",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if controller.target_rope is not None:
        x1, y1, x2, y2 = controller.target_rope
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 4)


def draw_player_labels(
    frame: np.ndarray,
    players: list[tuple[tuple[int, int, int, int], bool]],
    font: ImageFont.FreeTypeFont,
) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    for (x1, y1, x2, y2), is_self in players:
        color = (0, 255, 0) if is_self else (255, 255, 0)
        line_width = 4 if is_self else 2
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        draw.text(
            (x1, max(0, y1 - 29)),
            "自己" if is_self else "玩家",
            font=font,
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def main() -> None:
    pydirectinput.PAUSE = 0
    enable_dpi_awareness()
    hwnd, title = choose_window()
    using_cpu_inference = str(INFERENCE_DEVICE).lower() == "cpu"
    reapply_cpu_threads_after_predictor_setup = False
    if using_cpu_inference:
        if CPU_INFERENCE_THREADS < 1:
            raise ValueError("CPU_INFERENCE_THREADS must be at least 1")
        torch.set_num_threads(CPU_INFERENCE_THREADS)
        reapply_cpu_threads_after_predictor_setup = True
    model = YOLO(str(MODEL_PATH))
    missing_confidence_classes = sorted(
        set(model.names.values()) - set(CLASS_CONFIDENCE_THRESHOLDS)
    )
    if missing_confidence_classes:
        raise RuntimeError(
            "Missing confidence thresholds for model classes: "
            + ", ".join(missing_confidence_classes)
        )
    invalid_confidence_thresholds = {
        class_name: threshold
        for class_name, threshold in CLASS_CONFIDENCE_THRESHOLDS.items()
        if not 0.0 <= threshold <= 1.0
    }
    if invalid_confidence_thresholds:
        raise RuntimeError(
            "Class confidence thresholds must be between 0.0 and 1.0: "
            + ", ".join(
                f"{name}={threshold}"
                for name, threshold in invalid_confidence_thresholds.items()
            )
        )
    class_confidence_by_id = {
        class_id: CLASS_CONFIDENCE_THRESHOLDS[class_name]
        for class_id, class_name in model.names.items()
    }
    minimum_model_confidence = min(class_confidence_by_id.values())
    monster_motion_filter = MovingMonsterFilter()
    scroll_detector = ScrollDetector(
        load_scroll_templates(SCROLL_TEMPLATE_DIRECTORY)
    )
    ocr = RapidOCR()
    ocr_font = ImageFont.truetype(str(OCR_FONT_PATH), 24)
    player_class_id = next(class_id for class_id, name in model.names.items() if name == "player")
    ignored_class_ids = {
        class_id
        for class_id, name in model.names.items()
        if name in IGNORED_CLASS_NAMES
    }
    last_ocr_time = 0.0
    self_center: tuple[float, float] | None = None
    last_resource_ocr_time = 0.0
    last_terminal_status_time = float("-inf")
    hp_stat: tuple[int, int] | None = None
    mp_stat: tuple[int, int] | None = None
    critical_hp_active = False
    critical_hp_started_at: float | None = None
    next_critical_hp_heal_at = 0.0
    next_critical_hp_fallback_at = 0.0
    print(f"偵測視窗：{title}")
    print(
        f"自動補給：每 {RESOURCE_OCR_INTERVAL} 秒檢查，"
        f"HP <= {CRITICAL_HP_THRESHOLD} 時每 "
        f"{CRITICAL_HP_HEAL_INTERVAL} 秒按 {CRITICAL_HP_HEAL_KEY.upper()}，"
        f"{CRITICAL_HP_FALLBACK_DELAY} 秒後仍未恢復則按 {HP_POTION_KEY}；"
        f"MP 缺口 > {MP_DEFICIT_THRESHOLD} 按 {MP_POTION_KEY}"
    )
    print("DirectInput 只會在所選遊戲視窗位於前景時送出。")
    print("按 Q 或 Esc 結束。")

    preview_title = "MapleStory YOLO Detection"
    if RENDER_PREVIEW_WINDOW:
        cv2.namedWindow(preview_title, cv2.WINDOW_AUTOSIZE)
    else:
        print(
            f"Preview disabled; terminal status every "
            f"{TERMINAL_STATUS_INTERVAL:.1f}s. Press Ctrl+C to stop."
        )
    input_controller = DirectInputController(hwnd)
    automation_started_at = perf_counter()
    next_pickup_at = automation_started_at + AUTOMATION_START_DELAY
    next_player_discovery_jump_at = (
        automation_started_at + PLAYER_DISCOVERY_JUMP_INITIAL_DELAY
    )
    combat_controller = TwoLayerCombatController(
        input_controller,
        automation_started_at,
    )
    buff_controller = TimedBuffController(
        input_controller,
        automation_started_at,
    )
    rest_controller = RestCycleController(
        input_controller,
        automation_started_at + AUTOMATION_START_DELAY,
    )
    atexit.register(input_controller.shutdown)
    atexit.register(scroll_detector.shutdown)
    atexit.register(rest_controller.shutdown)
    atexit.register(buff_controller.shutdown)
    print(
        f"Automatic pickup starts after {AUTOMATION_START_DELAY:.1f}s; "
        f"{AUTO_PICKUP_KEY.upper()} every {AUTO_PICKUP_INTERVAL:.2f}s"
    )
    print(
        f"Combat: slime={SLIME_ATTACK_KEY.upper()}, "
        f"snake={SNAKE_ATTACK_KEY.upper()}, "
        f"interval={ATTACK_INTERVAL:.2f}s, "
        f"pursuit={SAME_LAYER_MONSTER_MAX_DISTANCE}px"
    )
    print(
        "Timed buffs: "
        + ", ".join(
            f"{key.upper()} every {interval:.0f}s"
            for key, interval in BUFF_SCHEDULES
        )
        + f"; {BUFF_TAP_COUNT} taps every {BUFF_TAP_INTERVAL:.1f}s"
    )
    print(
        f"Rest cycle: hunt {HUNT_DURATION_MINUTES} minutes, "
        f"rest {REST_DURATION_MINUTES} minute(s)."
    )

    with mss() as screen:
        while True:
            region = get_client_region(hwnd)
            if region is None:
                print("遊戲視窗已關閉或最小化。")
                break

            started = perf_counter()
            rest_controller.update(started)
            if rest_controller.resume_ready:
                rest_controller.resume_hunting(started)
                next_pickup_at = started
                next_player_discovery_jump_at = started
                combat_controller = TwoLayerCombatController(
                    input_controller,
                    started - AUTOMATION_START_DELAY,
                )
            resting = rest_controller.active

            frame = np.asarray(screen.grab(region))[:, :, :3]
            scroll_matches = scroll_detector.update(frame, perf_counter())

            pickup_now = perf_counter()
            if (
                not resting
                and not buff_controller.active
                and pickup_now >= next_pickup_at
            ):
                if input_controller.tap(AUTO_PICKUP_KEY):
                    next_pickup_at = pickup_now + AUTO_PICKUP_INTERVAL

            resource_now = perf_counter()
            if (
                not resting
                and resource_now - last_resource_ocr_time >= RESOURCE_OCR_INTERVAL
            ):
                last_resource_ocr_time = resource_now
                new_hp_stat = read_resource_stat(frame, ocr, HP_ROI_OFFSETS)
                new_mp_stat = read_resource_stat(frame, ocr, MP_ROI_OFFSETS)
                if new_hp_stat is not None:
                    hp_stat = new_hp_stat
                if hp_stat is not None and hp_stat[0] <= CRITICAL_HP_THRESHOLD:
                    critical_hp_active = True
                    if critical_hp_started_at is None:
                        critical_hp_started_at = resource_now
                        next_critical_hp_heal_at = resource_now
                        next_critical_hp_fallback_at = (
                            resource_now + CRITICAL_HP_FALLBACK_DELAY
                        )

                    if resource_now >= next_critical_hp_heal_at:
                        if input_controller.tap(CRITICAL_HP_HEAL_KEY):
                            next_critical_hp_heal_at = (
                                resource_now + CRITICAL_HP_HEAL_INTERVAL
                            )

                    if (
                        resource_now - critical_hp_started_at
                        >= CRITICAL_HP_FALLBACK_DELAY
                        and resource_now >= next_critical_hp_fallback_at
                    ):
                        if input_controller.tap(HP_POTION_KEY):
                            next_critical_hp_fallback_at = (
                                resource_now + CRITICAL_HP_FALLBACK_INTERVAL
                            )
                else:
                    critical_hp_active = False
                    critical_hp_started_at = None
                    next_critical_hp_heal_at = 0.0
                    next_critical_hp_fallback_at = 0.0
                if new_mp_stat is not None:
                    mp_stat = new_mp_stat
                    if mp_stat[1] - mp_stat[0] > MP_DEFICIT_THRESHOLD:
                        input_controller.tap(MP_POTION_KEY)

            result = model.predict(
                frame,
                imgsz=IMAGE_SIZE,
                conf=minimum_model_confidence,
                device=INFERENCE_DEVICE,
                verbose=False,
            )[0]
            if reapply_cpu_threads_after_predictor_setup:
                # Ultralytics select_device() resets CPU threads during the
                # first predictor initialization, so apply our value again.
                torch.set_num_threads(CPU_INFERENCE_THREADS)
                reapply_cpu_threads_after_predictor_setup = False
                print(
                    "CPU inference threads applied after predictor setup: "
                    f"{torch.get_num_threads()}",
                    flush=True,
                )

            if result.boxes is not None and len(result.boxes) > 0:
                keep = result.boxes.conf < 0
                for class_id, confidence_threshold in class_confidence_by_id.items():
                    keep = keep | (
                        (result.boxes.cls == class_id)
                        & (result.boxes.conf >= confidence_threshold)
                    )
                for ignored_class_id in ignored_class_ids:
                    keep = keep & (result.boxes.cls != ignored_class_id)
                result = result[keep]

            motion_detections = []
            if result.boxes is not None:
                for result_index, box in enumerate(result.boxes):
                    class_name = model.names[int(box.cls.item())]
                    if class_name not in MONSTER_CLASS_NAMES:
                        continue
                    coordinates = tuple(int(value) for value in box.xyxy[0])
                    motion_detections.append(
                        (result_index, coordinates, class_name)
                    )

            moving_monster_indices = monster_motion_filter.update(
                motion_detections,
                perf_counter(),
            )
            if result.boxes is not None and len(result.boxes) > 0:
                motion_keep = result.boxes.cls == -1
                for result_index, box in enumerate(result.boxes):
                    class_name = model.names[int(box.cls.item())]
                    if (
                        class_name not in MONSTER_CLASS_NAMES
                        or result_index in moving_monster_indices
                    ):
                        motion_keep[result_index] = True
                result = result[motion_keep]

            player_boxes = [] if result.boxes is None else [
                box for box in result.boxes if int(box.cls.item()) == player_class_id
            ]
            now = perf_counter()
            if player_boxes and now - last_ocr_time >= OCR_INTERVAL:
                last_ocr_time = now
                ocr_matches = []
                for index, player_box in enumerate(player_boxes):
                    name_crop = crop_player_name(frame, player_box)
                    if name_crop is None:
                        continue
                    ocr_result = ocr(name_crop, use_det=False, use_cls=False, use_rec=True)
                    if not ocr_result.txts or not ocr_result.scores:
                        continue
                    text = ocr_result.txts[0].strip()
                    score = float(ocr_result.scores[0])
                    if score >= OCR_CONFIDENCE and is_my_player_name(text):
                        ocr_matches.append((index, score))

                if ocr_matches:
                    if self_center is None:
                        matched_index = max(ocr_matches, key=lambda item: item[1])[0]
                    else:
                        matched_index = min(
                            (index for index, _score in ocr_matches),
                            key=lambda index: (
                                (player_box_center(player_boxes[index])[0] - self_center[0]) ** 2
                                + (player_box_center(player_boxes[index])[1] - self_center[1]) ** 2
                            ),
                        )
                    self_center = player_box_center(player_boxes[matched_index])

            self_index = None
            if self_center is not None and player_boxes:
                self_index = min(
                    range(len(player_boxes)),
                    key=lambda index: (
                        (player_box_center(player_boxes[index])[0] - self_center[0]) ** 2
                        + (player_box_center(player_boxes[index])[1] - self_center[1]) ** 2
                    ),
                )
                self_center = player_box_center(player_boxes[self_index])

            player_labels = []
            for index, player_box in enumerate(player_boxes):
                coordinates = tuple(int(value) for value in player_box.xyxy[0])
                player_labels.append((coordinates, index == self_index))

            monster_boxes = []
            coin_boxes = []
            platform_boxes = []
            rope_boxes = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_name = model.names[int(box.cls.item())]
                    coordinates = tuple(int(value) for value in box.xyxy[0])
                    if class_name in MONSTER_CLASS_NAMES:
                        monster_boxes.append((coordinates, class_name))
                    elif class_name in COIN_CLASS_NAMES:
                        coin_boxes.append((coordinates, class_name))
                    elif class_name in PLATFORM_CLASS_NAMES:
                        platform_boxes.append(coordinates)
                    elif class_name == "rope":
                        rope_boxes.append(coordinates)

            self_coordinates = (
                None
                if self_index is None
                else player_labels[self_index][0]
            )
            player_position = (
                None
                if self_coordinates is None
                else tuple(
                    int(round(value))
                    for value in coordinates_center(self_coordinates)
                )
            )

            player_discovery_active = (
                PLAYER_DISCOVERY_JUMP_ENABLED and self_center is None
            )
            discovery_now = perf_counter()
            if (
                player_discovery_active
                and not resting
                and buff_controller.initial_buffs_completed
                and not buff_controller.active
                and input_controller.has_focus()
                and discovery_now >= next_player_discovery_jump_at
            ):
                if input_controller.tap(JUMP_KEY):
                    next_player_discovery_jump_at = (
                        discovery_now + PLAYER_DISCOVERY_JUMP_INTERVAL
                    )

            scroll_target = choose_nearest_scroll(
                self_coordinates,
                scroll_matches,
            )

            attack_range = (
                None
                if self_coordinates is None
                else build_attack_range(self_coordinates, frame.shape)
            )
            pursuit_range = (
                None
                if self_coordinates is None
                else build_same_layer_pursuit_range(self_coordinates, frame.shape)
            )
            nearby_monsters = (
                []
                if self_coordinates is None
                else same_platform_nearby_monsters(
                    self_coordinates,
                    monster_boxes,
                    platform_boxes,
                )
            )
            monsters_in_attack_range = (
                []
                if attack_range is None
                else [
                    monster
                    for monster in nearby_monsters
                    if box_center_is_inside(monster[0], attack_range)
                ]
            )

            player_on_rope = (
                self_coordinates is not None
                and player_is_hanging_on_rope(
                    self_coordinates,
                    rope_boxes,
                )
            )
            if resting:
                buff_blocked_reason = "RESTING"
            elif critical_hp_active:
                buff_blocked_reason = "HEALING"
            elif (
                player_on_rope
                or combat_controller.state in BUFF_ROPE_BLOCKING_STATES
            ):
                buff_blocked_reason = "ON ROPE"
            elif (
                monsters_in_attack_range
                or combat_controller.state == "combat"
            ):
                buff_blocked_reason = "COMBAT"
            else:
                buff_blocked_reason = None
            buff_now = perf_counter()
            buff_status = buff_controller.update(
                buff_now,
                buff_blocked_reason,
            )

            if resting:
                combat_controller.current_layer = (
                    "unknown"
                    if self_coordinates is None
                    else player_layer(self_coordinates)
                )
                combat_controller.nearby_monster_count = len(nearby_monsters)
                combat_controller.attack_range_monster_count = len(
                    monsters_in_attack_range
                )
                combat_controller.target_platform = None
                combat_controller.target_rope = None
                combat_controller.current_player_platform = (
                    None
                    if self_coordinates is None
                    else find_player_platform(
                        self_coordinates,
                        platform_boxes,
                    )
                )
                combat_controller.target_upper_snake_count = 0
                automation_status = rest_controller.status(perf_counter())
            elif buff_controller.active:
                combat_controller.abort(buff_now)
                automation_status = buff_status or buff_controller.status
            else:
                automation_status = combat_controller.update(
                    perf_counter(),
                    self_coordinates,
                    monster_boxes,
                    coin_boxes,
                    platform_boxes,
                    rope_boxes,
                    frame.shape,
                    critical_hp_active,
                )
                if buff_status is not None:
                    automation_status = (
                        f"{automation_status} | {buff_status}"
                    )
                if player_discovery_active:
                    automation_status = (
                        f"{automation_status} | SEARCH PLAYER ID: JUMP"
                    )

            if RENDER_PREVIEW_WINDOW:
                non_player_result = result
                if result.boxes is not None:
                    non_player_result = result[
                        result.boxes.cls != player_class_id
                    ]
                annotated = non_player_result.plot(
                    labels=True,
                    conf=True,
                    line_width=2,
                    img=frame.copy(),
                )

                fps = 1.0 / max(perf_counter() - started, 1e-6)
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if hp_stat is not None:
                    cv2.putText(
                        annotated,
                        f"HP: {hp_stat[0]}/{hp_stat[1]}",
                        (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (
                            (0, 0, 255)
                            if hp_stat[0] <= CRITICAL_HP_THRESHOLD
                            else (0, 255, 0)
                        ),
                        2,
                        cv2.LINE_AA,
                    )
                if mp_stat is not None:
                    mp_missing = mp_stat[1] - mp_stat[0]
                    cv2.putText(
                        annotated,
                        f"MP: {mp_stat[0]}/{mp_stat[1]}",
                        (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (
                            (0, 0, 255)
                            if mp_missing > MP_DEFICIT_THRESHOLD
                            else (255, 200, 0)
                        ),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    annotated,
                    f"AUTO: {automation_status}",
                    (15, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                scan_interval = scroll_detector.scan_interval
                cv2.putText(
                    annotated,
                    f"SCROLL SCAN: {scan_interval:.2f}s",
                    (15, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (
                        (0, 255, 255)
                        if scroll_detector.active
                        else (180, 180, 180)
                    ),
                    2,
                    cv2.LINE_AA,
                )
                if SHOW_PLAYER_COORDINATES:
                    player_position_text = (
                        "PLAYER CENTER: NOT FOUND"
                        if player_position is None
                        else (
                            f"PLAYER CENTER: "
                            f"({player_position[0]}, {player_position[1]})"
                        )
                    )
                    cv2.putText(
                        annotated,
                        player_position_text,
                        (15, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    annotated,
                    (
                        f"LAYER: {combat_controller.current_layer.upper()} | "
                        f"NEAR: {len(nearby_monsters)} | "
                        f"ATTACK: {len(monsters_in_attack_range)}"
                    ),
                    (15, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                draw_navigation_debug(
                    annotated,
                    attack_range,
                    pursuit_range,
                    nearby_monsters,
                    monsters_in_attack_range,
                    combat_controller,
                )
                draw_scroll_matches(annotated, scroll_matches, scroll_target)
                annotated = draw_player_labels(
                    annotated,
                    player_labels,
                    ocr_font,
                )
                cv2.imshow(preview_title, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
            else:
                terminal_now = perf_counter()
                fps = 1.0 / max(terminal_now - started, 1e-6)
                if (
                    terminal_now - last_terminal_status_time
                    >= TERMINAL_STATUS_INTERVAL
                ):
                    last_terminal_status_time = terminal_now
                    hp_text = (
                        "HP: --/--"
                        if hp_stat is None
                        else (
                            f"HP: {hp_stat[0]}/{hp_stat[1]} "
                            f"(missing {hp_stat[1] - hp_stat[0]})"
                        )
                    )
                    mp_text = (
                        "MP: --/--"
                        if mp_stat is None
                        else (
                            f"MP: {mp_stat[0]}/{mp_stat[1]} "
                            f"(missing {mp_stat[1] - mp_stat[0]})"
                        )
                    )
                    player_text = (
                        ""
                        if not SHOW_PLAYER_COORDINATES
                        else (
                            " | PLAYER: NOT FOUND"
                            if player_position is None
                            else (
                                f" | PLAYER: "
                                f"({player_position[0]}, {player_position[1]})"
                            )
                        )
                    )
                    print(
                        f"[STATUS] FPS: {fps:.1f} | {hp_text} | {mp_text} | "
                        f"AUTO: {automation_status}{player_text} | "
                        f"LAYER: {combat_controller.current_layer.upper()} | "
                        f"MONSTERS: {len(monster_boxes)} | "
                        f"NEAR: {len(nearby_monsters)} | "
                        f"ATTACK: {len(monsters_in_attack_range)} | "
                        f"COINS: {len(coin_boxes)} | "
                        f"PLATFORMS: {len(platform_boxes)} | "
                        f"ROPES: {len(rope_boxes)} | "
                        f"SCROLLS: {len(scroll_matches)}",
                        flush=True,
                    )

    buff_controller.shutdown()
    rest_controller.shutdown()
    input_controller.shutdown()
    scroll_detector.shutdown()
    if RENDER_PREVIEW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
        cv2.destroyAllWindows()
