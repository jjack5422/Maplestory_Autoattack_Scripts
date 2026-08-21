import argparse
import atexit
import ctypes
import random
import re
from collections import deque
from ctypes import wintypes
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import hypot, isfinite
from pathlib import Path
from queue import Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter, sleep

import cv2
import numpy as np
import pydirectinput
from mss import MSS
from PIL import Image, ImageDraw, ImageFont
from rapidocr import RapidOCR
from ultralytics import YOLO

from capture_game_window import enable_dpi_awareness, get_client_region, list_visible_windows


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "maplestory_02.pt"
ROPE_CONFIDENCE = 0.10
PLAYER_CONFIDENCE = 0.40
OTHER_CONFIDENCE = 0.6
IMAGE_SIZE = 960
GLOBAL_STOP_KEY = 0x77  # F8
OCR_INTERVAL = 0.25
OCR_CONFIDENCE = 0.60
OCR_FONT_PATH = Path("C:/Windows/Fonts/msjhbd.ttc")
MY_PLAYER_NAME = "阿罵的手機"
NAME_MATCH_THRESHOLD = 0.80

# Attack-range hyperparameters, measured from the tracked player's box.
ATTACK_RANGE_HORIZONTAL_PADDING = 200
ATTACK_RANGE_VERTICAL_PADDING = 10
SAME_PLATFORM_CHASE_MARGIN = 200
MONSTER_CLASS_NAMES = frozenset(
    {"blue_snail", "mushroom", "red_snail", "snail", "slime", "wood"}
)
IGNORED_CLASS_NAMES = frozenset()

# A YOLO monster must demonstrate motion before it becomes a valid target.
MONSTER_TRACK_MATCH_DISTANCE = 80
MONSTER_TRACK_MAX_MISSING = 0.50
MONSTER_MOTION_WINDOW = 0.75
MONSTER_MIN_MOTION_PIXELS = 8
MONSTER_MOTION_MIN_OBSERVATIONS = 3
MONSTER_MOTION_CONFIRM_SAMPLES = 2

# Auto-combat input and timing hyperparameters.
AUTOMATION_START_DELAY = 3.0
AUTO_PICKUP_KEY = "z"
AUTO_PICKUP_INTERVAL = 0.15
ATTACK_KEY = "q"
ATTACK_INTERVAL = 0.15
STUCK_ATTACK_DURATION = 3.0
UNSTICK_MOVE_DURATION = 2.0
PLAYER_STATIONARY_TIMEOUT = 10.0
PLAYER_STATIONARY_MOVEMENT_THRESHOLD = 8
PLAYER_STATIONARY_RECOVERY_DURATION = 3.0
PLAYER_LOST_TIMEOUT = 3.0
PLAYER_LOST_RECOVERY_DURATION = 1.0
# Never let a temporarily missing self detection jump to a nearby player.
SELF_TRACK_MAX_DISTANCE = 45
SELF_TRACK_AMBIGUITY_MARGIN = 25
LEFT_KEY = "left"
RIGHT_KEY = "right"
UP_KEY = "up"
DOWN_KEY = "down"
JUMP_KEY = "alt"
ESCAPE_KEY = "esc"
ENTER_KEY = "enter"

# Take an in-game break after each 30-minute combat session.
COMBAT_SESSION_DURATION = 30 * 60.0
OFFLINE_BREAK_DURATION = 2 * 60.0
LOGOUT_MENU_DELAY = 1.0
LOGOUT_CONFIRM_DELAY = 1.0
LOGOUT_FINAL_CONFIRM_DELAY = 1.0
LOGIN_ENTER_INTERVAL = 1.50
LOGIN_SETTLE_DURATION = 5.0

# One-time final logout. Set to 0 to disable, or override it with
# --auto-logout-minutes when starting the script.
AUTO_LOGOUT_MINUTES = 0.0
AUTO_LOGOUT_COMPLETION_DELAY = 0.50

# Geometry and movement hyperparameters.
SAME_LEVEL_FEET_TOLERANCE = 45
WALK_TARGET_LOST_GRACE = 0.35
# Use a symmetric floor gap so lower monsters are not much harder to classify
# than upper monsters. Same-platform detections are filtered by feet first.
MONSTER_ABOVE_MIN_OFFSET = 80
MONSTER_BELOW_MIN_OFFSET = 80
ROPE_FINE_ALIGNMENT_DISTANCE = 30
ROPE_CENTER_ENTER_TOLERANCE = 8
ROPE_CENTER_EXIT_TOLERANCE = 18
ROPE_ALIGN_TAP_INTERVAL = 0.10
ROPE_CENTER_CONFIRM_DURATION = 0.50
ROPE_ON_HORIZONTAL_MARGIN = 18
ROPE_VERTICAL_MARGIN = 15
ROPE_REACH_VERTICAL_GAP = 80
ROPE_SEARCH_TIMEOUT = 3.0
ROPE_SEARCH_DIRECTION_DURATION = 0.50
DROP_PREPARE_DURATION = 1
DROP_JUMP_DURATION = 1
ROPE_ESCAPE_PREPARE_DURATION = 0.20
ROPE_ESCAPE_JUMP_DURATION = 0.50
CLIMB_JUMP_LEAD_DURATION = 0.20
CLIMB_ALT_HOLD_DURATION = 1.00
CLIMB_ATTACH_TIMEOUT = 1.20
# Dismount as soon as the detected rope top is reached. The fixed duration is
# retained only as a safety fallback when the top cannot be detected.
CLIMB_UP_MAX_DURATION = 3.75
CLIMB_TOP_VERTICAL_MARGIN = 24
CLIMB_TOP_CONFIRM_DURATION = 0.25
CLIMB_ROPE_TRACK_HORIZONTAL_MARGIN = 45
CLIMB_PROGRESS_MIN_PIXELS = 4
CLIMB_STALL_MIN_UP_DURATION = 0.80
CLIMB_STALL_CONFIRM_DURATION = 0.35
CLIMB_DISMOUNT_LEAD_DURATION = 0.12
CLIMB_DISMOUNT_JUMP_DURATION = 0.65
CLIMB_DISMOUNT_MAX_ATTEMPTS = 3

# Resource automation hyperparameters.
RESOURCE_OCR_INTERVAL = 0.25
RESOURCE_LOW_CONFIRMATIONS = 2
RESOURCE_POTION_COOLDOWN = 0.75
HP_REMAINING_RATIO_THRESHOLD = 0.50
HP_POTION_KEY = "1"
MP_DEFICIT_THRESHOLD = 150
MP_POTION_KEY = "2"

HP_ROI_OFFSETS = (-183, -104, -35, -15)
MP_ROI_OFFSETS = (-73, 7, -35, -15)
RESOURCE_OCR_SCALE = 5
DIRECTINPUT_KEY_HOLD = 0.060

# Scroll-template detection and navigation hyperparameters.
SCROLL_TEMPLATE_DIRECTORY = ROOT / "assests" / "scroll"
SCROLL_IDLE_SCAN_INTERVAL = 3.0
SCROLL_ACTIVE_SCAN_INTERVAL = 0.25
SCROLL_TEMPLATE_MATCH_THRESHOLD = 0.92
SCROLL_MAX_MEAN_COLOR_ERROR = 35.0
SCROLL_MAX_MATCHES_PER_TEMPLATE = 5
SCROLL_NMS_THRESHOLD = 0.30
SCROLL_PLATFORM_VERTICAL_OFFSET = 200
SCROLL_X_ALIGNMENT_TOLERANCE = 18
SCROLL_DROP_X_TOLERANCE = 80
SCROLL_JUMP_MIN_VERTICAL_OFFSET = 10
SCROLL_JUMP_INTERVAL = 0.50


user32 = ctypes.windll.user32
user32.GetForegroundWindow.argtypes = ()
user32.GetForegroundWindow.restype = wintypes.HWND


class DirectInputController:
    """Route DirectInput through independent key-group workers."""

    WORKER_KEYS = {
        "z": frozenset({AUTO_PICKUP_KEY}),
        "q": frozenset({ATTACK_KEY}),
        "horizontal": frozenset({LEFT_KEY, RIGHT_KEY}),
        "vertical": frozenset({UP_KEY, DOWN_KEY}),
        "alt": frozenset({JUMP_KEY}),
        "resource": frozenset({HP_POTION_KEY, MP_POTION_KEY}),
        "menu": frozenset({ESCAPE_KEY, ENTER_KEY}),
    }

    def __init__(self, hwnd: int) -> None:
        self.hwnd = hwnd
        self.held_keys: set[str] = set()
        self._physical_held_keys: set[str] = set()
        self._pending_taps: set[str] = set()
        self._lock = Lock()
        self._stop_event = Event()
        self._closed = False
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

    def _command_queue_for(self, key: str) -> Queue[tuple[str, object]] | None:
        worker_name = self._key_to_worker.get(key)
        return None if worker_name is None else self._commands[worker_name]

    def key_down(self, key: str) -> bool:
        command_queue = self._command_queue_for(key)
        if command_queue is None or not self.has_focus():
            return False

        with self._lock:
            if self._closed:
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
            if self._closed or key in self.held_keys or key in self._pending_taps:
                return False
            self._pending_taps.add(key)
        command_queue.put(("tap", key))
        return True

    def tap_attack(self, direction: str) -> bool:
        """Tap a facing direction, then wake the independent Q worker."""
        if direction not in (LEFT_KEY, RIGHT_KEY) or not self.has_focus():
            return False

        attack_keys = (direction, ATTACK_KEY)
        with self._lock:
            if self._closed or any(
                key in self.held_keys or key in self._pending_taps
                for key in attack_keys
            ):
                return False
            self._pending_taps.update(attack_keys)

        direction_completed = Event()
        self._commands["horizontal"].put(
            ("tap_signal", (direction, direction_completed))
        )
        self._commands["q"].put(
            ("tap_after", (ATTACK_KEY, direction_completed))
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
                        if self._stop_event.is_set():
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

    def _physical_key_down(self, key: str) -> bool:
        with self._lock:
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
        if not self.has_focus():
            return False

        key_down_ok = pydirectinput.keyDown(key)
        try:
            sleep(DIRECTINPUT_KEY_HOLD)
        finally:
            key_up_ok = pydirectinput.keyUp(key)
        return bool(key_down_ok and key_up_ok)


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
    if (
        x1 < 0
        or y1 < 0
        or x2 > frame_width
        or y2 > frame_height
        or x1 >= x2
        or y1 >= y2
    ):
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


@dataclass
class ResourcePotionTrigger:
    """Repeat potion requests at a safe interval while a resource remains low.

    OCR can occasionally drop the leading digit from four-digit HUD values.
    Requiring repeated low readings filters a one-frame error, while an OCR miss
    between valid readings must not erase all progress toward confirmation.
    """

    deficit_threshold: int | None = None
    remaining_ratio_threshold: float | None = None
    low_confirmations_required: int = RESOURCE_LOW_CONFIRMATIONS
    cooldown: float = RESOURCE_POTION_COOLDOWN
    consecutive_low_readings: int = 0
    last_triggered_at: float = float("-inf")

    def should_trigger(
        self,
        stat: tuple[int, int] | None,
        now: float | None = None,
    ) -> bool:
        if stat is None:
            # OCR commonly misses an isolated frame. Preserve the preceding
            # valid low reading so two good samples can still confirm low HP.
            return False

        current, maximum = stat
        if self.remaining_ratio_threshold is not None:
            is_low = current / maximum <= self.remaining_ratio_threshold
        else:
            is_low = (
                self.deficit_threshold is not None
                and maximum - current > self.deficit_threshold
            )

        if not is_low:
            self.consecutive_low_readings = 0
            return False

        self.consecutive_low_readings += 1
        if self.consecutive_low_readings < self.low_confirmations_required:
            return False
        return now is None or now - self.last_triggered_at >= self.cooldown

    def mark_triggered(self, now: float | None = None) -> None:
        self.last_triggered_at = perf_counter() if now is None else now
        self.consecutive_low_readings = 0


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


def find_window_by_title(title_query: str, timeout: float) -> tuple[int, str]:
    """Find a visible window without requiring interactive menu input."""
    deadline = perf_counter() + timeout
    query = title_query.casefold()
    while perf_counter() < deadline:
        windows = list_visible_windows()
        exact = [window for window in windows if window[1].casefold() == query]
        partial = [window for window in windows if query in window[1].casefold()]
        matches = exact or partial
        if matches:
            hwnd, title, _width, _height = max(
                matches,
                key=lambda window: window[2] * window[3],
            )
            return hwnd, title
        sleep(0.5)
    raise RuntimeError(
        f"Window containing {title_query!r} was not found within {timeout:.0f}s."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window-title",
        help="Automatically select a visible window containing this title.",
    )
    parser.add_argument(
        "--window-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for --window-title (default: 60).",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="Exit cleanly when this file appears.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the annotated OpenCV preview for debugging (default: off).",
    )
    parser.add_argument(
        "--auto-logout-minutes",
        type=float,
        default=AUTO_LOGOUT_MINUTES,
        help=(
            "Permanently log out after this many minutes, then stop the script. "
            "Uses Esc, Up, Enter, Enter. Set to 0 to disable "
            f"(default: {AUTO_LOGOUT_MINUTES:g})."
        ),
    )
    arguments = parser.parse_args()
    if (
        not isfinite(arguments.auto_logout_minutes)
        or arguments.auto_logout_minutes < 0
    ):
        parser.error("--auto-logout-minutes must be a finite number >= 0")
    return arguments


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


def expand_player_box(
    player_coordinates: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Build a frame-clipped attack range around the tracked player."""
    x1, y1, x2, y2 = player_coordinates
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
    """Return whether a detection's center point is inside an area."""
    x1, y1, x2, y2 = coordinates
    area_x1, area_y1, area_x2, area_y2 = area
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return area_x1 <= center_x <= area_x2 and area_y1 <= center_y <= area_y2


def coordinates_center(
    coordinates: tuple[int, int, int, int],
) -> tuple[float, float]:
    x1, y1, x2, y2 = coordinates
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def monster_is_on_player_level(
    player: tuple[int, int, int, int],
    monster: tuple[int, int, int, int],
) -> bool:
    """Compare feet positions without requiring a detected platform."""
    return abs(player[3] - monster[3]) <= SAME_LEVEL_FEET_TOLERANCE


def monster_is_within_same_platform_chase_range(
    player: tuple[int, int, int, int],
    monster: tuple[int, int, int, int],
) -> bool:
    """Only chase same-platform monsters up to 200 px beyond attack range."""
    if not monster_is_on_player_level(player, monster):
        return False

    monster_x, _monster_y = coordinates_center(monster)
    attack_left = player[0] - ATTACK_RANGE_HORIZONTAL_PADDING
    attack_right = player[2] + ATTACK_RANGE_HORIZONTAL_PADDING
    distance_beyond_attack_range = max(
        attack_left - monster_x,
        monster_x - attack_right,
        0.0,
    )
    return distance_beyond_attack_range <= SAME_PLATFORM_CHASE_MARGIN


def player_is_on_or_near_rope(
    player: tuple[int, int, int, int],
    ropes: list[tuple[int, int, int, int]],
) -> bool:
    """Treat overlap and close horizontal contact as being on a rope."""
    player_x1, player_y1, player_x2, player_y2 = player
    for rope_x1, rope_y1, rope_x2, rope_y2 in ropes:
        horizontal_contact = (
            player_x2 >= rope_x1 - ROPE_ON_HORIZONTAL_MARGIN
            and player_x1 <= rope_x2 + ROPE_ON_HORIZONTAL_MARGIN
        )
        vertical_contact = (
            player_y2 >= rope_y1 - ROPE_VERTICAL_MARGIN
            and player_y1 <= rope_y2 + ROPE_VERTICAL_MARGIN
        )
        if horizontal_contact and vertical_contact:
            return True
    return False


def player_rope_top_gap(
    player: tuple[int, int, int, int],
    ropes: list[tuple[int, int, int, int]],
    tracked_rope_x: float,
) -> float | None:
    """Measure the vertical distance from the player center to a tracked rope top."""
    _player_x, player_y = coordinates_center(player)
    matching_ropes = []
    for rope in ropes:
        rope_x, _rope_y = coordinates_center(rope)
        rope_x_distance = abs(rope_x - tracked_rope_x)
        if rope_x_distance > CLIMB_ROPE_TRACK_HORIZONTAL_MARGIN:
            continue

        _rope_x1, rope_y1, _rope_x2, _rope_y2 = rope
        # Ignore a different rope segment entirely below the character.
        if rope_y1 > player[3] + ROPE_VERTICAL_MARGIN:
            continue
        matching_ropes.append((rope_x_distance, rope_y1))

    if not matching_ropes:
        return None

    _rope_x_distance, rope_top_y = min(matching_ropes)
    return player_y - rope_top_y


def player_is_at_rope_top(
    player: tuple[int, int, int, int],
    ropes: list[tuple[int, int, int, int]],
    tracked_rope_x: float,
) -> bool:
    """Return true once the tracked rope's top reaches the player's center."""
    top_gap = player_rope_top_gap(player, ropes, tracked_rope_x)
    return top_gap is not None and top_gap <= CLIMB_TOP_VERTICAL_MARGIN


def nearest_monster(
    player: tuple[int, int, int, int],
    monsters: list[tuple[tuple[int, int, int, int], str]],
) -> tuple[tuple[int, int, int, int], str]:
    player_x, player_y = coordinates_center(player)
    return min(
        monsters,
        key=lambda monster: (
            (coordinates_center(monster[0])[0] - player_x) ** 2
            + (coordinates_center(monster[0])[1] - player_y) ** 2
        ),
    )


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


def split_monsters_vertically(
    player: tuple[int, int, int, int],
    monsters: list[tuple[tuple[int, int, int, int], str]],
) -> tuple[
    list[tuple[tuple[int, int, int, int], str]],
    list[tuple[tuple[int, int, int, int], str]],
]:
    _player_x, player_y = coordinates_center(player)
    above = []
    below = []
    for monster in monsters:
        if monster_is_on_player_level(player, monster[0]):
            continue
        _monster_x, monster_y = coordinates_center(monster[0])
        if monster_y < player_y - MONSTER_ABOVE_MIN_OFFSET:
            above.append(monster)
        elif monster_y >= player_y + MONSTER_BELOW_MIN_OFFSET:
            below.append(monster)
    return above, below


def choose_vertical_exploration_direction(
    above: list[tuple[tuple[int, int, int, int], str]],
    below: list[tuple[tuple[int, int, int, int], str]],
    current_direction: str | None,
) -> str:
    """Choose the floor containing more monsters without forcing upward travel."""
    if len(above) > len(below):
        return "up"
    if len(below) > len(above):
        return "down"
    if current_direction in ("up", "down"):
        return current_direction
    return random.choice(("up", "down"))


def choose_rope_for_climb(
    player: tuple[int, int, int, int],
    ropes: list[tuple[int, int, int, int]],
    upper_monsters: list[tuple[tuple[int, int, int, int], str]],
) -> tuple[int, int, int, int] | None:
    """Choose a reachable upward rope, biased toward the upper monsters."""
    player_x, _player_y = coordinates_center(player)
    target_x = player_x
    if upper_monsters:
        target_x = (
            sum(coordinates_center(monster[0])[0] for monster in upper_monsters)
            / len(upper_monsters)
        )
    return choose_rope_for_target_x(player, ropes, target_x)


def choose_rope_for_target_x(
    player: tuple[int, int, int, int],
    ropes: list[tuple[int, int, int, int]],
    target_x: float,
) -> tuple[int, int, int, int] | None:
    """Choose a reachable upward rope while biasing toward a target X."""
    player_x, player_y = coordinates_center(player)
    candidates = []
    for rope in ropes:
        rope_x, _rope_y = coordinates_center(rope)
        _rope_x1, rope_y1, _rope_x2, rope_y2 = rope
        if rope_y1 >= player_y or rope_y2 < player[1] - ROPE_REACH_VERTICAL_GAP:
            continue
        score = abs(rope_x - player_x) + 0.25 * abs(rope_x - target_x)
        candidates.append((score, rope))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


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


class AutoCombatController:
    """Non-blocking navigation and combat state machine."""

    def __init__(
        self,
        inputs: DirectInputController,
        started_at: float,
        auto_logout_after: float | None = None,
    ) -> None:
        self.inputs = inputs
        self.started_at = started_at
        self.next_pickup_at = started_at + AUTOMATION_START_DELAY
        self.last_attack_at = float("-inf")
        self.combat_started_at: float | None = None
        self.facing_direction: str | None = None
        self.unstick_direction: str | None = None
        self.walk_direction: str | None = None
        self.walk_target_name: str | None = None
        self.walk_target_last_seen_at = float("-inf")
        self.explore_direction: str | None = None
        self.rope_centered_at: float | None = None
        self.confirmed_rope_x: float | None = None
        self.last_rope_align_tap_at = float("-inf")
        self.rope_search_started_at: float | None = None
        self.rope_search_direction_started_at = 0.0
        self.rope_search_direction: str | None = None
        self.state = "idle"
        self.state_started_at = started_at
        self.climb_alt_started_at = 0.0
        self.climb_up_started_at = 0.0
        self.climb_seen_rope = False
        self.climb_exit_direction: str | None = None
        self.climb_rope_x: float | None = None
        self.climb_top_candidate_at: float | None = None
        self.climb_best_top_gap: float | None = None
        self.climb_last_progress_at = 0.0
        self.climb_dismount_attempts = 0
        self.last_climb_dismount_jump_at = float("-inf")
        self.scroll_mode_active = False
        self.last_scroll_jump_at = float("-inf")
        self.player_motion_anchor: tuple[float, float] | None = None
        self.player_last_moved_at = started_at
        self.stationary_recovery_direction: str | None = None
        self.stationary_recovery_started_at = 0.0
        self.stationary_recovery_context: str | None = None
        self.player_last_seen_at = started_at
        self.player_lost_recovery_direction: str | None = None
        self.player_lost_recovery_pending_direction: str | None = None
        self.player_lost_recovery_started_at = 0.0
        self.player_lost_rope_climb_started_at = 0.0
        self.break_phase = "playing"
        self.break_phase_started_at = started_at
        self.next_break_at = (
            started_at + AUTOMATION_START_DELAY + COMBAT_SESSION_DURATION
        )
        self.login_enter_target = 0
        self.login_enter_count = 0
        self.next_login_enter_at = 0.0
        self.final_logout_at = (
            None
            if auto_logout_after is None or auto_logout_after <= 0
            else started_at + auto_logout_after
        )
        self.final_logout_phase = "waiting"
        self.final_logout_phase_started_at = started_at
        self.final_logout_complete = False

    def set_state(self, state: str, now: float) -> None:
        self.state = state
        self.state_started_at = now

    def reset_rope_alignment(self) -> None:
        self.rope_centered_at = None
        self.confirmed_rope_x = None

    def reset_climb_tracking(self) -> None:
        self.climb_seen_rope = False
        self.climb_exit_direction = None
        self.climb_rope_x = None
        self.climb_top_candidate_at = None
        self.climb_best_top_gap = None
        self.climb_last_progress_at = 0.0
        self.climb_dismount_attempts = 0
        self.last_climb_dismount_jump_at = float("-inf")

    def reset_walk_target(self) -> None:
        self.walk_direction = None
        self.walk_target_name = None
        self.walk_target_last_seen_at = float("-inf")

    def reset_rope_search(self) -> None:
        self.rope_search_started_at = None
        self.rope_search_direction_started_at = 0.0
        self.rope_search_direction = None

    def reset_exploration(self) -> None:
        self.explore_direction = None
        self.reset_rope_search()

    def reset_stationary_tracking(self, now: float) -> None:
        self.player_motion_anchor = None
        self.player_last_moved_at = now
        self.stationary_recovery_direction = None
        self.stationary_recovery_started_at = 0.0
        self.stationary_recovery_context = None

    def stop_navigation(self, now: float) -> None:
        self.inputs.release_navigation()
        self.set_state("idle", now)
        self.combat_started_at = None
        self.unstick_direction = None
        self.reset_walk_target()
        self.reset_exploration()
        self.reset_rope_alignment()
        self.reset_climb_tracking()
        self.reset_stationary_tracking(now)

    def trigger_pickup_if_due(self, now: float) -> None:
        if (
            now >= self.next_pickup_at
            and self.inputs.tap(AUTO_PICKUP_KEY)
        ):
            self.next_pickup_at = now + AUTO_PICKUP_INTERVAL

    def update_final_logout(self, now: float) -> str | None:
        """Run a one-time Esc/Up/Enter/Enter logout, then remain offline."""
        if self.final_logout_at is None or self.final_logout_complete:
            return None

        if self.final_logout_phase == "waiting":
            if now < self.final_logout_at:
                return None
            # If the regular 30-minute break is in progress, let it finish its
            # login first so the logout keys are not sent on the wrong screen.
            if self.break_phase != "playing":
                return None
            self.stop_navigation(now)
            self.inputs.release_all()
            if self.inputs.tap(ESCAPE_KEY):
                self.final_logout_phase = "logout_menu"
                self.final_logout_phase_started_at = now
            return "FINAL LOGOUT: OPENING LOGOUT MENU"

        self.inputs.release_all()

        if self.final_logout_phase == "logout_menu":
            if now - self.final_logout_phase_started_at < LOGOUT_MENU_DELAY:
                return "FINAL LOGOUT: WAITING FOR LOGOUT MENU"
            if self.inputs.tap(UP_KEY):
                self.final_logout_phase = "logout_confirm"
                self.final_logout_phase_started_at = now
            return "FINAL LOGOUT: SELECTING OFFLINE"

        if self.final_logout_phase == "logout_confirm":
            if now - self.final_logout_phase_started_at < LOGOUT_CONFIRM_DELAY:
                return "FINAL LOGOUT: WAITING TO CONFIRM OFFLINE"
            if self.inputs.tap(ENTER_KEY):
                self.final_logout_phase = "logout_final_confirm"
                self.final_logout_phase_started_at = now
            return "FINAL LOGOUT: OPENING OFFLINE CONFIRMATION"

        if self.final_logout_phase == "logout_final_confirm":
            if now - self.final_logout_phase_started_at < LOGOUT_FINAL_CONFIRM_DELAY:
                return "FINAL LOGOUT: WAITING FOR OFFLINE CONFIRMATION"
            if self.inputs.tap(ENTER_KEY):
                self.final_logout_phase = "settle"
                self.final_logout_phase_started_at = now
            return "FINAL LOGOUT: CONFIRMING OFFLINE"

        if self.final_logout_phase == "settle":
            if now - self.final_logout_phase_started_at < AUTO_LOGOUT_COMPLETION_DELAY:
                return "FINAL LOGOUT: COMPLETING"
            self.final_logout_complete = True
            return "FINAL LOGOUT: COMPLETE"

        raise RuntimeError(
            f"Unknown final-logout phase: {self.final_logout_phase}"
        )

    def update_scheduled_break(self, now: float) -> str | None:
        """Run the Esc/Enter logout and Enter login sequence without blocking."""
        if self.break_phase == "playing":
            if now < self.next_break_at:
                return None
            self.stop_navigation(now)
            self.inputs.release_all()
            if self.inputs.tap(ESCAPE_KEY):
                self.break_phase = "logout_menu"
                self.break_phase_started_at = now
            return "BREAK: OPENING LOGOUT MENU"

        if self.break_phase == "logout_menu":
            self.inputs.release_all()
            if now - self.break_phase_started_at < LOGOUT_MENU_DELAY:
                return "BREAK: WAITING FOR LOGOUT MENU"
            if self.inputs.tap(UP_KEY):
                self.break_phase = "logout_confirm"
                self.break_phase_started_at = now
            return "BREAK: SELECTING OFFLINE"

        if self.break_phase == "logout_confirm":
            self.inputs.release_all()
            if now - self.break_phase_started_at < LOGOUT_CONFIRM_DELAY:
                return "BREAK: WAITING TO CONFIRM OFFLINE"
            if self.inputs.tap(ENTER_KEY):
                self.break_phase = "logout_final_confirm"
                self.break_phase_started_at = now
            return "BREAK: OPENING OFFLINE CONFIRMATION"

        if self.break_phase == "logout_final_confirm":
            self.inputs.release_all()
            if now - self.break_phase_started_at < LOGOUT_FINAL_CONFIRM_DELAY:
                return "BREAK: WAITING FOR OFFLINE CONFIRMATION"
            if self.inputs.tap(ENTER_KEY):
                self.break_phase = "offline"
                self.break_phase_started_at = now
            return "BREAK: CONFIRMING OFFLINE"

        if self.break_phase == "offline":
            self.inputs.release_all()
            remaining = OFFLINE_BREAK_DURATION - (now - self.break_phase_started_at)
            if remaining > 0:
                return f"OFFLINE BREAK: {remaining:.0f}s"

            self.login_enter_target = random.choice((2, 3))
            self.login_enter_count = 0
            self.next_login_enter_at = now
            self.break_phase = "login"

        if self.break_phase == "login":
            self.inputs.release_all()
            if (
                self.login_enter_count < self.login_enter_target
                and now >= self.next_login_enter_at
                and self.inputs.tap(ENTER_KEY)
            ):
                self.login_enter_count += 1
                self.next_login_enter_at = now + LOGIN_ENTER_INTERVAL

            if self.login_enter_count < self.login_enter_target:
                return (
                    "LOGIN: ENTER "
                    f"{self.login_enter_count}/{self.login_enter_target}"
                )

            self.break_phase = "login_settle"
            self.break_phase_started_at = now
            return "LOGIN: WAITING FOR GAME"

        if self.break_phase == "login_settle":
            self.inputs.release_all()
            remaining = LOGIN_SETTLE_DURATION - (now - self.break_phase_started_at)
            if remaining > 0:
                return f"LOGIN: SETTLING {remaining:.1f}s"

            self.break_phase = "playing"
            self.break_phase_started_at = now
            self.next_break_at = now + COMBAT_SESSION_DURATION
            self.started_at = now - AUTOMATION_START_DELAY
            self.next_pickup_at = now
            self.player_last_seen_at = now
            self.reset_stationary_tracking(now)
            return "LOGIN: COMBAT RESUMED"

        raise RuntimeError(f"Unknown scheduled-break phase: {self.break_phase}")

    def update_player_lost_recovery(self, now: float) -> str:
        """Move in a random direction after self detection is missing for 3 seconds."""
        self.combat_started_at = None
        self.reset_walk_target()
        self.reset_rope_alignment()

        if self.player_lost_recovery_direction is not None:
            elapsed = now - self.player_lost_recovery_started_at
            if elapsed < PLAYER_LOST_RECOVERY_DURATION:
                return (
                    "SELF HIDDEN: MOVE "
                    f"{self.player_lost_recovery_direction.upper()}"
                )

            self.inputs.release_navigation()
            self.player_lost_recovery_direction = None
            self.player_last_seen_at = now
            return "SELF HIDDEN: RECOVERY MOVE FINISHED"

        self.inputs.release_navigation()
        missing_duration = now - self.player_last_seen_at
        if missing_duration < PLAYER_LOST_TIMEOUT:
            return f"PAUSED: SELF MISSING {missing_duration:.1f}s"

        # Never press UP while the player is hidden: the character may be
        # standing on a portal, which could accidentally leave the map.
        self.player_lost_recovery_direction = random.choice(
            (LEFT_KEY, RIGHT_KEY, DOWN_KEY)
        )
        self.player_lost_recovery_pending_direction = (
            self.player_lost_recovery_direction
        )
        self.player_lost_recovery_started_at = now
        self.inputs.key_down(self.player_lost_recovery_direction)
        return (
            "SELF HIDDEN: MOVE "
            f"{self.player_lost_recovery_direction.upper()}"
        )

    def continue_player_lost_rope_climb(
        self,
        now: float,
        on_or_near_rope: bool | None,
    ) -> str | None:
        """Keep climbing after an UP recovery, then use the normal dismount."""
        if self.state != "player_lost_rope_climb":
            return None

        elapsed = now - self.player_lost_rope_climb_started_at
        if elapsed >= CLIMB_UP_MAX_DURATION:
            return self.begin_climb_dismount(now, "self-recovery timeout")

        self.inputs.key_down(UP_KEY)
        rope_status = (
            "ROPE DETECTED"
            if on_or_near_rope is True
            else "CLEARING PLATFORM EDGE"
            if on_or_near_rope is False
            else "SELF HIDDEN"
        )
        return (
            "SELF RECOVERY CLIMB: UP "
            f"{elapsed:.1f}/{CLIMB_UP_MAX_DURATION:.1f}s "
            f"({rope_status})"
        )

    def observe_player_movement(
        self,
        now: float,
        player: tuple[int, int, int, int],
    ) -> bool:
        player_center = coordinates_center(player)
        if self.player_motion_anchor is None:
            self.player_motion_anchor = player_center
            self.player_last_moved_at = now
            return False

        moved_distance = hypot(
            player_center[0] - self.player_motion_anchor[0],
            player_center[1] - self.player_motion_anchor[1],
        )
        if moved_distance < PLAYER_STATIONARY_MOVEMENT_THRESHOLD:
            return False

        self.player_motion_anchor = player_center
        self.player_last_moved_at = now
        return True

    def determine_action_context(
        self,
        player: tuple[int, int, int, int],
        monsters: list[tuple[tuple[int, int, int, int], str]],
        monsters_in_range: list[tuple[tuple[int, int, int, int], str]],
        ropes: list[tuple[int, int, int, int]],
        scroll_target: ScrollMatch | None,
    ) -> str:
        player_x, player_y = coordinates_center(player)

        if scroll_target is not None:
            scroll_x, scroll_y = coordinates_center(scroll_target.coordinates)
            horizontal_gap = scroll_x - player_x
            vertical_gap = scroll_y - player_y
            if vertical_gap <= -SCROLL_PLATFORM_VERTICAL_OFFSET:
                target_rope = choose_rope_for_target_x(
                    player,
                    ropes,
                    scroll_x,
                )
                return "scroll_up_rope" if target_rope else "scroll_up_search"
            if vertical_gap >= SCROLL_PLATFORM_VERTICAL_OFFSET:
                return (
                    "scroll_down_align"
                    if abs(horizontal_gap) > SCROLL_DROP_X_TOLERANCE
                    else "scroll_down_drop"
                )
            if abs(horizontal_gap) > SCROLL_X_ALIGNMENT_TOLERANCE:
                return "scroll_same_move"
            if vertical_gap <= -SCROLL_JUMP_MIN_VERTICAL_OFFSET:
                return "scroll_same_jump"
            return "scroll_collect"

        if monsters_in_range:
            return "combat"

        above, below = split_monsters_vertically(player, monsters)
        same_level_monsters = [
            monster
            for monster in monsters
            if monster_is_within_same_platform_chase_range(
                player,
                monster[0],
            )
        ]
        if same_level_monsters:
            return "move_on_level"

        vertical_direction = choose_vertical_exploration_direction(
            above,
            below,
            self.explore_direction,
        )

        if vertical_direction == "down":
            return "navigate_down"

        target_x = player_x
        if above:
            target_x = sum(
                coordinates_center(monster[0])[0] for monster in above
            ) / len(above)
        target_rope = choose_rope_for_target_x(player, ropes, target_x)
        if target_rope is None:
            return "navigate_up_search"

        rope_x, _rope_y = coordinates_center(target_rope)
        horizontal_gap = abs(rope_x - player_x)
        if horizontal_gap > ROPE_FINE_ALIGNMENT_DISTANCE:
            return "navigate_up_coarse"
        if horizontal_gap > ROPE_CENTER_ENTER_TOLERANCE:
            return "navigate_up_fine"
        return "navigate_up_confirm"

    def update_stationary_recovery(
        self,
        now: float,
        player_moved: bool,
        action_context: str,
    ) -> str | None:
        if self.stationary_recovery_direction is not None:
            if (
                player_moved
                or action_context != self.stationary_recovery_context
            ):
                self.inputs.set_horizontal(None)
                self.stationary_recovery_direction = None
                self.stationary_recovery_started_at = 0.0
                self.stationary_recovery_context = None
                self.player_last_moved_at = now
                self.set_state("idle", now)
                return None

            if (
                now - self.stationary_recovery_started_at
                >= PLAYER_STATIONARY_RECOVERY_DURATION
            ):
                self.stationary_recovery_direction = random.choice(
                    (LEFT_KEY, RIGHT_KEY)
                )
                self.stationary_recovery_started_at = now

            self.inputs.set_horizontal(self.stationary_recovery_direction)
            self.facing_direction = self.stationary_recovery_direction
            self.state = "stationary_recovery"
            elapsed = now - self.stationary_recovery_started_at
            return (
                "STATIONARY RECOVERY: "
                f"{self.stationary_recovery_direction.upper()} "
                f"{elapsed:.1f}/{PLAYER_STATIONARY_RECOVERY_DURATION:.1f}s"
            )

        if now - self.player_last_moved_at < PLAYER_STATIONARY_TIMEOUT:
            return None

        self.inputs.release_navigation()
        self.reset_walk_target()
        self.reset_exploration()
        self.reset_rope_alignment()
        self.stationary_recovery_direction = random.choice(
            (LEFT_KEY, RIGHT_KEY)
        )
        self.stationary_recovery_started_at = now
        self.stationary_recovery_context = action_context
        self.inputs.set_horizontal(self.stationary_recovery_direction)
        self.facing_direction = self.stationary_recovery_direction
        self.state = "stationary_recovery"
        return (
            "STATIONARY RECOVERY: "
            f"{self.stationary_recovery_direction.upper()} "
            f"0.0/{PLAYER_STATIONARY_RECOVERY_DURATION:.1f}s"
        )

    def begin_drop(self, now: float) -> str:
        self.inputs.release_navigation()
        self.reset_walk_target()
        self.reset_exploration()
        self.reset_rope_alignment()
        self.inputs.key_down(DOWN_KEY)
        self.set_state("drop_prepare", now)
        return "DROP: hold down before jump"

    def begin_rope_escape(self, now: float) -> str:
        self.inputs.release_navigation()
        self.reset_walk_target()
        self.reset_rope_alignment()
        self.inputs.key_down(RIGHT_KEY)
        self.facing_direction = RIGHT_KEY
        self.set_state("rope_escape_prepare", now)
        return "DROP: escape rope to the right"

    def begin_climb(
        self,
        now: float,
        exit_direction: str,
        rope_x: float,
    ) -> str:
        self.inputs.release_navigation()
        self.reset_walk_target()
        self.reset_exploration()
        self.reset_rope_alignment()
        self.inputs.key_down(JUMP_KEY)
        self.climb_alt_started_at = now
        self.climb_seen_rope = False
        self.climb_exit_direction = exit_direction
        self.climb_rope_x = rope_x
        self.climb_top_candidate_at = None
        self.climb_best_top_gap = None
        self.climb_last_progress_at = now
        self.set_state("climb_prepare", now)
        return "CLIMB: jump toward rope"

    def begin_climb_dismount(self, now: float, reason: str) -> str:
        """Jump sideways at the rope top instead of releasing UP in place."""
        self.inputs.key_up(UP_KEY)
        self.inputs.key_up(JUMP_KEY)
        direction = self.climb_exit_direction or self.facing_direction or RIGHT_KEY
        self.climb_exit_direction = direction
        self.inputs.set_horizontal(direction)
        self.facing_direction = direction
        self.climb_dismount_attempts = 1
        self.set_state("climb_dismount_prepare", now)
        return f"CLIMB: {reason}; dismount {direction.upper()}"

    def begin_unstick(self, now: float) -> str:
        self.inputs.release_navigation()
        self.reset_walk_target()
        self.reset_exploration()
        self.reset_rope_alignment()
        self.unstick_direction = random.choice((LEFT_KEY, RIGHT_KEY))
        self.inputs.key_down(self.unstick_direction)
        self.facing_direction = self.unstick_direction
        self.combat_started_at = None
        self.set_state("unstick_move", now)
        return f"UNSTICK: hold {self.unstick_direction.upper()}"

    def continue_timed_action(
        self,
        now: float,
        on_or_near_rope: bool | None,
        player: tuple[int, int, int, int] | None = None,
        ropes: list[tuple[int, int, int, int]] | None = None,
    ) -> str | None:
        elapsed = now - self.state_started_at

        if self.state == "unstick_move":
            if elapsed >= UNSTICK_MOVE_DURATION:
                if self.unstick_direction is not None:
                    self.inputs.key_up(self.unstick_direction)
                self.unstick_direction = None
                self.set_state("idle", now)
                return "UNSTICK: completed"
            return f"UNSTICK: moving {self.unstick_direction.upper()}"

        if self.state == "drop_prepare":
            if elapsed >= DROP_PREPARE_DURATION:
                self.inputs.key_down(JUMP_KEY)
                self.set_state("drop_jump", now)
            return "DROP: preparing" if elapsed < DROP_PREPARE_DURATION else "DROP: jumping"

        if self.state == "drop_jump":
            if elapsed >= DROP_JUMP_DURATION:
                self.inputs.key_up(JUMP_KEY)
                self.inputs.key_up(DOWN_KEY)
                self.set_state("idle", now)
                return "DROP: completed"
            return "DROP: jumping"

        if self.state == "rope_escape_prepare":
            if elapsed >= ROPE_ESCAPE_PREPARE_DURATION:
                self.inputs.key_down(JUMP_KEY)
                self.set_state("rope_escape_jump", now)
            return (
                "DROP: moving away from rope"
                if elapsed < ROPE_ESCAPE_PREPARE_DURATION
                else "DROP: right jump"
            )

        if self.state == "rope_escape_jump":
            if elapsed >= ROPE_ESCAPE_JUMP_DURATION:
                self.inputs.key_up(JUMP_KEY)
                self.inputs.key_up(RIGHT_KEY)
                self.set_state("idle", now)
                return "DROP: rope escape completed"
            return "DROP: right jump"

        if self.state == "climb_dismount_prepare":
            if elapsed >= CLIMB_DISMOUNT_LEAD_DURATION:
                if self.inputs.tap(JUMP_KEY):
                    self.last_climb_dismount_jump_at = now
                    self.set_state("climb_dismount_jump", now)
                    return f"CLIMB: jump off {self.climb_exit_direction.upper()}"
                return "CLIMB: waiting to send dismount jump"
            return f"CLIMB: preparing exit {self.climb_exit_direction.upper()}"

        if self.state == "climb_dismount_jump":
            # One jump per dismount attempt. Do not spam ALT while leaving the rope.
            if elapsed >= CLIMB_DISMOUNT_JUMP_DURATION:
                direction = self.climb_exit_direction
                if direction is not None:
                    self.inputs.key_up(direction)

                # If we are still detected on/near the rope, retry in the SAME
                # direction. Reversing direction here caused the character to
                # bounce left/right at the rope top.
                if on_or_near_rope is not False:
                    if (
                        direction is not None
                        and self.climb_dismount_attempts < CLIMB_DISMOUNT_MAX_ATTEMPTS
                    ):
                        self.climb_dismount_attempts += 1
                        self.inputs.set_horizontal(direction)
                        self.facing_direction = direction
                        self.set_state("climb_dismount_prepare", now)
                        return (
                            "CLIMB: still on rope; retry same direction "
                            f"{self.climb_dismount_attempts}/"
                            f"{CLIMB_DISMOUNT_MAX_ATTEMPTS} "
                            f"{direction.upper()}"
                        )

                    # Do not force another left/right jump or a drop-jump.
                    # Release controls and let the normal navigation logic
                    # re-evaluate the next frame.
                    self.inputs.release_navigation()
                    self.reset_climb_tracking()
                    self.set_state("idle", now)
                    return "CLIMB: dismount unresolved; released controls"

                self.reset_climb_tracking()
                self.set_state("idle", now)
                return "CLIMB: dismount completed"
            return f"CLIMB: jumping off {self.climb_exit_direction.upper()}"

        if self.state == "climb_prepare":
            if elapsed >= CLIMB_JUMP_LEAD_DURATION:
                self.inputs.key_down(UP_KEY)
                self.climb_up_started_at = now
                self.set_state("climb_up", now)
            return (
                "CLIMB: holding jump"
                if elapsed < CLIMB_JUMP_LEAD_DURATION
                else "CLIMB: trying to attach"
            )

        if self.state != "climb_up":
            return None

        if now - self.climb_alt_started_at >= CLIMB_ALT_HOLD_DURATION:
            self.inputs.key_up(JUMP_KEY)

        if on_or_near_rope:
            self.climb_seen_rope = True

        if not self.climb_seen_rope:
            if now - self.climb_up_started_at >= CLIMB_ATTACH_TIMEOUT:
                self.stop_navigation(now)
                return "CLIMB: attach failed; retrying"
            return "CLIMB: trying to attach"

        top_gap = None
        if (
            player is not None
            and ropes is not None
            and self.climb_rope_x is not None
        ):
            top_gap = player_rope_top_gap(player, ropes, self.climb_rope_x)

        if top_gap is not None and (
            self.climb_best_top_gap is None
            or top_gap
            <= self.climb_best_top_gap - CLIMB_PROGRESS_MIN_PIXELS
        ):
            self.climb_best_top_gap = top_gap
            self.climb_last_progress_at = now

        at_rope_top = (
            top_gap is not None
            and top_gap <= CLIMB_TOP_VERTICAL_MARGIN
        )
        if at_rope_top:
            if self.climb_top_candidate_at is None:
                self.climb_top_candidate_at = now
            elif now - self.climb_top_candidate_at >= CLIMB_TOP_CONFIRM_DURATION:
                return self.begin_climb_dismount(now, "rope top detected")
        else:
            self.climb_top_candidate_at = None

        climb_up_elapsed = now - self.climb_up_started_at
        stalled_at_rope_end = (
            top_gap is not None
            and self.climb_best_top_gap is not None
            and climb_up_elapsed >= CLIMB_STALL_MIN_UP_DURATION
            and now - self.climb_last_progress_at
            >= CLIMB_STALL_CONFIRM_DURATION
        )
        if stalled_at_rope_end:
            return self.begin_climb_dismount(now, "upward movement stopped")

        if climb_up_elapsed >= CLIMB_UP_MAX_DURATION:
            return self.begin_climb_dismount(now, "upward safety timeout")

        return "CLIMB: moving up"

    def navigate_up_to_x(
        self,
        now: float,
        player: tuple[int, int, int, int],
        ropes: list[tuple[int, int, int, int]],
        target_x: float,
        on_or_near_rope: bool,
        fallback_to_drop: bool,
        status_prefix: str = "",
    ) -> str:
        prefix = f"{status_prefix} " if status_prefix else ""
        target_rope = choose_rope_for_target_x(player, ropes, target_x)
        if target_rope is None:
            self.reset_rope_alignment()

            if self.rope_search_started_at is None:
                self.rope_search_started_at = now
                self.rope_search_direction_started_at = now
                self.rope_search_direction = random.choice(
                    (LEFT_KEY, RIGHT_KEY)
                )

            search_elapsed = now - self.rope_search_started_at
            if search_elapsed >= ROPE_SEARCH_TIMEOUT:
                if fallback_to_drop:
                    self.inputs.set_horizontal(None)
                    self.reset_rope_search()
                    self.explore_direction = "down"
                    if on_or_near_rope:
                        return self.begin_rope_escape(now)
                    return self.begin_drop(now)

                self.rope_search_started_at = now
                self.rope_search_direction_started_at = now
                self.rope_search_direction = random.choice(
                    (LEFT_KEY, RIGHT_KEY)
                )
                search_elapsed = 0.0

            if (
                now - self.rope_search_direction_started_at
                >= ROPE_SEARCH_DIRECTION_DURATION
            ):
                self.rope_search_direction = (
                    RIGHT_KEY
                    if self.rope_search_direction == LEFT_KEY
                    else LEFT_KEY
                )
                self.rope_search_direction_started_at = now

            self.inputs.set_horizontal(self.rope_search_direction)
            self.facing_direction = self.rope_search_direction
            self.state = "search_rope"
            return (
                f"{prefix}SEARCH ROPE: {self.rope_search_direction.upper()} "
                f"{search_elapsed:.1f}/{ROPE_SEARCH_TIMEOUT:.1f}s"
            )

        self.reset_rope_search()

        player_x, _player_y = coordinates_center(player)
        rope_x, _rope_y = coordinates_center(target_rope)
        horizontal_gap = rope_x - player_x
        absolute_gap = abs(horizontal_gap)
        desired_direction = RIGHT_KEY if horizontal_gap > 0 else LEFT_KEY

        confirming_same_rope = (
            self.rope_centered_at is not None
            and self.confirmed_rope_x is not None
            and abs(rope_x - self.confirmed_rope_x)
            <= ROPE_CENTER_EXIT_TOLERANCE
        )
        if confirming_same_rope and absolute_gap <= ROPE_CENTER_EXIT_TOLERANCE:
            self.inputs.set_horizontal(None)
            centered_duration = now - self.rope_centered_at
            if centered_duration < ROPE_CENTER_CONFIRM_DURATION:
                self.state = "confirm_rope_alignment"
                return (
                    f"{prefix}ALIGN ROPE: centered "
                    f"{centered_duration:.1f}/{ROPE_CENTER_CONFIRM_DURATION:.1f}s"
                )
            if target_x > rope_x + ROPE_CENTER_ENTER_TOLERANCE:
                exit_direction = RIGHT_KEY
            elif target_x < rope_x - ROPE_CENTER_ENTER_TOLERANCE:
                exit_direction = LEFT_KEY
            else:
                exit_direction = self.facing_direction or RIGHT_KEY
            return self.begin_climb(now, exit_direction, rope_x)

        if self.rope_centered_at is not None:
            self.reset_rope_alignment()

        if absolute_gap > ROPE_FINE_ALIGNMENT_DISTANCE:
            self.inputs.set_horizontal(desired_direction)
            self.facing_direction = desired_direction
            self.state = "align_rope_coarse"
            return f"{prefix}ALIGN ROPE HOLD: {desired_direction.upper()}"

        self.inputs.set_horizontal(None)
        if absolute_gap > ROPE_CENTER_ENTER_TOLERANCE:
            if now - self.last_rope_align_tap_at >= ROPE_ALIGN_TAP_INTERVAL:
                if self.inputs.tap(desired_direction):
                    self.facing_direction = desired_direction
                self.last_rope_align_tap_at = now
            self.state = "align_rope_fine"
            return f"{prefix}ALIGN ROPE TAP: {desired_direction.upper()}"

        self.rope_centered_at = now
        self.confirmed_rope_x = rope_x
        self.state = "confirm_rope_alignment"
        return (
            f"{prefix}ALIGN ROPE: centered "
            f"0.0/{ROPE_CENTER_CONFIRM_DURATION:.1f}s"
        )

    def update_scroll_navigation(
        self,
        now: float,
        player: tuple[int, int, int, int],
        scroll: ScrollMatch,
        ropes: list[tuple[int, int, int, int]],
        on_or_near_rope: bool,
    ) -> str:
        player_x, player_y = coordinates_center(player)
        scroll_x, scroll_y = coordinates_center(scroll.coordinates)
        horizontal_gap = scroll_x - player_x
        vertical_gap = scroll_y - player_y

        self.combat_started_at = None
        self.reset_walk_target()

        if vertical_gap <= -SCROLL_PLATFORM_VERTICAL_OFFSET:
            return self.navigate_up_to_x(
                now,
                player,
                ropes,
                scroll_x,
                on_or_near_rope,
                fallback_to_drop=False,
                status_prefix="SCROLL",
            )

        self.reset_rope_search()
        self.reset_rope_alignment()

        if vertical_gap >= SCROLL_PLATFORM_VERTICAL_OFFSET:
            if abs(horizontal_gap) > SCROLL_DROP_X_TOLERANCE:
                desired_direction = (
                    RIGHT_KEY if horizontal_gap > 0 else LEFT_KEY
                )
                self.inputs.set_horizontal(desired_direction)
                self.facing_direction = desired_direction
                self.state = "scroll_align_drop"
                return f"SCROLL LOWER: MOVE {desired_direction.upper()}"

            if on_or_near_rope:
                return self.begin_rope_escape(now)
            return self.begin_drop(now)

        if abs(horizontal_gap) > SCROLL_X_ALIGNMENT_TOLERANCE:
            desired_direction = RIGHT_KEY if horizontal_gap > 0 else LEFT_KEY
            self.inputs.set_horizontal(desired_direction)
            self.facing_direction = desired_direction
            self.state = "scroll_move"
            return f"SCROLL SAME LEVEL: MOVE {desired_direction.upper()}"

        self.inputs.set_horizontal(None)
        if vertical_gap <= -SCROLL_JUMP_MIN_VERTICAL_OFFSET:
            if (
                now - self.last_scroll_jump_at >= SCROLL_JUMP_INTERVAL
                and self.inputs.tap(JUMP_KEY)
            ):
                self.last_scroll_jump_at = now
            self.state = "scroll_jump"
            return "SCROLL SAME LEVEL: JUMP"

        self.state = "scroll_collect"
        return "SCROLL: ALIGNED FOR PICKUP"

    def update(
        self,
        now: float,
        player: tuple[int, int, int, int] | None,
        monsters: list[tuple[tuple[int, int, int, int], str]],
        monsters_in_range: list[tuple[tuple[int, int, int, int], str]],
        ropes: list[tuple[int, int, int, int]],
        scroll_target: ScrollMatch | None = None,
    ) -> str:
        final_logout_status = self.update_final_logout(now)
        if final_logout_status is not None:
            return final_logout_status

        if not self.inputs.has_focus():
            self.stop_navigation(now)
            return "PAUSED: game window is not focused"

        break_status = self.update_scheduled_break(now)
        if break_status is not None:
            return break_status

        remaining_delay = AUTOMATION_START_DELAY - (now - self.started_at)
        if remaining_delay > 0:
            self.inputs.release_navigation()
            return f"STARTING IN: {remaining_delay:.1f}s"

        self.trigger_pickup_if_due(now)

        if player is None:
            recovery_climb_status = self.continue_player_lost_rope_climb(
                now,
                None,
            )
            if recovery_climb_status is not None:
                return recovery_climb_status
            if self.state in {
                "climb_prepare",
                "climb_up",
                "climb_dismount_prepare",
                "climb_dismount_jump",
                "drop_prepare",
                "drop_jump",
            }:
                timed_status = self.continue_timed_action(now, None, None, ropes)
                if timed_status is not None:
                    return timed_status
            return self.update_player_lost_recovery(now)

        self.player_last_seen_at = now
        on_or_near_rope = player_is_on_or_near_rope(player, ropes)
        recovery_direction = (
            self.player_lost_recovery_direction
            or self.player_lost_recovery_pending_direction
        )
        if recovery_direction is not None:
            recovered_with_up = (
                recovery_direction == UP_KEY
                and on_or_near_rope
            )
            if recovered_with_up:
                self.player_lost_rope_climb_started_at = now
                self.set_state("player_lost_rope_climb", now)
                self.inputs.key_down(UP_KEY)
            else:
                self.inputs.release_navigation()
            self.player_lost_recovery_direction = None
            self.player_lost_recovery_pending_direction = None
            self.player_lost_recovery_started_at = 0.0
            if recovered_with_up:
                return self.continue_player_lost_rope_climb(
                    now,
                    True,
                ) or "SELF RECOVERY CLIMB: UP"

        player_moved = self.observe_player_movement(now, player)
        recovery_climb_status = self.continue_player_lost_rope_climb(
            now,
            on_or_near_rope,
        )
        if recovery_climb_status is not None:
            return recovery_climb_status
        timed_status = self.continue_timed_action(
            now,
            on_or_near_rope,
            player,
            ropes,
        )
        if timed_status is not None:
            return timed_status

        action_context = self.determine_action_context(
            player,
            monsters,
            monsters_in_range,
            ropes,
            scroll_target,
        )
        stationary_status = self.update_stationary_recovery(
            now,
            player_moved,
            action_context,
        )
        if stationary_status is not None:
            return stationary_status

        if scroll_target is not None:
            if not self.scroll_mode_active:
                self.inputs.release_navigation()
                self.reset_walk_target()
                self.reset_exploration()
                self.reset_rope_alignment()
                self.scroll_mode_active = True
            return self.update_scroll_navigation(
                now,
                player,
                scroll_target,
                ropes,
                on_or_near_rope,
            )

        if self.scroll_mode_active:
            self.scroll_mode_active = False
            self.inputs.set_horizontal(None)
            self.reset_rope_search()
            self.reset_rope_alignment()
            self.set_state("idle", now)

        if monsters_in_range:
            self.inputs.release_navigation()
            self.reset_walk_target()
            self.reset_exploration()
            self.reset_rope_alignment()
            target_box, target_name = nearest_monster(player, monsters_in_range)
            desired_direction = (
                horizontal_direction_to(player, target_box)
                or self.facing_direction
                or RIGHT_KEY
            )

            if self.combat_started_at is None:
                self.combat_started_at = now
            elif now - self.combat_started_at >= STUCK_ATTACK_DURATION:
                return self.begin_unstick(now)

            if now - self.last_attack_at >= ATTACK_INTERVAL:
                attack_queued = self.inputs.tap_attack(desired_direction)
                if attack_queued:
                    self.facing_direction = desired_direction
                    self.last_attack_at = now
            self.state = "combat"
            return f"COMBAT: {target_name}"

        self.combat_started_at = None

        above, below = split_monsters_vertically(player, monsters)
        same_level_monsters = [
            monster
            for monster in monsters
            if monster_is_within_same_platform_chase_range(
                player,
                monster[0],
            )
        ]

        if same_level_monsters:
            self.reset_exploration()
            self.reset_rope_alignment()
            target_box, target_name = nearest_monster(player, same_level_monsters)
            desired_direction = horizontal_direction_to(player, target_box)
            self.inputs.set_horizontal(desired_direction)
            if desired_direction is not None:
                self.facing_direction = desired_direction
                self.walk_direction = desired_direction
                self.walk_target_name = target_name
                self.walk_target_last_seen_at = now
                self.state = "move_on_level"
                return f"MOVE {desired_direction.upper()}: {target_name}"
            self.reset_walk_target()
            self.set_state("idle", now)
            return f"WAIT: {target_name} is vertically out of range"

        if (
            self.state == "move_on_level"
            and self.walk_direction is not None
            and now - self.walk_target_last_seen_at <= WALK_TARGET_LOST_GRACE
        ):
            self.inputs.set_horizontal(self.walk_direction)
            return (
                f"MOVE {self.walk_direction.upper()}: "
                f"{self.walk_target_name or 'target'} temporarily missing"
            )

        self.reset_walk_target()
        self.explore_direction = choose_vertical_exploration_direction(
            above,
            below,
            self.explore_direction,
        )
        vertical_direction = self.explore_direction

        if vertical_direction == "down":
            if on_or_near_rope:
                self.explore_direction = "down"
                return self.begin_rope_escape(now)
            return self.begin_drop(now)

        if vertical_direction == "up":
            player_x, _player_y = coordinates_center(player)
            target_x = player_x
            if above:
                target_x = sum(
                    coordinates_center(monster[0])[0] for monster in above
                ) / len(above)
            return self.navigate_up_to_x(
                now,
                player,
                ropes,
                target_x,
                on_or_near_rope,
                fallback_to_drop=True,
            )

        return self.begin_drop(now)


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


def draw_attack_range(
    frame: np.ndarray,
    attack_range: tuple[int, int, int, int] | None,
    monsters_in_range: list[tuple[tuple[int, int, int, int], str]],
) -> None:
    """Draw the attack range and highlight monsters whose centers are inside it."""
    if attack_range is None:
        return

    range_x1, range_y1, range_x2, range_y2 = attack_range
    cv2.rectangle(
        frame,
        (range_x1, range_y1),
        (range_x2, range_y2),
        (255, 0, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Attack range: {len(monsters_in_range)}",
        (range_x1, max(20, range_y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    for (x1, y1, x2, y2), monster_name in monsters_in_range:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            frame,
            f"IN RANGE: {monster_name}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


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
    arguments = parse_arguments()
    pydirectinput.PAUSE = 0
    enable_dpi_awareness()
    if arguments.window_title:
        hwnd, title = find_window_by_title(
            arguments.window_title,
            max(1.0, arguments.window_timeout),
        )
    else:
        hwnd, title = choose_window()
    model = YOLO(str(MODEL_PATH))
    monster_motion_filter = MovingMonsterFilter()
    scroll_detector = ScrollDetector(
        load_scroll_templates(SCROLL_TEMPLATE_DIRECTORY)
    )
    ocr = RapidOCR()
    ocr_font = ImageFont.truetype(str(OCR_FONT_PATH), 24)
    rope_class_id = next(class_id for class_id, name in model.names.items() if name == "rope")
    player_class_id = next(class_id for class_id, name in model.names.items() if name == "player")
    ignored_class_ids = {
        class_id
        for class_id, name in model.names.items()
        if name in IGNORED_CLASS_NAMES
    }
    last_ocr_time = 0.0
    self_center: tuple[float, float] | None = None
    last_resource_ocr_time = 0.0
    hp_stat: tuple[int, int] | None = None
    mp_stat: tuple[int, int] | None = None
    last_resource_report: tuple[
        tuple[int, int] | None,
        tuple[int, int] | None,
    ] | None = None
    hp_skill_trigger = ResourcePotionTrigger(
        remaining_ratio_threshold=HP_REMAINING_RATIO_THRESHOLD
    )
    mp_potion_trigger = ResourcePotionTrigger(MP_DEFICIT_THRESHOLD)
    print(f"偵測視窗：{title}")
    print(
        f"自動補給：每 {RESOURCE_OCR_INTERVAL} 秒檢查，"
        f"HP <= {HP_REMAINING_RATIO_THRESHOLD:.0%} 按一次 {HP_POTION_KEY}，"
        f"MP 缺口 > {MP_DEFICIT_THRESHOLD} 按 {MP_POTION_KEY}"
    )
    print("DirectInput 只會在所選遊戲視窗位於前景時送出。")
    print("按 Q 或 Esc 結束。")

    preview_title = "MapleStory YOLO Detection"
    if arguments.preview:
        cv2.namedWindow(preview_title, cv2.WINDOW_AUTOSIZE)
    print(f"Annotated preview: {'ON' if arguments.preview else 'OFF'}")
    print("Press F8 or Ctrl+C to stop.")
    input_controller = DirectInputController(hwnd)
    automation_started_at = perf_counter()
    auto_logout_after = (
        arguments.auto_logout_minutes * 60.0
        if arguments.auto_logout_minutes > 0
        else None
    )
    auto_combat = AutoCombatController(
        input_controller,
        automation_started_at,
        auto_logout_after,
    )
    last_auto_state: str | None = None
    atexit.register(input_controller.shutdown)
    atexit.register(scroll_detector.shutdown)
    print(
        f"Auto combat starts after {AUTOMATION_START_DELAY:.1f}s; "
        f"{AUTO_PICKUP_KEY.upper()} every {AUTO_PICKUP_INTERVAL:.2f}s"
    )
    print(
        f"Scheduled break: {COMBAT_SESSION_DURATION / 60:.0f} min combat, "
        f"{OFFLINE_BREAK_DURATION / 60:.0f} min offline"
    )
    if auto_logout_after is None:
        print("Final auto logout: OFF")
    else:
        print(
            "Final auto logout: "
            f"{arguments.auto_logout_minutes:g} min after startup"
        )

    with MSS() as screen:
        while True:
            if arguments.stop_file and arguments.stop_file.exists():
                arguments.stop_file.unlink(missing_ok=True)
                print("External stop requested.")
                break
            if user32.GetAsyncKeyState(GLOBAL_STOP_KEY) & 1:
                print("F8 stop requested.")
                break

            region = get_client_region(hwnd)
            if region is None:
                print("遊戲視窗已關閉或最小化。")
                break

            started = perf_counter()
            frame = np.asarray(screen.grab(region))[:, :, :3]
            scroll_matches = scroll_detector.update(frame, perf_counter())

            resource_now = perf_counter()
            if resource_now - last_resource_ocr_time >= RESOURCE_OCR_INTERVAL:
                last_resource_ocr_time = resource_now
                new_hp_stat = read_resource_stat(frame, ocr, HP_ROI_OFFSETS)
                new_mp_stat = read_resource_stat(frame, ocr, MP_ROI_OFFSETS)
                hp_stat = new_hp_stat
                mp_stat = new_mp_stat
                if hp_skill_trigger.should_trigger(new_hp_stat, resource_now):
                    if input_controller.tap(HP_POTION_KEY):
                        hp_skill_trigger.mark_triggered(resource_now)
                        print(
                            f"HP POTION: {new_hp_stat[0]}/{new_hp_stat[1]}, "
                            f"sent {HP_POTION_KEY}"
                        )
                    else:
                        print("HP POTION: key send rejected (game not focused or key busy)")
                if mp_potion_trigger.should_trigger(new_mp_stat, resource_now):
                    if input_controller.tap(MP_POTION_KEY):
                        mp_potion_trigger.mark_triggered(resource_now)
                        print(
                            f"MP POTION: {new_mp_stat[0]}/{new_mp_stat[1]}, "
                            f"sent {MP_POTION_KEY}"
                        )
                    else:
                        print("MP POTION: key send rejected (game not focused or key busy)")

                resource_report = (new_hp_stat, new_mp_stat)
                if resource_report != last_resource_report:
                    hp_text = (
                        "OCR FAIL"
                        if new_hp_stat is None
                        else f"{new_hp_stat[0]}/{new_hp_stat[1]}"
                    )
                    mp_text = (
                        "OCR FAIL"
                        if new_mp_stat is None
                        else f"{new_mp_stat[0]}/{new_mp_stat[1]}"
                    )
                    print(f"RESOURCE OCR: HP {hp_text} | MP {mp_text}")
                    last_resource_report = resource_report

            result = model.predict(
                frame,
                imgsz=IMAGE_SIZE,
                conf=min(
                    ROPE_CONFIDENCE,
                    PLAYER_CONFIDENCE,
                    OTHER_CONFIDENCE,
                ),
                device=0,
                verbose=False,
            )[0]

            if result.boxes is not None and len(result.boxes) > 0:
                is_rope = result.boxes.cls == rope_class_id
                is_player = result.boxes.cls == player_class_id
                keep = (
                    is_rope & (result.boxes.conf >= ROPE_CONFIDENCE)
                ) | (
                    is_player & (result.boxes.conf >= PLAYER_CONFIDENCE)
                ) | (
                    ~is_rope
                    & ~is_player
                    & (result.boxes.conf >= OTHER_CONFIDENCE)
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
            ocr_self_index = None
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
                    ocr_self_index = matched_index

            self_index = ocr_self_index
            if self_index is None and self_center is not None and player_boxes:
                tracking_candidates = []
                climbing_rope_x = (
                    auto_combat.climb_rope_x
                    if auto_combat.state
                    in {
                        "climb_prepare",
                        "climb_up",
                        "climb_dismount_prepare",
                        "climb_dismount_jump",
                    }
                    else None
                )
                for index in range(len(player_boxes)):
                    candidate_center = player_box_center(player_boxes[index])
                    if (
                        climbing_rope_x is not None
                        and abs(candidate_center[0] - climbing_rope_x)
                        > CLIMB_ROPE_TRACK_HORIZONTAL_MARGIN
                    ):
                        continue
                    tracking_distance = hypot(
                        candidate_center[0] - self_center[0],
                        candidate_center[1] - self_center[1],
                    )
                    if tracking_distance <= SELF_TRACK_MAX_DISTANCE:
                        tracking_candidates.append(
                            (tracking_distance, index, candidate_center)
                        )

                tracking_candidates.sort()
                tracking_is_unambiguous = (
                    len(tracking_candidates) == 1
                    or (
                        len(tracking_candidates) >= 2
                        and tracking_candidates[1][0]
                        - tracking_candidates[0][0]
                        >= SELF_TRACK_AMBIGUITY_MARGIN
                    )
                )
                if tracking_candidates and tracking_is_unambiguous:
                    _distance, self_index, self_center = tracking_candidates[0]

            player_labels = []
            for index, player_box in enumerate(player_boxes):
                coordinates = tuple(int(value) for value in player_box.xyxy[0])
                player_labels.append((coordinates, index == self_index))

            monster_boxes = []
            rope_boxes = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_name = model.names[int(box.cls.item())]
                    coordinates = tuple(int(value) for value in box.xyxy[0])
                    if class_name in MONSTER_CLASS_NAMES:
                        monster_boxes.append((coordinates, class_name))
                    elif class_name == "rope":
                        rope_boxes.append(coordinates)

            self_coordinates = None
            attack_range = None
            monsters_in_range = []
            if self_index is not None:
                self_coordinates = player_labels[self_index][0]
                attack_range = expand_player_box(self_coordinates, frame.shape)
                monsters_in_range = [
                    monster
                    for monster in monster_boxes
                    if box_center_is_inside(monster[0], attack_range)
                ]

            scroll_target = choose_nearest_scroll(
                self_coordinates,
                scroll_matches,
            )

            automation_status = auto_combat.update(
                perf_counter(),
                self_coordinates,
                monster_boxes,
                monsters_in_range,
                rope_boxes,
                scroll_target,
            )
            if auto_combat.state != last_auto_state:
                print(
                    f"AUTO STATE: {last_auto_state or 'startup'} -> "
                    f"{auto_combat.state} | {automation_status}"
                )
                last_auto_state = auto_combat.state
            if auto_combat.final_logout_complete:
                print("Final auto logout complete. Detector is stopping.")
                break

            if arguments.preview:
                non_player_result = result
                if result.boxes is not None:
                    non_player_result = result[result.boxes.cls != player_class_id]
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
                        (0, 0, 255)
                        if hp_stat[0] / hp_stat[1]
                        <= HP_REMAINING_RATIO_THRESHOLD
                        else (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        annotated,
                        "HP: OCR FAIL",
                        (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
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
                        (0, 0, 255)
                        if mp_missing > MP_DEFICIT_THRESHOLD
                        else (255, 200, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        annotated,
                        "MP: OCR FAIL",
                        (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
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
                    (0, 255, 255)
                    if scroll_detector.active
                    else (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )
                draw_attack_range(annotated, attack_range, monsters_in_range)
                draw_scroll_matches(annotated, scroll_matches, scroll_target)
                annotated = draw_player_labels(annotated, player_labels, ocr_font)
                cv2.imshow(preview_title, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break

    input_controller.shutdown()
    scroll_detector.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
