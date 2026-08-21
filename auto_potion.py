from __future__ import annotations

import argparse
import ctypes
import re
from ctypes import wintypes
from dataclasses import dataclass
from time import perf_counter, sleep

import cv2
import numpy as np
import pydirectinput
from mss import mss
from rapidocr import RapidOCR

from capture_game_window import (
    enable_dpi_awareness,
    get_client_region,
    list_visible_windows,
)


STOP_KEY = 0x76  # F7
HP_KEY = "d"
MP_KEY = "f"
DEFAULT_HP_DEFICIT = 150
DEFAULT_MP_REMAINING = 150
MP_POTION_PRESSES = 2
MP_POTION_PRESS_INTERVAL = 0.12
MP_LOW_CONFIRMATIONS = 2
DEFAULT_SCAN_INTERVAL = 0.25
DEFAULT_POTION_COOLDOWN = 0.50
KEY_HOLD_DURATION = 0.060

# Bottom-center anchored HUD regions: left, right, top, bottom.
HP_ROI_OFFSETS = (-183, -104, -35, -15)
# Leave extra space on both sides so OCR sees all digits of 4-digit MP values.
MP_ROI_OFFSETS = (-93, 27, -35, -15)
OCR_SCALE = 5


user32 = ctypes.windll.user32
user32.GetForegroundWindow.argtypes = ()
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = (
    wintypes.HWND,
    ctypes.POINTER(wintypes.DWORD),
)
user32.GetWindowThreadProcessId.restype = wintypes.DWORD


def choose_window() -> tuple[int, str]:
    windows = list_visible_windows()
    if not windows:
        raise RuntimeError("找不到可擷取的視窗。")

    print("可擷取的視窗：")
    for index, (_hwnd, title, width, height) in enumerate(windows, start=1):
        print(f"  {index:>2}. {title} [{width}x{height}]")

    while True:
        answer = input("請輸入遊戲視窗編號：").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(windows):
            hwnd, title, _width, _height = windows[int(answer) - 1]
            return hwnd, title
        print(f"請輸入 1 到 {len(windows)} 之間的編號。")


def find_window_by_title(title_query: str, timeout: float) -> tuple[int, str]:
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


def resource_roi(
    frame: np.ndarray,
    offsets: tuple[int, int, int, int],
) -> np.ndarray | None:
    frame_height, frame_width = frame.shape[:2]
    center_x = frame_width // 2
    left, right, top, bottom = offsets
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
    return frame[y1:y2, x1:x2]


def read_resource_stat(
    frame: np.ndarray,
    ocr: RapidOCR,
    offsets: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    crop = resource_roi(frame, offsets)
    if crop is None or crop.size == 0:
        return None

    enlarged = cv2.resize(
        crop,
        None,
        fx=OCR_SCALE,
        fy=OCR_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    result = ocr(enlarged, use_det=False, use_cls=False, use_rec=True)
    if not result.txts:
        return None

    return parse_resource_text(" ".join(result.txts))


def parse_resource_text(text: str) -> tuple[int, int] | None:
    """Parse every digit OCR returned; current/max values are not 3-digit limited."""

    normalized = text.translate(
        str.maketrans({"S": "5", "s": "5", "O": "0", "I": "1", "l": "1"})
    )
    match = re.search(r"(\d+)\s*/\s*(\d+)", normalized)
    if not match:
        return None

    current, maximum = map(int, match.groups())
    if maximum <= 0 or current < 0 or current > maximum:
        return None
    return current, maximum


@dataclass
class MpPotionTrigger:
    """Trigger one potion batch after MP is reliably at or below the limit."""

    remaining_threshold: int
    confirmations_required: int = MP_LOW_CONFIRMATIONS
    armed: bool = False
    consecutive_low_readings: int = 0
    last_healthy_stat: tuple[int, int] | None = None

    def should_trigger(self, stat: tuple[int, int] | None) -> bool:
        if stat is None:
            self.consecutive_low_readings = 0
            return False

        current, maximum = stat
        if current > self.remaining_threshold:
            self.armed = True
            self.consecutive_low_readings = 0
            self.last_healthy_stat = stat
            return False

        if not self.armed:
            return False

        # Common OCR failures turn e.g. 1150 into 150 (missing first digit) or
        # 115 (missing last digit). Ignore those abrupt changes. Normal MP drain
        # will provide intermediate healthy readings before reaching 150.
        if self.last_healthy_stat is not None:
            previous, previous_maximum = self.last_healthy_stat
            truncated_maximum = previous_maximum >= 1000 and maximum < 1000
            truncated_current = (
                previous >= 1000
                and previous_maximum == maximum
                and current in (previous % 1000, previous // 10)
            )
            if truncated_maximum or truncated_current:
                self.consecutive_low_readings = 0
                return False

        self.consecutive_low_readings += 1
        return self.consecutive_low_readings >= self.confirmations_required

    def mark_triggered(self) -> None:
        self.armed = False
        self.consecutive_low_readings = 0


def window_process_id(hwnd: int) -> int:
    process_id = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    return int(process_id.value)


def game_is_foreground(hwnd: int) -> bool:
    foreground = user32.GetForegroundWindow()
    return (
        bool(foreground)
        and window_process_id(foreground) == window_process_id(hwnd)
    )


def tap_key(hwnd: int, key: str) -> bool:
    if not game_is_foreground(hwnd):
        return False
    pydirectinput.keyDown(key)
    try:
        sleep(KEY_HOLD_DURATION)
    finally:
        pydirectinput.keyUp(key)
    return True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MapleStory automatic HP/MP potion")
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
        "--hp-deficit",
        type=int,
        default=DEFAULT_HP_DEFICIT,
        help="Press D when missing HP is greater than this value (default: 150).",
    )
    parser.add_argument(
        "--mp-threshold",
        type=int,
        default=DEFAULT_MP_REMAINING,
        help="Press F twice when remaining MP is at or below this value (default: 150).",
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=DEFAULT_SCAN_INTERVAL,
        help="OCR scan interval in seconds (default: 0.25).",
    )
    parser.add_argument(
        "--potion-cooldown",
        type=float,
        default=DEFAULT_POTION_COOLDOWN,
        help="Minimum seconds between presses of the same potion key (default: 0.5).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the captured HP/MP regions for calibration.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.hp_deficit < 0 or arguments.mp_threshold < 0:
        raise ValueError("Potion thresholds cannot be negative.")
    if arguments.scan_interval <= 0 or arguments.potion_cooldown < 0:
        raise ValueError("Scan interval must be positive and cooldown cannot be negative.")

    enable_dpi_awareness()
    pydirectinput.PAUSE = 0
    if arguments.window_title:
        hwnd, title = find_window_by_title(
            arguments.window_title,
            max(1.0, arguments.window_timeout),
        )
    else:
        hwnd, title = choose_window()

    ocr = RapidOCR()
    last_hp_potion_at = float("-inf")
    last_mp_potion_at = float("-inf")
    mp_potion_trigger = MpPotionTrigger(arguments.mp_threshold)
    last_status: tuple[tuple[int, int] | None, tuple[int, int] | None] | None = None
    preview_title = "MapleStory Auto Potion"
    if arguments.preview:
        cv2.namedWindow(preview_title, cv2.WINDOW_AUTOSIZE)

    print(f"已選擇視窗：{title}")
    print(
        f"HP 缺少達 {arguments.hp_deficit} 按 {HP_KEY.upper()}；"
        f"MP 剩餘不超過 {arguments.mp_threshold} 時按 "
        f"{MP_KEY.upper()} {MP_POTION_PRESSES} 次"
    )
    print("遊戲視窗必須位於前景；按 F7 或 Ctrl+C 停止。")
    last_foreground_state: bool | None = None

    try:
        with mss() as screen:
            while True:
                if user32.GetAsyncKeyState(STOP_KEY) & 1:
                    print("F7 stop requested.")
                    break

                loop_started_at = perf_counter()
                region = get_client_region(hwnd)
                if region is None:
                    print("遊戲視窗已關閉，自動補水停止。")
                    break

                frame = np.asarray(screen.grab(region))[:, :, :3]
                hp_stat = read_resource_stat(frame, ocr, HP_ROI_OFFSETS)
                mp_stat = read_resource_stat(frame, ocr, MP_ROI_OFFSETS)
                now = perf_counter()

                is_foreground = game_is_foreground(hwnd)
                if is_foreground != last_foreground_state:
                    print(
                        "按鍵狀態：遊戲位於前景，可以補給。"
                        if is_foreground
                        else "按鍵狀態：遊戲不在前景，暫停送出 D/F。"
                    )
                    last_foreground_state = is_foreground

                if is_foreground:
                    if (
                        hp_stat is not None
                        and hp_stat[1] - hp_stat[0] >= arguments.hp_deficit
                        and now - last_hp_potion_at >= arguments.potion_cooldown
                        and tap_key(hwnd, HP_KEY)
                    ):
                        last_hp_potion_at = now
                        print(
                            f"補 HP：缺少 {hp_stat[1] - hp_stat[0]}，"
                            f"已送出 {HP_KEY.upper()}"
                        )

                    if (
                        mp_potion_trigger.should_trigger(mp_stat)
                        and now - last_mp_potion_at >= arguments.potion_cooldown
                    ):
                        presses_sent = 0
                        for press_index in range(MP_POTION_PRESSES):
                            if not tap_key(hwnd, MP_KEY):
                                break
                            presses_sent += 1
                            if press_index + 1 < MP_POTION_PRESSES:
                                sleep(MP_POTION_PRESS_INTERVAL)

                        if presses_sent:
                            last_mp_potion_at = perf_counter()
                            mp_potion_trigger.mark_triggered()
                            print(
                                f"補 MP：剩餘 {mp_stat[0]}，已送出 "
                                f"{MP_KEY.upper()} {presses_sent} 次"
                            )
                else:
                    # Do not accumulate low-MP confirmations while keystrokes
                    # are paused because the game is not in the foreground.
                    mp_potion_trigger.consecutive_low_readings = 0

                status = (hp_stat, mp_stat)
                if status != last_status:
                    hp_text = "OCR失敗" if hp_stat is None else f"{hp_stat[0]}/{hp_stat[1]}"
                    mp_text = "OCR失敗" if mp_stat is None else f"{mp_stat[0]}/{mp_stat[1]}"
                    print(f"HP {hp_text} | MP {mp_text}")
                    last_status = status

                if arguments.preview:
                    hp_crop = resource_roi(frame, HP_ROI_OFFSETS)
                    mp_crop = resource_roi(frame, MP_ROI_OFFSETS)
                    if hp_crop is not None and mp_crop is not None:
                        hp_preview = cv2.resize(
                            hp_crop,
                            None,
                            fx=OCR_SCALE,
                            fy=OCR_SCALE,
                            interpolation=cv2.INTER_NEAREST,
                        )
                        mp_preview = cv2.resize(
                            mp_crop,
                            None,
                            fx=OCR_SCALE,
                            fy=OCR_SCALE,
                            interpolation=cv2.INTER_NEAREST,
                        )
                        cv2.imshow(preview_title, np.hstack((hp_preview, mp_preview)))
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                        break

                elapsed = perf_counter() - loop_started_at
                sleep(max(0.0, arguments.scan_interval - elapsed))
    except KeyboardInterrupt:
        print("Interrupted by keyboard.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
