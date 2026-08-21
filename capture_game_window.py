from __future__ import annotations

import ctypes
import time
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from mss import MSS


PREVIEW_TITLE = "Game Window Capture"
GLOBAL_STOP_KEY = 0x77  # F8
CHARACTER_TEMPLATE_PATHS = (
    Path("assests/character/character_stand_right.png"),
    Path("assests/character/character_stand_left.png"),
    Path("assests/character/character_down_right.png"),
    Path("assests/character/character_down_left.png"),
)
MONSTER_TEMPLATE_PATHS = (
    Path("assests/monster/wood_01.png"),
    Path("assests/monster/wood_02.png"),
    Path("assests/monster/snail_01.png"),
    Path("assests/monster/snail_02.png"),
    Path("assests/monster/slime_01.png"),
    Path("assests/monster/slime_02.png"),
)
MATCH_THRESHOLD = 0.80
MONSTER_MATCH_THRESHOLD = 0.88
MONSTER_ZNCC_THRESHOLD = 0.78
MONSTER_MAX_MEAN_COLOR_ERROR = 42.0
MONSTER_NMS_THRESHOLD = 0.35
MAX_MONSTER_CANDIDATES_PER_TEMPLATE = 30
MONSTER_TRACK_MARGIN_X = 40
MONSTER_TRACK_MARGIN_Y = 30
MONSTER_TRACK_THRESHOLD = 0.55
MONSTER_TRACK_MAX_MEAN_COLOR_ERROR = 65.0
MONSTER_TRACK_MAX_FAILURES = 1
MONSTER_TRACK_MAX_FRAMES_WITHOUT_DETECTION = 15
ENABLE_CHARACTER_MATCHING = True
ENABLE_MONSTER_MATCHING = True
LOCAL_SEARCH_MARGIN_X = 120
LOCAL_SEARCH_MARGIN_Y = 80

user32 = ctypes.windll.user32


@dataclass
class MonsterTrack:
    track_id: int
    name: str
    box: tuple[int, int, int, int]
    score: float
    appearance: np.ndarray
    foreground_mask: np.ndarray | None
    frames_since_detection: int = 0
    tracking_failures: int = 0
    source: str = "D"
    visible: bool = True


def enable_dpi_awareness() -> None:
    """Keep Win32 window coordinates aligned with physical screen pixels."""
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            user32.SetProcessDPIAware()


def get_window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""

    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value.strip()


def get_client_region(hwnd: int) -> dict[str, int] | None:
    """Return the drawable client area in desktop coordinates."""
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None

    origin = wintypes.POINT(rect.left, rect.top)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None

    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None

    return {
        "left": origin.x,
        "top": origin.y,
        "width": width,
        "height": height,
    }


def list_visible_windows() -> list[tuple[int, str, int, int]]:
    windows: list[tuple[int, str, int, int]] = []
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )

    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True

        title = get_window_title(hwnd)
        region = get_client_region(hwnd)
        if not title or region is None:
            return True

        if region["width"] >= 320 and region["height"] >= 200:
            windows.append((hwnd, title, region["width"], region["height"]))
        return True

    callback_ref = callback_type(callback)
    user32.EnumWindows(callback_ref, 0)
    windows.sort(key=lambda item: item[1].casefold())
    return windows


def choose_window() -> tuple[int, str]:
    windows = list_visible_windows()
    if not windows:
        raise RuntimeError("找不到可擷取的視窗。")

    print("\n目前可擷取的視窗：")
    for index, (_hwnd, title, width, height) in enumerate(windows, start=1):
        print(f"  {index:>2}. {title}  [{width} x {height}]")

    while True:
        answer = input("\n請輸入遊戲視窗編號：").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(windows):
            hwnd, title, _width, _height = windows[int(answer) - 1]
            return hwnd, title
        print("輸入無效，請輸入清單中的編號。")


def save_capture(frame: np.ndarray) -> Path:
    output_dir = Path("captures")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"maplestory_{timestamp}.png"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"無法儲存截圖：{path}")
    return path.resolve()


def load_character_template(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    template_image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if template_image is None:
        raise FileNotFoundError(f"找不到角色模板：{path.resolve()}")

    if template_image.ndim != 3 or template_image.shape[2] not in (3, 4):
        raise ValueError("角色模板必須是 RGB 或 RGBA 圖片。")

    template = template_image[:, :, :3]
    mask: np.ndarray | None = None
    if template_image.shape[2] == 4:
        alpha = template_image[:, :, 3]
        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        if cv2.countNonZero(mask) == 0:
            raise ValueError("角色模板的 Alpha 通道完全透明。")

    return template, mask


def find_character(
    frame: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[float, tuple[int, int, int, int] | None]:
    template_height, template_width = template.shape[:2]
    frame_height, frame_width = frame.shape[:2]
    if template_width > frame_width or template_height > frame_height:
        return 0.0, None

    if mask is not None:
        result = cv2.matchTemplate(
            frame,
            template,
            cv2.TM_CCORR_NORMED,
            mask=mask,
        )
    else:
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

    result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _min_score, max_score, _min_location, max_location = cv2.minMaxLoc(result)
    x, y = max_location
    box = (x, y, x + template_width, y + template_height)
    return float(max_score), box


def find_monsters_for_template(
    frame: np.ndarray,
    template_name: str,
    template: np.ndarray,
    mask: np.ndarray | None,
) -> list[
    tuple[float, str, tuple[int, int, int, int], np.ndarray | None]
]:
    template_height, template_width = template.shape[:2]
    frame_height, frame_width = frame.shape[:2]
    if template_width > frame_width or template_height > frame_height:
        return []

    if mask is not None:
        result = cv2.matchTemplate(
            frame,
            template,
            cv2.TM_CCORR_NORMED,
            mask=mask,
        )
    else:
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

    result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)

    # TM_CCORR_NORMED is fast, but dark or nearly uniform regions can still get
    # a deceptively high score. Use it only to propose separated candidates;
    # every proposal is verified below with zero-mean structure and color error.
    candidate_map = result.copy()
    candidates: list[tuple[float, int, int]] = []
    suppress_x = max(3, template_width // 2)
    suppress_y = max(3, template_height // 2)
    for _ in range(MAX_MONSTER_CANDIDATES_PER_TEMPLATE):
        _min_score, max_score, _min_location, max_location = cv2.minMaxLoc(
            candidate_map
        )
        if max_score < MONSTER_MATCH_THRESHOLD:
            break

        x, y = max_location
        candidates.append((float(max_score), x, y))
        candidate_map[
            max(0, y - suppress_y) : min(candidate_map.shape[0], y + suppress_y + 1),
            max(0, x - suppress_x) : min(candidate_map.shape[1], x + suppress_x + 1),
        ] = -1.0

    monster_name = template_name.rsplit("_", 1)[0]
    detections = []
    foreground = mask > 0 if mask is not None else np.ones(
        (template_height, template_width), dtype=bool
    )
    template_pixels = template[foreground].astype(np.float32)
    template_centered = template_pixels - template_pixels.mean(
        axis=0, keepdims=True
    )
    template_norm = float(np.linalg.norm(template_centered))

    for score, x, y in candidates:
        patch = frame[y : y + template_height, x : x + template_width]
        patch_pixels = patch[foreground].astype(np.float32)

        mean_color_error = float(np.mean(np.abs(template_pixels - patch_pixels)))
        if mean_color_error > MONSTER_MAX_MEAN_COLOR_ERROR:
            continue

        patch_centered = patch_pixels - patch_pixels.mean(axis=0, keepdims=True)
        denominator = template_norm * float(np.linalg.norm(patch_centered))
        if denominator <= 1e-6:
            continue
        zncc = float(np.sum(template_centered * patch_centered) / denominator)
        if zncc < MONSTER_ZNCC_THRESHOLD:
            continue

        detections.append(
            (
                score,
                monster_name,
                (
                    int(x),
                    int(y),
                    int(x + template_width),
                    int(y + template_height),
                ),
                mask,
            )
        )
    return detections


def merge_monster_detections(
    detections: list[
        tuple[float, str, tuple[int, int, int, int], np.ndarray | None]
    ],
) -> list[
    tuple[float, str, tuple[int, int, int, int], np.ndarray | None]
]:
    if not detections:
        return []

    boxes_xywh = []
    scores = []
    for score, _name, (left, top, right, bottom), _mask in detections:
        boxes_xywh.append((left, top, right - left, bottom - top))
        scores.append(score)

    selected = cv2.dnn.NMSBoxes(
        boxes_xywh,
        scores,
        score_threshold=MONSTER_MATCH_THRESHOLD,
        nms_threshold=MONSTER_NMS_THRESHOLD,
    )
    selected_indices = np.asarray(selected).reshape(-1)
    merged = [detections[int(index)] for index in selected_indices]
    merged.sort(key=lambda detection: detection[0], reverse=True)
    return merged


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    left, top, right, bottom = box
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def crop_box(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray | None:
    frame_height, frame_width = frame.shape[:2]
    left, top, right, bottom = box
    left = max(0, min(left, frame_width))
    top = max(0, min(top, frame_height))
    right = max(0, min(right, frame_width))
    bottom = max(0, min(bottom, frame_height))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right].copy()


def track_monster_appearance(
    frame: np.ndarray,
    track: MonsterTrack,
) -> tuple[float, tuple[int, int, int, int]] | None:
    """Track one monster using its appearance from the preceding frame."""
    template = track.appearance
    template_height, template_width = template.shape[:2]
    frame_height, frame_width = frame.shape[:2]
    left, top, right, bottom = track.box

    search_left = max(0, left - MONSTER_TRACK_MARGIN_X)
    search_top = max(0, top - MONSTER_TRACK_MARGIN_Y)
    search_right = min(frame_width, right + MONSTER_TRACK_MARGIN_X)
    search_bottom = min(frame_height, bottom + MONSTER_TRACK_MARGIN_Y)
    search = frame[search_top:search_bottom, search_left:search_right]
    if (
        search.shape[1] < template_width
        or search.shape[0] < template_height
    ):
        return None

    # A small blur makes pixel-art tracking tolerant of one-pixel animation
    # changes while retaining the monster's color and overall structure.
    search_soft = cv2.GaussianBlur(search, (3, 3), 0)
    template_soft = cv2.GaussianBlur(template, (3, 3), 0)
    result = cv2.matchTemplate(
        search_soft,
        template_soft,
        cv2.TM_CCOEFF_NORMED,
        mask=track.foreground_mask,
    )
    result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)

    # Prefer a nearby peak when two identical monsters are close together.
    result_height, result_width = result.shape
    expected_x = left - search_left
    expected_y = top - search_top
    grid_y, grid_x = np.ogrid[:result_height, :result_width]
    distance = np.sqrt(
        ((grid_x - expected_x) / max(1, MONSTER_TRACK_MARGIN_X)) ** 2
        + ((grid_y - expected_y) / max(1, MONSTER_TRACK_MARGIN_Y)) ** 2
    )
    ranked_result = result - distance.astype(np.float32) * 0.10
    _min_rank, _max_rank, _min_location, max_location = cv2.minMaxLoc(
        ranked_result
    )
    x, y = max_location
    score = float(result[y, x])
    if score < MONSTER_TRACK_THRESHOLD:
        return None

    new_left = search_left + x
    new_top = search_top + y
    new_box = (
        new_left,
        new_top,
        new_left + template_width,
        new_top + template_height,
    )
    new_appearance = crop_box(frame, new_box)
    if new_appearance is None or new_appearance.shape != template.shape:
        return None

    foreground = (
        track.foreground_mask > 0
        if track.foreground_mask is not None
        else np.ones((template_height, template_width), dtype=bool)
    )
    reference_pixels = template[foreground].astype(np.float32)
    tracked_pixels = new_appearance[foreground].astype(np.float32)
    mean_color_error = float(np.mean(np.abs(reference_pixels - tracked_pixels)))
    if mean_color_error > MONSTER_TRACK_MAX_MEAN_COLOR_ERROR:
        return None

    return score, new_box


def update_monster_tracks(
    frame: np.ndarray,
    tracks: list[MonsterTrack],
    strict_detections: list[
        tuple[float, str, tuple[int, int, int, int], np.ndarray | None]
    ],
    next_track_id: int,
) -> tuple[list[MonsterTrack], int]:
    """Combine strict template detections with short-range frame tracking."""
    for track in tracks:
        track.visible = False
        track.frames_since_detection += 1
        tracked = track_monster_appearance(frame, track)
        if tracked is None:
            track.tracking_failures += 1
            continue

        score, box = tracked
        track.box = box
        track.score = score
        track.tracking_failures = 0
        track.source = "T"
        track.visible = True

    matched_track_ids: set[int] = set()
    for score, name, detected_box, foreground_mask in strict_detections:
        detected_center_x, detected_center_y = box_center(detected_box)
        detected_width = detected_box[2] - detected_box[0]
        detected_height = detected_box[3] - detected_box[1]
        maximum_distance = max(30.0, max(detected_width, detected_height) * 0.9)

        nearest_track = None
        nearest_distance = float("inf")
        for track in tracks:
            if track.track_id in matched_track_ids or track.name != name:
                continue
            track_center_x, track_center_y = box_center(track.box)
            center_distance = float(
                np.hypot(
                    detected_center_x - track_center_x,
                    detected_center_y - track_center_y,
                )
            )
            if center_distance <= maximum_distance and center_distance < nearest_distance:
                nearest_track = track
                nearest_distance = center_distance

        appearance = crop_box(frame, detected_box)
        if appearance is None:
            continue

        if nearest_track is None:
            tracks.append(
                MonsterTrack(
                    track_id=next_track_id,
                    name=name,
                    box=detected_box,
                    score=score,
                    appearance=appearance,
                    foreground_mask=foreground_mask,
                )
            )
            matched_track_ids.add(next_track_id)
            next_track_id += 1
        else:
            nearest_track.box = detected_box
            nearest_track.score = score
            nearest_track.appearance = appearance
            nearest_track.foreground_mask = foreground_mask
            nearest_track.frames_since_detection = 0
            nearest_track.tracking_failures = 0
            nearest_track.source = "D"
            nearest_track.visible = True
            matched_track_ids.add(nearest_track.track_id)

    tracks = [
        track
        for track in tracks
        if track.tracking_failures <= MONSTER_TRACK_MAX_FAILURES
        and track.frames_since_detection <= MONSTER_TRACK_MAX_FRAMES_WITHOUT_DETECTION
    ]
    return tracks, next_track_id


def find_character_with_tracking(
    frame: np.ndarray,
    template_name: str,
    template: np.ndarray,
    mask: np.ndarray | None,
    previous_box: tuple[int, int, int, int] | None = None,
) -> tuple[
    float,
    str,
    tuple[int, int, int, int] | None,
    tuple[int, int, int, int],
]:
    def match_in_image(
        image: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> tuple[float, str, tuple[int, int, int, int] | None]:
        score, box = find_character(image, template, mask)
        if box is not None:
            left, top, right, bottom = box
            box = (
                left + offset_x,
                top + offset_y,
                right + offset_x,
                bottom + offset_y,
            )
        return score, template_name, box

    if previous_box is not None:
        frame_height, frame_width = frame.shape[:2]
        left, top, right, bottom = previous_box
        search_left = max(0, left - LOCAL_SEARCH_MARGIN_X)
        search_top = max(0, top - LOCAL_SEARCH_MARGIN_Y)
        search_right = min(frame_width, right + LOCAL_SEARCH_MARGIN_X)
        search_bottom = min(frame_height, bottom + LOCAL_SEARCH_MARGIN_Y)
        search_region = (
            search_left,
            search_top,
            search_right,
            search_bottom,
        )

        search_image = frame[
            search_top:search_bottom,
            search_left:search_right,
        ]
        local_match = match_in_image(search_image, search_left, search_top)
        if local_match[0] >= MATCH_THRESHOLD:
            return (*local_match, search_region)

    # Initial acquisition or local-search miss: scan the complete game frame.
    full_frame_region = (0, 0, frame.shape[1], frame.shape[0])
    return (*match_in_image(frame), full_frame_region)


def capture_window(hwnd: int, title: str) -> None:
    character_templates = []
    if ENABLE_CHARACTER_MATCHING:
        for template_path in CHARACTER_TEMPLATE_PATHS:
            template, mask = load_character_template(template_path)
            character_templates.append((template_path.stem, template, mask))

    monster_templates = []
    if ENABLE_MONSTER_MATCHING:
        for template_path in MONSTER_TEMPLATE_PATHS:
            template, mask = load_character_template(template_path)
            monster_templates.append((template_path.stem, template, mask))

    # WINDOW_AUTOSIZE keeps the preview image at the captured client's native size.
    cv2.namedWindow(PREVIEW_TITLE, cv2.WINDOW_AUTOSIZE)
    last_completed_time = time.perf_counter()
    smoothed_processed_fps = 0.0
    match_score = 0.0
    matched_template = ""
    character_box = None
    search_region = None
    monster_tracks: list[MonsterTrack] = []
    next_monster_track_id = 1
    minimized_message_shown = False

    print(f"\n正在擷取：{title}")
    print("預覽視窗有焦點時：按 S 儲存，按 Q 或 Esc 結束。")
    print("無論目前焦點在哪裡，都可以按 F8 結束程式。")
    print("請保持遊戲視窗沒有被其他視窗遮住。")
    if ENABLE_CHARACTER_MATCHING:
        template_names = ", ".join(
            name for name, _template, _mask in character_templates
        )
        print(f"角色模板：{template_names}，辨識門檻：{MATCH_THRESHOLD:.2f}")
    else:
        print("角色模板比對：已停用")

    with MSS() as screen_capture, ThreadPoolExecutor(
        max_workers=max(1, len(character_templates) + len(monster_templates)),
        thread_name_prefix="template-detector",
    ) as detector:
        while user32.IsWindow(hwnd):
            if user32.GetAsyncKeyState(GLOBAL_STOP_KEY) & 1:
                print("收到 F8，全域停止。")
                break

            if cv2.getWindowProperty(PREVIEW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                print("預覽視窗已關閉。")
                break

            if user32.IsIconic(hwnd):
                if not minimized_message_shown:
                    print("遊戲視窗已最小化，等待視窗恢復……")
                    minimized_message_shown = True
                time.sleep(0.1)
                continue

            minimized_message_shown = False
            region = get_client_region(hwnd)
            if region is None:
                time.sleep(0.05)
                continue

            screenshot = screen_capture.grab(region)
            frame = np.asarray(screenshot, dtype=np.uint8)[:, :, :3].copy()
            preview = frame.copy()

            character_futures = []
            monster_futures = []

            if ENABLE_CHARACTER_MATCHING:
                previous_box = (
                    character_box
                    if match_score >= MATCH_THRESHOLD
                    else None
                )
                character_futures = [
                    detector.submit(
                        find_character_with_tracking,
                        frame,
                        template_name,
                        character_template,
                        character_mask,
                        previous_box,
                    )
                    for template_name, character_template, character_mask
                    in character_templates
                ]

            if ENABLE_MONSTER_MATCHING:
                monster_futures = [
                    detector.submit(
                        find_monsters_for_template,
                        frame,
                        template_name,
                        monster_template,
                        monster_mask,
                    )
                    for template_name, monster_template, monster_mask
                    in monster_templates
                ]

            if character_futures:
                detection_results = [
                    future.result() for future in character_futures
                ]
                match_score, matched_template, character_box, search_region = max(
                    detection_results,
                    key=lambda match: match[0],
                )

            if monster_futures:
                monster_candidates = []
                for future in monster_futures:
                    monster_candidates.extend(future.result())
                strict_monster_detections = merge_monster_detections(
                    monster_candidates
                )
                monster_tracks, next_monster_track_id = update_monster_tracks(
                    frame,
                    monster_tracks,
                    strict_monster_detections,
                    next_monster_track_id,
                )
            else:
                monster_tracks = []

            if search_region is not None:
                search_left, search_top, search_right, search_bottom = search_region
                cv2.rectangle(
                    preview,
                    (search_left, search_top),
                    (max(search_left, search_right - 1), max(search_top, search_bottom - 1)),
                    (0, 0, 255),
                    1,
                )

            if character_box is not None and match_score >= MATCH_THRESHOLD:
                left, top, right, bottom = character_box
                cv2.rectangle(
                    preview,
                    (left, top),
                    (right, bottom),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    preview,
                    f"Character {match_score:.3f} ({matched_template})",
                    (left, max(top - 8, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            visible_monster_tracks = [
                track for track in monster_tracks if track.visible
            ]
            for monster_track in visible_monster_tracks:
                left, top, right, bottom = monster_track.box
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                monster_color = (
                    (0, 255, 255)
                    if monster_track.source == "D"
                    else (0, 165, 255)
                )
                cv2.rectangle(
                    preview,
                    (left, top),
                    (right, bottom),
                    monster_color,
                    2,
                )
                cv2.drawMarker(
                    preview,
                    (center_x, center_y),
                    monster_color,
                    cv2.MARKER_CROSS,
                    7,
                    1,
                    cv2.LINE_8,
                )
                cv2.putText(
                    preview,
                    (
                        f"{monster_track.name} {monster_track.source} "
                        f"{monster_track.score:.3f} "
                        f"({center_x},{center_y})"
                    ),
                    (left, max(top - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    monster_color,
                    1,
                    cv2.LINE_AA,
                )

            completed_time = time.perf_counter()
            completed_elapsed = max(
                completed_time - last_completed_time,
                1e-9,
            )
            current_processed_fps = 1.0 / completed_elapsed
            smoothed_processed_fps = (
                current_processed_fps
                if smoothed_processed_fps == 0.0
                else smoothed_processed_fps * 0.9 + current_processed_fps * 0.1
            )
            last_completed_time = completed_time

            cv2.putText(
                preview,
                (
                    f"Capture: {region['width']}x{region['height']}  "
                    f"FPS: {smoothed_processed_fps:.1f}  "
                    + (
                        f"Best: {match_score:.3f}"
                        if ENABLE_CHARACTER_MATCHING
                        else "Match: OFF"
                    )
                    + f"  Monsters: {len(visible_monster_tracks)}"
                ),
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(PREVIEW_TITLE, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord("s"):
                saved_path = save_capture(frame)
                print(f"已儲存截圖：{saved_path}")

    cv2.destroyAllWindows()
    print("擷取已結束。")


def main() -> None:
    enable_dpi_awareness()
    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)
    hwnd, title = choose_window()
    capture_window(hwnd, title)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n已由使用者中止。")
    except Exception as error:
        cv2.destroyAllWindows()
        raise SystemExit(f"錯誤：{error}") from error
