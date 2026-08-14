#!/usr/bin/env python3
"""
Plot and aggregate real-hardware UAV experiments.

Directory structure:

ROOT/
├── 2 UAVs/
│   └── exp 1/
│       ├── cf_0_direct_odom.csv
│       ├── cf_0_direct_status.csv
│       ├── priority_cf_0.csv
│       ├── cf_1_direct_odom.csv
│       ├── cf_1_direct_status.csv
│       └── priority_cf_1.csv
│
└── 3 UAVs/
    ├── exp 1/
    ├── exp 2/
    ├── ...
    └── exp 9/

Hardware data:
- cf_X_direct_odom.csv:
      t,x,y,z,yaw
- cf_X_direct_status.csv:
      time ... voltage
  The LAST column is used as voltage.
- priority_cf_X.csv:
      Time_s,Priority_0,...,Priority_6
  The FIRST column is time; Priority_0...Priority_5 are resources;
  Priority_6 is battery and is excluded from objective detection.
- No leader/photosynthesis data are used.

Times:
Every individual CSV is plotted on its own elapsed-time scale:
    t = timestamp - timestamp[0]

Crash:
A crash is detected when consecutive x/y positions move by more than
CRASH_DISPLACEMENT_THRESHOLD. When a crash is detected, the sample after
the jump is excluded and all time-series data for that UAV are truncated
using the crash timestamp.

Objective:
The objective is reached when, for EVERY resource column (Priority_0...5),
AT LEAST ONE UAV has a priority strictly below 1.

Example:
UAV 1: 1.0 1.0 1.0 0.99 0.8 0.0
UAV 2: 0.5 0.4 1.0 1.0 1.0 0.0
UAV 3: 1.0 1.0 0.2 1.0 1.0 0.0

Objective IS reached, because:
- Priority_0: UAV 2 < 1
- Priority_1: UAV 2 < 1
- Priority_2: UAV 3 < 1
- Priority_3: UAV 1 < 1
- Priority_4: UAV 1 < 1
- Priority_5: UAV 1/UAV 2/UAV 3 < 1

Map:
The hardware arena is represented by the dashed rectangle in the supplied
map. Red points are obstacles and blue circles are waypoints.

The map constants are deliberately kept together near the top so they can
be changed easily if the real arena dimensions differ.
"""

from __future__ import annotations

import argparse
import re
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Point, box
from shapely.ops import unary_union


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_SETTINGS = [
    "2 UAVs",
    "3 UAVs",
]

POSITION_FILE_RE = re.compile(
    r"^cf_(\d+)_direct_odom\.csv$",
    re.IGNORECASE,
)

STATUS_FILE_RE = re.compile(
    r"^cf_(\d+)_direct_status\.csv$",
    re.IGNORECASE,
)

PRIORITY_FILE_RE = re.compile(
    r"^priority_cf_(\d+)\.csv$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Crash detection
# ---------------------------------------------------------------------------

# Change this value easily according to the hardware experiment.
CRASH_DISPLACEMENT_THRESHOLD = 4.0  # metres

# ---------------------------------------------------------------------------
# Finish detection
# ---------------------------------------------------------------------------

FINISH_HOLD_TIME = 0.5  # seconds

# Tolerance used to determine whether the UAV is stationary.
# Position tolerance is in metres.
# Yaw tolerance is in radians.
FINISH_POSITION_TOLERANCE = 0.001  # metres
FINISH_YAW_TOLERANCE = 0.01       # radians

# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

COVERAGE_RADIUS = 2.5  # metres
COVERAGE_TIMESTEP = 0.25 # seconds
# Dashed rectangle visible in the supplied hardware map.
ARENA_XMIN = -5.0
ARENA_XMAX = 5.0
ARENA_YMIN = -5.0
ARENA_YMAX = 5.0

# Display limits of the complete map.
WORLD_XMIN = -5.0
WORLD_XMAX = 5.0
WORLD_YMIN = -5.0
WORLD_YMAX = 5.0

# ---------------------------------------------------------------------------
# Waypoints visible in the supplied map.
# ---------------------------------------------------------------------------

WAYPOINTS = [
    (-2.0, 3.5),
    (1.0, 2.0),
    (4.0, 3.0),
    (-2.0, 0.0),
    (3.0, -1.0),
]
HIDDEN_WAYPOINTS = [
    (3.0, 2.0),
]
# ---------------------------------------------------------------------------
# Red obstacle markers visible in the supplied map.
# These are used for plotting. They are treated as point markers and
# therefore do NOT subtract area from the coverage denominator.
# ---------------------------------------------------------------------------

OBSTACLES = []

# Top and bottom rows
for x in np.arange(-5.0, 5.01, 1.0):
    OBSTACLES.append((float(x), 5.0))
    OBSTACLES.append((float(x), -5.0))

# Left and right rows
for y in np.arange(-4.0, 4.01, 1.0):
    OBSTACLES.append((-5.0, float(y)))
    OBSTACLES.append((5.0, float(y)))

# Internal / additional red markers visible in the map
OBSTACLES.extend([
    (2.0, 1.0),
    (2.0, 1.5),
    (2.0, 2.0),
    (2.0, 4.0),
    (2.0, 4.5),
    (3.0, 1.0),
    (2.5, 1.0),
    (-4.5, 1.0),
    (-4.0, 1.0),
    (-3.5, 1.0),
    (-3.0, 1.0),
    (-2.5, 1.0),
    (-2.0, 1.0),
    (-1.5, 1.0),
    (-1.0, 1.0),
    (-0.5, 1.0),
    (0.0, 1.0),
])

DPI = 180
FIGSIZE = (10, 6)

# Number of points used for common-time interpolation in aggregate plots.
AGGREGATION_POINTS = 500


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def read_csv_numeric(path: Path) -> np.ndarray:
    """Read a CSV and convert all numeric columns to a numpy array."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read {path}: {exc}") from exc

    data = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Remove completely empty rows/columns.
    data = data[~np.all(np.isnan(data), axis=1)]
    data = data[:, ~np.all(np.isnan(data), axis=0)]

    return data


def read_csv_dataframe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read {path}: {exc}") from exc


def find_matching_file(folder: Path, regex: re.Pattern) -> Path | None:
    for path in folder.iterdir():
        if path.is_file() and regex.match(path.name):
            return path
    return None


def elapsed_time(timestamp: np.ndarray) -> np.ndarray:
    """Convert absolute timestamps to a time axis starting at zero."""
    timestamp = np.asarray(timestamp, dtype=float)

    if len(timestamp) == 0:
        return timestamp

    return timestamp - timestamp[0]


def make_valid_series(time: np.ndarray, values: np.ndarray):
    """Remove invalid samples and sort by time."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(values)

    mask = np.isfinite(time)

    if values.ndim == 1:
        mask &= np.isfinite(values)
    else:
        mask &= np.isfinite(values).all(axis=1)

    time = time[mask]
    values = values[mask]

    if len(time) == 0:
        return time, values

    order = np.argsort(time)
    time = time[order]
    values = values[order]

    # Remove duplicate timestamps, keeping the first occurrence.
    unique = np.concatenate(([True], np.diff(time) > 0))
    return time[unique], values[unique]


# =============================================================================
# MAP
# =============================================================================

def get_arena_polygon():
    return box(
        ARENA_XMIN,
        ARENA_YMIN,
        ARENA_XMAX,
        ARENA_YMAX,
    )


def plot_world_walls(ax):
    """
    Plot the hardware map.

    - Red points: obstacles
    - Blue circles: waypoints
    - Dashed rectangle: arena
    """
    # Arena
    ax.plot(
        [ARENA_XMIN, ARENA_XMAX, ARENA_XMAX, ARENA_XMIN, ARENA_XMIN],
        [ARENA_YMIN, ARENA_YMIN, ARENA_YMAX, ARENA_YMAX, ARENA_YMIN],
        linestyle="--",
        color="black",
        linewidth=2.5,
        zorder=20,
    )

    # Obstacles
    if OBSTACLES:
        ox, oy = zip(*OBSTACLES)
        ax.scatter(
            ox,
            oy,
            color="red",
            s=55,
            zorder=25,
            label="Obstacle",
        )

    # Waypoints
    if WAYPOINTS:
        wx, wy = zip(*WAYPOINTS)
        ax.scatter(
            wx,
            wy,
            facecolors="none",
            edgecolors="blue",
            linewidths=2.0,
            s=80,
            zorder=30,
            label="Waypoint",
        )
    if HIDDEN_WAYPOINTS:
        wx, wy = zip(*HIDDEN_WAYPOINTS)
        ax.scatter(
            wx,
            wy,
            facecolors="midnightblue",
            edgecolors="midnightblue",
            linewidths=2.0,
            s=80,
            zorder=30,
            label="Hidden Waypoint",
        )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_uav_data(exp_folder: Path, uav_id: str) -> dict:
    """Load odometry, voltage and priorities for one hardware UAV."""
    data = {}

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    position_path = exp_folder / f"cf_{uav_id}_direct_odom.csv"

    if position_path.exists():
        df = read_csv_dataframe(position_path)

        numeric = df.apply(pd.to_numeric, errors="coerce")

        if numeric.shape[1] < 5:
            raise ValueError(
                f"{position_path} must contain at least t,x,y,z,yaw."
            )

        arr = numeric.to_numpy(dtype=float)

        timestamp = arr[:, 0]
        position = arr[:, 1:5]

        timestamp, position = make_valid_series(timestamp, position)

        data["position_timestamp"] = timestamp
        data["position_time"] = elapsed_time(timestamp)
        data["position"] = position

    # ------------------------------------------------------------------
    # Voltage
    # ------------------------------------------------------------------

    status_path = exp_folder / f"cf_{uav_id}_direct_status.csv"

    if status_path.exists():
        df = read_csv_dataframe(status_path)
        numeric = df.apply(pd.to_numeric, errors="coerce")

        if numeric.shape[1] >= 2:
            arr = numeric.to_numpy(dtype=float)

            timestamp = arr[:, 0]
            voltage = arr[:, -1]

            timestamp, voltage = make_valid_series(
                timestamp,
                voltage,
            )

            data["voltage_timestamp"] = timestamp
            data["voltage_time"] = elapsed_time(timestamp)
            data["voltage"] = voltage

        elif numeric.shape[1] == 1:
            voltage = numeric.iloc[:, 0].to_numpy(dtype=float)
            mask = np.isfinite(voltage)

            data["voltage_timestamp"] = np.arange(np.sum(mask), dtype=float)
            data["voltage_time"] = np.arange(np.sum(mask), dtype=float)
            data["voltage"] = voltage[mask]

    # ------------------------------------------------------------------
    # Priorities
    # ------------------------------------------------------------------

    priority_path = exp_folder / f"priority_cf_{uav_id}.csv"

    if priority_path.exists():
        df = read_csv_dataframe(priority_path)
        numeric = df.apply(pd.to_numeric, errors="coerce")

        if numeric.shape[1] < 2:
            raise ValueError(
                f"{priority_path} does not contain priority columns."
            )

        arr = numeric.to_numpy(dtype=float)

        timestamp = arr[:, 0]
        priorities = arr[:, 1:]

        timestamp, priorities = make_valid_series(
            timestamp,
            priorities,
        )

        data["priority_timestamp"] = timestamp
        data["priority_time"] = elapsed_time(timestamp)
        data["priority"] = priorities

    return data


def discover_uavs(exp_folder: Path) -> list[str]:
    """Discover UAV IDs from cf_X_direct_odom.csv files."""
    ids = set()

    for path in exp_folder.iterdir():
        if not path.is_file():
            continue

        match = POSITION_FILE_RE.match(path.name)

        if match:
            ids.add(match.group(1))

    return sorted(ids, key=int)


def build_exp_data(exp_folder: Path) -> dict[str, dict]:
    result = {}

    for uav_id in discover_uavs(exp_folder):
        result[uav_id] = load_uav_data(exp_folder, uav_id)

    return result


# =============================================================================
# CRASH DETECTION
# =============================================================================

def detect_crash(position: np.ndarray,
                  timestamps: np.ndarray,
                  threshold: float):
    """
    Detect the first abnormal x/y displacement.

    The crash time is the timestamp of the first sample AFTER the jump.
    """
    if len(position) < 2:
        return None

    xy = position[:, :2]

    valid = np.isfinite(xy).all(axis=1)
    valid &= np.isfinite(timestamps)

    for i in range(1, len(position)):
        if not valid[i] or not valid[i - 1]:
            continue

        distance = np.linalg.norm(
            xy[i] - xy[i - 1]
        )

        if distance > threshold:
            return {
                "index": i,
                "timestamp": float(timestamps[i]),
                "displacement": float(distance),
            }

    return None


def trim_data_after_crash(data: dict,
                          crash_timestamp: float) -> dict:
    """
    Trim every time series using its own timestamp.

    Samples occurring after the crash timestamp are removed.
    """
    result = {}

    for key, values in data.items():

        if not isinstance(values, np.ndarray):
            result[key] = values
            continue

        if key == "position":
            timestamp_key = "position_timestamp"

        elif key == "voltage":
            timestamp_key = "voltage_timestamp"

        elif key == "priority":
            timestamp_key = "priority_timestamp"

        else:
            result[key] = values
            continue

        timestamp = data.get(timestamp_key)

        if timestamp is None:
            result[key] = values
            continue

        mask = timestamp <= crash_timestamp

        result[key] = values[mask]
        result[timestamp_key] = timestamp[mask]

        time_key = timestamp_key.replace(
            "_timestamp",
            "_time",
        )

        result[time_key] = elapsed_time(
            timestamp[mask]
        )

    return result


def apply_crash_detection(exp_data: dict[str, dict]):
    """
    Detect and trim crashes for all UAVs.

    Returns:
        crash_info: dict containing crash information.
    """
    crash_info = {}

    for uav_id, data in exp_data.items():

        if "position" not in data:
            crash_info[uav_id] = None
            continue

        crash = detect_crash(
            data["position"],
            data["position_timestamp"],
            CRASH_DISPLACEMENT_THRESHOLD,
        )

        crash_info[uav_id] = crash

        if crash is not None:
            print(
                f"    CRASH UAV {uav_id}: "
                f"iteration {crash['index']}, "
                f"displacement={crash['displacement']:.2f} m"
            )

            exp_data[uav_id] = trim_data_after_crash(
                data,
                crash["timestamp"],
            )

    return crash_info


def save_crash_info(exp_folder: Path, crash_info: dict):
    path = exp_folder / "crashinfo.txt"

    with path.open("w", encoding="utf-8") as f:

        f.write(
            f"Crash displacement threshold: "
            f"{CRASH_DISPLACEMENT_THRESHOLD:.3f} m\n\n"
        )

        any_crash = False

        for uav_id in sorted(crash_info, key=int):
            crash = crash_info[uav_id]

            if crash is None:
                f.write(
                    f"UAV {uav_id}: NO CRASH DETECTED\n"
                )
            else:
                any_crash = True

                f.write(
                    f"UAV {uav_id}: CRASH DETECTED\n"
                )
                f.write(
                    f"  Position sample index: "
                    f"{crash['index']}\n"
                )
                f.write(
                    f"  Absolute timestamp: "
                    f"{crash['timestamp']:.6f}\n"
                )
                f.write(
                    f"  Displacement: "
                    f"{crash['displacement']:.6f} m\n"
                )

        f.write("\n")
        f.write(
            "Experiment crash status: "
            + ("CRASH" if any_crash else "NO CRASH")
            + "\n"
        )


# =============================================================================
# TIME / MISSION FILES
# =============================================================================

def find_finish_timestamp(
    timestamp: np.ndarray,
    position: np.ndarray,
    hold_time: float = FINISH_HOLD_TIME,
):
    """
    Find the last timestamp at which x, y, z and yaw remain within
    the specified tolerances for at least hold_time seconds.

    The finish timestamp is the END of the last stationary interval.

    Returns:
        Absolute finish timestamp, or None if no valid interval exists.
    """
    if len(timestamp) < 2 or len(position) < 2:
        return None

    timestamp = np.asarray(timestamp, dtype=float)
    position = np.asarray(position, dtype=float)

    if position.shape[1] < 4:
        return None

    valid = (
        np.isfinite(timestamp)
        & np.isfinite(position[:, :4]).all(axis=1)
    )

    if not np.any(valid):
        return None

    timestamp = timestamp[valid]
    position = position[valid]

    if len(timestamp) < 2:
        return None

    # Differences between consecutive samples.
    delta_position = np.abs(
        position[1:, :3] - position[:-1, :3]
    )

    delta_yaw = np.abs(
        position[1:, 3] - position[:-1, 3]
    )

    # A transition is considered stationary when:
    #
    # |dx| <= position tolerance
    # |dy| <= position tolerance
    # |dz| <= position tolerance
    # |dyaw| <= yaw tolerance
    #
    # for consecutive samples.
    stationary = (
        np.all(
            delta_position
            <= FINISH_POSITION_TOLERANCE,
            axis=1,
        )
        & (
            delta_yaw
            <= FINISH_YAW_TOLERANCE
        )
    )

    last_finish_timestamp = None

    # Beginning of the current stationary interval.
    interval_start = timestamp[0]

    for i in range(1, len(timestamp)):

        if stationary[i - 1]:

            duration = (
                timestamp[i]
                - interval_start
            )

            if duration >= hold_time:
                last_finish_timestamp = (
                    timestamp[i]
                )

        else:
            # UAV moved outside the tolerance.
            interval_start = timestamp[i]
    if last_finish_timestamp is None:
        last_finish_timestamp = timestamp[len(timestamp) - 1]
    return last_finish_timestamp


def save_finish_times(
    exp_folder: Path,
    exp_data: dict[str, dict],
):
    """
    Save:
        timetofinish_cf_X.txt

    The finish time is the END of the last interval in which
    x, y, z and yaw remain within their respective tolerances
    for at least FINISH_HOLD_TIME seconds.

    The function also stores the elapsed finish time in:
        data["finish_time"]

    Returns:
        Dictionary:
            {
                uav_id: finish_time,
                ...
            }

        UAVs for which the finish condition is not reached have
        value None.
    """
    finish_times = {}

    for uav_id, data in exp_data.items():

        finish_time = None

        if (
            "position_timestamp" in data
            and "position" in data
        ):

            timestamp = data["position_timestamp"]
            position = data["position"]

            if len(timestamp) > 0:

                finish_timestamp = find_finish_timestamp(
                    timestamp,
                    position,
                )

                if finish_timestamp is not None:
                    finish_time = float(
                        finish_timestamp
                        - timestamp[0]
                    )

        # Keep the original finish time in the data structure.
        data["finish_time"] = finish_time

        finish_times[uav_id] = finish_time

        path = (
            exp_folder
            / f"timetofinish_cf_{uav_id}.txt"
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as f:

            if finish_time is None:
                f.write(
                    "Mission duration: NOT REACHED\n"
                )
            else:
                f.write(
                    f"Mission duration: "
                    f"{finish_time:.3f} seconds\n"
                )

    return finish_times

def trim_data_to_max_finish(
    exp_data: dict[str, dict],
    max_finish_time: float,
):
    """
    Trim all UAV time-series data to the maximum finish time
    of the experiment.

    The finish_time stored in each UAV is NOT modified.

    Each CSV has its own elapsed-time axis, therefore the
    corresponding absolute timestamp is reconstructed using:

        timestamp[0] + max_finish_time
    """
    for uav_id, data in exp_data.items():

        # --------------------------------------------------------------
        # Position
        # --------------------------------------------------------------

        if (
            "position_timestamp" in data
            and len(data["position_timestamp"]) > 0
        ):

            timestamp = data["position_timestamp"]

            limit = (
                timestamp[0]
                + max_finish_time
            )

            mask = timestamp <= limit

            data["position_timestamp"] = (
                timestamp[mask]
            )

            data["position"] = (
                data["position"][mask]
            )

            data["position_time"] = (
                elapsed_time(
                    data["position_timestamp"]
                )
            )

        # --------------------------------------------------------------
        # Voltage
        # --------------------------------------------------------------

        if (
            "voltage_timestamp" in data
            and len(data["voltage_timestamp"]) > 0
        ):

            timestamp = data["voltage_timestamp"]

            limit = (
                timestamp[0]
                + max_finish_time
            )

            mask = timestamp <= limit

            data["voltage_timestamp"] = (
                timestamp[mask]
            )

            data["voltage"] = (
                data["voltage"][mask]
            )

            data["voltage_time"] = (
                elapsed_time(
                    data["voltage_timestamp"]
                )
            )

        # --------------------------------------------------------------
        # Priorities
        # --------------------------------------------------------------

        if (
            "priority_timestamp" in data
            and len(data["priority_timestamp"]) > 0
        ):

            timestamp = data["priority_timestamp"]

            limit = (
                timestamp[0]
                + max_finish_time
            )

            mask = timestamp <= limit

            data["priority_timestamp"] = (
                timestamp[mask]
            )

            data["priority"] = (
                data["priority"][mask]
            )

            data["priority_time"] = (
                elapsed_time(
                    data["priority_timestamp"]
                )
            )


def load_saved_finish_time(path: Path) -> float | None:
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")

    match = re.search(
        r"Mission duration:\s*([0-9.eE+-]+)",
        text,
    )

    if not match:
        return None

    return float(match.group(1))


# =============================================================================
# OBJECTIVE DETECTION
# =============================================================================

def objective_reached_at_time(
    priority_states: dict[str, np.ndarray]
) -> bool:
    """
    Objective criterion:

    For EVERY resource column (excluding the last/battery column),
    at least ONE UAV must have priority < 1.

    Equivalently:
        min(priority[:, resource]) < 1
    for every resource.
    """
    if not priority_states:
        return False

    arrays = [
        values
        for values in priority_states.values()
        if values is not None
        and len(values) > 0
    ]

    if not arrays:
        return False

    n_resources = min(
        values.shape[0]
        for values in arrays
    )

    if n_resources <= 0:
        return False

    # Last priority column is battery and is excluded.
    resource_count = n_resources - 1

    if resource_count <= 0:
        return False

    for resource_index in range(resource_count):

        found = False

        for values in arrays:
            if values[resource_index] < 1.0:
                found = True
                break

        if not found:
            return False

    return True


def compute_objective_time(
    exp_data: dict[str, dict]
) -> float | None:
    """
    Determine the earliest elapsed time at which the objective is reached.

    Priority CSVs have independent absolute timestamps. The objective is
    evaluated using the real timestamps. The reported time is relative to
    the earliest first timestamp among all UAV priority files.

    At every event timestamp, the latest known priority sample of each UAV
    is used. The objective is valid only once all UAVs have at least one
    priority sample.
    """
    priority_data = {
        uav_id: data
        for uav_id, data in exp_data.items()
        if "priority" in data
        and "priority_timestamp" in data
        and len(data["priority_timestamp"]) > 0
    }

    if not priority_data:
        return None

    # All priority CSVs use their original absolute timestamp.
    reference_time = min(
        data["priority_timestamp"][0]
        for data in priority_data.values()
    )

    events = sorted(set(
        float(t)
        for data in priority_data.values()
        for t in data["priority_timestamp"]
    ))

    # Keep the most recent priority row for every UAV.
    current_index = {
        uav_id: -1
        for uav_id in priority_data
    }

    for event_time in events:

        for uav_id, data in priority_data.items():

            timestamps = data["priority_timestamp"]

            # Last sample <= current event time.
            idx = np.searchsorted(
                timestamps,
                event_time,
                side="right",
            ) - 1

            if idx >= 0:
                current_index[uav_id] = idx

        # Do not evaluate before every UAV has produced at least
        # one priority sample.
        if any(
            idx < 0
            for idx in current_index.values()
        ):
            continue

        states = {
            uav_id: priority_data[uav_id]["priority"][idx]
            for uav_id, idx in current_index.items()
        }

        if objective_reached_at_time(states):
            return float(event_time - reference_time)

    return None


def save_objective_time(exp_folder: Path,
                        objective_time: float | None):
    path = exp_folder / "timetoobjective.txt"

    with path.open("w", encoding="utf-8") as f:

        if objective_time is None:
            f.write(
                "Objective time: NOT REACHED\n"
            )
        else:
            f.write(
                f"Objective time: "
                f"{objective_time:.3f} seconds\n"
            )


def load_objective_time(path: Path) -> float | None:
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")

    match = re.search(
        r"Objective time:\s*([0-9.eE+-]+)",
        text,
    )

    if not match:
        return None

    return float(match.group(1))


# =============================================================================
# COVERAGE
# =============================================================================

def covered_area_geometry(positions: list[np.ndarray]):
    """Union of radius-COVERAGE_RADIUS circles around all visited positions."""
    circles = []

    for xy in positions:

        if len(xy) == 0:
            continue

        for point in xy:

            if not np.isfinite(point[:2]).all():
                continue

            circles.append(
                Point(
                    float(point[0]),
                    float(point[1]),
                ).buffer(COVERAGE_RADIUS)
            )

    if not circles:
        return None

    return unary_union(circles)


def safe_intersection(geometry, valid_area):
    """
    Robust Shapely intersection.

    buffer(0) is used to repair invalid geometries before intersection.
    """
    if geometry is None:
        return None

    try:
        geometry = geometry.buffer(0)
    except Exception:
        pass

    try:
        valid_area = valid_area.buffer(0)
    except Exception:
        pass

    try:
        return geometry.intersection(valid_area)
    except Exception:
        # Retry with a tiny topology repair.
        try:
            return geometry.buffer(0).intersection(
                valid_area.buffer(0)
            )
        except Exception:
            return None


def coverage_area_percentage(positions: list[np.ndarray]):
    arena = get_arena_polygon()
    coverage = covered_area_geometry(positions)

    if coverage is None:
        return 0.0, None

    clipped = safe_intersection(
        coverage,
        arena,
    )

    if clipped is None:
        return 0.0, None

    percentage = (
        clipped.area / arena.area
    ) * 100.0

    return percentage, clipped


def cumulative_coverage_area(
    time_position_pairs: list[tuple[np.ndarray, np.ndarray]]
):
    """
    Return cumulative covered-area percentage versus elapsed time.

    Coverage is computed as the union of circles with radius
    COVERAGE_RADIUS centered at all UAV positions visited up to
    each evaluation time.

    To avoid an extremely expensive geometry calculation at every
    CSV sample, the cumulative area is evaluated every
    COVERAGE_TIMESTEP seconds.
    """

    if not time_position_pairs:
        return np.array([]), np.array([])

    # --------------------------------------------------
    # Collect all position events
    # --------------------------------------------------

    events = []

    for time, xy in time_position_pairs:

        for t, point in zip(time, xy):

            if (
                np.isfinite(t)
                and np.isfinite(point[:2]).all()
            ):
                events.append(
                    (
                        float(t),
                        np.asarray(point[:2], dtype=float)
                    )
                )

    if not events:
        return np.array([]), np.array([])

    # Sort according to time
    events.sort(key=lambda x: x[0])

    # --------------------------------------------------
    # Arena
    # --------------------------------------------------

    arena = get_arena_polygon()

    # --------------------------------------------------
    # Cumulative geometry
    # --------------------------------------------------

    cumulative_coverage = None

    times = []
    areas = []

    batch_circles = []

    first_time = events[0][0]

    next_evaluation_time = first_time

    # --------------------------------------------------
    # Process events
    # --------------------------------------------------

    for t, point in events:

        # Create the coverage circle for this position.
        circle = Point(
            float(point[0]),
            float(point[1])
        ).buffer(
            COVERAGE_RADIUS,
            quad_segs=4
        )

        batch_circles.append(circle)

        # --------------------------------------------------
        # Evaluate coverage only every COVERAGE_TIMESTEP
        # --------------------------------------------------

        if t < next_evaluation_time:
            continue

        # Union all new circles accumulated since the
        # previous evaluation.
        if batch_circles:

            batch_geometry = unary_union(batch_circles)

            if cumulative_coverage is None:
                cumulative_coverage = batch_geometry
            else:
                cumulative_coverage = unary_union(
                    [
                        cumulative_coverage,
                        batch_geometry
                    ]
                )

            batch_circles = []

        # Clip coverage to the arena.
        clipped = safe_intersection(
            cumulative_coverage,
            arena
        )

        if clipped is not None:
            area_percentage = (
                clipped.area /
                arena.area *
                100.0
            )
        else:
            area_percentage = 0.0

        times.append(t)
        areas.append(area_percentage)

        next_evaluation_time += COVERAGE_TIMESTEP

    # --------------------------------------------------
    # Make sure the final position is included
    # --------------------------------------------------

    if batch_circles:

        batch_geometry = unary_union(
            batch_circles
        )

        if cumulative_coverage is None:
            cumulative_coverage = batch_geometry
        else:
            cumulative_coverage = unary_union(
                [
                    cumulative_coverage,
                    batch_geometry
                ]
            )

        clipped = safe_intersection(
            cumulative_coverage,
            arena
        )

        if clipped is not None:
            area_percentage = (
                clipped.area /
                arena.area *
                100.0
            )
        else:
            area_percentage = 0.0

        # Avoid duplicating the same final time.
        if not times or t > times[-1]:
            times.append(t)
            areas.append(area_percentage)
        else:
            areas[-1] = area_percentage

    return (
        np.asarray(times),
        np.asarray(areas)
    )

# =============================================================================
# PLOTS - INDIVIDUAL UAV
# =============================================================================

def plot_uav_position(
    uav_id: str,
    data: dict,
    out: Path,
    exp_name: str,
):
    if "position" not in data:
        return

    p = data["position"]
    t = data["position_time"]

    if len(p) == 0:
        return

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    labels = [
        "x [m]",
        "y [m]",
        "z [m]",
        "yaw [rad]",
    ]

    for i, ax in enumerate(axes):

        ax.plot(
            t,
            p[:, i],
        )

        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")

    fig.suptitle(
        f"{exp_name} - UAV {uav_id} - Position"
    )

    savefig(
        out / "position_time.png"
    )


def plot_uav_path(
    uav_id: str,
    data: dict,
    out: Path,
    exp_name: str,
):
    if "position" not in data:
        return

    p = data["position"]
    xy = p[:, :2]

    valid = np.isfinite(xy).all(axis=1)
    xy = xy[valid]

    if len(xy) == 0:
        return

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    plot_world_walls(ax)

    ax.plot(
        xy[:, 0],
        xy[:, 1],
        linewidth=1.2,
        label=f"UAV {uav_id}",
    )

    ax.scatter(
        xy[0, 0],
        xy[0, 1],
        marker="o",
        s=50,
        label="Start",
    )

    ax.scatter(
        xy[-1, 0],
        xy[-1, 1],
        marker="x",
        s=60,
        label="End",
    )

    coverage = covered_area_geometry([xy])
    clipped = safe_intersection(
        coverage,
        get_arena_polygon(),
    )

    if clipped is not None:

        if clipped.geom_type == "Polygon":

            x, y = clipped.exterior.xy

            ax.fill(
                x,
                y,
                alpha=0.15,
                label=f"Covered area",
            )

        elif clipped.geom_type == "MultiPolygon":

            first = True

            for polygon in clipped.geoms:

                x, y = polygon.exterior.xy

                ax.fill(
                    x,
                    y,
                    alpha=0.15,
                    label=(
                        f"Covered area"
                        if first
                        else None
                    ),
                )

                first = False

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.set_title(
        f"{exp_name} - UAV {uav_id} - 2D path"
    )

    ax.set_xlim(WORLD_XMIN, WORLD_XMAX)
    ax.set_ylim(WORLD_YMIN, WORLD_YMAX)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()

    savefig(
        out / "path_2d.png"
    )


def plot_uav_priorities(
    uav_id: str,
    data: dict,
    out: Path,
    exp_name: str,
):
    if "priority" not in data:
        return

    pr = data["priority"]
    t = data["priority_time"]

    if len(pr) == 0:
        return

    n = pr.shape[1]

    labels = []

    for i in range(n):

        if i == n - 1:
            labels.append("Battery")
        else:
            labels.append(
                f"Resource {i + 1}"
            )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    for i, label in enumerate(labels):
        ax.plot(
            t,
            pr[:, i],
            label=label,
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Priority")

    ax.set_title(
        f"{exp_name} - UAV {uav_id} - Resource priorities"
    )

    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    savefig(
        out / "priorities.png"
    )

    # Battery priority separately.
    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.plot(
        t,
        pr[:, -1],
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Battery priority")

    ax.set_title(
        f"{exp_name} - UAV {uav_id} - Battery priority"
    )

    ax.grid(True, alpha=0.3)

    savefig(
        out / "battery_priority.png"
    )


def plot_uav_voltage(
    uav_id: str,
    data: dict,
    out: Path,
    exp_name: str,
):
    if "voltage" not in data:
        return

    v = data["voltage"]
    t = data["voltage_time"]

    if len(v) == 0:
        return

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.plot(
        t,
        v,
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")

    ax.set_title(
        f"{exp_name} - UAV {uav_id} - Battery voltage"
    )

    ax.grid(True, alpha=0.3)

    savefig(
        out / "voltage.png"
    )


def make_individual_uav_plots(
    uav_id,
    data,
    out,
    exp_name,
):
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_uav_position(
        uav_id,
        data,
        out,
        exp_name,
    )

    plot_uav_path(
        uav_id,
        data,
        out,
        exp_name,
    )

    plot_uav_priorities(
        uav_id,
        data,
        out,
        exp_name,
    )

    plot_uav_voltage(
        uav_id,
        data,
        out,
        exp_name,
    )


# =============================================================================
# PLOTS - EXPERIMENT LEVEL
# =============================================================================

def pairwise_distances(exp_data):
    pairs = []
    series = []

    for a, b in combinations(
        sorted(exp_data.keys(), key=int),
        2,
    ):

        if (
            "position" not in exp_data[a]
            or "position" not in exp_data[b]
        ):
            continue

        ta = exp_data[a]["position_timestamp"]
        tb = exp_data[b]["position_timestamp"]

        pa = exp_data[a]["position"]
        pb = exp_data[b]["position"]

        # Interpolate UAV b onto UAV a's elapsed-time scale.
        if len(ta) == 0 or len(tb) == 0:
            continue

        ta_elapsed = elapsed_time(ta)
        tb_elapsed = elapsed_time(tb)

        common_start = max(
            ta_elapsed[0],
            tb_elapsed[0],
        )

        common_end = min(
            ta_elapsed[-1],
            tb_elapsed[-1],
        )

        if common_end < common_start:
            continue

        t = ta_elapsed[
            (ta_elapsed >= common_start)
            & (ta_elapsed <= common_end)
        ]

        if len(t) == 0:
            continue

        pb_interp = np.column_stack([
            np.interp(
                t,
                tb_elapsed,
                pb[:, j],
            )
            for j in range(3)
        ])

        pa_interp = np.column_stack([
            np.interp(
                t,
                ta_elapsed,
                pa[:, j],
            )
            for j in range(3)
        ])

        d = np.linalg.norm(
            pa_interp - pb_interp,
            axis=1,
        )

        pairs.append((a, b))
        series.append((t, d))

    return pairs, series


def plot_experiment_distances(
    exp_data,
    out,
    exp_name,
):
    pairs, series = pairwise_distances(
        exp_data
    )

    if not pairs:
        return

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    for pair, (t, d) in zip(
        pairs,
        series,
    ):

        ax.plot(
            t,
            d,
            label=f"UAV {pair[0]} - UAV {pair[1]}",
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance [m]")

    ax.set_title(
        f"{exp_name} - Inter-UAV distances"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    savefig(
        out / "uav_distances.png"
    )


def plot_experiment_path_and_area(
    exp_data,
    out,
    exp_name,
):
    positions = []
    labels = []

    for uav_id in sorted(
        exp_data.keys(),
        key=int,
    ):

        if "position" not in exp_data[uav_id]:
            continue

        xy = exp_data[uav_id]["position"][:, :2]

        valid = np.isfinite(xy).all(axis=1)
        xy = xy[valid]

        if len(xy):
            positions.append(xy)
            labels.append(uav_id)

    if not positions:
        return

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    plot_world_walls(ax)

    coverage = covered_area_geometry(
        positions
    )

    clipped = safe_intersection(
        coverage,
        get_arena_polygon(),
    )

    if clipped is not None:

        if clipped.geom_type == "Polygon":

            x, y = clipped.exterior.xy

            ax.fill(
                x,
                y,
                alpha=0.15,
                label=f"Covered area",
            )

        elif clipped.geom_type == "MultiPolygon":

            first = True

            for polygon in clipped.geoms:

                x, y = polygon.exterior.xy

                ax.fill(
                    x,
                    y,
                    alpha=0.15,
                    label=(
                        f"Covered area"
                        if first
                        else None
                    ),
                )

                first = False

    for xy, uav_id in zip(
        positions,
        labels,
    ):

        ax.plot(
            xy[:, 0],
            xy[:, 1],
            linewidth=1.2,
            label=f"UAV {uav_id}",
        )

        ax.scatter(
            xy[0, 0],
            xy[0, 1],
            s=35,
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.set_title(
        f"{exp_name} - UAV paths and covered area"
    )

    ax.set_xlim(WORLD_XMIN, WORLD_XMAX)
    ax.set_ylim(WORLD_YMIN, WORLD_YMAX)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()

    savefig(
        out / "paths_and_covered_area.png"
    )


def make_experiment_plots(
    exp_folder,
    exp_data,
):
    out = exp_folder / "plots"
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    exp_name = exp_folder.name

    plot_experiment_distances(
        exp_data,
        out,
        exp_name,
    )

    plot_experiment_path_and_area(
        exp_data,
        out,
        exp_name,
    )


# =============================================================================
# AGGREGATION UTILITIES
# =============================================================================

def interpolate_series(
    time,
    values,
    common_time,
):
    """
    Interpolate a time series onto common_time.

    Values outside the original time range are NaN.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(time) < 2:
        return np.full_like(
            common_time,
            np.nan,
            dtype=float,
        )

    result = np.full(
        len(common_time),
        np.nan,
        dtype=float,
    )

    mask = (
        (common_time >= time[0])
        & (common_time <= time[-1])
    )

    if np.any(mask):
        result[mask] = np.interp(
            common_time[mask],
            time,
            values,
        )

    return result


def make_common_time(
    series_list,
    n=AGGREGATION_POINTS,
):
    """
    Create a common elapsed-time axis from a list of
    (time, values) pairs.
    """
    valid = [
        (t, v)
        for t, v in series_list
        if len(t) > 0
    ]

    if not valid:
        return np.array([])

    max_time = max(
        float(t[-1])
        for t, _ in valid
    )

    if max_time <= 0:
        return np.array([0.0])

    return np.linspace(
        0.0,
        max_time,
        n,
    )


def aggregate_scalar_series(
    series_list,
):
    """
    series_list:
        [(time, values), ...]

    Returns:
        common_time, mean, variance
    """
    if not series_list:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )

    common_time = make_common_time(
        series_list
    )

    if len(common_time) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )

    stack = []

    for time, values in series_list:
        stack.append(
            interpolate_series(
                time,
                values,
                common_time,
            )
        )

    stack = np.asarray(stack)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        mean = np.nanmean(
            stack,
            axis=0,
        )

        variance = np.nanvar(
            stack,
            axis=0,
        )

    return (
        common_time,
        mean,
        variance,
    )


# =============================================================================
# AGGREGATED POSITION DATA
# =============================================================================

def aggregate_positions(
    all_exp_data,
):
    ids = sorted(
        {
            uav
            for exp in all_exp_data
            for uav in exp.keys()
        },
        key=int,
    )

    result = {}

    for uav_id in ids:

        experiments = []

        for exp in all_exp_data:

            if (
                uav_id not in exp
                or "position" not in exp[uav_id]
            ):
                continue

            data = exp[uav_id]

            experiments.append(
                (
                    data["position_time"],
                    data["position"],
                )
            )

        if not experiments:
            continue

        common_time = make_common_time(
            [
                (t, p[:, 0])
                for t, p in experiments
            ]
        )

        if len(common_time) == 0:
            continue

        stack = []

        for time, position in experiments:

            interpolated = np.column_stack([
                interpolate_series(
                    time,
                    position[:, i],
                    common_time,
                )
                for i in range(
                    position.shape[1]
                )
            ])

            stack.append(interpolated)

        stack = np.asarray(stack)

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=RuntimeWarning,
            )

            mean = np.nanmean(
                stack,
                axis=0,
            )

            variance = np.nanvar(
                stack,
                axis=0,
            )

        result[uav_id] = (
            common_time,
            mean,
            variance,
        )

    return result


def plot_aggregated_positions(
    position_stats,
    out,
    setting_name,
):
    for uav_id, (
        t,
        mean,
        variance,
    ) in position_stats.items():

        std = np.sqrt(variance)

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(10, 10),
            sharex=True,
        )

        labels = [
            "x [m]",
            "y [m]",
            "z [m]",
            "yaw [rad]",
        ]

        for i, ax in enumerate(axes):

            ax.plot(
                t,
                mean[:, i],
                label="Mean",
            )

            ax.fill_between(
                t,
                mean[:, i] - std[:, i],
                mean[:, i] + std[:, i],
                alpha=0.2,
                label="±1 std",
            )

            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time [s]")
        axes[0].legend()

        fig.suptitle(
            f"{setting_name} - UAV {uav_id} "
            f"- Mean position ± std"
        )

        savefig(
            out /
            f"mean_position_UAV_{uav_id}.png"
        )


# =============================================================================
# AGGREGATED PATHS + COVERED AREA
# =============================================================================

def build_path_uncertainty_area(
    mean_xy,
    std_xy,
):
    """
    Build a visual uncertainty area around a mean path.

    The local uncertainty radius is:
        sqrt(std_x^2 + std_y^2)

    The resulting buffers are united into one geometry.
    """
    circles = []

    for point, std in zip(
        mean_xy,
        std_xy,
    ):

        if (
            not np.isfinite(point).all()
            or not np.isfinite(std).all()
        ):
            continue

        radius = float(
            np.sqrt(
                std[0] ** 2
                + std[1] ** 2
            )
        )

        if radius <= 0:
            continue

        circles.append(
            Point(
                float(point[0]),
                float(point[1]),
            ).buffer(radius)
        )

    if not circles:
        return None

    return unary_union(circles)


def plot_aggregated_paths(
    position_stats,
    out,
    setting_name,
):
    """
    Plot mean paths.

    For each UAV, the path variance is represented as a filled
    uncertainty area around the mean trajectory.

    The same figure also contains the mean covered area over time,
    with ±1 standard deviation represented as a filled area.
    """
    if not position_stats:
        return

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    plot_world_walls(ax)

    # Matplotlib's standard color cycle gives each UAV a different color.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (
        uav_id,
        (
            t,
            mean,
            variance,
        ),
    ) in enumerate(
        sorted(
            position_stats.items(),
            key=lambda x: int(x[0]),
        )
    ):

        if mean.shape[1] < 2:
            continue

        color = colors[
            index % len(colors)
        ]

        mean_xy = mean[:, :2]
        std_xy = np.sqrt(
            variance[:, :2]
        )

        uncertainty = build_path_uncertainty_area(
            mean_xy,
            std_xy,
        )

        if uncertainty is not None:

            if uncertainty.geom_type == "Polygon":

                x, y = uncertainty.exterior.xy

                ax.fill(
                    x,
                    y,
                    color=color,
                    alpha=0.18,
                    label=f"UAV {uav_id} path ±1 std",
                )

            elif uncertainty.geom_type == "MultiPolygon":

                first = True

                for polygon in uncertainty.geoms:

                    x, y = polygon.exterior.xy

                    ax.fill(
                        x,
                        y,
                        color=color,
                        alpha=0.18,
                        label=(
                            f"UAV {uav_id} path ±1 std"
                            if first
                            else None
                        ),
                    )

                    first = False

        ax.plot(
            mean[:, 0],
            mean[:, 1],
            color=color,
            linewidth=2.0,
            #label=f"UAV {uav_id} mean path",
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.set_title(
        f"{setting_name} - Mean UAV paths "
        f"± std area"
    )

    ax.set_xlim(
        WORLD_XMIN,
        WORLD_XMAX,
    )

    ax.set_ylim(
        WORLD_YMIN,
        WORLD_YMAX,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    savefig(
        out / "mean_paths.png"
    )


def build_aggregated_coverage(
    all_exp_data,
):
    """
    Build cumulative covered-area percentage for every experiment.
    """
    series = []

    for exp in all_exp_data:

        time_position_pairs = []

        for uav_id, data in exp.items():

            if "position" not in data:
                continue

            time_position_pairs.append(
                (
                    data["position_time"],
                    data["position"][:, :2],
                )
            )

        t, area = cumulative_coverage_area(
            time_position_pairs
        )

        if len(t):
            series.append(
                (t, area)
            )

    return aggregate_scalar_series(
        series
    )


def plot_aggregated_coverage(
    coverage_stats,
    out,
    setting_name,
):
    t, mean, variance = coverage_stats

    if len(t) == 0:
        return

    std = np.sqrt(variance)

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.plot(
        t,
        mean,
        linewidth=2,
        label="Mean covered area",
    )

    ax.fill_between(
        t,
        np.maximum(
            0.0,
            mean - std,
        ),
        mean + std,
        alpha=0.2,
        label="±1 std",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Covered area [%]")

    ax.set_title(
        f"{setting_name} - Mean covered area ± std"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    savefig(
        out / "mean_covered_area.png"
    )


# =============================================================================
# AGGREGATED PRIORITIES
# =============================================================================

def aggregate_priorities(
    all_exp_data,
):
    ids = sorted(
        {
            uav
            for exp in all_exp_data
            for uav in exp.keys()
        },
        key=int,
    )

    result = {}

    for uav_id in ids:

        # Determine maximum number of priority columns.
        experiments = []

        for exp in all_exp_data:

            if (
                uav_id not in exp
                or "priority" not in exp[uav_id]
            ):
                continue

            data = exp[uav_id]

            experiments.append(
                (
                    data["priority_time"],
                    data["priority"],
                )
            )

        if not experiments:
            continue

        ncols = min(
            p.shape[1]
            for _, p in experiments
        )

        per_column = []

        for column in range(ncols):

            series = [
                (
                    t,
                    p[:, column],
                )
                for t, p in experiments
            ]

            per_column.append(
                aggregate_scalar_series(
                    series
                )
            )

        common_time = per_column[0][0]

        mean = np.column_stack([
            item[1]
            for item in per_column
        ])

        variance = np.column_stack([
            item[2]
            for item in per_column
        ])

        result[uav_id] = (
            common_time,
            mean,
            variance,
        )

    return result


def plot_aggregated_priorities(
    priority_stats,
    out,
    setting_name,
):
    for uav_id, (
        t,
        mean,
        variance,
    ) in priority_stats.items():

        std = np.sqrt(variance)
        n = mean.shape[1]

        labels = []

        for i in range(n):

            if i == n - 1:
                labels.append("Battery")
            else:
                labels.append(
                    f"Resource {i + 1}"
                )

        fig, ax = plt.subplots(
            figsize=FIGSIZE
        )

        for i, label in enumerate(labels):

            ax.plot(
                t,
                mean[:, i],
                label=label,
            )

            ax.fill_between(
                t,
                mean[:, i] - std[:, i],
                mean[:, i] + std[:, i],
                alpha=0.12,
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Priority")

        ax.set_title(
            f"{setting_name} - UAV {uav_id} "
            f"- Priorities - Mean ± std"
        )

        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)

        savefig(
            out /
            f"mean_priorities_UAV_{uav_id}.png"
        )

        # Battery priority separately
        fig, ax = plt.subplots(
            figsize=FIGSIZE
        )

        ax.plot(
            t,
            mean[:, -1],
            label="Mean",
        )

        ax.fill_between(
            t,
            mean[:, -1] - std[:, -1],
            mean[:, -1] + std[:, -1],
            alpha=0.2,
            label="±1 std",
        )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Battery priority")

        ax.set_title(
            f"{setting_name} - UAV {uav_id} "
            f"- Battery priority - Mean ± std"
        )

        ax.grid(True, alpha=0.3)
        ax.legend()

        savefig(
            out /
            f"mean_battery_priority_UAV_{uav_id}.png"
        )


# =============================================================================
# AGGREGATED VOLTAGE
# =============================================================================

def aggregate_voltage(
    all_exp_data,
):
    ids = sorted(
        {
            uav
            for exp in all_exp_data
            for uav in exp.keys()
        },
        key=int,
    )

    result = {}

    for uav_id in ids:

        series = []

        for exp in all_exp_data:

            if (
                uav_id in exp
                and "voltage" in exp[uav_id]
            ):
                data = exp[uav_id]

                series.append(
                    (
                        data["voltage_time"],
                        data["voltage"],
                    )
                )

        if series:
            result[uav_id] = (
                aggregate_scalar_series(
                    series
                )
            )

    return result


def plot_aggregated_voltage(
    voltage_stats,
    out,
    setting_name,
):
    for uav_id, (
        t,
        mean,
        variance,
    ) in voltage_stats.items():

        std = np.sqrt(variance)

        fig, ax = plt.subplots(
            figsize=FIGSIZE
        )

        ax.plot(
            t,
            mean,
            label="Mean",
        )

        ax.fill_between(
            t,
            mean - std,
            mean + std,
            alpha=0.2,
            label="±1 std",
        )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Voltage [V]")

        ax.set_title(
            f"{setting_name} - UAV {uav_id} "
            f"- Voltage - Mean ± std"
        )

        ax.grid(True, alpha=0.3)
        ax.legend()

        savefig(
            out /
            f"mean_voltage_UAV_{uav_id}.png"
        )


# =============================================================================
# AGGREGATED INTER-UAV DISTANCES
# =============================================================================

def build_distance_series(exp_data):
    pairs, series = pairwise_distances(
        exp_data
    )

    return {
        pair: values
        for pair, values in zip(
            pairs,
            series,
        )
    }


def aggregate_distances(
    all_exp_data,
):
    all_pairs = sorted(
        {
            pair
            for exp in all_exp_data
            for pair in combinations(
                sorted(
                    exp.keys(),
                    key=int,
                ),
                2,
            )
        },
        key=lambda p: (
            int(p[0]),
            int(p[1]),
        ),
    )

    result = {}

    for pair in all_pairs:

        series = []

        for exp in all_exp_data:

            distance_series = build_distance_series(
                exp
            )

            if pair in distance_series:
                series.append(
                    distance_series[pair]
                )

        if series:
            result[pair] = (
                aggregate_scalar_series(
                    series
                )
            )

    return result


def plot_aggregated_distances(
    distance_stats,
    out,
    setting_name,
):
    if not distance_stats:
        return

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    for pair, (
        t,
        mean,
        variance,
    ) in distance_stats.items():

        std = np.sqrt(variance)

        ax.plot(
            t,
            mean,
            label=(
                f"UAV {pair[0]} - UAV {pair[1]}"
            ),
        )

        ax.fill_between(
            t,
            mean - std,
            mean + std,
            alpha=0.12,
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance [m]")

    ax.set_title(
        f"{setting_name} - Inter-UAV distance "
        f"- Mean ± std"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    savefig(
        out / "mean_uav_distances.png"
    )


# =============================================================================
# REPORT
# =============================================================================

def hidden_waypoint_percentage(
    exp_data,
):
    """
    Hidden waypoint is detected when the penultimate priority column
    is non-zero at the end of the UAV experiment.

    Returns:
        percentage, detected_count, total_count
    """
    total = 0
    detected = 0

    for uav_id, data in exp_data.items():

        if (
            "priority" not in data
            or len(data["priority"]) == 0
        ):
            continue

        priority = data["priority"]

        if priority.shape[1] < 2:
            continue

        final_hidden_priority = priority[-1, -2]

        total += 1

        if (
            np.isfinite(final_hidden_priority)
            and final_hidden_priority != 0
        ):
            detected += 1

    if total == 0:
        return np.nan, 0, 0

    return (
        detected / total * 100.0,
        detected,
        total,
    )


def experiment_report_metrics(
    exp_folder,
    exp_data,
    crash_info,
):
    # --------------------------------------------------------------
    # Finish times
    # --------------------------------------------------------------

    finish_times = {}

    for uav_id, data in exp_data.items():

        finish_time = data.get(
            "finish_time"
        )

        if finish_time is not None:
            finish_times[uav_id] = float(
                finish_time
            )

    # --------------------------------------------------------------
    # Objective
    # --------------------------------------------------------------

    objective_time = compute_objective_time(
        exp_data
    )

    # --------------------------------------------------------------
    # Coverage
    # --------------------------------------------------------------

    positions = []

    for data in exp_data.values():

        if "position" not in data:
            continue

        xy = data["position"][:, :2]

        valid = np.isfinite(xy).all(axis=1)
        xy = xy[valid]

        if len(xy):
            positions.append(xy)

    coverage_percentage, _ = (
        coverage_area_percentage(
            positions
        )
    )

    # --------------------------------------------------------------
    # Hidden waypoint
    # --------------------------------------------------------------

    hidden_percentage, hidden_detected, hidden_total = (
        hidden_waypoint_percentage(
            exp_data
        )
    )

    # --------------------------------------------------------------
    # Crash
    # --------------------------------------------------------------

    crash_uavs = [
        uav_id
        for uav_id, crash in crash_info.items()
        if crash is not None
    ]

    experiment_crashed = len(
        crash_uavs
    ) > 0

    return {
        "finish_times": finish_times,
        "objective_time": objective_time,
        "coverage_percentage": coverage_percentage,
        "hidden_percentage": hidden_percentage,
        "hidden_detected": hidden_detected,
        "hidden_total": hidden_total,
        "crash_uavs": crash_uavs,
        "experiment_crashed": experiment_crashed,
        "uav_count": len(exp_data),
    }


def generate_setting_report(
    setting_folder,
    experiment_reports,
):
    """
    Generate report.txt containing the same main metrics as the
    previous simulation analysis.
    """
    out = (
        setting_folder /
        "report.txt"
    )

    n_experiments = len(
        experiment_reports
    )

    all_uavs = sorted(
        {
            uav
            for report in experiment_reports
            for uav in report["finish_times"]
        },
        key=int,
    )

    # --------------------------------------------------------------
    # Finish-time means
    # --------------------------------------------------------------

    mean_finish_per_uav = {}
    variance_finish_per_uav = {}
    std_finish_per_uav = {}

    for uav_id in all_uavs:

        values = [
            report["finish_times"][uav_id]
            for report in experiment_reports
            if uav_id in report["finish_times"]
        ]

        if values:
            mean_finish_per_uav[uav_id] = float(
                np.mean(values)
            )
            

            variance_finish_per_uav[uav_id] = float(
                np.var(values)
            )
            std_finish_per_uav[uav_id] = float(
                np.std(values)
            )

    all_finish_times = [
        value
        for report in experiment_reports
        for value in report["finish_times"].values()
    ]

    mean_finish_all = (
        float(np.mean(all_finish_times))
        if all_finish_times
        else np.nan
    )

    variance_finish_all = (
        float(np.var(all_finish_times))
        if all_finish_times
        else np.nan
    )

    std_finish_all = (
        float(np.std(all_finish_times))
        if all_finish_times
        else np.nan
    )

    # --------------------------------------------------------------
    # Objective time
    # --------------------------------------------------------------

    objective_values = [
        report["objective_time"]
        for report in experiment_reports
        if report["objective_time"] is not None
    ]

    mean_objective = (
        float(np.mean(objective_values))
        if objective_values
        else np.nan
    )
    variance_objective = (
        float(np.var(objective_values))
        if objective_values
        else np.nan
    )

    std_objective = (
        float(np.std(objective_values))
        if objective_values
        else np.nan
    )

    objective_success_percentage = (
        len(objective_values)
        / n_experiments
        * 100.0
        if n_experiments
        else np.nan
    )

    # --------------------------------------------------------------
    # Coverage
    # --------------------------------------------------------------

    coverage_values = [
        report["coverage_percentage"]
        for report in experiment_reports
        if np.isfinite(
            report["coverage_percentage"]
        )
    ]

    mean_coverage = (
        float(np.mean(coverage_values))
        if coverage_values
        else np.nan
    )
    variance_coverage = (
        float(np.var(coverage_values))
        if coverage_values
        else np.nan
    )

    std_coverage = (
        float(np.std(coverage_values))
        if coverage_values
        else np.nan
    )
    # --------------------------------------------------------------
    # Hidden waypoint
    # --------------------------------------------------------------

    hidden_values = [
        report["hidden_percentage"]
        for report in experiment_reports
        if np.isfinite(
            report["hidden_percentage"]
        )
    ]

    mean_hidden = (
        float(np.mean(hidden_values))
        if hidden_values
        else np.nan
    )
    variance_hidden = (
        float(np.var(hidden_values))
        if hidden_values
        else np.nan
    )

    std_hidden = (
        float(np.std(hidden_values))
        if hidden_values
        else np.nan
    )
    # --------------------------------------------------------------
    # Crash
    # --------------------------------------------------------------
    experiment_crash_values = [
    100.0 if report["experiment_crashed"] else 0.0
    for report in experiment_reports
    ]

    variance_experiment_crash = (
        float(np.var(experiment_crash_values))
        if experiment_crash_values
        else np.nan
    )

    std_experiment_crash = (
        float(np.std(experiment_crash_values))
        if experiment_crash_values
        else np.nan
    )
    uav_crash_values = [
        (
            len(report["crash_uavs"])
            / report["uav_count"]
            * 100.0
        )
        if report["uav_count"] > 0
        else np.nan
        for report in experiment_reports
    ]

    uav_crash_values = [
        value
        for value in uav_crash_values
        if np.isfinite(value)
    ]

    variance_uav_crash = (
        float(np.var(uav_crash_values))
        if uav_crash_values
        else np.nan
    )

    std_uav_crash = (
        float(np.std(uav_crash_values))
        if uav_crash_values
        else np.nan
    )
    experiment_crashes = sum(
        report["experiment_crashed"]
        for report in experiment_reports
    )

    experiment_crash_percentage = (
        experiment_crashes
        / n_experiments
        * 100.0
        if n_experiments
        else np.nan
    )

    total_uavs = sum(
        report["uav_count"]
        for report in experiment_reports
    )

    crashed_uavs = sum(
        len(report["crash_uavs"])
        for report in experiment_reports
    )

    uav_crash_percentage = (
        crashed_uavs
        / total_uavs
        * 100.0
        if total_uavs
        else np.nan
    )

    # --------------------------------------------------------------
    # Write report
    # --------------------------------------------------------------

    with out.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"SETTING: {setting_folder.name}\n"
        )
        f.write(
            f"Number of experiments: "
            f"{n_experiments}\n"
        )
        f.write("\n")

        f.write(
            "========================================\n"
        )
        f.write(
            "MISSION FINISH TIMES\n"
        )
        f.write(
            "========================================\n"
        )

        f.write(
            f"Mean finish time across all UAVs: "
            f"{mean_finish_all:.3f} s\n"
        )
        f.write(
            f"Variance finish time across all UAVs: "
            f"{variance_finish_all:.3f} s²\n"
        )
        f.write(
            f"Standard deviation finish time across all UAVs: "
            f"{std_finish_all:.3f} s\n"
        )

        for uav_id in all_uavs:
            value = mean_finish_per_uav.get(
                uav_id,
                np.nan,
            )

            variance = variance_finish_per_uav.get(
                uav_id,
                np.nan,
            )

            f.write(
                f"Mean finish time UAV {uav_id}: "
                f"{value:.3f} s\n"
            )

            f.write(
                f"Variance finish time UAV {uav_id}: "
                f"{variance:.3f} s²\n"
            )

            std = std_finish_per_uav.get(
                uav_id,
                np.nan,
            )

            f.write(
                f"Standard deviation finish time UAV {uav_id}: "
                f"{std:.3f} s\n"
            )

        f.write(
            "========================================\n"
        )
        f.write(
            "OBJECTIVE\n"
        )
        f.write(
            "========================================\n"
        )

        f.write(
            "Criterion: for every resource column "
            "Priority_0...Priority_5, at least one "
            "UAV must have priority < 1. "
            "Priority_6 (battery) is excluded.\n"
        )

        f.write(
            f"Mean objective time "
            f"(successful experiments only): "
            f"{mean_objective:.3f} s\n"
        )

        f.write(
            f"Variance objective time "
            f"(successful experiments only): "
            f"{variance_objective:.3f} s²\n"
        )
        f.write(
            f"Standard deviation objective time "
            f"(successful experiments only): "
            f"{std_objective:.3f} s\n"
        )

        f.write(
            f"Experiments reaching objective: "
            f"{len(objective_values)}/{n_experiments} "
            f"({objective_success_percentage:.2f}%)\n"
        )

        f.write("\n")

        f.write(
            "========================================\n"
        )
        f.write(
            "COVERAGE\n"
        )
        f.write(
            "========================================\n"
        )

        f.write(
            f"Mean covered area: "
            f"{mean_coverage:.3f}%\n"
        )
        f.write(
            f"Variance covered area: "
            f"{variance_coverage:.3f} %²\n"
        )
        f.write(
            f"Standard deviation covered area: "
            f"{std_coverage:.3f} %\n"
        )

        f.write("\n")

        f.write(
            "========================================\n"
        )
        f.write(
            "HIDDEN WAYPOINT\n"
        )
        f.write(
            "========================================\n"
        )

        f.write(
            "Criterion: penultimate priority column "
            "is non-zero at the end of the UAV experiment.\n"
        )

        f.write(
            f"Mean percentage of UAVs finding hidden "
            f"waypoint: {mean_hidden:.3f}%\n"
        )
        f.write(
            f"Variance percentage of UAVs finding hidden "
            f"waypoint: {variance_hidden:.3f} %²\n"
        )
        f.write(
            f"Standard deviation percentage of UAVs finding hidden "
            f"waypoint: {std_hidden:.3f} %\n"
        )

        f.write("\n")

        f.write(
            "========================================\n"
        )
        f.write(
            "CRASHES\n"
        )
        f.write(
            "========================================\n"
        )

        f.write(
            f"Experiments with at least one crash: "
            f"{experiment_crashes}/{n_experiments} "
            f"({experiment_crash_percentage:.2f}%)\n"
        )

        f.write(
            f"Variance experiment crash percentage: "
            f"{variance_experiment_crash:.3f} %²\n"
        )
        f.write(
            f"Standard deviation experiment crash percentage: "
            f"{std_experiment_crash:.3f} %\n"
        )

        f.write(
            f"UAV crash percentage: "
            f"{uav_crash_percentage:.2f}%\n"
        )

        f.write(
            f"Variance UAV crash percentage: "
            f"{variance_uav_crash:.3f} %²\n"
        )
        f.write(
            f"Standard deviation UAV crash percentage: "
            f"{std_uav_crash:.3f} %\n"
        )
        f.write(
            f"UAV crash percentage: "
            f"{uav_crash_percentage:.2f}%\n"
        )

        f.write("\n")

        f.write(
            "========================================\n"
        )
        f.write(
            "PER-EXPERIMENT RESULTS\n"
        )
        f.write(
            "========================================\n"
        )

        for i, report in enumerate(
            experiment_reports,
            start=1,
        ):

            f.write(
                f"\nExperiment {i}\n"
            )

            f.write(
                f"  Coverage: "
                f"{report['coverage_percentage']:.3f}%\n"
            )

            f.write(
                f"  Hidden waypoint: "
                f"{report['hidden_percentage']:.3f}%\n"
            )

            if report["objective_time"] is None:
                f.write(
                    "  Objective time: NOT REACHED\n"
                )
            else:
                f.write(
                    f"  Objective time: "
                    f"{report['objective_time']:.3f} s\n"
                )

            f.write(
                "  Crashed UAVs: "
                + (
                    ", ".join(
                        report["crash_uavs"]
                    )
                    if report["crash_uavs"]
                    else "None"
                )
                + "\n"
            )

    print(
        f"  Report saved to: {out}"
    )


# =============================================================================
# EXPERIMENT PROCESSING
# =============================================================================

def process_experiment(
    exp_folder: Path,
):
    print(
        f"  Processing {exp_folder.name}..."
    )

    exp_data = build_exp_data(
        exp_folder
    )

    if not exp_data:
        print(
            "    WARNING: no UAV data found."
        )
        return (
            exp_data,
            {},
        )

    # Detect crashes BEFORE plotting/reporting.
    crash_info = apply_crash_detection(
        exp_data
    )

    save_crash_info(
        exp_folder,
        crash_info,
    )

    # Generate timetofinish_cf_X.txt and determine
    # the individual finish time of every UAV.
    finish_times = save_finish_times(
        exp_folder,
        exp_data,
    )

    # --------------------------------------------------------------
    # Maximum finish time of this experiment
    # --------------------------------------------------------------

    valid_finish_times = [
        finish_time
        for finish_time in finish_times.values()
        if finish_time is not None
    ]

    if valid_finish_times:

        max_finish_time = max(
            valid_finish_times
        )

        # All plots/data from this experiment are limited
        # to the finish time of the slowest UAV.
        trim_data_to_max_finish(
            exp_data,
            max_finish_time,
        )

    else:

        max_finish_time = None

        print(
            "    WARNING: no UAV reached the finish condition."
        )

    # Generate timetoobjective.txt.
    objective_time = compute_objective_time(
        exp_data
    )

    save_objective_time(
        exp_folder,
        objective_time,
    )

    # Individual UAV plots.
    for uav_id, data in exp_data.items():

        make_individual_uav_plots(
            uav_id,
            data,
            exp_folder /
            f"plots_cf_{uav_id}",
            exp_folder.name,
        )

    # Experiment-level plots.
    make_experiment_plots(
        exp_folder,
        exp_data,
    )

    return (
        exp_data,
        crash_info,
    )


# =============================================================================
# SETTING PROCESSING
# =============================================================================

def get_experiment_folders(
    setting_folder: Path,
):
    folders = [
        p
        for p in setting_folder.iterdir()
        if (
            p.is_dir()
            and re.match(
                r"^exp\s*\d+$",
                p.name,
                re.IGNORECASE,
            )
        )
    ]

    def exp_number(path):
        match = re.search(
            r"(\d+)",
            path.name,
        )

        return (
            int(match.group(1))
            if match
            else 999999
        )

    return sorted(
        folders,
        key=exp_number,
    )


def process_setting(
    setting_folder: Path,
):
    print(
        f"\nProcessing setting: "
        f"{setting_folder.name}"
    )

    exp_folders = get_experiment_folders(
        setting_folder
    )

    if not exp_folders:
        print(
            "  WARNING: no exp folders found."
        )
        return

    all_exp_data = []
    experiment_reports = []

    for exp_folder in exp_folders:

        exp_data, crash_info = process_experiment(
            exp_folder
        )

        if not exp_data:
            continue

        all_exp_data.append(
            exp_data
        )

        experiment_reports.append(
            experiment_report_metrics(
                exp_folder,
                exp_data,
                crash_info,
            )
        )

    if not all_exp_data:
        return

    # ------------------------------------------------------------------
    # Aggregated plots
    # ------------------------------------------------------------------

    out = (
        setting_folder /
        "plots_mean_experiments"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Position mean ± std
    position_stats = aggregate_positions(
        all_exp_data
    )

    plot_aggregated_positions(
        position_stats,
        out,
        setting_folder.name,
    )

    # Mean paths with uncertainty area
    plot_aggregated_paths(
        position_stats,
        out,
        setting_folder.name,
    )

    # Mean covered area ± std
    coverage_stats = build_aggregated_coverage(
        all_exp_data
    )

    plot_aggregated_coverage(
        coverage_stats,
        out,
        setting_folder.name,
    )

    # Priorities
    priority_stats = aggregate_priorities(
        all_exp_data
    )

    plot_aggregated_priorities(
        priority_stats,
        out,
        setting_folder.name,
    )

    # Voltage
    voltage_stats = aggregate_voltage(
        all_exp_data
    )

    plot_aggregated_voltage(
        voltage_stats,
        out,
        setting_folder.name,
    )

    # Distances
    distance_stats = aggregate_distances(
        all_exp_data
    )

    plot_aggregated_distances(
        distance_stats,
        out,
        setting_folder.name,
    )

    # Report
    generate_setting_report(
        setting_folder,
        experiment_reports,
    )

    print(
        f"  Aggregated plots saved to: {out}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate plots and reports for "
            "real-hardware UAV experiments."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help=(
            "Root directory containing "
            "'2 UAVs' and/or '3 UAVs'."
        ),
    )

    parser.add_argument(
        "--settings",
        nargs="*",
        default=None,
        help=(
            "Optional setting folder names. "
            "Default: discover the expected settings."
        ),
    )

    args = parser.parse_args()

    root = (
        args.root
        .expanduser()
        .resolve()
    )

    if not root.exists():
        raise SystemExit(
            f"Root directory does not exist: "
            f"{root}"
        )

    if args.settings:
        settings = [
            root / name
            for name in args.settings
        ]

        settings = [
            path
            for path in settings
            if path.is_dir()
        ]

    else:
        expected = [
            root / name
            for name in DEFAULT_SETTINGS
        ]

        settings = [
            path
            for path in expected
            if path.is_dir()
        ]

        if not settings:
            settings = sorted(
                [
                    path
                    for path in root.iterdir()
                    if path.is_dir()
                ]
            )

    if not settings:
        raise SystemExit(
            "No setting folders found."
        )

    print(
        f"Root: {root}"
    )

    print(
        "Settings: "
        + str(
            [
                path.name
                for path in settings
            ]
        )
    )

    for setting in settings:
        process_setting(setting)

    print("\nDone.")


if __name__ == "__main__":
    main()