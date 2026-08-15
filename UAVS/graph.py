#!/usr/bin/env python3
"""
Plot and aggregate UAV experiments.

Expected directory structure:

ROOT/
├── 1 UAV/
│   ├── exp 1/
│   │   ├── 28/
│   │   │   ├── position.csv
│   │   │   ├── priority.csv
│   │   │   ├── Voltage.csv
│   │   │   └── photosynthesis.csv
│   │   ├── 29/
│   │   └── ...
│   ├── exp 2/
│   └── ...
├── 3 UAV/
├── 5 UAV/
├── no leader/
└── Normal APF/

The folders 28, 29, ... are interpreted as UAV IDs.

IMPORTANT:
- No timestamp is required. The CSV row number is used as the iteration/tick.
- For averaging, experiments are aligned by iteration index. If experiments have
  different lengths, missing samples are ignored at each iteration.
- "Covered area" is computed as the area of the convex hull of all positions
  visited up to that iteration. This is a geometric trajectory-envelope metric,
  NOT a true sensor/occupancy coverage metric. If you have a sensor radius or
  occupancy/coverage file later, this can be replaced.
- The first 5 columns of priority.csv are treated as visible resources,
  column 6 as the hidden resource, and column 7 as battery priority.
- The first column of Voltage.csv is used.
- The first numeric column of photosynthesis.csv is used.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.path import Path as MplPath
import matplotlib.patches as patches

from shapely.geometry import Point
from shapely.ops import unary_union

# --------------------------------------------------
# WALL GEOMETRY (EDIT THIS EASILY)
# --------------------------------------------------
# --------------------------------------------------
# Waypoints
# --------------------------------------------------

# Outer boundary of the world
WORLD_OUTER = {
    "xmin": -10.0,
    "xmax": 9.0,
    "ymin": -6.0,
    "ymax": 5.5
}

# --------------------------------------------------
# Waypoints
# --------------------------------------------------
WAYPOINTS = [
    (6.55, 0.95),
    (-0.56, 4.11),
    (0.78, -0.41),
    (-6.17, 2.00),
    (-8.27, -4.50),
    
]
HIDDEN_WAYPOINT = [(3.48,3.17),]

CHARGIN_STATION = [(6, -3),]
OBSTACLES = [

    # ============================================================
    # WALLS
    # ============================================================
    # ============================================================
    # INTERNAL WALL / STRUCTURE
    # ============================================================

    # Folding door / wall around bedroom
    (-2.46, -6.00),
    (-2.46, -5.50),
    (-2.46, -5.00),
    (-2.46, -4.50),
    (-2.46, -4.00),
    (-2.46, -3.50),
    (-2.46, -3.00),
    (-2.46, -2.50),
    (-2.46, -2.00),
    (-2.46,  0.00),
    (-2.46,  0.50),
    (-2.46,  1.00),
    (-2.46,  1.50),
    (-2.46,  2.00),
    (-2.46,  2.50),
    (-2.46,  3.00),


    # ============================================================
    # BED
    # ============================================================

    (-7.05, 1.45),
    (-7.05, 2.05),
    (-7.05, 2.60),
    (-6.45, 1.45),
    (-6.45, 2.05),
    (-6.45, 2.60),
    (-5.85, 1.45),
    (-5.85, 2.05),
    (-5.85, 2.60),
    (-5.25, 1.45),
    (-5.25, 2.05),
    (-5.25, 2.60),

    # Nightstands
    (-7.73, 2.86),
    (-4.41, 2.86),

    # Wardrobe
    (-3.65, 2.10),
    (-3.65, 2.50),
    (-3.65, 2.90),
    (-3.15, 2.10),
    (-3.15, 2.50),
    (-3.15, 2.90),
    (-2.65, 2.10),
    (-2.65, 2.50),
    (-2.65, 2.90),

    # Desk and chair
    (-8.99, 2.06),
    (-8.27, 1.92),

    # Trash
    (-8.70, 1.00),

    # ============================================================
    # LIVING ROOM
    # ============================================================

    # Sofa
    (-0.45, -1.05),
    (-0.45, -0.50),
    (-0.45,  0.05),
    ( 0.15, -1.05),
    ( 0.15, -0.50),
    ( 0.15,  0.05),
    ( 0.75, -1.05),
    ( 0.75, -0.50),
    ( 0.75,  0.05),
    ( 1.35, -1.05),
    ( 1.35, -0.50),
    ( 1.35,  0.05),

    # Coffee table
    (1.51, -1.73),

    # TV cabinet
    (-0.15, -5.18),
    ( 0.40, -5.18),
    ( 0.95, -5.18),
    ( 1.50, -5.18),

    # TV
    (0.82, -5.38),

    # Trash
    (2.36, -0.80),

    # Balls
    (3.30, 4.23),
    (-6.95, -4.22),

    # ============================================================
    # DINING AREA
    # ============================================================

    # Kitchen table
    (5.95, 0.35),
    (5.95, 0.95),
    (5.95, 1.55),
    (6.55, 0.35),
    (6.55, 0.95),
    (6.55, 1.55),
    (7.15, 0.35),
    (7.15, 0.95),
    (7.15, 1.55),

    # Chairs
    (7.12, 0.21),
    (6.26, 0.22),
    (6.07, 1.68),
    (7.00, 1.67),

    # ============================================================
    # BALCONY
    # ============================================================

    # Balcony table
    (-0.56, 4.11),

    # Balcony chairs
    (-1.38, 4.10),
    (0.33, 4.10),
    (-8.27, -4.50),

    # ============================================================
    # KITCHEN
    # ============================================================

    # Cooking bench
    (8.60, -3.80),
    (8.60, -3.35),
    (8.60, -2.90),
    (9.00, -3.80),
    (9.00, -3.35),
    (9.00, -2.90),

    # Kitchen cabinet
    (8.00, -3.84),

    # Refrigerator
    (8.70, -1.55),
    (8.70, -1.05),
    (8.70, -0.55),

    # Fitness equipment
    (3.48, 3.17),

    # Dumbbell
    (2.51, 2.72),

    # ============================================================
    # DOORS
    # ============================================================

    (6.00, -5.55),
    (-2.46, 1.84),
    (-2.46, -4.05),
    (4.67, 2.46),
    (4.89, -4.85),

    # ============================================================
    # SMALL OBJECTS
    # ============================================================

    (-2.00, -5.23),
]
# Cutouts (holes) inside the wall
# Each is a rectangle: (center x,y + width + height)
WALL_CUTOUTS = [
    # top-left cutout
    {"x": -7.0, "y": 4.2, "sx": 8.0, "sy": 2.5},

    # top-right cutout
    {"x": 7.5, "y": 4.2, "sx": 7.0, "sy": 2.5},
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum allowed displacement between two consecutive samples.
# If the UAV moves more than this distance, the trajectory is considered
# to have crashed and all data from that iteration onward are discarded.
CRASH_DISTANCE_THRESHOLD = 5.0 # meters

TIME_PER_ITERATION = 1# seconds

plt.rcParams.update({
    # =========================
    # FONT
    # =========================
    'font.family': 'serif',
    'font.size': 8,

    # =========================
    # AXES
    # =========================
    'axes.titlesize': 9,
    'axes.labelsize': 8,

    # =========================
    # TICKS
    # =========================
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,

    # =========================
    # LEGEND
    # =========================
    'legend.fontsize': 7,

    # =========================
    # LINES
    # =========================
    'lines.linewidth': 1.2,
    'lines.markersize': 0.1,
    # =========================
    # AXES STYLE
    # =========================
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,

    # =========================
    # FIGURE
    # =========================
    'figure.dpi': 150,
    'savefig.dpi': 300,

    # =========================
    # OUTPUT
    # =========================
    'savefig.bbox': 'tight',
})
SCATTER_MARKER_SIZE = 10
# ---------------------------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------------------------

def parse_mission_duration(txt_path: Path) -> float | None:
    """
    Read:
        Mission duration: 127.901seconds

    Returns the duration in seconds.
    """

    if not txt_path.is_file():
        return None

    try:
        text = txt_path.read_text().strip()

        match = re.search(
            r"Mission\s+duration\s*:\s*([-+]?\d*\.?\d+)",
            text,
            re.IGNORECASE
        )

        if match:
            return float(match.group(1))

    except Exception as exc:
        print(f"WARNING: cannot read {txt_path}: {exc}")

    return None


def parse_objective_time(txt_path: Path) -> float | None:
    """
    Read:
        Obejctive time: 76.857 seconds

    Also accepts the correctly spelled:
        Objective time: 76.857 seconds
    """

    if not txt_path.is_file():
        return None

    try:
        text = txt_path.read_text().strip()

        match = re.search(
            r"(?:Obejctive|Objective)\s+time\s*:\s*([-+]?\d*\.?\d+)",
            text,
            re.IGNORECASE
        )

        if match:
            return float(match.group(1))

    except Exception as exc:
        print(f"WARNING: cannot read {txt_path}: {exc}")

    return None


def valid_world_polygon():
    """
    Build the valid area of the world.

    Valid area =
        outer world rectangle
        minus wall cutouts.

    This construction is robust even when a cutout is very close
    to or intersects the external boundary.
    """

    from shapely.geometry import Polygon

    xmin = WORLD_OUTER["xmin"]
    xmax = WORLD_OUTER["xmax"]
    ymin = WORLD_OUTER["ymin"]
    ymax = WORLD_OUTER["ymax"]

    # --------------------------------------------------
    # Outer world
    # --------------------------------------------------

    world = Polygon([
        (xmin, ymin),
        (xmin, ymax),
        (xmax, ymax),
        (xmax, ymin)
    ])

    # --------------------------------------------------
    # Remove cutouts
    # --------------------------------------------------

    valid_area = world

    for cut in WALL_CUTOUTS:

        x0 = cut["x"] - cut["sx"] / 2.0
        x1 = cut["x"] + cut["sx"] / 2.0

        y0 = cut["y"] - cut["sy"] / 2.0
        y1 = cut["y"] + cut["sy"] / 2.0

        cutout = Polygon([
            (x0, y0),
            (x1, y0),
            (x1, y1),
            (x0, y1)
        ])

        valid_area = valid_area.difference(cutout)

    # --------------------------------------------------
    # Fix possible numerical topology problems
    # --------------------------------------------------

    if not valid_area.is_valid:
        valid_area = valid_area.buffer(0)

    return valid_area


def compute_experiment_covered_area_percentage(exp_data):
    """
    Compute the final covered area percentage for one experiment.

    Coverage is the union of circles with radius COVERAGE_RADIUS
    centered at all visited UAV positions.

    The coverage is clipped to the valid world area:

        outer rectangle - cutouts

    Overlapping coverage between UAVs is counted only once.
    """

    valid_area = valid_world_polygon()

    if valid_area.is_empty or valid_area.area <= 0:
        return np.nan

    circles = []

    # --------------------------------------------------
    # Build coverage circles
    # --------------------------------------------------

    for uav_id, data in exp_data.items():

        if "position" not in data:
            continue

        xy = data["position"][:, :2]

        for point in xy:

            if not np.isfinite(point).all():
                continue

            circle = Point(
                float(point[0]),
                float(point[1])
            ).buffer(
                COVERAGE_RADIUS
            )

            circles.append(circle)

    if not circles:
        return 0.0

    # --------------------------------------------------
    # Union of all UAV coverage
    # --------------------------------------------------

    coverage = unary_union(circles)

    # --------------------------------------------------
    # Fix possible topology problems
    # --------------------------------------------------

    if not coverage.is_valid:
        coverage = coverage.buffer(0)

    if not valid_area.is_valid:
        valid_area = valid_area.buffer(0)

    # --------------------------------------------------
    # Clip coverage to valid environment
    # --------------------------------------------------

    try:

        coverage = coverage.intersection(
            valid_area
        )

    except Exception:

        # Additional topology repair if necessary
        coverage = coverage.buffer(0)
        valid_area = valid_area.buffer(0)

        coverage = coverage.intersection(
            valid_area
        )

    # --------------------------------------------------
    # Percentage
    # --------------------------------------------------

    covered_area = coverage.area
    total_valid_area = valid_area.area

    if total_valid_area <= 0:
        return np.nan

    percentage = (
        covered_area /
        total_valid_area *
        100.0
    )

    return percentage

def get_time_axis(n_samples):
    return np.arange(n_samples) * TIME_PER_ITERATION

def detect_crash(data: dict, threshold: float) -> tuple[bool, int | None, float | None]:
    """
    Detect the first crash using exactly the same logic as
    truncate_after_crash().

    Returns:
        crash_detected
        crash_index
        displacement
    """

    if "position" not in data:
        return False, None, None

    position = data["position"]

    if len(position) < 2:
        return False, None, None

    xy = position[:, :2]

    valid = np.isfinite(xy).all(axis=1)

    for i in range(1, len(position)):

        if not valid[i] or not valid[i - 1]:
            continue

        distance = np.linalg.norm(
            xy[i] - xy[i - 1]
        )

        if distance > threshold:
            return True, i, distance

    return False, None, None


def compute_experiment_report(exp_folder: Path) -> dict:
    """
    Compute all report metrics for one experiment.
    """

    exp_data = build_exp_data(exp_folder)

    # --------------------------------------------------
    # Mission duration
    # --------------------------------------------------

    mission_times = []

    for uav_id in discover_uavs(exp_folder):

        txt_path = exp_folder / uav_id / "finishing_time.txt"

        value = parse_mission_duration(txt_path)

        if value is not None:
            mission_times.append(value)

    if mission_times:
        mean_mission_time = float(np.mean(mission_times))
    else:
        mean_mission_time = np.nan

    # --------------------------------------------------
    # Objective time
    # --------------------------------------------------

    objective_path = (
        exp_folder /
        "leader" /
        "timetoobjective.txt"
    )

    objective_time = parse_objective_time(objective_path)

    if objective_time is None:
        objective_time = np.nan

    # --------------------------------------------------
    # Covered area
    # --------------------------------------------------

    covered_area_percentage = (
        compute_experiment_covered_area_percentage(
            exp_data
        )
    )

    # --------------------------------------------------
    # Hidden waypoint
    # --------------------------------------------------

    uav_ids = discover_uavs(exp_folder)

    hidden_found = 0
    hidden_total = 0

    for uav_id in uav_ids:

        if uav_id not in exp_data:
            continue

        data = exp_data[uav_id]

        if "priority" not in data:
            continue

        priority = data["priority"]

        if len(priority) == 0:
            continue

        # Penultimate column = hidden resource.
        if priority.shape[1] < 2:
            continue

        hidden_priority = priority[-1, -2]

        hidden_total += 1

        if np.isfinite(hidden_priority) and hidden_priority != 0:
            hidden_found += 1

    if hidden_total > 0:
        hidden_percentage = (
            hidden_found /
            hidden_total *
            100.0
        )
    else:
        hidden_percentage = np.nan

    # --------------------------------------------------
    # Crashes
    # --------------------------------------------------

    crashed_uavs = 0
    crash_total = 0

    crash_details = []

    for uav_id in uav_ids:

        if uav_id not in exp_data:
            continue

        data = exp_data[uav_id]

        if "position" not in data:
            continue

        crash_total += 1

        crashed, crash_index, displacement = detect_crash(
            data,
            CRASH_DISTANCE_THRESHOLD
        )

        if crashed:

            crashed_uavs += 1

            crash_details.append({
                "uav_id": uav_id,
                "crash_index": crash_index,
                "displacement": displacement
            })

    if crash_total > 0:
        crash_percentage = (
            crashed_uavs /
            crash_total *
            100.0
        )
    else:
        crash_percentage = np.nan

    return {
        "experiment": exp_folder.name,

        "mission_duration_s": mean_mission_time,
        "objective_time_s": objective_time,

        "covered_area_percent": covered_area_percentage,

        "hidden_found_percent": hidden_percentage,

        "crash_percent": crash_percentage,

        "crashed_uavs": crashed_uavs,
        "total_uavs": crash_total,

        "hidden_found_uavs": hidden_found,
        "hidden_total_uavs": hidden_total,

        "crash_details": crash_details
    }


def generate_setting_report(setting_folder: Path):
    """
    Generate the final report for one setting.

    The report averages the metrics over all available experiments.
    """

    exp_folders = get_experiment_folders(setting_folder)

    if not exp_folders:
        print(
            f"WARNING: no experiments found for "
            f"{setting_folder.name}"
        )
        return None

    print(
        f"\nGenerating report for "
        f"{setting_folder.name}"
    )

    experiment_reports = []

    for exp_folder in exp_folders:

        print(
            f"  Reading {exp_folder.name}..."
        )

        report = compute_experiment_report(
            exp_folder
        )

        experiment_reports.append(report)

    # --------------------------------------------------
    # Average over experiments
    # --------------------------------------------------

    mission_values = [
        r["mission_duration_s"]
        for r in experiment_reports
        if np.isfinite(r["mission_duration_s"])
    ]

    objective_values = [
        r["objective_time_s"]
        for r in experiment_reports
        if np.isfinite(r["objective_time_s"])
    ]

    area_values = [
        r["covered_area_percent"]
        for r in experiment_reports
        if np.isfinite(r["covered_area_percent"])
    ]

    hidden_values = [
        r["hidden_found_percent"]
        for r in experiment_reports
        if np.isfinite(r["hidden_found_percent"])
    ]

    crash_values = [
        r["crash_percent"]
        for r in experiment_reports
        if np.isfinite(r["crash_percent"])
    ]

    summary = {
        "setting": setting_folder.name,

        "n_experiments": len(experiment_reports),

        "mean_mission_duration_s": (
            np.mean(mission_values)
            if mission_values else np.nan
        ),
        "variance_mission_duration_s": (
            np.var(mission_values)
            if mission_values else np.nan
        ),
        "std_mission_duration_s": (
            np.std(mission_values)
            if mission_values else np.nan
        ),

        "mean_objective_time_s": (
            np.mean(objective_values)
            if objective_values else np.nan
        ),
        "variance_objective_time_s": (
            np.var(objective_values)
            if objective_values else np.nan
        ),
        "std_objective_time_s": (
            np.std(objective_values)
            if objective_values else np.nan
        ),

        "mean_covered_area_percent": (
            np.mean(area_values)
            if area_values else np.nan
        ),
        "variance_covered_area_percent": (
            np.var(area_values)
            if area_values else np.nan
        ),
        "std_covered_area_percent": (
            np.std(area_values)
            if area_values else np.nan
        ),

        "mean_hidden_found_percent": (
            np.mean(hidden_values)
            if hidden_values else np.nan
        ),
        "variance_hidden_found_percent": (
            np.var(hidden_values)
            if hidden_values else np.nan
        ),
        "std_hidden_found_percent": (
            np.std(hidden_values)
            if hidden_values else np.nan
        ),

        "mean_crash_percent": (
            np.mean(crash_values)
            if crash_values else np.nan
        ),
        "variance_crash_percent": (
            np.var(crash_values)
            if crash_values else np.nan
        ),
        "std_crash_percent": (
            np.std(crash_values)
            if crash_values else np.nan
        )
    }

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    out = (
        setting_folder /
        "plots_mean_5_experiments"
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    csv_data = []

    for r in experiment_reports:

        csv_data.append({
            "experiment": r["experiment"],
            "mission_duration_s": r["mission_duration_s"],
            "objective_time_s": r["objective_time_s"],
            "covered_area_percent": r["covered_area_percent"],
            "hidden_found_percent": r["hidden_found_percent"],
            "crash_percent": r["crash_percent"],
            "crashed_uavs": r["crashed_uavs"],
            "total_uavs": r["total_uavs"],
            "hidden_found_uavs": r["hidden_found_uavs"],
            "hidden_total_uavs": r["hidden_total_uavs"]
        })

    df = pd.DataFrame(csv_data)

    # Add final mean row.
    df.loc[len(df)] = {
        "experiment": "MEAN",
        "mission_duration_s":
            summary["mean_mission_duration_s"],
        "objective_time_s":
            summary["mean_objective_time_s"],
        "covered_area_percent":
            summary["mean_covered_area_percent"],
        "hidden_found_percent":
            summary["mean_hidden_found_percent"],
        "crash_percent":
            summary["mean_crash_percent"],
        "crashed_uavs": np.nan,
        "total_uavs": np.nan,
        "hidden_found_uavs": np.nan,
        "hidden_total_uavs": np.nan
    }

    # Add final variance row.
    df.loc[len(df)] = {
        "experiment": "VARIANCE",
        "mission_duration_s":
            summary["variance_mission_duration_s"],
        "objective_time_s":
            summary["variance_objective_time_s"],
        "covered_area_percent":
            summary["variance_covered_area_percent"],
        "hidden_found_percent":
            summary["variance_hidden_found_percent"],
        "crash_percent":
            summary["variance_crash_percent"],
        "crashed_uavs": np.nan,
        "total_uavs": np.nan,
        "hidden_found_uavs": np.nan,
        "hidden_total_uavs": np.nan
    }

    # Add final standard-deviation row.
    df.loc[len(df)] = {
        "experiment": "STD",
        "mission_duration_s":
            summary["std_mission_duration_s"],
        "objective_time_s":
            summary["std_objective_time_s"],
        "covered_area_percent":
            summary["std_covered_area_percent"],
        "hidden_found_percent":
            summary["std_hidden_found_percent"],
        "crash_percent":
            summary["std_crash_percent"],
        "crashed_uavs": np.nan,
        "total_uavs": np.nan,
        "hidden_found_uavs": np.nan,
        "hidden_total_uavs": np.nan
    }

    csv_path = out / "report.csv"

    df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------
    # Save human-readable TXT report
    # --------------------------------------------------

    txt_path = out / "report.txt"

    with open(txt_path, "w") as f:

        f.write(
            "==================================================\n"
        )

        f.write(
            f"FINAL REPORT - {setting_folder.name}\n"
        )

        f.write(
            "==================================================\n\n"
        )

        f.write(
            f"Number of experiments: "
            f"{summary['n_experiments']}\n"
        )

        f.write(
            f"Crash threshold: "
            f"{CRASH_DISTANCE_THRESHOLD:.2f} m\n"
        )

        f.write(
            f"Coverage radius: "
            f"{COVERAGE_RADIUS:.2f} m\n\n"
        )

        f.write(
            "--------------------------------------------------\n"
        )

        f.write(
            "AVERAGE OVER EXPERIMENTS\n"
        )

        f.write(
            "--------------------------------------------------\n"
        )

        f.write(
            f"Mission duration: "
            f"{summary['mean_mission_duration_s']:.3f} s\n"
        )

        f.write(
            f"Objective time: "
            f"{summary['mean_objective_time_s']:.3f} s\n"
        )

        f.write(
            f"Covered area: "
            f"{summary['mean_covered_area_percent']:.3f} %\n"
        )

        f.write(
            f"Hidden waypoint found: "
            f"{summary['mean_hidden_found_percent']:.3f} %\n"
        )

        f.write(
            f"Crash rate: "
            f"{summary['mean_crash_percent']:.3f} %\n"
        )

        f.write("\n")

        f.write(
            "VARIANCE OVER EXPERIMENTS\n"
        )
        f.write(
            "--------------------------------------------------\n"
        )

        f.write(
            f"Mission duration variance: "
            f"{summary['variance_mission_duration_s']:.3f} s^2\n"
        )
        f.write(
            f"Mission duration standard deviation: "
            f"{summary['std_mission_duration_s']:.3f} s\n"
        )

        f.write(
            f"Objective time variance: "
            f"{summary['variance_objective_time_s']:.3f} s^2\n"
        )
        f.write(
            f"Objective time standard deviation: "
            f"{summary['std_objective_time_s']:.3f} s\n"
        )

        f.write(
            f"Covered area variance: "
            f"{summary['variance_covered_area_percent']:.3f} %^2\n"
        )
        f.write(
            f"Covered area standard deviation: "
            f"{summary['std_covered_area_percent']:.3f} %\n"
        )

        f.write(
            f"Hidden waypoint found variance: "
            f"{summary['variance_hidden_found_percent']:.3f} %^2\n"
        )
        f.write(
            f"Hidden waypoint found standard deviation: "
            f"{summary['std_hidden_found_percent']:.3f} %\n"
        )

        f.write(
            f"Crash rate variance: "
            f"{summary['variance_crash_percent']:.3f} %^2\n"
        )
        f.write(
            f"Crash rate standard deviation: "
            f"{summary['std_crash_percent']:.3f} %\n"
        )

        f.write("\n")

        f.write(
            "--------------------------------------------------\n"
        )

        f.write(
            "INDIVIDUAL EXPERIMENTS\n"
        )

        f.write(
            "--------------------------------------------------\n\n"
        )

        for r in experiment_reports:

            f.write(
                f"{r['experiment']}\n"
            )

            f.write(
                f"  Mission duration: "
                f"{r['mission_duration_s']:.3f} s\n"
            )

            if np.isfinite(r["objective_time_s"]):

                f.write(
                    f"  Objective time: "
                    f"{r['objective_time_s']:.3f} s\n"
                )

            else:

                f.write(
                    "  Objective time: N/A\n"
                )

            f.write(
                f"  Covered area: "
                f"{r['covered_area_percent']:.3f} %\n"
            )

            f.write(
                f"  Hidden waypoint: "
                f"{r['hidden_found_percent']:.3f} % "
                f"({r['hidden_found_uavs']}/"
                f"{r['hidden_total_uavs']} UAVs)\n"
            )

            f.write(
                f"  Crashes: "
                f"{r['crash_percent']:.3f} % "
                f"({r['crashed_uavs']}/"
                f"{r['total_uavs']} UAVs)\n"
            )

            if r["crash_details"]:

                for crash in r["crash_details"]:

                    f.write(
                        f"    UAV {crash['uav_id']}: "
                        f"iteration "
                        f"{crash['crash_index']}, "
                        f"displacement "
                        f"{crash['displacement']:.3f} m\n"
                    )

            f.write("\n")

    print(
        f"  Report saved to: {txt_path}"
    )

    print(
        f"  CSV saved to: {csv_path}"
    )

    return summary

DEFAULT_SETTINGS = [
    "1 UAV",
    "3 UAVS",
    "5 UAVS",
    "no leader",
    "Normal APF",
]



POSITION_FILE = "position.csv"
PRIORITY_FILE = "priority.csv"
VOLTAGE_FILE = "Voltage.csv"

# The script also checks these names, case-insensitively.
PHOTOSYNTHESIS_CANDIDATES = [
    "photosynthesis.csv",
    "photosynthetic.csv",
    "photosynthesis_production.csv",
]

DPI = 150
FIGSIZE = (4.0, 3.0)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def read_csv_numeric(path: Path) -> np.ndarray:
    """Read a CSV containing numeric values and return a 2D numpy array."""
    try:
        df = pd.read_csv(path, header=None)
        data = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    except Exception as exc:
        raise RuntimeError(f"Cannot read {path}: {exc}") from exc

    # Remove completely empty rows/columns.
    data = data[~np.all(np.isnan(data), axis=1)]
    data = data[:, ~np.all(np.isnan(data), axis=0)]

    return data

def truncate_after_crash(data: dict, threshold: float, uav_id: str, exp_folder: Path) -> dict:
    """
    Detect an abnormal displacement between consecutive UAV positions.

    If the distance between two consecutive positions is greater than
    'threshold', the data are truncated starting from the second position
    (the first sample after the jump).

    All available time-series data are truncated to the same length.

    Crash information is saved in:
        exp_folder / "crashinfo.txt"
    """

    crash_index = None
    crash_distance = None

    if "position" not in data:
        return data

    position = data["position"]

    if len(position) < 2:
        return data

    xy = position[:, :2]

    # Ignore samples containing NaN/inf.
    valid = np.isfinite(xy).all(axis=1)

    # --------------------------------------------------
    # Find first abnormal displacement
    # --------------------------------------------------

    for i in range(1, len(position)):

        if not valid[i] or not valid[i - 1]:
            continue

        distance = np.linalg.norm(
            xy[i] - xy[i - 1]
        )

        if distance > threshold:

            crash_index = i
            crash_distance = distance

            print(
                f"  2  CRASH detected for UAV {uav_id} "
                f"at iteration {i}: "
                f"displacement = {distance:.2f} m "
                f"(threshold = {threshold:.2f} m)"
            )

            break

    # --------------------------------------------------
    # Save crash information
    # --------------------------------------------------

    exp_folder.mkdir(parents=True, exist_ok=True)

    crashinfo_path = exp_folder / "crashinfo.txt"

    with open(crashinfo_path, "a") as f:
        f.write(
            f"UAV {uav_id}: No crash detected\n"
        )

        f.write(
            f"  Threshold: {threshold:.3f} m\n"
        )

        f.write(
            f"  Samples: {len(position)}\n"
        )

        f.write("\n")

        if crash_index is not None:
            print("writing crash info")
            f.truncate()
            f.write(
                f"UAV {uav_id}: CRASH detected\n"
            )

            f.write(
                f"  Crash iteration: {crash_index}\n"
            )

            f.write(
                f"  Displacement: {crash_distance:.3f} m\n"
            )

            f.write(
                f"  Threshold: {threshold:.3f} m\n"
            )

            f.write(
                f"  Samples before crash: {crash_index}\n"
            )

            f.write(
                f"  Original samples: {len(position)}\n"
            )

            f.write(
                f"  Samples after truncation: {crash_index}\n"
            )

            f.write("\n")

            

    # --------------------------------------------------
    # No crash
    # --------------------------------------------------

    if crash_index is None:
        return data

    # --------------------------------------------------
    # Truncate all time-series data
    # --------------------------------------------------

    truncated_data = {}

    for key, values in data.items():

        if isinstance(values, np.ndarray):

            truncated_data[key] = values[:crash_index - 1]

        else:

            truncated_data[key] = values

    print(
        f"    Data truncated: "
        f"{len(position)} -> {crash_index - 1} samples"
    )

    return truncated_data

def find_file(folder: Path, candidates: list[str]) -> Path | None:
    """Find a file case-insensitively."""
    files = {p.name.lower(): p for p in folder.iterdir() if p.is_file()}
    for name in candidates:
        if name.lower() in files:
            return files[name.lower()]
    return None


def safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()

def build_world_perimeter():
    """
    Build the actual world perimeter.

    Cutouts can intersect the left/right outer walls. When a cutout
    touches an outer wall, that portion of the outer wall is removed.
    """

    xmin = WORLD_OUTER["xmin"]
    xmax = WORLD_OUTER["xmax"]
    ymin = WORLD_OUTER["ymin"]
    ymax = WORLD_OUTER["ymax"]

    # --------------------------------------------------
    # Cutout coordinates
    # --------------------------------------------------

    cutouts = []

    for cut in WALL_CUTOUTS:
        x0 = cut["x"] - cut["sx"] / 2
        x1 = cut["x"] + cut["sx"] / 2
        y0 = cut["y"] - cut["sy"] / 2
        y1 = cut["y"] + cut["sy"] / 2

        # Clip to world boundary
        x0 = max(xmin, x0)
        x1 = min(xmax, x1)
        y0 = max(ymin, y0)
        y1 = min(ymax, y1)

        cutouts.append((x0, x1, y0, y1))

    # Sort cutouts from left to right
    cutouts.sort(key=lambda c: c[0])

    vertices = []
    codes = []

    # --------------------------------------------------
    # Start at bottom-left
    # --------------------------------------------------

    vertices.append((xmin, ymin))
    codes.append(MplPath.MOVETO)

    # --------------------------------------------------
    # LEFT OUTER WALL
    #
    # If a cutout touches xmin, stop the wall at its y0,
    # go horizontally to x1, then continue vertically
    # from x1 to ymax.
    # --------------------------------------------------

    left_cutouts = [
        c for c in cutouts
        if np.isclose(c[0], xmin)
    ]

    if left_cutouts:

        # For the current geometry, use the lowest cutout
        # touching the left boundary.
        cut = min(left_cutouts, key=lambda c: c[2])

        x0, x1, y0, y1 = cut

        # Left wall up to bottom of cutout
        vertices.append((xmin, y0))
        codes.append(MplPath.LINETO)

        # Bottom edge of cutout
        vertices.append((x1, y0))
        codes.append(MplPath.LINETO)

        # Right side of cutout
        vertices.append((x1, ymax))
        codes.append(MplPath.LINETO)

    else:
        # No cutout touches left wall
        vertices.append((xmin, ymax))
        codes.append(MplPath.LINETO)

    # --------------------------------------------------
    # TOP BOUNDARY
    #
    # Process cutouts that are not already connected
    # to the left wall.
    # --------------------------------------------------

    current_x = (
        left_cutouts[0][1]
        if left_cutouts
        else xmin
    )

    for x0, x1, y0, y1 in cutouts:

        # Skip cutout already handled on the left wall
        if np.isclose(x0, xmin):
            continue

        # Horizontal top wall before the cutout
        if x0 > current_x:
            vertices.append((x0, ymax))
            codes.append(MplPath.LINETO)

        # Down into cutout
        vertices.append((x0, y0))
        codes.append(MplPath.LINETO)

        # Bottom of cutout
        vertices.append((x1, y0))
        codes.append(MplPath.LINETO)

        # Back up
        vertices.append((x1, ymax))
        codes.append(MplPath.LINETO)

        current_x = x1

    # --------------------------------------------------
    # RIGHT OUTER WALL
    #
    # If a cutout touches xmax, the right wall must stop
    # at its y0 instead of continuing through the cutout.
    # --------------------------------------------------

    right_cutouts = [
        c for c in cutouts
        if np.isclose(c[1], xmax)
    ]

    if right_cutouts:

        cut = min(right_cutouts, key=lambda c: c[2])

        x0, x1, y0, y1 = cut

        # Go horizontally to left side of cutout
        if current_x < x0:
            vertices.append((x0, ymax))
            codes.append(MplPath.LINETO)

        # Down to bottom of cutout
        vertices.append((x0, y0))
        codes.append(MplPath.LINETO)

        # Bottom edge toward outer wall
        vertices.append((xmax, y0))
        codes.append(MplPath.LINETO)

        # Continue down the right outer wall
        vertices.append((xmax, ymin))
        codes.append(MplPath.LINETO)

    else:
        # No cutout touches right wall
        if current_x < xmax:
            vertices.append((xmax, ymax))
            codes.append(MplPath.LINETO)

        vertices.append((xmax, ymin))
        codes.append(MplPath.LINETO)

    # --------------------------------------------------
    # Bottom wall
    # --------------------------------------------------

    vertices.append((xmin, ymin))
    codes.append(MplPath.CLOSEPOLY)

    return MplPath(vertices, codes)

def plot_world_walls(ax):
    """
    Plot the actual world wall geometry.

    The cutouts are openings on the top side of the world.
    Therefore:
      - the top wall is interrupted by the cutouts;
      - the two vertical sides of each cutout are walls;
      - the bottom side of each cutout is a wall;
      - portions of cutouts outside the world boundary are clipped.
    """

    xmin = WORLD_OUTER["xmin"]
    xmax = WORLD_OUTER["xmax"]
    ymin = WORLD_OUTER["ymin"]
    ymax = WORLD_OUTER["ymax"]

    # --------------------------------------------------
    # CUTOUT LIMITS
    # --------------------------------------------------

    cutout_ranges = []

    for cut in WALL_CUTOUTS:

        x0 = cut["x"] - cut["sx"] / 2
        x1 = cut["x"] + cut["sx"] / 2

        y0 = cut["y"] - cut["sy"] / 2
        y1 = cut["y"] + cut["sy"] / 2

        # Clip the cutout to the world boundary
        x0 = max(xmin, x0)
        x1 = min(xmax, x1)

        y0 = max(ymin, y0)
        y1 = min(ymax, y1)

        # Ignore invalid cutouts
        if x0 >= x1 or y0 >= y1:
            continue

        cutout_ranges.append((x0, x1, y0, y1))

    # Sort from left to right
    cutout_ranges.sort(key=lambda c: c[0])

    # --------------------------------------------------
    # OUTER WALLS
    # --------------------------------------------------

    # Bottom wall
    ax.plot(
        [xmin, xmax],
        [ymin, ymin],
        color="black",
        linestyle="--",
        linewidth=3,
        zorder=20
    )

    # Left wall
    ax.plot(
        [xmin, xmin],
        [ymin, ymax],
        color="black",
        linestyle="--",
        linewidth=3,
        zorder=20
    )

    # Right wall
    ax.plot(
        [xmax, xmax],
        [ymin, ymax],
        color="black",
        linestyle="--",
        linewidth=3,
        zorder=20
    )

    # --------------------------------------------------
    # TOP WALL
    # --------------------------------------------------

    current_x = xmin

    for x0, x1, y0, y1 in cutout_ranges:

        # Top wall before cutout
        if x0 > current_x:

            ax.plot(
                [current_x, x0],
                [ymax, ymax],
                color="black",
                linestyle="--",
                linewidth=3,
                zorder=20
            )

        current_x = max(current_x, x1)

    # Top wall after last cutout
    if current_x < xmax:

        ax.plot(
            [current_x, xmax],
            [ymax, ymax],
            color="black",
            linestyle="--",
            linewidth=3,
            zorder=20
        )

    # --------------------------------------------------
    # CUTOUT WALLS
    # --------------------------------------------------

    for x0, x1, y0, y1 in cutout_ranges:

        # LEFT vertical side
        #
        # Only draw it if the cutout actually starts
        # inside the world.
        if x0 > xmin:

            ax.plot(
                [x0, x0],
                [y0, ymax],
                color="black",
                linestyle="--",
                linewidth=3,
                zorder=20
            )

        # RIGHT vertical side
        if x1 < xmax:

            ax.plot(
                [x1, x1],
                [y0, ymax],
                color="black",
                linestyle="--",
                linewidth=3,
                zorder=20
            )

        # Bottom side of cutout
        ax.plot(
            [x0, x1],
            [y0, y0],
            color="black",
            linestyle="--",
            linewidth=3,
            zorder=20
        )

def mean_var(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Align arrays by row/iteration index and calculate mean and population
    variance, ignoring missing values.
    """
    if not arrays:
        return np.array([]), np.array([])

    max_len = max(len(a) for a in arrays)
    max_cols = max((a.shape[1] if a.ndim > 1 else 1) for a in arrays)

    stack = np.full((len(arrays), max_len, max_cols), np.nan)

    for i, a in enumerate(arrays):
        if a.ndim == 1:
            a = a[:, None]
        rows = min(len(a), max_len)
        cols = min(a.shape[1], max_cols)
        stack[i, :rows, :cols] = a[:rows, :cols]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        var = np.nanvar(stack, axis=0)

    return mean, var


def align_scalar_series(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance for 1D time series."""
    if not arrays:
        return np.array([]), np.array([])

    max_len = max(len(a) for a in arrays)
    stack = np.full((len(arrays), max_len), np.nan)

    for i, a in enumerate(arrays):
        stack[i, :len(a)] = a

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(stack, axis=0), np.nanvar(stack, axis=0)


# ---------------------------------------------------------------------------
# Convex hull / covered area
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Covered area
# ---------------------------------------------------------------------------

COVERAGE_RADIUS = 3.0


def covered_area_geometry(positions: list[np.ndarray]):
    """
    Calculate the geometric union of circles with radius COVERAGE_RADIUS
    centered at all visited UAV positions.

    Overlapping areas are counted only once.
    """
    circles = []

    for xy in positions:
        for point in xy:
            if np.isfinite(point[:2]).all():
                circles.append(
                    Point(
                        float(point[0]),
                        float(point[1])
                    ).buffer(COVERAGE_RADIUS)
                )

    if not circles:
        return None

    return unary_union(circles)

# ---------------------------------------------------------------------------
# Spatial coverage statistics across experiments
# ---------------------------------------------------------------------------

def build_coverage_mask(xy, xmin, xmax, ymin, ymax, resolution,
                        coverage_radius):
    """
    Build a boolean spatial coverage mask for one experiment.

    A grid cell is considered covered if its center lies inside the
    union of circles of radius coverage_radius centered on the trajectory.
    """

    x = np.arange(xmin, xmax + resolution, resolution)
    y = np.arange(ymin, ymax + resolution, resolution)

    X, Y = np.meshgrid(x, y)

    covered = np.zeros_like(X, dtype=bool)

    for point in xy:
        px, py = point

        if not np.isfinite(px) or not np.isfinite(py):
            continue

        covered |= (
            (X - px) ** 2 +
            (Y - py) ** 2
            <= coverage_radius ** 2
        )

    return x, y, covered


def aggregate_spatial_coverage(all_exp_data, resolution=0.2):
    """
    Calculate mean and variance of spatial coverage across experiments.

    For every experiment:
        1 = covered
        0 = not covered

    Then calculate:
        mean coverage = fraction of experiments covering each cell
        variance     = variance of coverage across experiments
    """

    experiments = []

    for exp in all_exp_data:

        positions = []

        for uav_id in sorted(exp.keys(), key=int):

            if "position" not in exp[uav_id]:
                continue

            xy = exp[uav_id]["position"][:, :2]

            valid = np.isfinite(xy).all(axis=1)
            xy = xy[valid]

            if len(xy):
                positions.append(xy)

        if positions:
            experiments.append(positions)

    if not experiments:
        return None

    xmin = WORLD_OUTER["xmin"]
    xmax = WORLD_OUTER["xmax"]
    ymin = WORLD_OUTER["ymin"]
    ymax = WORLD_OUTER["ymax"]

    coverage_masks = []

    for positions in experiments:

        # All UAV trajectories belonging to this experiment
        all_positions = np.vstack(positions)

        x, y, mask = build_coverage_mask(
            all_positions,
            xmin,
            xmax,
            ymin,
            ymax,
            resolution,
            COVERAGE_RADIUS
        )

        coverage_masks.append(mask.astype(float))

    coverage_stack = np.stack(coverage_masks, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        mean_coverage = np.nanmean(
            coverage_stack,
            axis=0
        )

        variance_coverage = np.nanvar(
            coverage_stack,
            axis=0
        )

    return x, y, mean_coverage, variance_coverage

def cumulative_coverage_area(xy, resolution=0.2):
    """
    Calculate cumulative covered area over time using a grid.

    Returns one value per trajectory iteration.
    """

    if len(xy) == 0:
        return np.array([])

    xmin = WORLD_OUTER["xmin"]
    xmax = WORLD_OUTER["xmax"]
    ymin = WORLD_OUTER["ymin"]
    ymax = WORLD_OUTER["ymax"]

    x = np.arange(
        xmin,
        xmax + resolution,
        resolution
    )

    y = np.arange(
        ymin,
        ymax + resolution,
        resolution
    )

    X, Y = np.meshgrid(x, y)

    covered = np.zeros_like(X, dtype=bool)

    area_per_cell = resolution ** 2

    areas = []

    for px, py in xy:

        if not np.isfinite(px) or not np.isfinite(py):
            areas.append(np.nan)
            continue

        covered |= (
            (X - px) ** 2 +
            (Y - py) ** 2
            <= COVERAGE_RADIUS ** 2
        )

        areas.append(
            np.sum(covered) * area_per_cell
        )

    return np.asarray(areas)

def load_leader_photosynthesis(exp_folder: Path) -> np.ndarray | None:
    """
    Load photosynthesis.csv from the leader folder of one experiment.

    Each row corresponds to one iteration.
    """
    leader_folder = exp_folder / "leader"

    if not leader_folder.is_dir():
        print(f"    WARNING: leader folder not found: {leader_folder}")
        return None

    photo_path = find_file(
        leader_folder,
        PHOTOSYNTHESIS_CANDIDATES
    )

    if photo_path is None:
        print(f"    WARNING: photosynthesis.csv not found in {leader_folder}")
        return None

    data = read_csv_numeric(photo_path)

    if data.shape[1] == 0:
        return None

    # First numeric column
    return data[:, 0]

def load_uav_data(uav_folder: Path, uav_id: str, exp_folder: Path) -> dict:
    """Load all available data for one UAV."""
    result = {}

    position_path = find_file(uav_folder, [POSITION_FILE])
    priority_path = find_file(uav_folder, [PRIORITY_FILE])
    voltage_path = find_file(uav_folder, [VOLTAGE_FILE])

    if position_path is not None:
        pos = read_csv_numeric(position_path)
        if pos.shape[1] < 4:
            raise ValueError(f"{position_path} must have x,y,z,yaw.")
        result["position"] = pos[:, :4]

    if priority_path is not None:
        result["priority"] = read_csv_numeric(priority_path)

    if voltage_path is not None:
        voltage = read_csv_numeric(voltage_path)
        if voltage.shape[1] >= 1:
            result["voltage"] = voltage[:, 0]

    photo_path = find_file(uav_folder, PHOTOSYNTHESIS_CANDIDATES)
    if photo_path is not None:
        photo = read_csv_numeric(photo_path)
        if photo.shape[1] >= 1:
            result["photosynthesis"] = photo[:, 0]

    return result


def discover_uavs(exp_folder: Path) -> list[str]:
    """Return numeric UAV folder names, sorted numerically."""
    ids = []
    for p in exp_folder.iterdir():
        if p.is_dir() and p.name.isdigit():
            ids.append(p.name)

    def key(x):
        return int(x)

    return sorted(ids, key=key)


# ---------------------------------------------------------------------------
# Individual experiment plots
# ---------------------------------------------------------------------------
def plot_experiment_photosynthesis(
    exp_folder: Path,
    out: Path,
    exp_name: str
):
    """
    Plot leader photosynthesis for one experiment.
    """

    photosynthesis = load_leader_photosynthesis(exp_folder)

    if photosynthesis is None or len(photosynthesis) == 0:
        return

    # Each iteration = 0.15 s
    t = get_time_axis(len(photosynthesis))

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(
        t,
        photosynthesis,
        linewidth=1.5,
        label="Photosynthesis"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Photosynthetic production")
    # #ax.set_title(
    #     f"{exp_name} - Leader - Photosynthetic production"
    # )

    ax.grid(True, alpha=0.3)
    ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.33),
    ncol=3,
    frameon=False
)

    

    savefig(out / "leader_photosynthesis.png")

def plot_uav_position(uav_id: str, data: dict, out: Path, exp_name: str):
    if "position" not in data:
        return

    p = data["position"]
    t = get_time_axis(len(p))

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    labels = ["x [m]", "y [m]", "z [m]", "yaw [rad]"]

    for i, ax in enumerate(axes):
        ax.plot(t, p[:, i])
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Iteration")
    # fig.suptitle(f"{exp_name} - UAV {uav_id} - Position")
    savefig(out / "position_time.png")

def plot_world(out: Path):

    fig, ax = plt.subplots(figsize=FIGSIZE)
    init_pos = [
        (4.0, -3.0, 28),
        (6.0, -3.0, 29),
        (5.0, -2.0, 30),
        (4.0, -1.0, 31),
        (6.0, -1.0, 32),
    ]
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for i, (x,y,id) in enumerate(init_pos):
        ax.scatter(
            x,
            y,
            facecolors="none",
            edgecolors=colors[i],
            linewidths=1.0,
            s=SCATTER_MARKER_SIZE,
            zorder=30,
            label=f"UAV {id}",
        )
    plot_world_walls(ax)

    if OBSTACLES:
        ox, oy = zip(*OBSTACLES)
        ax.scatter(
            ox,
            oy,
            color="red",
            s=SCATTER_MARKER_SIZE,
            zorder=25,
            label="Obstacle",
        )
    for cs in CHARGIN_STATION:
        circle = plt.Circle((cs[0], cs[1]), 2, fill=False, color="cyan", linewidth=2.0, zorder=20, label="Charging area")
        ax.add_patch(circle)
    
    if WAYPOINTS:
        wx, wy = zip(*WAYPOINTS)
        ax.scatter(
            wx,
            wy,
            facecolors="green",
            edgecolors="green",
            linewidths=1.0,
            s=SCATTER_MARKER_SIZE*3,
            zorder=30,
            label="POI",
        )
    if HIDDEN_WAYPOINT:
        wx, wy = zip(*HIDDEN_WAYPOINT)
        ax.scatter(
            wx,
            wy,
            facecolors="midnightblue",
            edgecolors="midnightblue",
            linewidths=1.0,
            s=SCATTER_MARKER_SIZE*3,
            zorder=30,
            label="Hidden POI",
        )



        
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ##ax.set_title(f"world")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.33),
    ncol=3,
    frameon=False
)

    

    savefig(out / "world.png")

def plot_uav_path(uav_id: str, data: dict, out: Path, exp_name: str):
    if "position" not in data:
        return

    p = data["position"]
    xy = p[:, :2]
    valid = np.isfinite(xy).all(axis=1)
    xy = xy[valid]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_world_walls(ax)
    if OBSTACLES:
        ox, oy = zip(*OBSTACLES)
        ax.scatter(
            ox,
            oy,
            color="red",
            s=SCATTER_MARKER_SIZE,
            zorder=25,
            label="Obstacle",
        )
    for cs in CHARGIN_STATION:
        circle = plt.Circle((cs[0], cs[1]), 2, fill=False, color="cyan", linewidth=2.0, zorder=20, label="Charging area")
        ax.add_patch(circle)
    if WAYPOINTS:
        wx, wy = zip(*WAYPOINTS)
        ax.scatter(
            wx,
            wy,
            facecolors="green",
            edgecolors="green",
            linewidths=1.0,
            s=SCATTER_MARKER_SIZE*3,
            zorder=30,
            label="POI",
        )
    if HIDDEN_WAYPOINT:
        wx, wy = zip(*HIDDEN_WAYPOINT)
        ax.scatter(
            wx,
            wy,
            facecolors="midnightblue",
            edgecolors="midnightblue",
            linewidths=1.0,
            s=SCATTER_MARKER_SIZE*3,
            zorder=30,
            label="Hidden POI",
        )

    if len(xy):
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            linewidth=1.2,
            label="Path"
        )

        ax.scatter(
            xy[0, 0],
            xy[0, 1],
            marker="o",
            s=SCATTER_MARKER_SIZE*3,
            label="Start"
        )

        ax.scatter(
            xy[-1, 0],
            xy[-1, 1],
            marker="x",
            s=SCATTER_MARKER_SIZE*3,
            label="End"
        )

        coverage = covered_area_geometry([xy])

        if coverage is not None:

            if coverage.geom_type == "Polygon":
                x, y = coverage.exterior.xy
                ax.fill(
                    x,
                    y,
                    alpha=0.15,
                    label="Covered area"
                )

            elif coverage.geom_type == "MultiPolygon":
                first = True

                for polygon in coverage.geoms:
                    x, y = polygon.exterior.xy

                    ax.fill(
                        x,
                        y,
                        alpha=0.15,
                        label="Covered area"
                        if first else None
                    )

                    first = False

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    #ax.set_title(f"{exp_name} - UAV {uav_id} - 2D path")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(WORLD_OUTER["xmin"] - 1, WORLD_OUTER["xmax"] + 1)
    ax.set_ylim(WORLD_OUTER["ymin"] - 1, WORLD_OUTER["ymax"] + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.33),
    ncol=3,
    frameon=False
)

    

    savefig(out / "path_2d.png")





def plot_priorities(uav_id: str, data: dict, out: Path, exp_name: str):
    if "priority" not in data:
        return

    pr = data["priority"]
    t = get_time_axis(len(pr))

    fig, ax = plt.subplots(figsize=FIGSIZE)

    n = pr.shape[1]
    labels = []
    for i in range(n):
        if i == n - 1:
            labels.append("Battery")
        elif i == n - 2:
            labels.append("Hidden resource")
        else:
            labels.append(f"Resource {i + 1}")

    for i, label in enumerate(labels):
        ax.plot(t, pr[:, i], label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Priority")
    #ax.set_title(f"{exp_name} - UAV {uav_id} - Resource priorities")
    ax.grid(True, alpha=0.3)
    ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.33),
    ncol=3,
    frameon=False
)

    
    savefig(out / "priorities.png")

    # Dedicated battery-priority graph.
    if n >= 1:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(t, pr[:, -1])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Battery priority")
        #ax.set_title(f"{exp_name} - UAV {uav_id} - Battery priority")
        ax.grid(True, alpha=0.3)
        savefig(out / "battery_priority.png")


def plot_voltage(uav_id: str, data: dict, out: Path, exp_name: str):
    if "voltage" not in data:
        return

    v = data["voltage"]
    t = get_time_axis(len(v))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(t, v)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Voltage [V]")
    #ax.set_title(f"{exp_name} - UAV {uav_id} - Battery voltage")
    ax.grid(True, alpha=0.3)
    savefig(out / "voltage.png")


def plot_photosynthesis(uav_id: str, data: dict, out: Path, exp_name: str):
    if "photosynthesis" not in data:
        return

    p = data["photosynthesis"]
    t = get_time_axis(len(p))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(t, p)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Photosynthetic production")
    #ax.set_title(f"{exp_name} - UAV {uav_id} - Photosynthetic production")
    ax.grid(True, alpha=0.3)
    savefig(out / "photosynthesis.png")


def make_individual_uav_plots(uav_id, data, out, exp_name):
    out.mkdir(parents=True, exist_ok=True)
    plot_uav_position(uav_id, data, out, exp_name)
    plot_uav_path(uav_id, data, out, exp_name)
    plot_priorities(uav_id, data, out, exp_name)
    plot_voltage(uav_id, data, out, exp_name)
    plot_photosynthesis(uav_id, data, out, exp_name)


# ---------------------------------------------------------------------------
# Experiment-level plots
# ---------------------------------------------------------------------------
def build_exp_data(exp_folder: Path) -> dict[str, dict]:
    result = {}
    crash_infos = []

    for uav_id in discover_uavs(exp_folder):

        uav_folder = exp_folder / uav_id

        data = load_uav_data(uav_folder, uav_id, exp_folder)

        result[uav_id] = data


    # --------------------------------------------------
    # SAVE CRASH INFORMATION
    # --------------------------------------------------

    crash_file = exp_folder / "crashinfo.txt"

    with open(crash_file, "w") as f:

        f.write("CRASH DETECTION REPORT\n")
        f.write("======================\n\n")

        f.write(
            f"Crash threshold: "
            f"{CRASH_DISTANCE_THRESHOLD:.2f} m\n\n"
        )

        if not crash_infos:

            f.write("No crashes detected.\n")

        else:

            f.write(
                f"Detected crashes: {len(crash_infos)}\n\n"
            )

            for info in crash_infos:

                f.write(
                    f"UAV {info['uav_id']}\n"
                )

                f.write(
                    f"  Crash detected at iteration: "
                    f"{info['iteration']}\n"
                )

                f.write(
                    f"  Previous iteration: "
                    f"{info['previous_iteration']}\n"
                )

                f.write(
                    f"  Position before jump: "
                    f"x={info['previous_position'][0]:.4f}, "
                    f"y={info['previous_position'][1]:.4f}, "
                    f"z={info['previous_position'][2]:.4f}\n"
                )

                f.write(
                    f"  Position after jump: "
                    f"x={info['crash_position'][0]:.4f}, "
                    f"y={info['crash_position'][1]:.4f}, "
                    f"z={info['crash_position'][2]:.4f}\n"
                )

                f.write(
                    f"  Position jump: "
                    f"{info['distance']:.4f} m\n"
                )

                f.write(
                    f"  Data kept until iteration: "
                    f"{info['trim_length']}\n"
                )

                f.write("\n")

    return result


def pairwise_distances(exp_data: dict[str, dict]) -> tuple[list[tuple[str, str]], np.ndarray]:
    pairs = []
    series = []

    for a, b in combinations(sorted(exp_data.keys(), key=int), 2):
        if "position" not in exp_data[a] or "position" not in exp_data[b]:
            continue

        pa = exp_data[a]["position"]
        pb = exp_data[b]["position"]
        n = min(len(pa), len(pb))

        d = np.linalg.norm(pa[:n, :3] - pb[:n, :3], axis=1)
        pairs.append((a, b))
        series.append(d)

    if not series:
        return pairs, np.empty((0, 0))

    max_len = max(len(x) for x in series)
    arr = np.full((len(series), max_len), np.nan)
    for i, x in enumerate(series):
        arr[i, :len(x)] = x

    return pairs, arr


def plot_experiment_distances(exp_data, out, exp_name):
    pairs, arr = pairwise_distances(exp_data)
    if len(pairs) == 0:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    t = get_time_axis(arr.shape[1])

    for i, pair in enumerate(pairs):
        ax.plot(t, arr[i], label=f"UAV {pair[0]} - UAV {pair[1]}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance [m]")
    #ax.set_title(f"{exp_name} - Inter-UAV distances")
    ax.grid(True, alpha=0.3)
    ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.33),
    ncol=3,
    frameon=False
)

    
    savefig(out / "uav_distances.png")


def plot_experiment_path_and_area(exp_data, out, exp_name):
    positions = []
    labels = []

    for uav_id in sorted(exp_data.keys(), key=int):
        if "position" in exp_data[uav_id]:
            xy = exp_data[uav_id]["position"][:, :2]
            valid = np.isfinite(xy).all(axis=1)
            xy = xy[valid]

            if len(xy):
                positions.append(xy)
                labels.append(uav_id)

    if not positions:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_world_walls(ax)    
    if OBSTACLES:
        ox, oy = zip(*OBSTACLES)
        ax.scatter(
            ox,
            oy,
            color="red",
            s=SCATTER_MARKER_SIZE,
            zorder=25,
            label="Obstacle",
        )

    for cs in CHARGIN_STATION:
        circle = plt.Circle((cs[0], cs[1]), 2, fill=False, color="cyan", linewidth=2.0, zorder=20, label="Charging area")
        ax.add_patch(circle)
    if WAYPOINTS:
        wx, wy = zip(*WAYPOINTS)
        ax.scatter(
            wx,
            wy,
            facecolors="green",
            edgecolors="green",
            linewidths=1.0,
            s=SCATTER_MARKER_SIZE*3,
            zorder=30,
            label="POI",
        )
    if HIDDEN_WAYPOINT:
        wx, wy = zip(*HIDDEN_WAYPOINT)
        ax.scatter(
            wx,
            wy,
            facecolors="midnightblue",
            edgecolors="midnightblue",
            linewidths=1.0,
            s=SCATTER_MARKER_SIZE*3,
            zorder=30,
            label="Hidden POI",
        )


    # ------------------------------------------------------------------
    # Final covered area:
    # union of all radius-2 m circles centered at all visited positions
    # ------------------------------------------------------------------

    coverage = covered_area_geometry(positions)

    if coverage is not None:

        if coverage.geom_type == "Polygon":

            x, y = coverage.exterior.xy

            ax.fill(
                x,
                y,
                alpha=0.15,
                label="Covered area"
            )

        elif coverage.geom_type == "MultiPolygon":

            first = True

            for polygon in coverage.geoms:

                x, y = polygon.exterior.xy

                ax.fill(
                    x,
                    y,
                    alpha=0.15,
                    label="Covered area"
                    if first else None
                )

                first = False

    # ------------------------------------------------------------------
    # UAV paths
    # ------------------------------------------------------------------

    for xy, uav_id in zip(positions, labels):

        ax.plot(
            xy[:, 0],
            xy[:, 1],
            linewidth=1.2,
            label=f"UAV {uav_id}"
        )

        ax.scatter(
            xy[0, 0],
            xy[0, 1],
            s=SCATTER_MARKER_SIZE
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    #ax.set_title(f"{exp_name} - UAV paths and covered area")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(WORLD_OUTER["xmin"] - 1, WORLD_OUTER["xmax"] + 1)
    ax.set_ylim(WORLD_OUTER["ymin"] - 1, WORLD_OUTER["ymax"] + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.33),
    ncol=3,
    frameon=False
)

    

    savefig(out / "paths_and_covered_area.png")


def make_experiment_plots(exp_folder, exp_data):
    out = exp_folder / "plots"
    out.mkdir(exist_ok=True)

    exp_name = exp_folder.name
    plot_experiment_distances(exp_data, out, exp_name)
    plot_experiment_path_and_area(exp_data, out, exp_name)
    plot_experiment_photosynthesis(exp_folder,out,exp_name)


# ---------------------------------------------------------------------------
# Aggregation across the 5 experiments
# ---------------------------------------------------------------------------
def aggregate_leader_photosynthesis(
    exp_folders: list[Path]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and variance of leader photosynthesis
    across experiments, aligned by iteration.
    """

    arrays = []

    for exp_folder in exp_folders:
        photosynthesis = load_leader_photosynthesis(exp_folder)

        if photosynthesis is not None and len(photosynthesis) > 0:
            arrays.append(photosynthesis)

    if not arrays:
        return np.array([]), np.array([])

    return align_scalar_series(arrays)

def get_experiment_folders(setting_folder: Path) -> list[Path]:
    folders = [
        p for p in setting_folder.iterdir()
        if p.is_dir() and re.match(r"^exp\s*\d+$", p.name, re.IGNORECASE)
    ]

    def exp_number(p):
        m = re.search(r"(\d+)", p.name)
        return int(m.group(1)) if m else 999999

    return sorted(folders, key=exp_number)


def aggregate_positions(all_exp_data):
    """Return {uav_id: (mean, variance)} for x,y,z,yaw."""
    ids = sorted(
        {uav for exp in all_exp_data for uav in exp.keys()},
        key=int
    )

    result = {}
    for uav in ids:
        arrays = [
            exp[uav]["position"]
            for exp in all_exp_data
            if uav in exp and "position" in exp[uav]
        ]
        if arrays:
            result[uav] = mean_var(arrays)

    return result

def plot_aggregated_leader_photosynthesis(
    exp_folders: list[Path],
    out: Path,
    setting_name: str
):
    """
    Plot mean leader photosynthesis across experiments
    with ±1 standard deviation as a colored area.
    """

    mean, var = aggregate_leader_photosynthesis(exp_folders)

    if len(mean) == 0:
        return

    std = np.sqrt(var)

    t = get_time_axis(len(mean))

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(
        t,
        mean,
        linewidth=2.0,
        label="Mean"
    )

    ax.fill_between(
        t,
        mean - std,
        mean + std,
        alpha=0.25,
        label="±1 std"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Photosynthetic production")

    #ax.set_title(
    #     f"{setting_name} - Leader photosynthetic production "
    #     f"- Mean ± std"
    # )

    ax.grid(True, alpha=0.3)
    ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.33),
    ncol=3,
    frameon=False
)

    

    savefig(out / "mean_leader_photosynthesis.png")

def plot_aggregated_positions(position_stats, out, setting_name):
    for uav_id, (mean, var) in position_stats.items():
        t = get_time_axis(len(mean))
        std = np.sqrt(var)

        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        labels = ["x [m]", "y [m]", "z [m]", "yaw [rad]"]

        for i, ax in enumerate(axes):
            ax.plot(t, mean[:, i], label="Mean")
            ax.fill_between(
                t,
                mean[:, i] - std[:, i],
                mean[:, i] + std[:, i],
                alpha=0.2,
                label="±1 std",
            )
            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Iteration")
        axes[0].legend(loc="upper left", ncol=2)
        fig.suptitle(f"{setting_name} - UAV {uav_id} - Mean position ± std")
        savefig(out / f"mean_position_UAV_{uav_id}.png")


def aggregate_scalar_per_uav(all_exp_data, key):
    ids = sorted(
        {uav for exp in all_exp_data for uav in exp.keys()},
        key=int
    )
    result = {}

    for uav in ids:
        arrays = [
            exp[uav][key]
            for exp in all_exp_data
            if uav in exp and key in exp[uav]
        ]
        if arrays:
            result[uav] = align_scalar_series(arrays)

    return result


def plot_aggregated_scalar_per_uav(stats, out, setting_name, key, ylabel, filename):
    for uav_id, (mean, var) in stats.items():
        t = get_time_axis(len(mean))
        std = np.sqrt(var)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(t, mean, label="Mean")
        ax.fill_between(t, mean - std, mean + std, alpha=0.2, label="±1 std")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        #ax.set_title(f"{setting_name} - UAV {uav_id} - {ylabel} - Mean ± std")
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.33),
            ncol=3,
            frameon=False
        )

        
        savefig(out / f"{filename}_UAV_{uav_id}.png")


def aggregate_priorities(all_exp_data):
    ids = sorted(
        {uav for exp in all_exp_data for uav in exp.keys()},
        key=int
    )

    result = {}
    for uav in ids:
        arrays = [
            exp[uav]["priority"]
            for exp in all_exp_data
            if uav in exp and "priority" in exp[uav]
        ]
        if arrays:
            result[uav] = mean_var(arrays)

    return result


def plot_aggregated_priorities(priority_stats, out, setting_name):
    for uav_id, (mean, var) in priority_stats.items():
        t = get_time_axis(len(mean))
        std = np.sqrt(var)
        n = mean.shape[1]

        labels = []
        for i in range(n):
            if i == n - 1:
                labels.append("Battery")
            elif i == n - 2:
                labels.append("Hidden resource")
            else:
                labels.append(f"Resource {i + 1}")

        fig, ax = plt.subplots(figsize=FIGSIZE)

        for i, label in enumerate(labels):
            ax.plot(t, mean[:, i], label=label)
            ax.fill_between(
                t,
                mean[:, i] - std[:, i],
                mean[:, i] + std[:, i],
                alpha=0.12,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Priority")
        #ax.set_title(f"{setting_name} - UAV {uav_id} - Priorities - Mean ± std")
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.33),
            ncol=3,
            frameon=False
        )

        
        savefig(out / f"mean_priorities_UAV_{uav_id}.png")

        # Battery priority separately
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(t, mean[:, -1], label="Mean")
        ax.fill_between(
            t,
            mean[:, -1] - std[:, -1],
            mean[:, -1] + std[:, -1],
            alpha=0.2,
            label="±1 std",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Battery priority")
        #ax.set_title(f"{setting_name} - UAV {uav_id} - Battery priority - Mean ± std")
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.33),
            ncol=3,
            frameon=False
        )

        
        savefig(out / f"mean_battery_priority_UAV_{uav_id}.png")


def aggregate_distances(all_exp_data):
    """Aggregate each UAV-pair distance across experiments."""
    all_pairs = sorted(
        {
            pair
            for exp in all_exp_data
            for pair in combinations(sorted(exp.keys(), key=int), 2)
        },
        key=lambda p: (int(p[0]), int(p[1]))
    )

    result = {}

    for pair in all_pairs:
        arrays = []
        for exp in all_exp_data:
            a, b = pair
            if (
                a in exp and b in exp
                and "position" in exp[a]
                and "position" in exp[b]
            ):
                pa = exp[a]["position"]
                pb = exp[b]["position"]
                n = min(len(pa), len(pb))
                arrays.append(np.linalg.norm(
                    pa[:n, :3] - pb[:n, :3], axis=1
                ))

        if arrays:
            result[pair] = align_scalar_series(arrays)

    return result


def plot_aggregated_distances(distance_stats, out, setting_name):
    if not distance_stats:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for pair, (mean, var) in distance_stats.items():
        t = get_time_axis(len(mean))
        std = np.sqrt(var)
        label = f"UAV {pair[0]} - UAV {pair[1]}"

        ax.plot(t, mean, label=label)
        ax.fill_between(t, mean - std, mean + std, alpha=0.12)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance [m]")
    #ax.set_title(f"{setting_name} - Inter-UAV distance - Mean ± std")
    ax.grid(True, alpha=0.3)
    ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.33),
    ncol=3,
    frameon=False
)

    
    savefig(out / "mean_uav_distances.png")


def aggregate_paths(all_exp_data):
    """Mean x/y trajectory per UAV."""
    return aggregate_positions(all_exp_data)


def plot_aggregated_paths(path_stats, out, setting_name, all_exp_data):
    """
    Plot mean UAV paths with:

    - spatial ±1 std uncertainty area;
    - mean spatial coverage across experiments;
    - spatial coverage variance;
    - waypoints;
    - world walls.
    """

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # ------------------------------------------------------------------
    # WORLD
    # ------------------------------------------------------------------

    plot_world_walls(ax)

    # ------------------------------------------------------------------
    # WAYPOINTS
    # ------------------------------------------------------------------
    if OBSTACLES:
        ox, oy = zip(*OBSTACLES)
        ax.scatter(
            ox,
            oy,
            color="red",
            s=SCATTER_MARKER_SIZE,
            zorder=25,
            label="Obstacle",
        )
    for cs in CHARGIN_STATION:
        circle = plt.Circle((cs[0], cs[1]), 2, fill=False, color="cyan", linewidth=2.0, zorder=20, label="Charging area")
        ax.add_patch(circle)
    if WAYPOINTS:
        wx, wy = zip(*WAYPOINTS)
        ax.scatter(
            wx,
            wy,
            facecolors="green",
            edgecolors="green",
            linewidths=1.0,
            s=SCATTER_MARKER_SIZE*3,
            zorder=30,
            label="POI",
        )
    if HIDDEN_WAYPOINT:
        wx, wy = zip(*HIDDEN_WAYPOINT)
        ax.scatter(
            wx,
            wy,
            facecolors="midnightblue",
            edgecolors="midnightblue",
            linewidths=1.0,
            s=SCATTER_MARKER_SIZE*3,
            zorder=30,
            label="Hidden POI",
        )


    # ------------------------------------------------------------------
    # MEAN SPATIAL COVERAGE
    # ------------------------------------------------------------------

    coverage_result = aggregate_spatial_coverage(
        all_exp_data,
        resolution=0.2
    )

    if coverage_result is not None:

        x_cov, y_cov, mean_cov, var_cov = coverage_result

        X_cov, Y_cov = np.meshgrid(
            x_cov,
            y_cov
        )

        # --------------------------------------------------------------
        # Mean coverage
        #
        # 0   = never covered
        # 1   = covered in all experiments
        # --------------------------------------------------------------

        mean_mask = np.ma.masked_where(
            mean_cov <= 0,
            mean_cov
        )

        mean_plot = ax.pcolormesh(
            X_cov,
            Y_cov,
            mean_mask,
            cmap="Blues",
            shading="auto",
            alpha=0.30,
            vmin=0,
            vmax=1,
            zorder=1
        )

        # --------------------------------------------------------------
        # Coverage variance
        #
        # Higher values = stronger difference between experiments.
        #
        # Overlay it with a different colormap.
        # --------------------------------------------------------------

        variance_mask = np.ma.masked_where(
            var_cov <= 0.01,
            var_cov
        )

        ax.pcolormesh(
            X_cov,
            Y_cov,
            variance_mask,
            cmap="Oranges",
            shading="auto",
            alpha=0.25,
            vmin=0,
            vmax=0.25,
            zorder=2
        )

    # ------------------------------------------------------------------
    # MEAN UAV PATHS + SPATIAL STD
    # ------------------------------------------------------------------

    any_data = False

    for uav_id, (mean, var) in path_stats.items():

        if mean.shape[1] < 2:
            continue

        valid = np.isfinite(mean[:, 0]) & np.isfinite(mean[:, 1])

        if not np.any(valid):
            continue

        mean_x = mean[:, 0]
        mean_y = mean[:, 1]

        std_x = np.sqrt(var[:, 0])
        std_y = np.sqrt(var[:, 1])

        # --------------------------------------------------------------
        # Uncertainty area
        #
        # At every iteration create a rectangle:
        #
        # [mean_x ± std_x, mean_y ± std_y]
        #
        # and fill the union of these regions.
        # --------------------------------------------------------------

        from matplotlib.patches import Rectangle

        for i in range(len(mean_x) - 1):

            if not (
                np.isfinite(mean_x[i])
                and np.isfinite(mean_y[i])
                and np.isfinite(std_x[i])
                and np.isfinite(std_y[i])
            ):
                continue

            rect = Rectangle(
                (
                    mean_x[i] - std_x[i],
                    mean_y[i] - std_y[i]
                ),
                2 * std_x[i],
                2 * std_y[i],
                facecolor="red",
                edgecolor="none",
                alpha=0.025,
                zorder=5
            )

            ax.add_patch(rect)

        # --------------------------------------------------------------
        # Mean path
        # --------------------------------------------------------------

        ax.plot(
            mean_x[valid],
            mean_y[valid],
            linewidth=2.0,
            label=f"UAV {uav_id}",
            zorder=10
        )

        any_data = True

    if not any_data:
        plt.close(fig)
        return

    # ------------------------------------------------------------------
    # LEGEND HANDLES FOR COVERAGE
    # ------------------------------------------------------------------

    coverage_mean_handle = patches.Patch(
        facecolor="blue",
        alpha=0.30,
        label="Mean covered area"
    )

    coverage_variance_handle = patches.Patch(
        facecolor="orange",
        alpha=0.25,
        label="Coverage variance"
    )

    uncertainty_handle = patches.Patch(
        facecolor="red",
        alpha=0.15,
        label="Path ±1 std"
    )

    # ------------------------------------------------------------------
    # AXIS
    # ------------------------------------------------------------------

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    #ax.set_title(
    #     f"{setting_name} - Mean UAV paths, "
    #     f"coverage and variability"
    # )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )
    ax.set_xlim(WORLD_OUTER["xmin"] - 1, WORLD_OUTER["xmax"] + 1)
    ax.set_ylim(WORLD_OUTER["ymin"] - 1, WORLD_OUTER["ymax"] + 1)
    ax.grid(
        True,
        alpha=0.3
    )

    # Combine UAV handles with coverage handles
    handles, labels = ax.get_legend_handles_labels()

    handles.extend([
        coverage_mean_handle,
        coverage_variance_handle,
        uncertainty_handle
    ])

    ax.legend(
        handles=handles,
        loc="upper left", 
        ncol=2
    )

    savefig(
        out / "mean_paths.png"
    )

# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_setting(setting_folder: Path):
    
    print(f"\nProcessing setting: {setting_folder.name}")

    exp_folders = get_experiment_folders(setting_folder)

    if not exp_folders:
        print("  WARNING: no exp folders found.")
        return

    all_exp_data = []

    for exp_folder in exp_folders:
        print(f"  Processing {exp_folder.name}...")

        exp_data = build_exp_data(exp_folder)
        for uav_id, data in exp_data.items():
            exp_data[uav_id] = truncate_after_crash(
                data,
                CRASH_DISTANCE_THRESHOLD,
                uav_id,
                exp_folder
            )
        all_exp_data.append(exp_data)

        # UAV-specific plots go inside each 28/29/30/... folder.
        for uav_id, data in exp_data.items():
            make_individual_uav_plots(
                uav_id,
                data,
                exp_folder / uav_id / "plots",
                exp_folder.name,
            )

        # Swarm-level plots go inside exp X/plots.
        make_experiment_plots(exp_folder, exp_data)

    # Aggregated plots go directly in the setting folder.
    out = setting_folder / "plots_mean_5_experiments"
    out.mkdir(exist_ok=True)

    position_stats = aggregate_positions(all_exp_data)
    plot_aggregated_positions(position_stats, out, setting_folder.name)
    plot_aggregated_paths(position_stats, out, setting_folder.name, all_exp_data)

    priority_stats = aggregate_priorities(all_exp_data)
    plot_aggregated_priorities(priority_stats, out, setting_folder.name)

    voltage_stats = aggregate_scalar_per_uav(all_exp_data, "voltage")
    plot_aggregated_scalar_per_uav(
        voltage_stats, out, setting_folder.name,
        "voltage", "Voltage [V]", "mean_voltage"
    )

    photo_stats = aggregate_scalar_per_uav(all_exp_data, "photosynthesis")
    plot_aggregated_scalar_per_uav(
        photo_stats, out, setting_folder.name,
        "photosynthesis", "Photosynthetic production", "mean_photosynthesis"
    )

    distance_stats = aggregate_distances(all_exp_data)
    plot_aggregated_distances(distance_stats, out, setting_folder.name)
    plot_aggregated_leader_photosynthesis(exp_folders,out,setting_folder.name)

    print(f"  Aggregated plots saved to: {out}")
    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------

    generate_setting_report(setting_folder)


def main():
    parser = argparse.ArgumentParser(
        description="Generate UAV experiment plots and 5-experiment averages."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing the setting folders.",
    )
    parser.add_argument(
        "--settings",
        nargs="*",
        default=None,
        help="Optional setting folder names. By default all folders are discovered.",
    )

    args = parser.parse_args()

    root = args.root.expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    if args.settings:
        settings = [root / s for s in args.settings]
    else:
        # Prefer the expected setting names; otherwise use all root directories.
        expected = [root / s for s in DEFAULT_SETTINGS]
        settings = [p for p in expected if p.is_dir()]

        if not settings:
            settings = sorted([p for p in root.iterdir() if p.is_dir()])

    if not settings:
        raise SystemExit("No setting folders found.")

    print(f"Root: {root}")
    print(f"Settings: {[p.name for p in settings]}")
    print("Saving wolrd plot")
    plot_world(root)
    for setting in settings:
        process_setting(setting)

    print("\nDone.")


if __name__ == "__main__":
    main()