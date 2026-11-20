"""
Export one 64-band AlphaEarth cosine-medoid GeoTIFF per WSTS fire, on that fire's exact grid.
Uses the AEF layer of the year BEFORE the fire's year (no leakage). Earth Engine task state is
persisted locally so interrupted runs can resume without resubmitting known tasks.

    python scripts/02_export_aef_medoids.py --project YOUR_PROJECT --index data/events_index.csv --limit 1 --dry-run
    python scripts/02_export_aef_medoids.py --project YOUR_PROJECT --index data/events_index.csv
    python scripts/02_export_aef_medoids.py --project YOUR_PROJECT --index data/events_index.csv --years 2018 2019
"""
import argparse
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time

import pandas as pd

from aef_utils import build_medoid_image, export_to_drive, validate_grid, validate_medoid_image


ACTIVE_TASK_STATES = {"READY", "RUNNING", "CANCEL_REQUESTED", "CANCELING"}
FAILED_TASK_STATES = {"FAILED", "CANCELLED"}


def select_events(df, years=None, fire=None, limit=None):
    """Apply filters without sorting; selected events retain their CSV order."""
    selected = df
    if years:
        selected = selected.loc[selected["year"].isin(years)]
    if fire:
        selected = selected.loc[selected["fire"] == fire]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer")
        selected = selected.head(limit)
    return selected


def configure_logger(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("aef_export")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_state(path):
    path = Path(path)
    if not path.exists():
        return {"version": 1, "events": {}}
    with path.open(encoding="utf-8") as state_file:
        state = json.load(state_file)
    if (not isinstance(state, dict) or state.get("version") != 1
            or not isinstance(state.get("events"), dict)):
        raise ValueError(f"Unsupported or malformed task state file: {path}")
    return state


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as state_file:
            temp_path = Path(state_file.name)
            json.dump(state, state_file, indent=2, sort_keys=True)
            state_file.write("\n")
            state_file.flush()
            os.fsync(state_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _task_snapshot(ee):
    by_id = {}
    by_description = {}
    for task in ee.batch.Task.list():
        status = task.status()
        task_id = status.get("id") or getattr(task, "id", None)
        description = status.get("description")
        if task_id:
            by_id[task_id] = status
        if description:
            by_description.setdefault(description, []).append(status)
    return by_id, by_description


def refresh_task_states(ee, state):
    """Refresh journaled task states and recover a submission interrupted before its ID was saved."""
    if not state["events"]:
        return
    by_id, by_description = _task_snapshot(ee)
    for name, event in state["events"].items():
        status = event.get("status")
        task_id = event.get("task_id")
        remote = by_id.get(task_id) if task_id else None
        if remote is None and not task_id and status in {"BUILDING", "SUBMITTING"}:
            matches = by_description.get(name, [])
            if len(matches) > 1:
                raise RuntimeError(
                    f"Cannot safely recover {name}: multiple Earth Engine tasks share its description"
                )
            if matches:
                remote = matches[0]
                event["task_id"] = remote.get("id")
        if remote is not None:
            event["status"] = remote.get("state", "UNKNOWN")
            event["error"] = remote.get("error_message")
            event["updated_at"] = datetime.now(timezone.utc).isoformat()


def _grid_from_row(row):
    grid = {
        "crs": str(row.crs),
        "transform": [float(getattr(row, coefficient)) for coefficient in "abcdef"],
        "width": int(row.width),
        "height": int(row.height),
    }
    validate_grid(grid)
    return grid


def _skip_existing(event, retry_failed):
    status = event.get("status")
    if status in FAILED_TASK_STATES:
        return not retry_failed
    if event.get("task_id") or status in ACTIVE_TASK_STATES | {"COMPLETED", "SUBMITTED"}:
        return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Earth Engine Cloud project ID")
    ap.add_argument("--index", default="data/events_index.csv")
    ap.add_argument("--folder", default="wsts_aef", help="Google Drive folder for the exports")
    ap.add_argument("--years", type=int, nargs="*", help="only these fire years")
    ap.add_argument("--fire", help="only this fire name")
    ap.add_argument("--limit", type=int, help="export at most N fires")
    ap.add_argument("--dry-run", action="store_true", help="validate images but do not start tasks")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between task submissions")
    ap.add_argument("--log-file", default="data/aef_export.log", help="append-only run log")
    ap.add_argument("--state-file", default="data/aef_export_state.json",
                    help="atomic per-fire task journal used to resume interrupted runs")
    ap.add_argument("--retry-failed", action="store_true",
                    help="resubmit events whose previous Earth Engine task failed or was cancelled")
    args = ap.parse_args(argv)
    if not math.isfinite(args.sleep) or args.sleep < 0:
        ap.error("--sleep must be a finite non-negative number")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be a positive integer")

    logger = configure_logger(args.log_file)
    logger.info("Starting AEF export; index=%s project=%s", args.index, args.project)
    import ee

    ee.Initialize(project=args.project)
    df = select_events(pd.read_csv(args.index), args.years, args.fire, args.limit)
    logger.info("%d fires selected in CSV order", len(df))

    state = load_state(args.state_file) if not args.dry_run else {"version": 1, "events": {}}
    if not args.dry_run:
        refresh_task_states(ee, state)
        save_state(args.state_file, state)

    submitted = 0
    for position, row in enumerate(df.itertuples(index=False), start=1):
        name = f"{int(row.year)}__{row.fire}"
        if pd.isna(row.consistent) or not bool(row.consistent):
            logger.warning("[%d/%d] skip %s: source days do not share one grid",
                           position, len(df), name)
            continue

        try:
            grid = _grid_from_row(row)
        except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
            logger.exception("[%d/%d] invalid grid for %s", position, len(df), name)
            raise
        event = state["events"].get(name)
        if event is not None and event.get("grid") != grid:
            logger.error("Grid for %s differs from the saved task journal", name)
            raise ValueError(
                f"Grid for {name} differs from the saved task journal; use a new --state-file "
                "to start a separate export run"
            )
        if event is not None and _skip_existing(event, args.retry_failed):
            logger.info("[%d/%d] skip %s: task status=%s task_id=%s",
                        position, len(df), name, event.get("status"), event.get("task_id"))
            continue

        aef_year = int(row.year) - 1
        if not args.dry_run:
            event = event or {"grid": grid, "name": name}
            event.pop("task_id", None)
            event.update(status="BUILDING", updated_at=datetime.now(timezone.utc).isoformat())
            state["events"][name] = event
            save_state(args.state_file, state)
        logger.info("[%d/%d] checking %s (AEF %d)", position, len(df), name, aef_year)

        try:
            _, coarse = build_medoid_image(ee, grid, aef_year)
            validate_medoid_image(ee, coarse, grid)
            if args.dry_run:
                logger.info("[dry-run] %s passed band, grid, and unit-vector checks", name)
                continue

            if submitted and args.sleep:
                time.sleep(args.sleep)
            event.update(status="SUBMITTING", updated_at=datetime.now(timezone.utc).isoformat())
            save_state(args.state_file, state)
            task = export_to_drive(ee, coarse, grid, name, args.folder)
            if not getattr(task, "id", None):
                raise RuntimeError(f"Earth Engine did not return a task ID for {name}")
            event.update(
                status="SUBMITTED",
                task_id=task.id,
                updated_at=datetime.now(timezone.utc).isoformat(),
                error=None,
            )
            save_state(args.state_file, state)
            submitted += 1
            logger.info("[%d/%d] submitted %s (AEF %d) task_id=%s",
                        position, len(df), name, aef_year, task.id)
        except Exception as error:
            if not args.dry_run:
                event.update(
                    status="SUBMITTING" if event.get("status") == "SUBMITTING" else "FAILED_LOCAL",
                    error=str(error),
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                save_state(args.state_file, state)
            logger.exception("[%d/%d] failed processing %s", position, len(df), name)
            raise

    logger.info("Run complete: %d fires selected, %d tasks submitted", len(df), submitted)


if __name__ == "__main__":
    main()
