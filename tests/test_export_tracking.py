import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
exporter_spec = importlib.util.spec_from_file_location(
    "aef_exporter", SCRIPTS_DIR / "02_export_aef_medoids.py"
)
exporter = importlib.util.module_from_spec(exporter_spec)
exporter_spec.loader.exec_module(exporter)


def test_event_selection_preserves_filtered_csv_order():
    events = pd.DataFrame(
        {
            "year": [2020, 2019, 2020, 2021],
            "fire": ["first", "older", "second", "last"],
        }
    )

    selected = exporter.select_events(events, years=[2020, 2019], limit=2)

    assert selected["fire"].tolist() == ["first", "older"]


def test_task_state_is_atomically_persisted_and_loaded(tmp_path):
    path = tmp_path / "nested" / "state.json"
    state = {
        "version": 1,
        "events": {
            "2020__fire": {
                "status": "SUBMITTED",
                "task_id": "task-123",
                "grid": {"crs": "EPSG:32610"},
            }
        },
    }

    exporter.save_state(path, state)

    assert exporter.load_state(path) == state
    assert list(path.parent.iterdir()) == [path]


def test_refresh_recovers_submission_id_from_earth_engine_description():
    class Task:
        id = "task-123"

        def status(self):
            return {
                "id": self.id,
                "description": "2020__fire",
                "state": "RUNNING",
            }

    class FakeEE:
        batch = type("FakeBatch", (), {"Task": type("Tasks", (), {"list": staticmethod(lambda: [Task()])})})

    state = {
        "version": 1,
        "events": {
            "2020__fire": {
                "name": "2020__fire",
                "status": "SUBMITTING",
                "grid": {"crs": "EPSG:32610"},
            }
        },
    }

    exporter.refresh_task_states(FakeEE, state)

    assert state["events"]["2020__fire"]["task_id"] == "task-123"
    assert state["events"]["2020__fire"]["status"] == "RUNNING"


def test_failed_task_is_only_resubmitted_when_requested():
    event = {"status": "FAILED", "task_id": "task-123"}

    assert exporter._skip_existing(event, retry_failed=False)
    assert not exporter._skip_existing(event, retry_failed=True)


def test_main_submits_csv_order_and_skips_completed_tasks(tmp_path, monkeypatch):
    index = tmp_path / "events.csv"
    pd.DataFrame(
        [
            [2020, "first", True],
            [2019, "older", True],
            [2020, "third", True],
        ],
        columns=["year", "fire", "consistent"],
    ).assign(
        crs="EPSG:32610",
        a=375,
        b=0,
        c=500000,
        d=0,
        e=-375,
        f=5500000,
        width=2,
        height=2,
    ).to_csv(index, index=False)
    tasks = []
    submitted = []

    class RemoteTask:
        def __init__(self, task_id, description):
            self.id = task_id
            self.description = description
            self.state = "READY"

        def status(self):
            return {"id": self.id, "description": self.description, "state": self.state}

    class TaskAPI:
        @staticmethod
        def list():
            return tasks

    class FakeEE:
        batch = type("FakeBatch", (), {"Task": TaskAPI})

        @staticmethod
        def Initialize(project):
            assert project == "test-project"

    def fake_export(_ee, _image, _grid, name, _folder):
        submitted.append(name)
        task = RemoteTask(f"task-{len(submitted)}", name)
        tasks.append(task)
        return SimpleNamespace(id=task.id)

    monkeypatch.setitem(sys.modules, "ee", FakeEE)
    monkeypatch.setattr(exporter, "build_medoid_image", lambda *_args: (None, object()))
    monkeypatch.setattr(exporter, "validate_medoid_image", lambda *_args: None)
    monkeypatch.setattr(exporter, "export_to_drive", fake_export)

    args = [
        "--project", "test-project",
        "--index", str(index),
        "--years", "2020", "2019",
        "--limit", "2",
        "--sleep", "0",
        "--log-file", str(tmp_path / "export.log"),
        "--state-file", str(tmp_path / "state.json"),
    ]
    exporter.main(args)

    assert submitted == ["2020__first", "2019__older"]
    for task in tasks:
        task.state = "COMPLETED"
    exporter.main(args)

    assert submitted == ["2020__first", "2019__older"]
    state = json.loads((tmp_path / "state.json").read_text())
    assert {event["status"] for event in state["events"].values()} == {"COMPLETED"}