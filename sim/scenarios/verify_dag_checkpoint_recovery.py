"""Acceptance gate: durable DAG checkpoint/resume with no completed-stage recompute."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness, _kill_tree  # noqa: E402


CP_PORT = 8340  # distinct from verify_scale.py (8320); a shared port lets one
                # gate's lingering socket be mistaken for this gate's server.
DATA_DIR = str(Path(__file__).resolve().parent / "_data_dag_checkpoint_p4_batch_dag")


def _post_dag(cp: str) -> str:
    body = {
        "name": "checkpoint-map-reduce",
        "stages": [
            {
                "id": "map",
                "func": "sleep_echo",
                "payloads": [{"seconds": 0.2, "echo": i} for i in range(4)],
                "cpu_pct": 10,
                "ram_mb": 64,
            },
            {
                "id": "reduce",
                "func": "sleep_echo",
                "depends_on": ["map"],
                "reduce": True,
                "payloads": [{"seconds": 0.2, "echo": "reduced"}],
                "cpu_pct": 10,
                "ram_mb": 64,
            },
        ],
    }
    r = httpx.post(f"{cp}/api/v1/dags", json=body, timeout=10)
    r.raise_for_status()
    return r.json()["dag_id"]


def _dag(cp: str, dag_id: str) -> dict:
    r = httpx.get(f"{cp}/api/v1/dags/{dag_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def _stage(dag: dict, stage_id: str) -> dict:
    return next(s for s in dag["stages"] if s["id"] == stage_id)


def _wait_stage(cp: str, dag_id: str, stage_id: str, status: str,
                timeout: float = 45.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = _dag(cp, dag_id)
        if _stage(d, stage_id)["status"] == status:
            return d
        time.sleep(0.3)
    raise TimeoutError(f"stage {stage_id} did not reach {status}: {_dag(cp, dag_id)}")


def _wait_done(cp: str, dag_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = _dag(cp, dag_id)
        if str(d["status"]).startswith("completed"):
            return d
        time.sleep(0.4)
    raise TimeoutError(f"DAG did not finish: {_dag(cp, dag_id)}")


def _corrupt_map_checkpoint(dag_id: str) -> None:
    db_path = str(Path(DATA_DIR) / "meshpool.db")
    last = None
    for _ in range(20):
        try:
            conn = sqlite3.connect(db_path, timeout=10)
            try:
                row = conn.execute("SELECT data FROM dags WHERE id=?", (dag_id,)).fetchone()
                assert row, f"DAG {dag_id} missing from DB"
                data = json.loads(row[0])
                for stage in data["stages"]:
                    if stage["id"] == "map":
                        stage["status"] = "pending"
                        stage["job_id"] = None
                        stage["completed_ms"] = None
                data["status"] = "running"
                conn.execute("UPDATE dags SET status=?, data=? WHERE id=?",
                             ("running", json.dumps(data), dag_id))
                conn.commit()
                return
            finally:
                conn.close()
        except sqlite3.Error as exc:
            last = exc
            time.sleep(0.25)
    raise RuntimeError(f"could not corrupt checkpoint for non-vacuity run: {last}")


def main() -> int:
    h = Harness(n_agents=3, cp_port=CP_PORT, data_dir=DATA_DIR)
    ok = True
    try:
        print("== starting cluster for DAG checkpoint gate ==")
        h.start()
        cp = h.cp_url()
        nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        print(f"  control plane: {cp}")
        print(f"  nodes online: {len(nodes)}")
        assert len(nodes) >= 3, "non-vacuity: expected 3 real agent processes"

        print("\n== submit map-reduce DAG ==")
        dag_id = _post_dag(cp)
        print(f"  dag_id={dag_id}")

        print("\n== wait for map stage checkpoint ==")
        d = _wait_stage(cp, dag_id, "map", "succeeded")
        map_stage = _stage(d, "map")
        reduce_stage = _stage(d, "reduce")
        print(f"  map status={map_stage['status']} execution_count={map_stage['execution_count']}")
        print(f"  reduce status={reduce_stage['status']} execution_count={reduce_stage['execution_count']}")
        assert map_stage["execution_count"] == 4, (
            "non-vacuity: map execution counter must equal its 4 fan-out units before restart"
        )
        assert len(map_stage["results"]) == 4, "map did not produce four checkpointed results"

        print("\n== hard-kill control plane after completed map checkpoint ==")
        _kill_tree(h.cp_proc)
        h.cp_proc = None
        print("  control plane killed")

        if os.environ.get("MESHPOOL_DAG_CORRUPT_MAP_CHECKPOINT") == "1":
            print("  intentionally corrupting map checkpoint to prove the gate catches recompute")
            _corrupt_map_checkpoint(dag_id)

        print("\n== restart control plane and resume DAG ==")
        h.start_control_plane()
        final = _wait_done(cp, dag_id)
        final_map = _stage(final, "map")
        final_reduce = _stage(final, "reduce")
        print(f"  final status={final['status']}")
        print(f"  final map execution_count={final_map['execution_count']}")
        print(f"  final reduce execution_count={final_reduce['execution_count']}")
        assert final["status"] == "completed", final["status"]
        assert final_map["execution_count"] == 4, (
            f"completed map stage was recomputed: expected 4 executions, "
            f"got {final_map['execution_count']}"
        )
        assert final_reduce["execution_count"] >= 1, (
            "non-vacuity: reducer never executed, so the DAG did not really resume"
        )
        assert final_reduce["status"] == "succeeded", final_reduce
        print("\nALL DAG CHECKPOINT RECOVERY CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        h.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
