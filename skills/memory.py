"""Skill memory: SQLite for queries + an append-only JSONL journal of every write (the DB can be rebuilt from it).

Everything is keyed by setup_id (executor's SetupId: calibration version, robot profile + ranges, tool, table
layout). Registering a new setup in the same scope INVALIDATES the previous one: its skill versions are no longer
returned as incumbents, so a skill learned with one tissue / calibration is not silently reused with another.

Tables
  setups(setup_id, scope, components, created, invalidated, reason)
  skill_versions(version, skill, parent, setup_id, params, status, created_by, note, created, approved_by, approved_at)
      status: candidate | incumbent | retired | rejected
  experiments(exp_id, setup_id, skill, version, kind, initial, versions, change, outcome, timing, interventions, created)
  adaptations(id, skill, setup_id, from_version, to_version, accepted, record, created)
  promotions(id, skill, setup_id, candidate, incumbent, rule, evidence, decision, approved_by, created)
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

from .model import SkillVersion

SCHEMA = """
CREATE TABLE IF NOT EXISTS setups(setup_id TEXT PRIMARY KEY, scope TEXT, components TEXT, created REAL,
    invalidated REAL, reason TEXT);
CREATE TABLE IF NOT EXISTS skill_versions(version TEXT, skill TEXT, parent TEXT, setup_id TEXT, params TEXT,
    status TEXT, created_by TEXT, note TEXT, created REAL, approved_by TEXT, approved_at REAL,
    PRIMARY KEY(version, setup_id));
CREATE TABLE IF NOT EXISTS experiments(exp_id TEXT PRIMARY KEY, setup_id TEXT, skill TEXT, version TEXT, kind TEXT,
    initial TEXT, versions TEXT, change TEXT, outcome TEXT, timing TEXT, interventions TEXT, created REAL);
CREATE TABLE IF NOT EXISTS adaptations(id TEXT PRIMARY KEY, skill TEXT, setup_id TEXT, from_version TEXT,
    to_version TEXT, accepted INTEGER, record TEXT, created REAL);
CREATE TABLE IF NOT EXISTS promotions(id TEXT PRIMARY KEY, skill TEXT, setup_id TEXT, candidate TEXT, incumbent TEXT,
    rule TEXT, evidence TEXT, decision TEXT, approved_by TEXT, created REAL);
"""
JSON_COLS = {"components", "params", "initial", "versions", "outcome", "timing", "interventions", "record", "rule", "evidence"}


def code_version(repo: Path | str | None = None) -> str:
    """git HEAD (+ '-dirty' with uncommitted changes) of the repo holding this file."""
    cwd = str(repo or Path(__file__).resolve().parent)
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Memory:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal = self.root / "memory.jsonl"
        self.db = sqlite3.connect(self.root / "memory.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # low level ---------------------------------------------------------------------------------
    def _write(self, table: str, op: str, row: dict, where: dict | None = None, journal: bool = True):
        enc = {k: json.dumps(v, sort_keys=True) if k in JSON_COLS and v is not None else v for k, v in row.items()}
        if op == "insert":
            cols = ",".join(enc)
            self.db.execute(f"INSERT OR REPLACE INTO {table}({cols}) VALUES({','.join('?' * len(enc))})", list(enc.values()))
        else:
            sets = ",".join(f"{k}=?" for k in enc)
            conds = " AND ".join(f"{k}=?" for k in where)
            self.db.execute(f"UPDATE {table} SET {sets} WHERE {conds}", list(enc.values()) + list(where.values()))
        self.db.commit()
        if journal:
            with open(self.journal, "a") as f:
                f.write(json.dumps({"t": time.time(), "table": table, "op": op, "row": row, "where": where}, sort_keys=True) + "\n")

    def _rows(self, sql: str, args=()) -> list[dict]:
        out = []
        for r in self.db.execute(sql, args).fetchall():
            d = dict(r)
            for k in JSON_COLS & d.keys():
                if d[k] is not None:
                    d[k] = json.loads(d[k])
            out.append(d)
        return out

    @classmethod
    def rebuild(cls, journal: str | Path, root: str | Path) -> "Memory":
        """A fresh DB in `root` replayed from a journal (proves the journal holds everything)."""
        m = cls(root)
        for line in Path(journal).read_text().splitlines():
            e = json.loads(line)
            m._write(e["table"], e["op"], e["row"], e["where"], journal=False)
        return m

    # setups ------------------------------------------------------------------------------------
    def register_setup(self, setup_id: str, components: dict, scope: str = "default") -> dict:
        """Record the current setup. Any other valid setup in the same scope is invalidated (with what changed)."""
        known = self._rows("SELECT * FROM setups WHERE setup_id=?", (setup_id,))
        invalidated = []
        for s in self._rows("SELECT * FROM setups WHERE scope=? AND invalidated IS NULL AND setup_id!=?", (scope, setup_id)):
            changed = sorted(k for k in set(s["components"]) | set(components) if s["components"].get(k) != components.get(k))
            self.invalidate_setup(s["setup_id"], f"superseded by {setup_id}: changed {', '.join(changed) or 'id'}")
            invalidated.append(s["setup_id"])
        if not known:
            self._write("setups", "insert", {"setup_id": setup_id, "scope": scope, "components": components,
                                             "created": time.time(), "invalidated": None, "reason": None})
        elif known[0]["invalidated"] is not None:          # the old setup came back (e.g. the tissue was swapped back)
            self._write("setups", "update", {"invalidated": None, "reason": None}, {"setup_id": setup_id})
        return {"setup_id": setup_id, "new": not known, "invalidated": invalidated}

    def invalidate_setup(self, setup_id: str, reason: str):
        self._write("setups", "update", {"invalidated": time.time(), "reason": reason}, {"setup_id": setup_id})

    def setup_valid(self, setup_id: str) -> bool:
        r = self._rows("SELECT invalidated FROM setups WHERE setup_id=?", (setup_id,))
        return bool(r) and r[0]["invalidated"] is None

    def setup(self, setup_id: str) -> dict | None:
        r = self._rows("SELECT * FROM setups WHERE setup_id=?", (setup_id,))
        return r[0] if r else None

    # skill versions ----------------------------------------------------------------------------
    def add_version(self, v: SkillVersion, setup_id: str, status: str = "candidate"):
        assert status in ("candidate", "incumbent", "retired", "rejected")
        self._write("skill_versions", "insert", {"version": v.version, "skill": v.skill, "parent": v.parent,
                                                 "setup_id": setup_id, "params": v.params, "status": status,
                                                 "created_by": v.created_by, "note": v.note, "created": v.created,
                                                 "approved_by": None, "approved_at": None})

    def _sv(self, r: dict) -> SkillVersion:
        return SkillVersion(r["skill"], r["version"], r["parent"], r["params"], r["created_by"], r["note"] or "", r["created"])

    def version(self, version: str, setup_id: str | None = None) -> SkillVersion | None:
        q = "SELECT * FROM skill_versions WHERE version=?" + (" AND setup_id=?" if setup_id else "")
        r = self._rows(q, (version, setup_id) if setup_id else (version,))
        return self._sv(r[0]) if r else None

    def status(self, version: str, setup_id: str) -> str | None:
        r = self._rows("SELECT status FROM skill_versions WHERE version=? AND setup_id=?", (version, setup_id))
        return r[0]["status"] if r else None

    def versions(self, skill: str, setup_id: str, status: str | None = None) -> list[SkillVersion]:
        q, a = "SELECT * FROM skill_versions WHERE skill=? AND setup_id=?", [skill, setup_id]
        if status:
            q, a = q + " AND status=?", a + [status]
        return [self._sv(r) for r in self._rows(q + " ORDER BY created", a)]

    def incumbent(self, skill: str, setup_id: str) -> SkillVersion | None:
        """The current incumbent for this setup, or None if there is none OR the setup was invalidated."""
        if not self.setup_valid(setup_id):
            return None
        v = self.versions(skill, setup_id, "incumbent")
        return v[-1] if v else None

    def set_status(self, version: str, setup_id: str, status: str):
        self._write("skill_versions", "update", {"status": status}, {"version": version, "setup_id": setup_id})

    def promote(self, version: str, setup_id: str, approved_by: str) -> SkillVersion:
        """Make `version` the incumbent. Needs a named approver (the human approval step for hardware)."""
        if not approved_by or not approved_by.strip():
            raise PermissionError("promotion needs an explicit approver")
        if not self.setup_valid(setup_id):
            raise ValueError(f"setup {setup_id} is invalidated: re-validate the skill on the current setup")
        v = self.version(version, setup_id)
        if v is None:
            raise KeyError(version)
        for old in self.versions(v.skill, setup_id, "incumbent"):
            self.set_status(old.version, setup_id, "retired")
        self._write("skill_versions", "update", {"status": "incumbent", "approved_by": approved_by, "approved_at": time.time()},
                    {"version": version, "setup_id": setup_id})
        return v

    def lineage(self, version: str) -> list[str]:
        out, v = [], self.version(version)
        while v is not None and v.version not in out:
            out.append(v.version)
            v = self.version(v.parent) if v.parent else None
        return out

    # experiments, adaptations, promotions -------------------------------------------------------------
    def record_experiment(self, *, setup_id: str, skill: str, version: str, kind: str, initial: dict, versions: dict,
                          change: str, outcome: dict, timing: dict, interventions: list, exp_id: str | None = None) -> str:
        exp_id = exp_id or "exp-" + uuid.uuid4().hex[:10]
        self._write("experiments", "insert", {"exp_id": exp_id, "setup_id": setup_id, "skill": skill, "version": version,
                                              "kind": kind, "initial": initial, "versions": versions, "change": change,
                                              "outcome": outcome, "timing": timing, "interventions": interventions,
                                              "created": time.time()})
        return exp_id

    def experiments(self, skill: str | None = None, setup_id: str | None = None) -> list[dict]:
        q, a = "SELECT * FROM experiments WHERE 1=1", []
        if skill:
            q, a = q + " AND skill=?", a + [skill]
        if setup_id:
            q, a = q + " AND setup_id=?", a + [setup_id]
        return self._rows(q + " ORDER BY created", a)

    def record_adaptation(self, skill: str, setup_id: str, record: dict) -> str:
        rid = "adapt-" + uuid.uuid4().hex[:10]
        self._write("adaptations", "insert", {"id": rid, "skill": skill, "setup_id": setup_id,
                                              "from_version": record.get("from_version"), "to_version": record.get("to_version"),
                                              "accepted": int(bool(record.get("accepted"))), "record": record, "created": time.time()})
        return rid

    def adaptations(self, skill: str, setup_id: str) -> list[dict]:
        return self._rows("SELECT * FROM adaptations WHERE skill=? AND setup_id=? ORDER BY created", (skill, setup_id))

    def record_promotion(self, *, skill: str, setup_id: str, candidate: str, incumbent: str | None, rule: dict,
                         evidence: dict, decision: str, approved_by: str | None = None) -> str:
        pid = "promo-" + uuid.uuid4().hex[:10]
        self._write("promotions", "insert", {"id": pid, "skill": skill, "setup_id": setup_id, "candidate": candidate,
                                             "incumbent": incumbent, "rule": rule, "evidence": evidence,
                                             "decision": decision, "approved_by": approved_by, "created": time.time()})
        return pid

    def promotions(self, skill: str | None = None) -> list[dict]:
        if skill:
            return self._rows("SELECT * FROM promotions WHERE skill=? ORDER BY created", (skill,))
        return self._rows("SELECT * FROM promotions ORDER BY created")
