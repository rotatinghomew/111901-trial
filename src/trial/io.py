"""Readers for the synthetic CSV/JSON review materials.

Empty fields are treated as "not recorded".  In particular an empty date or
link end never means "closed" or "ended"; it means the material does not
record a boundary.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Iterable

from .models import (
    Allocation,
    DailyRecord,
    Link,
    Resident,
    ReviewPolicy,
    Site,
    Study,
)


def _text(value: str | None) -> str:
    return (value or "").strip()


def _date(value: str | None) -> date | None:
    text = _text(value)
    if not text:
        return None
    return date.fromisoformat(text)


def _bool(value: str | None) -> bool | None:
    text = _text(value).lower()
    if text in {"", "na", "null", "none"}:
        return None
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"cannot read boolean value: {value!r}")


def _int(value: str | None) -> int | None:
    text = _text(value)
    return int(text) if text else None


def _rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def read_residents(path: Path) -> dict[str, Resident]:
    residents: dict[str, Resident] = {}
    for row in _rows(path):
        resident = _text(row.get("resident"))
        if not resident:
            continue
        cluster = _text(row.get("cluster"))
        if not cluster:
            raise ValueError(f"resident {resident} has no cluster")
        residents[resident] = Resident(
            resident=resident,
            cluster=cluster,
            eligible=bool(_bool(row.get("eligible"))),
            stratum=_text(row.get("stratum")),
        )
    return residents


def read_allocations(path: Path) -> dict[str, Allocation]:
    allocations: dict[str, Allocation] = {}
    for row in _rows(path):
        cluster = _text(row.get("cluster"))
        if not cluster:
            continue
        open_day = _date(row.get("assigned_open_day"))
        if open_day is None:
            raise ValueError(f"cluster {cluster} has no assigned open day")
        allocations[cluster] = Allocation(
            cluster=cluster,
            wave=int(_text(row.get("wave"))),
            assigned_open_day=open_day,
            baseline_eligible=int(_text(row.get("baseline_eligible")) or 1),
        )
    return allocations


def read_sites(path: Path) -> dict[str, Site]:
    sites: dict[str, Site] = {}
    for row in _rows(path):
        site = _text(row.get("site"))
        if not site:
            continue
        open_day = _date(row.get("open_day"))
        if open_day is None:
            raise ValueError(f"site {site} has no open day")
        sites[site] = Site(
            site=site,
            cluster=_text(row.get("cluster")),
            open_day=open_day,
            close_day=_date(row.get("close_day")),
            seats=_int(row.get("seats")),
        )
    return sites


def read_links(path: Path) -> tuple[Link, ...]:
    links: list[Link] = []
    for row in _rows(path):
        source = _text(row.get("from_cluster"))
        target = _text(row.get("to_cluster"))
        valid_from = _date(row.get("valid_from"))
        if not source or not target or valid_from is None:
            continue
        links.append(
            Link(
                from_cluster=source,
                to_cluster=target,
                valid_from=valid_from,
                valid_to=_date(row.get("valid_to")),
            )
        )
    return tuple(links)


def read_daily(path: Path) -> tuple[DailyRecord, ...]:
    rows: list[DailyRecord] = []
    for row in _rows(path):
        resident = _text(row.get("resident"))
        day = _date(row.get("day"))
        if not resident or day is None:
            continue
        site = _text(row.get("site")) or None
        rows.append(
            DailyRecord(
                resident=resident,
                day=day,
                site=site,
                attendance=_bool(row.get("attendance")),
                report_complete=_bool(row.get("report_complete")),
            )
        )
    return tuple(sorted(rows, key=lambda item: (item.resident, item.day)))


def read_policy(path: Path) -> ReviewPolicy:
    return ReviewPolicy.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def load_study(data_dir: Path) -> Study:
    data_dir = Path(data_dir)
    return Study(
        residents=read_residents(data_dir / "residents.csv"),
        allocations=read_allocations(data_dir / "allocation.csv"),
        sites=read_sites(data_dir / "sites.csv"),
        links=read_links(data_dir / "links.csv"),
        daily=read_daily(data_dir / "daily.csv"),
        policy=read_policy(data_dir / "review.json"),
    )
