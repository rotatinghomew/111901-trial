"""Core data model for the staged community fitness programme review.

The design is a cluster (service-area) randomised stepped rollout.  Every
analysis in this package keeps the wave/cluster allocation fixed: where a
resident actually attends, and whether a resident attends at all, never moves
that resident into another comparison group.

All dates are local calendar dates.  A site is open on both ``open_day`` and
``close_day``; a link is usable on both ``valid_from`` and ``valid_to``.  An
empty ``close_day`` or ``valid_to`` means "not recorded", never "closed".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Mapping

CONTINUE = "continue"
EXTEND_OBSERVATION = "extend_observation"
NEED_MORE_EVIDENCE = "need_more_evidence"

DECISION_LABELS = {
    CONTINUE: "可继续",
    EXTEND_OBSERVATION: "延长观察",
    NEED_MORE_EVIDENCE: "待补证",
}


@dataclass(frozen=True)
class Resident:
    resident: str
    cluster: str
    eligible: bool
    stratum: str

    @property
    def baseline_eligible(self) -> bool:
        return self.eligible


@dataclass(frozen=True)
class Allocation:
    cluster: str
    wave: int
    assigned_open_day: date
    baseline_eligible: int = 1


@dataclass(frozen=True)
class Site:
    site: str
    cluster: str
    open_day: date
    close_day: date | None = None
    seats: int | None = None

    def is_open(self, day: date) -> bool:
        if day < self.open_day:
            return False
        return self.close_day is None or day <= self.close_day


@dataclass(frozen=True)
class Link:
    from_cluster: str
    to_cluster: str
    valid_from: date
    valid_to: date | None = None

    def is_valid(self, day: date) -> bool:
        if day < self.valid_from:
            return False
        return self.valid_to is None or day <= self.valid_to

    def connects(self, cluster: str) -> str | None:
        if cluster == self.from_cluster:
            return self.to_cluster
        if cluster == self.to_cluster:
            return self.from_cluster
        return None


@dataclass(frozen=True)
class DailyRecord:
    resident: str
    day: date
    site: str | None
    attendance: bool | None
    report_complete: bool | None


@dataclass(frozen=True)
class ReviewPolicy:
    minimum_gain: float
    max_spillover: float
    max_missing: float
    required_strata: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ReviewPolicy":
        return cls(
            minimum_gain=float(value["minimum_gain"]),
            max_spillover=float(value["max_spillover"]),
            max_missing=float(value["max_missing"]),
            required_strata=tuple(str(item) for item in value["required_strata"]),
        )


@dataclass(frozen=True)
class ReportCorrection:
    """A supplementary or corrected daily report."""

    resident: str
    day: date
    site: str | None
    attendance: bool | None
    report_complete: bool | None
    reason: str = ""


@dataclass(frozen=True)
class SitePeriodCorrection:
    """Correction to the formally registered opening period of a site."""

    site: str
    open_day: date | None = None
    close_day: date | None = None
    reason: str = ""


@dataclass(frozen=True)
class BorrowedVenueFact:
    """A resident-day confirmed to have used a temporary/borrowed venue.

    This is kept apart from "no service": a borrowed venue is a service
    arrangement, while an unreported day carries no service information.
    """

    resident: str
    day: date
    reason: str = ""


@dataclass(frozen=True)
class CorrectionSet:
    reports: tuple[ReportCorrection, ...] = ()
    site_periods: tuple[SitePeriodCorrection, ...] = ()
    borrowed_venues: tuple[BorrowedVenueFact, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.reports or self.site_periods or self.borrowed_venues)


@dataclass
class Study:
    residents: dict[str, Resident]
    allocations: dict[str, Allocation]
    sites: dict[str, Site]
    links: tuple[Link, ...]
    daily: tuple[DailyRecord, ...]
    policy: ReviewPolicy
    borrowed_days: frozenset[tuple[str, date]] = frozenset()
    applied_corrections: tuple[str, ...] = ()
    daily_index: dict[tuple[str, date], DailyRecord] = field(
        default_factory=dict, repr=False
    )
    daily_by_resident: dict[str, tuple[DailyRecord, ...]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        index: dict[tuple[str, date], DailyRecord] = {}
        for record in self.daily:
            key = (record.resident, record.day)
            if key in index:
                raise ValueError(f"duplicate daily report: {record.resident} {record.day}")
            index[key] = record
        self.daily_index = index
        grouped: dict[str, list[DailyRecord]] = {}
        for record in self.daily:
            grouped.setdefault(record.resident, []).append(record)
        self.daily_by_resident = {
            resident: tuple(sorted(rows, key=lambda row: row.day))
            for resident, rows in grouped.items()
        }

    # ------------------------------------------------------------------
    # cluster / wave helpers
    # ------------------------------------------------------------------
    @property
    def waves(self) -> tuple[int, ...]:
        return tuple(sorted({item.wave for item in self.allocations.values()}))

    def clusters_in_wave(self, wave: int) -> tuple[str, ...]:
        return tuple(
            sorted(
                cluster
                for cluster, item in self.allocations.items()
                if item.wave == wave
            )
        )

    def assigned_open_day(self, wave: int) -> date:
        days = {
            item.assigned_open_day
            for item in self.allocations.values()
            if item.wave == wave
        }
        if len(days) != 1:
            raise ValueError(f"wave {wave} does not have one assigned open day: {days}")
        return next(iter(days))

    def observation_window(self, wave: int, follow_up_days: int) -> tuple[date, date]:
        start = self.assigned_open_day(wave)
        if follow_up_days < 1:
            raise ValueError("follow_up_days must be at least 1")
        return start, start + timedelta(days=follow_up_days - 1)

    def residents_in_cluster(self, cluster: str) -> tuple[Resident, ...]:
        return tuple(
            sorted(
                (item for item in self.residents.values() if item.cluster == cluster),
                key=lambda item: item.resident,
            )
        )

    def eligible_residents_in_cluster(self, cluster: str) -> tuple[Resident, ...]:
        return tuple(item for item in self.residents_in_cluster(cluster) if item.eligible)

    def clusters_with_open_site(self, day: date) -> tuple[str, ...]:
        return tuple(
            sorted({site.cluster for site in self.sites.values() if site.is_open(day)})
        )

    def linked_clusters(self, cluster: str, day: date) -> tuple[str, ...]:
        linked: set[str] = set()
        for link in self.links:
            if not link.is_valid(day):
                continue
            other = link.connects(cluster)
            if other is not None:
                linked.add(other)
        return tuple(sorted(linked))

    def has_own_service(self, cluster: str, day: date) -> bool:
        return any(
            site.cluster == cluster and site.is_open(day) for site in self.sites.values()
        )

    def has_neighbour_service(self, cluster: str, day: date) -> bool:
        neighbours = set(self.linked_clusters(cluster, day))
        return any(
            site.cluster in neighbours and site.is_open(day) for site in self.sites.values()
        )

    def site_is_formally_open(self, site_name: str | None, day: date) -> bool:
        if site_name is None:
            return False
        site = self.sites.get(site_name)
        return bool(site and site.is_open(day))

    def site_cluster(self, site_name: str | None) -> str | None:
        if site_name is None:
            return None
        site = self.sites.get(site_name)
        return site.cluster if site else None

    def record_for(self, resident: str, day: date) -> DailyRecord | None:
        return self.daily_index.get((resident, day))

    # ------------------------------------------------------------------
    # corrections
    # ------------------------------------------------------------------
    def with_corrections(self, corrections: CorrectionSet) -> "Study":
        """Return a corrected copy without touching allocation or grouping.

        Corrections may add/correct daily reports, correct a site's formal
        opening period, or record that a resident-day used a temporary venue.
        No correction path can change resident cluster, eligibility or wave
        allocation, so the randomised groups stay exactly as drawn.  A later
        correction therefore changes crosstalk, missing reporting, evidence
        bounds and the recommendation -- never the group assignment.
        """

        if corrections.empty:
            return Study(
                residents=dict(self.residents),
                allocations=dict(self.allocations),
                sites=dict(self.sites),
                links=tuple(self.links),
                daily=tuple(self.daily),
                policy=self.policy,
                borrowed_days=self.borrowed_days,
                applied_corrections=self.applied_corrections,
            )

        sites = dict(self.sites)
        notes: list[str] = []
        for item in corrections.site_periods:
            site = sites.get(item.site)
            if site is None:
                raise ValueError(f"unknown site in correction: {item.site}")
            new_open = item.open_day if item.open_day is not None else site.open_day
            new_close = item.close_day if item.close_day is not None else site.close_day
            if new_close is not None and new_close < new_open:
                raise ValueError(f"corrected site period is inverted: {item.site}")
            sites[item.site] = replace(site, open_day=new_open, close_day=new_close)
            notes.append(
                f"site_period:{item.site}:{new_open.isoformat()}:"
                f"{new_close.isoformat() if new_close else ''}:{item.reason}"
            )

        by_key: dict[tuple[str, date], DailyRecord] = {
            (record.resident, record.day): record for record in self.daily
        }
        for item in corrections.reports:
            if item.resident not in self.residents:
                raise ValueError(f"unknown resident in correction: {item.resident}")
            if item.site is not None and item.site not in sites:
                raise ValueError(f"unknown site in correction: {item.site}")
            by_key[(item.resident, item.day)] = DailyRecord(
                resident=item.resident,
                day=item.day,
                site=item.site,
                attendance=item.attendance,
                report_complete=item.report_complete,
            )
            notes.append(f"report:{item.resident}:{item.day.isoformat()}:{item.reason}")

        borrowed = set(self.borrowed_days)
        for item in corrections.borrowed_venues:
            if item.resident not in self.residents:
                raise ValueError(f"unknown resident in correction: {item.resident}")
            borrowed.add((item.resident, item.day))
            notes.append(
                f"borrowed_venue:{item.resident}:{item.day.isoformat()}:{item.reason}"
            )

        daily = tuple(sorted(by_key.values(), key=lambda row: (row.resident, row.day)))
        return Study(
            residents=dict(self.residents),
            allocations=dict(self.allocations),
            sites=sites,
            links=tuple(self.links),
            daily=daily,
            policy=self.policy,
            borrowed_days=frozenset(borrowed),
            applied_corrections=self.applied_corrections + tuple(notes),
        )


def default_follow_up_days(study: Study, fallback: int = 14) -> int:
    """Use the shortest planned gap between waves as the default window.

    The materials do not carry a separate follow-up setting.  The planned wave
    spacing is the only design-based value that keeps one wave's observation
    window from running into the next wave's opening.  Callers can override it.
    """

    days = sorted({item.assigned_open_day for item in study.allocations.values()})
    gaps = [(right - left).days for left, right in zip(days, days[1:]) if right > left]
    return min(gaps) if gaps else fallback
