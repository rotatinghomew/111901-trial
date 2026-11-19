"""Estimation for the staged rollout review.

Design choices behind every number in this module:

* Groups are fixed at randomisation.  A resident belongs to the wave of the
  cluster recorded in ``residents.csv``; attending elsewhere, or not
  attending, never moves the resident between arms.
* The primary comparison uses the resident list and assigned start time fixed
  at each wave's lottery.  A resident contributes at most once to the
  denominator.
* An attendance event is *valid evidence* only when the report is complete,
  attendance is reported, AND the recorded site is formally open that day (or
  a confirmed temporary-venue fact covers that resident-day).  The looser
  ``RECORDED`` rule is kept as a sensitivity analysis only.
* A resident without valid attendance is a known non-attender only when every
  day of the window carries a complete report showing non-attendance.
  Otherwise the resident is unknown.  Each estimate therefore carries a lower
  bound (unknown counted as non-attendance) and an upper bound (unknown
  counted as attendance).
* Interval estimates use the service area (cluster) as the unit of
  uncertainty.  Residents are never treated as independent draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt
from typing import Sequence

from .models import Resident, Study

STRICT = "strict"
RECORDED = "recorded"

ATTENDED = "attended"
NOT_ATTENDED = "not_attended"
UNKNOWN = "unknown"

# Two-sided 95% Student-t critical values.  For df above the table we fall back
# to the nearest lower tabulated df, widending the interval.  This keeps the
# package free of third-party dependencies.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    40: 2.021, 60: 2.000, 120: 1.980,
}


def critical_value(df: int, level: float = 0.95) -> float:
    if level != 0.95:
        raise ValueError("only the 95% interval is tabulated")
    if df < 1:
        raise ValueError("degrees of freedom must be positive")
    if df in _T95:
        return _T95[df]
    eligible = [value for value in _T95 if value <= df]
    return _T95[max(eligible)] if eligible else 1.96


@dataclass(frozen=True)
class ResidentOutcome:
    resident: str
    cluster: str
    stratum: str
    status: str
    attendance_days: tuple[date, ...] = ()
    observed_site_days: tuple[tuple[date, str | None], ...] = ()
    borrowed_venue_days: tuple[date, ...] = ()
    unreported_days: int = 0
    unknown_reason: str | None = None

    @property
    def attended(self) -> bool:
        return self.status == ATTENDED

    @property
    def lower_indicator(self) -> int:
        return 1 if self.status == ATTENDED else 0

    @property
    def upper_indicator(self) -> int:
        # An unknown resident may have attended; a known non-attender may not.
        return 1 if self.status in (ATTENDED, UNKNOWN) else 0


@dataclass(frozen=True)
class ClusterOutcome:
    cluster: str
    wave: int
    strata: tuple[str, ...]
    eligible: int
    attended: int
    known_non_attenders: int
    unknown: int
    lower_count: int
    upper_count: int

    @property
    def lower_rate(self) -> float | None:
        return self.lower_count / self.eligible if self.eligible else None

    @property
    def upper_rate(self) -> float | None:
        return self.upper_count / self.eligible if self.eligible else None


@dataclass(frozen=True)
class ArmSummary:
    clusters: tuple[str, ...]
    residents: int
    attended: int
    known_non_attenders: int
    unknown: int
    lower_count: int
    upper_count: int
    missing_share: float
    cluster_rates: tuple[float, ...]
    strata: tuple[str, ...]
    cluster_details: tuple[ClusterOutcome, ...]

    @property
    def point_rate(self) -> float | None:
        return self.lower_count / self.residents if self.residents else None

    @property
    def lower_rate(self) -> float | None:
        return self.lower_count / self.residents if self.residents else None

    @property
    def upper_rate(self) -> float | None:
        return self.upper_count / self.residents if self.residents else None


@dataclass(frozen=True)
class Interval:
    estimate: float | None
    low: float | None
    high: float | None
    df: int
    method: str
    estimable: bool
    reason: str | None = None


@dataclass(frozen=True)
class WindowEstimate:
    wave: int
    start: date
    end: date
    follow_up_days: int
    arm: ArmSummary
    control: ArmSummary | None
    unknown_range: tuple[float | None, float | None]
    gain: float | None
    gain_interval: Interval | None
    passes_minimum_gain: bool
    passes_max_missing: bool
    non_randomised_clusters: tuple[str, ...]
    control_exclusion_reason: str | None = None
    evidence_rule: str = STRICT

    @property
    def unknown_share(self) -> float:
        return self.arm.missing_share

    @property
    def gain_lower_bound(self) -> float | None:
        """Worst-case gain if every unknown resident did not attend."""
        if self.control is None:
            return None
        return (self.arm.lower_rate or 0.0) - (self.control.upper_rate or 0.0)

    @property
    def gain_upper_bound(self) -> float | None:
        """Best-case gain if every unknown resident did attend."""
        if self.control is None:
            return None
        return (self.arm.upper_rate or 0.0) - (self.control.lower_rate or 0.0)


def _in_window(day: date, start: date, end: date) -> bool:
    return start <= day <= end


def classify_resident(
    study: Study,
    resident: Resident,
    start: date,
    end: date,
    evidence_rule: str = STRICT,
) -> ResidentOutcome:
    rows = [
        record
        for record in study.daily_by_resident.get(resident.resident, ())
        if _in_window(record.day, start, end)
    ]

    valid_days: list[date] = []
    borrowed_days: list[date] = []
    observed: list[tuple[date, str | None]] = []
    for record in rows:
        observed.append((record.day, record.site))
        if record.report_complete is not True or record.attendance is not True:
            continue
        usable = (
            evidence_rule == RECORDED
            or study.site_is_formally_open(record.site, record.day)
            or (resident.resident, record.day) in study.borrowed_days
        )
        if not usable:
            continue
        valid_days.append(record.day)
        if (resident.resident, record.day) in study.borrowed_days:
            borrowed_days.append(record.day)

    if valid_days:
        return ResidentOutcome(
            resident=resident.resident,
            cluster=resident.cluster,
            stratum=resident.stratum,
            status=ATTENDED,
            attendance_days=tuple(sorted(set(valid_days))),
            observed_site_days=tuple(observed),
            borrowed_venue_days=tuple(sorted(set(borrowed_days))),
        )

    days = (end - start).days + 1
    complete_days = {record.day for record in rows if record.report_complete is True}
    missing_days = days - len(complete_days)
    if missing_days == 0:
        return ResidentOutcome(
            resident=resident.resident,
            cluster=resident.cluster,
            stratum=resident.stratum,
            status=NOT_ATTENDED,
            observed_site_days=tuple(observed),
        )
    return ResidentOutcome(
        resident=resident.resident,
        cluster=resident.cluster,
        stratum=resident.stratum,
        status=UNKNOWN,
        observed_site_days=tuple(observed),
        unreported_days=missing_days,
        unknown_reason=f"{missing_days}/{days} follow-up days lack complete reporting",
    )


def classify_cluster(
    study: Study,
    cluster: str,
    start: date,
    end: date,
    evidence_rule: str = STRICT,
) -> tuple[ClusterOutcome, tuple[ResidentOutcome, ...]]:
    residents = study.eligible_residents_in_cluster(cluster)
    outcomes = tuple(
        classify_resident(study, resident, start, end, evidence_rule)
        for resident in residents
    )
    attended = sum(1 for item in outcomes if item.status == ATTENDED)
    known_non = sum(1 for item in outcomes if item.status == NOT_ATTENDED)
    unknown = sum(1 for item in outcomes if item.status == UNKNOWN)
    allocation = study.allocations.get(cluster)
    return (
        ClusterOutcome(
            cluster=cluster,
            wave=allocation.wave if allocation else -1,
            strata=tuple(sorted({item.stratum for item in residents if item.stratum})),
            eligible=len(outcomes),
            attended=attended,
            known_non_attenders=known_non,
            unknown=unknown,
            lower_count=attended,
            upper_count=attended + unknown,
        ),
        outcomes,
    )


def arm_summary(
    study: Study,
    clusters: Sequence[str],
    start: date,
    end: date,
    evidence_rule: str = STRICT,
) -> tuple[ArmSummary, dict[str, tuple[ResidentOutcome, ...]]]:
    outcomes: dict[str, tuple[ResidentOutcome, ...]] = {}
    details: list[ClusterOutcome] = []
    attended = known_non = unknown = eligible = 0
    rates: list[float] = []
    strata: set[str] = set()
    for cluster in clusters:
        summary, rows = classify_cluster(study, cluster, start, end, evidence_rule)
        outcomes[cluster] = rows
        details.append(summary)
        attended += summary.attended
        known_non += summary.known_non_attenders
        unknown += summary.unknown
        eligible += summary.eligible
        if summary.eligible:
            rates.append(summary.attended / summary.eligible)
        strata.update(summary.strata)
    return (
        ArmSummary(
            clusters=tuple(clusters),
            residents=eligible,
            attended=attended,
            known_non_attenders=known_non,
            unknown=unknown,
            lower_count=attended,
            upper_count=attended + unknown,
            missing_share=(unknown / eligible) if eligible else 0.0,
            cluster_rates=tuple(rates),
            strata=tuple(sorted(strata)),
            cluster_details=tuple(details),
        ),
        outcomes,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    centre = _mean(values)
    return sum((value - centre) ** 2 for value in values) / (len(values) - 1)


def difference_interval(
    treatment: ArmSummary,
    control: ArmSummary,
    method: str = "cluster-t",
) -> Interval:
    """Difference in attendance proportions with clusters as the unit.

    The estimate is the unweighted difference of cluster means; the variance
    is the sum of the two cluster-level variance terms, with Welch degrees of
    freedom.  With fewer than two clusters in either arm the interval is not
    estimable and the reason is carried through to the report.
    """

    t_rates = treatment.cluster_rates
    c_rates = control.cluster_rates
    if len(t_rates) < 2 or len(c_rates) < 2:
        return Interval(
            None, None, None,
            min(len(t_rates), len(c_rates)) - 1,
            method,
            False,
            "任一臂少于两个抽签片区，无法由片区层面估计区间",
        )
    estimate = _mean(t_rates) - _mean(c_rates)
    a = _variance(t_rates) / len(t_rates)
    b = _variance(c_rates) / len(c_rates)
    variance = a + b
    if variance == 0:
        df = len(t_rates) + len(c_rates) - 2
    else:
        df = max(1, int((a + b) ** 2 / ((a ** 2 / (len(t_rates) - 1)) + (b ** 2 / (len(c_rates) - 1)))))
    critical = critical_value(df)
    standard_error = sqrt(variance)
    return Interval(
        estimate,
        max(-1.0, estimate - critical * standard_error),
        min(1.0, estimate + critical * standard_error),
        df,
        method,
        True,
    )


def choose_controls(
    study: Study,
    wave: int,
    start: date,
    end: date,
    explicit: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Clusters usable as controls for ``wave`` in this window.

    A cluster is usable only when it had not been allocated to open by the end
    of the window AND no site in it was formally open (even after a period
    correction) during the window.  A cluster that has already opened is never
    reused as a control.  Returns ``(usable, excluded)``.
    """

    candidates = (
        [item for item in study.allocations if item not in study.clusters_in_wave(wave)]
        if explicit is None
        else list(explicit)
    )
    usable: list[str] = []
    excluded: list[str] = []
    for cluster in sorted(set(candidates)):
        allocation = study.allocations.get(cluster)
        if allocation is None:
            raise ValueError(f"control cluster is not allocated: {cluster}")
        if cluster in study.clusters_in_wave(wave):
            raise ValueError(f"control cluster is in the treatment wave: {cluster}")
        actually_open = any(
            site.cluster == cluster
            and site.open_day <= end
            and (site.close_day is None or site.close_day >= start)
            for site in study.sites.values()
        )
        if allocation.assigned_open_day > end and not actually_open:
            usable.append(cluster)
        else:
            excluded.append(cluster)
    return tuple(usable), tuple(excluded)


def compare_wave(
    study: Study,
    wave: int,
    follow_up_days: int,
    evidence_rule: str = STRICT,
    control_clusters: Sequence[str] | None = None,
) -> WindowEstimate:
    """Intention-to-treat comparison for one wave in one observation window."""

    start, end = study.observation_window(wave, follow_up_days)
    arms = study.clusters_in_wave(wave)
    usable, excluded = choose_controls(study, wave, start, end, control_clusters)

    treatment, _ = arm_summary(study, arms, start, end, evidence_rule)
    control: ArmSummary | None = None
    gain: float | None = None
    interval: Interval | None = None
    if usable:
        control, _ = arm_summary(study, usable, start, end, evidence_rule)
        if treatment.residents and control.residents:
            gain = (treatment.point_rate or 0.0) - (control.point_rate or 0.0)
            interval = difference_interval(treatment, control)

    return WindowEstimate(
        wave=wave,
        start=start,
        end=end,
        follow_up_days=follow_up_days,
        arm=treatment,
        control=control,
        unknown_range=(treatment.lower_rate, treatment.upper_rate),
        gain=gain,
        gain_interval=interval,
        passes_minimum_gain=bool(gain is not None and gain >= study.policy.minimum_gain),
        passes_max_missing=treatment.missing_share <= study.policy.max_missing,
        non_randomised_clusters=excluded,
        control_exclusion_reason=(
            "窗口结束前已分配开放的片区不能充作对照" if excluded else None
        ),
        evidence_rule=evidence_rule,
    )
