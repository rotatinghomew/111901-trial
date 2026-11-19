"""Service-pathway and cross-talk diagnostics.

These diagnostics are separate from the intention-to-treat comparison.  They
describe what service was reachable and what residents actually did.  They are
not used to redefine the randomised groups and are not presented as causal.

Each resident-day is described on independent axes:

* reachable service -- the resident's own cluster has a formally open site;
  otherwise a cluster linked to it by a pre-registered, currently valid link
  has one (``neighbour_only``); otherwise ``none``.  A missing report is never
  used to decide that no site was open.
* reporting -- a complete daily report or not.
* temporary venue -- a confirmed borrowed-venue fact.  A borrowed venue is a
  service arrangement and is never recorded as "not open".

Attendance is *valid evidence* only when the report is complete, attendance is
reported, and the recorded site is formally open on that day (or a confirmed
borrowed-venue fact covers the resident-day).  Observed attendance at a site
that was not formally open and has no borrowed-venue confirmation is reported
as an execution anomaly, not silently accepted or silently discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

from .estimation import ATTENDED, STRICT, UNKNOWN, classify_resident
from .models import Study

OWN_SERVICE = "own"
NEIGHBOUR_ONLY = "neighbour_only"
NO_SERVICE = "none"


@dataclass(frozen=True)
class PathwayCounts:
    own: int = 0
    neighbour_only: int = 0
    none: int = 0
    missing_report: int = 0
    borrowed: int = 0
    cross_attendance_days: int = 0
    attendance_at_unopened_site_days: int = 0

    def add(self, other: "PathwayCounts") -> "PathwayCounts":
        return PathwayCounts(
            own=self.own + other.own,
            neighbour_only=self.neighbour_only + other.neighbour_only,
            none=self.none + other.none,
            missing_report=self.missing_report + other.missing_report,
            borrowed=self.borrowed + other.borrowed,
            cross_attendance_days=self.cross_attendance_days
            + other.cross_attendance_days,
            attendance_at_unopened_site_days=self.attendance_at_unopened_site_days
            + other.attendance_at_unopened_site_days,
        )

    @property
    def resident_days(self) -> int:
        return self.own + self.neighbour_only + self.none


@dataclass(frozen=True)
class ResidentPathway:
    resident: str
    cluster: str
    stratum: str
    status: str
    own_service_days: int
    neighbour_only_days: int
    no_service_days: int
    missing_report_days: int
    borrowed_venue_days: int
    own_cluster_attendance_days: tuple[date, ...]
    cross_cluster_attendance_days: tuple[date, ...]
    attendance_at_unopened_site_days: tuple[date, ...]


@dataclass(frozen=True)
class ClusterDiagnostics:
    cluster: str
    wave: int
    stratum: str
    start: date
    end: date
    eligible_residents: int
    pathways: tuple[ResidentPathway, ...]
    counts: PathwayCounts
    residents_with_own_service: int
    residents_with_neighbour_only_service: int
    residents_with_no_reachable_service: int
    residents_attending_own_cluster: int
    residents_attending_other_cluster: int
    residents_with_confirmed_borrowed_venue: int
    residents_with_unopened_site_attendance: int
    residents_unknown_without_valid_evidence: int

    @property
    def neighbour_only_person_day_share(self) -> float:
        days = self.counts.resident_days
        return self.counts.neighbour_only / days if days else 0.0

    @property
    def no_service_person_day_share(self) -> float:
        days = self.counts.resident_days
        return self.counts.none / days if days else 0.0


def _service_state(study: Study, cluster: str, day: date) -> str:
    if study.has_own_service(cluster, day):
        return OWN_SERVICE
    if study.has_neighbour_service(cluster, day):
        return NEIGHBOUR_ONLY
    return NO_SERVICE


def resident_pathway(
    study: Study,
    resident_id: str,
    start: date,
    end: date,
    evidence_rule: str = STRICT,
) -> ResidentPathway:
    resident = study.residents[resident_id]
    days = (end - start).days + 1
    own_days = neighbour_days = none_days = 0
    missing_days = 0
    borrowed_days = 0

    for offset in range(days):
        day = start + timedelta(days=offset)
        state = _service_state(study, resident.cluster, day)
        if state == OWN_SERVICE:
            own_days += 1
        elif state == NEIGHBOUR_ONLY:
            neighbour_days += 1
        else:
            none_days += 1
        record = study.record_for(resident_id, day)
        if record is None or record.report_complete is not True:
            missing_days += 1
        if (resident_id, day) in study.borrowed_days:
            borrowed_days += 1

    outcome = classify_resident(study, resident, start, end, evidence_rule)
    own_attendance: list[date] = []
    cross_attendance: list[date] = []
    unopened_attendance: list[date] = []
    for day in outcome.attendance_days:
        record = study.record_for(resident_id, day)
        site = study.sites.get(record.site) if record and record.site else None
        if site is not None and site.is_open(day) and site.cluster == resident.cluster:
            own_attendance.append(day)
        elif site is not None and site.is_open(day):
            cross_attendance.append(day)
        elif (resident_id, day) in study.borrowed_days:
            # Confirmed temporary venue: a service arrangement, not a site.
            pass
    # Days with a reported attendance at a site that is not formally open and
    # with no borrowed-venue confirmation are an execution anomaly.
    for record in study.daily_by_resident.get(resident_id, ()):
        if not (start <= record.day <= end):
            continue
        if record.report_complete is True and record.attendance is True and record.site:
            if not study.site_is_formally_open(record.site, record.day):
                if (resident_id, record.day) not in study.borrowed_days:
                    unopened_attendance.append(record.day)

    return ResidentPathway(
        resident=resident_id,
        cluster=resident.cluster,
        stratum=resident.stratum,
        status=outcome.status,
        own_service_days=own_days,
        neighbour_only_days=neighbour_days,
        no_service_days=none_days,
        missing_report_days=missing_days,
        borrowed_venue_days=borrowed_days,
        own_cluster_attendance_days=tuple(own_attendance),
        cross_cluster_attendance_days=tuple(cross_attendance),
        attendance_at_unopened_site_days=tuple(sorted(set(unopened_attendance))),
    )


def cluster_diagnostics(
    study: Study,
    cluster: str,
    start: date,
    end: date,
    evidence_rule: str = STRICT,
) -> ClusterDiagnostics:
    pathways = tuple(
        resident_pathway(study, resident.resident, start, end, evidence_rule)
        for resident in study.eligible_residents_in_cluster(cluster)
    )
    counts = PathwayCounts()
    for item in pathways:
        counts = counts.add(
            PathwayCounts(
                own=item.own_service_days,
                neighbour_only=item.neighbour_only_days,
                none=item.no_service_days,
                missing_report=item.missing_report_days,
                borrowed=item.borrowed_venue_days,
                cross_attendance_days=len(item.cross_cluster_attendance_days),
                attendance_at_unopened_site_days=len(item.attendance_at_unopened_site_days),
            )
        )
    days = (end - start).days + 1
    return ClusterDiagnostics(
        cluster=cluster,
        wave=study.allocations[cluster].wave,
        stratum=next(
            (item.stratum for item in pathways),
            "",
        ),
        start=start,
        end=end,
        eligible_residents=len(pathways),
        pathways=pathways,
        counts=counts,
        residents_with_own_service=sum(1 for item in pathways if item.own_service_days),
        residents_with_neighbour_only_service=sum(
            1 for item in pathways if item.neighbour_only_days and not item.own_service_days
        ),
        residents_with_no_reachable_service=sum(
            1 for item in pathways if item.no_service_days == days
        ),
        residents_attending_own_cluster=sum(
            1 for item in pathways if item.own_cluster_attendance_days
        ),
        residents_attending_other_cluster=sum(
            1 for item in pathways if item.cross_cluster_attendance_days
        ),
        residents_with_confirmed_borrowed_venue=sum(
            1 for item in pathways if item.borrowed_venue_days
        ),
        residents_with_unopened_site_attendance=sum(
            1 for item in pathways if item.attendance_at_unopened_site_days
        ),
        residents_unknown_without_valid_evidence=sum(
            1 for item in pathways if item.status == UNKNOWN
        ),
    )


@dataclass(frozen=True)
class SpilloverDiagnostic:
    """Cross-talk for one comparison, measured on the control clusters.

    ``neighbour_reachable_share`` is the share of control residents with at
    least one day on which only a linked, already-open cluster could serve
    them.  ``observed_cross_attendance_share`` is the share actually observed
    attending a site outside their own cluster during the window.  Neither
    changes the control group; both are reported for the reviewer.
    """

    wave: int
    control_clusters: tuple[str, ...]
    control_residents: int
    neighbour_reachable_residents: int
    borrowed_venue_residents: int
    observed_cross_attendance_residents: int
    unopened_site_attendance_residents: int
    neighbour_reachable_share: float
    observed_cross_attendance_share: float
    treatment_neighbour_reachable_share: float
    treatment_observed_cross_attendance_share: float
    borrowed_venue_days: int
    policy_cap: float

    @property
    def exceeds_cap(self) -> bool:
        return (
            self.neighbour_reachable_share > self.policy_cap
            or self.observed_cross_attendance_share > self.policy_cap
        )


def spillover_diagnostic(
    study: Study,
    wave: int,
    control_clusters: Sequence[str],
    start: date,
    end: date,
    evidence_rule: str = STRICT,
) -> SpilloverDiagnostic:
    controls = [
        cluster_diagnostics(study, cluster, start, end, evidence_rule)
        for cluster in control_clusters
    ]
    treatments = [
        cluster_diagnostics(study, cluster, start, end, evidence_rule)
        for cluster in study.clusters_in_wave(wave)
    ]

    def share(items: Sequence[ClusterDiagnostics], field: str) -> float:
        total = sum(item.eligible_residents for item in items)
        count = sum(getattr(item, field) for item in items)
        return count / total if total else 0.0

    control_residents = sum(item.eligible_residents for item in controls)
    treatment_residents = sum(item.eligible_residents for item in treatments)
    control_cross = sum(item.residents_attending_other_cluster for item in controls)
    treatment_cross = sum(item.residents_attending_other_cluster for item in treatments)
    control_neighbour = sum(
        item.residents_with_neighbour_only_service for item in controls
    )
    treatment_neighbour = sum(
        item.residents_with_neighbour_only_service for item in treatments
    )
    return SpilloverDiagnostic(
        wave=wave,
        control_clusters=tuple(control_clusters),
        control_residents=control_residents,
        neighbour_reachable_residents=control_neighbour,
        borrowed_venue_residents=sum(
            item.residents_with_confirmed_borrowed_venue for item in controls
        ),
        observed_cross_attendance_residents=control_cross,
        unopened_site_attendance_residents=sum(
            item.residents_with_unopened_site_attendance for item in controls
        ),
        neighbour_reachable_share=(
            control_neighbour / control_residents if control_residents else 0.0
        ),
        observed_cross_attendance_share=(
            control_cross / control_residents if control_residents else 0.0
        ),
        treatment_neighbour_reachable_share=(
            treatment_neighbour / treatment_residents if treatment_residents else 0.0
        ),
        treatment_observed_cross_attendance_share=(
            treatment_cross / treatment_residents if treatment_residents else 0.0
        ),
        borrowed_venue_days=sum(item.counts.borrowed for item in controls + treatments),
        policy_cap=study.policy.max_spillover,
    )


@dataclass(frozen=True)
class ExposureProfile:
    """Descriptive actual-exposure table.  Not a causal contrast."""

    wave: int
    denominator: int
    attended_own_cluster_site: int
    attended_other_open_site: int
    attended_confirmed_borrowed_venue: int
    no_valid_attendance: int
    unknown_reporting: int
    note: str = (
        "按实际暴露描述，仅说明执行情况；暴露是自选结果，不解读为因果效应。"
    )


def exposure_profile(
    study: Study,
    wave: int,
    start: date,
    end: date,
    evidence_rule: str = STRICT,
) -> ExposureProfile:
    own = other = borrowed_only = none = unknown = 0
    denominator = 0
    for cluster in study.clusters_in_wave(wave):
        for resident in study.eligible_residents_in_cluster(cluster):
            denominator += 1
            item = resident_pathway(study, resident.resident, start, end, evidence_rule)
            if item.status == UNKNOWN:
                unknown += 1
                continue
            if item.status != ATTENDED:
                none += 1
                continue
            if item.own_cluster_attendance_days:
                own += 1
            elif item.cross_cluster_attendance_days:
                other += 1
            elif item.borrowed_venue_days:
                borrowed_only += 1
            else:
                none += 1
    return ExposureProfile(
        wave=wave,
        denominator=denominator,
        attended_own_cluster_site=own,
        attended_other_open_site=other,
        attended_confirmed_borrowed_venue=borrowed_only,
        no_valid_attendance=none,
        unknown_reporting=unknown,
    )
