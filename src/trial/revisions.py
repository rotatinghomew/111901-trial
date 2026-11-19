"""Signed recommendations, revisions, and dependent schedules.

A signed recommendation is immutable.  New evidence never overwrites it under
its original number: it produces a new revision that cites the record it
supersedes.  If the new evidence removes the basis for an earlier
recommendation, the still-unstarted sessions that depended on it are flagged
as affected and stay suspended until a *new* approval decision is signed for
that batch.  Some facts -- crosstalk, missing reporting, evidence bounds and
the recommendation -- can change with a correction.  The randomised group
assignment cannot.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Iterable, Mapping

from .models import CONTINUE, CorrectionSet, Study
from .report import BatchRecommendation, WaveReport, assess_batch

SCHEDULED = "scheduled"
STARTED = "started"
SUSPENDED = "suspended"


@dataclass(frozen=True)
class SessionSlot:
    """A planned future session batch that depends on a recommendation."""

    session_id: str
    wave: int
    start_day: date
    status: str = SCHEDULED
    suspended_by: str | None = None

    @property
    def not_started(self) -> bool:
        return self.status == SCHEDULED


def _digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _wave_evidence(report: WaveReport) -> dict[str, object]:
    est = report.estimate
    return {
        "wave": est.wave,
        "start": est.start,
        "end": est.end,
        "treatment_clusters": report.treatment_clusters,
        "control_clusters": report.control_clusters,
        "withheld_clusters": report.opened_clusters_withheld,
        "treatment_residents": est.arm.residents,
        "treatment_attended": est.arm.attended,
        "treatment_unknown": est.arm.unknown,
        "control_residents": None if est.control is None else est.control.residents,
        "control_attended": None if est.control is None else est.control.attended,
        "control_unknown": None if est.control is None else est.control.unknown,
        "gain": est.gain,
        "gain_low": None if est.gain_interval is None else est.gain_interval.low,
        "gain_high": None if est.gain_interval is None else est.gain_interval.high,
        "unknown_share": est.unknown_share,
        "spillover_pass": report.spillover_pass,
        "strata_pass": report.strata_pass,
        "missing_pass": report.missing_pass,
        "decision": report.decision,
        "corrections_applied": list(report.corrections_applied),
    }


@dataclass(frozen=True)
class ApprovalRecord:
    record_id: str
    batch_id: str
    decision: str
    issued_on: date
    follow_up_days: int
    evidence_digest: str
    evidence: tuple[Mapping[str, object], ...]
    supersedes: str | None = None
    revision: int = 1
    reason: str = ""
    schedule: tuple[SessionSlot, ...] = ()
    basis_lost: bool = False
    affected_unstarted: tuple[SessionSlot, ...] = ()

    @property
    def is_revision(self) -> bool:
        return self.supersedes is not None

    @property
    def schedule_ids(self) -> tuple[str, ...]:
        return tuple(item.session_id for item in self.schedule)


@dataclass
class RecommendationLedger:
    """Append-only ledger of signed recommendations and schedule states."""

    records: list[ApprovalRecord] = field(default_factory=list)
    schedule: list[SessionSlot] = field(default_factory=list)

    # ------------------------------------------------------------------
    def add_schedule(self, slots: Iterable[SessionSlot]) -> None:
        known = {item.session_id for item in self.schedule}
        for slot in slots:
            if slot.session_id in known:
                raise ValueError(f"duplicate session id: {slot.session_id}")
            self.schedule.append(slot)
            known.add(slot.session_id)

    def record(self, record_id: str) -> ApprovalRecord:
        for item in self.records:
            if item.record_id == record_id:
                return item
        raise KeyError(record_id)

    def by_batch(self, batch_id: str) -> tuple[ApprovalRecord, ...]:
        return tuple(item for item in self.records if item.batch_id == batch_id)

    def latest_for_batch(self, batch_id: str) -> ApprovalRecord | None:
        items = self.by_batch(batch_id)
        return max(items, key=lambda item: (item.issued_on, item.revision)) if items else None

    def _schedule_for_waves(self, waves: Iterable[int]) -> tuple[SessionSlot, ...]:
        wanted = set(waves)
        return tuple(item for item in self.schedule if item.wave in wanted)

    def issue(
        self,
        batch_id: str,
        recommendation: BatchRecommendation,
        issued_on: date,
        follow_up_days: int,
        record_id: str | None = None,
        reason: str = "",
        supersedes: str | None = None,
        revision: int = 1,
        schedule: Iterable[SessionSlot] | None = None,
    ) -> ApprovalRecord:
        evidence = tuple(_wave_evidence(report) for report in recommendation.waves)
        if record_id is None:
            record_id = _next_id(self.records, batch_id, revision)
        if any(item.record_id == record_id for item in self.records):
            raise ValueError(f"record id already used: {record_id}")
        slots = tuple(schedule) if schedule is not None else self._schedule_for_waves(
            report.wave for report in recommendation.waves
        )
        record = ApprovalRecord(
            record_id=record_id,
            batch_id=batch_id,
            decision=recommendation.decision,
            issued_on=issued_on,
            follow_up_days=follow_up_days,
            evidence_digest=_digest({"waves": evidence}),
            evidence=evidence,
            supersedes=supersedes,
            revision=revision,
            reason=reason,
            schedule=slots,
        )
        if record.decision != CONTINUE:
            self._suspend(record.schedule, record.record_id)
        self.records.append(record)
        return record

    # ------------------------------------------------------------------
    def revise(
        self,
        study: Study,
        record_id: str,
        corrections: CorrectionSet,
        issued_on: date,
        follow_up_days: int | None = None,
        reason: str = "",
    ) -> ApprovalRecord:
        """Re-assess with corrections and sign a new, linked revision.

        The original record is never modified.  Unstarted sessions that lose
        their basis are suspended and listed on the new record.
        """

        original = self.record(record_id)
        if any(item.supersedes == record_id for item in self.records):
            raise ValueError(f"{record_id} already has a revision")
        corrected = study.with_corrections(corrections)
        days = follow_up_days or original.follow_up_days
        waves = tuple(int(item["wave"]) for item in original.evidence)
        recommendation = assess_batch(corrected, days, waves=waves)

        before = {item.session_id: item.status for item in self.schedule}
        new_record = self.issue(
            batch_id=original.batch_id,
            recommendation=recommendation,
            issued_on=issued_on,
            follow_up_days=days,
            record_id=f"{original.record_id}-r{original.revision + 1}",
            reason=reason,
            supersedes=original.record_id,
            revision=original.revision + 1,
        )
        # Sessions this revision suspended are recorded on it.
        suspended_now = tuple(
            item
            for item in self.schedule
            if before[item.session_id] == SCHEDULED
            and item.status == SUSPENDED
            and item.session_id in new_record.schedule_ids
        )
        lost = original.decision == CONTINUE and recommendation.decision != CONTINUE
        if suspended_now:
            self.records[-1] = replace(
                self.records[-1],
                basis_lost=lost,
                affected_unstarted=suspended_now,
            )
        return self.records[-1]

    # ------------------------------------------------------------------
    def _suspend(self, slots: Iterable[SessionSlot], record_id: str) -> None:
        wanted = {item.session_id for item in slots}
        for index, slot in enumerate(self.schedule):
            if slot.session_id in wanted and slot.not_started:
                self.schedule[index] = replace(
                    slot, status=SUSPENDED, suspended_by=record_id
                )

    def restore(self, session_id: str, new_record: ApprovalRecord) -> None:
        """Restore a suspended session only under a newly signed approval.

        ``new_record`` must be a signed decision that comes *after* the
        decision that suspended the session -- the very record that suspended
        it cannot also restore it -- and must itself be a continue decision.
        """

        if new_record.decision != CONTINUE:
            raise ValueError("only a continue decision can restore a session")
        if not any(item.record_id == new_record.record_id for item in self.records):
            raise ValueError("restoring decision is not in the ledger")
        slot = next((item for item in self.schedule if item.session_id == session_id), None)
        if slot is None:
            raise KeyError(session_id)
        if slot.status != SUSPENDED:
            raise ValueError(f"session {session_id} is not suspended")
        if new_record.record_id == slot.suspended_by:
            raise ValueError(
                "the decision that suspended the session cannot restore it; "
                "a new approval decision is required"
            )
        if new_record.issued_on < self.record(slot.suspended_by).issued_on:
            raise ValueError("restoring decision predates the suspension")
        if session_id not in new_record.schedule_ids:
            raise ValueError("restoring decision does not cover this session")
        for index, item in enumerate(self.schedule):
            if item.session_id == session_id:
                self.schedule[index] = replace(
                    item, status=SCHEDULED, suspended_by=None
                )
                return
        raise KeyError(session_id)

    def affected_unstarted(self) -> tuple[SessionSlot, ...]:
        return tuple(item for item in self.schedule if item.status == SUSPENDED)

    def mark_started(self, session_id: str) -> None:
        for index, item in enumerate(self.schedule):
            if item.session_id == session_id:
                if item.status == SUSPENDED:
                    raise ValueError(
                        f"session {session_id} is suspended and needs a new approval"
                    )
                self.schedule[index] = replace(item, status=STARTED)
                return
        raise KeyError(session_id)


def _next_id(records: Iterable[ApprovalRecord], batch_id: str, revision: int) -> str:
    existing = {item.record_id for item in records}
    candidate = f"{batch_id}-{revision:03d}"
    suffix = 1
    while candidate in existing:
        suffix += 1
        candidate = f"{batch_id}-{revision:03d}-{suffix}"
    return candidate
