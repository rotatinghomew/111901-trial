"""Reviewer-facing reports and the internal audit trace.

Two different products come out of the same assessment:

* ``render_reviewer_report`` -- the routine material for the sign-off panel.
  Aggregate only; it never contains resident identifiers.
* ``audit_trace`` -- the pilot-comparison record for the project team, which
  can be traced by resident and by service area, including the reason each
  resident was included or excluded.

The reviewer report always shows, for every wave, the comparable groups, the
denominator, the interval estimate, the unknown range, and -- when an
inference is not supported -- the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

from .diagnostics import (
    ExposureProfile,
    SpilloverDiagnostic,
    exposure_profile,
    spillover_diagnostic,
)
from .estimation import (
    ATTENDED,
    NOT_ATTENDED,
    STRICT,
    UNKNOWN,
    WindowEstimate,
    arm_summary,
    choose_controls,
    classify_resident,
    compare_wave,
)
from .models import (
    CONTINUE,
    DECISION_LABELS,
    EXTEND_OBSERVATION,
    NEED_MORE_EVIDENCE,
    Study,
)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


@dataclass(frozen=True)
class WaveReport:
    estimate: WindowEstimate
    treatment_clusters: tuple[str, ...]
    control_clusters: tuple[str, ...]
    opened_clusters_withheld: tuple[str, ...]
    spillover: SpilloverDiagnostic | None
    exposure: ExposureProfile
    missing_pass: bool
    spillover_pass: bool | None
    strata_pass: bool
    required_strata: tuple[str, ...]
    decision: str
    decision_reasons: tuple[str, ...]
    unsupported_reasons: tuple[str, ...]
    corrections_applied: tuple[str, ...] = ()

    @property
    def wave(self) -> int:
        return self.estimate.wave

    @property
    def comparable(self) -> bool:
        return self.estimate.control is not None and self.estimate.gain_interval is not None


def _strata_coverage(
    study: Study,
    treatment_clusters: Sequence[str],
    control_clusters: Sequence[str],
    start: date,
    end: date,
    evidence_rule: str,
) -> tuple[bool, tuple[str, ...]]:
    treatment, _ = arm_summary(study, treatment_clusters, start, end, evidence_rule)
    control, _ = arm_summary(study, control_clusters, start, end, evidence_rule)
    missing: list[str] = []
    for stratum in study.policy.required_strata:
        if stratum not in treatment.strata or stratum not in control.strata:
            missing.append(stratum)
    return (not missing, tuple(missing))


def assess_wave(
    study: Study,
    wave: int,
    follow_up_days: int,
    evidence_rule: str = STRICT,
    control_clusters: Sequence[str] | None = None,
) -> WaveReport:
    estimate = compare_wave(
        study, wave, follow_up_days, evidence_rule, control_clusters
    )
    treatment_clusters = study.clusters_in_wave(wave)
    usable, excluded = choose_controls(
        study, wave, estimate.start, estimate.end, control_clusters
    )
    spill = (
        spillover_diagnostic(
            study, wave, usable, estimate.start, estimate.end, evidence_rule
        )
        if usable
        else None
    )
    exposure = exposure_profile(
        study, wave, estimate.start, estimate.end, evidence_rule
    )

    strata_ok, missing_strata = _strata_coverage(
        study,
        treatment_clusters,
        usable,
        estimate.start,
        estimate.end,
        evidence_rule,
    )

    unsupported: list[str] = []
    if estimate.control is None:
        unsupported.append("本波观察窗口内没有尚未开放的片区可作为同期对照，不能做意向分配比较。")
    if estimate.gain_interval is not None and not estimate.gain_interval.estimable:
        unsupported.append(
            estimate.gain_interval.reason
            or "片区数量不足，无法由片区层面给出区间估计。"
        )
    if not estimate.passes_max_missing:
        unsupported.append(
            f"缺失回报比例 {_pct(estimate.arm.missing_share)} 超过上限 "
            f"{_pct(study.policy.max_missing)}；未完整回报且无有效出席证据者只能记为未知。"
        )
    if spill is not None and spill.exceeds_cap:
        unsupported.append(
            f"对照片区串扰 {_pct(max(spill.neighbour_reachable_share, spill.observed_cross_attendance_share))} "
            f"超过允许上限 {_pct(study.policy.max_spillover)}。"
        )
    if not strata_ok:
        unsupported.append(
            "本波可比组未同时覆盖以下片区类型（批次整体覆盖情况见总建议）："
            + "、".join(missing_strata)
            + "。"
        )

    reasons: list[str] = []
    quality_ok = (
        estimate.control is not None
        and estimate.passes_max_missing
        and (spill is None or not spill.exceeds_cap)
        and not (
            estimate.gain_interval is not None
            and not estimate.gain_interval.estimable
        )
    )

    if quality_ok and estimate.passes_minimum_gain and estimate.gain_interval is not None:
        lower = estimate.gain_interval.low
        if lower is not None and lower > 0:
            decision = CONTINUE
            reasons.append(
                f"点估计增益 {_pct(estimate.gain)} 达到最低增益 "
                f"{_pct(study.policy.minimum_gain)}，且片区层面区间下限 {_pct(lower)} 大于 0；"
                "缺报、串扰和片区类型覆盖均在阈值内。"
            )
        else:
            decision = EXTEND_OBSERVATION
            reasons.append(
                f"可见增益 {_pct(estimate.gain)}，但片区层面区间下限为 {_pct(lower)}，"
                "证据未过门槛；不得据此自动推荐继续，建议延长观察。"
            )
    elif quality_ok:
        decision = EXTEND_OBSERVATION
        reasons.append(
            f"数据质量在阈值内，但增益 {_pct(estimate.gain)} 未达到最低增益 "
            f"{_pct(study.policy.minimum_gain)}；建议延长观察，不自动推荐继续。"
        )
    else:
        decision = NEED_MORE_EVIDENCE
        reasons.append("可比性或数据质量未达门槛，现有证据不支持作出继续决定，需待补证。")

    return WaveReport(
        estimate=estimate,
        treatment_clusters=treatment_clusters,
        control_clusters=usable,
        opened_clusters_withheld=excluded,
        spillover=spill,
        exposure=exposure,
        missing_pass=estimate.passes_max_missing,
        spillover_pass=None if spill is None else not spill.exceeds_cap,
        strata_pass=strata_ok,
        required_strata=study.policy.required_strata,
        decision=decision,
        decision_reasons=tuple(reasons),
        unsupported_reasons=tuple(unsupported),
        corrections_applied=tuple(study.applied_corrections),
    )


@dataclass(frozen=True)
class BatchRecommendation:
    """Recommendation for the next batch of sessions.

    It is deliberately conservative: a single wave that shows a gain but
    cannot support it does not authorise opening the next batch.  The required
    stratum coverage is a *batch-level* condition: it is not required that
    every single wave contains both urban and rural clusters, but the set of
    comparable waves must jointly cover each required stratum in both a
    treatment and a control arm.
    """

    decision: str
    reasons: tuple[str, ...]
    waves: tuple[WaveReport, ...]
    covered_strata: tuple[str, ...] = ()
    missing_strata: tuple[str, ...] = ()
    required_strata: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return DECISION_LABELS[self.decision]


def _batch_strata_coverage(
    reports: Sequence[WaveReport],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    treatment: set[str] = set()
    control: set[str] = set()
    required: set[str] = set()
    for report in reports:
        required.update(report.required_strata)
        treatment.update(report.estimate.arm.strata)
        if report.estimate.control is not None:
            control.update(report.estimate.control.strata)
    covered = treatment & control
    missing = tuple(sorted(required - covered))
    return tuple(sorted(covered & required)), missing


def assess_batch(
    study: Study,
    follow_up_days: int,
    evidence_rule: str = STRICT,
    waves: Sequence[int] | None = None,
) -> BatchRecommendation:
    selected = tuple(sorted(waves)) if waves is not None else study.waves
    reports = tuple(
        assess_wave(study, wave, follow_up_days, evidence_rule) for wave in selected
    )
    covered, missing = _batch_strata_coverage(reports)

    if any(report.decision == NEED_MORE_EVIDENCE for report in reports):
        decision = NEED_MORE_EVIDENCE
        reasons = [
            f"第 {report.wave} 波："
            + "；".join(report.unsupported_reasons or report.decision_reasons)
            for report in reports
            if report.decision == NEED_MORE_EVIDENCE
        ]
    elif all(report.decision == CONTINUE for report in reports):
        decision = CONTINUE
        reasons = ["所有已审波次的增益、缺报和串扰均过门槛。"]
    else:
        decision = EXTEND_OBSERVATION
        reasons = [
            f"第 {report.wave} 波：" + "；".join(report.decision_reasons)
            for report in reports
            if report.decision != CONTINUE
        ]

    if missing:
        decision = NEED_MORE_EVIDENCE
        reasons.append(
            "批次整体未在可比的试验与对照臂中共同覆盖必须包含的片区类型："
            + "、".join(missing)
            + "，证据不足以支持扩围。"
        )
    elif decision == CONTINUE:
        reasons.append(
            "必须覆盖的片区类型（" + "、".join(sorted(set(study.policy.required_strata)))
            + "）已在可比的试验与对照臂中共同覆盖。"
        )

    return BatchRecommendation(
        decision=decision,
        reasons=tuple(reasons),
        waves=reports,
        covered_strata=covered,
        missing_strata=missing,
        required_strata=tuple(sorted(set(study.policy.required_strata))),
    )


# ----------------------------------------------------------------------
# reviewer-facing rendering (no resident identifiers)
# ----------------------------------------------------------------------
def render_reviewer_report(
    study: Study,
    recommendation: BatchRecommendation,
    follow_up_days: int,
) -> str:
    lines: list[str] = []
    lines.append("社区健身指导课扩围审签材料")
    lines.append("=" * 40)
    lines.append(
        f"审签建议：{recommendation.label}（{recommendation.decision}）"
    )
    for reason in recommendation.reasons:
        lines.append(f"  - {reason}")
    lines.append("")
    lines.append(f"各波观察窗口固定为抽签后 {follow_up_days} 天；分组按抽签名单固定，"
                 "跨区参加或未参加均不改变归属。不确定性以抽签片区为单位。")
    if recommendation.required_strata:
        lines.append(
            "  必须覆盖的片区类型："
            + "、".join(recommendation.required_strata)
            + "；已同时在可比试验与对照臂中覆盖："
            + ("、".join(recommendation.covered_strata) if recommendation.covered_strata else "无")
            + ("；仍缺：" + "、".join(recommendation.missing_strata) if recommendation.missing_strata else "")
        )
    lines.append("")

    for report in recommendation.waves:
        est = report.estimate
        lines.append(f"第 {est.wave} 波（{est.start.isoformat()} 至 {est.end.isoformat()}）")
        lines.append("-" * 40)
        lines.append(
            "  可比组：试验区 "
            + "、".join(report.treatment_clusters)
            + "；同期尚未开放对照区 "
            + ("、".join(report.control_clusters) if report.control_clusters else "无")
        )
        if report.opened_clusters_withheld:
            lines.append(
                "  已开放片区不得重新充当对照，已排除："
                + "、".join(report.opened_clusters_withheld)
            )
        arm = est.arm
        lines.append(
            f"  主指标分母（基线合资格居民）：{arm.residents}；"
            f"有至少一次有效出席：{arm.attended}；"
            f"完整回报且未出席：{arm.known_non_attenders}；"
            f"缺报且无有效证据（未知）：{arm.unknown}"
        )
        lines.append(
            f"  试验区出席比例范围（未知计为未出席至未知计为出席）："
            f"{_pct(est.unknown_range[0])} 至 {_pct(est.unknown_range[1])}"
        )
        if est.control is not None:
            ctl = est.control
            lines.append(
                f"  对照分母：{ctl.residents}；有至少一次有效出席：{ctl.attended}；"
                f"完整回报且未出席：{ctl.known_non_attenders}；未知：{ctl.unknown}"
            )
            lines.append(
                f"  对照出席比例范围：{_pct(ctl.lower_rate)} 至 {_pct(ctl.upper_rate)}"
            )
            interval = est.gain_interval
            lines.append(f"  意向分配增益（点估计）：{_pct(est.gain)}")
            if interval is not None and interval.estimable:
                lines.append(
                    f"  片区层面 95% 区间：{_pct(interval.low)} 至 {_pct(interval.high)}"
                    f"（df={interval.df}，以 {len(est.arm.cluster_rates)} 个试验区、"
                    f"{len(est.control.cluster_rates)} 个对照区为不确定性单位）"
                )
            else:
                lines.append(
                    "  区间估计：不可估计（"
                    + (interval.reason if interval and interval.reason else "片区数不足")
                    + "）"
                )
            lines.append(
                f"  未知可能改变估计的范围（最不利至最有利）："
                f"{_pct(est.gain_lower_bound)} 至 {_pct(est.gain_upper_bound)}"
            )
        else:
            lines.append("  区间估计与增益：不支持推断（无同期对照）。")

        if report.spillover is not None:
            s = report.spillover
            lines.append(
                f"  串扰诊断（对照区）：仅邻区可服务 {s.neighbour_reachable_residents}/"
                f"{s.control_residents}（{_pct(s.neighbour_reachable_share)}）；"
                f"实际跨区出席 {s.observed_cross_attendance_residents}/"
                f"{s.control_residents}（{_pct(s.observed_cross_attendance_share)}）；"
                f"借场登记 {s.borrowed_venue_days} 人日；"
                f"上限 {_pct(s.policy_cap)} → "
                + ("未超过" if not s.exceeds_cap else "已超过")
            )
        exp = report.exposure
        lines.append(
            f"  实际暴露描述（非因果）：本区站点 {exp.attended_own_cluster_site}；"
            f"其他已开站点 {exp.attended_other_open_site}；"
            f"已确认借场 {exp.attended_confirmed_borrowed_venue}；"
            f"无有效出席 {exp.no_valid_attendance}；"
            f"回报未知 {exp.unknown_reporting}（分母 {exp.denominator}）"
        )
        lines.append(
            f"  缺报是否在阈值内：{'是' if report.missing_pass else '否'}；"
            f"本波试验与对照臂是否覆盖全部必须片区类型："
            f"{'是' if report.strata_pass else '否（批次层面见总建议）'}"
        )
        if report.unsupported_reasons:
            lines.append("  不支持推断的原因：")
            for reason in report.unsupported_reasons:
                lines.append(f"    - {reason}")
        lines.append(f"  本波建议：{DECISION_LABELS[report.decision]}")
        for reason in report.decision_reasons:
            lines.append(f"    - {reason}")
        lines.append("")
    lines.append("说明：点估计与区间只描述按抽签分组的意向分配效果；实际暴露分布受自选影响，"
                 "不作为因果效应。缺报者只能记为未知，故同时给出未知可能改变估计的范围。")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# internal audit trace (may include resident identifiers)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AuditRow:
    wave: int
    cluster: str
    resident: str
    stratum: str
    role: str
    included: bool
    status: str
    reason: str


def audit_trace(
    study: Study,
    follow_up_days: int,
    waves: Sequence[int] | None = None,
) -> tuple[AuditRow, ...]:
    selected = tuple(sorted(waves)) if waves is not None else study.waves
    rows: list[AuditRow] = []
    for wave in selected:
        report = assess_wave(study, wave, follow_up_days)
        est = report.estimate
        for role, clusters in (
            ("treatment", report.treatment_clusters),
            ("control", report.control_clusters),
        ):
            for cluster in clusters:
                for resident in study.residents_in_cluster(cluster):
                    if not resident.eligible:
                        rows.append(
                            AuditRow(
                                wave=wave,
                                cluster=cluster,
                                resident=resident.resident,
                                stratum=resident.stratum,
                                role=role,
                                included=False,
                                status="excluded",
                                reason="基线不合资格，不进入分母",
                            )
                        )
                        continue
                    outcome = classify_resident(
                        study, resident, est.start, est.end, est.evidence_rule
                    )
                    if outcome.status == ATTENDED:
                        status = ATTENDED
                        reason = "窗口内有至少一次有效出席（站点正式开放或已确认借场）"
                    elif outcome.status == NOT_ATTENDED:
                        status = NOT_ATTENDED
                        reason = "窗口内逐日完整回报且无出席"
                    else:
                        status = UNKNOWN
                        reason = outcome.unknown_reason or "缺报且无有效出席证据"
                    rows.append(
                        AuditRow(
                            wave=wave,
                            cluster=cluster,
                            resident=resident.resident,
                            stratum=resident.stratum,
                            role=role,
                            included=True,
                            status=status,
                            reason=reason,
                        )
                    )
    return tuple(rows)


def render_audit_csv(rows: Iterable[AuditRow]) -> str:
    header = "wave,cluster,resident,stratum,role,included,status,reason"
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.wave},{row.cluster},{row.resident},{row.stratum},{row.role},"
            f"{int(row.included)},{row.status},{row.reason}"
        )
    return "\n".join(lines)
