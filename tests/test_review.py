"""Behavioural tests for the review analysis, built on small synthetic studies.

These tests pin the design decisions that the reviewer depends on; they do not
re-test the fixture files (see ``test_fixtures.py``).
"""

import datetime as dt
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trial import (  # noqa: E402
    ATTENDED,
    CONTINUE,
    EXTEND_OBSERVATION,
    NEED_MORE_EVIDENCE,
    UNKNOWN,
    Allocation,
    BorrowedVenueFact,
    CorrectionSet,
    DailyRecord,
    Link,
    RecommendationLedger,
    ReportCorrection,
    Resident,
    ReviewPolicy,
    SCHEDULED,
    SUSPENDED,
    SessionSlot,
    Site,
    SitePeriodCorrection,
    Study,
    assess_batch,
    assess_wave,
    audit_trace,
    classify_resident,
    compare_wave,
    default_follow_up_days,
    load_study,
    render_reviewer_report,
    spillover_diagnostic,
)

D = dt.date


def _day(offset: int) -> dt.date:
    return D(2026, 6, 1) + dt.timedelta(days=offset)


def _policy(**overrides) -> ReviewPolicy:
    values = dict(
        minimum_gain=0.03, max_spillover=0.25, max_missing=0.1,
        required_strata=("urban", "rural"),
    )
    values.update(overrides)
    return ReviewPolicy(**values)


def _study(
    residents,
    allocations,
    sites,
    links,
    daily,
    policy=None,
    borrowed=frozenset(),
) -> Study:
    return Study(
        residents={item.resident: item for item in residents},
        allocations={item.cluster: item for item in allocations},
        sites={item.site: item for item in sites},
        links=tuple(links),
        daily=tuple(daily),
        policy=policy or _policy(),
        borrowed_days=borrowed,
    )


def _full_reports(resident_ids, start, days=14, attended=None, day_site=None):
    """Complete reports for every day; attend only on the given days."""

    attended = attended or set()
    rows = []
    for resident in resident_ids:
        for offset in range(days):
            day = start + dt.timedelta(days=offset)
            hit = (resident, offset) in attended
            rows.append(
                DailyRecord(
                    resident=resident,
                    day=day,
                    site=(day_site or {}).get(resident, "tx-site"),
                    attendance=hit,
                    report_complete=True,
                )
            )
    return rows


def _simple_study(
    n_per_cluster=10,
    treatment_clusters=("a1", "a2"),
    control_clusters=("b1", "b2"),
    wave=1,
    attended_treatment=0.5,
    attended_control=0.0,
    treatment_rates=None,
    control_rates=None,
    missing_control=0.0,
    missing_treatment=0.0,
):
    residents = []
    allocations = []
    sites = []
    wave_map = {}
    for cluster in treatment_clusters:
        allocations.append(Allocation(cluster, wave, _day(0)))
        wave_map[cluster] = wave
        sites.append(Site(f"tx-{cluster}", cluster, _day(0)))
    for cluster in control_clusters:
        allocations.append(Allocation(cluster, wave + 1, _day(60)))
        wave_map[cluster] = wave + 1
        sites.append(Site(f"tx-{cluster}", cluster, _day(60)))

    daily = []
    strata = ("urban", "rural")
    for cluster in (*treatment_clusters, *control_clusters):
        is_treatment = cluster in treatment_clusters
        if is_treatment:
            share = (treatment_rates or {}).get(cluster, attended_treatment)
            missing_share = missing_treatment
        else:
            share = (control_rates or {}).get(cluster, attended_control)
            missing_share = missing_control
        for index in range(n_per_cluster):
            resident = f"{cluster}-r{index:02d}"
            residents.append(Resident(resident, cluster, True, strata[index % 2]))
            attends = index < round(share * n_per_cluster)
            complete = index >= round(missing_share * n_per_cluster)
            for offset in range(14):
                daily.append(
                    DailyRecord(
                        resident,
                        _day(offset),
                        f"tx-{cluster}",
                        attends,
                        complete,
                    )
                )
    return _study(residents, allocations, sites, [], daily), wave_map


class GroupingTest(unittest.TestCase):
    def test_cross_cluster_attendance_does_not_move_resident(self):
        study, _ = _simple_study()
        # Make one control resident attend the open treatment site throughout.
        control_resident = "b1-r00"
        rows = [
            DailyRecord(
                r.resident,
                r.day,
                "tx-a1",
                True,
                True,
            )
            if r.resident == control_resident
            else r
            for r in study.daily
        ]
        study = Study(
            residents=study.residents,
            allocations=study.allocations,
            sites=study.sites,
            links=study.links,
            daily=tuple(rows),
            policy=study.policy,
        )
        estimate = compare_wave(study, 1, 14)
        self.assertEqual(estimate.arm.residents, 20)
        self.assertEqual(estimate.control.residents, 20)
        # The cross-attending control resident counts in the control arm and
        # does raise the control attendance -- it is not silently moved.
        self.assertEqual(estimate.control.attended, 1)
        self.assertFalse(any("b1-r00" in c for c in estimate.arm.clusters))

    def test_denominator_uses_baseline_eligible_only_and_counts_once(self):
        study, _ = _simple_study(n_per_cluster=10)
        residents = dict(study.residents)
        residents["a1-ghost"] = Resident("a1-ghost", "a1", False, "urban")
        study = Study(
            residents=residents,
            allocations=study.allocations,
            sites=study.sites,
            links=study.links,
            daily=study.daily,
            policy=study.policy,
        )
        estimate = compare_wave(study, 1, 14)
        self.assertEqual(estimate.arm.residents, 20)

    def test_known_attendance_regardless_of_remaining_missing_days(self):
        # An attended day is sufficient evidence even if other days are missing.
        resident = Resident("r1", "a1", True, "urban")
        study = _study(
            [resident],
            [Allocation("a1", 1, _day(0))],
            [Site("s1", "a1", _day(0))],
            [],
            [DailyRecord("r1", _day(3), "s1", True, True)],
        )
        outcome = classify_resident(study, resident, _day(0), _day(13))
        self.assertEqual(outcome.status, ATTENDED)


class UnknownAndIntervalTest(unittest.TestCase):
    def test_missing_days_become_unknown_not_non_attendance(self):
        study, _ = _simple_study(
            n_per_cluster=10, attended_treatment=0.0, missing_treatment=0.5
        )
        estimate = compare_wave(study, 1, 14)
        self.assertEqual(estimate.arm.unknown, 10)
        self.assertEqual(estimate.arm.known_non_attenders, 10)
        low, high = estimate.unknown_range
        self.assertLess(low, high)

    def test_attendance_at_unopened_site_is_not_valid_evidence(self):
        resident = Resident("r1", "a1", True, "urban")
        site = Site("s1", "a1", _day(10))
        study = _study(
            [resident],
            [Allocation("a1", 1, _day(0))],
            [site],
            [],
            [DailyRecord("r1", _day(2), "s1", True, True)],
        )
        outcome = classify_resident(study, resident, _day(0), _day(13))
        self.assertEqual(outcome.status, UNKNOWN)

    def test_confirmed_borrowed_venue_is_valid_and_not_no_service(self):
        resident = Resident("r1", "a1", True, "urban")
        study = _study(
            [resident],
            [Allocation("a1", 1, _day(0))],
            [Site("s1", "a1", _day(10))],
            [],
            [DailyRecord("r1", _day(2), None, True, True)],
            borrowed=frozenset({("r1", _day(2))}),
        )
        outcome = classify_resident(study, resident, _day(0), _day(13))
        self.assertEqual(outcome.status, ATTENDED)
        self.assertEqual(outcome.borrowed_venue_days, (_day(2),))

    def test_interval_uses_clusters_not_residents(self):
        study, _ = _simple_study(
            n_per_cluster=200,
            treatment_rates={"a1": 0.6, "a2": 0.4},
            control_rates={"b1": 0.05, "b2": 0.15},
        )
        small = compare_wave(study, 1, 14)
        self.assertTrue(small.gain_interval.estimable)
        # With residents treated independently the interval would shrink with
        # sample size; here it depends only on the cluster-level spread.
        larger, _ = _simple_study(
            n_per_cluster=2000,
            treatment_rates={"a1": 0.6, "a2": 0.4},
            control_rates={"b1": 0.05, "b2": 0.15},
        )
        big = compare_wave(larger, 1, 14)
        self.assertEqual(small.gain_interval.df, big.gain_interval.df)
        self.assertAlmostEqual(
            small.gain_interval.low, big.gain_interval.low, places=6
        )

    def test_single_control_cluster_is_not_estimable(self):
        study, _ = _simple_study(control_clusters=("b1",))
        estimate = compare_wave(study, 1, 14)
        self.assertFalse(estimate.gain_interval.estimable)
        self.assertIsNone(estimate.gain_interval.low)
        report = assess_wave(study, 1, 14)
        self.assertTrue(any("区间" in reason for reason in report.unsupported_reasons))
        self.assertEqual(report.decision, NEED_MORE_EVIDENCE)


class ControlSelectionTest(unittest.TestCase):
    def test_opened_cluster_is_never_reused_as_control(self):
        study, _ = _simple_study()
        # Wave 1 starts day 0; b1/b2 are allocated day 60, so out of window.
        estimate = compare_wave(study, 2, 14)
        self.assertIn("a1", estimate.non_randomised_clusters)

    def test_explicit_control_already_open_is_rejected(self):
        study, _ = _simple_study()
        estimate = compare_wave(study, 2, 14, control_clusters=("a1",))
        self.assertIsNone(estimate.control)
        self.assertEqual(estimate.non_randomised_clusters, ("a1",))
        self.assertIsNone(estimate.gain)

    def test_cluster_formally_open_early_is_not_a_control(self):
        study, _ = _simple_study()
        # b1 was allocated day 60 but a corrected site period opens it day 5.
        study = study.with_corrections(
            CorrectionSet(
                site_periods=(SitePeriodCorrection("tx-b1", open_day=_day(5)),)
            )
        )
        estimate = compare_wave(study, 1, 14)
        self.assertEqual(estimate.control.clusters, ("b2",))
        self.assertEqual(estimate.non_randomised_clusters, ("b1",))
        self.assertFalse(estimate.gain_interval.estimable)


class SpilloverTest(unittest.TestCase):
    def test_neighbour_service_differs_from_missing_report(self):
        residents = [
            Resident("r1", "a1", True, "urban"),
            Resident("r2", "a2", True, "urban"),
            Resident("r3", "a3", True, "urban"),
        ]
        allocations = [
            Allocation("a1", 1, _day(0)),
            Allocation("a2", 2, _day(30)),
            Allocation("a3", 3, _day(60)),
        ]
        sites = [
            Site("s1", "a1", _day(0)),
            Site("s2", "a2", _day(30)),
            Site("s3", "a3", _day(60)),
        ]
        links = [Link("a1", "a2", _day(0)), Link("a1", "a3", _day(0))]
        # No daily reports at all for a2/a3: absence of reports is not
        # evidence that no site was open.
        daily = [DailyRecord("r1", _day(1), "s1", True, True)]
        study = _study(residents, allocations, sites, links, daily)
        diag = spillover_diagnostic(study, 1, ("a2", "a3"), _day(0), _day(13))
        self.assertEqual(diag.neighbour_reachable_residents, 2)
        self.assertEqual(diag.neighbour_reachable_share, 1.0)
        self.assertTrue(diag.exceeds_cap)

    def test_borrowed_venue_is_reported_but_not_counted_as_no_service(self):
        residents = [Resident("r1", "a1", True, "urban")]
        study = _study(
            residents,
            [Allocation("a1", 1, _day(0)), Allocation("a2", 2, _day(30))],
            [Site("s1", "a1", _day(0)), Site("s2", "a2", _day(30))],
            [],
            [],
            borrowed=frozenset({("r1", _day(1))}),
        )
        diag = spillover_diagnostic(study, 1, ("a2",), _day(0), _day(13))
        self.assertEqual(diag.borrowed_venue_days, 1)
        self.assertEqual(diag.neighbour_reachable_share, 0.0)


class DecisionTest(unittest.TestCase):
    def test_gap_below_minimum_is_not_recommended_for_continuation(self):
        study, _ = _simple_study(attended_treatment=0.02, attended_control=0.0)
        report = assess_wave(study, 1, 14)
        self.assertNotEqual(report.decision, CONTINUE)

    def test_visible_gain_failing_interval_is_not_continued(self):
        study, _ = _simple_study(
            n_per_cluster=10,
            treatment_rates={"a1": 0.6, "a2": 0.1},
            control_rates={"b1": 0.0, "b2": 0.6},
        )
        report = assess_wave(study, 1, 14)
        self.assertGreater(report.estimate.gain, 0)
        self.assertLess(report.estimate.gain_interval.low, 0)
        self.assertEqual(report.decision, EXTEND_OBSERVATION)
        self.assertIn("不得据此自动推荐继续", " ".join(report.decision_reasons))

    def test_missing_above_cap_requires_evidence(self):
        study, _ = _simple_study(
            n_per_cluster=10,
            attended_treatment=1.0,
            missing_treatment=0.5,
        )
        report = assess_wave(study, 1, 14)
        self.assertFalse(report.missing_pass)
        self.assertEqual(report.decision, NEED_MORE_EVIDENCE)

    def test_no_control_cluster_gives_no_comparison(self):
        study, _ = _simple_study(treatment_clusters=("a1", "a2"), control_clusters=())
        report = assess_wave(study, 1, 14)
        self.assertIsNone(report.estimate.gain)
        self.assertEqual(report.decision, NEED_MORE_EVIDENCE)

    def _urban_only_study(self):
        study, _ = _simple_study(
            treatment_rates={"a1": 0.9, "a2": 0.9},
            control_rates={"b1": 0.1, "b2": 0.1},
        )
        residents = {
            key: Resident(item.resident, item.cluster, item.eligible, "urban")
            for key, item in study.residents.items()
        }
        return Study(
            residents=residents,
            allocations=study.allocations,
            sites=study.sites,
            links=study.links,
            daily=study.daily,
            policy=study.policy,
        )

    def test_required_strata_are_a_batch_level_condition(self):
        study = self._urban_only_study()
        wave_one = assess_wave(study, 1, 14)
        self.assertFalse(wave_one.strata_pass)
        self.assertEqual(wave_one.decision, CONTINUE)
        batch = assess_batch(study, 14, waves=(1,))
        self.assertEqual(batch.missing_strata, ("rural",))
        self.assertEqual(batch.decision, NEED_MORE_EVIDENCE)

    def test_batch_with_both_strata_in_both_arms_can_continue(self):
        residents = []
        allocations = []
        sites = []
        daily = []
        for index, cluster in enumerate(("a1", "a2", "b1", "b2")):
            stratum = "urban" if index % 2 == 0 else "rural"
            is_treatment = cluster.startswith("a")
            day = _day(0) if is_treatment else _day(60)
            allocations.append(Allocation(cluster, 1 if is_treatment else 2, day))
            sites.append(Site(f"tx-{cluster}", cluster, day))
            for member in range(10):
                resident = f"{cluster}-r{member:02d}"
                residents.append(Resident(resident, cluster, True, stratum))
                attends = member < (9 if is_treatment else 1)
                for offset in range(14):
                    daily.append(
                        DailyRecord(resident, _day(offset), f"tx-{cluster}", attends, True)
                    )
        study = _study(residents, allocations, sites, [], daily)
        batch = assess_batch(study, 14, waves=(1,))
        self.assertEqual(batch.missing_strata, ())
        self.assertEqual(batch.decision, CONTINUE)


class CorrectionsAndRevisionsTest(unittest.TestCase):
    def setUp(self):
        self.study, _ = _simple_study(
            n_per_cluster=10, attended_treatment=1.0, missing_control=0.0
        )

    def test_correction_changes_evidence_but_not_groups(self):
        baseline = compare_wave(self.study, 1, 14)
        corrected = self.study.with_corrections(
            CorrectionSet(
                reports=tuple(
                    ReportCorrection(
                        f"{cluster}-r{index:02d}",
                        _day(offset),
                        f"tx-{cluster}",
                        False,
                        True,
                        "更正",
                    )
                    for cluster in ("a1", "a2")
                    for index in range(10)
                    for offset in range(14)
                )
            )
        )
        after = compare_wave(corrected, 1, 14)
        self.assertEqual(after.arm.residents, baseline.arm.residents)
        self.assertEqual(after.arm.attended, 0)
        self.assertNotEqual(after.gain, baseline.gain)

    def test_site_period_correction_fixes_valid_attendance_evidence(self):
        resident = Resident("r1", "a1", True, "urban")
        study = _study(
            [resident],
            [Allocation("a1", 1, _day(0))],
            [Site("s1", "a1", _day(5))],
            [],
            [DailyRecord("r1", _day(1), "s1", True, True)],
        )
        self.assertEqual(
            classify_resident(study, resident, _day(0), _day(13)).status, UNKNOWN
        )
        corrected = study.with_corrections(
            CorrectionSet(
                site_periods=(SitePeriodCorrection("s1", open_day=_day(0)),)
            )
        )
        self.assertEqual(
            classify_resident(corrected, resident, _day(0), _day(13)).status,
            ATTENDED,
        )
        self.assertEqual(corrected.residents["r1"].cluster, "a1")

    def test_signed_record_is_immutable_and_supersede_is_linked(self):
        ledger = RecommendationLedger()
        ledger.add_schedule([SessionSlot("s1", 1, _day(80))])
        recommendation = assess_batch(self.study, 14, waves=(1,))
        self.assertEqual(recommendation.decision, CONTINUE)
        original = ledger.issue("batch", recommendation, D(2026, 7, 1), 14)
        digest = original.evidence_digest

        corrections = CorrectionSet(
            reports=tuple(
                ReportCorrection(
                    f"{cluster}-r{index:02d}",
                    _day(offset),
                    f"tx-{cluster}",
                    False,
                    True,
                    "反对证据",
                )
                for cluster in ("a1", "a2")
                for index in range(10)
                for offset in range(14)
            )
        )
        revised = ledger.revise(
            self.study, original.record_id, corrections, D(2026, 7, 5), reason="补报"
        )
        self.assertEqual(ledger.record(original.record_id).evidence_digest, digest)
        self.assertEqual(ledger.record(original.record_id).decision, CONTINUE)
        self.assertEqual(revised.supersedes, original.record_id)
        self.assertNotEqual(revised.evidence_digest, digest)
        self.assertTrue(revised.basis_lost)
        self.assertEqual(
            [item.session_id for item in revised.affected_unstarted], ["s1"]
        )
        self.assertEqual(ledger.schedule[0].status, SUSPENDED)

    def test_suspended_session_needs_new_approval_to_restore(self):
        ledger = RecommendationLedger()
        ledger.add_schedule([SessionSlot("s1", 1, _day(80))])
        recommendation = assess_batch(self.study, 14, waves=(1,))
        original = ledger.issue("batch", recommendation, D(2026, 7, 1), 14)
        ledger._suspend(ledger.schedule, original.record_id)
        self.assertEqual(ledger.schedule[0].status, SUSPENDED)
        with self.assertRaises(ValueError):
            ledger.restore("s1", original)  # same record is not a new decision

        fresh = assess_batch(self.study, 14, waves=(1,))
        new = ledger.issue("batch", fresh, D(2026, 7, 9), 14, record_id="batch-002", revision=2)
        ledger.restore("s1", new)
        self.assertEqual(ledger.schedule[0].status, SCHEDULED)
        with self.assertRaises(KeyError):
            ledger.mark_started("missing")

    def test_non_continue_record_does_not_restore(self):
        ledger = RecommendationLedger()
        ledger.add_schedule([SessionSlot("s1", 1, _day(80))])
        rec = assess_batch(self.study, 14, waves=(1,))
        signed = ledger.issue("batch", rec, D(2026, 7, 1), 14)
        ledger._suspend(ledger.schedule, signed.record_id)
        bad = ledger.issue(
            "batch",
            assess_batch(self.study, 14, waves=(1,)),
            D(2026, 7, 9),
            14,
            record_id="batch-002",
            revision=2,
        )
        suspended = ledger.schedule[0]
        object.__setattr__(bad, "decision", EXTEND_OBSERVATION)
        with self.assertRaises(ValueError):
            ledger.restore("s1", bad)
        self.assertEqual(ledger.schedule[0].status, suspended.status)


class ReportingTest(unittest.TestCase):
    def test_reviewer_report_has_no_resident_identifiers(self):
        study, _ = _simple_study(n_per_cluster=3)
        report = assess_batch(study, 14)
        text = render_reviewer_report(study, report, 14)
        self.assertNotIn("a1-r00", text)
        self.assertIn("分母", text)
        self.assertIn("区间", text)

    def test_audit_trace_is_traceable_by_resident_and_cluster(self):
        study, _ = _simple_study(n_per_cluster=3)
        rows = audit_trace(study, 14, waves=(1,))
        self.assertTrue(rows)
        self.assertTrue(all(row.cluster and row.resident and row.reason for row in rows))
        self.assertTrue(all(row.wave == 1 for row in rows))

    def test_default_window_uses_planned_wave_spacing(self):
        study, _ = _simple_study()
        self.assertEqual(default_follow_up_days(study), 60)


class FixtureRunTest(unittest.TestCase):
    def test_fixture_material_yields_auditable_recommendation(self):
        study = load_study(ROOT / "data")
        report = assess_batch(study, 14)
        # The fixture is deliberately sparse: 36 residents, one day each.
        self.assertEqual(report.decision, NEED_MORE_EVIDENCE)
        self.assertEqual(report.waves[0].estimate.arm.residents, 12)
        # Wave 3 has no not-yet-opened cluster to compare with.
        self.assertIsNone(report.waves[2].estimate.control)
        # The only valid attendance in wave 1 is at the wave-1 site.
        self.assertGreater(report.waves[1].estimate.gain, 0)
        self.assertFalse(report.waves[1].estimate.passes_max_missing)


if __name__ == "__main__":
    unittest.main()
