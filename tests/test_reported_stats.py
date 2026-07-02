"""
Tests for the statcheck-style reported-statistics parser.

These tests verify extraction fidelity, ground-truth p recomputation
against scipy, and — most importantly — the interval-safe verdict
rules: a statistic must never FAIL because of honest rounding of the
printed value.  All inputs are synthetic values constructed to exercise
a specific code path; they do not correspond to any real publication.
"""
import os
import sys

import pytest
from scipy import stats as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from paperscope.analysis.reported_stats import (
    extract_reported_tests, recompute_p, check_reported_tests,
    parse_reported_p, reported_p_verdict,
)
from paperscope.analysis.forensic_report import verdict_from_result
from paperscope.analysis.forensic_stats import effect_size_consistency


# A paragraph containing all five stat types in varied formats:
# Welch t (decimal df), F, unicode chi-squared with N, r with a unicode
# minus and no leading zero, z, and an "ns" report.
PARAGRAPH = (
    "The treatment effect was significant, t(37.4) = 2.10, p = .04, "
    "as was the omnibus test, F(2, 45) = 4.53, p = .016. "
    "Attrition differed by arm, χ²(1, N = 320) = 22.31, p < .001, "
    "and age correlated negatively with score, r(28) = −.42, p = .02. "
    "The final contrast was z=2.58, p=0.01, while the manipulation "
    "check was not significant, t(12) = 1.10, ns."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtraction:

    def test_all_five_stat_types_extracted(self):
        exts = extract_reported_tests(PARAGRAPH)
        assert len(exts) == 6
        assert [e['stat'] for e in exts] == ['t', 'F', 'chi2', 'r', 'z', 't']

    def test_welch_t_with_decimal_df(self):
        e = extract_reported_tests(PARAGRAPH)[0]
        assert e['stat'] == 't'
        assert e['df1'] == pytest.approx(37.4)
        assert e['df2'] is None
        assert e['value'] == pytest.approx(2.10)
        assert e['value_str'] == '2.10'
        assert e['p_op'] == '='
        assert e['p'] == pytest.approx(0.04)
        assert e['p_str'] == '.04'
        assert e['anchor'] == 't(37.4) = 2.10, p = .04'

    def test_f_with_two_dfs(self):
        e = extract_reported_tests(PARAGRAPH)[1]
        assert e['stat'] == 'F'
        assert e['df1'] == 2 and e['df2'] == 45
        assert e['value'] == pytest.approx(4.53)

    def test_unicode_chi_squared_with_n(self):
        e = extract_reported_tests(PARAGRAPH)[2]
        assert e['stat'] == 'chi2'
        assert e['df1'] == 1
        assert e['value'] == pytest.approx(22.31)
        assert e['p_op'] == '<'
        assert e['p'] == pytest.approx(0.001)

    def test_r_with_unicode_minus(self):
        e = extract_reported_tests(PARAGRAPH)[3]
        assert e['stat'] == 'r'
        assert e['df1'] == 28
        assert e['value'] == pytest.approx(-0.42)
        assert e['value_str'] == '−.42'

    def test_z_has_no_df_and_tight_spacing(self):
        e = extract_reported_tests(PARAGRAPH)[4]
        assert e['stat'] == 'z'
        assert e['df1'] is None and e['df2'] is None
        assert e['value'] == pytest.approx(2.58)
        assert e['p_str'] == '0.01'

    def test_ns_report(self):
        e = extract_reported_tests(PARAGRAPH)[5]
        assert e['p_op'] == 'ns'
        assert e['p'] is None
        assert e['anchor'] == 't(12) = 1.10, ns'

    def test_offsets_slice_back_to_anchor(self):
        for e in extract_reported_tests(PARAGRAPH):
            assert PARAGRAPH[e['start']:e['end']] == e['anchor']

    def test_no_stats_yields_empty(self):
        text = ("The mean age was 38.2 years (SD = 4.1) across both "
                "groups, and 62% of participants completed the study.")
        assert extract_reported_tests(text) == []

    def test_near_misses_not_extracted(self):
        # No inferential statistic here: extraction must stay silent.
        text = ("See part 2 of 38 for details; page(3) = 4 lists the "
                "materials, and t-shirts (n = 12) were provided.")
        assert extract_reported_tests(text) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Recomputation ground truth vs scipy
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecomputeP:

    def test_t_two_tailed(self):
        assert recompute_p('t', 3.16, 38) == pytest.approx(
            2 * sp.t.sf(3.16, 38))
        # t(38) = 3.16 -> p ≈ 0.0031
        assert recompute_p('t', 3.16, 38) == pytest.approx(0.0031, abs=2e-4)

    def test_t_sign_invariant(self):
        assert recompute_p('t', -3.16, 38) == recompute_p('t', 3.16, 38)

    def test_f(self):
        assert recompute_p('F', 4.53, 2, 45) == pytest.approx(
            float(sp.f.sf(4.53, 2, 45)))

    def test_chi2(self):
        assert recompute_p('chi2', 22.31, 1) == pytest.approx(
            float(sp.chi2.sf(22.31, 1)))

    def test_r_via_t_conversion(self):
        r, df = 0.42, 28
        t_val = r * (df / (1 - r ** 2)) ** 0.5
        assert recompute_p('r', -r, df) == pytest.approx(
            2 * sp.t.sf(t_val, df))

    def test_z(self):
        assert recompute_p('z', 2.58) == pytest.approx(
            2 * sp.norm.sf(2.58))

    def test_impossible_inputs_raise_clean_errors(self):
        with pytest.raises(ValueError):
            recompute_p('r', 1.0, 28)       # |r| >= 1
        with pytest.raises(ValueError):
            recompute_p('t', 2.0, 0)        # df <= 0
        with pytest.raises(ValueError):
            recompute_p('F', -1.0, 2, 45)   # negative F
        with pytest.raises(ValueError):
            recompute_p('bogus', 1.0, 2)    # unknown statistic


# ═══════════════════════════════════════════════════════════════════════════════
# Verdicts (interval-safe statcheck semantics)
# ═══════════════════════════════════════════════════════════════════════════════

def _one_finding(text):
    report = check_reported_tests(text, source='test')
    assert len(report['findings']) == 1
    return report['findings'][0]


class TestVerdicts:

    def test_consistent_exact_p_passes(self):
        # t(38) = 3.16 -> two-tailed p ≈ 0.0031, consistent with "p = .003"
        f = _one_finding("t(38) = 3.16, p = .003")
        assert f['verdict'] == 'PASS'

    def test_reporting_error_flags_not_fails(self):
        # p ≈ .003 reported as .03: wrong, but both are significant at .05
        f = _one_finding("t(38) = 3.16, p = .03")
        assert f['verdict'] == 'FLAG'

    def test_decision_error_fails(self):
        # t(38) = 1.00 -> p ≈ .32 (not significant), reported p = .003
        f = _one_finding("t(38) = 1.00, p = .003")
        assert f['verdict'] == 'FAIL'
        assert 'decision error' in f['detail'].lower()

    def test_p_less_than_consistent_on_rounding_interval(self):
        # Borderline case: the .05 critical t for df=38 is ~2.0244, so the
        # point value t=2.02 gives p > .05 (would falsely fail), but the
        # printed "2.02" means t ∈ [2.015, 2.025] and the top of that
        # interval gives p < .05 — must PASS.
        assert 2 * sp.t.sf(2.02, 38) > 0.05     # the point-value trap
        assert 2 * sp.t.sf(2.025, 38) < 0.05    # the interval rescue
        f = _one_finding("t(38) = 2.02, p < .05")
        assert f['verdict'] == 'PASS'
        assert f['inputs']['recomputed_p_range'][0] < 0.05

    def test_one_tailed_rescue_flags_with_explanation(self):
        # t(30) = 1.80: two-tailed p ≈ .082, one-tailed ≈ .041 — the
        # reported p = .04 only matches one-tailed, so FLAG, never FAIL.
        f = _one_finding("t(30) = 1.80, p = .04")
        assert f['verdict'] == 'FLAG'
        assert 'one-tailed' in f['detail'].lower()

    def test_ns_consistent_passes(self):
        f = _one_finding("t(12) = 1.10, ns")
        assert f['verdict'] == 'PASS'

    def test_ns_with_significant_stat_is_decision_error(self):
        # p ≈ .003 reported as "ns" flips the significance decision
        f = _one_finding("t(38) = 3.16, ns")
        assert f['verdict'] == 'FAIL'

    def test_impossible_r_fails(self):
        f = _one_finding("r(28) = 1.05, p < .001")
        assert f['verdict'] == 'FAIL'
        assert 'impossible' in f['detail'].lower()

    def test_negative_f_fails(self):
        f = _one_finding("F(2, 45) = −4.53, p = .016")
        assert f['verdict'] == 'FAIL'

    def test_zero_df_fails(self):
        f = _one_finding("t(0) = 2.10, p = .04")
        assert f['verdict'] == 'FAIL'


# ═══════════════════════════════════════════════════════════════════════════════
# Report shape (shared Finding/Report contract)
# ═══════════════════════════════════════════════════════════════════════════════

class TestReportShape:

    def test_report_contract(self):
        report = check_reported_tests(PARAGRAPH, source='paper.txt')
        assert report['source'] == 'paper.txt'
        assert report['mode'] == 'text'
        assert set(report['counts']) == {'PASS', 'FLAG', 'FAIL',
                                         'UNDETERMINED'}
        assert sum(report['counts'].values()) == len(report['findings'])
        assert isinstance(report['summary'], str)

    def test_finding_contract(self):
        report = check_reported_tests(PARAGRAPH, source='paper.txt')
        for f in report['findings']:
            assert f['check'] == 'p_recalculation'
            assert f['verdict'] in ('PASS', 'FLAG', 'FAIL', 'UNDETERMINED')
            assert f['anchor'] in PARAGRAPH
            assert f['ref']
            for key in ('stat', 'value', 'dfs', 'reported_p',
                        'recomputed_p_range'):
                assert key in f['inputs']

    def test_empty_text_yields_empty_report(self):
        report = check_reported_tests("No statistics here.", source='x')
        assert report['findings'] == []
        assert sum(report['counts'].values()) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Shared verdict-mapping helper (forensic_report)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerdictFromResult:

    def test_possible_tristate(self):
        assert verdict_from_result({'possible': True}) == 'PASS'
        assert verdict_from_result({'possible': False}) == 'FAIL'
        assert verdict_from_result({'possible': None}) == 'UNDETERMINED'
        assert verdict_from_result({'plausible': None}) == 'UNDETERMINED'

    def test_flags_list(self):
        assert verdict_from_result({'flags': []}) == 'PASS'
        assert verdict_from_result({'flags': ['SD is odd']}) == 'FLAG'
        assert verdict_from_result(
            {'flags': ['SD exceeds maximum — IMPOSSIBLE']}) == 'FAIL'

    def test_carlisle_suspicious(self):
        assert verdict_from_result({'suspicious': True,
                                    'flags': ['x']}) == 'FLAG'
        assert verdict_from_result({'suspicious': False,
                                    'flags': []}) == 'PASS'

    def test_unknown_shape_softens(self):
        assert verdict_from_result({'detail': 'SKIP: whatever'}) == \
            'UNDETERMINED'


class TestCodexReviewFalseAccusations:
    """Round-2 adversarial review (Codex): three false-accusation paths.

    A forensic tool must never brand correct reporting an error; each of
    these was a FAIL that should not have been.
    """

    def test_r_exactly_one_is_not_impossible(self):
        # r printed "1.00" is a perfect/near-perfect correlation rounded
        # from [0.995, 1.005], not arithmetically impossible.
        r = check_reported_tests("r(24) = 1.00, p < .001")
        assert r["findings"][0]["verdict"] != "FAIL"

    def test_r_clearly_above_one_still_fails(self):
        r = check_reported_tests("r(24) = 1.07, p < .001")
        assert r["findings"][0]["verdict"] == "FAIL"

    def test_r_one_is_no_longer_impossible_as_printed(self):
        # r = 1.00 with an inconsistent p is a legitimate decision-error
        # FAIL, but must NOT be branded "impossible as printed" — a
        # perfect correlation is possible, its reported p just disagrees
        f = check_reported_tests("r(24) = 1.00, p = .50")["findings"][0]
        assert "impossible as printed" not in f["detail"]
        assert "decision error" in f["detail"]

    def test_p_above_one_is_undetermined_not_fail(self):
        # p = 2 is a garbled probability (typo), not a decision error
        r = check_reported_tests("z = 3, p = 2")
        assert r["findings"][0]["verdict"] == "UNDETERMINED"

    def test_negative_p_is_undetermined(self):
        r = check_reported_tests("t(30) = 2.0, p = -.03")
        f = [x for x in r["findings"]]
        if f:
            assert f[0]["verdict"] == "UNDETERMINED"


# ═══════════════════════════════════════════════════════════════════════════════
# Shared table-mode p semantics: parse_reported_p + reported_p_verdict (bug 5)
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseReportedP:
    """The reported p may be a bare float (back-compat) or a string carrying
    an operator and precision; parsing must recover operator, value, and the
    printed digits."""

    def test_float_is_equality_claim(self):
        assert parse_reported_p(0.03) == ('=', 0.03, '0.03')

    def test_less_than_operator(self):
        op, val, s = parse_reported_p("<0.001")
        assert op == '<' and val == pytest.approx(0.001) and s == '0.001'

    def test_leading_p_and_no_leading_zero(self):
        op, val, s = parse_reported_p("p < .001")
        assert op == '<' and val == pytest.approx(0.001)

    def test_greater_than(self):
        op, val, _ = parse_reported_p("> .05")
        assert op == '>' and val == pytest.approx(0.05)

    def test_ns_token(self):
        op, val, _ = parse_reported_p("ns")
        assert op == 'ns' and val is None

    def test_none_is_empty(self):
        assert parse_reported_p(None) == ('=', None, '')


class TestReportedPVerdict:
    """The shared consistency judgement routed through by the table-mode
    recomputation checks (t-test / ANOVA / chi-squared)."""

    def test_less_than_0001_consistent_passes(self):
        """A recomputed p far below the reported '<0.001' is consistent — the
        operator is honoured, not treated as an exact equality (which a crude
        ratio band would have mis-flagged)."""
        v = reported_p_verdict(2.9e-11, "<0.001")
        assert v['verdict'] == 'PASS'
        assert v['reported_op'] == '<'
        assert not v['decision_flip']

    def test_less_than_violated_flags(self):
        """A recomputed p ABOVE the '<0.001' bound is inconsistent."""
        v = reported_p_verdict(0.02, "<0.001")
        assert v['verdict'] == 'FLAG'

    def test_near_alpha_decision_flip_flags(self):
        """Bug 5 pattern (chi-squared): reported significant p=0.03,
        recomputed 0.081 — both sides of alpha=.05, so the significance
        decision flips.  The old ratio band accepted 0.081/0.03 = 2.7 and
        missed it; the shared machinery must FLAG the flip."""
        v = reported_p_verdict(0.081, "0.03")
        assert v['verdict'] == 'FLAG'
        assert v['decision_flip']

    def test_equal_within_rounding_passes(self):
        """A recomputed p close to the reported one (same side of alpha) is a
        PASS — recomputation from rounded summary stats carries its own noise."""
        v = reported_p_verdict(0.019, "0.02")
        assert v['verdict'] == 'PASS'
        assert not v['decision_flip']

    def test_float_backcompat(self):
        """A float reported_p still works (equality claim)."""
        v = reported_p_verdict(0.081, 0.03)
        assert v['decision_flip'] and v['verdict'] == 'FLAG'


# ═══════════════════════════════════════════════════════════════════════════════
# Operator coverage + crash-proofing (adversarial review A6, B4, B7)
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseReportedPOperators:
    """'<=', '>=', '≤', '≥' are real reported forms (inclusive bounds) and
    must parse; trailing significance markers (*, **, †) must be stripped."""

    def test_less_equal_ascii(self):
        op, val, s = parse_reported_p("<=0.05")
        assert op == '<=' and val == pytest.approx(0.05)

    def test_greater_equal_ascii(self):
        op, val, _ = parse_reported_p(">= .05")
        assert op == '>=' and val == pytest.approx(0.05)

    def test_less_equal_unicode(self):
        op, val, _ = parse_reported_p("≤0.05")
        assert op == '<=' and val == pytest.approx(0.05)

    def test_greater_equal_unicode(self):
        op, val, _ = parse_reported_p("p ≥ 0.05")
        assert op == '>=' and val == pytest.approx(0.05)

    def test_significance_stars_stripped(self):
        assert parse_reported_p("0.03*")[:2] == ('=', 0.03)
        assert parse_reported_p("0.001**")[:2] == ('=', 0.001)
        assert parse_reported_p("<0.05 †")[:2] == ('<', 0.05)


class TestReportedPVerdictCrashProof:
    """A crash on any string input is never acceptable: malformed input
    yields UNDETERMINED, not an exception (and never a PASS that quietly
    skipped the check on a parse failure)."""

    @pytest.mark.parametrize("junk", [
        "<=0.05", "abc", "p<", "≥", "", "0.03;0.04", "NaN-ish", "< or =",
    ])
    def test_malformed_p_calc_is_undetermined(self, junk):
        v = reported_p_verdict(junk, 0.05)
        assert v['verdict'] == 'UNDETERMINED'

    @pytest.mark.parametrize("junk", ["abc", "p<", "< or =", "?"])
    def test_unparseable_reported_string_is_undetermined(self, junk):
        v = reported_p_verdict(0.04, junk)
        assert v['verdict'] == 'UNDETERMINED'

    def test_inclusive_bound_operators_judged(self):
        assert reported_p_verdict(0.001, "<=0.05")['verdict'] == 'PASS'
        assert reported_p_verdict(0.5, "≥0.05")['verdict'] == 'PASS'
        assert reported_p_verdict(0.5, "<=0.05")['verdict'] == 'FLAG'


class TestReportedPZeroInterval:
    """SPSS prints tiny ps as '0.000': that is the rounding interval
    [0, 0.0005), not an exact zero to feed a ratio band."""

    def test_recomputed_0004_passes(self):
        assert reported_p_verdict(0.0004, "0.000")['verdict'] == 'PASS'

    def test_recomputed_002_does_not_pass(self):
        assert reported_p_verdict(0.002, "0.000")['verdict'] != 'PASS'

    def test_equals_prefix_form(self):
        assert reported_p_verdict(0.0004, "=0.000")['verdict'] == 'PASS'

    def test_exact_zero_recomputed_passes(self):
        assert reported_p_verdict(0.0, "0.000")['verdict'] == 'PASS'


class TestLessThanBoundarySlack:
    """B4: '<0.05' with a recomputed p barely over the bound (0.0501 from
    rounded inputs) must not be branded a decision flip while '=0.05' with
    the same recomputed value PASSes; the calc-side alpha comparison gets
    the same half-ulp slack the consistency check already grants."""

    def test_barely_over_bound_passes(self):
        v = reported_p_verdict(0.0501, "<0.05")
        assert v['verdict'] == 'PASS', v
        assert not v['decision_flip']

    def test_clearly_over_bound_still_flags(self):
        v = reported_p_verdict(0.09, "<0.05")
        assert v['verdict'] == 'FLAG'

    def test_genuine_flip_still_flags(self):
        v = reported_p_verdict(0.081, "0.03")
        assert v['verdict'] == 'FLAG' and v['decision_flip']


class TestEffectSizeConsistencyReportedP:
    """B5: effect_size_consistency must route reported_p through the shared
    machinery — string operators work, and no string input can crash it."""

    def _stats(self):
        # two clearly different groups: Welch p ~ 2e-4
        return dict(mean1=10.0, sd1=2.0, n1=30, mean2=12.0, sd2=2.0, n2=30)

    def test_string_p_consistent(self):
        r = effect_size_consistency(**self._stats(), reported_p='<0.001')
        assert r['consistent'], r['detail']

    def test_string_p_equals_form(self):
        r = effect_size_consistency(**self._stats(), reported_p='0.0002')
        assert r['consistent'], r['detail']

    def test_string_p_inconsistent_flags(self):
        r = effect_size_consistency(**self._stats(), reported_p='0.9')
        assert not r['consistent']

    def test_malformed_string_p_does_not_crash_or_flag(self):
        r = effect_size_consistency(**self._stats(), reported_p='p<')
        assert r['consistent']  # unparseable -> cannot check -> no accusation
