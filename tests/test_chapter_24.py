"""Numerical, simulation, visual, and CLI tests for Chapter 24."""
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import t, ttest_1samp

from analytics_foundations.chapter_21 import generate_population
from analytics_foundations.chapter_23 import mean_confidence_interval
from analytics_foundations.chapter_24 import (REQUIRED_FIGURES, create_figures,
    one_sample_t_test, outlier_comparison, reject_null, sample_size_signal,
    simulate_rejection_rate)
from analytics_foundations.chapters import get_chapter


def test_manual_statistic_tails_df_and_scipy_agreement():
    values=np.array([9., 13., 17., 21., 25.])
    greater=one_sample_t_test(values,null_mean=14,alternative="greater")
    assert greater.sample_size == 5 and greater.degrees_of_freedom == 4
    assert greater.sample_sd == pytest.approx(values.std(ddof=1))
    assert greater.standard_error == pytest.approx(greater.sample_sd/np.sqrt(5))
    assert greater.effect == pytest.approx(values.mean()-14)
    assert greater.t_statistic == pytest.approx(greater.effect/greater.standard_error)
    assert greater.p_value == pytest.approx(t.sf(greater.t_statistic,4))
    less=one_sample_t_test(values,null_mean=14,alternative="less")
    two=one_sample_t_test(values,null_mean=14,alternative="two-sided")
    assert less.p_value == pytest.approx(t.cdf(less.t_statistic,4))
    assert two.p_value == pytest.approx(2*greater.p_value)
    scipy=ttest_1samp(values,14,alternative="greater")
    assert greater.t_statistic == pytest.approx(scipy.statistic)
    assert greater.p_value == pytest.approx(scipy.pvalue)
    assert all(0 <= r.p_value <= 1 for r in (greater,less,two))


@pytest.mark.parametrize("values", ([1], [1,np.nan], [2,2,2]))
def test_input_validation(values):
    with pytest.raises(ValueError): one_sample_t_test(values,null_mean=1)


def test_alternative_null_and_decision_validation():
    with pytest.raises(ValueError): one_sample_t_test([1,2],null_mean=1,alternative="upper")
    with pytest.raises(ValueError): one_sample_t_test([1,2],null_mean=np.inf)
    assert reject_null(.05,.05) and not reject_null(.051,.05)
    for p,a in ((-1,.05),(.2,0),(.2,1)):
        with pytest.raises(ValueError): reject_null(p,a)


def test_matching_two_sided_ci_consistency():
    for values in (np.arange(10.,20.), np.arange(20.,30.)):
        test=one_sample_t_test(values,null_mean=15,alternative="two-sided")
        ci=mean_confidence_interval(values)
        assert (test.p_value <= .05) == (not ci.lower <= 15 <= ci.upper)


def test_size_signal_and_outlier_demonstration():
    rows=sample_size_signal(); ses=[r[1] for r in rows]; stats=[r[2] for r in rows]
    assert ses == sorted(ses,reverse=True) and stats == sorted(stats)
    before,after=outlier_comparison(np.array([14.,15.,16.,17.,18.]))
    assert after.sample_mean > before.sample_mean and after.sample_sd > before.sample_sd


def test_type_i_power_monotonicity_and_reproducibility():
    rate=simulate_rejection_rate(true_mean=15,repetitions=10000,seed=7)
    assert rate == pytest.approx(.05,abs=.015)
    assert rate == simulate_rejection_rate(true_mean=15,repetitions=10000,seed=7)
    by_size=[simulate_rejection_rate(true_mean=17,sample_size=n,repetitions=6000,seed=10+n) for n in (16,64,256)]
    by_effect=[simulate_rejection_rate(true_mean=m,sample_size=64,repetitions=6000,seed=20) for m in (15.5,16,17,19)]
    assert by_size == sorted(by_size) and by_effect == sorted(by_effect)


def test_figures(tmp_path: Path):
    values=np.random.default_rng(24).normal(17,8,64)
    result=one_sample_t_test(values,null_mean=15,alternative="greater")
    paths=create_figures(result,values,tmp_path)
    assert [p.name for p in paths] == list(REQUIRED_FIGURES)
    assert all(p.read_bytes().startswith(b"\x89PNG") for p in paths)


def test_registration_and_execution(tmp_path: Path,capsys):
    chapter=get_chapter("chapter-24")
    assert chapter and chapter.available and chapter.title == "Hypothesis Testing"
    from analytics_foundations.chapter_24 import run
    assert run(tmp_path) == 0
    output=capsys.readouterr().out
    assert "H0: mu = 15" in output and "Manual/SciPy" in output
    assert "P(H0 | data)" in output and "no group test" in output
    assert len(output.splitlines()) < 20
