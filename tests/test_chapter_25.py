"""Data, numerical, visual, and CLI tests for Chapter 25."""
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import f_oneway,ttest_1samp,ttest_ind,ttest_rel
from analytics_foundations.chapter_25 import (REQUIRED_FIELDS,REQUIRED_FIGURES,create_figures,generate_wait_data,group_summary,one_way_anova_components,paired_comparison,tukey_table,welch_t_test)
from analytics_foundations.chapters import get_chapter

def test_dataset_fields_sizes_summaries_and_reproducibility():
    df=generate_wait_data(); assert REQUIRED_FIELDS <= set(df); assert df.equals(generate_wait_data())
    summary=group_summary(df); assert summary.n.to_dict()=={'Colonial':43,'Harbor':37,'Midtown':51,'Riverfront':58}
    for name,g in df.groupby('location_name'):
        assert summary.loc[name,'mean']==pytest.approx(g.wait_minutes.mean()); assert summary.loc[name,'sd']==pytest.approx(g.wait_minutes.std())

def test_welch_arithmetic_scipy_ci_and_unequal_inputs():
    a=np.arange(10.,22.); b=np.array([3.,7.,8.,9.,10.,12.,18.]); result=welch_t_test(a,b)
    v1=a.var(ddof=1)/len(a); v2=b.var(ddof=1)/len(b); expected_df=(v1+v2)**2/(v1**2/(len(a)-1)+v2**2/(len(b)-1)); scipy=ttest_ind(a,b,equal_var=False)
    assert result.n1 != result.n2 and result.sd1 != pytest.approx(result.sd2)
    assert result.standard_error==pytest.approx(np.sqrt(v1+v2)); assert result.degrees_of_freedom==pytest.approx(expected_df)
    assert result.t_statistic==pytest.approx(scipy.statistic); assert result.p_value==pytest.approx(scipy.pvalue)
    assert result.confidence_interval.lower < result.difference < result.confidence_interval.upper
    assert (result.p_value<=.05) == (not result.confidence_interval.lower<=0<=result.confidence_interval.upper)

def test_paired_differences_and_scipy_equivalence():
    before=np.array([18.,20.,14.,22.,17.]); after=np.array([15.,17.,13.,19.,16.]); differences,manual=paired_comparison(before,after); related=ttest_rel(after,before)
    assert np.array_equal(differences,after-before); assert manual.statistic==pytest.approx(related.statistic); assert manual.pvalue==pytest.approx(related.pvalue); assert manual.statistic==pytest.approx(ttest_1samp(differences,0).statistic)

def test_anova_tiny_decomposition_df_means_and_scipy():
    groups={'A':np.array([4.,5.,6.]),'B':np.array([7.,8.,9.]),'C':np.array([5.,6.,7.])}; result=one_way_anova_components(groups); scipy=f_oneway(*groups.values())
    assert result.group_means=={'A':5.,'B':8.,'C':6.}; assert result.grand_mean==pytest.approx(19/3)
    assert result.ss_between==pytest.approx(14); assert result.ss_within==pytest.approx(6); assert result.ss_total==pytest.approx(20); assert result.ss_total==pytest.approx(result.ss_between+result.ss_within)
    assert result.df_total==result.df_between+result.df_within; assert result.ms_between==pytest.approx(7); assert result.ms_within==pytest.approx(1); assert result.f_statistic==pytest.approx(7)
    assert result.f_statistic==pytest.approx(scipy.statistic); assert result.p_value==pytest.approx(scipy.pvalue); assert 0<=result.eta_squared<=1

def test_weighted_grand_mean_and_tukey():
    result=one_way_anova_components({'A':np.full(10,20.)+np.linspace(-.1,.1,10),'B':np.full(90,30.)+np.linspace(-.1,.1,90)})
    assert result.grand_mean==pytest.approx(29); table=tukey_table(generate_wait_data()); assert len(table)==6; assert {'meandiff','lower','upper','reject'}<=set(table)

def test_figures(tmp_path: Path):
    df=generate_wait_data(); groups={n:g.wait_minutes.to_numpy() for n,g in df.groupby('location_name')}; w=welch_t_test(groups['Riverfront'],groups['Colonial']); a=one_way_anova_components(groups); paths=create_figures(df,w,a,tmp_path)
    assert [p.name for p in paths]==list(REQUIRED_FIGURES); assert all(p.read_bytes().startswith(b'\x89PNG') for p in paths)

def test_registration_and_execution(tmp_path: Path,capsys):
    chapter=get_chapter('chapter-25'); assert chapter and chapter.available and chapter.title=='Comparing Groups'
    from analytics_foundations.chapter_25 import run
    assert run(tmp_path)==0; output=capsys.readouterr().out
    assert 'Inspect distributions before tests' in output and 'Welch t' in output and 'Paired scheduling' in output and 'All-location' in output and 'not causation' in output and 'no regression' in output
    assert len(output.splitlines())<30
