"""Comparing independent, paired, and multiple groups for Chapter 25."""
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f, f_oneway, t, ttest_1samp, ttest_ind, ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from analytics_foundations.datasets import PROJECT_ROOT

REQUIRED_FIELDS = frozenset({"observation_id","date","location_id","location_name","wait_minutes","party_size","reservation","day_of_week"})
REQUIRED_FIGURES = tuple(f"chapter-25-{name}.png" for name in ("distributions","mean-cis","difference","anova-decomposition","f-distribution","tukey"))

@dataclass(frozen=True)
class ConfidenceInterval:
    level: float; critical_value: float; margin_of_error: float; lower: float; upper: float

@dataclass(frozen=True)
class TwoSampleResult:
    n1: int; n2: int; mean1: float; mean2: float; difference: float; sd1: float; sd2: float
    standard_error: float; degrees_of_freedom: float; t_statistic: float; p_value: float; confidence_interval: ConfidenceInterval

@dataclass(frozen=True)
class AnovaResult:
    group_sizes: dict[str,int]; group_means: dict[str,float]; grand_mean: float
    ss_between: float; ss_within: float; ss_total: float; df_between: int; df_within: int
    df_total: int; ms_between: float; ms_within: float; f_statistic: float; p_value: float; eta_squared: float

def generate_wait_data(seed: int=2525) -> pd.DataFrame:
    """Create a stable fictional observation-level Friday dinner dataset."""
    rng=np.random.default_rng(seed); specs=(('RF','Riverfront',58,18.4,6.2),('CO','Colonial',43,15.9,3.8),('MI','Midtown',51,13.8,4.6),('HA','Harbor',37,20.3,5.2)); rows=[]
    for code,name,n,center,spread in specs:
        waits=np.clip(rng.normal(center,spread,n),2,42); parties=rng.integers(1,9,n); reserved=rng.random(n)<.48
        for i,(wait,party,res) in enumerate(zip(waits,parties,reserved)):
            date=pd.Timestamp('2026-01-02')+pd.Timedelta(days=7*(i%26))
            rows.append((f'{code}-{i+1:03}',date,code,name,round(float(wait),1),int(party),bool(res),'Friday'))
    return pd.DataFrame(rows,columns=['observation_id','date','location_id','location_name','wait_minutes','party_size','reservation','day_of_week'])

def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Report each group's size, center, spread, range, and quartiles."""
    return df.groupby('location_name',sort=True).agg(n=('wait_minutes','size'),mean=('wait_minutes','mean'),median=('wait_minutes','median'),sd=('wait_minutes','std'),minimum=('wait_minutes','min'),q1=('wait_minutes',lambda x:x.quantile(.25)),q3=('wait_minutes',lambda x:x.quantile(.75)),maximum=('wait_minutes','max'))

def welch_t_test(group_a: np.ndarray, group_b: np.ndarray, confidence: float=.95) -> TwoSampleResult:
    """Manually calculate a two-sided Welch comparison and difference CI."""
    a=np.asarray(group_a,dtype=float).reshape(-1); b=np.asarray(group_b,dtype=float).reshape(-1)
    if min(a.size,b.size)<2 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)): raise ValueError('each group needs at least two finite observations')
    if not 0<confidence<1: raise ValueError('confidence must be between zero and one')
    n1,n2=a.size,b.size; m1,m2=float(a.mean()),float(b.mean()); s1,s2=float(a.std(ddof=1)),float(b.std(ddof=1))
    if s1==0 and s2==0: raise ValueError('at least one group must vary')
    v1,v2=s1*s1/n1,s2*s2/n2; se=float(np.sqrt(v1+v2)); df=float((v1+v2)**2/(v1*v1/(n1-1)+v2*v2/(n2-1)))
    difference=m1-m2; statistic=difference/se; pvalue=float(2*t.sf(abs(statistic),df)); critical=float(t.ppf((1+confidence)/2,df)); margin=critical*se
    ci=ConfidenceInterval(confidence,critical,margin,difference-margin,difference+margin)
    return TwoSampleResult(n1,n2,m1,m2,difference,s1,s2,se,df,float(statistic),pvalue,ci)

def one_way_anova_components(groups: dict[str,np.ndarray]) -> AnovaResult:
    """Expose the classical one-way ANOVA between/within decomposition."""
    if len(groups)<2: raise ValueError('at least two groups are required')
    clean={k:np.asarray(v,dtype=float).reshape(-1) for k,v in groups.items()}
    if any(x.size<2 or not np.all(np.isfinite(x)) for x in clean.values()): raise ValueError('each group needs at least two finite observations')
    sizes={k:int(v.size) for k,v in clean.items()}; means={k:float(v.mean()) for k,v in clean.items()}; all_values=np.concatenate(list(clean.values())); grand=float(all_values.mean())
    ssb=float(sum(sizes[k]*(means[k]-grand)**2 for k in clean)); ssw=float(sum(np.sum((v-means[k])**2) for k,v in clean.items())); sst=float(np.sum((all_values-grand)**2)); n=all_values.size; k=len(clean)
    dfb,dfw=k-1,n-k; msb,msw=ssb/dfb,ssw/dfw; stat=msb/msw; p=float(f.sf(stat,dfb,dfw))
    return AnovaResult(sizes,means,grand,ssb,ssw,sst,dfb,dfw,n-1,msb,msw,float(stat),p,ssb/sst)

def paired_comparison(before: np.ndarray, after: np.ndarray):
    """Return differences and their one-sample test; pairing defines the analysis."""
    before=np.asarray(before,dtype=float); after=np.asarray(after,dtype=float)
    if before.shape != after.shape: raise ValueError('paired arrays must have matching shapes')
    differences=after-before
    return differences,ttest_1samp(differences,0)

def tukey_table(df: pd.DataFrame) -> pd.DataFrame:
    result=pairwise_tukeyhsd(df.wait_minutes,df.location_name)
    return pd.DataFrame(result._results_table.data[1:],columns=result._results_table.data[0])

def _save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig); return path

def create_figures(df: pd.DataFrame, welch: TwoSampleResult, anova: AnovaResult, output_dir: Path) -> list[Path]:
    output_dir=Path(output_dir); order=sorted(anova.group_means); arrays=[df.loc[df.location_name==g,'wait_minutes'] for g in order]; paths=[]
    fig,ax=plt.subplots(figsize=(8,4)); ax.boxplot(arrays,tick_labels=order); ax.set(title='Inspect distributions before testing',ylabel='Wait (minutes)'); paths.append(_save(fig,output_dir/REQUIRED_FIGURES[0]))
    means=np.array([x.mean() for x in arrays]); ses=np.array([x.std(ddof=1)/np.sqrt(len(x)) for x in arrays]); crit=np.array([t.ppf(.975,len(x)-1) for x in arrays]); fig,ax=plt.subplots(figsize=(8,4)); ax.errorbar(order,means,yerr=crit*ses,fmt='o',capsize=6); ax.set(title='95% intervals for each group mean (not differences)',ylabel='Wait (minutes)'); paths.append(_save(fig,output_dir/REQUIRED_FIGURES[1]))
    ci=welch.confidence_interval; fig,ax=plt.subplots(figsize=(8,2.5)); ax.errorbar(welch.difference,0,xerr=ci.margin_of_error,fmt='o',capsize=7); ax.axvline(0,color='C3',ls='--'); ax.set(title='Riverfront − Colonial mean difference (95% CI)',xlabel='Minutes',yticks=[]); paths.append(_save(fig,output_dir/REQUIRED_FIGURES[2]))
    fig,ax=plt.subplots(figsize=(8,4)); ax.scatter(df.location_name,df.wait_minutes,alpha=.25); ax.axhline(anova.grand_mean,color='black',ls='--',label='weighted grand mean'); ax.scatter(order,means,color='C3',zorder=3,label='group means'); ax.set(title='ANOVA separates between- and within-group variation',ylabel='Wait (minutes)'); ax.legend(); paths.append(_save(fig,output_dir/REQUIRED_FIGURES[3]))
    xmax=max(anova.f_statistic*1.15,f.ppf(.999,anova.df_between,anova.df_within)); x=np.linspace(.001,xmax,900); density=f.pdf(x,anova.df_between,anova.df_within); fig,ax=plt.subplots(figsize=(8,4)); ax.plot(x,density); ax.fill_between(x,density,where=x>=anova.f_statistic,alpha=.4); ax.axvline(anova.f_statistic,color='C3',label=f'observed F={anova.f_statistic:.2f}'); ax.set(title='ANOVA F reference distribution: upper-tail p-value',xlabel='F',ylabel='Density'); ax.legend(); paths.append(_save(fig,output_dir/REQUIRED_FIGURES[4]))
    tab=tukey_table(df); y=np.arange(len(tab)); dif=tab.meandiff.astype(float); lo=tab.lower.astype(float); hi=tab.upper.astype(float); labels=tab.group1.astype(str)+' − '+tab.group2.astype(str); fig,ax=plt.subplots(figsize=(8,5)); ax.errorbar(dif,y,xerr=[dif-lo,hi-dif],fmt='o',capsize=4); ax.axvline(0,color='C3',ls='--'); ax.set(yticks=y,yticklabels=labels,title='Tukey-adjusted pairwise 95% intervals',xlabel='Reported mean difference (minutes)'); paths.append(_save(fig,output_dir/REQUIRED_FIGURES[5])); return paths

def run(output_dir: Path|None=None) -> int:
    df=generate_wait_data(); summary=group_summary(df); river=df.loc[df.location_name=='Riverfront','wait_minutes'].to_numpy(); colonial=df.loc[df.location_name=='Colonial','wait_minutes'].to_numpy(); w=welch_t_test(river,colonial); scipy_w=ttest_ind(river,colonial,equal_var=False)
    before=np.array([18,20,14,22,17,25,19,21.]); after=np.array([15,17,13,19,16,21,17,19.]); differences,paired=paired_comparison(before,after); paired_direct=ttest_rel(after,before)
    groups={name:g.wait_minutes.to_numpy() for name,g in df.groupby('location_name')}; a=one_way_anova_components(groups); scipy_a=f_oneway(*groups.values()); paths=create_figures(df,w,a,output_dir or PROJECT_ROOT/'figures'); tukey=tukey_table(df)
    print('Chapter 25 — Comparing Groups'); print('James River Restaurant Group: Friday dinner waits; grain = one customer-party observation. Inspect distributions before tests.')
    print(summary[['n','mean','median','sd','minimum','maximum']].round(2).to_string())
    print('\nTwo-location comparison (Riverfront − Colonial):'); print(f'Estimated difference: {w.difference:.2f} minutes; Welch SE={w.standard_error:.3f}')
    print(f'Welch t({w.degrees_of_freedom:.1f})={w.t_statistic:.3f}; p={w.p_value:.4g}; 95% CI [{w.confidence_interval.lower:.2f}, {w.confidence_interval.upper:.2f}]')
    print(f'SciPy verification: |Δt|={abs(w.t_statistic-scipy_w.statistic):.2g}, |Δp|={abs(w.p_value-scipy_w.pvalue):.2g}. Welch supports unequal n/variance without a preliminary variance test.')
    print(f'Paired scheduling example: mean after−before={differences.mean():.2f}; paired t={paired.statistic:.3f}; ttest_rel agreement={abs(paired.statistic-paired_direct.statistic):.2g}.')
    print('\nAll-location comparison:'); print(f'F({a.df_between}, {a.df_within})={a.f_statistic:.3f}; p={a.p_value:.4g}; eta^2={a.eta_squared:.3f}; SciPy |ΔF|={abs(a.f_statistic-scipy_a.statistic):.2g}')
    print(f'Decomposition: SS_B={a.ss_between:.2f}, SS_W={a.ss_within:.2f}, SS_T={a.ss_total:.2f}; Tukey adjusted follow-ups={len(tukey)}, rejected={int(tukey.reject.sum())}.')
    print('Interpretation: the global result is evidence that not all population means are equal—not that every pair differs. Report raw differences, intervals, p-values, eta-squared, and operational relevance together.')
    print('Limitations: these observational groups require independent, reliable measurements; inspect shape/outliers and group n. Classical ANOVA assumes equal population variances. Association with location is not causation: staffing, volume, party size, and reservation mix may confound it.')
    print(f'Generated {len(paths)} figures. Next question: how can several relevant variables be accounted for at once? That is Chapter 26; no regression is begun here.')
    return 0
