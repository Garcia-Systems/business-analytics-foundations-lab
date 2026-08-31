# Chapter 25 — Comparing Groups

> **Central outcome.** Group comparisons ask whether observed differences in sample means are large relative to the variability and sampling uncertainty within those groups. Welch's *t* test handles a direct comparison between two independent means, paired tests preserve matched designs, and ANOVA generalizes the logic to several groups by comparing between-group variation with within-group variation.

## James River Restaurant Group: pictures before tests

Management asks whether Friday dinner wait times differ among Riverfront, Colonial, Midtown, and Harbor. The response is \(Y=\text{wait time in minutes}\), the grouping variable is \(G=\text{restaurant location}\), and each row is one customer party. Begin with each group's \(n\), mean, median, SD, range, quartiles, and side-by-side distributions. Are centers clearly separated? How much overlap, skew, and variability appear? Are there outliers? **Statistical tests do not replace looking at the data.** Tiny groups make every estimate unstable, so show \(n_g\), not only total \(N\).

The locations have different operational variability and unequal sample sizes. Boxplots make that visible. Mean intervals describe each group mean; their overlap is **not** a formal substitute for directly testing a difference.

## Two independent locations

For Riverfront and Colonial, an observation belongs to one location, not both. Customers at distinct stores, different employees in departments, and orders from distinct channels are likewise independent-group designs. Write

\[
H_0:\mu_R-\mu_C=0\quad(\text{equivalently }\mu_R=\mu_C),\qquad H_A:\mu_R-\mu_C\ne0,
\]

with the directly useful parameter \(\Delta=\mu_R-\mu_C\). If \(\bar x_R=18.4\) and \(\bar x_C=15.9\), then \(18.4-15.9=2.5\) minutes: Riverfront's sample mean is 2.5 minutes longer. Is that large relative to sampling uncertainty?

### Welch's method is the default

For two independent means, both groups contribute uncertainty:

\[
SE(\bar X_1-\bar X_2)=\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}},\qquad
t=\frac{(\bar x_1-\bar x_2)-0}{SE}.
\]

Welch does not impose equal population variances, naturally permits \(n_1\ne n_2\), and does not require a preliminary variance-equality test. With \(\bar x_1=18,\bar x_2=15,s_1=6,n_1=36,s_2=4,n_2=25\), the difference is 3 minutes,

\[
SE=\sqrt{36/36+16/25}=\sqrt{1.64}\approx1.28,
\qquad t=3/1.28\approx2.34.
\]

Software calculates the possibly noninteger Welch degrees of freedom:

\[
\nu=\frac{(s_1^2/n_1+s_2^2/n_2)^2}
{(s_1^2/n_1)^2/(n_1-1)+(s_2^2/n_2)^2/(n_2-1)}.
\]

The implementation is verified with `ttest_ind(group_a, group_b, equal_var=False)`. A \(100(1-\alpha)\%\) interval is

\[
(\bar x_1-\bar x_2)\pm t^*_{\nu}SE.
\]

For example, a 3-minute estimate, \(SE=1.28\), and supplied \(t^*=2.01\) give \(3\pm2.57=[0.43,5.57]\) minutes. The interval estimates the *size* of the population difference. In a matching two-sided 5% test, zero outside the 95% interval means reject equal means; zero inside means do not reject. A 0.4-minute estimate can have \(p<.001\) in a huge sample, but management must still ask whether 24 seconds matters. Report minutes, CI, and p-value.

A one-sided, prespecified question such as “Is Riverfront slower?” uses \(H_A:\mu_R-\mu_C>0\). Tail logic does not change the design. Historically, the pooled test assumes \(\sigma_1^2=\sigma_2^2\), estimates
\(s_p^2=((n_1-1)s_1^2+(n_2-1)s_2^2)/(n_1+n_2-2)\), and uses \(SE=s_p\sqrt{1/n_1+1/n_2}\). **For this lab, use Welch unless equal variance is substantively justified.**

## Paired data preserve the design

The same restaurants before/after a change, the same employees before/after training, matched stores, or the same customers under two conditions are paired. Define \(D_i=X_{after,i}-X_{before,i}\); the paired test is Chapter 24's one-sample test of \(H_0:\mu_D=0\). For before \(18,20,14\) and after \(15,17,13\), differences are \(-3,-3,-1\). In code, `differences = after - before`; `ttest_rel(after, before)` agrees with `ttest_1samp(differences, 0)`. Strong within-unit relationships make pairing valuable. **Study design determines the method**, not column names.

## Four locations: ANOVA intuition

For \(k=4\), ask \(H_0:\mu_1=\mu_2=\mu_3=\mu_4\) versus “at least one population mean differs.” Do not write that every mean must differ. Four groups create \({4\choose2}=6\) pairs (five create 10); many unadjusted tests inflate false-positive risk. A global ANOVA asks whether group means are farther apart than ordinary within-group variation suggests.

With unequal sizes, the grand mean must be weighted:

\[
\bar x_\cdot=\frac{\sum_g n_g\bar x_g}{N}.
\]

Thus groups A (\(n=10,\bar x=20\)) and B (\(n=90,\bar x=30\)) have grand mean \(29\), not 25. This is Chapter 3's weighted-average idea.

### Decomposing variability

\[
SS_T=\sum_g\sum_i(x_{ig}-\bar x_\cdot)^2,
\quad SS_B=\sum_gn_g(\bar x_g-\bar x_\cdot)^2,
\quad SS_W=\sum_g\sum_i(x_{ig}-\bar x_g)^2,
\quad SS_T=SS_B+SS_W.
\]

For A = \(4,5,6\), B = \(7,8,9\), C = \(5,6,7\), group means are 5, 8, 6 and the grand mean is \(19/3\). Direct calculation gives \(SS_B=14\), \(SS_W=6\), and \(SS_T=20\). With \(N=9,k=3\):

\[
df_B=k-1=2,\quad df_W=N-k=6,\quad df_T=N-1=8=2+6,
\]
\[
MS_B=14/2=7,\quad MS_W=6/6=1,\quad F=MS_B/MS_W=7.
\]

Under the null and model assumptions, \(F\) follows an F distribution with \((k-1,N-k)\) degrees of freedom. Similar centers relative to within-group scatter commonly produce \(F\approx1\) (though F may be below 1); greater separation produces larger F. The transparent calculation is verified using `f_oneway`.

Classical one-way ANOVA assumes independent observations, appropriate sampling/assignment, reliable measurement, suitable within-group distributions, and equal population variances. Its p-value is global: a small result supports “not all means are equal,” **not** “every pair differs.” The sample effect size \(\eta^2=SS_B/SS_T\) is the proportion of sample variability associated with the between-group decomposition; it is not causal variance explained.

After a useful global result, adjusted pairwise questions may follow. Tukey HSD reports differences, simultaneous intervals, and adjusted decisions and controls familywise error under its assumptions. A practical introductory workflow is: inspect distributions; ask the global question; if pairwise questions matter, use adjusted follow-ups; report effect sizes and operational relevance. This is useful, not an inviolable law.

## Assumptions and interpretation

- **Independent Welch:** appropriate sampling/assignment; independence within/across groups; reliable measurement; a meaningful mean; and no extreme small-sample skew/outliers. Equal variance is not required.
- **Paired t:** meaningful one-to-one pairs; independent pairs; analyze differences; differences reasonably suitable, especially for small \(n\).
- **Classical ANOVA:** independent observations, appropriate assignment/sampling, reliable measurement, suitable within-group shapes, and equal population variances.

Means alone can miss skew, tails, multimodality, or variance differences. One extreme wait can alter a mean, SD, t, and F. Inspect data rather than conducting ritualistic assumption checks. Statistically significant means 15.1, 15.3, 15.5, and 15.4 differ by only 24 seconds; 11, 15, 19, and 24 minutes tell a different operational story.

Location is observational. Staffing, layout, volume, reservation mix, kitchen capacity, neighborhood, management, and party size can differ systematically. A location association does **not** show location caused the wait difference. Aggregation can even reverse comparisons when composition differs (Simpson's paradox).

## Common misconceptions

1. Different sample means prove population differences. **No:** samples vary naturally.
2. A small p-value proves a large effect. **No:** variability and sample size also matter.
3. Welch requires equal variances. **No:** it avoids that assumption.
4. Paired observations are unrelated groups. **No:** preserve the study's links.
5. Significant ANOVA means every pair differs. **No:** it rejects only the global equal-means claim.
6. Every post-ANOVA pair can use an unadjusted t test. **No:** multiplicity raises false positives.
7. A significant location result proves causation. **No:** observational confounding remains.
8. Overlapping separate group CIs prove no difference. **No:** directly analyze the difference.
9. ANOVA tests variances because of its name. **No:** it uses variance decomposition to test means.

## Mastery checkpoints

1. Classify designs: customers at A versus B (independent); same stores before/after rollout (paired); same employees before/after training (paired); customers randomly assigned landing pages (independent).
2. For \(\bar x_1=52,\bar x_2=48,s_1=10,s_2=8,n_1=25,n_2=36\), find difference \(4\) and Welch \(SE=\sqrt{100/25+64/36}\approx2.40\).
3. A 95% difference CI [1.2, 4.8] excludes zero: reject the matching two-sided 5% null. Does that establish operational importance? No.
4. Why not treat restaurant before/after observations as independent? Links within restaurant preserve design and remove irrelevant between-unit variability.
5. What does ANOVA compare? Between-group variation relative to within-group variation.
6. ANOVA gives \(p=.003\) for four groups. Does every pair differ? No.
7. Find the weighted grand mean for \(n=10,\bar x=20\) and \(n=90,\bar x=30\): 29.
8. Does a longer mean wait prove physical location caused it? No; systematic group differences may confound it.

## Preparation and transition

This chapter prepares for BUAD 512A's two-sample inference and ANOVA, BUAD 512B's transparent pandas/NumPy/SciPy/statsmodels workflows, and BUAD 5112's comparisons of locations, products, segments, and processes. It does not reproduce any William & Mary course.

Riverfront may have longer waits while also serving larger parties, more walk-ins, and busier evenings. Group tests ask, “Are means different?” They do not ask how much location predicts wait after accounting for several relevant variables. A later chapter will take up that next question; this chapter does not begin regression.
