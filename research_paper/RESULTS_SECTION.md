# 6 Results

We evaluate the LinUCB contextual bandit model on the FCPS dataset using sequential simulation over the historical timeline. Performance is assessed through cumulative reward, regret relative to the oracle policy, and operational metrics including raw sales and health scores. To smooth the variability inherent in real cafeteria operations, we employ expanding-window rolling averages computed as:

\[
\text{roll metric}_t = \frac{1}{t} \sum_{i=1}^{t} \text{metric}_i
\]

This approach stabilizes early fluctuations and highlights long-term learning trends, providing a clearer view of model performance evolution.

## 6.1 Continuous Regret Calculation

Regret is computed at each time step \(t\) as the difference between the oracle reward (the maximum available reward among all feasible actions) and the actual reward received by the selected action:

\[
\text{regret}_t = \text{oracle}_t - r_t  
\]

where \(\text{oracle}_t = \max_{a \in \mathcal{A}_t} r_{t,a}\) represents the optimal reward achievable at time \(t\), and \(r_t\) is the reward obtained from the action selected by the bandit algorithm. To normalize regret across different reward scales, we also compute percentage regret:

\[
\text{regret\%}_t = \frac{\text{regret}_t}{\text{oracle}_t} \times 100\%
\]

![Figure 3: Continuous Regret Calculation](data/results/figure3_continuous_regret_calculation.png)

**Figure 3: Continuous Regret Calculation.** Rolling average regret percentage over time for LinUCB with \(\lambda = 0.3\). The regret trajectory demonstrates a clear learning pattern: initial regret values exceed 70%, reflecting the model's uncertainty during early exploration, but decrease steadily to approximately 27.6% by the end of training. This reduction indicates that the model successfully learns context-dependent reward patterns and improves decision-making quality through sequential feedback.

The continuous regret calculation enables real-time monitoring of model performance and provides a principled metric for comparing different recommendation strategies. The decreasing trend in regret percentage confirms that LinUCB effectively balances exploration of underutilized items with exploitation of historically successful meals.

## 6.2 Impact of Dataset Size on Performance

To assess how model performance scales with dataset size, we conduct an ablation study training LinUCB on progressively larger fractions of the available data. The dataset is partitioned chronologically, ensuring that each fraction represents a contiguous time period and maintains temporal ordering. Table 4 summarizes the results across data fractions ranging from 0.10 to 1.00 in increments of 0.10.

**Table 4: LinUCB Data Fraction Ablation (chronological)**

| Fraction | Slots | Rows | Total Reward | Oracle Reward | Regret | Regret % |
|----------|-------|------|--------------|---------------|--------|----------|
| 0.10 | 2,166 | 22,277 | 16,918.79 | 25,012.85 | 8,094.06 | 32.4% |
| 0.20 | 4,332 | 44,699 | 34,074.13 | 49,730.29 | 15,656.16 | 31.5% |
| 0.30 | 6,497 | 67,660 | 52,881.29 | 74,733.73 | 21,852.44 | 29.2% |
| 0.40 | 8,663 | 90,413 | 71,320.84 | 99,779.62 | 28,458.78 | 28.5% |
| 0.50 | 10,828 | 112,680 | 89,238.93 | 124,645.65 | 35,406.72 | 28.4% |
| 0.60 | 12,994 | 134,919 | 106,523.76 | 149,227.04 | 42,703.29 | 28.6% |
| 0.70 | 15,160 | 157,636 | 125,271.24 | 174,257.62 | 48,986.38 | 28.1% |
| 0.80 | 17,325 | 180,389 | 143,634.66 | 199,184.84 | 55,550.18 | 27.9% |
| 0.90 | 19,491 | 202,673 | 161,980.02 | 223,722.06 | 61,742.04 | 27.6% |
| 1.00 | 21,656 | 224,536 | 180,809.28 | 248,626.74 | 67,817.46 | 27.3% |

The results reveal several important patterns. As the data fraction increases, both total reward and oracle reward increase proportionally, reflecting the larger number of decision opportunities. Absolute regret also increases with dataset size, which is expected given the cumulative nature of the metric. However, percentage regret demonstrates a decreasing trend, dropping from 32.4% at 10% data fraction to 27.3% at full dataset size. This improvement indicates that the model's relative performance improves as more training data becomes available.

The most substantial improvement in percentage regret occurs between the 10% and 30% data fractions, suggesting that the initial learning phase benefits significantly from additional data. Beyond 50% data fraction, the improvement rate diminishes, indicating that the model approaches a performance plateau. This pattern aligns with theoretical expectations for contextual bandit algorithms, where early exploration is critical for learning effective policies.

## 6.3 Model-Wise Performance Comparison

We compare the LinUCB contextual bandit against two baseline recommendation strategies: (1) a **Random** baseline that selects uniformly at random from available items at each time step, and (2) a **Health-First** rule-based policy that selects the item with the highest health score (NRF9.3) at each decision point, independent of contextual or historical reward signals. All three models are evaluated under identical sequential simulation conditions with \(\lambda = 0.3\), ensuring fair comparison.

![Figure 4: Model-Wise Performance Comparison](data/results/model_comparison_lambda_0.3_rolling4_raw_simplified.png)

**Figure 4: Model-Wise Performance Comparison.** Four-panel comparison showing rolling averages of reward, regret percentage, raw sales, and raw health scores over the full 21,656 time steps. The visualizations employ expanding-window rolling averages to smooth temporal fluctuations and highlight long-term performance trends.

**Panel 1: Rolling Average Reward**  
The reward panel demonstrates that LinUCB achieves the highest final rolling average reward (8.35), substantially outperforming both baselines. Health-First achieves a final reward of 4.50, while Random achieves 4.51. LinUCB's reward trajectory shows consistent improvement over time, indicating effective learning from sequential feedback. In contrast, Random maintains a relatively flat reward profile, as expected for a non-adaptive strategy, while Health-First shows moderate growth but remains constrained by its purely health-driven selection criterion.

**Panel 2: Rolling Average Regret Percentage**  
The regret panel reveals that LinUCB achieves the lowest final regret percentage (27.6%), representing a substantial improvement over both baselines. Health-First achieves 60.4% regret, while Random achieves 60.0% regret. LinUCB's regret trajectory exhibits a clear decreasing trend, confirming that the model learns to make increasingly effective decisions over time. The baselines show minimal improvement, as expected for non-learning strategies. LinUCB achieves a 54.3% relative reduction in regret compared to Health-First (from 60.4% to 27.6%) and a 54.0% relative reduction compared to Random (from 60.0% to 27.6%), demonstrating the value of contextual decision-making and adaptive exploration.

**Panel 3: Rolling Average Raw Sales**  
The sales panel shows that LinUCB achieves the highest final rolling average sales (100.18 units), followed by Random (48.33 units) and Health-First (38.10 units). LinUCB's sales trajectory demonstrates steady growth over time, reflecting the model's ability to learn which items are most popular in different contexts. Health-First's lower sales performance is expected, as it prioritizes nutritional quality over student preference. LinUCB achieves 163% higher sales than Health-First and 107% higher sales than Random, while maintaining competitive health scores, illustrating the effectiveness of the balanced reward formulation.

**Panel 4: Rolling Average Raw Health Scores**  
The health panel shows that Health-First achieves the highest final rolling average health score (5.62), followed by LinUCB (5.24) and Random (4.95). While Health-First prioritizes health maximization by design, LinUCB maintains health scores within 7.2% of Health-First's performance while achieving substantially higher sales and overall reward. This demonstrates that the contextual bandit approach successfully balances nutritional quality with student preference, achieving a favorable trade-off that neither baseline can match.

The comprehensive comparison reveals that LinUCB successfully balances popularity and health objectives, achieving superior overall reward through context-aware decision-making that improves over time. The model's ability to learn from sequential feedback enables it to identify meals that perform well across different schools, time periods, and contextual conditions, resulting in recommendations that are both appealing to students and nutritionally sound.

