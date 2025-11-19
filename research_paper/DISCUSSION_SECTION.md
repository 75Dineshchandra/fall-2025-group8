# 7 Discussion

This section interprets the empirical findings presented in the Results section. The analysis synthesizes the patterns observed in the visualizations and tables, focusing on comparative performance across models, temporal learning behavior, and the relationship between reward components. We examine how the contextual bandit approach balances student preference and nutritional quality, and discuss the implications for school meal planning.

## 7.1 Interpretation of Continuous Regret Trends

The continuous regret trajectory (Figure 3) demonstrates a clear learning pattern characterized by a steep decline during the early portion of the sequence, followed by gradual convergence toward the end. The regret percentage decreases from initial values exceeding 70% to approximately 27.6% by the conclusion of training, representing a reduction of over 42 percentage points. This pattern indicates that the LinUCB model successfully adjusts its decision strategy as it observes more contextual interactions, progressively improving its alignment with optimal decision-making.

The rapid decline in regret during the initial phase (approximately the first 30% of time steps) reflects the model's exploration of the action space and its ability to quickly identify high-performing meal items across different contexts. As training progresses, the rate of improvement diminishes, with the regret curve stabilizing in later stages. This stabilization reflects diminishing marginal returns in action-value estimation once sufficient data have been accumulated, consistent with theoretical expectations for contextual bandit algorithms. The convergence to approximately 27.6% regret indicates that the model achieves substantial but not perfect alignment with the oracle policy, leaving room for further improvement through additional exploration or refined feature engineering.

## 7.2 Interpretation of Table 4: Data Fraction Ablation

The data fraction ablation study (Table 4) reveals a consistent decline in regret percentage as the dataset fraction increases from 0.10 to 1.00. The regret percentage decreases from 32.4% at 10% data fraction to 27.3% at full dataset size, representing a 5.1 percentage point improvement. This pattern demonstrates that model performance scales favorably with dataset size, with relative performance improving as more training data becomes available.

The most substantial reductions occur between the 10% and 30% data fractions, where regret percentage drops from 32.4% to 29.2%—a reduction of 3.2 percentage points. This suggests that early model performance is highly sensitive to additional data, as the model benefits significantly from increased exploration opportunities during the initial learning phase. Beyond the 50% data fraction, regret percentage changes become smaller, with improvements of less than 1 percentage point between consecutive fractions. This indicates that the model approaches a performance plateau once a sufficient number of decision points (approximately 10,000-11,000 time steps) have been observed.

The linear growth in total reward and oracle reward with dataset size confirms that regret metrics are comparable across fractions, as both components scale proportionally with the number of decision opportunities. The fact that percentage regret decreases while absolute regret increases reflects the cumulative nature of the metric and demonstrates that the model's relative performance improves even as the total number of decisions grows.

## 7.3 Comparative Analysis Across Models

The multi-panel comparison (Figure 4) highlights distinct behavioral characteristics of each recommendation strategy, revealing fundamental differences in how each approach balances student preference and nutritional quality:

**Random Baseline:**

The Random baseline exhibits flat trajectories across all metrics, consistent with a non-adaptive policy that does not learn from historical data. Reward, sales, and health trajectories remain relatively constant throughout the sequence, with final values of 4.51, 48.33 units, and 4.95, respectively. Regret percentages remain high (60.0%) because selections do not incorporate contextual information or historical performance patterns. This baseline serves as a lower bound, demonstrating the value of any structured decision-making approach.

**Health-First Baseline:**

The Health-First rule-based policy maintains the highest health scores throughout (final value: 5.62), reflecting its design objective of maximizing nutritional quality. However, this comes at a significant cost: the model exhibits lower reward (4.50) and sales performance (38.10 units) due to its emphasis on nutritional quality at the expense of student preference. Its regret trajectory remains close to that of the Random baseline (60.4%), reflecting limited alignment with the oracle benchmark, which considers both popularity and health in its reward formulation. This demonstrates that a purely health-driven approach, while achieving superior nutritional outcomes, fails to balance the dual objectives of the recommendation task.

**LinUCB Contextual Bandit:**

The LinUCB model shows steady increases in reward and sales along with a decreasing regret percentage, indicating that the model successfully adjusts its selection strategy based on observed contextual patterns. Final performance metrics (reward: 8.35, sales: 100.18 units, regret: 27.6%) substantially outperform both baselines. Its health scores (5.24) remain between the two baselines, demonstrating that the model selects items with moderate-to-high nutritional density while also incorporating popularity signals. This balanced approach enables LinUCB to achieve 85.7% higher reward than Health-First and 163% higher sales while maintaining health scores within 7.2% of the Health-First baseline.

Across all metrics, LinUCB exhibits the most dynamic behavior, with clear evidence of adaptation over time. The model's ability to learn context-dependent patterns enables it to identify meals that perform well across different schools, time periods, and contextual conditions, resulting in recommendations that are both appealing to students and nutritionally sound. In contrast, the baseline models remain stable due to their non-learning structures, highlighting the value of adaptive, data-driven decision-making.

## 7.4 Cross-Metric Relationships

Examining reward, regret, sales, and health together reveals several important relationships that illuminate how the contextual bandit framework integrates popularity and health weighting:

**Reward-Sales Relationship:**

Increases in reward correspond strongly with increases in sales, indicating that popularity contributes significantly to the combined reward signal at λ = 0.3. LinUCB's reward trajectory (final: 8.35) closely mirrors its sales trajectory (final: 100.18 units), suggesting that the model learns to prioritize items that are both popular and reasonably healthy. This relationship demonstrates that at the chosen lambda value, student preference remains a primary driver of reward, while health serves as a secondary consideration that guides selection toward more nutritious options.

**Health-Reward Trade-off:**

Health scores for LinUCB (5.24) remain higher than Random (4.95) but lower than Health-First (5.62), suggesting that the reward weighting encourages a balance between nutritional density and student preference. The 7.2% gap between LinUCB and Health-First health scores represents an acceptable trade-off given LinUCB's substantially superior sales and overall reward performance. This pattern indicates that the λ = 0.3 weighting successfully prevents the model from prioritizing popularity exclusively while avoiding the extreme health focus that limits Health-First's effectiveness.

**Regret-Reward Alignment:**

The decreasing regret trajectory aligns inversely with the increasing reward trajectory, demonstrating that improved decision-making corresponds to closer alignment with the oracle action. As LinUCB's regret decreases from over 70% to 27.6%, its reward increases from initial low values to 8.35, confirming that the model progressively learns to make decisions that approximate the optimal policy. This inverse relationship validates the regret metric as a meaningful indicator of model performance and learning effectiveness.

**Temporal Learning Patterns:**

The temporal evolution of all metrics reveals consistent learning behavior: LinUCB shows steady improvement across reward, sales, and regret, while health scores remain relatively stable. This suggests that the model learns to identify popular items without sacrificing nutritional quality, achieving a sustainable balance that improves over time. The stability of health scores indicates that the model maintains its commitment to nutritional objectives even as it optimizes for overall reward.

These cross-metric patterns demonstrate how the contextual bandit framework successfully integrates popularity and health weighting into a unified decision process, enabling the model to balance competing objectives through adaptive learning rather than fixed rules.

## 7.5 Summary of Findings

Overall, the combined analysis of visualizations and tables demonstrates that LinUCB successfully adapts to the sequential structure of the FCPS dataset and achieves performance that shifts progressively toward the oracle benchmark as more data is accumulated. The model's ability to learn context-dependent patterns enables it to identify meals that perform well across different schools, time periods, and contextual conditions, resulting in recommendations that are both appealing to students and nutritionally sound.

The baseline models exhibit stable but limited performance due to their non-contextual or deterministic nature. The Random baseline provides a lower bound, demonstrating that any structured approach improves upon random selection. The Health-First baseline achieves superior nutritional outcomes but fails to balance popularity and health, resulting in lower overall reward and student engagement.

The comparative trends illustrate the critical role of contextual learning in balancing nutritional quality and student engagement within the FCPS environment. LinUCB's superior performance across multiple metrics—achieving 85.7% higher reward, 163% higher sales, and 54.3% lower regret than Health-First while maintaining competitive health scores—demonstrates the value of adaptive, data-driven decision-making for school meal planning.

These findings have practical implications for school nutrition programs: the contextual bandit approach enables cafeteria managers to make evidence-based menu decisions that improve student participation while maintaining nutritional standards. The model's ability to learn from historical data and adapt to contextual patterns provides a scalable framework for optimizing meal offerings across diverse school environments, ultimately supporting both student health and program sustainability.

