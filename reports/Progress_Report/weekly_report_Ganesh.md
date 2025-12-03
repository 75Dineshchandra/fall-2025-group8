Ganesh Kumar Boini – Personal Weekly Report
Project: Health-Aware Meal Recommendation using Contextual Multi-Armed Bandits (CMAB)
Duration: September 10 – December 3, 2025
 
Date: Week 1 – September 10 – September 17, 2025
Topics of Discussion
•	Understood the project problem: balancing meal popularity with nutrition
•	Studied Contextual Multi-Armed Bandits and LinUCB algorithm
•	Reviewed course project requirements and grading rubric
•	Explored FCPS dataset format and fields
•	Identified variables for modeling (time slots, items, sales, health)
Action Items
•	 Reviewed problem statement and literature
•	 Studied LinUCB and contextual bandits
•	 Reviewed grading rubric and deliverables
•	Explored FCPS data structure
•	Noted initial modeling assumptions
 
Date: Week 2 – September 17 – September 24, 2025
Topics of Discussion
•	Created initial environment pipeline design
•	Added dummy version of env.py
•	Checked initial code into GitHub repository
•	Continued reinforcement learning study
Action Items
•	 Added initial env.py stub
•	 Pushed first commit to repo
•	Verified project folder structure
•	 Reviewed RL fundamentals and materials
 
Date: Week 3 – September 24 – October 1, 2025
Topics of Discussion
•	Designed and implemented nutrition ingestion pipeline from LINQ Connect API
•	Built configurable date range system (calendar year, school year, custom)
•	Implemented robust API handling:
o	Timeout
o	Connection failures
o	Rate limiting
•	Flattened nested API responses into structured tabular format
•	Exported outputs to CSV and JSON
•	Added summary statistics for health checks
Action Items
•	 Integrated LINQ Connect API
Built automated one-year nutrition collection pipeline
•	Implemented error handling and request throttling
•	Parsed menu JSON into flat records
•	Generated nutrition dataset files
 
Date: Week 4 – October 1 – October 8, 2025
Topics of Discussion
•	Implemented Action Matrix generator (time_slots × items)
•	Implemented Feature Matrix generator for LinUCB
•	Added nutrient normalization (z-score)
•	Added cyclical encoding for weekday context
•	Created consistent item_idx mappings
•	Enabled strict vs non-strict item handling
Action Items
•	 Implemented build_action_matrix()
•	Implemented build_feature_matrix()
•	Added feature scaling
•	Built time grouping logic for training
•	Returned metadata for downstream use
 
Date: Week 5 – October 8 – October 15, 2025
Topics of Discussion
•	Refined utility functions
•	Improved health score evaluation logic and added merge logic
•	Improved feature preprocessing
•	Standardized numeric handling and missing values
•	Refactored helper code for clarity and reuse
Action Items
•	 Implemented health scoring utility
•	 Refactored helper functions
•	 Improved NaN handling
•	 Cleaned feature utilities
•	 Improved code structure
 
Date: Week 6 – October 15 – October 22, 2025
Topics of Discussion
•	Implemented Random baseline for benchmarking
•	Implemented oracle reward calculation
•	Added reproducibility via random seed
•	Implemented Top-K support
•	Ensured alignment for evaluation
Action Items
•	Built Random baseline evaluation function
•	 Added oracle computation
•	 Implemented deterministic selection
•	 Logged metrics per run
•	 Verified sanity of output
 
Date: Week 7 – October 22 – October 29, 2025
Topics of Discussion
•	Implemented Health-First baseline
•	Selection based on highest health score
•	Added λ-reward as secondary tie-breaker
•	Implemented oracle comparison
•	Added Top-K capability and seed control
Action Items
•	 Built Health-First evaluation function
•	 Added deterministic selection logic
•	 Implemented regret tracking
•	 Parameterized Top-K and lambda
•	Validated baseline behavior
 
Date: Week 8 – October 29 – November 5, 2025
Topics of Discussion
•	Prepared midterm presentation
•	Documented system architecture
•	Explained data pipeline and modeling
•	Prepared visual explanations
Action Items
•	Created slide deck
•	 Prepared explanation notes
•	Structured diagrams
•	Rehearsed presentation
 
Date: Week 9 – November 5 – November 12, 2025
Topics of Discussion
•	Designed ablation study for LinUCB
•	Evaluated training data fractions from 10–100%
•	Logged metrics per fraction
•	Ensured consistency across runs
Action Items
•	 Implemented ablation loop
•	 Parameterized training fractions
•	 Logged experimental outputs
•	 Verified learning trends
•	Prepared results for tables
 
Date: Week 10 – November 12 – November 19, 2025
Topics of Discussion
•	Fixed Health-First tie-breaking policy
•	Added secondary sorting using λ-reward
•	Eliminated row-order bias
•	Re-ran baseline experiments
Action Items
•	Updated Health-First policy
•	 Enforced deterministic ordering
•	 Revalidated output
•	Updated evaluation results
•	Documented logic
 
Date: Week 11 – November 19 – November 26, 2025
Topics of Discussion
•	Updated research paper
•	Refined Models section
•	Updated Results with metrics
•	Expanded Discussion
Action Items
•	 Revised paper models section
•	 Updated results
•	Refined discussion
•	Aligned narrative and code
•	Finalized academic tone
 
Date: Week 12 – November 26 – December 3, 2025
Topics of Discussion
•	Updated final presentation
•	Recorded final video
•	Revised research paper
Action Items
•	 Updated slides
•	Recorded video
•	Updated paper
•	 Final submissions
•	 Checked consistency

