Overview
Help science build better Parkinson's evaluation
For each visit, predict the unbiased (“true”) OFF MDS-UPDRS motor score. Minimize RMSE.

The competition runs only during the workshop: 7 October 2026, 09:30–17:00 (Paris time, CEST).

Work in Bob (IDE or CLI) with skore. A Submission is valid only if the Kaggle Submission Description contains the corresponding Skore Hub EstimatorReport URL.

Start

18 hours ago
Close

7 days to go
Description
Parkinson's Disease: Predicting and Correcting Bias in Motor Score Evaluation
Private guided lab / Kaggle prediction competition bobathon-esilv (Probabl × IBM). Competitive regression: for each visit, predict the unbiased (“true”) OFF MDS-UPDRS motor score. Minimize RMSE on the unlabeled test set.

The competition runs only during the workshop: 7 October 2026, 09:30–17:00 (Paris time, CEST). Entry, team mergers, and Submissions are valid only in that window. Final deadline: 17:00.

Scientific background and modelling pitfalls: docs/CONTEXT.md in the lab repo.
Setup (Kaggle team, Python, Skore Hub workspace, skore and Bob skills): docs/SETUP.md.
Guided techniques: docs/DAY1.md (data, EDA, dummy baseline) and docs/DAY2.md (models, patient-grouped cross-validation, Submissions).
Eligibility and Submission validity: the Rules tab of this competition.
Goal
Observed ON/OFF MDS-UPDRS motor scores are biased (rater subjectivity, missing values, levodopa timing, OFF exams often skipped). The target is a true OFF score estimated by removing those biases. It is not available in real care. Predict it for every test visit.

Toolchain (mandatory)
All modelling, evaluation, and the file you upload to Kaggle must be done with:

Bob IDE and/or Bob CLI as the only coding agent.
skore for evaluation, reports, and experiment tracking, with reports pushed to your team's Skore Hub workspace on skore.probabl.ai.
The lab skore skills, installed with skore skills install all --repo probabl-ai/skills-hackathon --agent bob (see docs/SETUP.md).
Cursor, Claude Code, Copilot, ChatGPT, and Kaggle Notebooks are not allowed as the coding agent. A CSV produced outside Bob + skore is not a valid Submission.

Data
The competition files are in the lab git repo.

File	Role
X_train.csv	Training features (44,590 visits, 5,576 patients)
y_train.csv	Training target: true OFF (Index,target)
X_test.csv	Test features, no labels (11,013 visits, 1,395 patients)
sample_submission.csv	Shape template (Index,target)
Test patients do not overlap train patients (holdout by patient_id). Test labels are not distributed. Each patient has several visits, so evaluate with patient-grouped cross-validation (GroupKFold on patient_id). A random row split leaks patients and overestimates your score.

Columns: Index (submission key), patient_id, cohort, sexM, gene, age_at_diagnosis, age, ledd, time_since_intake_on, time_since_intake_off, rater_id, on, off. Years since diagnosis is not given; derive it as age - age_at_diagnosis. on, off, ledd, gene, and the intake delays are often missing, and that missingness is informative.

Metric
Visit-level RMSE. Lower is better.

RMSE(y, ŷ) = √[ (1 / N) Σᵢ (yᵢ − ŷᵢ)² ]

The public leaderboard is computed on a patient-grouped slice of the test set. The private leaderboard at the deadline is official. As a reference, a dummy mean predictor (Day 1) sets the floor every model must beat.

Submission format
CSV with header Index,target, one row per test visit. Index is the visit id from X_test.csv.
Unlimited Submissions during the event. Each team designates one Submission as its scored leaderboard entry.
A Submission is valid only if it was produced with the allowed toolchain and the Kaggle Submission Description contains the URL of the corresponding Skore Hub EstimatorReport (https://skore.probabl.ai/…, printed when you put the report). Use a new report key for each Submission (01_dummy, 02_ridge, …). A scored file without that URL may be removed.
Teams
Maximum four people per team. Create one Skore Hub workspace per team, named after your Kaggle team.

Evaluation
Evaluation
Regression, RMSE on visits. Lower is better.

RMSE(y, ŷ) = √[ (1 / N) Σᵢ (yᵢ − ŷᵢ)² ]

yᵢ — true OFF at visit i
ŷᵢ — prediction at visit i
Submissions must be a CSV with columns Index,target, one row per test visit. Index is the visit id from X_test.csv. sample_submission.csv is the shape template.

Unlimited submissions during the event. Each team designates one Submission as its scored leaderboard entry.

The public leaderboard is a patient-grouped slice of the test set. The private leaderboard (used at the deadline) is the remaining patients.

A Submission is scored by Kaggle only if the file is well-formed. It is valid only if the Submission Description contains the URL of the corresponding EstimatorReport on the lab Skore Hub (https://ibm.skore.probabl.ai/…). We may remove a scored row that has no such URL.