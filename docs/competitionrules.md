Official Competition Rules — IBM × Probabl Hackaton
Entry in this Competition constitutes acceptance of these Rules and of Kaggle’s foundational-rules on the Competition Website.

This is a skills-based, private lab. Register on the Competition Website to enter. A Submission is an agreement to these Rules.

You may not enter or submit from more than one Kaggle account.

1. Competition terms
Title	IBM × Probabl Hackaton
Winner license	Non-exclusive. A winner grants us a worldwide, non-exclusive, royalty-free, perpetual right to use, reproduce, and display the winning Submission and the source used to produce it, for reporting, teaching, and communication about this event. You keep ownership of your code.
Data use	Competition use only. You may use the Competition Data solely to participate in this Competition and related lab forums. No other transmission, republication, or commercial use.
Data license	Subject to these Rules.
Any small prize we may hand out at the event is announced there; it is not part of these Rules and is not listed on Kaggle.

2. Timeline and teams
The Competition runs only during the workshop: 29 September 2026, 09:30–17:00 (Paris time, CEST). Entry, team mergers, and Submissions are valid only in that window. The final Submission deadline is 17:00.
Maximum team size: four (4).
3. Submissions
Score: visit-level RMSE between your target and the hidden true OFF. Lower is better.
File: CSV with header Index,target, one row per test visit. Index is the visit id from X_test.csv. Use sample_submission.csv as the shape template.
Unlimited Submissions during the event. Each team may designate one (1) Submission as its scored leaderboard entry. Only that row counts.
The public leaderboard is a patient-grouped slice of the test set. The private leaderboard at the deadline is official.
A Submission is valid only if all of the following hold:

It was produced with the Allowed toolchain in §4. A CSV written by hand, by another agent, by a Kaggle Notebook, or by a script run outside Bob + skore is not a valid Submission.
It corresponds to one EstimatorReport on Skore Hub.
The Kaggle Submission Description contains the URL of that EstimatorReport (https://ibm.skore.probabl.ai/…).
A Submission whose Description has no EstimatorReport URL, or whose URL does not open that report on the hub, is invalid and may be removed even if Kaggle scored the file.

4. Allowed toolchain (mandatory)
All modelling, evaluation, and the file you upload to Kaggle must be done with this stack and nothing else:

Bob IDE and/or Bob CLI (bob / bobide) as the only coding agent and the only environment in which experiment code is written or run for a Submission.
skore for evaluation, reports, and experiment tracking (skore.evaluate, skore.Project, and skore-cli as specified in the lab setup).
The lab skore skills, install with skore skills ....
Allowed as libraries inside that stack: the Python packages the lab environment installs (including scikit-learn, skrub, pandas, and skore). They may be used only from code authored and executed through Bob, and evaluation/reporting must go through skore.

Forbidden (non-exhaustive):

Any other coding agent or IDE agent: Cursor, Claude Code, GitHub Copilot Chat, ChatGPT, Gemini CLI, OpenCode, Windsurf, Cline, Codex, and equivalents.
Kaggle Notebooks (or any other notebook host) to train, predict, or build the Submission file.
Another experiment tracker as the system of record (MLflow, W&B, ad-hoc metric printouts) instead of skore.
AutoML / external training APIs that bypass Bob + skore.
Copying a Submission or pipeline from another team.
We may require the EstimatorReport, the experiment journal, and evidence that the run was issued from Bob. A leaderboard row without that evidence may be removed.

5. Competition Data
Use only the files on the Competition Website Data tab: X_train.csv, y_train.csv, X_test.csv, sample_submission.csv. Download them from Kaggle into your workspace.
Test labels are not provided. Do not use another copy of this table or any external dataset to train, tune, or score.
No pretrained models on this same task, and no scraped labels.
Keep the data accessible only to teammates who accepted these Rules. Notify us if it leaks.
6. External tools and data
Section 4 overrides the usual “external data and tools are OK if cheap and public” default. No external data. No extra agents. Public documentation (library docs, papers) may be read; they must not be used as a side channel to another agent writing the Submission.

7. Conduct and enforcement
No multiple accounts, no private sharing of code or predictions across teams.
We may inspect code, Skore Hub EstimatorReports, Bob history, and Kaggle Submission Descriptions; may rescore or remove Submissions; and may disqualify a team for a toolchain, Description, or data violation even after the deadline.
Our decision on eligibility and ranking is final for this event.
8. Winner obligations
A winning team must provide, within a reasonable time:

the Bob-authored source that produced the designated Submission;
the Skore Hub EstimatorReport linked in that Submission’s Description;
enough instructions to reproduce the Index,target file from the Competition Data.
Kaggle’s foundational-rules apply to everything not stated here. If they conflict with §3–§6, these Competition-Specific Rules control.