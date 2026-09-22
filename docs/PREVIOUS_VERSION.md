# Previous version and evaluation history

The earlier project included multiple feature ablations, histogram gradient boosting, rolling validation and PSI monitoring. The main learning path has been reduced to descriptive statistics, logistic regression, one small decision tree and simple validation-period monitoring.

The complete earlier version remains in [Git history at the published snapshot](https://github.com/Lavicy/fraud-risk-monitoring/tree/c4519cb66e21e7f3aa88f990cfd301e423b0a276). A local copy is also retained under the ignored `.local_history/advanced-2026-09-18/` directory. The smaller version does not require running it.

Two small reference tables are retained in `reports/reference/`: the previous logistic-regression balance comparison and the original fixed model's results. These are copied historical results, not newly executed experiments. The balance comparison used training steps 1–323 and validation steps 324–377. Its full specification included opening balance and amount/balance relationships; its balance-free specification used amount and type alone.

The previous balance-dependent model had historical precision 1.0000 and recall 0.9955. The publisher warns that the recorded balances are affected by simulation cancellation behaviour. Those results are therefore not evidence that the model could have been used at transaction initiation in a real bank. They are not the performance of the simplified model.

Steps 378–743 were already inspected before this revision. The revised models are fitted on steps 1–323 and compared on steps 324–377 only. They do not receive a new test score. The validation period is reused for model choice, threshold choice and monitoring, so all revised results are development evidence with selection optimism. A new independent dataset would be needed for a defensible final evaluation.
