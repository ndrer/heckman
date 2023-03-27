# heckman
Heckman implementation from bert9bert's[1] unmerged branch for statsmodels[2], with additional wrapper for regularized fit (elastic net, ridge, lasso, and square-root lasso) inspired by fabmarcher's Stata package[3].
Plus a function to call shorter result summary as pandas DataFrame and Henze-Zirkler multivariate normality test to see whether the residuals from the first (Probit) and second (OLS) stage of Heckman regression are jointly normal.

## Reference:
1) https://github.com/bert9bert/
2) https://github.com/statsmodels/statsmodels/blob/92ea62232fd63c7b60c60bee4517ab3711d906e3/statsmodels/regression/heckman.py
3) https://github.com/farbmacher/heckman_lasso