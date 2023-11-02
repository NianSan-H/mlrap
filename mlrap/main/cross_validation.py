import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._validation import (_check_multimetric_scoring,
                                                 _fit_and_score, check_scoring)
from sklearn.utils.parallel import Parallel, delayed


def stratified_cv(model, X, y=None, scoring="r2", cv=5, n_jobs=None, verbose=0, fit_params=None, 
                  pre_dispatch="2*n_jobs", shuffle=False, random_state=None, num_bins=None):
    """
    Implement stratified cross validation of regression tasks.
    """
    if type(cv) is int:
        cv = StratifiedKFold(n_splits=cv, shuffle=shuffle, 
                                  random_state=random_state)
        
    y.columns = ['target']
    data = pd.concat([X, y], axis=1)
    if num_bins is None:
        num_bins = cv.n_splits
    bins_df = pd.cut(data['target'], bins=num_bins, labels=range(num_bins))

    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(model, scoring)
    else:
        scorers = _check_multimetric_scoring(model, scoring)

    indices = cv.split(X, bins_df)
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(_fit_and_score)(
            clone(model), X, y, scorers, train,
            test, verbose, None, fit_params
        )
        for train, test in indices
    )
    return [entry['test_scores'] for entry in results]