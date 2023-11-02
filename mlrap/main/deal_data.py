import os
import warnings
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


def load_data(data_path, df=pd.DataFrame([])):
    """
    Load or save data.
    """
    if df.empty:
        if not os.path.exists(data_path):
            raise FileExistsError(
                f"Please check if the input file {data_path} exists."
            )
        df = pd.read_csv(data_path)
        return df
    else:
        df.to_csv(data_path, index=False)


def identify_df(df):
    """
    Identify and separate non data columns in df.
    """
    other_df = df.select_dtypes(exclude=['number'])
    df = df.select_dtypes(include=['number'])
    
    return df, other_df


def embedded_select(model, dataset, series, threshold=10):
    """
    Embedded feature filtering, which filters out features with a given model's \\
    importance greater than the threshold.

    Parameters
    ----------
    model: scikit-learn regressor
    dataset: pd.DataFrame
        Feature set that requires feature filtering.
    series: pd.Series
        Predict target attributes.
    threshold: int or float
        Feature filtering threshold, which can be filtered by importance or ranking.

    Returns
    ----------
    pd.DataFrame Feature subset
    """
    model.fit(dataset, series)

    data={'feature_names': dataset.columns,
          'feature_importance': model.feature_importances_}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    
    if threshold < 0:
        warnings.warn(
            f"\nInvalid threshold {threshold} < 0, descriptors not filtered.", UserWarning
        )

    if type(threshold) == float:
        fi_feature = fi_df[fi_df['feature_importance'] >= threshold]
    elif type(threshold) == int:
        fi_feature = fi_df.iloc[:threshold]
    else:
        fi_feature = pd.DataFrame([])

    if fi_feature.empty:
        raise ValueError(f"The provided threshold {threshold} is out of range, "
                        f"the correct threshold range is: int (0 < threshold); "
                        f"float (0 < threshold < {max(fi_df['feature_importance']):.2f})")

    return dataset[fi_feature['feature_names']]


def correlation_select(dataset, series, threshold=0.75):
    """
    Correlation feature selection.

    Parameters:
    ----------
    dataset: pd.DataFrame
        Feature set that requires feature filtering.
    series: pd.Series
        Predict target attributes.
    threshold: float
        Feature filtering threshold.

    Return: 
    ----------
    pd.DataFrame Feature subset
    """
    unique_counts = dataset.apply(pd.Series.nunique)
    equal_columns = unique_counts[unique_counts == 1].index
    dataset = dataset.drop(equal_columns, axis=1)
    def check_corr(pair):
        i, j = pair
        col1 = dataset.columns[i]
        col2 = dataset.columns[j]
        df = pd.concat([dataset.iloc[:, [i, j]], series], axis=1, join='inner')  
        corr_matrix = df.corr()
        if abs(corr_matrix.iloc[0, 2]) > abs(corr_matrix.iloc[1, 2]):
            return col2
        elif abs(corr_matrix.iloc[0, 2]) == abs(corr_matrix.iloc[1, 2]):
            return col2
        else:
            return col1
    corr_matrix = dataset.corr()
    corr_pairs = [(i, j) for i in range(len(corr_matrix.columns))  
                  for j in range(i+1, len(corr_matrix.columns))
                  if abs(corr_matrix.iloc[i, j]) > threshold]
    col_corr = Parallel(n_jobs=-1)(delayed(check_corr)(pair) for pair in corr_pairs)
    col_corr = set(col_corr)
    return dataset.drop(col_corr, axis=1)


def stratified_split(data, num_bins=5, test_size=0.2, random_state=None):
    """
    Stratified sampling regression dataset.

    Parameters:
    ----------
    data: pd.DataFrame
        A dataset waiting for sampling.
    test_size: float or int
        The proportion or quantity of samples in the test set.

    Return: 
    ----------
    Train-test split of inputs.
    """
    bins_df = pd.cut(data['target'], bins=num_bins, labels=range(num_bins))
    if test_size < 1:
        test_size = test_size * len(data)
    try:
        X_train, X_test, _, _ = train_test_split(data, bins_df, 
            test_size=test_size, stratify=bins_df, random_state=random_state)
    except ValueError:
        raise ValueError(f'The num_bins {num_bins} you set is too large. Please lower the num_bins')

    return X_train, X_test