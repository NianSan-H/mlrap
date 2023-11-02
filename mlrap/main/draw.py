import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error


def plot_corr(dataset, figure_size: tuple=(12,8), title=None, cmap=None, mask=None, show=True):
    """
    Draw a heat map of Pearson correlation coefficients.

    Parameters
    ----------
    dataset: pd.DataFrame
        A dataset that requires a heat map to be drawn.
    title: str
        Figure name.
    """
    corr = dataset.corr(numeric_only=True)
    if cmap is None:
        cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
    if mask is None:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1) 
    plt.figure(figsize=figure_size)
    plt.title(title)
    sns.heatmap(corr, mask=mask, center=0, annot=True, fmt='.2f', square=True, cmap=cmap)
    if show:
        plt.show()


def plot_model(dict_data, min_value=None, max_value=None, 
               scores: dict={'R\u00b2': r2_score,
                             'RMSE': lambda true, pred: np.sqrt(mean_squared_error(true, pred))}, 
               xlabel='True Data', ylabel='Predicted Data', title=None, show=True):
    """
    Draw Prediction-True to evaluate model performance.

    Parameters
    ----------
    dict_data: dict
        A dictionary containing True-Prediction data.
    min_value & max_value: float
        The upper and lower bounds of the image will be automatically selected as the maximum and \\
        minimum values in the dict_data if not specified.
    scores: dict
        Dictionary of scoring functions.
    title: str
        Figure name.
    
    Example
    ----------
    XGBoost Prediction-True plot example.
    ```python
        xgb = xgboost.XGBRegressor()
        xgb.fit(X_train, y_train)
        data_dict['train'] = (y_train.T.values.tolist(), xgb.predict(X_train).T.tolist())
        data_dict['val'] = (y_test.T.values.tolist(), xgb.predict(X_test).T.tolist())
        tool.plot_model(data_dict)
    ```
    """

    if min_value is None and max_value is None:
        flat_list = [item for sublist in dict_data.values() for item in sublist]
        min_value = min(min(sublist) for sublist in flat_list)
        max_value = max(max(sublist) for sublist in flat_list)
    plt.plot([min_value,max_value], [min_value,max_value], 'r--')

    text_list = []
    for set_name, true_pre in dict_data.items():
        plt.scatter(true_pre[0], true_pre[1], label=set_name)
        text = f'{set_name}-'
        for score_type, score_func in scores.items():
            score = score_func(true_pre[0], true_pre[1])
            text += f' {score_type}: {score:.2f} '
        text_list.append(text+'\n')

    plt.legend(loc='lower right')
    plt.title(title, fontsize=25)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    texts = ''
    for text in text_list:
        texts += text
    plt.text(x=min_value, y=max_value, s=texts, verticalalignment='top')
    if show:
        plt.show()


def plot_feature_importance(importance, feature_names: list, show_feature=None, 
                            xlabel:str=None, ylabel:str=None, model_type:str=None, show=True):
    """
    Plot the feature importance graph. If the 'show_feature' parameter is specified, the \\
    remaining feature importances will be summed and presented as the final row.

    Parameters
    ----------
    importance: np.ndarray
        The weight or importance of feature descriptors can be obtained using 'feature_importances_.
    feature_names: list[str]
        The names of corresponding feature descriptors.
    show_feature: int or float
        Set the number of features to display. Integer for the number of features, float for the display threshold.
    model_type: str
        Figure name.
    """
    feature_importance = np.array(importance)
    feature_names = np.array(feature_names)
    data={'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Extract features that are not displayed.
    if type(show_feature) == float:
        other_df = fi_df[fi_df['feature_importance'] < show_feature]
    elif type(show_feature) == int:
        other_df = fi_df.iloc[show_feature-1:]
    else:
        other_df = pd.DataFrame([])

    # Calculate the last row.
    fi_df = fi_df.iloc[:len(fi_df)-len(other_df)]
    if len(other_df) != 0:
        other_row = pd.DataFrame({
            'feature_names': [f'{len(other_df)} other features'], 
            'feature_importance': [other_df['feature_importance'].sum()]})
        fi_df = pd.concat([fi_df, other_row], ignore_index=True)
    
    # draw
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    if model_type != None:
        plt.title(str(model_type) + ' FEATURE IMPORTANCE')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()


def sample_distribution(samples: dict, num_bins=20, show=True, alpha=0.9,
                        ylabel='Number of Samples', xlabel='Range of Samples'):
    """
    Draw a distribution image of sample data.
    """
    def sort_dict(item):
        _, series = item
        return -len(series)
    samples = dict(sorted(samples.items(), key=sort_dict))

    sample_max = max([s.max() for s in samples.values()])
    sample_min = min([s.min() for s in samples.values()])

    for label, sample_set in samples.items():
        hist, bin_edges = np.histogram(sample_set, 
            range=(sample_min, sample_max), bins=num_bins, density=False)
        label = label
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), 
                align='edge', alpha=0.5, label=label)
        
    plt.grid(alpha=alpha)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()

    if show:
        plt.show()


def save_plot(plot_func, style_path, save_path=None, save_type='tiff', fig_name='figure', *args, **kwargs):
    """
    An image saving function.
    
    Parameters
    ----------
    plot_func: callable
        A drawing function that needs to save images.
    style_path: str
        Style path of figure.
    save_path: str
        Save path of figure.
    save_type: str
        Image type, default: tiff
    fig_name: str
        Figure name.
    kwargs: Any
        Other necessary parameters for plot_func.
    """
    if save_path is None:
        save_path = os.path.split(os.path.realpath(__file__))[0]
        save_path = os.path.join(save_path, 'save_image')
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, fig_name + '.' + save_type)

    # reset plot style
    plt.style.use('default')
    plt.style.use(style_path)
    plot_func(*args, **kwargs)
    plt.savefig(save_path, bbox_inches='tight')
    print('figure was save in: ', save_path)
    plt.close()