import importlib
import itertools
import json
import os

import numpy as np
import yaml
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score


def optimization(model, X, y, config_file):
    """
    Hyperparametric optimization search.

    Parameters
    ----------
    model: scikit-learn or XGBoost regressor
    X: pd.DataFrame
        The dataset to be searched for.
    y: pd.Series
        Predict target attributes.
    config_file: str
        Hyperparameter optimization configuration file path.

    Return
    ----------
    The search will save or update parameter logs based on the settings in \\
    the configuration file.
    """
    search_type = {
        "bayes":bayes_search,
        "grid": grid_search
    }

    with open(config_file, "r", encoding='utf-8') as optim_config:
        config = yaml.safe_load(optim_config)
    params = load_param_space(config)

    search_type[config["search type"]](
        X = X,
        y = y,
        model = model,
        param_dist = params,
        **config
    )


def model_import(model_type: str, **kwargs):
    """
    Import the model based on the input string and return it.

    Parameters
    ----------
    model_type: str
        Model import path.
    kwargs: Any
        Model initialization hyperparameters.

    Return
    ----------
    Import model instances under the path.
    """
    module_name, model_name = model_type.rsplit('.', 1)
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, model_name)

    model = ModelClass(**kwargs)

    return model


def load_param_space(data):
    """
    Load parameters space from config file.
    """
    type_dict = {"int": int, "float": float, "str": str}
    parameters = {}

    for key, value in data.get("parameters space", {}).items():
        param_values = []

        if value["list"]:
            param_values.append(np.array(value["include"]))
        else:
            start, end, step = value["start"], value["end"], value["step"]
            if step is None:
                param_values.append((start, end))
            else:
                param_values.append(np.arange(start, end, step))
            
            if "func" in value and value["func"] is not None:
                exec(value["func"])
        
        param_type = type_dict.get(value["type"], str)
        param_values.append(param_type)
        parameters[key] = param_values
    
    return parameters


def bayes_search(model, X, y, param_dist, **kwargs):
    """
    Bayesian Optimization

    Parameters
    ----------
    model: scikit-learn or XGBoost regressor
    X: pd.DataFrame
        The dataset to be searched for.
    y: pd.Series
        Predict target attributes.
    param_dist: dict
        The hyperparameter space to be searched.
    """
    def score_func(**args):

        for key, value in args.items():
            args[key] = param_dist[key][1](value)

        for key, value in param_dist.items():
            model.set_params(**args)
        
        return cross_val_score(model, X, y, scoring=kwargs["scoring"], cv=kfold).mean()
    
    kfold = KFold(n_splits=kwargs["cv"], shuffle=True, random_state=kwargs["random state"])
    score_func = score_func
    space = {key: value[0] for key, value in param_dist.items()}

    print(param_dist)

    if kwargs["init_points"] is None and kwargs["n_iter"] is None:
        kwargs["init_points"] = 5
        kwargs["n_iter"] = kwargs["log times"] - 5
    for i in range(kwargs["epoch"]):
        params_list = []
        optimizer = BayesianOptimization(score_func, space)
        optimizer.maximize(init_points=kwargs["init_points"], n_iter=kwargs["n_iter"])
        if kwargs["save log"] is not None:
            params = optimizer.res
            sorted_data = sorted(params, key=lambda x: x['target'], reverse=True)
            best_param = sorted_data[0]
            for key, value in best_param["params"].items():
                best_param["params"][key] = param_dist[key][1](best_param["params"][key])
                best_param["random state"] = kwargs["random state"]
                best_param["K-Fold"] = kwargs["cv"]
            params_list.append(best_param)
            save_search_data(params_list,
                             save_path=kwargs["log path"], 
                             search_type=kwargs["search type"])
            print(f"Round {i+1} of the search is finished, and parameter logs are updated.\n")


def grid_search(X, y, model, param_dist, **kwargs):
    """
    grid search, not recommended.
    """
    param_dist = to_nparray(common_params(param_dist))
    dist_len = [len(value) for value in param_dist.values()]
    factor_list = generate_factors(kwargs["log times"], dist_len)
    
    i = 0
    for key, value in param_dist.items():
        new_shape = (dist_len[i] // factor_list[i], factor_list[i])
        param_dist[key] = value.reshape(new_shape)
        i += 1

    factors_list = [range(int(i / factor)) for i, factor in zip(dist_len, factor_list)]
    subset_combinations = generate_subset(factors_list)
    kfold = KFold(n_splits=kwargs["cv"], shuffle=True, random_state=kwargs["random state"])
    for subset_combination in subset_combinations:
        subset = create_subset(param_dist, subset_combination)
        
        search = GridSearchCV(
            estimator=model, 
            param_grid=subset, 
            scoring=kwargs["scoring"],
            cv=kfold
        )
        search.fit(X, y)
        print(search.best_score_)


def common_params(params):
    param_dist = {}
    for key, value in params.items():
        param_dist[key] = value[0]
    return param_dist


def generate_factors(log_times, dist_len):
    def factors(n):
        return [i for i in range(1, n + 1) if n % i == 0]

    def closest_combination(lists, target_product):
        arrays = [np.array(lst) for lst in lists]
        all_combinations = list(itertools.product(*arrays))
        product_combinations = np.prod(np.array(all_combinations), axis=1)
        closest_index = np.argmin(np.abs(product_combinations - target_product))
        best_combination = all_combinations[closest_index]

        return best_combination

    factors_list = [factors(n) for n in dist_len]
    return closest_combination(factors_list, log_times)


def generate_subset(factor_list):
    for permutation in itertools.product(*factor_list):
        yield permutation


def to_nparray(param_dist):
    for key, value in param_dist.items():
        param_dist[key] = np.array(value)
    
    return param_dist


def create_subset(param_dist, subset_combination):
    subset = {}
    i = 0
    for key, value in param_dist.items():
        subset[key] = value[subset_combination[i]]
        i += 1
    return subset


def save_search_data(params, save_path=None, search_type='Bayes'):
    """
    Save Bayesian optimization parameter logs.

    Parameters
    ----------
    params: list[dict]
        List of optimal hyperparameters to be saved.
    save_path: str
        Path for saving parameter logs.
    """
    file_name = f"{search_type}_log.json"

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_name = os.path.join(save_path, file_name)

    # If the log file exists, write new parameters to the log file.
    saved_data = []
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            saved_data = json.load(file)
    saved_data.extend(params)

    with open(file_name, 'w') as file:
        json.dump(saved_data, file, indent=4)


if __name__=="__main__":
    pass