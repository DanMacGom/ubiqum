from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import xgboost as xgb
import pandas as pd


def algorithm_pipeline(X_train_data, X_test_data, y_train_data, model, param_grid, cv=10,
                       scoring_fit='neg_mean_absolute_error', do_probabilities=False, n_jobs=-1):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train_data, y_train_data.values.ravel())

    if do_probabilities:
        pred = fitted_model.predict_proba(X_test_data)
    else:
        pred = fitted_model.predict(X_test_data)

    return fitted_model, pred


def select_predictor_variables(df, name):
    return df.drop(name, axis=1)


def select_target_variable(df, name):
    return df[name]


train = pd.read_csv(".\\3.- IoT Analytics\\Wifi Locationing\\Datasets\\trainingData.csv")
validation = pd.read_csv(".\\3.- IoT Analytics\\Wifi Locationing\\Datasets\\validationData.csv")

anonymized_vars = ["LATITUDE", "LONGITUDE", "FLOOR", "BUILDINGID", "SPACEID",
                   "RELATIVEPOSITION", "USERID", "TIMESTAMP", "PHONEID"]

vars_to_predict = ["LATITUDE", "LONGITUDE", "FLOOR", "BUILDINGID"]

X_train = select_predictor_variables(train, anonymized_vars)
X_test = select_predictor_variables(validation, anonymized_vars)
y_train = select_target_variable(train, "LATITUDE")
y_test = select_target_variable(validation, "LATITUDE")

modelRFR = RandomForestRegressor()
param_gridRFR = {
    'n_estimators': [400],  # , 700, 1000
    'max_depth': [15],  # , 20, 25
    'max_leaf_nodes': [50]  # , 100, 200
}

modelRFR, predRFR = algorithm_pipeline(X_train, X_test, y_train, modelRFR, param_gridRFR, cv=5)

print(-modelRFR.best_score_)
print(modelRFR.best_params_)


modelXGB = xgb.XGBRegressor(objective="reg:squarederror")
param_gridXGB = {
    'n_estimators': [400],  # 700, 1000
    'colsample_bytree': [0.7],  # , 0.8
    'max_depth': [15],  # , 20, 25
    'reg_alpha': [1.1],  # , 1.2, 1.3
    'reg_lambda': [1.1],  # , 1.2, 1.3
    'subsample': [0.7]  # , 0.8, 0.9
}

modelXGB, predXGB = algorithm_pipeline(X_train, X_test, y_train, modelXGB, param_gridXGB, cv=3, n_jobs=-1)

# Mean absolute error
print(-modelXGB.best_score_)
print(modelXGB.best_params_)

# xgb_param = alg.get_xgb_params()
xgtrain = xgb.DMatrix(X_train.values)
xgtest = xgb.DMatrix(y_train.values)

cvresult = xgb.cv(params={
    'max_depth': 6,
    'min_child_weight': 1,
    'eta': .3,
    'subsample': 1,
    'colsample_bytree': 1,
    "eval_metric": "mae",
    "objective": "reg:squarederror",
    "num_boost_round": 999},
    dtrain=xgtrain,
    # num_boost_round=alg.get_params()['n_estimators'],
    nfold=5,
    metrics='mae',
    early_stopping_rounds=50,
    seed=42)


model1 = xgb.XGBRegressor(objective="reg:squarederror")

model1.fit(X_train, y_train)

preds = model1.predict(X_test)

mean_absolute_error(y_test, preds)

# dtrain = xgb.DMatrix('train.svm.txt')
# dtest = xgb.DMatrix('test.svm.buffer')
