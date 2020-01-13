def main(min_transf=-105, cv=False):
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import os

    from pathlib import Path
    from sklearn.metrics import mean_absolute_error, accuracy_score
    from sklearn.model_selection import RandomizedSearchCV

    # XGBoost with RandomizedSearchCV
    def select_predictor_variables(df, name):
        return df.drop(name, axis=1)

    def select_target_variable(df, name):
        return df[name]

    # Path shenanigans
    path_handler = {}

    for word in (str(Path.cwd()) + "\\3.- IoT Analytics\\Wifi Locationing\\Datasets").split("\\"):
        if word in path_handler:
            path_handler[word] += 1
        else:
            path_handler[word] = 1

    datasets_path = "\\".join(path_handler.keys())

    train = pd.read_csv(datasets_path + "\\trainingData.csv")
    validation = pd.read_csv(datasets_path + "\\validationData.csv")

    if min_transf:
        train = train.apply(lambda x: np.where(x == 100, min_transf, x))
        validation = validation.apply(lambda x: np.where(x == 100, min_transf, x))

    param_distributions = {
        "learning_rate": [0.25],
        "max_depth": [15],
        "min_child_weight": [1],
        "gamma": [0.4],
        "colsample_bytree": [0.4],
        "n_estimators": [400]
    }

    anonymized_vars = ["LATITUDE", "LONGITUDE", "FLOOR", "BUILDINGID", "SPACEID",
                       "RELATIVEPOSITION", "USERID", "TIMESTAMP", "PHONEID"]

    vars_to_predict = ["LATITUDE", "LONGITUDE", "FLOOR", "BUILDINGID"]

    d_xgb = {}
    xgbc_model = {}
    xgbr_model = {}

    for var in vars_to_predict:
        x_train = select_predictor_variables(train, anonymized_vars)
        x_test = select_predictor_variables(validation, anonymized_vars)
        y_train = select_target_variable(train, var)
        # y_test = select_target_variable(validation, var)

        if var in ["LATITUDE", "LONGITUDE"]:
            if cv:
                xgbr = RandomizedSearchCV(estimator=xgb.XGBRegressor(verbosity=1, objective="reg:squarederror"),
                                          param_distributions=param_distributions, scoring="neg_mean_absolute_error",
                                          verbose=10, n_jobs=4, random_state=42, cv=5)

            else:
                xgbr = xgb.XGBRegressor(max_depth=15, objective="reg:squarederror", min_child_weight=1,
                                        n_estimators=400, colsample_bytree=0.4, gamma=0.4, learning_rate=0.25,
                                        nthread=4)

            xgbr_fit = xgbr.fit(x_train, y_train.values.ravel())

            d_xgb[var] = xgbr_fit.predict(x_test)
            xgbr_model[var] = xgbr_fit

        else:
            if cv:
                xgbc = RandomizedSearchCV(estimator=xgb.XGBClassifier(verbosity=1),
                                          param_distributions=param_distributions, scoring="accuracy", verbose=10,
                                          n_jobs=4, random_state=42, cv=5)
            else:
                xgbc = xgb.XGBClassifier(max_depth=15, objective="reg:squarederror", min_child_weight=1,
                                         n_estimators=400, colsample_bytree=0.4, gamma=0.4, learning_rate=0.25,
                                         nthread=4)

            xgbc_fit = xgbc.fit(x_train, y_train.values.ravel())

            d_xgb[var] = xgbc_fit.predict(x_test)
            xgbc_model[var] = xgbc_fit

    distance_75_xgb = np.percentile(
        np.sqrt((d_xgb["LONGITUDE"] - validation["LONGITUDE"])**2 + (d_xgb["LATITUDE"] - validation["LATITUDE"])**2) +
        4*abs(d_xgb["FLOOR"] - validation["FLOOR"]) + 50*(d_xgb["BUILDINGID"] != validation["BUILDINGID"]),
        75
    )

    # ------------------------------------------------------------------ NoChanges / -110
    print(mean_absolute_error(validation["LATITUDE"], d_xgb["LATITUDE"]))  # 135.52 / 136.15
    print(mean_absolute_error(validation["LONGITUDE"], d_xgb["LONGITUDE"]))  # 25.16 / 25.99
    print(accuracy_score(validation["FLOOR"], d_xgb["FLOOR"]))  # 0.7778 / 0.8541
    print(accuracy_score(validation["BUILDINGID"], d_xgb["BUILDINGID"]))  # 0.9892 / 0.9901
    print(distance_75_xgb)  # 174.23 / 175.86
    return xgbr_model, xgbc_model


# xgbc_fits, xgbr_fits = main()

if __name__ == "__main__":
    xgbr_fits, xgbc_fits = main()

# best params BUILDINGID:
# params = {'min_child_weight': 1, 'max_depth': 15, 'learning_rate': 0.15, 'gamma': 0.4, 'colsample_bytree': 0.3}
# acc = 0.999

# best params LATITUDE:
# params = {'n_estimators': 400, 'learning_rate': 0.3}

# best params LONGITUDE:
# params = {'n_estimators': 500, 'learning_rate': 0.25}

# best params FLOOR:
# params = {'n_estimators': 400, 'learning_rate': 0.3}

# best params BUILDINGID:
# params = {'n_estimators': 400, 'learning_rate': 0.25}
# results: 9.69, 14.61, 0.91, 0.999, 24.43

