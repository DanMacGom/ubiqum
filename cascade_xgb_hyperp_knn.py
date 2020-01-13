def main(min_transf=-105, cv_buildingid=False, cv_floor=False, cv_lat_lon=False, knn=True):
    import pandas as pd
    import numpy as np
    import xgboost as xgb

    from pathlib import Path
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.preprocessing import normalize
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

    # Selector of variables functions
    def select_predictor_variables(df, name):
        return df.drop(name, axis=1)

    def select_target_variable(df, name):
        return df[name]

    # CV hyper parameter tuning list. Used it to find the best parameters for the xgboost model. Fixed the values for
    # subsequent models.
    param_distributions = {
        "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
    }

    # Path shenanigans
    path_handler = {}

    # Depending if the code is executed through the console or the IDE the relative path changes.
    # This is a workaround that discards the repeated folders.
    for word in (str(Path.cwd()) + "\\3.- IoT Analytics\\Wifi Locationing\\Datasets").split("\\"):
        if word in path_handler:
            path_handler[word] += 1
        else:
            path_handler[word] = 1

    datasets_path = "\\".join(path_handler.keys())

    train = pd.read_csv(datasets_path + "\\trainingData.csv")
    train2 = pd.read_csv(datasets_path + "\\validationData.csv")
    train = train.append(train2, ignore_index=True)

    validation = pd.read_csv(datasets_path + "\\TEST_DATA.csv")

    theta = np.radians(29)  # 29 is almost horizontal
    c, s = np.cos(theta), np.sin(theta)
    r = np.array(((c, -s), (s, c)))

    train[["LONGITUDE", "LATITUDE"]] = np.transpose(np.dot(r, np.transpose(train[["LONGITUDE", "LATITUDE"]])))

    # Transform 100s values to something more meaningful for the model

    if min_transf:
        train = train.apply(lambda x: np.where(x == 100, min_transf, x))
        validation = validation.apply(lambda x: np.where(x == 100, min_transf, x))

    anonymized_vars = ["LATITUDE", "LONGITUDE", "FLOOR", "BUILDINGID", "SPACEID",
                       "RELATIVEPOSITION", "USERID", "TIMESTAMP", "PHONEID"]

    # Cascade: first BUILDINGID as target variable.
    x_train = select_predictor_variables(train, anonymized_vars)
    x_test = select_predictor_variables(validation, anonymized_vars)

    # Shift all the values in the data set by 105.
    x_train = x_train.apply(lambda x: x + 105)
    x_test = x_test.apply(lambda x: x + 105)

    # Apply l2 row normalization. The sum of each row will be 1. This should help in taking into account the variance
    # in height, mobile phone model, etc.
    x_train = pd.DataFrame(normalize(X=x_train, norm="l2", axis=1), columns=x_train.columns)
    x_test = pd.DataFrame(normalize(X=x_test, norm="l2", axis=1), columns=x_test.columns)

    y_train = select_target_variable(train, "BUILDINGID")

    if cv_buildingid:
        xgbc_building = RandomizedSearchCV(estimator=xgb.XGBClassifier(verbosity=1),
                                           param_distributions=param_distributions,
                                           scoring="accuracy", verbose=10,
                                           n_jobs=4, random_state=42, cv=5)
    elif knn:
        xgbc_building = KNeighborsClassifier(n_neighbors=5)
    else:
        xgbc_building = xgb.XGBClassifier(
            max_depth=15, objective="reg:squarederror", min_child_weight=1,
            n_estimators=400, colsample_bytree=0.4, gamma=0.4, learning_rate=0.25,
            nthread=4)

    xgbc_fit_building = xgbc_building.fit(x_train, y_train.values.ravel())

    # Cascade: use the prediction as a predictor for other target variables.
    x_test["BUILDINGID"] = xgbc_fit_building.predict(x_test)
    preds_building = x_test["BUILDINGID"]
    x_train["BUILDINGID"] = train["BUILDINGID"].sort_index()

    n_buildings = train["BUILDINGID"].unique().tolist()

    preds_floor = pd.DataFrame()

    # Once BUILDINGID is predicted, the test set can be subset. Apply a different xgboost to each BUILDINGID.
    for build in n_buildings:
        x_train_subset = x_train[x_train["BUILDINGID"] == build]
        y_train_subset = select_target_variable(train[train["BUILDINGID"] == build], "FLOOR")
        x_test_subset = x_test[x_test["BUILDINGID"] == build]

        if cv_floor:
            xgbc_floor = RandomizedSearchCV(estimator=xgb.XGBClassifier(verbosity=1),
                                            param_distributions=param_distributions,
                                            scoring="accuracy", verbose=10,
                                            n_jobs=4, random_state=42, cv=5)
        else:
            xgbc_floor = xgb.XGBClassifier(
                max_depth=15, objective="reg:squarederror", min_child_weight=1,
                n_estimators=400, colsample_bytree=0.4, gamma=0.4, learning_rate=0.25,
                nthread=4)

        xgbr_fit_floor_build = xgbc_floor.fit(x_train_subset, y_train_subset.values.ravel())

        preds_floor = preds_floor.append(pd.DataFrame(xgbr_fit_floor_build.predict(x_test_subset),
                                                      index=x_test_subset.index))

    x_test["FLOOR"] = preds_floor
    x_train["FLOOR"] = train["FLOOR"]

    lat_lon_preds = pd.DataFrame()

    # The best result obtained is through knn with k = 5 for latitude and longitude.
    for var in ["LATITUDE", "LONGITUDE"]:
        y_train = select_target_variable(train, var)

        if cv_lat_lon:
            xgbr_lat_lon = RandomizedSearchCV(estimator=xgb.XGBRegressor(verbosity=1, objective="reg:squarederror"),
                                              param_distributions=param_distributions,
                                              scoring="neg_mean_absolute_error", verbose=10, n_jobs=4, random_state=42,
                                              cv=5)
        elif knn:
            xgbr_lat_lon = KNeighborsRegressor(n_neighbors=5)
        else:
            xgbr_lat_lon = xgb.XGBRegressor(
                max_depth=15, objective="reg:squarederror", min_child_weight=1,
                n_estimators=400, colsample_bytree=0.4, gamma=0.4, learning_rate=0.25,
                nthread=4)

        xgbr_fit_lat_lon = xgbr_lat_lon.fit(x_train, y_train.values.ravel())
        lat_lon_preds[var] = xgbr_fit_lat_lon.predict(x_test)

    theta = np.radians(-29)  # 29 is almost horizontal
    c, s = np.cos(theta), np.sin(theta)
    r = np.array(((c, -s), (s, c)))

    lat_lon_preds[["LONGITUDE", "LATITUDE"]] = np.transpose(np.dot(r, np.transpose(lat_lon_preds[["LONGITUDE",
                                                                                                  "LATITUDE"]])))

    final_preds = pd.concat([lat_lon_preds, preds_floor, preds_building], axis=1, verify_integrity=True)
    final_preds.rename(columns={0: "FLOOR"}, inplace=True)

    final_preds.to_csv(datasets_path + "\\cascade_xgb_hyperp_knn.csv")


if __name__ == "__main__":
    main()
