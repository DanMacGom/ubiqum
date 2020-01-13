import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

train = pd.read_csv(".\\3.- IoT Analytics\\Wifi Locationing\\Datasets\\trainingData.csv")
validation = pd.read_csv(".\\3.- IoT Analytics\\Wifi Locationing\\Datasets\\validationData.csv")

train = train.apply(lambda x: np.where(x == 100, -110, x))
validation = validation.apply(lambda x: np.where(x == 100, -110, x))
train.iloc[:, 0:520] = normalize(train.iloc[:, 0:520], norm="l1", axis=1)
validation.iloc[:, 0:520] = normalize(validation.iloc[:, 0:520], norm="l1", axis=1)
sum(train.iloc[1, 0:520])

# Group_by and mean()
# train = train.groupby(by=["LATITUDE", "LONGITUDE", "FLOOR"]).mean().reset_index()
# train_test = pd.concat([train, validation], sort=False)

phoneid_brand_no_version = {
    0: "Celkon A27", 1: "GT-I8160", 2: "GT-I8160", 3: "GT-I9100", 4: "GT-I9300", 5: "GT-I9505", 6: "GT-S5360",
    7: "GT-S6500", 8: "Galaxy Nexus", 9: "Galaxy Nexus", 10: "HTC Desire HD", 11: "HTC One", 12: "HTC One",
    13: "HTC Wildfire S", 14: "LT22i", 15: "LT22i", 16: "LT26i", 17: "M1005D", 18: "MT11i", 19: "Nexus 4",
    20: "Nexus 4", 21: "Nexus S", 22: "Orange Monte Carlo", 23: "Transformer TF101", 24: "bq Curie", 25: "Nexus 5",
    26: "Orange Rono", 27: "D2303", 28: "Wildfire S A510e", 29: "GT-I9505"
}

phone_id_android_version = {
    0: "4.0.4(6577)", 1: "2.3.6", 2: "4.1.2", 3: "4.0.4", 4: "4.1.2", 5: "4.2.2", 6: "2.3.6", 7: "2.3.6", 8: "4.2.2",
    9: "4.3", 10: "2.3.5", 11: "4.1.2", 12: "4.2.2", 13: "2.3.5", 14: "4.0.4", 15: "4.1.2", 16: "4.0.4", 17: "4.0.4",
    18: "2.3.4", 19: "4.2.2", 20: "4.3", 21: "4.1.2", 22: "2.3.5", 23: "4.0.3", 24: "4.1.1", 25: "5.0.1", 26: "4.4.2",
    27: "4.4.4", 28: "4.2.2", 29: "4.4.2"
}


def select_predictor_variables(df, name):
    return df.drop(name, axis=1)


def select_target_variable(df, name):
    return df[name]


def data_transf(df):
    # date_time_series = pd.to_datetime(df["TIMESTAMP"], unit="s")
    # df["SECONDS"] = date_time_series.apply(lambda x: x.second)
    # df["MINUTES"] = date_time_series.apply(lambda x: x.minute)
    # df["HOURS"] = date_time_series.apply(lambda x: x.hour)
    # df["DAY"] = date_time_series.apply(lambda x: x.day)
    # df["MONTH"] = date_time_series.apply(lambda x: x.month)

    # df["PHONE_BRAND"] = df["PHONEID"].apply(lambda x: phoneid_brand_no_version[x])
    # df["ANDROID_VERSION"] = df["PHONEID"].apply(lambda x: phone_id_android_version[x])
    #
    # le_android_version = LabelEncoder()
    # encoders.append(le_android_version)
    # df["ANDROID_VERSION"] = le_android_version.fit_transform(df["ANDROID_VERSION"])
    #
    # le_phone_brand = LabelEncoder()
    # encoders.append(le_phone_brand)
    # df["PHONE_BRAND"] = le_phone_brand.fit_transform(df["PHONE_BRAND"])

    return df


anonymized_vars = ["LATITUDE", "LONGITUDE", "FLOOR", "BUILDINGID", "SPACEID",
                   "RELATIVEPOSITION", "USERID", "TIMESTAMP", "PHONEID"]

vars_to_predict = ["LATITUDE", "LONGITUDE", "FLOOR", "BUILDINGID"]

d_rfr_mean = pd.DataFrame()

# Mean of train Random Forest
y_test_total = []

for var in vars_to_predict:
    X_train, X_test, y_train, y_test = train_test_split(train_test.iloc[:, 9:529], train_test[var], random_state=42)
    y_test_total.append(y_test)
    # X_train = select_predictor_variables(X_train, var)
    # X_test = select_predictor_variables(X_test, anonymized_vars)
    # y_train = select_target_variable(train, var)
    # y_test = select_target_variable(validation, var)

    if var in ["LATITUDE", "LONGITUDE"]:
        RFR = RandomForestRegressor(random_state=42)
        RFR_fit = RFR.fit(X_train, y_train.values.ravel())
        predictions = RFR_fit.predict(X_test)

        d_rfr_mean[var] = predictions

    else:
        RFC = RandomForestClassifier(random_state=42)
        RFC_fit = RFC.fit(X_train, y_train)
        predictions_class = RFC_fit.predict(X_test)

        d_rfr_mean[var] = predictions_class

Distance_75_rfr_mean = np.percentile(
    abs(d_rfr_mean["LONGITUDE"].values - y_test_total[1].values) +
    (abs(d_rfr_mean["LATITUDE"].values - y_test_total[0].values)) +
    (4*abs(d_rfr_mean["FLOOR"] - y_test_total[2].values)) +
    (50*abs((d_rfr_mean["BUILDINGID"] - d_rfr_mean["BUILDINGID"]))),
    75
)

# ------------------------------------------------------------------ NoChanges / -110 / mean by local. /
print(mean_absolute_error(y_test_total[0].values, d_rfr_mean["LATITUDE"]))  # 8.76 / 7.99 / 6.37
print(mean_absolute_error(y_test_total[1].values, d_rfr_mean["LONGITUDE"]))  # 10.14 / 9.27 / 9.63
print(accuracy_score(y_test_total[2].values, d_rfr_mean["FLOOR"]))  # 0.8352 / 0.8857 / 0.9256
print(accuracy_score(y_test_total[3].values, d_rfr_mean["BUILDINGID"]))  # 0.9981 / 0.9955 / 0.9980
print(Distance_75_rfr_mean)  # 24.28 / 21.165 / 19.1449


d_rfr = pd.DataFrame()

# Random Forest
for var in vars_to_predict:
    X_train = select_predictor_variables(train, anonymized_vars)
    X_test = select_predictor_variables(validation, anonymized_vars)
    y_train = select_target_variable(train, var)
    y_test = select_target_variable(validation, var)

    if var in ["LATITUDE", "LONGITUDE"]:
        RFR = RandomForestRegressor(random_state=42)
        RFR_fit = RFR.fit(X_train, y_train.values.ravel())
        predictions = RFR_fit.predict(X_test)

        d_rfr[var] = predictions

    else:
        RFC = RandomForestClassifier(random_state=42)
        RFC_fit = RFC.fit(X_train, y_train)
        predictions_class = RFC_fit.predict(X_test)

        d_rfr[var] = predictions_class


Distance_75_rfr = np.percentile(
    abs(d_rfr["LONGITUDE"] - validation["LONGITUDE"]) + (abs(d_rfr["LATITUDE"] - validation["LATITUDE"])) +
    (4*abs(d_rfr["FLOOR"] - validation["FLOOR"])) + (50*abs((d_rfr["BUILDINGID"] - validation["BUILDINGID"]))),
    75
)

# ------------------------------------------------------------------ NoChanges / -110
print(mean_absolute_error(validation["LATITUDE"], d_rfr["LATITUDE"]))  # 8.76 / 7.99
print(mean_absolute_error(validation["LONGITUDE"], d_rfr["LONGITUDE"]))  # 10.14 / 9.27
print(accuracy_score(validation["FLOOR"], d_rfr["FLOOR"]))  # 0.8352 / 0.8857
print(accuracy_score(validation["BUILDINGID"], d_rfr["BUILDINGID"]))  # 0.9981 / 0.9955
print(Distance_75_rfr)  # 24.28 / 21.165


# XGBoost
d_xgb = pd.DataFrame()

for var in vars_to_predict:
    X_train = select_predictor_variables(train, anonymized_vars)
    X_test = select_predictor_variables(validation, anonymized_vars)
    y_train = select_target_variable(train, var)
    y_test = select_target_variable(validation, var)

    if var in ["LATITUDE", "LONGITUDE"]:
        XGBCR = xgb.XGBRegressor(objective="reg:squarederror", seed=42)
        XGBCR_fit = XGBCR.fit(X_train, y_train.values.ravel())
        predictions = XGBCR_fit.predict(X_test)

        d_xgb[var] = predictions

    else:
        XGBC = xgb.XGBClassifier(seed=42)
        XGBC_fit = XGBC.fit(X_train, y_train)
        predictions_class = XGBC_fit.predict(X_test)

        d_xgb[var] = predictions_class

Distance_75_xgb = np.percentile(
    abs(d_xgb["LONGITUDE"] - validation["LONGITUDE"]) + abs(d_xgb["LATITUDE"] - validation["LATITUDE"]) +
    4*abs(d_xgb["FLOOR"] - validation["FLOOR"]) + 50*abs((d_xgb["BUILDINGID"] - validation["BUILDINGID"])),
    75
)

# ------------------------------------------------------------------ NoChanges / -110
print(mean_absolute_error(validation["LATITUDE"], d_xgb["LATITUDE"]))  # 135.52 / 136.15
print(mean_absolute_error(validation["LONGITUDE"], d_xgb["LONGITUDE"]))  # 25.16 / 25.99
print(accuracy_score(validation["FLOOR"], d_xgb["FLOOR"]))  # 0.7778 / 0.8541
print(accuracy_score(validation["BUILDINGID"], d_xgb["BUILDINGID"]))  # 0.9892 / 0.9901
print(Distance_75_xgb)  # 174.23 / 175.86


param_grid = {
    "C": [10**(-5), 10**(-4), 10**(-3), 10**(-2)],
    "gamma": [10**-8, 10**-7, 10**-6], #10**-5, 10**-4, 10**(-3), 10**(-2), 10**(-1), 10**0, 10**1, 10**2]
}


# SVM classifier
svm = SVC(kernel='linear', random_state=42)

cv_svm = GridSearchCV(estimator=svm, param_grid=param_grid, scoring="accuracy", n_jobs=7, verbose=2)

# Train the classifier
cv_svm.fit(X_train, y_train)

cv_svm.best_score_
cv_svm.best_params_

preds_svm = svm.predict(X_test)

accuracy_score(y_test, preds_svm)  # 0.9982

# K-Nearest Neighbors
acc_knn = []  # Best result 0.9945 for k = 1

for _ in range(1, 20):
    classifier = KNeighborsClassifier(n_neighbors=_)
    classifier.fit(X_train, y_train)

    preds_knn = classifier.predict(X_test)

    acc_knn.append((accuracy_score(y_test, preds_knn), _))


acc_knn.sort(reverse=True)

train.any(0, 1)

