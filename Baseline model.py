import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, r2_score

train = pd.read_csv(".\\3.- IoT Analytics\\Wifi Locationing\\Datasets\\trainingData.csv")
validation = pd.read_csv(".\\3.- IoT Analytics\\Wifi Locationing\\Datasets\\validationData.csv")

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

encoders = []


def select_predictor_variables(df, name):
    return df.drop(name, axis=1)


def select_target_variable(df, name):
    return df[name]


anonymized_vars = ["LATITUDE", "LONGITUDE", "FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "TIMESTAMP",
                   "PHONEID"]
vars_to_predict = ["LATITUDE", "LONGITUDE", "FLOOR", "BUILDINGID"]

acc = []
d = {}

for var in vars_to_predict:
    X_train = select_predictor_variables(train, anonymized_vars)
    X_test = select_predictor_variables(validation, anonymized_vars)
    y_train = select_target_variable(train, var)
    y_test = select_target_variable(validation, var)

    if var in ["LATITUDE", "LONGITUDE"]:
        RFR = RandomForestRegressor(random_state=42)
        RFR_fit = RFR.fit(X_train, y_train.values.ravel())
        predictions = RFR_fit.predict(X_test)

        d[var] = predictions

    else:
        RFC = RandomForestClassifier(random_state=42)
        RFC_fit = RFC.fit(X_train, y_train)
        predictionss = RFC_fit.predict(X_test)

        d[var] = predictionss
        acc.append(accuracy_score(y_test, predictionss))


Distance_75 = np.percentile(
    abs(d["LONGITUDE"] - validation["LONGITUDE"]) + abs(d["LATITUDE"] - validation["LATITUDE"]) +
    4*abs(d["FLOOR"] - validation["FLOOR"]) + 50*(d["BUILDINGID"] - validation["BUILDINGID"]), 75
)

print(mean_absolute_error(validation["LATITUDE"], d["LATITUDE"]))
print(mean_absolute_error(validation["LONGITUDE"], d["LONGITUDE"]))
print(r2_score(validation["LONGITUDE"], d["LONGITUDE"]))
print(Distance_75)
print(acc)
