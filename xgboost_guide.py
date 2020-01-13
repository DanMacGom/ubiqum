import matplotlib.pylab as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error

train = pd.read_csv(".\\3.- IoT Analytics\\Wifi Locationing\\Datasets\\trainingData.csv")
validation = pd.read_csv(".\\3.- IoT Analytics\\Wifi Locationing\\Datasets\\validationData.csv")
target = train["BUILDINGID"]
train = train.iloc[:, 0:520]

xgb1 = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    objective="multi:softmax",
    num_class=3,
    seed=27
)

alg = xgb1


def evaluate_model(alg, train, target, predictors, cv_folds=5, early_stopping_rounds=1):
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(train[predictors].values, target.values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                      metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(train[predictors], target['diagnosis'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(train[predictors])
    dtrain_predprob = alg.predict_proba(train[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % accuracy_score(target['diagnosis'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % roc_auc_score(target['diagnosis'], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importance', color='g')
    plt.ylabel('Feature Importance Score')


# Choose all predictors except target & IDcols
predictors = train.columns

evaluate_model(xgb1, train, target, predictors)
