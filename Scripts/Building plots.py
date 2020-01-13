import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale

train = pd.read_csv(".\\3.- IoT Analytics\\Wifi Locationing\\Datasets\\trainingData.csv")
validation = pd.read_csv(".\\3.- IoT Analytics\\Wifi Locationing\\Datasets\\validationData.csv")

theta = np.radians(29)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s), (s, c)))

train[["LONGITUDE", "LATITUDE"]] = np.transpose(np.dot(R, np.transpose(train[["LONGITUDE", "LATITUDE"]])))
validation[["LONGITUDE", "LATITUDE"]] = np.transpose(np.dot(R, np.transpose(validation[["LONGITUDE", "LATITUDE"]])))

plt.scatter(x=validation["LONGITUDE"], y=validation["LATITUDE"])

train[["LATITUDE", "LONGITUDE"]] = np.transpose(np.dot(R, np.transpose(train[["LATITUDE", "LONGITUDE"]])))
validation[["LATITUDE", "LONGITUDE"]] = np.transpose(np.dot(R, np.transpose(validation[["LATITUDE", "LONGITUDE"]])))


dtrain = train
dtrain["isvalidation"] = 0
dvalidation = validation
dvalidation["isvalidation"] = 1
alltogether = pd.concat([dtrain, dvalidation])

alltogether[["LONGITUDE", "LATITUDE"]].scatter()

ax = Axes3D(plt.gcf())
ax.scatter(xs=alltogether["LONGITUDE"], ys=alltogether["LATITUDE"], zs=alltogether["FLOOR"], zdir='z',
           c=alltogether["FLOOR"],
           )



