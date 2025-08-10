import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
from lazypredict.Supervised import  LazyClassifier, LazyRegressor

from classification import model

# read data
data = pd.read_csv("diabetes.csv")
target = "Outcome"

# data visualisation
profile = ProfileReport(data, title="data report", explorative=True)
profile.to_file("diabetes_report.html")

# split data
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2323)

# preprocessing data
scaler = StandardScaler
x_train = scaler.fit_transform(x_train)
y_train = scaler.transform(y_train)

# choose params
params = [

]

# choose model
model = GridSearchCV(RandomForestClassifier(random_state=34343),
                    param_grid=params,
                     scoring="recall",



                     )




