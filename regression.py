import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("StudentScore.xls")

print(data[["math score", "writing score", "reading score"]].corr())

# profile = ProfileReport(data, title= "Student Score data", explorative=True)
# profile.to_file("Student_report.html")

target = "math score"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=232)

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

education_levels = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
genders = x_train["gender"].unique()
lunchs = x_train["lunch"].unique()
test_preps = x_train["test preparation course"].unique()

ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_levels, genders, lunchs, test_preps]))
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("ord_bool_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_features", nom_transformer, ["race/ethnicity"])
])

# processed_data = ord_transformer.fit_transform(x_train[["parental level of education", "gender", "lunch", "test preparation course"]])
# for i,j in zip(x_train[["parental level of education", "gender", "lunch", "test preparation course"]].values, processed_data):
#     print("Before: {}, After: {}".format(i,j))

# processed_data = preprocessor.fit_transform(x_train)

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=32))
])

params = {
    "preprocessor__num_features_imputer__strategy": ["median", "mean"],
    "regressor__n_estimators": [50, 100, 200],
    "regressor__criterion": ["squared_error", "absolute_error", "friedman_mse"]
}

model = GridSearchCV(
    reg,
    param_grid=params,
    scoring="r2",
    cv=6,
    verbose=0,
    # n_jobs=-1
)

model.fit(x_train, y_train)
print(model.best_score_)
print(model.best_params_)

y_predict = model.predict(x_test)

print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))


# for i,j in zip(y_test, y_predict):
#     print("Actual: {} Predict: {}".format(i,j))
