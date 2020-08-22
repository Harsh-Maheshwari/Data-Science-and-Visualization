import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from boruta import BorutaPy
import pickle


df = pd.read_excel("../static/ml/Customer-Churn-Dataset.xls")
df = df.replace(r'^\s+$', np.nan, regex=True)
df = df.dropna()
del df["customerID"]
del df["TotalCharges"]

all_columns_list = df.columns.tolist()
numerical_columns_list = ['tenure','MonthlyCharges']
categorical_columns_list = [e for e in all_columns_list if e not in numerical_columns_list]
for index in categorical_columns_list:
    df[index] = pd.Categorical(df[index])
for index in numerical_columns_list:
    df[index] = pd.to_numeric(df[index])

num = ['float64', 'int64']
num_df = df.select_dtypes(include=num)
obj_df = df.select_dtypes(exclude=num)
num_df = pd.concat([num_df,df["Churn"]],axis=1)
tenure_bins=pd.cut(num_df["tenure"], bins=[0,20,60,80], labels=['low','medium','high'])
MonthlyCharges_bins=pd.cut(num_df["MonthlyCharges"], bins=[0,35,60,130], labels=['low','medium','high'])
df['SeniorCitizen'] = df.SeniorCitizen.map({0:'No', 1:'Yes'})
bins = pd.DataFrame([tenure_bins, MonthlyCharges_bins]).T
transformed_df = pd.concat([bins,obj_df],axis=1)
dummy_columns = [e for e in transformed_df.columns if e != 'Churn']
df_dummies = pd.get_dummies(data=transformed_df, columns=dummy_columns)
df_dummies_features = df_dummies.drop(["Churn"], axis=1).columns
X_all = df_dummies[df_dummies_features]
y_all = df_dummies["Churn"]
X_boruta = X_all.values
y_boruta = y_all.values
rfc = RandomForestClassifier(n_jobs = -1)
feature_selector = BorutaPy(rfc, n_estimators='auto', verbose=2, random_state=1)
feature_selector.fit(X_boruta, y_boruta)
all_feature = df_dummies.drop(['Churn'],axis=1).columns.tolist()
df_features_rank = pd.DataFrame(index = all_feature)
df_features_rank['Boruta_Rank'] = feature_selector.ranking_
df_features_rank['Feature']=  df_features_rank.index
df_features_rank = df_features_rank.sort_values('Boruta_Rank')
df_top2_ranked_feature = df_features_rank.loc[df_features_rank['Boruta_Rank'].isin([1,2])]
df_top2_ranked_feature
selected_features = df_top2_ranked_feature.index

X_selected = df_dummies[selected_features]
y_selected = df_dummies["Churn"]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_selected, y_selected, test_size=0.20, random_state=7)

scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression(solver = 'lbfgs')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators=10)))
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cv_results = model_selection.cross_val_score(model, X_train, y_train,  cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #print(msg)

models = []
accuracy_list = []
trained_models = {}


models.append(('LogisticRegression', LogisticRegression(solver = 'lbfgs')))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier(n_estimators=10)))

for name, model in models:
    model.fit(X_train, y_train)
    trained_models[name] = model
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    accuracy_list.append((name,acc))
models_metrics = pd.DataFrame(accuracy_list, columns=["Model", "Accuracy"]) 
models_metrics['Model_Rank'] = models_metrics['Accuracy'].rank(ascending=False, method='first')
models_metrics.to_csv('../static/ml/metrics_score.csv', index=False)
rank_dict = pd.Series(models_metrics.Model_Rank.values, index=models_metrics.Model.values).to_dict()
trained_models_with_rank = {}

for key, value in rank_dict.items():
    trained_models_with_rank[rank_dict[key]] = [value1 for key1, value1 in trained_models.items() if key == key1]
    trained_models_with_rank[rank_dict[key]].append(key)

filename = '../static/ml/pickled_models.pkl'
pickle.dump(trained_models_with_rank, open(filename, 'wb'), protocol=2)








