import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
import time

#%%
# Load train data
train_data = pd.read_csv('D:/#NTU/IN6227 DataMining/Assignment1/Census Income Data Set/adult.data', header=None)
train_data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
                      'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
test_data = pd.read_csv('D:/#NTU/IN6227 DataMining/Assignment1/Census Income Data Set/adult.test', skiprows=1, header=None)
test_data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
                     'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

train_data['income'] = train_data['income'].str.strip()
test_data['income'] = test_data['income'].str.replace('.', '').str.strip()

X_train = train_data.drop('income', axis=1)
y_train = train_data['income']
X_test = test_data.drop('income', axis=1)
y_test = test_data['income']

#%% Preprocessing
numeric_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', StandardScaler())]) 

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])  

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)])

#%% Decision Tree 
tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('classifier', DecisionTreeClassifier(
        max_depth=10,  
        min_samples_split=10, 
        min_samples_leaf=5,  
        criterion='gini'  
    ))
])
start_time = time.time()
tree_pipeline.fit(X_train, y_train)
train_time_tree = time.time() - start_time

y_pred_tree = tree_pipeline.predict(X_test)
tree_accuracy = accuracy_score(y_test, y_pred_tree)
tree_loss = log_loss(y_test, tree_pipeline.predict_proba(X_test))

conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
class_report_tree = classification_report(y_test, y_pred_tree)

print(f"Decision Tree Accuracy: {tree_accuracy}")
print(f"Decision Tree Loss: {tree_loss}")
print(f"Confusion Matrix:\n {conf_matrix_tree}")
print(f"Classification Report:\n {class_report_tree}")
print(f"Decision Tree Training Time: {train_time_tree} seconds")
"""
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

decision_tree_model = tree_pipeline.named_steps['classifier']

plt.figure(figsize=(20, 10))  
plot_tree(decision_tree_model, 
          feature_names=numeric_columns + tree_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out().tolist(),  
          class_names=['<=50K', '>50K'], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.show()"""

#%% KNN 
knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                               ('classifier', KNeighborsClassifier(n_neighbors=5))]) 

start_time = time.time()
knn_pipeline.fit(X_train, y_train)
train_time_knn = time.time() - start_time

y_pred_knn = knn_pipeline.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_loss = log_loss(y_test, knn_pipeline.predict_proba(X_test))

conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
class_report_knn = classification_report(y_test, y_pred_knn)

print(f"kNN Accuracy: {knn_accuracy}")
print(f"kNN Loss: {knn_loss}")
print(f"Confusion Matrix:\n {conf_matrix_knn}")
print(f"Classification Report:\n {class_report_knn}")
print(f"kNN Training Time: {train_time_knn} seconds")