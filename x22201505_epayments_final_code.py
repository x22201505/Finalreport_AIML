#!/usr/bin/env python
# coding: utf-8

# # Online Payments Fraud Detection

# # Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Import dataset

# In[2]:


epayment = pd.read_csv('C:/Users/user/Desktop/dataset.csv')
epayment


# # Exploratory data analysis 

# In[4]:


epayment.head()


# In[4]:


epayment.info()   


# In[5]:


epayment.isnull().sum()


# In[6]:


epayment['isFraud'].value_counts()


# In[7]:


epayment.describe()


# In[3]:


import plotly.express as px
type_transaction = epayment["type"].value_counts()
transaction = type_transaction.index
quantity = type_transaction.values

custom_colors = px.colors.qualitative.Set1
fig = px.pie(epayment,
             values=quantity,
             names=transaction,
             hole=0.7,
             title="Distribution of Transaction Type",
             color_discrete_sequence=custom_colors)
fig.show()


# In[4]:


import plotly.express as px
fig = px.histogram(epayment, x="amount", title="Transaction Amount Distribution")
fig.show()


# In[49]:


epayment['isFraud'].value_counts().plot.pie(autopct='%1.1f%%');


# In[60]:


#distribuition of FRAUD transactions
import seaborn as sns
ax = sns.lmplot(y="amount", x="type", fit_reg=False,aspect=1.8,
                data=epayment, hue='isFraud')
plt.title("Frauds and Normal Transactions",fontsize=16)
plt.show()


# In[10]:


corr = epayment.corr(numeric_only=True)
corr["isFraud"].sort_values(ascending=False)


# In[59]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix')
plt.show()


# In[11]:


epayment["type"] = epayment["type"].map({"CASH_OUT": 1, 
                                 "PAYMENT": 2, 
                                 "CASH_IN": 3, 
                                 "TRANSFER": 4,
                                 "DEBIT": 5})
epayment["isFraud"] = epayment["isFraud"].map({0: "No Fraud", 1: "Fraud"})
epayment.head()


# # MODEL BUILDING

# # logistic regression

# In[13]:


from sklearn.model_selection import train_test_split

#  independent variables and target variable
indep_variables = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
X = epayment[indep_variables]
y = epayment['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=22201505)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# In[14]:


epay = epayment.copy()
epay


# In[15]:


epay.info()


# In[16]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
epay['nameOrig'] = label_encoder.fit_transform(epay['nameOrig'])
epay['nameDest'] = label_encoder.fit_transform(epay['nameDest'])
epay['isFraud'] = label_encoder.fit_transform(epay['isFraud'])


# In[17]:


epay.info()


# In[19]:


from sklearn.model_selection import train_test_split
indep_variables = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
X = epay[indep_variables]
y = epay['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=22201505)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Creating a logistic regression model
logistic_reg = LogisticRegression()
# Fitting the model on the training data
logistic_reg.fit(X_train, y_train)
# Making predictions on the testing data
y_pred = logistic_reg.predict(X_test)
# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[21]:


from sklearn.metrics import confusion_matrix
# Generating the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[22]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(conf_matrix, annot=True)
plt.title('Confusion Matrix - Test Data')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')


# In[43]:


from sklearn.metrics import precision_score, f1_score
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1] 
threshold = 0.5
y_pred_binary = (y_pred_proba > threshold).astype(int)
precision = precision_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
print("Precision:", precision)
print("F1 Score:", f1)


# # DECISION TREE

# In[25]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=7)
# Fit the regressor to the training data
dt.fit(X_train, y_train)


# In[24]:


#Evaluation model performance


# In[26]:


y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)


# In[27]:


from sklearn.metrics import mean_squared_error, r2_score
print("Mean Squared Error:", mean_squared_error(y_train, y_train_pred))
print("R-squared Score:", r2_score(y_train, y_train_pred))


# In[28]:


#Above r square value is underfitting


# In[29]:


from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
import numpy as np
indep_variables = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
X = epay[indep_variables]
y = epay['isFraud']
model = DecisionTreeRegressor()
k = 3
kf = KFold(n_splits=k, shuffle=True, random_state=70)
# Perform k-fold cross-validation
fold_accuracies = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test) 
    fold_accuracies.append(accuracy)
# Calculate the average accuracy across all folds
avg_accuracy = np.mean(fold_accuracies)
print("Average R-squared:", avg_accuracy)


# In[30]:


from sklearn.tree import DecisionTreeRegressor
regularized_model = DecisionTreeRegressor(max_depth=15, min_samples_split=40, min_samples_leaf=20)
# Train the regularized model on the dataset
regularized_model.fit(X, y)


# In[ ]:


# Calculate R-squared value
r_squared = regularized_model.score(X, y)

print("R-squared value:", r_squared)


# In[42]:


from sklearn.metrics import precision_score, f1_score
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train, y_train)
y_pred = tree_classifier.predict(X_test)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Precision:", precision)
print("F1 Score:", f1)


# In[41]:


#Tree Node


# In[32]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
feature_names_list = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
plt.figure(figsize=(20, 15))
plot_tree(dt,
          filled=True,                 
          feature_names=feature_names_list, 
          class_names=['Fraud','No Fraud Detected'],  
          rounded=True,               
          fontsize=10)                  
plt.show()


# # Gradient Boosting Machine

# In[37]:


# Import necessary libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X, y = make_classification(n_samples=10000, n_features=11, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, max_depth=6, random_state=42)
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[46]:


#Gradient Boosting Machine

from sklearn.metrics import classification_report
gbm = GradientBoostingClassifier()
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)
# Generate classification report
report = classification_report(y_test, y_pred)
print(report)


# In[47]:


#Decision Tree

tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train, y_train)
y_pred = tree_classifier.predict(X_test)
# Generate classification report
report = classification_report(y_test, y_pred)
print(report)



# In[48]:


#logistic regression

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
# Generate classification report
report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print(report)


# In[69]:


import matplotlib.pyplot as plt
model_names = ['Logistic Regression', 'Decision Tree', 'Gradient Boosting Machine']
accuracy_scores = [0.90, 0.90, 0.94]  

# Create bar chart
plt.figure(figsize=(6, 5))
plt.bar(model_names, accuracy_scores, color=['blue', 'green', 'orange'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.ylim(0, 1)  
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.tight_layout()  
plt.show()


# In[ ]:




