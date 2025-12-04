import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix

# dataset = pd.read_csv("/Users/mihailluzin/Documents/ПРОЕКТЫ/Хакатон /(2)test_data.csv")
dataset = pd.read_csv("/Users/watermaroon/Desktop/old-chem-shiiit/test_data.csv")
dataset.head()
dataset.info()
dataset.describe().T

dataset['Toxicity'].unique()

dataset['Toxicity'] = dataset['Toxicity'].replace('False', 0).replace('True', 1)
y = dataset['Toxicity']
X = dataset.drop(['Toxicity'], axis=1)


SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

rfc = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=SEED)

# Fit RandomForestClassifier
rfc.fit(X_train, y_train)
# Predict the test set labels
y_pred = rfc.predict(X_test)

features = X.columns.values  # The name of each column
classes = ['0', '1', '2']  # The name of each class
# You can also use low, medium and high risks in the same order instead
# classes = ['low risk', 'medium risk', 'high risk']

for estimator in rfc.estimators_:
    print(estimator)
    plt.figure(figsize=(12, 6))
    tree.plot_tree(estimator,
                   feature_names=features,
                   class_names=classes,
                   fontsize=8,
                   filled=True,
                   rounded=True)
    plt.show()


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d').set_title(
    'Maternal risks confusion matrix (0 = low risk, 1 = medium risk, 2 = high risk)')

print(classification_report(y_test, y_pred))

# Organizing feature names and importances in a DataFrame
features_df = pd.DataFrame({'features': rfc.feature_names_in_, 'importances': rfc.feature_importances_})

# Sorting data from highest to lowest
features_df_sorted = features_df.sort_values(by='importances', ascending=False)

# Barplot of the result without borders and axis lines
g = sns.barplot(data=features_df_sorted, x='importances', y='features', palette="rocket")
sns.despine(bottom=True, left=True)
g.set_title('Feature importances')
g.set(xlabel=None)
g.set(ylabel=None)
g.set(xticks=[])
for value in g.containers:
    g.bar_label(value, padding=2)

rfc_ = RandomForestClassifier(n_estimators=900, max_depth=7, random_state=SEED)
rfc_.fit(X_train, y_train)
y_pred = rfc_.predict(X_test)

cm_ = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_, annot=True, fmt='d').set_title(
    'Maternal risks confusion matrix (0 = low risk, 1 = medium risk, 2 = high risk) for 900 trees with 8 levels')

print(classification_report(y_test, y_pred))
