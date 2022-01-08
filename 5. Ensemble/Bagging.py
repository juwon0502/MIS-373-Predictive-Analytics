import streamlit as st

st.title("Bagging Model")
# st.write("#### The decision tree classifier is one of the most basic but useful classification models\n")

st.write("Begin first by importing all necessary libraries")
with st.echo():
  # import all libraries
  import pandas as pd
  import numpy as np
  from scipy.io import arff
  from matplotlib import pyplot as plt
  from sklearn import tree
  from sklearn.ensemble import BaggingClassifier
  from sklearn.tree import plot_tree, export_text
  from sklearn.model_selection import train_test_split, cross_val_score
  from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, accuracy_score

# remove warnings on webstie
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("Import data")
with st.echo():
  # load data as pandas Dataframe
  arff = arff.loadarff('../datasets/HR_employee_attrition.arff')
  arff_df = pd.DataFrame(arff[0])
  meta = arff[1]

# cache data so computing time is saved
@st.cache(suppress_st_warning = True)
def clean_df(df):
  # decode str values
  cols = list(df.columns)
  for col in cols:
    try:
      df[col] = df[col].str.decode('utf-8')
    except:
      df[col] = pd.to_numeric(df[col])
      pass
    try:
      df = df.replace({col: {'Yes': 1, 'No': 0}})
    except:
      pass
  # return pd.get_dummies(df)
  return df

df = pd.get_dummies(clean_df(arff_df))

st.write("Training Data:", df.head(10))

# st.write("## Visualize Attributes")
# # display attributes
# def display_attribute(df, meta, col_name):
#   pep = df.loc[df['Attrition'] == 1]
#   pep_col_name = []
#   no_pep_col_name = []
#   if meta.types()[meta.names().index(col_name)] == 'nominal':
#     labels = get_labels(col_name)
#     for label in labels:
#       no_pep_col_name.append(len(df.loc[df[col_name] == label]))
#       pep_col_name.append(len(pep.loc[pep[col_name] == label]))

#   else:
#     labels = []
#     min_val = int(min(df[col_name]))
#     max_val = int(max(df[col_name]))
#     rg = max_val - min_val
#     if rg < 12:
#       for x in range(min_val, max_val + 1):
#         no_pep_col_name.append(len(df.loc[df[col_name] == x]))
#         pep_col_name.append(len(pep.loc[pep[col_name] == x]))
#         labels.append(x)
#     else:
#       for y in range(min_val, max_val, (rg//8)):
#         no_pep_col_name.append(len(df.loc[df[col_name].between(y, y + (rg//8))]))
#         pep_col_name.append(len(pep.loc[pep[col_name].between(y, y + (rg//8))]))
#         labels.append(f"{y}-{y+(rg//8-1)}")

#   if type(labels[0]) != str:
#     labels = [str(label) for label in labels]
#   plt.figure(dpi = 300)
#   plt.bar(labels, no_pep_col_name, label = 'No attrition')
#   plt.bar(labels, pep_col_name, label = 'Yes attrition')
#   plt.legend()
#   plt.title(f'{col_name} distribution')
#   plt.show()
#   st.pyplot()
  
# def get_labels(col_name):
#   label = []
#   x = meta.names().index(col_name) + 1
#   values = (str(meta).split('\n')[x].split("range is ")[1].lstrip('(').rstrip(')').split(','))
#   for value in values:
#     label.append(value.strip().strip("''"))
#   if label == ['NO', 'YES']:
#     return [False, True]
#   return label

# option = st.selectbox("column", arff_df.columns, index = 0)
# display_attribute(arff_df, meta, option)

# create model
st.title("Create Model")
st.write("Python Code to create model")
with st.echo():
  X = df.drop(columns=['Attrition'])
  y = df.Attrition
  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
  # max_depth = st.number_input(label = 'max_depth', value = 5, min_value = 1, step = 1)
  clf = tree.DecisionTreeClassifier(criterion = 'entropy')
  n_estimators = st.number_input(label = "Number of estimators", min_value = 5, max_value = 30)
  model = BaggingClassifier(base_estimator=clf,n_estimators=n_estimators, random_state=0).fit(X, y)


# st.write("#### Classification Accuracy: ", round(float(accuracy_score(model.predict(X_test), y_test)),4))
# st.write("#### Classification Accuracy: ", round(float(cross_validation(model.predict(X), y)),4))
scores = cross_val_score(model, X, y, cv=10)
st.write(f"#### {round(scores.mean(),4)} accuracy with a standard deviation of {round(scores.std(),4)}")

st.write("\n\n")

# # evaluate model performance
# st.title("Model Evaluation")
# st.write("#### Evaluated on an out-of-sample test set\n")
# y_test = testing_df_dummy.pep
# X_test = testing_df_dummy.drop(columns=['pep'])
# # y_pred = model.predict(X_test)

# if st.checkbox('Confusion Matrix'):
#   st.subheader("Confusion Matrix") 
#   plot_confusion_matrix(model, X_test, y_test)
#   st.pyplot()

# if st.checkbox("ROC Curve"):
#   st.subheader("ROC Curve") 
#   plot_roc_curve(model, X_test, y_test)
#   st.pyplot()
