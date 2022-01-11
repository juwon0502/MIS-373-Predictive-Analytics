import streamlit as st

st.title("Decision Tree Classifier")
st.write("#### The decision tree classifier is one of the most basic but useful classification models\n")

st.write("Begin first by importing all necessary libraries")
with st.echo():
  # import all libraries
  import pandas as pd
  import numpy as np
  import arff
  from matplotlib import pyplot as plt
  from sklearn import tree
  from sklearn.tree import plot_tree, export_text
  from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, accuracy_score

# remove warnings on webstie
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("Import data")
with st.echo():
  # load data as pandas Dataframe
  training_arff = arff.load(open('../datasets/bank-training.arff'))
  testing_arff = arff.load(open('../datasets/bank-NewCustomers.arff'))
  col_val = [attribute[0] for attribute in training_arff['attributes']]
  training_df = pd.DataFrame(training_arff['data'], columns = col_val)
  testing_df = pd.DataFrame(testing_arff['data'], columns = col_val)
  meta = training_arff['attributes']

# cache data so computing time is saved
@st.cache(suppress_st_warning = True)
def clean_df(df):
  # decode str values
  cols = list(df.columns)
  for col in cols:
    try:
      df = df.replace({col: {'YES': True, 'NO': False}})
    except:
      pass
    pass
  return df

training_df = clean_df(training_df)
training_df_dummy = pd.get_dummies(training_df)
testing_df_dummy = pd.get_dummies(clean_df(testing_df))


st.write("Training Data:", training_df_dummy.head(10))

st.write("## Visualize Attributes")
# display attributes
def display_dt_attribute(df, meta, col_name):
  col_val = [item[0] for item in meta]
  pep = df.loc[df['pep'] == True]
  pep_col_name = []
  no_pep_col_name = []
  if type(meta[col_val.index(col_name)][1]) == list:
    labels = meta[col_val.index(col_name)][1]
    for label in labels:
      no_pep_col_name.append(len(df.loc[df[col_name] == label]))
      pep_col_name.append(len(pep.loc[pep[col_name] == label]))

  else:
    labels = []
    min_val = int(min(df[col_name]))
    max_val = int(max(df[col_name]))
    rg = max_val - min_val
    if rg < 12:
      for x in range(min_val, max_val + 1):
        no_pep_col_name.append(len(df.loc[df[col_name] == x]))
        pep_col_name.append(len(pep.loc[pep[col_name] == x]))
        labels.append(x)
    else:
      for y in range(min_val, max_val, (rg//8)):
        no_pep_col_name.append(len(df.loc[df[col_name].between(y, y + (rg//8))]))
        pep_col_name.append(len(pep.loc[pep[col_name].between(y, y + (rg//8))]))
        labels.append(f"{y}-{y+(rg//8-1)}")

  if type(labels[0]) != str:
    labels = [str(label) for label in labels]
  plt.figure(dpi = 200)
  plt.bar(labels, no_pep_col_name, label = 'No PEP')
  plt.bar(labels, pep_col_name, label = 'Yes PEP')
  plt.legend()
  plt.title(f'{col_name} distribution')
  plt.show()
  st.pyplot()
  
dt_option = st.selectbox("column", training_df.columns, index = 0)
display_dt_attribute(training_df, meta, dt_option)

# create model
st.title("Create Model")
st.write("Python Code to create model")
with st.echo():
  X = training_df_dummy.drop(columns=['pep'])
  y = training_df_dummy.pep
  max_depth = st.number_input(label = 'max_depth', value = 5, min_value = 1, step = 1)
  clf = tree.DecisionTreeClassifier(max_depth = max_depth, criterion = 'entropy')
  model = clf.fit(X,y)

st.write("Adjust depth of tree")
# plot tree
plt.figure(figsize = (20, 12), dpi = 150)
tree.plot_tree(model, fontsize = 15, feature_names = X.columns, impurity = True,
               class_names = ["True", "False"], label = 'root', filled = True)
plt.show()
st.pyplot()

st.write("#### Classification Accuracy: ", round(float(accuracy_score(model.predict(X), y)),4))
st.write("\n\n")

# evaluate model performance
st.title("Model Evaluation")
st.write("#### Evaluated on an out-of-sample test set\n")
y_test = testing_df_dummy.pep
X_test = testing_df_dummy.drop(columns=['pep'])
# y_pred = model.predict(X_test)

if st.checkbox('Confusion Matrix'):
  st.subheader("Confusion Matrix") 
  plot_confusion_matrix(model, X_test, y_test)
  st.pyplot()

if st.checkbox("ROC Curve"):
  st.subheader("ROC Curve") 
  plot_roc_curve(model, X_test, y_test)
  st.pyplot()

###############################################################################################################################################

st.title("Ensemble Models")
st.title("Bagging + Random Forest Model")
# st.write("#### The decision tree classifier is one of the most basic but useful classification models\n")

st.write("Begin first by importing all necessary libraries")
with st.echo():
  # import all libraries
  import pandas as pd
  import numpy as np
  import arff
  from matplotlib import pyplot as plt
  from sklearn import tree
  from sklearn.ensemble import BaggingClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.tree import plot_tree, export_text, DecisionTreeClassifier
  from sklearn.model_selection import train_test_split, cross_val_score
  from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, accuracy_score

# remove warnings on webstie
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("Import data")
with st.echo():
  # load data as pandas Dataframe
  hr_arff = arff.load(open('../datasets/HR_employee_attrition.arff'))
  col_val = [attribute[0] for attribute in hr_arff['attributes']]
  hr_df = pd.DataFrame(hr_arff['data'], columns = col_val)
  meta = hr_arff['attributes']

# cache data so computing time is saved
@st.cache(suppress_st_warning = True)
def clean_ens_df(df):
  # decode str values
  cols = list(df.columns)
  for col in cols:
    try:
      df = df.replace({col: {'Yes': 1, 'No': 0}})
    except:
      pass
  return df

hr_df = clean_ens_df(hr_df)
hr_df_dummies = pd.get_dummies(hr_df)

st.write("Training Data:", hr_df_dummies.head(10))

st.write("## Visualize Attributes")
# display attributes
def display_ensemble_attribute(df, meta, col_name):
  att = df.loc[df['Attrition'] == 1]
  att_col_name = []
  no_att_col_name = []
  if type(meta[col_val.index(col_name)][1]) == list:
    labels = meta[col_val.index(col_name)][1]
    for label in labels:
      no_att_col_name.append(len(df.loc[df[col_name] == label]))
      att_col_name.append(len(att.loc[att[col_name] == label]))

  else:
    labels = []
    min_val = int(min(df[col_name]))
    max_val = int(max(df[col_name]))
    rg = max_val - min_val
    if rg < 12:
      for x in range(min_val, max_val + 1):
        no_att_col_name.append(len(df.loc[df[col_name] == x]))
        att_col_name.append(len(att.loc[att[col_name] == x]))
        labels.append(x)
    else:
      for y in range(min_val, max_val, (rg//8)):
        no_att_col_name.append(len(df.loc[df[col_name].between(y, y + (rg//8))]))
        att_col_name.append(len(att.loc[att[col_name].between(y, y + (rg//8))]))
        labels.append(f"{y}-{y+(rg//8-1)}")

  if type(labels[0]) != str:
    labels = [str(label) for label in labels]
  plt.figure(dpi = 300)
  plt.bar(labels, no_att_col_name, label = 'No attrition')
  plt.bar(labels, att_col_name, label = 'Yes attrition')
  plt.legend()
  plt.title(f'{col_name} distribution')
  plt.show()
  st.pyplot()
  

esb_option = st.selectbox("column", hr_df.columns, index = 0)
display_ensemble_attribute(hr_df, meta, esb_option)

# create model
st.title("Create Bagging Model")
st.write("Python Code to create model")
with st.echo():
  X = hr_df_dummies.drop(columns=['Attrition'])
  y = hr_df_dummies.Attrition
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
  n_estimators = st.number_input(label = "Number of estimators", min_value = 5, max_value = 30)
  bagging_model = BaggingClassifier(base_estimator=clf, n_estimators=n_estimators, random_state=0).fit(X, y)

# create model
st.title("Create Random Forest Model")
st.write("Python Code to create model")
with st.echo():
  X = hr_df_dummies.drop(columns=['Attrition'])
  y = hr_df_dummies.Attrition
  max_features = st.number_input(label = 'Number of features', min_value = 4, max_value = 8, value = 4)
  rf_model = RandomForestClassifier(criterion = 'entropy', max_features = max_features).fit(X,y)


# st.write("#### Classification Accuracy: ", round(float(accuracy_score(model.predict(X_test), y_test)),4))
# st.write("#### Classification Accuracy: ", round(float(cross_validation(model.predict(X), y)),4))
with st.echo():
  # Cross validation for accuracy
  bagging_scores = cross_val_score(bagging_model, X, y, cv=10)
  bagging_accuracy = bagging_scores.mean()
  bagging_std = bagging_scores.std()

  rf_scores = cross_val_score(rf_model, X, y, cv=10)
  rf_accuracy = rf_scores.mean()
  rf_std = rf_scores.std()

st.write(f"#### Bagging: {round(bagging_accuracy,4)} accuracy with a standard deviation of {round(bagging_std,4)}")
st.write(f"#### Random Forest: {round(rf_accuracy,4)} accuracy with a standard deviation of {round(rf_std,4)}")

dt_model = DecisionTreeClassifier(criterion = 'entropy').fit(X,y)
dt_scores = cross_val_score(dt_model, X, y, cv=10)
st.write(f"#### Decision Tree: {round(dt_scores.mean(),4)} accuracy with a standard deviation of {round(dt_scores.std(),4)}")

st.write("\n\n")

# ###############################################################################################################################################

# st.title("Random Forest Model")
# # st.write("#### The decision tree classifier is one of the most basic but useful classification models\n")


# # remove warnings on webstie
# st.set_option('deprecation.showPyplotGlobalUse', False)

# st.write("Import data")
# with st.echo():
#   # load data as pandas Dataframe
#   hr_arff = arff.load(open('../datasets/HR_employee_attrition.arff'))
#   col_val = [attribute[0] for attribute in hr_arff['attributes']]
#   hr_df = pd.DataFrame(hr_arff['data'], columns = col_val)
#   meta = hr_arff['attributes']


# st.write("Training Data:", hr_df_dummies.head(10))

# st.write("## Visualize Attributes")
# # display attributes
# def display_attribute(df, meta, col_name):
#   pep = df.loc[df['Attrition'] == 1]
#   pep_col_name = []
#   no_pep_col_name = []
#   if type(meta[col_val.index(col_name)][1]) == list:
#     labels = meta[col_val.index(col_name)][1]
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
  

# option = st.selectbox("column", hr_df.columns, index = 0)
# display_attribute(hr_df, meta, option)

# # create model
# st.title("Create Model")
# st.write("Python Code to create model")
# with st.echo():
#   X = hr_df_dummies.drop(columns=['Attrition'])
#   y = hr_df_dummies.Attrition
#   max_features = st.number_input(label = 'Number of features', min_value = 4, max_value = 8, value = 4)
#   model = RandomForestClassifier(criterion = 'entropy', max_features = max_features).fit(X,y)


# # st.write("#### Classification Accuracy: ", round(float(accuracy_score(model.predict(X_test), y_test)),4))
# # st.write("#### Classification Accuracy: ", round(float(cross_validation(model.predict(X), y)),4))
# with st.echo():
#   # Cross validation for accuracy
#   scores = cross_val_score(model, X, y, cv=10)
#   accuracy = scores.mean()
#   std = scores.std()
# st.write(f"#### {round(accuracy,4)} accuracy with a standard deviation of {round(std,4)}")

# st.write("### Compare to regular Decision Tree")
# model = DecisionTreeClassifier(criterion = 'entropy').fit(X,y)
# scores = cross_val_score(model, X, y, cv=10)
# st.write(scores)
# st.write(f"#### {round(scores.mean(),4)} accuracy with a standard deviation of {round(scores.std(),4)}")

# st.write("\n\n")