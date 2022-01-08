import streamlit as st

st.title("Model Evaluation")
st.write("#### Model evaluation\n")

st.write("Begin first by importing all necessary libraries")
with st.echo():
  # import all libraries
  import pandas as pd
  import numpy as np
  from scipy.io import arff
  from matplotlib import pyplot as plt
  from sklearn import tree
  from sklearn.metrics import accuracy_score
  from sklearn.model_selection import train_test_split
  from sklearn.model_selection import cross_val_score

st.write("Import data")
with st.echo():
  # load data as pandas Dataframe
  training_arff = arff.loadarff('./datasets/bank-training.arff')
  testing_arff = arff.loadarff('./datasets/bank-NewCustomers.arff')
  training_df = pd.DataFrame(training_arff[0])
  testing_df = pd.DataFrame(testing_arff[0])
  meta = training_arff[1]

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
      df = df.replace({col: {'YES': True, 'NO': False}})
    except:
      pass
  # return pd.get_dummies(df)
  return df

training_df = clean_df(training_df)
training_df_dummy = pd.get_dummies(training_df)
testing_df_dummy = pd.get_dummies(clean_df(testing_df))

st.write("Split training data into testing and training")
st.write(training_df_dummy.head(10))
slider = st.slider(min_value = 0.0, max_value = 1.0, step = 0.05, value = 0.65, label = 'Train/Test Split Percentage')

with st.echo():
    X = training_df_dummy.drop(columns = ['pep'])
    y = training_df_dummy.pep
    if slider == 1.0:
      X_train = X_test = X
      y_train = y_test = y
    else:
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (1 - slider))
    
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
model = clf.fit(X_train,y_train) 
st.write(len(X_train))
st.write("#### Classification Accuracy: ", round(float(accuracy_score(model.predict(X_test), y_test)),4))

st.write("### Cross Validation")

with st.echo():
  num_folds = st.number_input(label = "number of folds", min_value = 1, max_value = len(X) + 1, value = 5)
  scores = cross_val_score(clf, X, y, cv=num_folds)
  st.write(scores)
  st.write(f"{round(scores.mean(),4)} accuracy with a standard deviation of {round(scores.std(),4)}")