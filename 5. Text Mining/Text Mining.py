import streamlit as st

st.title("Text Mining Model")
# st.write("#### The decision tree classifier is one of the most basic but useful classification models\n")

st.write("Begin first by importing all necessary libraries")
with st.echo():
  # import all libraries
  import pandas as pd
  import numpy as np
  import arff
  from matplotlib import pyplot as plt
  from sklearn import tree
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import train_test_split, cross_val_score
  from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, accuracy_score
  from sklearn.feature_extraction.text import CountVectorizer

# remove warnings on webstie
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("Import data")
with st.echo():
  # load data as pandas Dataframe
  movie_arff = arff.load(open('../datasets/Movie_reviews-sentiments.arff'))
  col_val = [attribute[0] for attribute in movie_arff['attributes']]
  movie_df = pd.DataFrame(movie_arff['data'], columns = col_val)

# cache data so computing time is saved
@st.cache(suppress_st_warning = True)
def clean_df(df):
  # decode str values
  cols = list(df.columns)
  for col in cols:
    try:
      df = df.replace({col: {'Yes': 1, 'No': 0}})
    except:
      pass
  # return pd.get_dummies(df)
  return df

st.write(movie_df.head(20))

st.write("### Bag of Words Vectorization")

with st.echo():
  # corpus = movie_df.text
  vectorizer = CountVectorizer(binary = True)
  train_X = vectorizer.fit_transform(movie_df.text)
  st.write(train_X[0])
  # st.write(vectorizer.get_feature_names())
  # st.write(X)

# df = pd.get_dummies(clean_df(arff_df))

# st.write("Training Data:", df.head(10))

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

# # create model
# st.title("Create Model")
# st.write("Python Code to create model")
# with st.echo():
#   X = df.drop(columns=['Attrition'])
#   y = df.Attrition
#   # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
#   # max_depth = st.number_input(label = 'max_depth', value = 5, min_value = 1, step = 1)
#   max_features = st.number_input(label = 'Number of features', min_value = 4, max_value = 8, value = 4)
#   model = RandomForestClassifier(criterion = 'entropy', max_features = max_features).fit(X,y)


# st.write("#### Classification Accuracy: ", round(float(accuracy_score(model.predict(X_test), y_test)),4))
# st.write("#### Classification Accuracy: ", round(float(cross_validation(model.predict(X), y)),4))
scores = cross_val_score(model, X, y, cv=10)
st.write(f"#### {round(scores.mean(),4)} accuracy with a standard deviation of {round(scores.std(),4)}")

# @st.cache(suppress_st_warning = True)
# def four_to_six():
#   for i in range(4, 7):
#       X = df.drop(columns=['Attrition'])
#       y = df.Attrition
#       max_features = i
#       model = RandomForestClassifier(criterion = 'entropy', max_features = max_features).fit(X,y)
#       scores = cross_val_score(model, X, y, cv=10)
#       st.write(f"{i} features has an accuracy of {round(scores.mean(),4)}")
#       return

# four_to_six()

st.write("### Compare to regular Decision Tree")
model = DecisionTreeClassifier().fit(X,y)
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
