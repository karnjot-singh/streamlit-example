import streamlit as st
#from sklearn import tree
from joblib import load
from sklearn import datasets

st.title("ML model deploy demo")
clf = load("iris.joblib")

iris = datasets.load_iris()

labels = iris.target_names
inputs = iris.feature_names

input1 = st.slider(inputs[0], max_value=10)
input2 = st.slider(inputs[1], max_value=10)
input3 = st.slider(inputs[2], max_value=10)
input4 = st.slider(inputs[3], max_value=10)

prediction = clf.predict([[input1, input2, input3, input4]])

st.write(labels[prediction[0]])