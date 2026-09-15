# Step 1: Load Important Modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import streamlit as st
# this streamlit is for web based application project


# Web Page Code
st.title("HEALTH INSURANCE PREDICTION")
img_url = "https://cdn.zeebiz.com/sites/default/files/2026/03/09/401943-health-insurance.png"
st.image(img_url)

# LOAD DATA and ML MODEL PART

# Step 2: load Insurance data
url = "https://raw.githubusercontent.com/ankitmisk/UIT-data/refs/heads/main/Insurance.csv"
df = pd.read_csv(url)

# Step 3: EDA: Exploratory Data Analysis
df.drop("Customer_ID", axis = 1, inplace = True)

df['Previous_Insurance'] = df['Previous_Insurance'].map({'No':0,"Yes":1})
df['Insurance_Bought'] = df['Insurance_Bought'].map({'No':0,"Yes":1})


# Step 4: Divide dataset into features and target
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Step 5: Divide data into Training & testing part
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.3)

# Step  6: Train Model
model = LogisticRegression()
model.fit(X_train,y_train)



# show data sample
st.write(df.head())
# Create Side bar for user input form
st.sidebar.title("Fill Customer Details")
st.sidebar.image(img_url)


# To get user input
all_ans = []
for index, col_name in enumerate(X.columns):
  min_v = X[col_name].min()
  max_v = X[col_name].max()
  if col_name != "Previous_Insurance":
    value = st.sidebar.slider(f"Select value for {col_name}",
                             min_value = min_v,
                             max_value = max_v)
  else:
    value = st.sidebar.number_input(f"Select value for {col_name} (0:No, 1: yes): ")

  all_ans.append(value)

ud = {j:all_ans[i] for i,j in enumerate(X.columns)}
user_df = pd.DataFrame(ud, index = [1])
st.write(user_df)

#=========================Prediction==============
if st.button("Click to Predict: "):
  with st.spinner("Predicting.."):
    import time
    time.sleep(2)
  final_ans = model.predict([all_ans])[0]
  if final_ans == 0:
    st.info("❎Customer will not Buy the Insurance❎")
  else:
    st.success("✅Customer will buy the Insurance✅")
