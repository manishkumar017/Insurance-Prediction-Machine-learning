import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import streamlit as st
#this streamlit is for web based application project


#web page code
st.title("HEALTH INSURANCE PREDICION")
img_url = "https://cdn.zeebiz.com/sites/default/files/2026/03/09/401943-health-insurance.png?im=FitAndFill=(848,477)&format=webp&quality=medium"
st.image(img_url)

#load data and ml model part
# step2:load insurance data
url = "https://raw.githubusercontent.com/ankitmisk/UIT-data/refs/heads/main/Insurance.csv"
df = pd.read_csv(url)


#Step 3: EDA
#to get the top 5 rows
df.drop("Customer_ID", axis = 1, inplace = True)
df['Previous_Insurance'] = df['Previous_Insurance'].map({'No': 0, 'Yes': 1})
df['Insurance_Bought'] = df['Insurance_Bought'].map({'No': 0, 'Yes': 1})

# Step 4:divide data set into features and target
#iloc : iNdexed ,baSed searching loacation
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Step 5: divide data imto traiNiNg / teStiNg
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42, test_size =0.3)

#Step 6 : Train Model
model = LogisticRegression()
model.fit(X_train, y_train)
#fit-methods :it understand the data pattern and build model


#show data sample 
st.write(df.head())

#create side bar for uver input form
st.sidebar.title("Fill Customer Details")
st.sidebar.image.(img_url)

#to get user input
all_ns = []
for index, col_name in enumerate(X.columns):
  min_y = X[col_name].min()
  max_v = X[col_name].max()
  if col_name != "Previous_Insurance":
    value = st.sidebar.slider(f"Select value for {col_name}",
                             min_value = min_v,
                             max_value = max_v)
  else:
    value = st.sidebar.number_input(f"Select value for {col_name}: ")
  all_ans.append(value)
ud = {j:all_ans[i] for i,j in enumerate(X.columns)}
user_df = pd.DataFrame(ud, index = [1])
sr.write(user_df)



#=======================Prediction====================
if(st.button(" click to Predict: "):
  with  st.spinner("Predicting.."):
    import time
    time.sleep(2)
  final_ans = model.predict([all_ans])[0]
  if final_ans == 0:
    st.info("❌Customer will not buy the Insurance❌")
  else:
    st.success("✅customer will buy the insurance✅")
    
