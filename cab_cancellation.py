#!/usr/bin/env python
# coding: utf-8

# In[46]:


#Loading the important libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from dmba import classificationSummary
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[2]:


#LoadingA the data
df = pd.read_csv('YourCabs_training.csv')


# In[3]:


df #print the data


# In[5]:


# Check for null values in each column
null_columns = df.isnull().sum()

# Filter the columns with null values
columns_with_null = null_columns[null_columns > 0]

# Print the columns with null values
print(columns_with_null)


# In[9]:


df=df.drop(['id','user_id','package_id','to_area_id','from_city_id','to_city_id','to_date'],axis=1)


# In[10]:


df


# In[12]:


null_values = df.isnull().sum()
print(null_values)


# In[13]:


# Replace null values with median in specific columns
columns_to_fill = ['from_area_id', 'from_lat', 'from_long', 'to_lat', 'to_long']
for column in columns_to_fill:
    median = df[column].median()
    df[column].fillna(median, inplace=True)

df


# In[14]:


null_values = df.isnull().sum()
print(null_values)


# In[17]:


# Convert booking_created column to datetime
df['booking_created'] = pd.to_datetime(df['booking_created'])

# Extract day of the week, time, month, and year into separate columns
df['booking_dayofweek'] = df['booking_created'].dt.strftime('%A')
df['booking_time'] = df['booking_created'].dt.time
df['booking_month'] = df['booking_created'].dt.month
df['booking_year'] = df['booking_created'].dt.year


# In[20]:


df.head(10)


# # Neural Network model

# In[26]:


predictors = ['vehicle_model_id','online_booking','mobile_site_booking','from_area_id','from_lat','from_long','to_lat','to_long','booking_dayofweek','booking_time','booking_month','booking_year']
outcome = 'Car_Cancellation'
X = df[predictors]
y = df[outcome]


# In[32]:


# Perform one-hot encoding on categorical columns
train_X_encoded = pd.get_dummies(train_X, columns=['booking_dayofweek'])
valid_X_encoded = pd.get_dummies(valid_X, columns=['booking_dayofweek'])

# Convert booking_time column to seconds since midnight
train_X_encoded['booking_time'] = train_X_encoded['booking_time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
valid_X_encoded['booking_time'] = valid_X_encoded['booking_time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

# Train neural network with 2 hidden nodes
clf = MLPClassifier(hidden_layer_sizes=(2), activation='logistic', solver='lbfgs', random_state=1)
clf.fit(train_X_encoded, train_y.values)


# In[36]:


# Encode categorical labels to numeric values
label_encoder = LabelEncoder()
train_y_encoded = label_encoder.fit_transform(train_y)
valid_y_encoded = label_encoder.transform(valid_y)

# Training performance
classificationSummary(train_y_encoded, clf.predict(train_X_encoded))

# Validation performance
classificationSummary(valid_y_encoded, clf.predict(valid_X_encoded))


# In[38]:


# Obtain the predicted labels for the training and validation sets
train_y_pred = clf.predict(train_X_encoded)
valid_y_pred = clf.predict(valid_X_encoded)

# Encode categorical labels to numeric values
train_y_encoded = label_encoder.inverse_transform(train_y_encoded)
valid_y_encoded = label_encoder.inverse_transform(valid_y_encoded)

# Create confusion matrices
train_cm = confusion_matrix(train_y_encoded, train_y_pred)
valid_cm = confusion_matrix(valid_y_encoded, valid_y_pred)

# Plot the heatmaps
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Training set heatmap
sns.heatmap(train_cm, annot=True, cmap='Blues', fmt='g',
            xticklabels=['Prediction 0', 'Prediction 1'],
            yticklabels=['Actual 0', 'Actual 1'], ax=axes[0])
axes[0].set_title('Training Set Confusion Matrix')

# Validation set heatmap
sns.heatmap(valid_cm, annot=True, cmap='Blues', fmt='g',
            xticklabels=['Prediction 0', 'Prediction 1'],
            yticklabels=['Actual 0', 'Actual 1'], ax=axes[1])
axes[1].set_title('Validation Set Confusion Matrix')

# Adjust the layout
plt.tight_layout()

# Display the heatmaps
plt.show()


# # Logistic regression

# In[42]:


# Define the predictors and outcome variable
predictors = ['vehicle_model_id', 'online_booking', 'mobile_site_booking', 'from_area_id', 'from_lat', 'from_long', 'to_lat', 'to_long', 'booking_dayofweek', 'booking_time', 'booking_month', 'booking_year']
outcome = 'Car_Cancellation'
X = df[predictors]
y = df[outcome]

# Perform one-hot encoding on categorical columns
X_encoded = pd.get_dummies(X, columns=['booking_dayofweek'])

# Convert booking_time column to string
X_encoded['booking_time'] = X_encoded['booking_time'].astype(str)

# Convert booking_time column to seconds since midnight
X_encoded['booking_time'] = pd.to_datetime(X_encoded['booking_time']).dt.hour * 3600 + pd.to_datetime(X_encoded['booking_time']).dt.minute * 60 + pd.to_datetime(X_encoded['booking_time']).dt.second


# In[43]:


# Encode the outcome variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=1)


# In[44]:


# Create an instance of the LogisticRegression model
logreg = LogisticRegression()

# Fit the model to the training data
logreg.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = logreg.predict(X_train)
y_test_pred = logreg.predict(X_test)


# In[47]:


# Print the classification report for training set
print("Training Set Classification Report:")
print(classification_report(y_train, y_train_pred))

# Print the classification report for test set
print("Test Set Classification Report:")
print(classification_report(y_test, y_test_pred))

# Print the confusion matrix for test set
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))


# In[49]:


# Compute the confusion matrix
cm = confusion_matrix(train_y_encoded, clf.predict(train_X_encoded))

# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

# Add labels, title, and ticks
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Training Set")
plt.xticks([0, 1], labels=["Class 0", "Class 1"])
plt.yticks([0, 1], labels=["Class 0", "Class 1"])

# Show the plot
plt.show()


# In[ ]:




