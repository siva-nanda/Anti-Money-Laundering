import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

# Load dataset
df = pd.read_csv( "ABC.csv")
df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)

# One-hot encoding for categorical variable
df = pd.get_dummies(df, columns=['type'])

df_sample = df.sample(n=20000)
X_train, X_test, y_train, y_test = train_test_split(df_sample.drop(['isFraud'], axis=1), df_sample['isFraud'], test_size=0.3, random_state=0)

# Oversampling with SMOTE
sm = SMOTE(random_state=10, sampling_strategy=1.0)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Feature scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train_res)
X_test_scaled = sc.transform(X_test)

# Model definition
model = Sequential([
    Dense(6, input_dim=11, activation='relu', kernel_initializer='uniform'),
    Dense(6, activation='relu', kernel_initializer='uniform'),
    Dense(1, activation='sigmoid', kernel_initializer='uniform')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train_res, batch_size=10, epochs=10)

# Predict and evaluate
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

print('Accuracy:', round(accuracy_score(y_test, y_pred) * 100, 2))
print('Confusion matrix:', confusion_matrix(y_test, y_pred))
print('Classification report:', classification_report(y_test, y_pred))
