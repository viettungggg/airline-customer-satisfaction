import pandas as pd
from sklearn.preprocessing import StandardScaler

train_data_path = './data/train.csv'
test_data_path = './data/test.csv'

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

train_df = train_df.drop(train_df.columns[[0, 1]], axis=1)
test_df = test_df.drop(test_df.columns[[0, 1]], axis=1)

train_df = train_df.dropna()
test_df = test_df.dropna()

X_train = train_df.drop('satisfaction', axis=1)
y_train = train_df['satisfaction']
X_test = test_df.drop('satisfaction', axis=1)
y_test = test_df['satisfaction']

X_train_encoded = pd.get_dummies(X_train, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'])
X_test_encoded = pd.get_dummies(X_test, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'])

X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='inner', axis=1)

numerical_columns = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                     'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
                     'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
                     'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
                     'Departure Delay in Minutes', 'Arrival Delay in Minutes']

scaler = StandardScaler()
X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()

X_train_scaled[numerical_columns] = scaler.fit_transform(X_train_encoded[numerical_columns])
X_test_scaled[numerical_columns] = scaler.transform(X_test_encoded[numerical_columns])

X_train_scaled.to_csv('data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/X_test_scaled.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)