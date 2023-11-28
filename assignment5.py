#Ahnaf Ahmad
#1001835014

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


data = pd.read_csv('spambase.data', header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X = (X - X.mean()) / X.std()

model = Sequential()

model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=50, batch_size=32)
accuracy = model.evaluate(X,y)

print("Loss: " + str(accuracy[0]))
print("Accuracy: " + str(accuracy[1]*100) + "%")