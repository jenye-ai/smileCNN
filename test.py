import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load train data
train_data = pd.read_csv('randomized_data.txt')
print(train_data)

y_label_numpy = train_data.iloc[:,0].values - 1
x_data_numpy = train_data.iloc[:,1:].values
# print(x_data_numpy)

# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data_numpy, y_label_numpy, test_size=0.20)


# for testing different nodes, use later on
num_hidden_nodes = [3, 5, 10]


# Build the model
# Build the model: 3 hidden layers
model = Sequential()
model.add(Dense(10, input_shape=(13,), activation='linear', name='input'))
model.add(Dense(3, activation='softmax', name='output'))
print(model.summary())

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, verbose=1, batch_size=64, epochs=50)

# Test on unseen data
results = model.evaluate(X_test, y_test)
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))