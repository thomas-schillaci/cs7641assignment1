from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plot
import utils

data = x_train, x_test, y_train, y_test = utils.import_wine(y_transform='to_categorical')

model = Sequential()

model.add(Dense(10, input_dim=len(x_train.keys()), activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(len(y_train[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=50)

utils.plot_ann_history(history)

plot.show()

y_predict = model.predict(x_test[:5])
for i in range(5):
    plot.bar([80, 84, 88, 92, 96, 100], [*y_test[i], 0], width=4, align='edge')
    plot.bar([80, 84, 88, 92, 96, 100], [*y_predict[i], 0], width=4, align='edge')
    plot.title(f'Example {i + 1}')
    plot.xticks([80, 84, 88, 92, 96, 100])
    plot.xlabel('Points')
    plot.ylabel('Degree of certainty')
    plot.legend(['Real', 'Predicted'], loc='upper left')
    plot.show()
