from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Dense, Dropout, BatchNormalization

import utils

data = x_train, x_test, y_train, y_test = utils.import_adult(use_to_categorical=True, normalize=True)

model = Sequential()

model.add(Dense(15, input_dim=len(x_train.keys()), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(15, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(len(y_train[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

filepath="best-model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=200,
    batch_size=50,
    callbacks=callbacks_list
)

model = load_model(filepath)

print(f'Model accuracy: {model.evaluate(x_test, y_test)[1]*100}%')

utils.plot_ann_history(history)
