# Libraries for math and graphing
import numpy as np
import tensorflow as tensorflow
import itertools
import matplotlib.pyplot as plt
import csv

from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix

# Inputs
age = []
sex = []
chest_pain_type = []
resting_blood_pressure = []
serium_cholestoral = []
fasting_blood_sugar = []
resting_ECG = []
maximum_heart_rate = []
exercise_induced_angina = []
oldpeak = []
peak_exercise_slope = []
major_vessels = []
thal = []

# Labels
labels = []

with open('heart.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for line in csv_reader:
        {
            age.append(line[0]),
            sex.append(line[1]),
            chest_pain_type.append(line[2]),
            resting_blood_pressure.append(line[3]),
            serium_cholestoral.append(line[4]),
            fasting_blood_sugar.append(line[5]),
            resting_ECG.append(line[6]),
            maximum_heart_rate.append(line[7]),
            exercise_induced_angina.append(line[8]),
            oldpeak.append(line[9]),
            peak_exercise_slope.append(line[10]),
            major_vessels.append(line[11]),
            thal.append(line[12]),
            labels.append(line[13])
        }

age = np.array(age)
sex = np.array(sex)
chest_pain_type = np.array(chest_pain_type)
resting_blood_pressure = np.array(resting_blood_pressure)
serium_cholestoral = np.array(serium_cholestoral)
fasting_blood_sugar = np.array(fasting_blood_sugar)
resting_ECG = np.array(resting_ECG)
maximum_heart_rate = np.array(maximum_heart_rate)
exercise_induced_angina = np.array(exercise_induced_angina)
oldpeak = np.array(oldpeak)
peak_exercise_slope = np.array(peak_exercise_slope)
major_vessels = np.array(major_vessels)
thal = np.array(thal)



labels = np.array(labels)

scaler = MinMaxScaler(feature_range=(0, 1))

age = scaler.fit_transform(age.reshape(-1, 1))
sex = scaler.fit_transform(sex.reshape(-1, 1))
chest_pain_type = scaler.fit_transform(chest_pain_type.reshape(-1, 1))
resting_blood_pressure = scaler.fit_transform(resting_blood_pressure.reshape(-1, 1))
serium_cholestoral = scaler.fit_transform(serium_cholestoral.reshape(-1, 1))
fasting_blood_sugar = scaler.fit_transform(fasting_blood_sugar.reshape(-1, 1))
resting_ECG = scaler.fit_transform(resting_ECG.reshape(-1, 1))
maximum_heart_rate = scaler.fit_transform(maximum_heart_rate.reshape(-1, 1))
exercise_induced_angina = scaler.fit_transform(exercise_induced_angina.reshape(-1, 1))
oldpeak = scaler.fit_transform(oldpeak.reshape(-1, 1))
peak_exercise_slope = scaler.fit_transform(peak_exercise_slope.reshape(-1, 1))
major_vessels = scaler.fit_transform(major_vessels.reshape(-1, 1))
thal = scaler.fit_transform(thal.reshape(-1, 1))

labels = scaler.fit_transform(labels.reshape(-1, 1))

age_train, age_test, sex_train, sex_test, chest_pain_type_train, chest_pain_type_test, resting_blood_pressure_train, resting_blood_pressure_test, serium_cholestoral_train, serium_cholestoral_test, fasting_blood_sugar_train, fasting_blood_sugar_test, resting_ECG_train, resting_ECG_test, maximum_heart_rate_train, maximum_heart_rate_test, exercise_induced_angina_train, exercise_induced_angina_test, oldpeak_train, oldpeak_test, peak_exercise_slope_train, peak_exercise_slope_test, major_vessels_train, major_vessels_test, thal_train, thal_test, labels_train, labels_test = train_test_split(
    age, 
    sex, 
    chest_pain_type,
    resting_blood_pressure,
    serium_cholestoral,
    fasting_blood_sugar,
    resting_ECG,
    maximum_heart_rate,
    exercise_induced_angina,
    oldpeak,
    peak_exercise_slope,
    major_vessels,
    thal,
    labels,
    test_size = 0.2, random_state = 1)

input1 = Input(shape = (1,))
input2 = Input(shape = (1,))
inputs = Concatenate()([input1, input2])

dense1 = keras.layers.Dense(512, activation = 'relu')
dense2 = keras.layers.Dense(256, activation = 'relu')
dense3 = keras.layers.Dense(2, activation = 'softmax')

x = dense1(inputs)
x = dense2(x)
outputs = dense3(x)
model = keras.Model(inputs = inputs, outputs = outputs)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer = keras.optimizers.Adam(lr = 0.001),
    metrics = ["accuracy"],
)


history = model.fit([age_train, sex_train], labels_train, batch_size= 32, epochs = 30, verbose = 2, validation_split = 0.2)
scores = model.evaluate([age_test, sex_test], labels_test, verbose = 2)
print("Test Loss:", scores[0])
print("Test Accuracy:", scores[1])
print(model.summary())

# model = keras.Model(inputs = inputs, outputs = outputs, name = 'functional-model')

# age_train = keras.Input(shape = (820, ))
# sex_train = keras.Input(shape = (820, ))

# input = Concatenate()([age_train, sex_train])
# x = Dense(2)(input)
# x = Dense(1)(x)

# model = Model(inputs = [age_train, sex_train], outputs = x)
# model.summary()

# model.compile(
#     optimizer = Adam(learning_rate=0.0001), 
#     loss = 'sparse_categorical_crossentropy', 
#     metrics = ['accuracy']
#     )

# model.fit(
#     x = [age_test, sex_test], 
#     y = labels_test, 
#     validation_split = 0.1)




# inputs = keras.Input(shape = (1))
# x = keras.layers.Dense(512, activation = 'relu')(inputs)
# x = keras.layers.Dense(256, activation = 'relu')(x)
# outputs = keras.layers.Dense(2, activation = 'softmax')(x)
# model = keras.Model(inputs = inputs, outputs = outputs)

# model.compile(
#     optimizer = Adam(learning_rate=0.0001), 
#     loss = 'sparse_categorical_crossentropy', 
#     metrics = ['accuracy']
#     )


# model.fit(x = age_train, y = labels_train, batch_size=32, epochs = 50, verbose = 2)
# model.evaluate(age_test, labels_test, batch_size = 32, verbose = 2)
# print(model.summary())


# flatten = keras.layers.Flatten()
# dense1 = keras.layers.Dense(128, activation = 'relu')
# dense2 = keras.layers.Dense(256)
# dense3 = keras.layers.Dense(2)

# x = flatten(inputs)
# x = dense1(x)
# x = dense2(x)
# x = dense3(x)
# outputs = dense3(x)


# merged = keras.layers.Concatenate(axis = 1)([age_train, sex_train])
# dense1 = keras.layers.Dense(2, input_dim = 2, activation = keras.activations.sigmoid, use_bias = True)(merged)
# output = keras.layers.Dense(1, activation =keras.activations.relu, use_bias = True)(dense1)
# model = keras.models.Model(inputs = [age_train, sex_train], output = output)

# model.fit([age_test, sex_test], output, batch_size = 16, epochs = 30)





# # matplotlib inline


# cm = confusion_matrix(y_true = test_labels, y_pred = rounded_predictions)

# def plot_confusion_matrix(
#     cm, 
#     classes, 
#     normalize = False, 
#     title = 'Confusion matrix', 
#     cmap = plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting 'normalize=True'.
#     """

#     plt.imshow(cm, interpolation='nearest', cmap = cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation = 45)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis]
#     else:
#         print("Confusion matrix, without normalization")

#     print(cm)

#     thresh = cm.max()/2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#             horizontalalignment = "center",
#             color = "white" if cm[i, j] > thresh else "black")

# plt.tight_layout()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')

# cm_plot_labels = ['no_side_effects', 'had_side_effects']
# plot_confusion_matrix(cm = cm, classes = cm_plot_labels, title = "Confusion Matrix")

# import os.path
# model.save('medical_trial_model.h5')