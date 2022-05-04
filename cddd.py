#!/usr/bin/env python
# coding: utf-8

# Imports


from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
from tensorflow import keras
import random


# Load Data

df = pd.read_csv('AAAA.csv')
df = df.drop_duplicates()
df.columns = ['smiles', 'label']
smiles = df['smiles']

cansmiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smile)) for smile in smiles]
cansmiles = pd.DataFrame(cansmiles)
cansmiles.columns = ['smiles']


# Extract Molecular properties

properties = pd.DataFrame()
properties['label'] = df['label']
properties['smiles'] = cansmiles['smiles']
# add properties
properties['logP'] = [Descriptors.MolLogP(
    Chem.MolFromSmiles(smile)) for smile in properties['smiles']]
properties['maxPC'] = [Descriptors.MaxPartialCharge(
    Chem.MolFromSmiles(smile)) for smile in properties['smiles']]
properties['minPC'] = [Descriptors.MinPartialCharge(
    Chem.MolFromSmiles(smile)) for smile in properties['smiles']]
properties['valElectrons'] = [Descriptors.NumValenceElectrons(
    Chem.MolFromSmiles(smile)) for smile in properties['smiles']]
properties['HbondDonor'] = [Descriptors.NumHDonors(
    Chem.MolFromSmiles(smile)) for smile in properties['smiles']]
properties['HbondAcceptor'] = [Descriptors.NumHAcceptors(
    Chem.MolFromSmiles(smile)) for smile in properties['smiles']]
properties['BalaJ'] = [Descriptors.BalabanJ(
    Chem.MolFromSmiles(smile)) for smile in properties['smiles']]
# properties['molarRefractivity'] = [Descriptors.MolarRefractivity(Chem.MolFromSmiles(smile)) for smile in properties['smiles']]
properties['TPSA'] = [Descriptors.TPSA(
    Chem.MolFromSmiles(smile)) for smile in properties['smiles']]

# replace all nan's to average
properties['logP'].fillna(properties['logP'].mean(), inplace=True)
properties['maxPC'].fillna(properties['maxPC'].mean(), inplace=True)
properties['minPC'].fillna(properties['minPC'].mean(), inplace=True)
properties['valElectrons'].fillna(
    properties['valElectrons'].mean(), inplace=True)
properties['HbondDonor'].fillna(properties['HbondDonor'].mean(), inplace=True)
properties['HbondAcceptor'].fillna(
    properties['HbondAcceptor'].mean(), inplace=True)
properties['BalaJ'].fillna(properties['BalaJ'].mean(), inplace=True)
properties['TPSA'].fillna(properties['TPSA'].mean(), inplace=True)

properties.head()

molprops = np.array(properties[['logP', 'maxPC', 'minPC',
                    'valElectrons', 'HbondDonor', 'HbondAcceptor', 'BalaJ', 'TPSA']])


# Preprocessing


# get set of all characters
chars = []
for cansmile in cansmiles['smiles']:
    for char in cansmile:
        if char not in chars:
            chars.append(char)
chars = sorted(chars)
print("Total characters:", len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# create one-hot encodings
maxlen = 0
for cansmile in cansmiles['smiles']:
    if len(cansmile) > maxlen:
        maxlen = len(cansmile)
print("Max length:", maxlen)
x = np.zeros((len(cansmiles), maxlen, len(chars)), dtype=np.bool)
for i, cansmile in enumerate(cansmiles['smiles']):
    for j, char in enumerate(cansmile):
        x[i, j, char_indices[char]] = 1


# split data to train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, molprops, test_size=0.1, random_state=42)


### Autoencoder (experiments)


encoder = keras.Sequential(
    [
        keras.Input(shape=(maxlen, len(chars))),
        keras.layers.LSTM(units=128, return_sequences=False),
        keras.layers.Dense(units=len(chars), activation='relu')
    ]
)
decoder = keras.Sequential(
    [
        keras.Input(shape=(len(chars),)),
        keras.layers.RepeatVector(maxlen),
        keras.layers.LSTM(units=128, return_sequences=True),
        keras.layers.Dense(units=len(chars), activation='softmax')

    ]
)

autoencoder = keras.Model(inputs=encoder.input,
                          outputs=decoder(encoder.output))
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')


# train the autoencoder
autoencoder.fit(x_train, x_train, epochs=10, batch_size=32)


# predict the test data
x_test_pred = autoencoder.predict(x_test)

# normalize, and convert to boolean
x_test_pred = x_test_pred / np.max(x_test_pred)
x_test_pred = x_test_pred > 0.5

# test accuracy of the autoencoder
score = 0.0
for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        for k in range(len(x_test[i][j])):
            if x_test[i][j][k] == x_test_pred[i][j][k]:
                score += 1
score /= len(x_test) * maxlen * len(chars)
print("Test accuracy:", score)

# check for false positives
false_pos = 0
for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        for k in range(len(x_test[i][j])):
            if x_test[i][j][k] != x_test_pred[i][j][k] and x_test[i][j][k] == False:
                false_pos += 1
print("False positives:", false_pos / (len(x_test) * maxlen * len(chars)))

# check for false negatives
false_neg = 0
for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        for k in range(len(x_test[i][j])):
            if x_test[i][j][k] != x_test_pred[i][j][k] and x_test[i][j][k] == True:
                false_neg += 1
print("False negatives:", false_neg / (len(x_test) * maxlen * len(chars)))


# Autoencoder + Effective Translation (experiments)


# autoencoder with multiple losses
encoderl = keras.Sequential(
    [
        keras.Input(shape=(maxlen, len(chars))),
        keras.layers.LSTM(units=128, return_sequences=False),
        keras.layers.Dense(units=len(chars), activation='relu')
    ]
)

fnn = keras.Sequential(
    [
        keras.Input(shape=(len(chars),)),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=16, activation='relu'),
        keras.layers.Dense(units=8, activation='relu')
    ]
)

decoderl = keras.Sequential(
    [
        keras.Input(shape=(len(chars),)),
        keras.layers.RepeatVector(maxlen),
        keras.layers.LSTM(units=128, return_sequences=True),
        keras.layers.Dense(units=len(chars), activation='softmax')

    ]
)

autoencoderl = keras.Model(inputs=encoderl.input, outputs=[
                           decoderl(encoderl.output), fnn(encoderl.output)])
autoencoderl.compile(optimizer='adam', loss=[
                     'categorical_crossentropy', 'mse'])


# train the autoencoder
autoencoderl.fit(x_train, [x_train, y_train], epochs=100, batch_size=32)


# predict the test data
x_test_predl = autoencoderl.predict(x_test)[0]

# normalize, and convert to boolean
x_test_predl = x_test_predl / np.max(x_test_predl)
x_test_predl = x_test_predl > 0.5

# test accuracy of the autoencoder
score = 0.0
for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        for k in range(len(x_test[i][j])):
            if x_test[i][j][k] == x_test_predl[i][j][k]:
                score += 1
score /= len(x_test) * maxlen * len(chars)
print("Test accuracy:", score)

# check for false positives
false_pos = 0
for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        for k in range(len(x_test[i][j])):
            if x_test[i][j][k] != x_test_predl[i][j][k] and x_test[i][j][k] == False:
                false_pos += 1
print("False positives:", false_pos / (len(x_test) * maxlen * len(chars)))

# check for false negatives
false_neg = 0
for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        for k in range(len(x_test[i][j])):
            if x_test[i][j][k] != x_test_predl[i][j][k] and x_test[i][j][k] == True:
                false_neg += 1
print("False negatives:", false_neg / (len(x_test) * maxlen * len(chars)))


molpropspred = autoencoderl.predict(x_test)[1]

# test error
error = 0.0
for i in range(len(y_test)):
    for j in range(len(y_test[i])):
        # check if error is nan
        if np.isnan(error):
            print("Error is nan")
            print("stopped at:", i, j)
        # mean squared error
        error += np.sum(np.square(y_test[i][j] - molpropspred[i][j]))
error /= len(y_test) * len(y_test[0])
print("Test error:", error)
