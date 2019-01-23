# IMPORTS
import tensorflow as tf
from tensorflow import keras
from xml.etree import ElementTree
import glob
import numpy as np
#from nltk.corpus import stopwords
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
import matplotlib.pyplot as plt
from PIL import Image

# FUNCTIONS
        

# DATA READING
# Reading the report files and recording the descriptions provided as well as their associated images and diagnosis
i = 0
descriptions = []
images = []
diagnosis = []
for file in glob.glob('NLMCXR_reports/ecgen-radiology/*.xml'):
    tree = ElementTree.parse(file)
    root = tree.getroot()
    # Iterating over all AbstarctText tags
    for description in root.iter('AbstractText'):
        # if the Label is findings, record text as the description
        if(description.get('Label') == 'FINDINGS'):
            if(description.text != None):
                descriptions.append(description.text)
            else:
                descriptions.append('')
        # if the label is impression, record text as the diagnosis
        if(description.get('Label') == 'IMPRESSION'):
            if(description.text != None):
                diagnosis.append(description.text)
            else:
                diagnosis.append('')
    # Now we must read the images that are relevant to each report
    # Reminder: there may be more than one image associated to each report!
    temp = []
    for image in root.iter('parentImage'):
        temp.append(image.get('id'))
    images.append(temp)
    i += 1

# import all of the images that are associated to the reports and assign them to the images array
# be careful to keep the images in the correct order as some reports have multiply images and some have none
for i in range(len(images)):
    reports = []
    for image in images[i]:
        image = Image.open('NLMCXR_png/' + image + '.png')
        reports.append(image)
    images[i] = reports

# tokenize the description so that the labels can be made by detecting certain words or phrases in the line
labels = [0]*len(descriptions)
i = 0
for line in diagnosis:
    line = line.lower()
    # some of the diagnosis' are clearly identified as having no active diseases
    if 'no acute' in line or 'no active' in line or 'normal' in line or 'no radiographic evidence' in line or 'no evidence' in line:
        # if there are no active diseases, assign a label value of 0, meaning normal
        labels[i] = 0
    else:
        # else, if there are indications of an active disease, assign label value of 1, to indicate a diseased sample
        labels[i] = 1
    i += 1

#for i in range(50):
#    print(descriptions[i])
#    print(diagnosis[i])
#    print(labels[i])
#    print()

# converting the sequences to one hot vectors to represent words as integers, with a max vocabulary size of 1000
i = 0
for line in descriptions:
    descriptions[i] = one_hot(line, 1000, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower = True, split = ' ')
    i += 1

# regulating the size of all descriptions
descriptions = pad_sequences(descriptions, maxlen = 100, dtype = 'int32')

# allocating some of the descriptions and label data to be training, validation, and testing data
trainX = descriptions[:2500]
trainY = labels[:2500]
valX = descriptions[2500:3250]
valY = labels[2500:3250]
testX = descriptions[3250:]
testY = labels[3250:]

# regulating the dimensions of all of the data to ensure the model will be able to properly work with it
trainX = np.asarray(trainX)
trainY = np.asarray(trainY)
valX = np.asarray(valX)
valY = np.asarray(valY)
testX = np.asarray(testX)
testY = np.asarray(testY)

# constructing the convolutional neural network model
model = Sequential()
model.add(Embedding(1000, 128, input_length=100))
model.add(Conv1D(32, 7, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(32, 7, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation = 'sigmoid'))

# compiling the model with adam optimizer, using binary crossentropy and recording the accuracy
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# saving the fit data as a dictionary to be used to create graphs to depict the results
history = model.fit(trainX, trainY, epochs = 10, batch_size = 64, validation_data = (valX, valY))

# evaluating the neural network using the allocated test data and printing the results
results = model.evaluate(testX,  testY)
print(results)

# taking the values form the fitting dictionary to create graphs to represent the change in validation and loss over time
acc = history.history['acc']
valAcc = history.history['val_acc']
loss = history.history['loss']
valLoss = history.history['val_loss']
        
epochs = range(1, len(acc) + 1)

# creating a graph to represent the Loss that occured during the fitting process
plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, valLoss, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# creating a graph to represent the Accuracy that was recorded during the fitting process
plt.clf()
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, valAcc, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
