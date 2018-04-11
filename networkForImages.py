import sys
import math
import struct
import numpy
import random

def sigmoid(x):
    return 1/(1+numpy.exp(-x))

class perceptron(object):
    def __init__(self):
        self.learningRate = 0.1
        self.output = 0
        self.inputs = numpy.zeros(1)
        self.weights = numpy.zeros(1)
        self.polarity = 0
        self.error = 0
        self.weightChange = 0

    def setNumberOfWeights(self, i):
        self.weights = numpy.zeros(i)
        for x in range(len(self.weights)):
            self.weights[x] = (random.random() * 0.01)

    def printWeights(self):
        for x in range(len(self.weightNames)):
            print(self.weightNames[x] + " = " + str(self.weights[x]))

    def saveWeightsAsImage(self, imageName):
        image = open(imageName, "w")
        image.write("P2")
        image.write("\n")
        imageinfo = "28" + " " + "28"
        image.write(imageinfo)
        image.write("\n")
        image.write("255")
        image.write("\n")
        for x in range(28):
            for y in range (28):
                image.write(str(int(self.weights[x+y*28]*2550)))
                image.write(" ")
            image.write("\n")

    def printState(self):
        print("-------------------------")
        print("learing rate: ", self.learningRate)
        print("polarity: ", self.polarity)       
        print("output: ", self.output)
        print("classification errors: ", self.classErrors)
        print("lenght of inputs: ", len(self.inputs))
        print("length of weights: ", len(self.weights))
        print("-------------------------")


class network(object):
    def __init__(self):
        self.layers = []
        self.layerOutputs = []
        self.numberOfLayers = 0
        self.networkOutput = []
        self.bias = -1

    def printState(self):
        #print("Layers: " + str(self.layers))
        print("===========================================================================")
        print("percep outputs: ")
        for layer in self.layers:
            print("[", end="")
            for percep in layer:
                print(percep.output, end="")
                print(" ", end="")
            print("]")
        print("Weight Changes: ")
        for layer in self.layers:
            print("[", end="")
            for percep in layer:
                print(percep.weightChange, end="")
                print(" ", end="")
            print("]")

        print("Number of layers: ", self.numberOfLayers)
        print("===========================================================================")

    def printWeightsAsImage(self, layer):
        for percep in layer:
            for weight in percep.weights:
                print(int(weight*16), end="")
            print("")

    def saveWeightsAsImage(self, layer, imageName):
        image = open(imageName, "w")
        image.write("P2")
        image.write("\n")
        imageinfo = str(len(layer)) + " " + str(len(layer[0].weights)) 
        image.write(imageinfo)
        image.write("\n")
        image.write("255")
        image.write("\n")
        for percep in layer:
            for weight in percep.weights:
                image.write(str(int(weight*2550)))
                image.write(" ")
            image.write("\n")

    # build a layer of perceptrons
    def buildLayer(self, numberOfPerceps, numInputs=0):
        layer = []
        for x in range(numberOfPerceps):
            percep = perceptron()
            if(numInputs == 0):
                percep.setNumberOfWeights(len(self.layerOutputs[len(self.layerOutputs)-1]))
            else:
                percep.setNumberOfWeights(numInputs+1)
            layer.append(percep)
        self.layers.append(layer)
        layerOutput = numpy.zeros(numberOfPerceps + 1)
        layerOutput[len(layerOutput)-1] = self.bias # set the bias
        self.layerOutputs.append(numpy.zeros(numberOfPerceps + 1)) # +1 for bias

        self.numberOfLayers += 1

    def calculateNetworkOutput(self, inputs):
        inputs2 = numpy.zeros(len(inputs) + 1)
        # add a the bias to the end of the inputs
        inputs2[len(inputs2) - 1] = self.bias

        for x in range(len(inputs)):
            inputs2[x] = inputs[x]/255

        # get the output of the first layer
        for x in range(len(self.layers[0])):
            percep = self.layers[0][x]
            percep.inputs = inputs2
            percep.output = sigmoid(numpy.dot(percep.inputs, percep.weights))
            self.layerOutputs[0][x] = percep.output

        # do all the other layers
        for x in range(len(self.layers) - 1):
            for i in range(len(self.layers[x+1])):
                percep = self.layers[x+1][i]
                percep.inputs = self.layerOutputs[x]
                percep.output = sigmoid(numpy.dot(percep.inputs, percep.weights))
                self.layerOutputs[x+1][i] = percep.output
        self.networkOutput = self.layerOutputs[len(self.layerOutputs)-1]

    def changeWeights(self, labels):
        labelsForLearning = [0] * 10
        labelsForLearning[labels[0]] = 1

        # change the weights of the output layer
        layers = self.layers
        for x in range(len(layers[len(layers) - 1])):
            # set the desired output in the percep
            percep = layers[len(layers)-1][x]
            percep.polarity = labelsForLearning[x]
            percep.error = percep.output * (1 - percep.output) * (percep.polarity - percep.output)
            percep.weightChange = percep.error * percep.learningRate 
            numpy.add(percep.weights, percep.inputs*percep.weightChange, out=percep.weights)
            
            #print("-------------------------------")
            #print("percep output: ", percep.output)
            #print("percep polarity: ", percep.polarity)
            #print("percep error: ", percep.error)
            #print("percep weightChange: ", percep.weightChange)
            #percep.printState()

        #change the weights of the other layers, propergating backwards
        for x in reversed(range(len(layers) - 1)): # go back through the layers starting at the second to last
            for i in range(len(layers[x])): # for every perceptron in the layer
                errorSum = 0
                for percep in layers[x+1]:
                    errorSum += percep.weights[i]*percep.error

                percep = layers[x][i]
                percep.error = percep.output * (1 - percep.output) * errorSum
                percep.weightChange = percep.error * percep.learningRate
                numpy.add(percep.weights, percep.inputs*percep.weightChange, out=percep.weights)


                #print("-------------------------------")
                #print("percep errorSum: ", errorSum)
                #print("percep output: ", layers[x][i].output)
                #print("percep error: ", layers[x][i].error)
                #print("percep weightChange: ", layers[x][i].weightChange)


images = open(sys.argv[1], "rb")
labels = open(sys.argv[2], "rb")
imagesTest = open(sys.argv[3], "rb")
labelsTest = open(sys.argv[4], "rb")

#----------------------------load training stuff----------------------------
# Load the images into an array
magicNumber = struct.unpack('I', images.read(4))
if(magicNumber[0] != 0x3080000):
    print("unexpected magic number")
numberOfImages = struct.unpack('>I', images.read(4))[0]
print("Number of images: " + str(numberOfImages))
iRow = struct.unpack('>I', images.read(4))[0]
iCol = struct.unpack('>I', images.read(4))[0]
print("Number of columns | rows: " + str(iCol) + " | " + str(iRow))
imageArray = numpy.fromstring(images.read(numberOfImages * iRow * iCol), dtype = numpy.uint8)

imageArray = imageArray.reshape((numberOfImages, iRow * iCol))

# Load the labels into an array
magicNumber2 = struct.unpack('I', labels.read(4))
if magicNumber2[0] != 0x1080000:
    print("unexpected magic number")
numberOfLabels = struct.unpack('>I', labels.read(4))[0]
print("Number of labels: " + str(numberOfLabels))
labelArray = numpy.fromstring(labels.read(numberOfLabels), dtype = numpy.uint8)
labelArray = labelArray.reshape((numberOfLabels, 1))
#----------------------------load testing stuff----------------------------
# Load the images into an array
magicNumber = struct.unpack('I', imagesTest.read(4))
if(magicNumber[0] != 0x3080000):
    print("unexpected magic number")
numberOfImages = struct.unpack('>I', imagesTest.read(4))[0]
print("Number of images: " + str(numberOfImages))
iRow = struct.unpack('>I', imagesTest.read(4))[0]
iCol = struct.unpack('>I', imagesTest.read(4))[0]
print("Number of columns | rows: " + str(iCol) + " | " + str(iRow))
imageArrayTest = numpy.fromstring(imagesTest.read(numberOfImages * iRow * iCol), dtype = numpy.uint8)

imageArrayTest = imageArrayTest.reshape((numberOfImages, iRow * iCol))

# Load the labels into an array
magicNumber2 = struct.unpack('I', labelsTest.read(4))
if magicNumber2[0] != 0x1080000:
    print("unexpected magic number")
numberOfLabels = struct.unpack('>I', labelsTest.read(4))[0]
print("Number of labels: " + str(numberOfLabels))
labelArrayTest = numpy.fromstring(labelsTest.read(numberOfLabels), dtype = numpy.uint8)
labelArrayTest = labelArrayTest.reshape((numberOfLabels, 1))

percepalNet = network()
percepalNet.buildLayer(40, 28*28)
percepalNet.buildLayer(20)
percepalNet.buildLayer(10)

right = 0
wrong = 0
prediction = 0
total = 0

# percepalNet.calculateNetworkOutput(imageArray[0])
# prediction = max(range(len(percepalNet.networkOutput)), key = lambda i: percepalNet.networkOutput[i])
# percepalNet.printState()

# print("Image Number | Prediction | Actual: ", 0+1, " | ", prediction, " | ", labelArray[0][0])
# percepalNet.changeWeights(labelArray[0])
# percepalNet.calculateNetworkOutput(imageArray[0])
# prediction = max(range(len(percepalNet.networkOutput)), key = lambda i: percepalNet.networkOutput[i])
# percepalNet.printState()
# print("Image Number | Prediction | Actual: ", 0+1, " | ", prediction, " | ", labelArray[0][0])

percepalNet.saveWeightsAsImage(percepalNet.layers[0], "layer1-before.pgm")
percepalNet.saveWeightsAsImage(percepalNet.layers[1], "layer2-before.pgm")
percepalNet.layers[0][0].saveWeightsAsImage("percep0-before.pgm")


for y in range(10):
    for x in range(0, 60000):
        percepalNet.calculateNetworkOutput(imageArray[x])
        #percepalNet.printState()
        percepalNet.changeWeights(labelArray[x])
        # if (x%999==0): 
        #     percepalNet.printState()
        #     print("Image Number | Prediction | Actual: ", x+1, " | ", prediction, " | ", labelArray[x][0])
        prediction = max(range(len(percepalNet.networkOutput)), key = lambda i: percepalNet.networkOutput[i])

        #print("Image Number | Prediction | Actual: ", x+1, " | ", prediction, " | ", labelArray[x][0])

        if (prediction == labelArray[x][0]): right += 1
        else: wrong += 1
        total += 1

    permutation = numpy.random.permutation(len(imageArray))
    imageArray = imageArray[permutation]
    labelArray = labelArray[permutation]
    print("epoch: ", y)


print("Result: ", right, " | ", wrong, " | ", total, " | ", right/total)

percepalNet.saveWeightsAsImage(percepalNet.layers[0], "layer1-after.pgm")
percepalNet.saveWeightsAsImage(percepalNet.layers[1], "layer2-after.pgm")
percepalNet.layers[0][0].saveWeightsAsImage("percep0-after.pgm")
percepalNet.layers[0][1].saveWeightsAsImage("percep0-after.pgm")
percepalNet.layers[0][2].saveWeightsAsImage("percep0-after.pgm")
percepalNet.layers[0][3].saveWeightsAsImage("percep0-after.pgm")
percepalNet.layers[0][4].saveWeightsAsImage("percep0-after.pgm")
percepalNet.layers[0][5].saveWeightsAsImage("percep0-after.pgm")
percepalNet.layers[0][6].saveWeightsAsImage("percep0-after.pgm")
percepalNet.layers[0][7].saveWeightsAsImage("percep0-after.pgm")
percepalNet.layers[0][8].saveWeightsAsImage("percep0-after.pgm")
percepalNet.layers[0][9].saveWeightsAsImage("percep0-after.pgm")

right = 0
wrong = 0
prediction = 0
total = 0

for x in range(0, 10000):
    percepalNet.calculateNetworkOutput(imageArrayTest[x])
    prediction = max(range(len(percepalNet.networkOutput)), key = lambda i: percepalNet.networkOutput[i])
    #print("Image Number | Prediction | Actual: ", x+1, " | ", prediction, " | ", labelArrayTest[x][0])
    if (prediction == labelArrayTest[x][0]): right += 1
    else: wrong += 1
    total += 1

print("Result: ", right, " | ", wrong, " | ", total, " | ", right/total)