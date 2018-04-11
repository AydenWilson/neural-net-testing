import sys
import math

class perceptron(object):
    def __init__(self):
        self.learningRate = 1
        self.output = 0
        self.inputs = []
        self.weights = [0] * 101
        self.polarity = -99 # in training, this is the sentinement of the tweet
        self.bias = -0.1
        self.classErrors = 0
        self.weightNames = []

    def calculateOutput(self):
        inputSum = 0;
        # save the polarity and remove it from the inputs
        self.polarity = self.inputs[len(self.inputs) - 1]
        del self.inputs[len(self.inputs) - 1]

        # add a -1 to the start of the inputs
        self.inputs.insert(0, self.bias)

        # multiply the inputs by their weights and then sum them
        for i in range(len(self.inputs)):
            inputSum += self.inputs[i] * self.weights[i]

        # sigmoid function to squach answer between 0 and 1
        self.output = 1/(1+(math.pow(math.e, -inputSum)))
        if self.polarity == 1:
            if self.output < 0.5:
                self.classErrors += 1
        else:
            if self.output > 0.5:
                self.classErrors += 1

    def changeWeights(self):
        g = self.output * (1 - self.output)
        error = self.polarity - self.output
        #change the weights
        for i in range(len(self.weights) - 1):
            self.weights[i] = self.weights[i] + float(self.learningRate) * error * g * float(self.inputs[i])

    def printWeights(self):
        for x in range(len(self.weightNames)):
            print(self.weightNames[x] + " = " + str(self.weights[x]))

    def printState(self):
        print("-------------------------")
        print("learing rate: ", self.learningRate)
        print("polarity: ", self.polarity)       
        print("output: ", self.output)
        print("classification errors: ", self.classErrors)
        print("lenght of inputs: ", len(self.inputs))
        print("length of weights: ", len(self.weights))
        print("-------------------------")


f = open(sys.argv[1])
words = f.readline()
words = words.split(",")

# remove the \n off the end of the last word
words[len(words)-1] = words[len(words) - 1][:-1]

percep = perceptron()
percep.weightNames = words
percep.learningRate = sys.argv[2]


for line in f:
    inputs = list(map(int, line.split(",")))
    percep.inputs = inputs
    percep.calculateOutput()
    percep.changeWeights()

#print(",".join(map(str, percep.weights)))
percep.printWeights()
percep.printState()
