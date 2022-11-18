import numpy as np
import random as rand
from determine_weights import *
import os

def runGeneration(numOfGenerations, numOfWeights, mutationProbability, populationSize, maxDepth):

    # Generate starting population, each chromosome represents a tuple of values (weights) with total sum = 1 .
    currentPopulation = generateStartingWeights(populationSize, numOfWeights)
    logging = Log(numOfGenerations, mutationProbability, populationSize, maxDepth)
    
    bestWeight = None
    winProbability = 0

    for iteration in range(numOfGenerations):
        print(f'============================ GENERATION #{iteration + 1} ===========================')
        newPopulation = []
        selectionProbability = evaluate(currentPopulation, maxDepth)
        for i in range(populationSize // 2):
            firstIdx, secondIdx = np.random.choice(populationSize, size=2, p=selectionProbability)

            # Make sure that chromosomes that reproduce are different.
            while firstIdx == secondIdx:
                secondIdx = np.random.choice(populationSize, size=1, p=selectionProbability)[0]
            
            firstParent = currentPopulation[firstIdx]
            secondParent = currentPopulation[secondIdx]

            firstChild, secondChild = reproduce(firstParent, secondParent)
            firstChild, secondChild = mutate(firstChild, mutationProbability), mutate(secondChild, mutationProbability)

            newPopulation.append(firstChild)
            newPopulation.append(secondChild)
    
        # This doesn't keep the max correctly
        if max(selectionProbability) > winProbability:
            winProbability = max(selectionProbability)
            bestWeight = currentPopulation[selectionProbability.index(winProbability)]

        logging.log(bestWeight, winProbability)
        currentPopulation = newPopulation

    return bestWeight


# Chromosomes represent weights for bot-players.
# Each player battles with every other player, the number of wins of every player is counted
# and probability of selection for reproduction is calculated for each chromosome.
# The more wins the higher the chance to be selected for reproduction.
def evaluate(evaluationChromosomes, maxDepth):
    populationSize = len(evaluationChromosomes)
    results = [0] * populationSize 
    totalGames = (populationSize - 1) * populationSize
    print(f'Battle progress: 0/{totalGames}', end="\r")
    cnt = 0
    for i in range(populationSize):
        for j in range(populationSize):
            if i != j:
                _, winner = battle(evaluationChromosomes[i], evaluationChromosomes[j], maxDepth)
                if winner == 0:
                    results[j] += 1
                elif winner == 1:
                    results[i] += 1
                else:
                    totalGames -= 1
                cnt += 1
                print(f'Battle progress: {cnt}/{populationSize * (populationSize - 1)}', end="\r")
    print()
    return [i / totalGames for i in results]


def reproduce(x, y):
    splitIdx = np.random.randint(len(x))

    firstChild = x[:splitIdx] + y[splitIdx:]
    secondChild = y[:splitIdx] + x[splitIdx:]


    return firstChild, secondChild


def mutate(x, mutationProbability):
    idxes = []
    weightsSum = 0 # total weight that will go unchanged.

    for i in range(len(x)):
        if  np.random.uniform() < mutationProbability:
            idxes.append(i)
        else:
            weightsSum += x[i]
    
    if len(idxes) == 0: return x

    # distribute remaining weight randomly for the selected weights to be altered.
    mutatedWeights = generateTuple(len(idxes), 1 - weightsSum)

    newChromosome = list(x)
    for i, idx in enumerate(idxes):
        newChromosome[idx] = mutatedWeights[i]
    
    return tuple(newChromosome)

class Log:
    def __init__(self, numOfGenerations, mutationProbability, populationSize, maxDepth):
        directoryName = 'genetic_weights'
        if not os.path.exists(directoryName): os.mkdir(directoryName)
        self.fileName = f'{numOfGenerations}_{str(mutationProbability).replace(".", ",")}_{populationSize}_{maxDepth}.txt'
        self.file = open(directoryName + '/' + self.fileName, 'w')
        self.currGeneration = 1

    def log(self, bestWeight, winProbability):
        self.file.write(f'{"="*15} Generation #{self.currGeneration} {"="*15}\nWin Probability:\n{winProbability}\nbestWeights:\n{bestWeight}\n\n')
        self.currGeneration += 1
    
    def closeFile(self):
        self.file.close()

numOfGenerations = int(input("Num of generations: "))
numOfWeights = 4
mutationProbability = 0.01
populationSize = int(input("Population size: "))
while populationSize % 2 == 1:
    populationSize = int(input("Populatin size (must be even): "))
maxDepth = int(input("Max depth: "))
runGeneration(numOfGenerations, numOfWeights, mutationProbability, populationSize, maxDepth)