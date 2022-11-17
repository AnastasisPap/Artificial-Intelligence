import numpy as np
import random as rand
from determine_weights import *

def runGeneration(numOfGenerations, numOfWeights, mutationProbability, populationSize):
    currentPopulation = generateStartingWeights(populationSize, numOfWeights)
    bestWeight = None
    winProbability = 0

    for iteration in range(numOfGenerations):
        print(f'============================ GENERATION #{iteration + 1} ===========================')
        newPopulation = []
        selectionProbability = evaluate(currentPopulation)
        for i in range(populationSize // 2):
            firstIdx, secondIdx = np.random.choice(populationSize, size=2, p=selectionProbability)
            while firstIdx == secondIdx:
                secondIdx = np.random.choice(populationSize, size=1, p=selectionProbability)
            
            firstParent = currentPopulation[firstIdx]
            secondParent = currentPopulation[secondIdx]

            firstChild, secondChild = reproduce(firstParent, secondParent)
            firstChild, secondChild = mutate(firstChild, mutationProbability), mutate(secondChild, mutationProbability)

            newPopulation.append(firstChild)
            newPopulation.append(secondChild)
            currentPopulation = newPopulation
    
        if max(selectionProbability) > winProbability:
            winProbability = max(selectionProbability)
            bestWeight = currentPopulation[selectionProbability.index(winProbability)]

    print(f'Best weight from {numOfGenerations} generations: {bestWeight}')
    return bestWeight


def evaluate(evaluationChromosomes):
    populationSize = len(evaluationChromosomes)
    results = [0] * populationSize 
    for i in range(populationSize):
        for j in range(populationSize):
            if i != j:
                _, winner = battle(evaluationChromosomes[i], evaluationChromosomes[j])
                if winner == 0:
                    results[j] += 1
                else:
                    results[i] += 1
    
    totalGames = (populationSize - 1) * populationSize
    return [i / totalGames for i in results]


def reproduce(x, y):
    splitIdx = np.random.randint(len(x))

    firstChild = x[:splitIdx] + y[splitIdx:]
    secondChild = y[:splitIdx] + x[splitIdx:]


    return firstChild, secondChild


def mutate(x, mutationProbability):
    idxes = []
    weightsSum = 0
    for i in range(len(x)):
        if  np.random.uniform() < mutationProbability:
            idxes.append(i)
        else:
            weightsSum += x[i]
    
    if len(idxes) == 0: return x
    mutatedWeights = generateTuple(len(idxes), 1 - weightsSum)

    newChromosome = list(x)
    for i, idx in enumerate(idxes):
        newChromosome[idx] = mutatedWeights[i]
    
    return tuple(newChromosome)

runGeneration(4, 4, 0.01, 10)