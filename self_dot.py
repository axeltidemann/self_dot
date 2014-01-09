# Attempt at self-generating code: self_dot
# 
# Author: Axel Tidemann and Oeyvind Brandtsegg, (axel.tidemann@gmail.com, obrandts@gmail.com)

from pyevolve import G1DList, GSimpleGA, Mutators

i = 0
n = 20
#language = [ 'for ', 'i ', 'in ', 'range ', 'n ', 'print ', ' ', '(', ')', ':']
language = [ 'range', '(', 'n', ')']

def combinations(L):
    combos = []
    for i in range(len(L)):
        for j in range(len(L)):
	        if j+i < len(L):
		        combos.append(L[j:j+i+1])
    return combos

def chromosome_to_expression(chromosome):
    return ''.join([ language[gene] for gene in chromosome ])

def fitness(chromosome):
    #print 'e:', chromosome_to_expression(chromosome)
    score = 0
    for c in combinations(chromosome):
        try:
            temp = eval(chromosome_to_expression(c))
            length = len(c)#len(chromosome_to_expression(c))
            #print 'length', length, chromosome_to_expression(c)
            score += length
        except:
            pass 
    return score

genome = G1DList.G1DList(4)
genome.evaluator.set(fitness)
genome.mutator.set(Mutators.G1DListMutatorIntegerRange)
genome.setParams(rangemin=0, rangemax=len(language)-1)
ga = GSimpleGA.GSimpleGA(genome)
ga.setMutationRate(0.5)
populationSize = 10
ga.setPopulationSize(populationSize)
ga.setGenerations(10)
ga.evolve(freq_stats=1)
pop = ga.getPopulation()
pop.sort()
index = 0
good = 0
while index < populationSize:
    try:
        print 'genome:', pop[index].genomeList
        print 'trying ...'
        print chromosome_to_expression(pop[index].genomeList)
        print 'Evaluates to', eval(chromosome_to_expression(pop[index].genomeList))
        index += 1
        good = index
    except:
        print '\n Sorry, no more valid expressions'
        index = populationSize+1
print 'done, producing %i good statements' %good
