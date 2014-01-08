# First version of self-generating code: self_dot
# 
# Author: Axel Tidemann, axel.tidemann@gmail.com

from pyevolve import G1DList, GSimpleGA, Mutators

language = [ '1.0', '3.0', '+', '*', '4.0', '.5', '/', ' ' ]

def chromosome_to_expression(chromosome):
    return ''.join([ language[gene] for gene in chromosome ])

def fitness(chromosome):
   try:
       return eval(chromosome_to_expression(chromosome))
   except:
       return 0

genome = G1DList.G1DList(6)
genome.evaluator.set(fitness)
genome.mutator.set(Mutators.G1DListMutatorIntegerRange)
genome.setParams(rangemin=0, rangemax=len(language))
ga = GSimpleGA.GSimpleGA(genome)
ga.setMutationRate(.5)
ga.setPopulationSize(100)
ga.setGenerations(100)
ga.evolve(freq_stats=10)
print 'Best individual was the expression', chromosome_to_expression(ga.bestIndividual().genomeList), 'which equals', eval(chromosome_to_expression(ga.bestIndividual().genomeList))


