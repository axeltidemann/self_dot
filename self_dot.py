# Attempt at self-generating code: self_dot
# 
# Authors: Axel Tidemann and Oeyvind Brandtsegg, (axel.tidemann@gmail.com, obrandts@gmail.com)

from pyevolve import G1DList, GSimpleGA, Mutators

language = [ 'for', 'i', 'in', 'range', '10', 'print', ' ', '(', ')', ':']

def chromosome_to_expression(chromosome):
    return ''.join([ language[gene] for gene in chromosome ])

def fitness(chromosome):
    try:
        eval(chromosome_to_expression(chromosome))
        return (len(chromosome) - chromosome_to_expression(chromosome).count(' ') )*100
    except:
        return chromosome_to_expression(chromosome).count(' ')

genome = G1DList.G1DList(20)
genome.evaluator.set(fitness)
genome.mutator.set(Mutators.G1DListMutatorIntegerRange)
genome.setParams(rangemin=0, rangemax=len(language)-1)
ga = GSimpleGA.GSimpleGA(genome)
ga.setMutationRate(0.5)
ga.setPopulationSize(1000)
ga.setGenerations(1000)
ga.setMultiProcessing(True)
ga.evolve(freq_stats=10)
print chromosome_to_expression(ga.bestIndividual().genomeList)
