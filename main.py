import argparse
import logging
import numpy
import random
import sys


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

__author__ = "Manolomon"
__license__ = "MIT"
__version__ = "1.0"

import pickle

def fitness_func(individuo): # Fitness Function
    global X_train
    global y_train
    classifier = MultinomialNB(alpha=individuo[0], 
                               fit_prior = False if individuo[1] < 0.5 else True)
    classifier.fit(X_train, y_train)

    acc = cross_val_score(classifier, X_train, y_train, cv=5)
    del classifier
    return acc.mean()

def reflection(vector): # Boundary constraint-handling
    global limits

    for i in range(len(limits)):
        param_range = limits[i]
        if vector[i] <= param_range[0]: 
            vector[i] = (2 * param_range[0]) - vector[i]
        if vector[i] >= param_range[1]:
            vector[i] = (2 * param_range[1]) - vector[i]
    return vector

def ed_rand_1_bin(np, max_gen, f, cr):
    global limits
    random.seed(8)

    #print(str(np) + ' - ' + str(max_gen) + ' - ' + str(f) + ' - ' + str(cr))
    # Initialize population
    alphas = numpy.random.uniform(low=limits[0][0], high=limits[0][1], size=(np,1))
    fit_priors = numpy.random.uniform(low=limits[1][0], high=limits[1][1], size=(np,1))
    population = numpy.concatenate((alphas, fit_priors), axis=1)
    # First evaluation of population
    logging.debug("Start of evolution")
    fitness = numpy.apply_along_axis(fitness_func, 1, population)
    order = numpy.argsort(fitness)
    population = population[order]

    logging.debug("  Evaluated %i individuals" % len(population))
    # Evolutionary process
    for g in range(max_gen):
        logging.debug("\n-- Generation %i --" % g)
        for i in range (np):
            # Mutation
            no_parent = numpy.delete(population, i, 0)
            # Random pick of individuals
            row_i = numpy.random.choice(no_parent.shape[0], 3, replace=False)
            r = no_parent[row_i, :]
            v_mutation = ((r[0]-r[1]) * f) + r[2]
            # Reflection for boundaries constrain-handling
            v_mutation = reflection(v_mutation)
            # Crossover
            jrand = random.randint(0, 1)
            v_son = numpy.empty([1, 2])
            for j in range(2):
                if random.uniform(0, 1) < cr or j == jrand:
                    v_son[0,j] = v_mutation[j]
                else:
                    v_son[0,j] = population[i,j]
            population = numpy.concatenate((population, v_son), axis=0)
            # Reevaluation
            fitness = numpy.apply_along_axis(fitness_func, 1, population)
            order = numpy.argsort(fitness)[::-1]
            population = population[order]
        logging.debug("Best individual gets %s" % (fitness[0]))
        # Surplus disposal
        population = population[:np]
        fitness = fitness[:np]
    logging.debug("-- End of (successful) evolution --")
    logging.debug("Best individual is [alpha=%s, fit_prior=%s], %s" % (alphas[0], False if fit_priors[0] < 0.5 else True, fitness[0]))
    return fitness[0]

def main(POP, CXPB, MUTPB, DATFILE='main.dat'):
    NGEN = 100
    score = ed_rand_1_bin(POP, NGEN, float(MUTPB), float(CXPB))

    # save the fo values in DATFILE
    with open(DATFILE, 'w') as f:
        f.write(str(score*100))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Feature Selection using GA with DecisionTreeClassifier')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    ap.add_argument('--pop', dest='pop', type=int, required=True, help='Population size')
    ap.add_argument('--cros', dest='cros', type=float, required=True, help='Crossover probability')
    ap.add_argument('--mut', dest='mut', type=float, required=True, help='Mutation probability')
    ap.add_argument('--datfile', dest='datfile', type=str, required=True, help='File where it will be save the score (result)')

    with open('../text-representation/X_train.pickle', 'rb') as data:
        X_train = pickle.load(data)
    with open('../text-representation/y_train.pickle', 'rb') as data:
        y_train = pickle.load(data)
    
    # just check if args are ok
    with open('args.txt', 'w') as f:
        f.write(str(sys.argv))

    limits = [[0,100], # alpha
              [0,1]] # fir_prior

    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug(args)
    main(args.pop, args.mut, args.cros, args.datfile)