import sys
from beagle import *

def getTable():
    table={}
    table['A']=0
    table['C']=1
    table['G']=2
    table['T']=3
    table['a']=0
    table['c']=1
    table['g']=2
    table['t']=3
    table['-']=4
    return table


mars = "CCGAG-AGCAGCAATGGAT-GAGGCATGGCG"
saturn  = "GCGCGCAGCTGCTGTAGATGGAGGCATGACG"
jupiter = "GCGCGCAGCAGCTGTGGATGGAAGGATGACG"

nPatterns = len(mars)

returnInfo = BeagleInstanceDetails()

instance = beagleCreateInstance(3,
                                2,
                                3,
                                4,
                                nPatterns,
                                1,
                                4,
                                1,
                                0,
                                None,
                                0,
                                0,
                                0,
                                returnInfo)

if instance<0:
    print "Failed to obtain BEAGLE instance"
    sys.exit()

table = getTable()

marsStates = createStates(mars,table)
saturnStates = createStates(saturn,table)
jupiterStates = createStates(jupiter,table)

beagleSetTipStates(instance,0,marsStates)
beagleSetTipStates(instance,1,saturnStates)
beagleSetTipStates(instance,2,jupiterStates)

patternWeights = createPatternWeights([1]*len(mars))
beagleSetPatternWeights(instance, patternWeights);

freqs = createPatternWeights([0.25]*4)
beagleSetStateFrequencies(instance,0,freqs)

weights = createPatternWeights([1.0])
rates = createPatternWeights([1.0])
beagleSetCategoryWeights(instance, 0, weights)
beagleSetCategoryRates(instance, rates)

evec = createPatternWeights([1.0,  2.0,  0.0,  0.5,
		 1.0,  -2.0,  0.5,  0.0,
		 1.0,  2.0, 0.0,  -0.5,
		 1.0,  -2.0,  -0.5,  0.0])
ivec = createPatternWeights([0.25,  0.25,  0.25,  0.25,
		 0.125,  -0.125,  0.125,  -0.125,
		 0.0,  1.0,  0.0,  -1.0,
		 1.0,  0.0,  -1.0,  0.0])
eval = createPatternWeights([0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333])

beagleSetEigenDecomposition(instance, 0, evec, ivec, eval)

nodeIndices = make_intarray([0,1,2,3])
edgeLengths = make_doublearray([0.1,0.1,0.2,0.1])

beagleUpdateTransitionMatrices(instance,     # instance
	                         0,             # eigenIndex
	                         nodeIndices,   # probabilityIndices
	                         None,          # firstDerivativeIndices
	                         None,          # secondDerivativeIndices
	                         edgeLengths,   # edgeLengths
	                         4);            # count

operations = new_BeagleOperationArray(2)
op0 = make_operation([3,BEAGLE_OP_NONE,BEAGLE_OP_NONE,0,0,1,1])
op1 = make_operation([4,BEAGLE_OP_NONE,BEAGLE_OP_NONE,2,2,3,3])
BeagleOperationArray_setitem(operations,0,op0)
BeagleOperationArray_setitem(operations,1,op1)

beagleUpdatePartials(instance,
                     operations,
                     2,
                     BEAGLE_OP_NONE)
    
logLp = new_doublep()
rootIndex = make_intarray([4])
categoryWeightIndex = make_intarray([0])
stateFrequencyIndex = make_intarray([0])
cumulativeScaleIndex = make_intarray([BEAGLE_OP_NONE])

beagleCalculateRootLogLikelihoods(instance,
                                  rootIndex,
                                  categoryWeightIndex,
                                  stateFrequencyIndex,
                                  cumulativeScaleIndex,
                                  1,
                                  logLp)

logL=doublep_value(logLp)
print(logL)
print("Woof!")
