/**
 * An example hello world program for libhmsbeagle
 * @author Aaron Darling
 */

#include <config.h>
#include <libhmsbeagle/beagle.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

/**
 * DNA of three imaginary organisms from exotic locales
 */
string mars    = "CCGAG-AGCAGCAATGGAT-GAGGCATGGCG";
string saturn  = "GCGCGCAGCTGCTGTAGATGGAGGCATGACG";
string jupiter = "GCGCGCAGCAGCTGTGGATGGAAGGATGACG";

/**
 * Machinery to translate nucleotides into numeric states
 */
static int* getTable(){
	static int* table = new int[128];
	table['A']=0;
	table['C']=1;
	table['G']=2;
	table['T']=3;
	table['a']=0;
	table['c']=1;
	table['g']=2;
	table['t']=3;
	table['-']=4;
	return table;
}

void createStates(const string& nucleotides, vector<int>& states){
	static int* ttab = getTable();
	states.resize(nucleotides.size());
	for(int i=0; i<nucleotides.size(); i++){
		 states[i] = ttab[ nucleotides[i] ];
	}
}

/**
 * begin a simple example to illustrate how the BEAGLE library might be used to calculate likelihood for a three leaf tree
 */
int main(int argc, char* argv[]){

	// get the number of site patterns.  These could optionally be compressed
	int nPatterns = mars.length();	
	BeagleInstanceDetails* returnInfo = new BeagleInstanceDetails();

	// create an instance of the BEAGLE library
	int instance = beagleCreateInstance(
				3,		/**< Number of tip data elements (input) */
				2,	        /**< Number of partials buffers to create (input) -- internal node count */
				3,		/**< Number of compact state representation buffers to create -- for use with setTipStates (input) */
				4,		/**< Number of states in the continuous-time Markov chain (input) -- DNA */
				nPatterns,	/**< Number of site patterns to be handled by the instance (input) -- not compressed in this case */
				1,		/**< Number of eigen-decomposition buffers to allocate (input) */
				4,		/**< Number of transition matrix buffers (input) -- one per edge */
				1,		/**< Number of rate categories */
				0,		/**< Number of scaling buffers -- can be zero if scaling is not needed*/
				NULL,		/**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
				0,		/**< Length of resourceList list (input) -- not needed to use the default hardware config */
				0,		/**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
				0,		/**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
				returnInfo
				);

	if (instance < 0) {
		cerr << "Failed to obtain BEAGLE instance\n\n";
		return -1;
	}

	// set the states at each of the tree tips
	vector<int> marsStates, saturnStates, jupiterStates;
	createStates(mars, marsStates);
	createStates(saturn, saturnStates);
	createStates(jupiter, jupiterStates);

	beagleSetTipStates(instance, 0, marsStates.data());
	beagleSetTipStates(instance, 1, saturnStates.data());
	beagleSetTipStates(instance, 2, jupiterStates.data());

	// let all sites have equal weight
	vector<double> patternWeights( marsStates.size(), 1 );
	beagleSetPatternWeights(instance, patternWeights.data());

	// create array of state background frequencies
	double freqs[4] = { 0.25, 0.25, 0.25, 0.25 };
	beagleSetStateFrequencies(instance, 0, freqs);

	// create an array containing site category weights and rates
	const double weights[1] = { 1.0 };
	const double rates[1] = { 1.0 };
	beagleSetCategoryWeights(instance, 0, weights);
	beagleSetCategoryRates(instance, rates);


	// an eigen decomposition for the JC69 model
	double evec[4 * 4] = {
		 1.0,  2.0,  0.0,  0.5,
		 1.0,  -2.0,  0.5,  0.0,
		 1.0,  2.0, 0.0,  -0.5,
		 1.0,  -2.0,  -0.5,  0.0
	};

	double ivec[4 * 4] = {
		 0.25,  0.25,  0.25,  0.25,
		 0.125,  -0.125,  0.125,  -0.125,
		 0.0,  1.0,  0.0,  -1.0,
		 1.0,  0.0,  -1.0,  0.0
	};

	double eval[4] = { 0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333 };

	// set the Eigen decomposition
	beagleSetEigenDecomposition(instance, 0, evec, ivec, eval);

	// a list of indices and edge lengths
	// these get used to tell beagle which edge length goes with which node
	int nodeIndices[4] = { 0, 1, 2, 3 };
	double edgeLengths[4] = { 0.1, 0.1, 0.2, 0.1 };

	// tell BEAGLE to populate the transition matrices for the above edge lengthss
	beagleUpdateTransitionMatrices(instance,     // instance
	                         0,             // eigenIndex
	                         nodeIndices,   // probabilityIndices
	                         NULL,          // firstDerivativeIndices
	                         NULL,          // secondDervativeIndices
	                         edgeLengths,   // edgeLengths
	                         4);            // count

	// create a list of partial likelihood update operations
	// the order is [dest, sourceScaling, destScaling, source1, matrix1, source2, matrix2]
	// these operations say: first peel node 0 and 1 to calculate the per-site partial likelihoods, and store them
        // in buffer 3.  Then peel node 2 and buffer 3 and store the per-site partial likelihoods in buffer 4.
	BeagleOperation operations[2] = {
		{3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1},
		{4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2, 2, 3, 3}
	};

	// this invokes all the math to carry out the likelihood calculation
	beagleUpdatePartials( instance,      // instance
	                operations,     // eigenIndex
	                2,              // operationCount
	                BEAGLE_OP_NONE);             // cumulative scale index

	double logL = 0;
	int rootIndex[1] = {4};
	int categoryWeightIndex[1] = {0};
	int stateFrequencyIndex[1] = {0};
	int cumulativeScaleIndex[1] = {BEAGLE_OP_NONE};

	// calculate the site likelihoods at the root node
	// this integrates the per-site root partial likelihoods across sites, background state frequencies, and rate categories
	// results in a single log likelihood, output here into logL
	beagleCalculateRootLogLikelihoods(instance,               // instance
	                            rootIndex,// bufferIndices
	                            categoryWeightIndex,                // weights
	                            stateFrequencyIndex,                 // stateFrequencies
				    cumulativeScaleIndex,	   // scaleBuffer to use
	                            1,                      // count
	                            &logL);         // outLogLikelihoods

	cout << "logL = " << logL << "\n\n";
	cout << "Woof!\n";
	return 0;
}
