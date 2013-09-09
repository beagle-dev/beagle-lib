#include <iostream>
#include <iomanip>
#include <fstream>
#include <numeric>	// needed for accumulate algorithm
#include <cmath>
#include <cstdlib>
#include "fourtaxon.h"

/*-----------------------------------------------------------------------------
| 	Allocates a two-dimensional array of doubles as one contiguous block of
|	memory the dimensions are f by s. The array is set up so that each
|	successive row follows the previous row in memory.
*/
double **NewTwoDArray(unsigned f , unsigned s)
	{
	double **ptr;
	ptr = new double *[f];
	*ptr = new double [f * s];
	for (unsigned fIt = 1 ; fIt < f ; fIt++)
		ptr[fIt] = ptr[fIt -1] +  s ;
	return ptr;
	}

/*-----------------------------------------------------------------------------
|	Delete a two-dimensional array (e.g. one created by NewTwoDArray) and set
|	ptr to NULL.
*/
void DeleteTwoDArray (double ** & ptr)
	{
	if (ptr)
		{
		delete [] * ptr;
		delete [] ptr;
		ptr = NULL;
		}
	}

unsigned rnseed = 1;

/*-----------------------------------------------------------------------------
|	A uniform random number generator. Should initialize global variable
|	`rnseed' to something before calling this function.
*/
double uniform()
	{
#	define MASK32BITS 0x00000000FFFFFFFFL
#	define A 				397204094			// multiplier
#	define M				2147483647			// modulus = 2^31 - 1
#	define MASK_SIGN_BIT	0x80000000
#	define MASK_31_BITS	0x7FFFFFFF

	unsigned	x, y;

	typedef unsigned long long uint64_t;

	uint64_t	w;

	w = (uint64_t)A * rnseed;
	x = (unsigned)(w & MASK32BITS);
	y = (unsigned)(w >> 32);

	y = (y << 1) | (x >> 31);		// isolate high-order 31 bits
	x &= MASK_31_BITS;				// isolate low-order 31 bits
	x += y;							// x'(i + 1) unless overflows
	if (x & MASK_SIGN_BIT) 			// overflow check
		x -= M;						// deal with overflow

	rnseed = x;

	return (1.0 / (M-2)) * (rnseed - 1);
	}

/*-----------------------------------------------------------------------------
|	Constructor simply calls init().
*/
FourTaxonExample::FourTaxonExample()
  : quiet(false)
  , ntaxa(4)
  , niters(10)
  , like_root_node(5)
  , like_parent_index(4)
  , like_child_index(5)
  , transmat_index(4)
  , nsites(0)
  , nrates(4)
  , seed(1)
  , delta(0.2)
  , mu(1.0)
  , instance_handle(-1)
  , rsrc_number(BEAGLE_OP_NONE)
  , use_tip_partials(true)
  , scaling(false)
  , do_rescaling(false)
  , accumulate_on_the_fly(false)
  , dynamic_scaling(false)
  , auto_scaling(false)
  , single(false)
  , require_double(false)
  , calculate_derivatives(0)
  , empirical_derivatives(false)
  , sse_vectorization(false)
    {
	data_file_name = "fourtaxon.dat";
	}

/*-----------------------------------------------------------------------------
|	This function is called if the program encounters an unrecoverable error.
|	After issuing the supplied error message, the program exits, returning 1.
*/
void FourTaxonExample::abort(
  std::string msg)	/**< is the error message to report to the user */
	{
	std::cerr << msg << "\nAborting..." << std::endl;
	std::exit(1);
	}

/*-----------------------------------------------------------------------------
|	This function builds the operations vector based on the value of the 
|	likeroot command line option.
*/
void FourTaxonExample::defineOperations()
	{
	// If the user specified a node to serve as likelihood root, that
	// node will be considered the child, with the parent being the internal
	// node to which it is connected.
	//
	//    A(0)     C(2)
	//      \      /
	//       4----5
	//      /      \
	//    B(1)     D(3)
	//
	like_child_index = like_root_node - 1;
	if (like_child_index == 0 || like_child_index == 1)
		like_parent_index = 4;
	else
		like_parent_index = 5;
	transmat_index = (like_child_index == 4 ? 4 : like_child_index);
	
	// 	std::cerr << "Tip chosen to serve as likelihood root:\n";
	// 	std::cerr << "  like_root_node     = " << like_root_node << '\n';
	// 	std::cerr << "  like_parent_index  = " << like_parent_index << '\n';
	// 	std::cerr << "  like_child_index   = " << like_child_index << '\n';
	// 	std::cerr << "  transmat_index     = " << transmat_index << '\n';
	
	operations.clear();	
	switch (like_child_index)
		{
		case 0:
			// assuming node 0 is the child and node 4 is the parent
			
			// first_cherry uses 2 and 3 to build 5
			operations.push_back(5);	// destination partial to be calculated
			operations.push_back(scaling ? 1 : BEAGLE_OP_NONE);	// destination scaling buffer index to write to
			operations.push_back(BEAGLE_OP_NONE);	// destination scaling buffer index to read from
			operations.push_back(2);	// left child partial index
			operations.push_back(2);	// left child transition matrix index
			operations.push_back(3);	// right child partial index
			operations.push_back(3);	// right child transition matrix index
			
			// second cherry uses 1 and 5 to build 4
			operations.push_back(4);	// destination partial to be calculated
			operations.push_back(scaling ? 2 : BEAGLE_OP_NONE);	// destination scaling buffer index to write to
			operations.push_back(BEAGLE_OP_NONE);	// destination scaling buffer index to read from
			operations.push_back(1);	// left child partial index
			operations.push_back(1);	// left child transition matrix index
			operations.push_back(5);	// right child partial index
			operations.push_back(4);	// right child transition matrix index (note: using transition matrix 4 because there are only 5 of them total)
			
			break;
		case 1:
			// assuming node 1 is the child and node 4 is the parent
			
			// first_cherry uses 2 and 3 to build 5
			operations.push_back(5);	// destination partial to be calculated
			operations.push_back(scaling ? 1 : BEAGLE_OP_NONE);	// destination scaling buffer index to write to
			operations.push_back(BEAGLE_OP_NONE);	// destination scaling buffer index to read from
			operations.push_back(2);	// left child partial index
			operations.push_back(2);	// left child transition matrix index
			operations.push_back(3);	// right child partial index
			operations.push_back(3);	// right child transition matrix index
			
			// second cherry uses 0 and 5 to build 4
			operations.push_back(4);	// destination partial to be calculated
			operations.push_back(scaling ? 2 : BEAGLE_OP_NONE);	// destination scaling buffer index to write to
			operations.push_back(BEAGLE_OP_NONE);	// destination scaling buffer index to read from
			operations.push_back(0);	// left child partial index
			operations.push_back(0);	// left child transition matrix index
			operations.push_back(5);	// right child partial index
			operations.push_back(4);	// right child transition matrix index (note: using transition matrix 4 because there are only 5 of them total)
			
			break;
		case 2:
			// assuming node 2 is the child and node 5 is the parent
			
			// first_cherry uses 0 and 1 to build 4
			operations.push_back(4);	// destination partial to be calculated
			operations.push_back(scaling ? 1 : BEAGLE_OP_NONE);	// destination scaling buffer index to write to
			operations.push_back(BEAGLE_OP_NONE);	// destination scaling buffer index to read from
			operations.push_back(0);	// left child partial index
			operations.push_back(0);	// left child transition matrix index
			operations.push_back(1);	// right child partial index
			operations.push_back(1);	// right child transition matrix index
			
			// second cherry uses 3 and 4 to build 5
			operations.push_back(5);	// destination partial to be calculated
			operations.push_back(scaling ? 2 : BEAGLE_OP_NONE);	// destination scaling buffer index to write to
			operations.push_back(BEAGLE_OP_NONE);	// destination scaling buffer index to read from
			operations.push_back(3);	// left child partial index
			operations.push_back(3);	// left child transition matrix index
			operations.push_back(4);	// right child partial index
			operations.push_back(4);	// right child transition matrix index
			
			break;
		case 3:
			// assuming node 3 is the child and node 5 is the parent
			
			// first_cherry uses 0 and 1 to build 4
			operations.push_back(4);	// destination partial to be calculated
			operations.push_back(scaling ? 1 : BEAGLE_OP_NONE);	// destination scaling buffer index to write to
			operations.push_back(BEAGLE_OP_NONE);	// destination scaling buffer index to read from
			operations.push_back(0);	// left child partial index
			operations.push_back(0);	// left child transition matrix index
			operations.push_back(1);	// right child partial index
			operations.push_back(1);	// right child transition matrix index
			
			// second cherry uses 2 and 4 to build 5
			operations.push_back(5);	// destination partial to be calculated
			operations.push_back(scaling ? 2 : BEAGLE_OP_NONE);	// destination scaling buffer index to write to
			operations.push_back(BEAGLE_OP_NONE);	// destination scaling buffer index to read from
			operations.push_back(2);	// left child partial index
			operations.push_back(2);	// left child transition matrix index
			operations.push_back(4);	// right child partial index
			operations.push_back(4);	// right child transition matrix index
			
			break;
		default:
			// assuming node 4 is the child and node 5 is the parent
			
			// first_cherry uses 0 and 1 to build 4
			operations.push_back(4);	// destination partial to be calculated
			operations.push_back(scaling ? 1 : BEAGLE_OP_NONE);	// destination scaling buffer index to write to
			operations.push_back(BEAGLE_OP_NONE);	// destination scaling buffer index to read from
			operations.push_back(0);	// left child partial index
			operations.push_back(0);	// left child transition matrix index
			operations.push_back(1);	// right child partial index
			operations.push_back(1);	// right child transition matrix index
			
			// second cherry uses 2 and 3 to build 5
			operations.push_back(5);	// destination partial to be calculated
			operations.push_back(scaling ? 2 : BEAGLE_OP_NONE);	// destination scaling buffer index to write to
			operations.push_back(BEAGLE_OP_NONE);	// destination scaling buffer index to read from
			operations.push_back(2);	// left child partial index
			operations.push_back(2);	// left child transition matrix index
			operations.push_back(3);	// right child partial index
			operations.push_back(3);	// right child transition matrix index
		}
	}
	
/*-----------------------------------------------------------------------------
|	This function sets up the beagle library and initializes all data members.
*/
void FourTaxonExample::initBeagleLib()
	{
	int code;
	partial.resize(ntaxa);

	// hard coded tree topology is (A,B,(C,D))
	// where taxon order is A, B, C, D in the data file
	// Assume nodes 0..3 are the tip node indices
	// Assume node 4 is ancestor of A,B (0,1)
	// Assume node 5 is ancestor of C,D (2,3)
	//    B(1)     C(2)
	//      \      /
	//       4----5
	//      /      \
	//    A(0)     D(3)
	
	int* rsrcList = NULL;
	int  rsrcCnt = 0;
	if (rsrc_number != BEAGLE_OP_NONE) {
		rsrcList = new int[1];
		rsrcList[0] = rsrc_number;
		rsrcCnt = 1;
	}
        
    long requirementFlags = 0;
    if (single) {
        requirementFlags |= BEAGLE_FLAG_PRECISION_SINGLE;
    }
        
    if (require_double) {
        requirementFlags |= BEAGLE_FLAG_PRECISION_DOUBLE;
    }
        
	int mtrxCount = ntaxa + 1; 
		
	if (calculate_derivatives == 1)
		mtrxCount *= 2;
    else if (calculate_derivatives == 2)
        mtrxCount *= 3;

    BeagleInstanceDetails instDetails;
        
	instance_handle = beagleCreateInstance(
				ntaxa,		// tipCount
				ntaxa + 2,	// partialsBufferCount
				(use_tip_partials ? 0 : ntaxa),			// compactBufferCount
				4, 			// stateCount
				nsites,		// patternCount
				1,			// eigenBufferCount
				mtrxCount,	// matrixBufferCount,
                nrates,     // categoryCount
                3,          // scalingBuffersCount                
				rsrcList,	// resourceList
				rsrcCnt,	// resourceCount
				(sse_vectorization ? BEAGLE_FLAG_VECTOR_SSE : 0) | (auto_scaling ? BEAGLE_FLAG_SCALING_AUTO : 0),         // preferenceFlags
				requirementFlags,			// requirementFlags
				&instDetails);
	
	if (rsrc_number != BEAGLE_OP_NONE)
		delete[] rsrcList;

	if (instance_handle < 0)
		abort("Failed: beagleCreateInstance returned a negative instance handle (and that's not good)");
        
    int rNumber = instDetails.resourceNumber;
    //BeagleResourceList* rList = beagleGetResourceList();
    fprintf(stdout, "Using resource %i:\n", rNumber);
    fprintf(stdout, "\tRsrc Name : %s\n", instDetails.resourceName);
    fprintf(stdout, "\tImpl Name : %s\n", instDetails.implName);
    fprintf(stdout, "\n");        
        
	brlens.resize(5);
	transition_matrix_index.resize(5 * 3);
	for (unsigned i = 0; i < 5; ++i)
		{
		brlens[i] = 0.01;
		transition_matrix_index[i] = i;
		transition_matrix_index[i + 5] = i + 5; // first derivative indices
		transition_matrix_index[i + 10] = i + 10; // second derivative indices
		}

	for (unsigned i = 0; i < ntaxa; ++i)
		{
		
		if (use_tip_partials)
			{
			code = beagleSetTipPartials(
						instance_handle,			// instance
						i,							// bufferIndex
						&partial[i][0]);			// inPartials
			if (code != 0)
				abort("beagleSetTipPartials encountered a problem");
			}
		else 
			{
			code = beagleSetTipStates(
						instance_handle,			// instance);
						i,							// bufferIndex
						(int*)&data[i][0]);
			if (code != 0)
				abort("beagleSetTipStates encountered a problem");
			}
		}

#ifdef _WIN32
	std::vector<double> rates(nrates);
#else
    double rates[nrates];
#endif

	if (nrates == 4)
		{
		// This branch will be visited unless original value of nrates has been changed
		rates[0] = 0.03338775;
		rates[1] = 0.25191592;
		rates[2] = 0.82026848;
		rates[3] = 2.89442785;
		}
	else
		{
		// If value of nrates set in the constructor is not 4 (the original value),
		// drop back to setting all rates equal since we have no code in this example
		// for computing the discrete gamma rate boundaries and mean rates
		for (int i = 0; i < nrates; i++) {
		    rates[i] = 1.0;
			}
		}
        
    beagleSetCategoryRates(instance_handle,
#ifdef _WIN32		
		&rates[0]
#else
		rates
#endif
		);

#ifdef _WIN32
        std::vector<double> patternWeights(nsites);
#else
        double patternWeights[nsites];
#endif
        
    for (int i = 0; i < nsites; i++) {
        patternWeights[i] = 1.0;
    }    
        
    beagleSetPatternWeights(instance_handle, &patternWeights[0]);
        
        
	// JC69 model eigenvector matrix
	double evec[4 * 4] = {
		 1.0,  2.0,  0.0,  0.5,
		 1.0,  -2.0,  0.5,  0.0,
		 1.0,  2.0, 0.0,  -0.5,
		 1.0,  -2.0,  -0.5,  0.0
	};

	// JC69 model inverse eigenvector matrix
	double ivec[4 * 4] = {
		 0.25,  0.25,  0.25,  0.25,
		 0.125,  -0.125,  0.125,  -0.125,
		 0.0,  1.0,  0.0,  -1.0,
		 1.0,  0.0,  -1.0,  0.0
	};

	// JC69 model eigenvalues
	double eval[4] = { 0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333 };

	code = beagleSetEigenDecomposition(
			   instance_handle,					// instance
			   0,								// eigenIndex,
			   (const double *)evec,			// inEigenVectors,
			   (const double *)ivec,			// inInverseEigenVectors,
			   eval);							// inEigenValues

	if (code != 0)
		abort("beagleSetEigenDecomposition encountered a problem");
        
    if (auto_scaling) {
        scaleIndices.resize(2);
        scaleIndices[0] = 4;
        scaleIndices[1] = 5;        
    } else if (scaling && !accumulate_on_the_fly) {
        scaleIndices.resize(2);
        scaleIndices[0] = 1;
        scaleIndices[1] = 2;
    }
            
	}

/*-----------------------------------------------------------------------------
|	Calculates the log likelihood by calling the beagle functions
|	beagleUpdateTransitionMatrices, beagleUpdatePartials and beagleCalculateEdgeLogLikelihoods.
*/
double FourTaxonExample::calcLnL(int return_value)
	{
	int code = beagleUpdateTransitionMatrices(
			instance_handle,				// instance,
			0,								// eigenIndex,
			&transition_matrix_index[0],	// probabilityIndices,
			(calculate_derivatives > 0 ? &transition_matrix_index[5] : NULL),	// firstDerivativeIndices,
			(calculate_derivatives > 1 ? &transition_matrix_index[10] : NULL),	// secondDerivativeIndices,
			&brlens[0],						// edgeLengths,
			5);								// count

	if (code != 0)
		abort("beagleUpdateTransitionMatrices encountered a problem");
        
    int cumulativeScalingFactorIndex = (scaling ? 0 : BEAGLE_OP_NONE);
        
    if (do_rescaling) // Perform rescaling during this likelihood evaluation
        beagleResetScaleFactors(instance_handle, cumulativeScalingFactorIndex);
      
	code = beagleUpdatePartials(
		   instance_handle,                                 // instance
		   (BeagleOperation*)&operations[0],                                   // operations
		   2,                                                // operationCount
           (accumulate_on_the_fly ?                          
            cumulativeScalingFactorIndex : BEAGLE_OP_NONE)); // cumulative scale index

	if (code != 0)
		abort("beagleUpdatePartials encountered a problem");

	int parentBufferIndex = like_parent_index;
	int childBufferIndex  = like_child_index;
	int transitionMatrixIndex  = transmat_index;
	int firstDerivMatrixIndex  = transmat_index + 5;
	int secondDerivMatrixIndex  = transmat_index + 10;
    int stateFrequencyIndex = 0;
    int categoryWeightsIndex = 0;
        
#ifdef _WIN32
	std::vector<double> relativeRateProb(nrates);
#else
	double relativeRateProb[nrates];
#endif

    for (int i = 0; i < nrates; i++) {
        relativeRateProb[i] = 1.0 / nrates;
    }
        
    if (auto_scaling) {
        code = beagleAccumulateScaleFactors(instance_handle, &scaleIndices[0], 2, BEAGLE_OP_NONE);
    } else if (do_rescaling && !accumulate_on_the_fly) { // Accumulate scale factors if not on-the-fly
        code = beagleAccumulateScaleFactors(
             instance_handle,
             &scaleIndices[0],
             2, 
             cumulativeScalingFactorIndex);
    }
        
	double stateFreqs[4] = { 0.25, 0.25, 0.25, 0.25 };
        
    beagleSetStateFrequencies(instance_handle, 0, stateFreqs);        
        
    beagleSetCategoryWeights(instance_handle, 0, &relativeRateProb[0]);

    double lnL = 0.0;
	double firstDeriv = 0.0;
	double secondDeriv = 0.0;

	code = beagleCalculateEdgeLogLikelihoods(
		 instance_handle,					// instance,
		 &parentBufferIndex,				// parentBufferIndices
		 &childBufferIndex,					// childBufferIndices
		 &transitionMatrixIndex,			// probabilityIndices
		 (calculate_derivatives > 0 ? &firstDerivMatrixIndex : NULL),	// firstDerivativeIndices
		 (calculate_derivatives > 1 ? &secondDerivMatrixIndex : NULL),	// secondDerivativeIndices
		 &categoryWeightsIndex,	// weights
		 &stateFrequencyIndex,			// stateFrequencies,
         &cumulativeScalingFactorIndex,
		 1,									// count
		 &lnL,							// outLogLikelihoods,
		 (calculate_derivatives > 0 ? &firstDeriv : NULL),	  // outFirstDerivatives,
		 (calculate_derivatives > 1 ? &secondDeriv : NULL));	  // outSecondDerivatives

	if (code != 0)
		abort("beagleCalculateEdgeLogLikelihoods encountered a problem");

    if (dynamic_scaling) {
        operations[1] = BEAGLE_OP_NONE; // Set write scale buffer (op1) to NONE
        operations[8] = BEAGLE_OP_NONE; // Set write scale buffer (op2) to NONE
        operations[2] = 1;              // Set read scale buffer (op1)
        operations[9] = 2;              // Set read scale buffer (op2)
        do_rescaling = false;           // Turn off calculating of scale factors
    }
	
	double return_sum = 0;
		
	if (return_value == 0)
		return_sum = lnL;
	else if (return_value == 1)
		return_sum = firstDeriv;
	else if (return_value == 2)
		return_sum = secondDeriv;


	return return_sum;
	}

/*-----------------------------------------------------------------------------
|	Updates a single branch length using a simple sliding-window Metropolis-
|	Hastings proposal. A window of width `delta' is centered over the current
|	branch length (x0) and the proposed new branch length is chosen from a
|	uniform(x0 - delta/2, x0 + delta/2) distribution.
*/
void FourTaxonExample::updateBrlen(
  unsigned brlen_index)		/**< is the index of the branch length to update */
	{
	// current state is x0
	double x0 = brlens[brlen_index];

	// proposed new state is x
	double x = x0 - delta/2.0 + delta*uniform();

	// reflect back into valid range if necessary
	if (x < 0.0)
		x = -x;

	// branch length prior is exponential with mean mu
	// (note: leaving out log(mu) because it will cancel anyway)
	double log_prior_before = -x0/mu;
	double log_prior_after  = -x/mu;

	// compute log-likelihood before and after move
	// (not being particularly efficient here because we are testing the beagle library
	// so the more calls to calcLnL the better)
	double log_like_before = calcLnL(0);
	brlens[brlen_index] = x;
	double log_like_after = calcLnL(0);

	double log_accept_ratio = log_prior_after + log_like_after - log_prior_before - log_like_before;
	double u = log(uniform());
	if (u > log_accept_ratio)
		{
		// proposed new branch length rejected, restore original branch length
		brlens[brlen_index] = x0;
		}
	}

/*-----------------------------------------------------------------------------
|	Reads the data file the name of which is supplied. This function expects
|	the data to be in pseufoPHYLIP format: ntaxa followed by nsites on first
|	line, then name and sequence data (separated by whitespace) for each
|	taxon on subsequent lines. Converts the DNA states A, C, G, and T to int
|	codes 0, 1, 2 and 3, respectively, storing these in the `data' data member.
|	Also calculates the partials for the tip nodes, which are stored in the
|	`partial' data member.
*/
void FourTaxonExample::readData()
	{
	std::string sequence;
	std::ifstream inf(data_file_name.c_str());
	if(!inf.good())
		abort("problem reading data file");	
	inf >> nsites;	// ntaxa is a constant (4) in this example
	taxon_name.resize(ntaxa);
	data.resize(ntaxa);
	partial.resize(ntaxa);
	for (unsigned i = 0; i < ntaxa; ++i)
		{
		inf >> taxon_name[i];
		inf >> sequence;
		data[i].resize(nsites);
		partial[i].reserve(nsites*4);
		for (unsigned j = 0; j < nsites; ++j)
			{
			switch (sequence[j])
				{
				case 'a':
				case 'A':
					data[i][j] = 0;
					partial[i].push_back(1.0);
					partial[i].push_back(0.0);
					partial[i].push_back(0.0);
					partial[i].push_back(0.0);
					break;
				case 'c':
				case 'C':
					data[i][j] = 1;
					partial[i].push_back(0.0);
					partial[i].push_back(1.0);
					partial[i].push_back(0.0);
					partial[i].push_back(0.0);
					break;
				case 'g':
				case 'G':
					data[i][j] = 2;
					partial[i].push_back(0.0);
					partial[i].push_back(0.0);
					partial[i].push_back(1.0);
					partial[i].push_back(0.0);
					break;
				case 't':
				case 'T':
					data[i][j] = 3;
					partial[i].push_back(0.0);
					partial[i].push_back(0.0);
					partial[i].push_back(0.0);
					partial[i].push_back(1.0);
					break;
				default:
					data[i][j] = 4;
					partial[i].push_back(1.0);
					partial[i].push_back(1.0);
					partial[i].push_back(1.0);
					partial[i].push_back(1.0);
				}
			}
		}
	inf.close();
	}

/*-----------------------------------------------------------------------------
|	This function spits out the data as a nexus formatted data file with a
|	paup block that can be used to check the starting likelihood in PAUP*.
*/
void FourTaxonExample::writeData()
	{
	std::ofstream outf("check_lnL_using_paup.nex", std::ios::out);
	outf << "#nexus\n\n";
	outf << "begin data;\n";
	outf << "  dimensions ntax=" << ntaxa << " nchar=" << nsites << ";\n";
	outf << "  format datatype=dna missing=? gap=-;\n";
	outf << "  matrix\n";
	for (unsigned i = 0; i < ntaxa; ++i)
		{
		outf << "    " << taxon_name[i] << "\t\t\t";
		for (unsigned j = 0; j < nsites; ++j)
			{
			switch (data[i][j])
				{
				case 0:
					outf << 'a';
					break;
				case 1:
					outf << 'c';
					break;
				case 2:
					outf << 'g';
					break;
				case 3:
					outf << 't';
					break;
				default:
					outf << '?';
				}
			}
			outf << std::endl;
		}
	outf << "  ;\n";
	outf << "end;\n\n";
	outf << "begin paup;\n";
	outf << "  set criterion=likelihood storebrlens;\n";
	outf << "end;\n\n";
	outf << "begin trees;\n";
	outf << "  tree starting = [&U](alga_D86836:0.01, fern_D14882:0.01, (hops_AF206777:0.01, corn_Z11973:0.01):0.01);\n";
	outf << "end;\n\n";
	outf << "begin paup;\n";
	if (nrates == 4)
		outf << "  lset nst=1 basefreq=equal rates=gamma shape=0.5;\n";
	else
		outf << "  lset nst=1 basefreq=equal rates=equal;\n";
	outf << "  lscores 1 / userbrlen;\n";
	outf << "end;\n";
	outf.close();
	}

/*-----------------------------------------------------------------------------
|	Reads in the data file (which must contain data for exactly four taxa), 
|	then calls calcLnL `n' times before calling finalize to inform the beagle 
|	library that it is ok to destroy all allocated memory.
*/
void FourTaxonExample::run()
	{
	::rnseed = seed;
	readData();
	initBeagleLib();
	writeData();

	if (!quiet)
		{
		std::cout.setf(std::ios::showpoint);
		std::cout.setf(std::ios::floatfield, std::ios::fixed);
		std::cout << std::setw(12) << "iter" << std::setw(24) << "log-likelihood" << std::setw(24) << "tree length";
		if (calculate_derivatives > 0)
			std::cout << std::setw(24) << "first derivative";
        if (calculate_derivatives > 1)
            std::cout << std::setw(24) << "second derivative";
        if (empirical_derivatives)
            std::cout << std::setw(24) << "first deriv (emp)" << std::setw(24) << "second deriv (emp)";
		std::cout << std::endl;
		}
		
	for (unsigned rep = 0; rep <= niters; ++rep)
		{
        if (rep > 0)
            {
            for (unsigned b = 0; b < 5; ++b)
                updateBrlen(b);
            }
			
		if (!quiet)
			{
			std::cout << std::setw(12) << rep;
			std::cout << std::setw(24) << std::setprecision(5) << calcLnL(0);
			std::cout << std::setw(24) << std::setprecision(5) << std::accumulate(brlens.begin(), brlens.end(), 0.0);
            if (calculate_derivatives > 0)
				std::cout << std::setw(24) << std::setprecision(5) << calcLnL(1);
            if (calculate_derivatives > 1)
                std::cout << std::setw(24) << std::setprecision(5) << calcLnL(2);
            if (empirical_derivatives)
                {
                double startBrlens = brlens[transmat_index];
                double incr = startBrlens * 0.001;
                double startLnL = calcLnL(0);
                brlens[transmat_index] = startBrlens - incr;
                double deltaMinus = ((calcLnL(0) - startLnL) / -incr);
                brlens[transmat_index] = startBrlens + incr;
                double deltaPlus = ((calcLnL(0) - startLnL) / incr);
                double empD1 = (deltaMinus + deltaPlus) /  2;                
                double empD2 = (deltaPlus - deltaMinus) / incr;
                std::cout << std::setw(24) << std::setprecision(5) << empD1 << std::setw(24) << empD2;
                brlens[transmat_index] = startBrlens;
                }
                
			std::cout << std::endl;
			}
		}

	int code = beagleFinalizeInstance(
		instance_handle);		// instance

	if (code != 0)
		abort("beagleFinalizeInstance encountered a problem");
	}

/*-----------------------------------------------------------------------------
|	Reads command line arguments and interprets them.
*/
void FourTaxonExample::helpMessage()
	{
	std::cerr << "Usage:\n\n";
	std::cerr << "fourtaxon [--help] [--quiet] [--niters <integer>] [--datafile <string>]";
	std::cerr << " [--rsrc <integer>] [--likeroot <integer>]  [--scaling <integer>] [--single] [--double] [--calcderivs] [--empiricalderivs] [--sse]\n\n";
	std::cerr << "If --help is specified, this usage message is shown\n\n";
	std::cerr << "If --quiet is specified, no progress reports will be issued (allowing for\n";
	std::cerr << "        more accurate timing).\n\n";
	std::cerr << "If --niters is specified, the MCMC sampler will be run for the specified\n";
	std::cerr << "        number of iterations. The default number of iterations is 10.\n\n";
	std::cerr << "If --datafile is specified, the file should have the same format as the\n";
	std::cerr << "        (default) data file \"fourtaxon.dat\" and should only contain sequences for\n";
	std::cerr << "        4 taxa (although the sequences can be of arbitrary length).\n\n";
	std::cerr << "If --rsrc is specified, the BEAGLE resource specified will be employed.\n\n";
	std::cerr << "If --likeroot is specified, the likelihood will be computed across the\n";
	std::cerr << "        edge associated with the specified node. The nodes are indexed thusly:\n";
	std::cerr << "        1          3    If 1, 2, 3 or 4 is specified, the likelihood will   \n";
	std::cerr << "          \\       /    be computed across the corresponding terminal edge. \n";
	std::cerr << "           5-----+      If 5 is chosen, the likelihood will be computed     \n";
	std::cerr << "          /       \\    across the internal edge. Default is 5.             \n";
	std::cerr << "        2           4                                                       \n\n";
    std::cerr << "If --scaling is specified, 0 = no rescaling,\n";
    std::cerr << "                           1 = rescale and accumulate scale factors on the fly\n";
    std::cerr << "                           2 = rescale and accumulate scale factors at once\n";
    std::cerr << "                           3 = rescale once at first evaluation (dynamic)\n";
    std::cerr << "                           4 = automatically rescale when necessary\n\n";
    std::cerr << "If --single is specified, then require single precision mode\n\n";
    std::cerr << "If --double is specified, then require double precision mode\n\n";
    std::cerr << "If --calcderivs is specified, 0 = no calculation of edge likelihood derivatives\n";
    std::cerr << "                              1 = calculate first order edge likelihood derivatives\n";
    std::cerr << "                              2 = calculate first and second order edge likelihood derivatives\n\n";
    std::cerr << "If --empiricalderivs is specified, then empirically calculate first and second order edge likelihood derivatives\n\n";
    std::cerr << "If --sse is specified, then the SSE implementation is enabled\n\n";        
	std::cerr << std::endl;
	std::exit(0);
	}

/*-----------------------------------------------------------------------------
|	Reads command line arguments and interprets them.
*/
void FourTaxonExample::interpretCommandLineParameters(
  int argc, 		/**< is the number of command line arguments */
  char* argv[])		/**< is the array of command line arguments */
	{
	bool expecting_niters = false;
	bool expecting_filename = false;
	bool expecting_rsrc_number = false;
	bool expecting_likerootnode = false;
    bool expecting_scaling_number = false;
    bool expecting_calculate_derivatives = false;
	for (unsigned i = 1; i < argc; ++i)
		{
		std::string option = argv[i];
		if (expecting_niters)
			{
			std::cerr << "niters option: " << option << std::endl;
			niters = (unsigned)atoi(option.c_str());
			std::cerr << "niters = " << niters << std::endl;
			expecting_niters = false;
			}
		else if (expecting_filename)
			{
			data_file_name = option.c_str();
			expecting_filename = false;
			}
		else if (expecting_rsrc_number) {
			std::cerr << "rsrc_number option: " << option << std::endl;
			rsrc_number = (unsigned)atoi(option.c_str());
			std::cerr << "rsrc_number = " << rsrc_number << std::endl;
			expecting_rsrc_number = false;
			if (rsrc_number < 0)
				abort("invalid BEAGLE resource number supplied on the command line");
			}
		else if (expecting_likerootnode)
			{
			std::cerr << "likeroot option: " << option << std::endl;
			like_root_node = (unsigned)atoi(option.c_str());
			std::cerr << "like_root_node = " << like_root_node << std::endl;
			expecting_likerootnode = false;
			}
        else if (expecting_scaling_number)
            {
            std::cerr << "scaling option: " << option << std::endl;
            int noption = (unsigned)atoi(option.c_str());
            scaling = false;
            accumulate_on_the_fly = true;
            do_rescaling = false;
            auto_scaling = false;
            if (noption >= 1 && noption < 4)
                {                
                scaling = true;
                do_rescaling = true;
                }
            if (noption >= 2 && noption < 4)
                accumulate_on_the_fly = false;
            if (noption == 3)                 
                dynamic_scaling = true;
            if (noption == 4)
                auto_scaling = true;
            expecting_scaling_number = false;
            if (noption < 0 || noption > 4)
                abort("invalid scaling option supplied on the command line");
             }
		else if (expecting_calculate_derivatives)
            {
			std::cerr << "calcderivs option: " << option << std::endl;
			calculate_derivatives = (unsigned)atoi(option.c_str());
			std::cerr << "calculate_derivatives = " << calculate_derivatives << std::endl;
			expecting_calculate_derivatives = false;
            }
		else if (option == "--help")
			{
			helpMessage();
			}
		else if (option == "--quiet")
			{
			quiet = true;
			}
        else if (option == "--single")
            {
            single = true;
            }
        else if (option == "--double")
        {
            require_double = true;
        }
		else if (option == "--likeroot")
			{
			expecting_likerootnode = true;
			}
		else if (option == "--niters")
			{
			expecting_niters = true;
			}
		else if (option == "--filename")
			{
			expecting_filename = true;
			}
		else if (option == "--rsrc")
			{
			expecting_rsrc_number = true;
			}
        else if (option == "--scaling")
            {
            expecting_scaling_number = true;
            }
		else if (option == "--usetipstates") 
			{
			use_tip_partials = false;
			}
		else if (option == "--calcderivs")
			{
			expecting_calculate_derivatives = true;
			}
        else if (option == "--empiricalderivs")
            {
            empirical_derivatives = true;
            }
        else if (option == "--sse")
        {
            sse_vectorization = true;
        }
		else 
			{
			std::string msg("Unknown command line parameter \"");
			msg.append(option);
			msg.append("\"  If this was intended to be the file path, it should be preceded by a --filename argument");
			abort(msg.c_str());
			}
		}

	if (expecting_niters)
		abort("read last command line option without finding value associated with --niters");
		
	if (expecting_filename)
		abort("read last command line option without finding value associated with --filename");
		
	if (expecting_rsrc_number)
		abort("read last command line option without finding value associated with --rsrc");
	
	if (expecting_likerootnode)
		abort("read last command line option without finding value associated with --likeroot");
	
	if (niters < 1)
		abort("invalid number of iterations supplied on the command line");
		
	if (like_root_node < 1)
		abort("invalid node number specified for --likeroot option (should be 1, 2, 3, 4, or 5)");
		
	if (like_root_node > 5)
		abort("invalid node number specified for --likeroot option (should be 1, 2, 3, 4, or 5)");
        
    if (expecting_scaling_number)
        abort("read last command line options with finding value associated with --scaling");

	// Now that we know what edge will be used for the likelihood calculation, we can
	// define the operations vector that the library will use to perform updates of partials
	defineOperations();

	std::cout << "quiet                = " << (quiet ? "true" : "false") << '\n';
	std::cout << "number of iterations = " << niters << '\n';
	std::cout << "data file name       = " << data_file_name << '\n';
	std::cout << std::endl;
	}

/*-----------------------------------------------------------------------------
|	Constructs a FourTaxonExample object, calls its
|	interpretCommandLineParameters to (optionally) change the number of
|	iterations and data file name, and then calls the run() function to
|	initialize the beagle library and start the analysis.
*/
int main(
  int argc, 		/**< is the number of command line arguments */
  char* argv[])		/**< is the array of command line arguments */
	{
	FourTaxonExample fourtax;
	fourtax.interpretCommandLineParameters(argc, argv);
	fourtax.run();
#ifdef _WIN32
    std::cout << "\nPress ENTER to exit...\n";
    fflush( stdout);
    fflush( stderr);
    getchar();
#endif
	}
