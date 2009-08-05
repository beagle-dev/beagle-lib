#include <iostream>
#include <iomanip>
#include <fstream>
#include <numeric>	// needed for accumulate algorithm
#include <cmath>
#include <cstdlib>
#include "fourtaxon.hpp"

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
  : ntaxa(4), niters(0), nsites(0), seed(1), delta(0.2), mu(1.0), instance_handle(-1)
	{
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
|	This function sets up the beagle library and initializes all data members.
*/
void FourTaxonExample::initBeagleLib()
	{
	int code;
	partial.resize(4);

	// hard coded tree topology is (A,B,(C,D))
	// where taxon order is A, B, C, D in the data file
	// Assume nodes 0..3 are the tip node indices
	// Assume node 4 is ancestor of A,B (0,1)
	// Assume node 5 is ancestor of C,D (2,3)
	operations.push_back(4);	// destination (to be calculated)
	operations.push_back(1);	// destination scaling buffer index to write to
	operations.push_back(BEAGLE_OP_NONE);	// destination scaling buffer index to read from
	operations.push_back(0);	// left child partial index
	operations.push_back(0);	// left child transition matrix index
	operations.push_back(1);	// right child partial index
	operations.push_back(1);	// right child transition matrix index

	operations.push_back(5);	// destination (to be calculated)
	operations.push_back(2);	// destination scaling buffer index to write to
    operations.push_back(BEAGLE_OP_NONE);	// destination scaling buffer index to read from
	operations.push_back(2);	// left child partial index
	operations.push_back(2);	// left child transition matrix index
	operations.push_back(3);	// right child partial index
	operations.push_back(3);	// right child transition matrix index

	instance_handle = createInstance(
				4,			// tipCount
				6,			// partialsBufferCount
				0,			// compactBufferCount
				4, 			// stateCount
				nsites,		// patternCount
				1,			// eigenBufferCount
				5,			// matrixBufferCount,
                1,          // categoryCount
                3,          // scalingBuffersCount
				NULL,		// resourceList
				0,			// resourceCount
				0L,			// preferenceFlags
				0L			// requirementFlags
				);

	if (instance_handle < 0)
		abort("createInstance returned a negative instance handle (and that's not good)");
        
    BeagleInstanceDetails instDetails;
    code = initializeInstance(instance_handle, &instDetails);    
    if (code != 0) {
			abort("initializeInstance encountered a problem");
    }
        
    int rNumber = instDetails.resourceNumber;
    BeagleResourceList* rList = BeagleGetResourceList();
    fprintf(stdout, "Using resource %i:\n", rNumber);
    fprintf(stdout, "\tName : %s\n", rList->list[rNumber].name);
    fprintf(stdout, "\tDesc : %s\n", rList->list[rNumber].description);
    fprintf(stdout, "\n");        
        
	transition_matrix_index.resize(5);
	transition_matrix_index[0] = 0;
	transition_matrix_index[1] = 1;
	transition_matrix_index[2] = 2;
	transition_matrix_index[3] = 3;
	transition_matrix_index[4] = 4;

	brlens.resize(5);
	brlens[0] = 0.01;
	brlens[1] = 0.02;
	brlens[2] = 0.03;
	brlens[3] = 0.04;
	brlens[4] = 0.05;

	for (unsigned i = 0; i < 4; ++i)
		{
		code = setTipPartials(
						instance_handle,			// instance
						i,							// bufferIndex
						&partial[i][0]);			// inPartials
		if (code != 0)
			abort("setPartials encountered a problem");
		}
        
    double rates[1] = { 1.0 };
    setCategoryRates(instance_handle, rates);

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

	code = setEigenDecomposition(
			   instance_handle,					// instance
			   0,								// eigenIndex,
			   (const double *)evec,			// inEigenVectors,
			   (const double *)ivec,			// inInverseEigenVectors,
			   eval);							// inEigenValues

	if (code != 0)
		abort("setEigenDecomposition encountered a problem");
	}

/*-----------------------------------------------------------------------------
|	Calculates the log likelihood by calling the beagle functions
|	updateTransitionMatrices, updatePartials and calculateEdgeLogLikelihoods.
*/
double FourTaxonExample::calcLnL()
	{
	int code = updateTransitionMatrices(
			instance_handle,				// instance,
			0,								// eigenIndex,
			&transition_matrix_index[0],	// probabilityIndices,
			NULL, 							// firstDerivativeIndices,
			NULL,							// secondDervativeIndices,
			&brlens[0],						// edgeLengths,
			5);								// count

	if (code != 0)
		abort("updateTransitionMatrices encountered a problem");
        
    int cumulativeScalingFactorIndex = 0;
        
    resetScaleFactors(instance_handle,
                      cumulativeScalingFactorIndex);

	code = updatePartials(
		   &instance_handle,	// instance
		   1,					// instanceCount
		   &operations[0],		// operations
		   2,					// operationCount
		   cumulativeScalingFactorIndex); // cumulative scale index

	if (code != 0)
		abort("updatePartials encountered a problem");

	int parentBufferIndex = 4;
	int childBufferIndex  = 5;
	int transitionMatrixIndex  = 4;
	double relativeRateProb  = 1.0;
        
	double stateFreqs[4] = { 0.25, 0.25, 0.25, 0.25 };

	std::vector<double> lnL(nsites);
        
	code = calculateEdgeLogLikelihoods(
		 instance_handle,					// instance,
		 &parentBufferIndex,				// parentBufferIndices
		 &childBufferIndex,					// childBufferIndices
		 &transitionMatrixIndex,			// probabilityIndices
		 NULL,								// firstDerivativeIndices
		 NULL,								// secondDerivativeIndices
		 (const double*)&relativeRateProb,	// weights
		 (const double*)stateFreqs,		// stateFrequencies,
         &cumulativeScalingFactorIndex,
		 1,									// count
		 &lnL[0],								// outLogLikelihoods,
		 NULL,								// outFirstDerivatives,
		 NULL);								// outSecondDerivatives

	if (code != 0)
		abort("calculateEdgeLogLikelihoods encountered a problem");

	return std::accumulate(lnL.begin(), lnL.end(), 0.0);
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
	double log_like_before = calcLnL();
	brlens[brlen_index] = x;
	double log_like_after = calcLnL();

	double log_accept_ratio = log_prior_after + log_like_after - log_prior_before - log_like_before;
	double u = log(uniform());
	if (u > log_accept_ratio)
		{
		// proposed new branch length rejected, restore original branch length
		brlens[brlen_index] = x0;
		}
	}

/*-----------------------------------------------------------------------------
|	Reads in the data file (which must be named fourtaxon.dat and must contain
|	data for four taxa), then calls calcLnL `n' times before calling finalize
|	to inform the beagle library that it is ok to destroy all allocated memory.
*/
void FourTaxonExample::run()
	{
	::rnseed = seed;
	readData();
	initBeagleLib();
	writeData();

	std::cout.setf(std::ios::showpoint);
	std::cout.setf(std::ios::floatfield, std::ios::fixed);

	std::cout << std::setw(12) << "iter" << std::setw(24) << "log-likelihood" << std::setw(24) << "tree length" << std::endl;
	for (unsigned rep = 1; rep <= niters; ++rep)
		{
		for (unsigned b = 0; b < 5; ++b)
			updateBrlen(b);
		std::cout << std::setw(12) << rep;
		std::cout << std::setw(24) << std::setprecision(5) << calcLnL();
		std::cout << std::setw(24) << std::setprecision(5) << std::accumulate(brlens.begin(), brlens.end(), 0.0);
		std::cout << std::endl;
		}

	int code = finalize(
		instance_handle);		// instance

	if (code != 0)
		abort("finalize encountered a problem");
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
	outf << "  tree starting = [&U](alga_D86836:0.01, fern_D14882:0.02, (hops_AF206777:0.03, corn_Z11973:0.04):0.05);\n";
	outf << "end;\n\n";
	outf << "begin paup;\n";
	outf << "  lset nst=1 basefreq=equal;\n";
	outf << "  lscores 1 / userbrlen;\n";
	outf << "end;\n";
	outf.close();
	}

/*-----------------------------------------------------------------------------
|	Reads command line arguments and interprets them as follows:
|>
|	fourtaxon [<niters> [<data_file_name>]]
|>
|	If niters is not specified, the default value 1 million is used.
|	If data_file_name	is not specified, the default value "fourtaxon.dat" is
|	used. If data_file_name is specified, the file should have the same format
|	as the file "fourtaxon.dat" and should only contain sequences for 4 taxa
|	(although the sequences can be of arbitrary length).
*/
void FourTaxonExample::interpretCommandLineParameters(
  int argc, 		/**< is the number of command line arguments */
  char* argv[])		/**< is the array of command line arguments */
	{
	// see if the user specified the number of MCMC iterations on the command line
	// and, if so, replace the default value of niters
	niters = 4;
	if (argc > 1)
		niters = (unsigned)atoi(argv[1]);
	if (niters < 1)
		abort("invalid number of iterations supplied on the command line");

	// see if the user specified a data file name on the command line
	// and, if so, replace the default value of data_file_name
	data_file_name = "fourtaxon.dat";
	if (argc > 2)
		data_file_name = std::string(argv[2]);
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
	}

