#include "fourtaxon.hpp"
using namespace std;

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

/*-----------------------------------------------------------------------------
|	Constructor simply calls init().
*/
FourTaxonExample::FourTaxonExample()
	{
	init();
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
|	The init() function sets up the beagle library and initializes all data
|	members.
*/
void FourTaxonExample::init()
	{
	int code;
	
	taxon_name.resize(4);
	data.resize(4);
	partial.resize(4);
	
	// hard coded tree topology is (A,B,(C,D))
	// where taxon order is A, B, C, D in the data file
	// Assume nodes 0..3 are the tip node indices
	// Assume node 4 is ancestor of A,B (0,1)
	// Assume node 5 is ancestor of C,D (2,3)
	operations.push_back(4);	// destination (to be calculated)
	operations.push_back(0);	// left child partial index
	operations.push_back(0);	// left child transition matrix index
	operations.push_back(1);	// right child partial index
	operations.push_back(1);	// right child transition matrix index
	
	operations.push_back(5);	// destination (to be calculated)
	operations.push_back(2);	// left child partial index
	operations.push_back(2);	// left child transition matrix index
	operations.push_back(3);	// right child partial index
	operations.push_back(3);	// right child transition matrix index
	
	instance_handle = createInstance(
				4,			// tipCount
				7,			// partialsBufferCount
				0,			// compactBufferCount
				4, 			// stateCount
				nsites,		// patternCount
				1,			// eigenBufferCount
				5,			// matrixBufferCount,
				NULL,		// resourceList
				0,			// resourceCount
				0,			// preferenceFlags
				0			// requirementFlags		
				);
				
	transition_matrix_index.resize(5);
	transition_matrix_index.push_back(0);
	transition_matrix_index.push_back(1);
	transition_matrix_index.push_back(2);
	transition_matrix_index.push_back(3);
	transition_matrix_index.push_back(4);
	
	brlens.resize(5);
	brlens.push_back(0.01);
	brlens.push_back(0.02);
	brlens.push_back(0.03);
	brlens.push_back(0.04);
	brlens.push_back(0.05);
	
	for (unsigned i = 0; i < 4; ++i)
		{
		code = setPartials(
						instance_handle,			// instance
						i,							// bufferIndex
						&partial[i][0]);			// inPartials
		if (code != 0)
			abort("setPartials encountered a problem");
		}
		
	double evec[4][4] = {
		{ 1.0,  2.0,  0.0,  0.5},
		{ 1.0,  -2.0,  0.5,  0.0},
		{ 1.0,  2.0, 0.0,  -0.5},
		{ 1.0,  -2.0,  -0.5,  0.0}
	};
	
	double ivec[4][4] = {
		{ 0.25,  0.25,  0.25,  0.25},
		{ 0.125,  -0.125,  0.125,  -0.125},
		{ 0.0,  1.0,  0.0,  -1.0},
		{ 1.0,  0.0,  -1.0,  0.0}
	};
	
	double eval[4] = { 0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333 };
	
	double ** evecP = NewTwoDArray(4,4);
	double ** ivecP = NewTwoDArray(4,4);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			evecP[i][j] = evec[i][j];
			ivecP[i][j] = ivec[i][j];
		}
	}
		
	code = setEigenDecomposition(
			   instance_handle,					// instance
			   0,								// eigenIndex,
			   (const double **)evecP,			// inEigenVectors,
			   (const double **)ivecP,			// inInverseEigenVectors,
			   eval);							// inEigenValues
	
	DeleteTwoDArray(evecP);
	DeleteTwoDArray(ivecP);
			   
	if (code != 0)
		abort("setEigenDecomposition encountered a problem");
	}
	
/*-----------------------------------------------------------------------------
|	
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
		
	code = updatePartials(
		   &instance_handle,	// instance
		   1,					// instanceCount
		   &operations[0],		// operations				
		   2,					// operationCount
		   0);					// rescale
		   
	if (code != 0)
		abort("updatePartials encountered a problem");
		
	int parentBufferIndex = 4;
	int childBufferIndex  = 5;
	int transitionMatrixIndex  = 4;
	double relativeRateProb  = 1.0;
	double stateFreqs[] = {0.25, 0.25, 0.25, 0.25};
	double lnL = 0.0;
	
	code = calculateEdgeLogLikelihoods(
		 instance_handle,					// instance,
		 &parentBufferIndex,				// parentBufferIndices
		 &childBufferIndex,					// childBufferIndices		                   
		 &transitionMatrixIndex,			// probabilityIndices
		 NULL,								// firstDerivativeIndices
		 NULL,								// secondDerivativeIndices
		 (const double*)&relativeRateProb,	// weights
		 (const double**)&stateFreqs,		// stateFrequencies,
		 1,									// count
		 &lnL,								// outLogLikelihoods,
		 NULL,								// outFirstDerivatives,
		 NULL);								// outSecondDerivatives

	if (code != 0)
		abort("calculateEdgeLogLikelihoods encountered a problem");
		
	return lnL;
	}

/*-----------------------------------------------------------------------------
|	
*/
void FourTaxonExample::run()
	{
	readData("example_data.txt");
	//writeData("example_data.check.txt");

	unsigned nreps = 1000;
	for (unsigned rep = 0; rep < nreps; ++rep)
		{
		std::cerr << rep << ": lnL = " << calcLnL() << std::endl;
		}
		
	finalize(
		&instance_handle,		// instance
		1);						// instanceCount
	}

/*-----------------------------------------------------------------------------
|	
*/
void FourTaxonExample::readData(const std::string file_name)
	{
	std::string sequence;
	std::ifstream inf(file_name.c_str());
	inf >> ntaxa >> nsites;
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
|	
*/
void FourTaxonExample::writeData(const std::string file_name)
	{
	std::ofstream outf(file_name.c_str(), std::ios::out);
	outf << ntaxa << " " << nsites << std::endl;
	for (unsigned i = 0; i < ntaxa; ++i)
		{
		outf << taxon_name[i] << " ";
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
	outf.close();
	}

/*-----------------------------------------------------------------------------
|	
*/
int main(int argc, char* argv[])
	{
	FourTaxonExample().run();
	}
	