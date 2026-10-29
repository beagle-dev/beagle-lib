#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <libhmsbeagle/BeagleImpl.h>
#include <cmath>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <sstream>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Map<MatrixXd> MapType;
typedef Eigen::SparseMatrix<double> SpMatrix;
typedef Eigen::Triplet<double> Triplet;
//#define JC

#ifdef _WIN32
#include <vector>
#endif

#include "libhmsbeagle/beagle.h"

char *human = (char*)"GAGTC";
char *chimp = (char*)"GAGGT";
char *gorilla = (char*)"AAATC";

//char *human = (char*)"G";
//char *chimp = (char*)"G";
//char *gorilla = (char*)"A";


//char *human = (char*)"GAGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGGAGCTTAAACCCCCTTATTTCTACTAGGACTATGAGAATCGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTATACCCTTCCCGTACTAAGAAATTTAGGTTAAATACAGACCAAGAGCCTTCAAAGCCCTCAGTAAGTTG-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGACCAATGGGACTTAAACCCACAAACACTTAGTTAACAGCTAAGCACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCGGAGCTTGGTAAAAAGAGGCCTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGGCCTCCATGACTTTTTCAAAAGGTATTAGAAAAACCATTTCATAACTTTGTCAAAGTTAAATTATAGGCT-AAATCCTATATATCTTA-CACTGTAAAGCTAACTTAGCATTAACCTTTTAAGTTAAAGATTAAGAGAACCAACACCTCTTTACAGTGA";
//char *chimp = (char*)"GGGAAATATGTCTGATAAAAGAATTACTTTGATAGAGTAAATAATAGGAGTTCAAATCCCCTTATTTCTACTAGGACTATAAGAATCGAACTCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTACACCCTTCCCGTACTAAGAAATTTAGGTTAAGCACAGACCAAGAGCCTTCAAAGCCCTCAGCAAGTTA-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATTAATGGGACTTAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCAGAGCTTGGTAAAAAGAGGCTTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCTAAAGCTGGTTTCAAGCCAACCCCATGACCTCCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAAGTTAAATTACAGGTT-AACCCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCATTAACCTTTTAAGTTAAAGATTAAGAGGACCGACACCTCTTTACAGTGA";
//char *gorilla = (char*)"AGAAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGAGGTTTAAACCCCCTTATTTCTACTAGGACTATGAGAATTGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTGTCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTCACATCCTTCCCGTACTAAGAAATTTAGGTTAAACATAGACCAAGAGCCTTCAAAGCCCTTAGTAAGTTA-CAACACTTAATTTCTGTAAGGACTGCAAAACCCTACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATCAATGGGACTCAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAGTCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAT-TCACCTCGGAGCTTGGTAAAAAGAGGCCCAGCCTCTGTCTTTAGATTTACAGTCCAATGCCTTA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGACCTTCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAGGTTAAATTACGGGTT-AAACCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCGTTAACCTTTTAAGTTAAAGATTAAGAGTATCGGCACCTCTTTGCAGTGA";

int* getStates(char *sequence, int repeats) {
    int n = strlen(sequence);
    int *states = (int*) malloc(sizeof(int) * n * repeats);

    int k = 0;
    for (int r = 0; r < repeats; ++r) {
        for (int i = 0; i < n; i++) {
            switch (sequence[i]) {
                case 'A':
                    states[k++] = 0;
                    break;
                case 'C':
                    states[k++] = 1;
                    break;
                case 'G':
                    states[k++] = 2;
                    break;
                case 'T':
                    states[k++] = 3;
                    break;
                default:
                    states[k++] = 4;
                    break;
            }
        }
    }
    return states;
}

double* getPartials(char *sequence, int repeats) {
    int n = strlen(sequence);
    double *partials = (double*)malloc(sizeof(double) * n * 4);

    int k = 0;
    for (int i = 0; i < n; i++) {
        switch (sequence[i]) {
            case 'A':
                partials[k++] = 1;
                partials[k++] = 0;
                partials[k++] = 0;
                partials[k++] = 0;
                break;
            case 'C':
                partials[k++] = 0;
                partials[k++] = 1;
                partials[k++] = 0;
                partials[k++] = 0;
                break;
            case 'G':
                partials[k++] = 0;
                partials[k++] = 0;
                partials[k++] = 1;
                partials[k++] = 0;
                break;
            case 'T':
                partials[k++] = 0;
                partials[k++] = 0;
                partials[k++] = 0;
                partials[k++] = 1;
                break;
            default:
                partials[k++] = 1;
                partials[k++] = 1;
                partials[k++] = 1;
                partials[k++] = 1;
                break;
        }
    }
    return partials;
}


std::ostream& operator<<(std::ostream& o, BeagleFlags inFlags)
{
    if (inFlags & BEAGLE_FLAG_PRECISION_SINGLE)   o<<" PRECISION_SINGLE";
    if (inFlags & BEAGLE_FLAG_PRECISION_DOUBLE)   o<<" PRECISION_DOUBLE";
    if (inFlags & BEAGLE_FLAG_COMPUTATION_SYNCH)  o<<" COMPUTATION_SYNCH";
    if (inFlags & BEAGLE_FLAG_COMPUTATION_ASYNCH) o<<" COMPUTATION_ASYNCH";
    if (inFlags & BEAGLE_FLAG_EIGEN_REAL)         o<<" EIGEN_REAL";
    if (inFlags & BEAGLE_FLAG_EIGEN_COMPLEX)      o<<" EIGEN_COMPLEX";

    if (inFlags & BEAGLE_FLAG_SCALING_MANUAL)     o<<" SCALING_MANUAL";
    if (inFlags & BEAGLE_FLAG_SCALING_AUTO)       o<<" SCALING_AUTO";
    if (inFlags & BEAGLE_FLAG_SCALING_ALWAYS)     o<<" SCALING_ALWAYS";
    if (inFlags & BEAGLE_FLAG_SCALING_DYNAMIC)    o<<" SCALING_DYNAMIC";
    if (inFlags & BEAGLE_FLAG_SCALERS_RAW)        o<<" SCALERS_RAW";
    if (inFlags & BEAGLE_FLAG_SCALERS_LOG)        o<<" SCALERS_LOG";

    if (inFlags & BEAGLE_FLAG_INVEVEC_STANDARD)   o<<" INVENV_STANDARD";
    if (inFlags & BEAGLE_FLAG_INVEVEC_TRANSPOSED) o<<" INVENV_STANDARD";

    if (inFlags & BEAGLE_FLAG_VECTOR_SSE)         o<<" VECTOR_SSE";
    if (inFlags & BEAGLE_FLAG_VECTOR_AVX)         o<<" VECTOR_AVX";
    if (inFlags & BEAGLE_FLAG_VECTOR_NONE)        o<<" VECTOR_NONE";

    if (inFlags & BEAGLE_FLAG_THREADING_CPP)      o<<" THREADING_CPP";
    if (inFlags & BEAGLE_FLAG_THREADING_OPENMP)   o<<" THREADING_OPENMP";
    if (inFlags & BEAGLE_FLAG_THREADING_NONE)     o<<" THREADING_NONE";

    if (inFlags & BEAGLE_FLAG_PROCESSOR_CPU)      o<<" PROCESSOR_CPU";
    if (inFlags & BEAGLE_FLAG_PROCESSOR_GPU)      o<<" PROCESSOR_GPU";
    if (inFlags & BEAGLE_FLAG_PROCESSOR_FPGA)     o<<" PROCESSOR_FPGA";
    if (inFlags & BEAGLE_FLAG_PROCESSOR_CELL)     o<<" PROCESSOR_CELL";
    if (inFlags & BEAGLE_FLAG_PROCESSOR_PHI)      o<<" PROCESSOR_PHI";
    if (inFlags & BEAGLE_FLAG_PROCESSOR_OTHER)    o<<" PROCESSOR_OTHER";

    if (inFlags & BEAGLE_FLAG_FRAMEWORK_CUDA)     o<<" FRAMEWORK_CUDA";
    if (inFlags & BEAGLE_FLAG_FRAMEWORK_OPENCL)   o<<" FRAMEWORK_OPENCL";
    if (inFlags & BEAGLE_FLAG_FRAMEWORK_CPU)      o<<" FRAMEWORK_CPU";

    if (inFlags & BEAGLE_FLAG_PARALLELOPS_STREAMS) o<<" PARALLEL_OPS_STREAMS";
    if (inFlags & BEAGLE_FLAG_PARALLELOPS_GRID) o<<" PARALLEL_OPS_GRID";

    if (inFlags & BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL) o<<" PREORDER_TRANSPOSE_MANUAL";
    if (inFlags & BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO) o<<" PREORDER_TRANSPOSE_MANUAL";

    if (inFlags & BEAGLE_FLAG_COMPUTATION_ACTION) o<<" COMPUTATION_ACTION";

    return o;
}

std::string showFlags(BeagleFlags flags)
{
    std::ostringstream o;
    o<<flags;
    return o.str();
}

bool has_only_digits(const std::string& s)
{
    return s.find_first_not_of( "0123456789" ) == std::string::npos;
}

int main( int argc, const char* argv[] )
{
    // print resource list
    BeagleResourceList* rList;
    rList = beagleGetResourceList();
    std::cout<<"Available resources:\n";
    for (int i = 0; i < rList->length; i++) {
        std::cout<<"\tResource "<<i<<":\n\t\tName : "<<rList->list[i].name<<"\n";
	std::cout<<"\t\tDesc : "<<rList->list[i].description<<"\n";
	std::cout<<"\t\tFlags:"<<BeagleFlags(rList->list[i].supportFlags )<<"\n";
    }
    std::cout<<"\n";

    bool doJC = true;

    bool singlePrecision = false;
    bool useSSE = true;

    // is nucleotides...
    int stateCount = 4;

    int nRepeats = 1;

    // get the number of site patterns
    int nPatterns = strlen(human) * nRepeats;

    // change # rate category to 2
//    int rateCategoryCount = 4;
    int rateCategoryCount = 2;

    bool scaling = true;
    for(int i=1;i<argc;i++)
	if (!strcmp(argv[i],"--noscaling"))
	    scaling = false;
    int scaleCount = (scaling ? 7 : 0);

    int whichDevice = -1;
    bool useGpu = false;
    for(int i=1;i<argc;i++)
    {
	if (!strcmp(argv[i],"--gpu"))
	{
	    useGpu = true;
	    if (i+1<argc and has_only_digits(argv[i+1]))
		whichDevice = std::stoi(argv[i+1]);
	}
    }

    bool useThreading = false;
    for(int i=1;i<argc;i++)
	if (!strcmp(argv[i],"--threading"))
	    useThreading = true;

    for(int i=1;i<argc;i++)
	if (!strcmp(argv[i],"--help"))
	{
	    std::cerr<<"Flag: --gpu\n";
	    std::cerr<<"Flag: --gpu <integer>\n";
	    std::cerr<<"Flag: --threading\n";
	    std::cerr<<"Flag: --noscaling\n";
	    std::exit(1);
	}

    bool useTipStates = false;

    if (useGpu) useSSE = false;

    BeagleInstanceDetails instDetails;

//    long requirementFlags = BEAGLE_FLAG_EIGEN_REAL;
    long long requirementFlags = 0;
    long long preferenceFlags = BEAGLE_FLAG_COMPUTATION_ACTION;

    if (useGpu) {
        requirementFlags |= BEAGLE_FLAG_PROCESSOR_GPU;
    } else {
        requirementFlags |= BEAGLE_FLAG_PROCESSOR_CPU;
	if (useSSE) {
	  requirementFlags |= BEAGLE_FLAG_VECTOR_SSE;
	} else {
	  requirementFlags |= BEAGLE_FLAG_VECTOR_NONE;
	}
    }

    if (singlePrecision) {
        preferenceFlags |= BEAGLE_FLAG_PRECISION_SINGLE;
    } else {
        preferenceFlags |= BEAGLE_FLAG_PRECISION_DOUBLE;
    }

    if (useThreading)
	preferenceFlags |= BEAGLE_FLAG_THREADING_CPP;

    // create an instance of the BEAGLE library
    int instance = beagleCreateInstance(
            3,				/**< Number of tip data elements (input) */
            20,	            /**< Number of partials buffers to create (input) */
            useTipStates ? 3 : 0,		        /**< Number of compact state representation buffers to create (input) */
            stateCount,		/**< Number of states in the continuous-time Markov chain (input) */
            nPatterns,		/**< Number of site patterns to be handled by the instance (input) */
            1,		        /**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
            6 * 2,		    /**< Number of rate matrix buffers (input) */
            rateCategoryCount,/**< Number of rate categories (input) */
            scaleCount,       /**< Number of scaling buffers */
            whichDevice >= 0 ? &whichDevice : NULL, /**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
            whichDevice >= 0 ? 1 : 0,			    /**< Length of resourceList list (input) */
            preferenceFlags,
            requirementFlags, /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
            &instDetails);
    if (instance < 0) {
        std::cerr<<"Failed to obtain BEAGLE instance\n\n";
        exit(1);
    }


    int rNumber = instDetails.resourceNumber;
    std::cout<<"Using resource "<<rNumber<<":\n";
    std::cout<<"\tRsrc Name : "<<instDetails.resourceName<<"\n";
    std::cout<<"\tImpl Name : "<<instDetails.implName<<"\n";
    std::cout<<"\tImpl Desc : "<<instDetails.implDescription<<"\n";
    std::cout<<"\n";

    if (useTipStates) {
        // set the sequences for each tip using state likelihood arrays
        int *humanStates = getStates(human, nRepeats);
        int *chimpStates = getStates(chimp, nRepeats);
        int *gorillaStates = getStates(gorilla, nRepeats);

        beagleSetTipStates(instance, 0, humanStates);
        beagleSetTipStates(instance, 1, chimpStates);
        beagleSetTipStates(instance, 2, gorillaStates);

        free(humanStates);
        free(chimpStates);
        free(gorillaStates);

    } else {
        // set the sequences for each tip using partial likelihood arrays
        double *humanPartials = getPartials(human, nRepeats);
        double *chimpPartials = getPartials(chimp, nRepeats);
        double *gorillaPartials = getPartials(gorilla, nRepeats);

        beagleSetTipPartials(instance, 0, humanPartials);
        beagleSetTipPartials(instance, 1, chimpPartials);
        beagleSetTipPartials(instance, 2, gorillaPartials);

        free(humanPartials);
        free(chimpPartials);
        free(gorillaPartials);
    }

#ifdef _WIN32
    std::vector<double> rates(rateCategoryCount);
#else
    double rates[rateCategoryCount];
#endif
//    for (int i = 0; i < rateCategoryCount; i++) {
//        rates[i] = 1.0;
////        rates[i] = 3.0 * (i + 1) / (2 * rateCategoryCount + 1);
//    }

    rates[0] = 0.14251623900062188;
    rates[1] = 1.857483760999378;

//    rates[0] = 1.0;

    beagleSetCategoryRates(instance, &rates[0]);

    double* patternWeights = (double*) malloc(sizeof(double) * nPatterns);

    for (int i = 0; i < nPatterns; i++) {
        patternWeights[i] = 1.0;
    }

    beagleSetPatternWeights(instance, patternWeights);

    // create base frequency array
    double freqs[4] = { 0.1, 0.3, 0.2, 0.4 };
//    double freqs[4] = { 0.25, 0.25, 0.25, 0.25 };

    beagleSetStateFrequencies(instance, 0, freqs);

    // create an array containing site category weights
#ifdef _WIN32
    std::vector<double> weights(rateCategoryCount);
#else
    double weights[rateCategoryCount];
#endif
    for (int i = 0; i < rateCategoryCount; i++) {
        weights[i] = 1.0/rateCategoryCount;
//        weights[i] = 2.0 * double(i + 1)/ double(rateCategoryCount * (rateCategoryCount + 1));
    }

    beagleSetCategoryWeights(instance, 0, &weights[0]);


    ///eigen decomposition of the HKY85 model
//    double evec[4 * 4] = {
//            0.9819805,  0.040022305,  0.04454354,  -0.5,
//            -0.1091089, -0.002488732, 0.81606029,  -0.5,
//            -0.1091089, -0.896939683, -0.11849713, -0.5,
//            -0.1091089,  0.440330814, -0.56393254, -0.5
//    };
//
//    double ivec[4 * 4] = {
//            0.9165151, -0.3533241, -0.1573578, -0.4058332,
//            0.0,  0.2702596, -0.8372848,  0.5670252,
//            0.0,  0.8113638, -0.2686725, -0.5426913,
//            -0.2, -0.6, -0.4, -0.8
//    };
//
//    ///array of real parts + array of imaginary parts
//    double eval[8] = { -1.42857105618099456, -1.42857095607719153, -1.42857087221423851, 0.0,
//                       0.0, 0.0, 0.0, 0.0 };

    ///Q^T matrix
//    double QT[4 * 4] = {
//            -1.2857138,  0.1428570,  0.1428570,  0.1428570,
//            0.4285712, -0.9999997,  0.4285714,  0.4285713,
//            0.2857142,  0.2857143, -1.1428568,  0.2857142,
//            0.5714284,  0.5714284,  0.5714284, -0.8571426
//    };

    double Q[4 * 4 * 2] = {
            -1.285714,  0.4285712,  0.2857142,  0.5714284,
            0.142857, -0.9999997,  0.2857143,  0.5714284,
            0.142857,  0.4285714, -1.1428568,  0.5714284,
            0.142857,  0.4285713,  0.2857142, -0.8571426,
            -1.285714,  0.4285712,  0.2857142,  0.5714284,
            0.142857, -0.9999997,  0.2857143,  0.5714284,
            0.142857,  0.4285714, -1.1428568,  0.5714284,
            0.142857,  0.4285713,  0.2857142, -0.8571426
    };

    double Q2[4 * 4 * 2] = {
            1.8367333, -0.6122443, -0.4081629, -0.8163261,
            -0.2040814,  1.4285705, -0.4081632, -0.8163259,
            -0.2040814, -0.6122447,  1.6326522, -0.8163261,
            -0.2040814, -0.6122446, -0.4081630,  1.2244890,
            1.8367333, -0.6122443, -0.4081629, -0.8163261,
            -0.2040814,  1.4285705, -0.4081632, -0.8163259,
            -0.2040814, -0.6122447,  1.6326522, -0.8163261,
            -0.2040814, -0.6122446, -0.4081630,  1.2244890
    };

    std::vector<double> scaledQ(4 * 4 * 2);
    std::vector<double> scaledQ2(4 * 4 * 2);
//    std::vector<double> scaledQT(4 * 4 * 2);

    double evec[16 * 2] = {
            0, 0,
            0, 1,
            0, 2,
            0, 3,
            1, 0,
            1, 1,
            1, 2,
            1, 3,
            2, 0,
            2, 1,
            2, 2,
            2, 3,
            3, 0,
            3, 1,
            3, 2,
            3, 3,
    };

    int rowIndices[16] = {
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
    };
    int colIndices[16] = {
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
    };

    double ivec[1] = {16};

    ///array of real parts + array of imaginary parts
    double eval[16] = { -1.285714,  0.4285712,  0.2857142,  0.5714284,
                        0.142857, -0.9999997,  0.2857143,  0.5714284,
                        0.142857,  0.4285714, -1.1428568,  0.5714284,
                        0.142857,  0.4285713,  0.2857142, -0.8571426 };

    for (int rate = 0; rate < rateCategoryCount; ++rate) {
        for (int entry = 0; entry < stateCount * stateCount; ++entry) {
            scaledQ[entry + rate * stateCount * stateCount] = Q[entry + rate * stateCount * stateCount] * rates[rate];
            scaledQ2[entry + rate * stateCount * stateCount] = Q2[entry + rate * stateCount * stateCount] * rates[rate] * rates[rate];
        }
    }

//    SpMatrix* mPtr = (SpMatrix *) malloc(sizeof(SpMatrix) * rateCategoryCount);
    SpMatrix * mPtr = new SpMatrix[rateCategoryCount];
    for (int i = 0; i < rateCategoryCount; i++) {
//        mPtr[i] = (SpMatrix *) malloc(sizeof(SpMatrix));
        SpMatrix matrix(4, 4);
        mPtr[i] = matrix;
        std::cout<<"mPtr: "<<mPtr[i]<<std::endl;
    }
    const int numNonZeros = 16;
    std::vector<Triplet> tripletList;
    for (int i = 0; i < numNonZeros; i++) {
        tripletList.push_back(Triplet((int) evec[2 * i], (int) evec[2 * i + 1], eval[i]));
//        mPtr[0] -> insert((int) evec[2 * i], (int) evec[2 * i + 1]) = eval[i];
    }
    SpMatrix matrix(4, 4);
    matrix.setFromTriplets(tripletList.begin(), tripletList.end());
//    mPtr[0]->setFromTriplets(tripletList.begin(), tripletList.end());
//    (*mPtr[0]).setFromTriplets(tripletList.begin(), tripletList.end());
//    mPtr[0].setFromTriplets(tripletList.begin(), tripletList.end());
    std::cout<<mPtr[0]<<std::endl;


    // set the Eigen decomposition
//    beagleSetEigenDecomposition(instance, 0, evec, ivec, eval);
    beagleSetSparseMatrix(instance, 0, rowIndices, colIndices, eval, 16);

    // a list of indices and edge lengths
    int nodeIndices[4] = { 0, 1, 2, 3 };
//    double edgeLengths[4] = { 0.6, 0.6, 1.3, 0.7};
    double edgeLengths[4] = { 0.6, 0.6, 1.3, 0.7};

    // tell BEAGLE to populate the transition matrices for the above edge lengths
    beagleUpdateTransitionMatrices(instance,     // instance
                                   0,             // eigenIndex
                                   nodeIndices,   // probabilityIndices
                                   NULL,          // firstDerivativeIndices
                                   NULL,          // secondDervativeIndices
                                   edgeLengths,   // edgeLengths
                                   4);            // count

    // create a list of partial likelihood update operations
    // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
    BeagleOperation operations[2] = {
            3, (scaling ? 0 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 0, 0, 1, 1,
            4, (scaling ? 1 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 2, 2, 3, 3
    };

    int rootIndex = 4;


    double * seeprePartials  = (double*) malloc(sizeof(double) * stateCount * nPatterns * rateCategoryCount);

    beagleGetPartials(instance, 0, BEAGLE_OP_NONE, seeprePartials);
    MapType * testMaps = (MapType *) malloc(sizeof(MapType) * rateCategoryCount);
    for (int i = 0; i < rateCategoryCount; i++) {
        new (& testMaps[i]) MapType(seeprePartials + i * nPatterns * stateCount, stateCount, nPatterns);
//        MapType mappedPartial(seeprePartials + i * nPatterns * stateCount, stateCount, nPatterns);
//        mappedPartial(0, 2) = 1.0;
//        testMaps[i] = mappedPartial;
    }
    testMaps[0](0, 2) = 1.0;

    int l = 0;
    for(int s = 0; s < rateCategoryCount; s++){
        std::cout<<"See partial"<< 0 <<" rate category"<< s+1<< ": \n";
        for(int k = 0; k<nPatterns; k++){
            for(int j=0; j < stateCount; j++){
                std::cout<<seeprePartials[l++]<<", ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }


    // update the partials
    beagleUpdatePartials(instance,      // instance
                         operations,     // eigenIndex
                         2,              // operationCount
                         BEAGLE_OP_NONE);          // cumulative scaling index



    beagleGetPartials(instance, 3, BEAGLE_OP_NONE, seeprePartials);

    l = 0;
    for(int s = 0; s < rateCategoryCount; s++){
        std::cout<<"See partial"<< 3 <<" rate category"<< s+1<< ": \n";
        for(int k = 0; k<nPatterns; k++){
            for(int j=0; j < stateCount; j++){
                std::cout<<seeprePartials[l++]<<", ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
    beagleGetPartials(instance, 4, BEAGLE_OP_NONE, seeprePartials);

    l = 0;
    for(int s = 0; s < rateCategoryCount; s++){
        std::cout<<"See partial"<< 4 <<" rate category"<< s+1<< ": \n";
        for(int k = 0; k<nPatterns; k++){
            for(int j=0; j < stateCount; j++){
                std::cout<<seeprePartials[l++]<<", ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }



    ///XJ: I decided to store the pre-order partials vector in reverse order as those of post-orders
    ///This means that the two indices to the partials of root nodes are adjacent.
    ///For any node, the indices of the two partials sum to 2*(partialsBufferCount + compactBufferCount) - 1


    int categoryWeightsIndex = 0;
    int stateFrequencyIndex = 0;
    int transpose = (stateCount == 4 || !useGpu) ? 0 : 6;
    int rootPreIndex = 5;

    double *patternLogLik = (double*)malloc(sizeof(double) * nPatterns);

    int cumulativeScalingIndex = (scaling ? 2 : BEAGLE_OP_NONE);

    if (scaling) {
        int scalingFactorsCount = 2;
        int scalingFactorsIndices[2] = {0, 1};

        beagleResetScaleFactors(instance,
                                cumulativeScalingIndex);

        beagleAccumulateScaleFactors(instance,
                                     scalingFactorsIndices,
                                     scalingFactorsCount,
                                     cumulativeScalingIndex);
    }

    double logL = 0.0;

    // calculate the site likelihoods at the root node
    beagleCalculateRootLogLikelihoods(instance,               // instance
                                      (const int *)&rootIndex,// bufferIndices
                                      &categoryWeightsIndex,                // weights
                                      &stateFrequencyIndex,                  // stateFrequencies
                                      &cumulativeScalingIndex,// cumulative scaling index
                                      1,                      // count
                                      &logL);         // outLogLikelihoods


    std::vector<double> siteLogLikelihoods(nPatterns);
    beagleGetSiteLogLikelihoods(instance, siteLogLikelihoods.data());

    std::cout << "site-log-like:";
    for (double logLike : siteLogLikelihoods) {
        std::cout << " " << logLike;
    }
    std::cout << std::endl;

//    beagleSetRootPrePartials(instance, // TODO Remove from API -- not necessary?
//                             (const int *) &rootPreIndex,               // bufferIndices
//                             &stateFrequencyIndex,                  // stateFrequencies
//                             1);                                    // count


    std::cout<<"logL = "<<logL<<" (R = -21.827146738233282)\n\n";

    double * seerootPartials = (double*) malloc(sizeof(double) * stateCount * nPatterns * rateCategoryCount);
    int offset = 0;
    for (int c = 0; c < rateCategoryCount; ++c) {
        for (int p = 0; p < nPatterns; ++p) {
            for (int s = 0; s < stateCount; ++s) {
                seerootPartials[offset++] = freqs[s];
            }
        }
    }
    beagleSetPartials(instance, rootPreIndex, seerootPartials);
    beagleGetPartials(instance, rootPreIndex, BEAGLE_OP_NONE, seeprePartials);

    std::cout<<"Pre-order Partial for root \n";

    l = 0;
    for(int s = 0; s < rateCategoryCount; s++){
        std::cout<<"  rate category"<< s+1<< ": \n";
        for(int k = 0; k<nPatterns; k++){
            for(int j=0; j < stateCount; j++){
                std::cout<<seeprePartials[l++]<<", ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
//    std::cout
    // create a list of partial likelihood update operations
    // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
    // destPartials point to the pre-order partials
    // partials1 = pre-order partials of the parent node
    // matrices1 = Ptr matrices of the current node (to the parent node)
    // partials2 = post-order partials of the sibling node
    // matrices2 = Ptr matrices of the sibling node (to the parent node)
    BeagleOperation pre_order_operations[4] = {
            6, (scaling ? 3 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 5, 3 + transpose, 2, 2,
            7, (scaling ? 4 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 5, 2 + transpose, 3, 3,
            8, (scaling ? 5 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 6, 1 + transpose, 0, 0,
            9, (scaling ? 6 : BEAGLE_OP_NONE), BEAGLE_OP_NONE, 6, 0 + transpose, 1, 1,
    };

    // update the pre-order partials
    beagleUpdatePrePartials(instance,
                            pre_order_operations,
                            4,
                            BEAGLE_OP_NONE);
    beagleGetPartials(instance, rootPreIndex, BEAGLE_OP_NONE, seeprePartials);

    std::cout<<"Pre-order Partial for root \n";

    l = 0;
    for(int s = 0; s < rateCategoryCount; s++){
        std::cout<<"  rate category"<< s+1<< ": \n";
        for(int k = 0; k<nPatterns; k++){
            for(int j=0; j < stateCount; j++){
                std::cout<<seeprePartials[l++]<<", ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
    int postBufferIndices[4] = {1, 0, 2, 3};
    int preBufferIndices[4] = {8, 9, 7, 6};
    int firstDervIndices[4] = {4, 4, 4, 4};
    int secondDervIndices[4] = {5, 5, 5, 5};
    int cumulativeScalingInices[4] = {6, 5, 4, 3};
    int categoryRatesIndex = categoryWeightsIndex;

    double* gradient = (double*) malloc(sizeof(double) * nPatterns * 4);

    for(int i = 0; i < 5; i++){

        int postBufferIndex = 4-i;
        int preBufferIndex = 5+i;

        beagleGetPartials(instance, preBufferIndex, BEAGLE_OP_NONE, seeprePartials);

        std::cout<<"Pre-order Partial for node "<< 4-i << ": \n";

        int l = 0;
        for(int s = 0; s < rateCategoryCount; s++){
            std::cout<<"  rate category"<< s+1<< ": \n";
            for(int k = 0; k<nPatterns; k++){
                for(int j=0; j < stateCount; j++){
                    std::cout<<seeprePartials[l++]<<", ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
    }

    beagleSetDifferentialMatrix(instance, 4, scaledQ.data());
    beagleSetDifferentialMatrix(instance, 5, scaledQ2.data());

    beagleSetSparseDifferentialMatrix(instance, 4, rowIndices, colIndices, eval, 16);


    std::vector<double> firstBuffer(nPatterns * 5 * 2); // Get both numerator and denominator
    std::vector<double> sumBuffer(5);
//    int cumulativeScalingIndices[4] = {BEAGLE_OP_NONE, BEAGLE_OP_NONE, BEAGLE_OP_NONE, BEAGLE_OP_NONE};

    beagleCalculateEdgeDerivatives(instance,
                                   postBufferIndices, preBufferIndices,
                                   firstDervIndices,
                                   &categoryWeightsIndex,
                                   4,
                                   firstBuffer.data(),
                                   sumBuffer.data(),
                                   NULL);

    std::cout << "check gradients  :";
    for (int i = 0; i < 4 * nPatterns; ++i) {
        std::cout << " " << firstBuffer[i];
    }
    std::cout << std::endl;



    free(patternWeights);

    free(patternLogLik);
    free(seeprePartials);

    beagleFinalizeInstance(instance);

#ifdef _WIN32
    std::cout << "\nPress ENTER to exit...\n";
    fflush( stdout);
    fflush( stderr);
    getchar();
#endif

}

//Gradient:
//-0.248521  -0.194621  -0.248521  0.36811
//-0.248521  -0.194621  -0.248521  0.114741
//0.221279  -0.171686  0.221279  -0.00658093
//0.22128  -0.171686  0.22128  -0.00658095
