/*
 *  synthetictest.cpp
 *  Created by Aaron Darling on 14/06/2009.
 *  @author Aaron Darling
 *  @author Daniel Ayres
 *  Based on tinyTest.cpp by Andrew Rambaut.
 */
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <stack>
#include <queue>

#ifdef _WIN32
    #include <winsock.h>
    #include <string>
#else
    #include <sys/time.h>
#endif

#include "libhmsbeagle/beagle.h"
#include "linalg.h"

#define MAX_DIFF    0.01        //max discrepancy in scoring between reps
#define GT_RAND_MAX 0x7fffffff

#ifdef _WIN32
    //From January 1, 1601 (UTC). to January 1,1970
    #define FACTOR 0x19db1ded53e8000 

    int gettimeofday(struct timeval *tp,void * tz) {
        FILETIME f;
        ULARGE_INTEGER ifreq;
        LONGLONG res; 
        GetSystemTimeAsFileTime(&f);
        ifreq.HighPart = f.dwHighDateTime;
        ifreq.LowPart = f.dwLowDateTime;

        res = ifreq.QuadPart - FACTOR;
        tp->tv_sec = (long)((LONGLONG)res/10000000);
        tp->tv_usec =(long)(((LONGLONG)res%10000000)/10); // Micro Seconds

        return 0;
    }
#endif

double cpuTimeSetPartitions, cpuTimeUpdateTransitionMatrices, cpuTimeUpdatePartials, cpuTimeAccumulateScaleFactors, cpuTimeCalculateRootLogLikelihoods, cpuTimeTotal;

bool useStdlibRand;

static unsigned int rand_state = 1;

int gt_rand_r(unsigned int *seed)
{
    *seed = *seed * 1103515245 + 12345;
    return (*seed % ((unsigned int)GT_RAND_MAX + 1));
}

int gt_rand(void)
{
    if (!useStdlibRand) {
        return (gt_rand_r(&rand_state));
    } else {
        return rand();
    }
}

void gt_srand(unsigned int seed)
{
    if (!useStdlibRand) {
        rand_state = seed;
    } else {
        srand(seed);
    }
}

void abort(std::string msg) {
    std::cerr << msg << "\nAborting..." << std::endl;
    std::exit(1);
}

double* getRandomTipPartials( int nsites, int stateCount )
{
    double *partials = (double*) calloc(sizeof(double), nsites * stateCount); // 'malloc' was a bug
    for( int i=0; i<nsites*stateCount; i+=stateCount )
    {
        int s = gt_rand()%stateCount;
        // printf("%d ", s);
        partials[i+s]=1.0;
    }
    return partials;
}

int* getRandomTipStates( int nsites, int stateCount )
{
    int *states = (int*) calloc(sizeof(int), nsites); 
    for( int i=0; i<nsites; i++ )
    {
        int s = gt_rand()%stateCount;
        states[i]=s;
    }
    return states;
}

void printTiming(double timingValue,
                 int timePrecision,
                 bool printSpeedup,
                 double cpuTimingValue,
                 int speedupPrecision,
                 bool printPercent,
                 double totalTime,
                 int percentPrecision) {
    std::cout << std::setprecision(timePrecision) << timingValue << "s";
    if (printSpeedup) std::cout << " (" << std::setprecision(speedupPrecision) << cpuTimingValue/timingValue << "x CPU)";
    if (printPercent) std::cout << " (" << std::setw(3+percentPrecision) << std::setfill('0') << std::setprecision(percentPrecision) << (double)(timingValue/totalTime)*100 << "%)";
    std::cout << "\n";
}

double getTimeDiff(struct timeval t1,
                   struct timeval t2) {
    return ((t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec-t1.tv_usec)/1000000.0);
}

struct node
{
    int data;
    struct node* left;
    struct node* right;
    struct node* parent;
};

/* Given a binary tree, print its nodes in reverse level order */
void reverseLevelOrder(node* root, std::stack <node*> &S)
{
    std::queue <node*> Q;
    Q.push(root);
 
    // Do something like normal level order traversal order. Following are the
    // differences with normal level order traversal
    // 1) Instead of printing a node, we push the node to stack
    // 2) Right subtree is visited before left subtree
    while (Q.empty() == false)
    {
        /* Dequeue node and make it root */
        root = Q.front();
        Q.pop();

        if (root->left!=NULL) {
            S.push(root);
        }
 
        /* Enqueue right child */
        if (root->right)
            Q.push(root->right); // NOTE: RIGHT CHILD IS ENQUEUED BEFORE LEFT
 
        /* Enqueue left child */
        if (root->left)
            Q.push(root->left);
    }
 
}


/* Given a binary tree, count number of parallel launches */
int countLaunches(node* root)
{
    std::stack <node *> S;
    reverseLevelOrder(root, S);

    int opCount = S.size();

    int launchCount = 0;
	std::vector<int> gridStartOp(opCount);
	std::vector<int> operationsTmp(opCount);
	int parentMinIndex = 0;

    for(int op=0; op<opCount; op++){
        node* parent = S.top();
        S.pop();
        int parentIndex = parent->data;
        int child1Index = parent->left->data;
        int child2Index = parent->right->data;

        operationsTmp[op] = parentIndex;
        
        // printf("op %02d dest %02d c1 %02d c2 %02d\n",
        //        op, parentIndex, child1Index, child2Index);

        bool newLaunch = false;

        if (op == 0) {
            newLaunch = true;
        } else if (child1Index >= parentMinIndex || child2Index >= parentMinIndex) {
            for (int i=gridStartOp[launchCount-1]; i < op; i++) {
                int previousParentIndex = operationsTmp[i];
                if (child1Index == previousParentIndex || child2Index == previousParentIndex) {
                    newLaunch = true;
                    break;
                }
            }
        }

       if (newLaunch) {
            gridStartOp[launchCount] = op;
            parentMinIndex = parentIndex;

            launchCount++;
        } 

        if (parentIndex < parentMinIndex)
            parentMinIndex = parentIndex;
    }

    return launchCount;
}


node* createNewNode(int data)
{
    node* temp = new node;
    temp->data = data;
    temp->left = NULL;
    temp->right = NULL;
    temp->parent = NULL;
 
    return (temp);
}

void addChildren(node* newNode, node* originalNode, std::vector<node*> newNodes)
{
    if (originalNode->left != NULL) {
        newNode->left = createNewNode(originalNode->left->data);
        newNode->left->parent = newNode;
        newNodes.push_back(newNode->left);
        
        addChildren(newNode->left, originalNode->left, newNodes);

        newNode->right = createNewNode(originalNode->right->data);
        newNode->right->parent = newNode;
        newNodes.push_back(newNode->right);

        addChildren(newNode->right, originalNode->right, newNodes);
    }
}


void addParentChildren(node* newNode, node* originalNode, std::vector<node*> newNodes)
{

    if (originalNode->parent != NULL) {

        if (originalNode->left->data == newNode->parent->data) {
            newNode->left = createNewNode(originalNode->parent->data);
            newNode->left->parent = newNode;
            newNodes.push_back(newNode->left);

            addParentChildren(newNode->left, originalNode->parent, newNodes);

            newNode->right = createNewNode(originalNode->right->data);
            newNode->right->parent = newNode;
            newNodes.push_back(newNode->right);

            addChildren(newNode->right, originalNode->right, newNodes);
        } else {
            newNode->right = createNewNode(originalNode->parent->data);
            newNode->right->parent = newNode;
            newNodes.push_back(newNode->right);

            addParentChildren(newNode->right, originalNode->parent, newNodes);

            newNode->left = createNewNode(originalNode->left->data);
            newNode->left->parent = newNode;
            newNodes.push_back(newNode->left);

            addChildren(newNode->left, originalNode->left, newNodes);
        }

    } else { // original is root node

        if (newNode->parent->data == originalNode->left->data) {
            newNode->data = originalNode->right->data;
            addChildren(newNode, originalNode->right, newNodes);
        } else {
            newNode->data = originalNode->left->data;
            addChildren(newNode, originalNode->left, newNodes);
        }
    }
}

node* reroot(node* rerootNode, node* root, std::vector<node*> newNodes)
{
    struct node* newRoot = createNewNode(rerootNode->data);
    newNodes.push_back(newRoot);

        if (rerootNode->parent->left == rerootNode) {

            newRoot->left = createNewNode(rerootNode->data);
            newRoot->left->parent = newRoot;
            newNodes.push_back(newRoot->left);

            addChildren(newRoot->left, rerootNode, newNodes);

            newRoot->right = createNewNode(rerootNode->parent->data);
            newRoot->right->parent = newRoot;
            newNodes.push_back(newRoot->right);

            addParentChildren(newRoot->right, rerootNode->parent, newNodes);

        } else {

            newRoot->right = createNewNode(rerootNode->data);
            newRoot->right->parent = newRoot;
            newNodes.push_back(newRoot->right);

            addChildren(newRoot->right, rerootNode, newNodes);

            newRoot->left = createNewNode(rerootNode->parent->data);
            newRoot->left->parent = newRoot;
            newNodes.push_back(newRoot->left);

            addParentChildren(newRoot->left, rerootNode->parent, newNodes);

        }

        newRoot->data = root->data;

        return newRoot;
}



void printFlags(long inFlags) {
    if (inFlags & BEAGLE_FLAG_PROCESSOR_CPU)      fprintf(stdout, " PROCESSOR_CPU");
    if (inFlags & BEAGLE_FLAG_PROCESSOR_GPU)      fprintf(stdout, " PROCESSOR_GPU");
    if (inFlags & BEAGLE_FLAG_PROCESSOR_FPGA)     fprintf(stdout, " PROCESSOR_FPGA");
    if (inFlags & BEAGLE_FLAG_PROCESSOR_CELL)     fprintf(stdout, " PROCESSOR_CELL");
    if (inFlags & BEAGLE_FLAG_PRECISION_DOUBLE)   fprintf(stdout, " PRECISION_DOUBLE");
    if (inFlags & BEAGLE_FLAG_PRECISION_SINGLE)   fprintf(stdout, " PRECISION_SINGLE");
    if (inFlags & BEAGLE_FLAG_COMPUTATION_ASYNCH) fprintf(stdout, " COMPUTATION_ASYNCH");
    if (inFlags & BEAGLE_FLAG_COMPUTATION_SYNCH)  fprintf(stdout, " COMPUTATION_SYNCH");
    if (inFlags & BEAGLE_FLAG_EIGEN_REAL)         fprintf(stdout, " EIGEN_REAL");
    if (inFlags & BEAGLE_FLAG_EIGEN_COMPLEX)      fprintf(stdout, " EIGEN_COMPLEX");
    if (inFlags & BEAGLE_FLAG_SCALING_MANUAL)     fprintf(stdout, " SCALING_MANUAL");
    if (inFlags & BEAGLE_FLAG_SCALING_AUTO)       fprintf(stdout, " SCALING_AUTO");
    if (inFlags & BEAGLE_FLAG_SCALING_ALWAYS)     fprintf(stdout, " SCALING_ALWAYS");
    if (inFlags & BEAGLE_FLAG_SCALING_DYNAMIC)    fprintf(stdout, " SCALING_DYNAMIC");
    if (inFlags & BEAGLE_FLAG_SCALERS_RAW)        fprintf(stdout, " SCALERS_RAW");
    if (inFlags & BEAGLE_FLAG_SCALERS_LOG)        fprintf(stdout, " SCALERS_LOG");
    if (inFlags & BEAGLE_FLAG_VECTOR_NONE)        fprintf(stdout, " VECTOR_NONE");
    if (inFlags & BEAGLE_FLAG_VECTOR_SSE)         fprintf(stdout, " VECTOR_SSE");
    if (inFlags & BEAGLE_FLAG_VECTOR_AVX)         fprintf(stdout, " VECTOR_AVX");
    if (inFlags & BEAGLE_FLAG_THREADING_NONE)     fprintf(stdout, " THREADING_NONE");
    if (inFlags & BEAGLE_FLAG_THREADING_OPENMP)   fprintf(stdout, " THREADING_OPENMP");
    if (inFlags & BEAGLE_FLAG_THREADING_CPP)      fprintf(stdout, " THREADING_CPP");
    if (inFlags & BEAGLE_FLAG_FRAMEWORK_CPU)      fprintf(stdout, " FRAMEWORK_CPU");
    if (inFlags & BEAGLE_FLAG_FRAMEWORK_CUDA)     fprintf(stdout, " FRAMEWORK_CUDA");
    if (inFlags & BEAGLE_FLAG_FRAMEWORK_OPENCL)   fprintf(stdout, " FRAMEWORK_OPENCL");
}



void runBeagle(int resource, 
               int stateCount, 
               int ntaxa, 
               int nsites, 
               bool manualScaling, 
               bool autoScaling,
               bool dynamicScaling,
               int rateCategoryCount,
               int nreps,
               bool fullTiming,
               bool requireDoublePrecision,
               bool requireSSE,
               bool requireAVX,
               int compactTipCount,
               int randomSeed,
               int rescaleFrequency,
               bool unrooted,
               bool calcderivs,
               bool logscalers,
               int eigenCount,
               bool eigencomplex,
               bool ievectrans,
               bool setmatrix,
               bool opencl,
               int partitionCount,
               bool sitelikes,
               bool newDataPerRep,
               bool randomTree,
               bool rerootTrees,
               bool pectinate)
{
    
    int edgeCount = ntaxa*2-2;
    int internalCount = ntaxa-1;
    int partialCount = ((ntaxa+internalCount)-compactTipCount)*eigenCount;
    int scaleCount = ((manualScaling || dynamicScaling) ? ntaxa : 0);

    int modelCount = eigenCount * partitionCount;
    
    BeagleInstanceDetails instDetails;
    
    // create an instance of the BEAGLE library
    int instance = beagleCreateInstance(
                ntaxa,            /**< Number of tip data elements (input) */
                partialCount, /**< Number of partials buffers to create (input) */
                compactTipCount,    /**< Number of compact state representation buffers to create (input) */
                stateCount,       /**< Number of states in the continuous-time Markov chain (input) */
                nsites,           /**< Number of site patterns to be handled by the instance (input) */
                modelCount,               /**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
                (calcderivs ? (3*edgeCount*modelCount) : edgeCount*modelCount),/**< Number of rate matrix buffers (input) */
                rateCategoryCount,/**< Number of rate categories */
                scaleCount*eigenCount,          /**< scaling buffers */
                &resource,        /**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
                1,                /**< Length of resourceList list (input) */
                0,         /**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
                // BEAGLE_FLAG_PARALLELOPS_STREAMS |
                (opencl ? BEAGLE_FLAG_FRAMEWORK_OPENCL : 0) |
                (ievectrans ? BEAGLE_FLAG_INVEVEC_TRANSPOSED : BEAGLE_FLAG_INVEVEC_STANDARD) |
                (logscalers ? BEAGLE_FLAG_SCALERS_LOG : BEAGLE_FLAG_SCALERS_RAW) |
                (eigencomplex ? BEAGLE_FLAG_EIGEN_COMPLEX : BEAGLE_FLAG_EIGEN_REAL) |
                (dynamicScaling ? BEAGLE_FLAG_SCALING_DYNAMIC : 0) |
                (autoScaling ? BEAGLE_FLAG_SCALING_AUTO : 0) |
                (requireDoublePrecision ? BEAGLE_FLAG_PRECISION_DOUBLE : BEAGLE_FLAG_PRECISION_SINGLE) |
                (requireSSE ? BEAGLE_FLAG_VECTOR_SSE :
                          (requireAVX ? BEAGLE_FLAG_VECTOR_AVX : BEAGLE_FLAG_VECTOR_NONE)),   /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
                &instDetails);
    if (instance < 0) {
        fprintf(stderr, "Failed to obtain BEAGLE instance\n\n");
        return;
    }
        
    int rNumber = instDetails.resourceNumber;
    fprintf(stdout, "Using resource %i:\n", rNumber);
    fprintf(stdout, "\tRsrc Name : %s\n",instDetails.resourceName);
    fprintf(stdout, "\tImpl Name : %s\n", instDetails.implName);    
    fprintf(stdout, "\tFlags:");
    printFlags(instDetails.flags);
    fprintf(stdout, "\n\n");
    

    if (!(instDetails.flags & BEAGLE_FLAG_SCALING_AUTO))
        autoScaling = false;
    
    // set the sequences for each tip using partial likelihood arrays
    gt_srand(randomSeed);   // fix the random seed...
    for(int i=0; i<ntaxa; i++)
    {
        if (compactTipCount == 0 || (i >= (compactTipCount-1) && i != (ntaxa-1))) {
            double* tmpPartials = getRandomTipPartials(nsites, stateCount);
            beagleSetTipPartials(instance, i, tmpPartials);
            free(tmpPartials);
        } else {
            int* tmpStates = getRandomTipStates(nsites, stateCount);
            beagleSetTipStates(instance, i, tmpStates);
            free(tmpStates);                
        }
    }

#ifdef _WIN32
    std::vector<double> rates(rateCategoryCount);
#else
    double rates[rateCategoryCount];
#endif
    
    for (int i = 0; i < rateCategoryCount; i++) {
        rates[i] = gt_rand() / (double) GT_RAND_MAX;
    }
    
    if (partitionCount > 1) {
        for (int i=0; i < partitionCount; i++) {
            beagleSetCategoryRatesWithIndex(instance, i, &rates[0]);
        }
    } else {
        beagleSetCategoryRates(instance, &rates[0]);
    }

    
    double* patternWeights = (double*) malloc(sizeof(double) * nsites);
    
    for (int i = 0; i < nsites; i++) {
        patternWeights[i] = gt_rand() / (double) GT_RAND_MAX;
    }    

    beagleSetPatternWeights(instance, patternWeights);
    
    // free(patternWeights);
    

    int* patternPartitions;
    double* partitionLogLs;
    double* partitionD1;
    double* partitionD2;
    if (partitionCount > 1) {
        partitionLogLs = (double*) malloc(sizeof(double) * partitionCount);
        partitionD1 = (double*) malloc(sizeof(double) * partitionCount);
        partitionD2 = (double*) malloc(sizeof(double) * partitionCount);
        patternPartitions = (int*) malloc(sizeof(int) * nsites);
        int partitionSize = nsites/partitionCount;
        for (int i = 0; i < nsites; i++) {
            // int sitePartition =  gt_rand()%partitionCount;
            int sitePartition =  i%partitionCount;
            // int sitePartition = i/partitionSize;
            if (sitePartition > partitionCount - 1)
                sitePartition = partitionCount - 1;
            patternPartitions[i] = sitePartition;
            // printf("patternPartitions[%d] = %d\n", i, patternPartitions[i]);
        }    
        // beagleSetPatternPartitions(instance, partitionCount, patternPartitions);
    }


    // create base frequency array

#ifdef _WIN32
    std::vector<double> freqs(stateCount);
#else
    double freqs[stateCount];
#endif
    
    // create an array containing site category weights
#ifdef _WIN32
    std::vector<double> weights(rateCategoryCount);
#else
    double weights[rateCategoryCount];
#endif

    for (int eigenIndex=0; eigenIndex < eigenCount; eigenIndex++) {
        for (int i = 0; i < rateCategoryCount; i++) {
            weights[i] = gt_rand() / (double) GT_RAND_MAX;
        } 
    
        beagleSetCategoryWeights(instance, eigenIndex, &weights[0]);
    }
    
    double* eval;
    if (!eigencomplex)
        eval = (double*)malloc(sizeof(double)*stateCount);
    else
        eval = (double*)malloc(sizeof(double)*stateCount*2);
    double* evec = (double*)malloc(sizeof(double)*stateCount*stateCount);
    double* ivec = (double*)malloc(sizeof(double)*stateCount*stateCount);
    
    for (int eigenIndex=0; eigenIndex < modelCount; eigenIndex++) {
        if (!eigencomplex && ((stateCount & (stateCount-1)) == 0)) {
            
            for (int i=0; i<stateCount; i++) {
                freqs[i] = 1.0 / stateCount;
            }

            // an eigen decomposition for the general state-space JC69 model
            // If stateCount = 2^n is a power-of-two, then Sylvester matrix H_n describes
            // the eigendecomposition of the infinitesimal rate matrix
             
            double* Hn = evec;
            Hn[0*stateCount+0] = 1.0; Hn[0*stateCount+1] =  1.0; 
            Hn[1*stateCount+0] = 1.0; Hn[1*stateCount+1] = -1.0; // H_1
         
            for (int k=2; k < stateCount; k <<= 1) {
                // H_n = H_1 (Kronecker product) H_{n-1}
                for (int i=0; i<k; i++) {
                    for (int j=i; j<k; j++) {
                        double Hijold = Hn[i*stateCount + j];
                        Hn[i    *stateCount + j + k] =  Hijold;
                        Hn[(i+k)*stateCount + j    ] =  Hijold;
                        Hn[(i+k)*stateCount + j + k] = -Hijold;
                        
                        Hn[j    *stateCount + i + k] = Hn[i    *stateCount + j + k];
                        Hn[(j+k)*stateCount + i    ] = Hn[(i+k)*stateCount + j    ];
                        Hn[(j+k)*stateCount + i + k] = Hn[(i+k)*stateCount + j + k];                                
                    }
                }        
            }
            
            // Since evec is Hadamard, ivec = (evec)^t / stateCount;    
            for (int i=0; i<stateCount; i++) {
                for (int j=i; j<stateCount; j++) {
                    ivec[i*stateCount+j] = evec[j*stateCount+i] / stateCount;
                    ivec[j*stateCount+i] = ivec[i*stateCount+j]; // Symmetric
                }
            }
           
            eval[0] = 0.0;
            for (int i=1; i<stateCount; i++) {
                eval[i] = -stateCount / (stateCount - 1.0);
            }
       
        } else if (!eigencomplex) {
            for (int i=0; i<stateCount; i++) {
                freqs[i] = gt_rand() / (double) GT_RAND_MAX;
            }
        
            double** qmat=New2DArray<double>(stateCount, stateCount);    
            double* relNucRates = new double[(stateCount * stateCount - stateCount) / 2];
            
            int rnum=0;
            for(int i=0;i<stateCount;i++){
                for(int j=i+1;j<stateCount;j++){
                    relNucRates[rnum] = gt_rand() / (double) GT_RAND_MAX;
                    qmat[i][j]=relNucRates[rnum] * freqs[j];
                    qmat[j][i]=relNucRates[rnum] * freqs[i];
                    rnum++;
                }
            }

            //set diags to sum rows to 0
            double sum;
            for(int x=0;x<stateCount;x++){
                sum=0.0;
                for(int y=0;y<stateCount;y++){
                    if(x!=y) sum+=qmat[x][y];
                        }
                qmat[x][x]=-sum;
            } 
            
            double* eigvalsimag=new double[stateCount];
            double** eigvecs=New2DArray<double>(stateCount, stateCount);//eigenvecs
            double** teigvecs=New2DArray<double>(stateCount, stateCount);//temp eigenvecs
            double** inveigvecs=New2DArray<double>(stateCount, stateCount);//inv eigenvecs    
            int* iwork=new int[stateCount];
            double* work=new double[stateCount];
            
            EigenRealGeneral(stateCount, qmat, eval, eigvalsimag, eigvecs, iwork, work);
            memcpy(*teigvecs, *eigvecs, stateCount*stateCount*sizeof(double));
            InvertMatrix(teigvecs, stateCount, work, iwork, inveigvecs);
            
            for(int x=0;x<stateCount;x++){
                for(int y=0;y<stateCount;y++){
                    evec[x * stateCount + y] = eigvecs[x][y];
                    if (ievectrans)
                        ivec[x * stateCount + y] = inveigvecs[y][x];
                    else
                        ivec[x * stateCount + y] = inveigvecs[x][y];
                }
            } 
            
            Delete2DArray(qmat);
            delete[] relNucRates;
            
            delete[] eigvalsimag;
            Delete2DArray(eigvecs);
            Delete2DArray(teigvecs);
            Delete2DArray(inveigvecs);
            delete[] iwork;
            delete[] work;
        } else if (eigencomplex && stateCount==4 && eigenCount==1) {
            // create base frequency array
            double temp_freqs[4] = { 0.25, 0.25, 0.25, 0.25 };
            
            // an eigen decomposition for the 4-state 1-step circulant infinitesimal generator
            double temp_evec[4 * 4] = {
                -0.5,  0.6906786606674509,   0.15153543380548623, 0.5,
                0.5, -0.15153543380548576,  0.6906786606674498,  0.5,
                -0.5, -0.6906786606674498,  -0.15153543380548617, 0.5,
                0.5,  0.15153543380548554, -0.6906786606674503,  0.5
            };
            
            double temp_ivec[4 * 4] = {
                -0.5,  0.5, -0.5,  0.5,
                0.6906786606674505, -0.15153543380548617, -0.6906786606674507,   0.15153543380548645,
                0.15153543380548568, 0.6906786606674509,  -0.15153543380548584, -0.6906786606674509,
                0.5,  0.5,  0.5,  0.5
            };
            
            double temp_eval[8] = { -2.0, -1.0, -1.0, 0, 0, 1, -1, 0 };
            
            for(int x=0;x<stateCount;x++){
                freqs[x] = temp_freqs[x];
                eval[x] = temp_eval[x];
                eval[x+stateCount] = temp_eval[x+stateCount];
                for(int y=0;y<stateCount;y++){
                    evec[x * stateCount + y] = temp_evec[x * stateCount + y];
                    if (ievectrans)
                        ivec[x * stateCount + y] = temp_ivec[x + y * stateCount];
                    else
                        ivec[x * stateCount + y] = temp_ivec[x * stateCount + y];
                }
            } 
        } else {
            abort("should not be here");
        }
            
        beagleSetStateFrequencies(instance, eigenIndex, &freqs[0]);
        
        if (!setmatrix) {
            // set the Eigen decomposition
            beagleSetEigenDecomposition(instance, eigenIndex, &evec[0], &ivec[0], &eval[0]);
        }
    }
    
    free(eval);
    free(evec);
    free(ivec);


    
    // a list of indices and edge lengths
    int* edgeIndices = new int[edgeCount*modelCount];
    int* edgeIndicesD1 = new int[edgeCount*modelCount];
    int* edgeIndicesD2 = new int[edgeCount*modelCount];
    for(int i=0; i<edgeCount*modelCount; i++) {
        edgeIndices[i]=i;
        edgeIndicesD1[i]=(edgeCount*modelCount)+i;
        edgeIndicesD2[i]=2*(edgeCount*modelCount)+i;
    }
    double* edgeLengths = new double[edgeCount*modelCount];
    for(int i=0; i<edgeCount; i++) {
        edgeLengths[i]=gt_rand() / (double) GT_RAND_MAX;
    }
    
    // create a list of partial likelihood update operations
    // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
    int operationCount = internalCount*modelCount;
    int beagleOpCount = BEAGLE_OP_COUNT;
    if (partitionCount > 1)
        beagleOpCount = BEAGLE_PARTITION_OP_COUNT;
    int* operations = new int[beagleOpCount*operationCount];
    int unpartOpsCount = internalCount*eigenCount;
    int* scalingFactorsIndices = new int[unpartOpsCount]; // internal nodes



    for(int i=0; i<unpartOpsCount; i++){
        int child1Index;
        if (((i % internalCount)*2) < ntaxa)
            child1Index = (i % internalCount)*2;
        else
            child1Index = i*2 - internalCount * (int)(i / internalCount);
        int child2Index;
        if (((i % internalCount)*2+1) < ntaxa)
            child2Index = (i % internalCount)*2+1;
        else
            child2Index = i*2+1 - internalCount * (int)(i / internalCount);

        for (int j=0; j<partitionCount; j++) {
            int op = partitionCount*i + j;
            operations[op*beagleOpCount+0] = ntaxa+i;
            operations[op*beagleOpCount+1] = (dynamicScaling ? i : BEAGLE_OP_NONE);
            operations[op*beagleOpCount+2] = (dynamicScaling ? i : BEAGLE_OP_NONE);
            operations[op*beagleOpCount+3] = child1Index;
            operations[op*beagleOpCount+4] = child1Index + j*edgeCount;
            operations[op*beagleOpCount+5] = child2Index;
            operations[op*beagleOpCount+6] = child2Index + j*edgeCount;
            if (partitionCount > 1) {
                operations[op*beagleOpCount+7] = j;
                operations[op*beagleOpCount+8] = (dynamicScaling ? internalCount : BEAGLE_OP_NONE);
            }
            // printf("op %d i %d j %d dest %d c1 %d c2 %d c1m %d c2m %d p %d\n",
            //        op, i, j, ntaxa+i, child1Index, child2Index,
            //        operations[op*beagleOpCount+4], operations[op*beagleOpCount+6], j);
        }

        scalingFactorsIndices[i] = i;

        if (autoScaling)
            scalingFactorsIndices[i] += ntaxa;
    }   

    int* rootIndices = new int[eigenCount * partitionCount];
    int* lastTipIndices = new int[eigenCount * partitionCount];
    int* lastTipIndicesD1 = new int[eigenCount * partitionCount];
    int* lastTipIndicesD2 = new int[eigenCount * partitionCount];
    int* categoryWeightsIndices = new int[eigenCount * partitionCount];
    int* stateFrequencyIndices = new int[eigenCount * partitionCount];
    int* cumulativeScalingFactorIndices = new int[eigenCount * partitionCount];
    int* partitionIndices = new int[partitionCount];
    
    for (int eigenIndex=0; eigenIndex < eigenCount; eigenIndex++) {
        int pOffset = partitionCount*eigenIndex;

        for (int partitionIndex=0; partitionIndex < partitionCount; partitionIndex++) {
            if (eigenIndex == 0)
                partitionIndices[partitionIndex] = partitionIndex;
            rootIndices[partitionIndex + pOffset] = ntaxa+(internalCount*(eigenIndex+1))-1;//ntaxa*2-2;
            lastTipIndices[partitionIndex + pOffset] = ntaxa-1;
            lastTipIndicesD1[partitionIndex + pOffset] = (ntaxa-1) + (edgeCount*modelCount);
            lastTipIndicesD2[partitionIndex + pOffset] = (ntaxa-1) + 2*(edgeCount*modelCount);
            categoryWeightsIndices[partitionIndex + pOffset] = eigenIndex;
            stateFrequencyIndices[partitionIndex + pOffset] = 0;
            cumulativeScalingFactorIndices[partitionIndex + pOffset] = ((manualScaling || dynamicScaling) ? (scaleCount*eigenCount-1)-eigenCount+eigenIndex+1 : BEAGLE_OP_NONE);
        }

        if (dynamicScaling)
            beagleResetScaleFactors(instance, cumulativeScalingFactorIndices[eigenIndex]);
    }

    // start timing!
    struct timeval time0, time1, time2, time3, time4, time5;
    double bestTimeSetPartitions, bestTimeUpdateTransitionMatrices, bestTimeUpdatePartials, bestTimeAccumulateScaleFactors, bestTimeCalculateRootLogLikelihoods, bestTimeTotal;
    
    double logL = 0.0;
    double deriv1 = 0.0;
    double deriv2 = 0.0;
    
    double previousLogL = 0.0;
    double previousDeriv1 = 0.0;
    double previousDeriv2 = 0.0;

    int* eigenIndices = new int[edgeCount * modelCount];
    int* categoryRateIndices = new int[edgeCount * modelCount];
    for (int eigenIndex=0; eigenIndex < modelCount; eigenIndex++) {
        for(int j=0; j<edgeCount; j++) {
            eigenIndices[eigenIndex*edgeCount + j] = eigenIndex;
            categoryRateIndices[eigenIndex*edgeCount + j] = eigenIndex;
            edgeLengths[eigenIndex*edgeCount + j] = edgeLengths[j];
        }
    }

    if (randomTree && eigenCount==1 && !unrooted) {

        std::vector <node*> nodes;
        nodes.push_back(createNewNode(0));
        int tipsAdded = 1;
        node* newParent;
        while (tipsAdded < ntaxa) {
            int sibling;
            if (pectinate)
                sibling = nodes.size()-1;
            else
                sibling = gt_rand() % nodes.size();
            node* newTip = createNewNode(tipsAdded);
            newParent = createNewNode(ntaxa + tipsAdded - 1);
            nodes.push_back(newTip);
            nodes.push_back(newParent);
            tipsAdded++;            
            newParent->left  = nodes[sibling];
            newParent->right = newTip;            
            if (nodes[sibling]->parent != NULL) {
                newParent->parent = nodes[sibling]->parent;
                if (nodes[sibling]->parent->left == nodes[sibling]) {
                    nodes[sibling]->parent->left = newParent;
                } else {
                    nodes[sibling]->parent->right = newParent;
                }
            }
            nodes[sibling]->parent = newParent;
            newTip->parent         = newParent;
        }
        node* root = nodes[0];
        while(root->parent != NULL) {
            root = root->parent;
        }
        int rootIndex = newParent->data;
        newParent->data = root->data;
        root->data = rootIndex;

        if (rerootTrees) {
            int bestRerootNode = -1;
            int bestLaunchCount = countLaunches(root);

            // printf("\nroot node   = %d\tparallel launches = %d\n", root->data, bestLaunchCount);


            std::vector<node*> newNodes;

            for(int i = 0; i < nodes.size(); i++) {

                // printf("reroot node = %02d\t", nodes[i]->data);

                node* rerootNode = nodes[i];

                if (rerootNode->parent != NULL && rerootNode->parent != root) {
                    
                    node* newRoot = reroot(rerootNode, root, newNodes);

                    int launchCount = countLaunches(newRoot);

                    newNodes.clear();

                    // printf("parallel launches = %d\n", launchCount);

                    if (launchCount < bestLaunchCount) {
                        bestLaunchCount = launchCount;
                        bestRerootNode = i;
                    }

                }
                // else {printf("doesn't change tree\n");}

            }

            if (bestRerootNode != -1) {
                // printf("\nbestLaunchCount = %d, node index = %d\n\n", bestLaunchCount, bestRerootNode);
                node* rerootNode = nodes[bestRerootNode];
                node* oldRoot = root;
                root = reroot(rerootNode, oldRoot, newNodes);
            }

        } 

        std::stack <node *> S;
        reverseLevelOrder(root, S);

        // while (S.empty() == false) {
        //     node* tmpNode = S.top();
        //     std::cout << tmpNode->data << " ";
        //     S.pop();
        // }
        // std::cout << std::endl;
        // reverseLevelOrder(root, S);

        // struct node *root = createNewNode(4);
        // root->left        = createNewNode(0);
        // root->right       = createNewNode(6);
        // root->right->left  = createNewNode(5);
        // root->right->right = createNewNode(3);
        // root->right->left->left  = createNewNode(1);
        // root->right->left->right = createNewNode(2);
        // std::stack <node *> S;
        // reverseLevelOrder(root, S);

        // printf("launch count = %03d", countLaunches(root));

        for(int op=0; op<unpartOpsCount; op++){
            node* parent = S.top();
            S.pop();
            int parentIndex = parent->data;
            int child1Index = parent->left->data;
            int child2Index = parent->right->data;

            for (int j=0; j<partitionCount; j++) {
                int opJ = partitionCount*op + j;
                operations[opJ*beagleOpCount+0] = parentIndex;
                operations[opJ*beagleOpCount+1] = (dynamicScaling ? parentIndex : BEAGLE_OP_NONE);
                operations[opJ*beagleOpCount+2] = (dynamicScaling ? parentIndex : BEAGLE_OP_NONE);
                operations[opJ*beagleOpCount+3] = child1Index;
                operations[opJ*beagleOpCount+4] = child1Index + j*edgeCount;;
                operations[opJ*beagleOpCount+5] = child2Index;
                operations[opJ*beagleOpCount+6] = child2Index + j*edgeCount;
                if (partitionCount > 1) {
                    operations[opJ*beagleOpCount+7] = j;
                    operations[opJ*beagleOpCount+8] = (dynamicScaling ? internalCount : BEAGLE_OP_NONE);
                }
            // printf("op %02d part %02d dest %02d c1 %02d c2 %02d\n",
            //        opJ, j, parentIndex, child1Index, child2Index);
            }
            // printf("\n");
        }   
    }


    for (int i=0; i<nreps; i++){

        if (newDataPerRep) {
            for(int ii=0; ii<ntaxa; ii++)
            {
                if (compactTipCount == 0 || (ii >= (compactTipCount-1) && ii != (ntaxa-1))) {
                    double* tmpPartials = getRandomTipPartials(nsites, stateCount);
                    beagleSetTipPartials(instance, ii, tmpPartials);
                    free(tmpPartials);
                } else {
                    int* tmpStates = getRandomTipStates(nsites, stateCount);
                    beagleSetTipStates(instance, ii, tmpStates);
                    free(tmpStates);                
                }
            }
        }

        if (manualScaling && (!(i % rescaleFrequency) || !((i-1) % rescaleFrequency))) {
            for(int j=0; j<operationCount; j++){
                int sIndex = j / partitionCount;
                operations[beagleOpCount*j+1] = (((manualScaling && !(i % rescaleFrequency))) ? sIndex : BEAGLE_OP_NONE);
                operations[beagleOpCount*j+2] = (((manualScaling && (i % rescaleFrequency))) ? sIndex : BEAGLE_OP_NONE);
            }
        }
        
        gettimeofday(&time0,NULL);

        if (partitionCount > 1 && i==0) { //!(i % rescaleFrequency)) {
            if (beagleSetPatternPartitions(instance, partitionCount, patternPartitions) != BEAGLE_SUCCESS) {
                printf("ERROR: No BEAGLE implementation for beagleSetPatternPartitions\n");
                exit(-1);
            }
        }

        gettimeofday(&time1,NULL);

        if (partitionCount > 1) {
            int totalEdgeCount = edgeCount * modelCount;
            beagleUpdateTransitionMatricesWithMultipleModels(
                                           instance,     // instance
                                           eigenIndices,   // eigenIndex
                                           categoryRateIndices,   // category rate index
                                           edgeIndices,   // probabilityIndices
                                           (calcderivs ? edgeIndicesD1 : NULL), // firstDerivativeIndices
                                           (calcderivs ? edgeIndicesD2 : NULL), // secondDerivativeIndices
                                           edgeLengths,   // edgeLengths
                                           totalEdgeCount);            // count
        } else {
            for (int eigenIndex=0; eigenIndex < modelCount; eigenIndex++) {
                if (!setmatrix) {
                    // tell BEAGLE to populate the transition matrices for the above edge lengths
                    beagleUpdateTransitionMatrices(instance,     // instance
                                                   eigenIndex,             // eigenIndex
                                                   &edgeIndices[eigenIndex*edgeCount],   // probabilityIndices
                                                   (calcderivs ? &edgeIndicesD1[eigenIndex*edgeCount] : NULL), // firstDerivativeIndices
                                                   (calcderivs ? &edgeIndicesD2[eigenIndex*edgeCount] : NULL), // secondDerivativeIndices
                                                   edgeLengths,   // edgeLengths
                                                   edgeCount);            // count
                } else {
                    double* inMatrix = new double[stateCount*stateCount*rateCategoryCount];
                    for (int matrixIndex=0; matrixIndex < edgeCount; matrixIndex++) {
                        for(int z=0;z<rateCategoryCount;z++){
                            for(int x=0;x<stateCount;x++){
                                for(int y=0;y<stateCount;y++){
                                    inMatrix[z*stateCount*stateCount + x*stateCount + y] = gt_rand() / (double) GT_RAND_MAX;
                                }
                            } 
                        }
                        beagleSetTransitionMatrix(instance, edgeIndices[eigenIndex*edgeCount + matrixIndex], inMatrix, 1);
                        if (calcderivs) {
                            beagleSetTransitionMatrix(instance, edgeIndicesD1[eigenIndex*edgeCount + matrixIndex], inMatrix, 0);
                            beagleSetTransitionMatrix(instance, edgeIndicesD2[eigenIndex*edgeCount + matrixIndex], inMatrix, 0);
                        }
                    }
                }
            }

        }

        // std::cout.setf(std::ios::showpoint);
        // // std::cout.setf(std::ios::floatfield, std::ios::fixed);
        // std::cout.precision(4);
        // unsigned int partialsOps = internalCount * eigenCount;
        // unsigned int flopsPerPartial = (stateCount * 4) - 2 + 1;
        // unsigned long long partialsSize = stateCount * nsites * rateCategoryCount;
        // unsigned long long partialsTotal = partialsSize * partialsOps;
        // unsigned long long flopsTotal = partialsTotal * flopsPerPartial;

        // std::cout << " compute throughput:   ";

        // for (int pRep=0; pRep < 50; pRep++) {
            gettimeofday(&time2, NULL);

            // update the partials
            if (partitionCount > 1) {
                beagleUpdatePartialsByPartition( instance,                   // instance
                                (BeagleOperationByPartition*)operations,     // operations
                                internalCount*eigenCount*partitionCount);    // operationCount
            } else {
                beagleUpdatePartials( instance,      // instance
                                (BeagleOperation*)operations,     // operations
                                internalCount*eigenCount,              // operationCount
                                (dynamicScaling ? internalCount : BEAGLE_OP_NONE));             // cumulative scaling index
            }

            gettimeofday(&time3, NULL);

            // struct timespec ts;
            // ts.tv_sec = 0;
            // ts.tv_nsec = 100000000;
            // nanosleep(&ts, NULL);

            // std::cout << (flopsTotal/getTimeDiff(time2, time3))/1000000000.0 << ", ";
        // }
        // std::cout << " GFLOPS " << std::endl<< std::endl;

        // std::cout << " compute throughput:   " << (flopsTotal/getTimeDiff(time2, time3))/1000000000.0 << " GFLOPS " << std::endl;


        int scalingFactorsCount = internalCount;
                
        for (int eigenIndex=0; eigenIndex < eigenCount; eigenIndex++) {
            if (manualScaling && !(i % rescaleFrequency)) {
                beagleResetScaleFactors(instance,
                                        cumulativeScalingFactorIndices[eigenIndex]);
                
                beagleAccumulateScaleFactors(instance,
                                       &scalingFactorsIndices[eigenIndex*internalCount],
                                       scalingFactorsCount,
                                       cumulativeScalingFactorIndices[eigenIndex]);
            } else if (autoScaling) {
                beagleAccumulateScaleFactors(instance, &scalingFactorsIndices[eigenIndex*internalCount], scalingFactorsCount, BEAGLE_OP_NONE);
            }
        }
        
        gettimeofday(&time4, NULL);

        // calculate the site likelihoods at the root node
        if (!unrooted) {
            if (partitionCount > 1) {
                beagleCalculateRootLogLikelihoodsByPartition(
                                            instance,               // instance
                                            rootIndices,// bufferIndices
                                            categoryWeightsIndices,                // weights
                                            stateFrequencyIndices,                 // stateFrequencies
                                            cumulativeScalingFactorIndices,
                                            partitionIndices,
                                            partitionCount,
                                            eigenCount,                      // count
                                            partitionLogLs,
                                            &logL);         // outLogLikelihoods
            } else {
                beagleCalculateRootLogLikelihoods(instance,               // instance
                                            rootIndices,// bufferIndices
                                            categoryWeightsIndices,                // weights
                                            stateFrequencyIndices,                 // stateFrequencies
                                            cumulativeScalingFactorIndices,
                                            eigenCount,                      // count
                                            &logL);         // outLogLikelihoods
            }
        } else {
            if (partitionCount > 1) {
                beagleCalculateEdgeLogLikelihoodsByPartition(
                                                  instance,
                                                  rootIndices,
                                                  lastTipIndices,
                                                  lastTipIndices,
                                                  (calcderivs ? lastTipIndicesD1 : NULL),
                                                  (calcderivs ? lastTipIndicesD2 : NULL),
                                                  categoryWeightsIndices,
                                                  stateFrequencyIndices,
                                                  cumulativeScalingFactorIndices,
                                                  partitionIndices,
                                                  partitionCount,
                                                  eigenCount,
                                                  partitionLogLs,
                                                  &logL,
                                                  (calcderivs ? partitionD1 : NULL),
                                                  (calcderivs ? &deriv1 : NULL),
                                                  (calcderivs ? partitionD2 : NULL),
                                                  (calcderivs ? &deriv2 : NULL));
            } else {            
                beagleCalculateEdgeLogLikelihoods(instance,               // instance
                                                  rootIndices,// bufferIndices
                                                  lastTipIndices,
                                                  lastTipIndices,
                                                  (calcderivs ? lastTipIndicesD1 : NULL),
                                                  (calcderivs ? lastTipIndicesD2 : NULL),
                                                  categoryWeightsIndices,                // weights
                                                  stateFrequencyIndices,                 // stateFrequencies
                                                  cumulativeScalingFactorIndices,
                                                  eigenCount,                      // count
                                                  &logL,    // outLogLikelihood
                                                  (calcderivs ? &deriv1 : NULL),
                                                  (calcderivs ? &deriv2 : NULL));
            }

        }
        // end timing!
        gettimeofday(&time5,NULL);
        
        // std::cout.setf(std::ios::showpoint);
        // std::cout.setf(std::ios::floatfield, std::ios::fixed);
        // int timePrecision = 6;
        // int speedupPrecision = 2;
        // int percentPrecision = 2;
        // std::cout << "run " << i << ": ";
        // printTiming(getTimeDiff(time1, time5), timePrecision, resource, cpuTimeTotal, speedupPrecision, 0, 0, 0);
        // fprintf(stdout, "logL = %.5f  ", logL);

            // unsigned int partialsOps = internalCount * eigenCount;
            // unsigned int flopsPerPartial = (stateCount * 4) - 2 + 1;
            // unsigned long long partialsSize = stateCount * nsites * rateCategoryCount;
            // unsigned long long partialsTotal = partialsSize * partialsOps;
            // unsigned long long flopsTotal = partialsTotal * flopsPerPartial;
            // std::cout << " compute throughput:   " << (flopsTotal/getTimeDiff(time2, time3))/1000000000.0 << " GFLOPS " << std::endl;
    
        if (i == 0 || getTimeDiff(time0, time5) < bestTimeTotal) {
            bestTimeTotal = getTimeDiff(time0, time5);
            bestTimeSetPartitions = getTimeDiff(time0, time1);
            bestTimeUpdateTransitionMatrices = getTimeDiff(time1, time2);
            bestTimeUpdatePartials = getTimeDiff(time2, time3);
            bestTimeAccumulateScaleFactors = getTimeDiff(time3, time4);
            bestTimeCalculateRootLogLikelihoods = getTimeDiff(time4, time5);
        }
        
        if (!(logL - logL == 0.0))
            fprintf(stdout, "error: invalid lnL\n");

        if (!newDataPerRep) {        
            if (i > 0 && std::abs(logL - previousLogL) > MAX_DIFF)
                fprintf(stdout, "error: large lnL difference between reps\n");
        }
        
        if (calcderivs) {
            if (!(deriv1 - deriv1 == 0.0) || !(deriv2 - deriv2 == 0.0))
                fprintf(stdout, "error: invalid deriv\n");
            
            if (i > 0 && ((std::abs(deriv1 - previousDeriv1) > MAX_DIFF) || (std::abs(deriv2 - previousDeriv2) > MAX_DIFF)) )
                fprintf(stdout, "error: large deriv difference between reps\n");
        }

        previousLogL = logL;
        previousDeriv1 = deriv1;
        previousDeriv2 = deriv2;        
    }

    if (resource == 0) {
        cpuTimeSetPartitions = bestTimeSetPartitions;
        cpuTimeUpdateTransitionMatrices = bestTimeUpdateTransitionMatrices;
        cpuTimeUpdatePartials = bestTimeUpdatePartials;
        cpuTimeAccumulateScaleFactors = bestTimeAccumulateScaleFactors;
        cpuTimeCalculateRootLogLikelihoods = bestTimeCalculateRootLogLikelihoods;
        cpuTimeTotal = bestTimeTotal;
    }
    
    if (!calcderivs)
        fprintf(stdout, "logL = %.5f \n", logL);
    else
        fprintf(stdout, "logL = %.5f d1 = %.5f d2 = %.5f\n", logL, deriv1, deriv2);

    if (partitionCount > 1) {
        fprintf(stdout, " (");
        for (int p=0; p < partitionCount; p++) {
            fprintf(stdout, "p%d = %.5f", p, partitionLogLs[p]);
            if (p < partitionCount - 1)
                fprintf(stdout, ", ");
        }
        fprintf(stdout, ")\n");
    }
    
    if (partitionCount > 1) {
        fprintf(stdout, " (");
        for (int p=0; p < partitionCount; p++) {
            fprintf(stdout, "p%dD1 = %.5f", p, partitionD1[p]);
            if (p < partitionCount - 1)
                fprintf(stdout, ", ");
        }
        fprintf(stdout, ")\n");
    }
    
    if (partitionCount > 1) {
        fprintf(stdout, " (");
        for (int p=0; p < partitionCount; p++) {
            fprintf(stdout, "p%dD2 = %.5f", p, partitionD2[p]);
            if (p < partitionCount - 1)
                fprintf(stdout, ", ");
        }
        fprintf(stdout, ")\n");
    }


    if (sitelikes) {
        double* siteLogLs = (double*) malloc(sizeof(double) * nsites);
        beagleGetSiteLogLikelihoods(instance, siteLogLs);
        double sumLogL = 0.0;
        fprintf(stdout, "site likelihoods = ");
        for (int i=0; i<nsites; i++) {
            fprintf(stdout, "%.5f \t", siteLogLs[i]);
            sumLogL += siteLogLs[i] * patternWeights[i];
        }
        fprintf(stdout, "\nsumLogL = %.5f\n", sumLogL);
        free(siteLogLs);
    }

    free(patternWeights);
    if (partitionCount > 1) {
        free(patternPartitions);
    }

    std::cout.setf(std::ios::showpoint);
    std::cout.setf(std::ios::floatfield, std::ios::fixed);
    int timePrecision = 6;
    int speedupPrecision = 2;
    int percentPrecision = 2;
    std::cout << "best run: ";
    printTiming(bestTimeTotal, timePrecision, resource, cpuTimeTotal, speedupPrecision, 0, 0, 0);
    if (fullTiming) {
        std::cout << " setPartitions:  ";
        printTiming(bestTimeSetPartitions, timePrecision, resource, cpuTimeSetPartitions, speedupPrecision, 1, bestTimeTotal, percentPrecision);
        std::cout << " transMats:  ";
        printTiming(bestTimeUpdateTransitionMatrices, timePrecision, resource, cpuTimeUpdateTransitionMatrices, speedupPrecision, 1, bestTimeTotal, percentPrecision);
        std::cout << " partials:   ";
        printTiming(bestTimeUpdatePartials, timePrecision, resource, cpuTimeUpdatePartials, speedupPrecision, 1, bestTimeTotal, percentPrecision);
        unsigned int partialsOps = internalCount * eigenCount;
        unsigned int flopsPerPartial = (stateCount * 4) - 2 + 1;
        unsigned int bytesPerPartial = 3 * (requireDoublePrecision ? 8 : 4);
        if (manualScaling) {
            flopsPerPartial++;
            bytesPerPartial += (requireDoublePrecision ? 8 : 4);
        }
        unsigned int matrixBytes = partialsOps * 2 * stateCount*stateCount*rateCategoryCount * (requireDoublePrecision ? 8 : 4);
        unsigned long long partialsSize = stateCount * nsites * rateCategoryCount;
        unsigned long long partialsTotal = partialsSize * partialsOps;
        unsigned long long flopsTotal = partialsTotal * flopsPerPartial;
        std::cout << " partials throughput:   " << (partialsTotal/bestTimeUpdatePartials)/1000000.0 << " M partials/second " << std::endl;
        std::cout << " compute throughput:   " << (flopsTotal/bestTimeUpdatePartials)/1000000000.0 << " GFLOPS " << std::endl;
        std::cout << " memory bandwidth:   " << (((partialsTotal * bytesPerPartial + matrixBytes)/bestTimeUpdatePartials))/1000000000.0 << " GB/s " << std::endl;
        if (manualScaling || autoScaling) {
            std::cout << " accScalers: ";
            printTiming(bestTimeAccumulateScaleFactors, timePrecision, resource, cpuTimeAccumulateScaleFactors, speedupPrecision, 1, bestTimeTotal, percentPrecision);
        }
        std::cout << " rootLnL:    ";
        printTiming(bestTimeCalculateRootLogLikelihoods, timePrecision, resource, cpuTimeCalculateRootLogLikelihoods, speedupPrecision, 1, bestTimeTotal, percentPrecision);

        std::cout << " tree throughput total:   " << (partialsTotal/bestTimeTotal)/1000000.0 << " M partials/second " << std::endl;

    }
    std::cout << "\n";
    
    beagleFinalizeInstance(instance);
}

void printResourceList() {
    // print version and citation info
    fprintf(stdout, "BEAGLE version %s\n", beagleGetVersion());
    fprintf(stdout, "%s\n", beagleGetCitation());     

    // print resource list
    BeagleResourceList* rList;
    rList = beagleGetResourceList();
    fprintf(stdout, "Available resources:\n");
    for (int i = 0; i < rList->length; i++) {
        fprintf(stdout, "\tResource %i:\n\t\tName : %s\n", i, rList->list[i].name);
        fprintf(stdout, "\t\tDesc : %s\n", rList->list[i].description);
        fprintf(stdout, "\t\tFlags:");
        printFlags(rList->list[i].supportFlags);
        fprintf(stdout, "\n");
    }    
    fprintf(stdout, "\n");
    std::exit(0);
}

void helpMessage() {
    std::cerr << "Usage:\n\n";
    std::cerr << "synthetictest [--help] [--resourcelist] [--states <integer>] [--taxa <integer>] [--sites <integer>] [--rates <integer>] [--manualscale] [--autoscale] [--dynamicscale] [--rsrc <integer>] [--reps <integer>] [--doubleprecision] [--SSE] [--AVX] [--compact-tips <integer>] [--seed <integer>] [--rescale-frequency <integer>] [--full-timing] [--unrooted] [--calcderivs] [--logscalers] [--eigencount <integer>] [--eigencomplex] [--ievectrans] [--setmatrix] [--opencl] [--partitions <integer>] [--sitelikes] [--newdata] [--randomtree] [--reroot] [--stdrand] [--pectinate]\n\n";
    std::cerr << "If --help is specified, this usage message is shown\n\n";
    std::cerr << "If --manualscale, --autoscale, or --dynamicscale is specified, BEAGLE will rescale the partials during computation\n\n";
    std::cerr << "If --full-timing is specified, you will see more detailed timing results (requires BEAGLE_DEBUG_SYNCH defined to report accurate values)\n\n";
    std::exit(0);
}

void interpretCommandLineParameters(int argc, const char* argv[],
                                    int* stateCount,
                                    int* ntaxa,
                                    int* nsites,
                                    bool* manualScaling,
                                    bool* autoScaling,
                                    bool* dynamicScaling,
                                    int* rateCategoryCount,
                                    std::vector<int>* rsrc,
                                    int* nreps,
                                    bool* fullTiming,
                                    bool* requireDoublePrecision,
                                    bool* requireSSE,
                                    bool* requireAVX,
                                    int* compactTipCount,
                                    int* randomSeed,
                                    int* rescaleFrequency,
                                    bool* unrooted,
                                    bool* calcderivs,
                                    bool* logscalers,
                                    int* eigenCount,
                                    bool* eigencomplex,
                                    bool* ievectrans,
                                    bool* setmatrix,
                                    bool* opencl,
                                    int*  partitions,
                                    bool* sitelikes,
                                    bool* newDataPerRep,
                                    bool* randomTree,
                                    bool* rerootTrees,
                                    bool* pectinate)    {
    bool expecting_stateCount = false;
    bool expecting_ntaxa = false;
    bool expecting_nsites = false;
    bool expecting_rateCategoryCount = false;
    bool expecting_nreps = false;
    bool expecting_rsrc = false;
    bool expecting_compactTipCount = false;
    bool expecting_seed = false;
    bool expecting_rescaleFrequency = false;
    bool expecting_eigenCount = false;
    bool expecting_partitions = false;
    
    for (unsigned i = 1; i < argc; ++i) {
        std::string option = argv[i];
        
        if (expecting_stateCount) {
            *stateCount = (unsigned)atoi(option.c_str());
            expecting_stateCount = false;
        } else if (expecting_ntaxa) {
            *ntaxa = (unsigned)atoi(option.c_str());
            expecting_ntaxa = false;
        } else if (expecting_nsites) {
            *nsites = (unsigned)atoi(option.c_str());
            expecting_nsites = false;
        } else if (expecting_rateCategoryCount) {
            *rateCategoryCount = (unsigned)atoi(option.c_str());
            expecting_rateCategoryCount = false;
        } else if (expecting_rsrc) {
            std::stringstream ss(option);
            int j;
            while (ss >> j) {
                rsrc->push_back(j);
                if (ss.peek() == ',')
                    ss.ignore();
            }
            expecting_rsrc = false;            
        } else if (expecting_nreps) {
            *nreps = (unsigned)atoi(option.c_str());
            expecting_nreps = false;
        } else if (expecting_compactTipCount) {
            *compactTipCount = (unsigned)atoi(option.c_str());
            expecting_compactTipCount = false;
        } else if (expecting_seed) {
            *randomSeed = (unsigned)atoi(option.c_str());
            expecting_seed = false;
        } else if (expecting_rescaleFrequency) {
            *rescaleFrequency = (unsigned)atoi(option.c_str());
            expecting_rescaleFrequency = false;
        } else if (expecting_eigenCount) {
            *eigenCount = (unsigned)atoi(option.c_str());
            expecting_eigenCount = false;
        } else if (expecting_partitions) {
            *partitions = (unsigned)atoi(option.c_str());
            expecting_partitions = false;
        } else if (option == "--help") {
            helpMessage();
        } else if (option == "--resourcelist") {
            printResourceList();
        } else if (option == "--manualscale") {
            *manualScaling = true;
        } else if (option == "--autoscale") {
            *autoScaling = true;
        } else if (option == "--dynamicscale") {
            *dynamicScaling = true;
        } else if (option == "--doubleprecision") {
            *requireDoublePrecision = true;
        } else if (option == "--states") {
            expecting_stateCount = true;
        } else if (option == "--taxa") {
            expecting_ntaxa = true;
        } else if (option == "--sites") {
            expecting_nsites = true;
        } else if (option == "--rates") {
            expecting_rateCategoryCount = true;
        } else if (option == "--rsrc") {
            expecting_rsrc = true;
        } else if (option == "--reps") {
            expecting_nreps = true;
        } else if (option == "--compact-tips") {
            expecting_compactTipCount = true;
        } else if (option == "--rescale-frequency") {
            expecting_rescaleFrequency = true;
        } else if (option == "--seed") {
            expecting_seed = true;
        } else if (option == "--full-timing") {
            *fullTiming = true;
        } else if (option == "--SSE") {
            *requireSSE = true;
        } else if (option == "--AVX") {
            *requireAVX = true;
        } else if (option == "--unrooted") {
            *unrooted = true;
        } else if (option == "--calcderivs") {
            *calcderivs = true;
        } else if (option == "--logscalers") {
            *logscalers = true;
        } else if (option == "--eigencount") {
            expecting_eigenCount = true;
        } else if (option == "--eigencomplex") {
            *eigencomplex = true;
        } else if (option == "--ievectrans") {
            *ievectrans = true;
        } else if (option == "--setmatrix") {
            *setmatrix = true;
        } else if (option == "--opencl") {
            *opencl = true;
        } else if (option == "--partitions") {
            expecting_partitions = true;
        } else if (option == "--sitelikes") {
            *sitelikes = true;
        } else if (option == "--newdata") {
            *newDataPerRep = true;
        } else if (option == "--randomtree") {
            *randomTree = true;
        } else if (option == "--stdrand") {
            useStdlibRand = true;
        } else if (option == "--reroot") {
            *rerootTrees = true;
        } else if (option == "--pectinate") {
            *pectinate = true;
        } else {
            std::string msg("Unknown command line parameter \"");
            msg.append(option);         
            abort(msg.c_str());
        }
    }
    
    if (expecting_stateCount)
        abort("read last command line option without finding value associated with --states");
    
    if (expecting_ntaxa)
        abort("read last command line option without finding value associated with --taxa");
    
    if (expecting_nsites)
        abort("read last command line option without finding value associated with --sites");
    
    if (expecting_rateCategoryCount)
        abort("read last command line option without finding value associated with --rates");

    if (expecting_rsrc)
        abort("read last command line option without finding value associated with --rsrc");
    
    if (expecting_nreps)
        abort("read last command line option without finding value associated with --reps");
    
    if (expecting_seed)
        abort("read last command line option without finding value associated with --seed");
    
    if (expecting_rescaleFrequency)
        abort("read last command line option without finding value associated with --rescale-frequency");

    if (expecting_compactTipCount)
        abort("read last command line option without finding value associated with --compact-tips");

    if (expecting_eigenCount)
        abort("read last command line option without finding value associated with --eigencount");

    if (expecting_partitions)
        abort("read last command line option without finding value associated with --partitions");

    if (*stateCount < 2)
        abort("invalid number of states supplied on the command line");
        
    if (*ntaxa < 2)
        abort("invalid number of taxa supplied on the command line");
      
    if (*nsites < 1)
        abort("invalid number of sites supplied on the command line");
    
    if (*rateCategoryCount < 1)
        abort("invalid number of rates supplied on the command line");
        
    if (*nreps < 1)
        abort("invalid number of reps supplied on the command line");

    if (*randomSeed < 1)
        abort("invalid number for seed supplied on the command line");   
        
    if (*manualScaling && *rescaleFrequency < 1)
        abort("invalid number for rescale-frequency supplied on the command line");   
    
    if (*compactTipCount < 0 || *compactTipCount > *ntaxa)
        abort("invalid number for compact-tips supplied on the command line");
    
    if (*calcderivs && !(*unrooted))
        abort("calcderivs option requires unrooted tree option");
    
    if (*eigenCount < 1)
        abort("invalid number for eigencount supplied on the command line");
    
    if (*eigencomplex && (*stateCount != 4 || *eigenCount != 1))
        abort("eigencomplex option only works with stateCount=4 and eigenCount=1");

    if (*partitions < 1 || *partitions > *nsites)
        abort("invalid number for partitions supplied on the command line");

    if (*randomTree && (*eigenCount!=1 || *unrooted))
        abort("random tree topology can only be used with eigencount=1 and unrooted trees");
}

int main( int argc, const char* argv[] )
{
    // Default values
    int stateCount = 4;
    int ntaxa = 16;
    int nsites = 10000;
    bool manualScaling = false;
    bool autoScaling = false;
    bool dynamicScaling = false;
    bool requireDoublePrecision = false;
    bool requireSSE = false;
    bool requireAVX = false;
    bool unrooted = false;
    bool calcderivs = false;
    int compactTipCount = 0;
    int randomSeed = 1;
    int rescaleFrequency = 1;
    bool logscalers = false;
    int eigenCount = 1;
    bool eigencomplex = false;
    bool ievectrans = false;
    bool setmatrix = false;
    bool opencl = false;
    bool sitelikes = false;
    int partitions = 1;
    bool newDataPerRep = false;
    bool randomTree = false;
    bool rerootTrees = false;
    bool pectinate = false;
    useStdlibRand = false;

    std::vector<int> rsrc;
    rsrc.push_back(-1);

    int nreps = 5;
    bool fullTiming = false;
    
    int rateCategoryCount = 4;
    
    interpretCommandLineParameters(argc, argv, &stateCount, &ntaxa, &nsites, &manualScaling, &autoScaling,
                                   &dynamicScaling, &rateCategoryCount, &rsrc, &nreps, &fullTiming,
                                   &requireDoublePrecision, &requireSSE, &requireAVX, &compactTipCount, &randomSeed,
                                   &rescaleFrequency, &unrooted, &calcderivs, &logscalers,
                                   &eigenCount, &eigencomplex, &ievectrans, &setmatrix, &opencl,
                                   &partitions, &sitelikes, &newDataPerRep, &randomTree, &rerootTrees, &pectinate);
    
    std::cout << "\nSimulating genomic ";
    if (stateCount == 4)
        std::cout << "DNA";
    else
        std::cout << stateCount << "-state data";
    if (partitions > 1) {
        std::cout << " with " << ntaxa << " taxa, " << nsites << " site patterns, and " << partitions << " partitions (" << nreps << " rep" << (nreps > 1 ? "s" : "");
    } else {
        std::cout << " with " << ntaxa << " taxa and " << nsites << " site patterns (" << nreps << " rep" << (nreps > 1 ? "s" : "");
    }
    std::cout << (manualScaling ? ", manual scaling":(autoScaling ? ", auto scaling":(dynamicScaling ? ", dynamic scaling":""))) << ", random seed " << randomSeed << ")\n\n";


    BeagleResourceList* rl = beagleGetResourceList();
    if(rl != NULL){
        for(int i=0; i<rl->length; i++){
            if (rsrc.size() == 1 || std::find(rsrc.begin(), rsrc.end(), i)!=rsrc.end()) {
                runBeagle(i,
                          stateCount,
                          ntaxa,
                          nsites,
                          manualScaling,
                          autoScaling,
                          dynamicScaling,
                          rateCategoryCount,
                          nreps,
                          fullTiming,
                          requireDoublePrecision,
                          requireSSE,
                          requireAVX,
                          compactTipCount,
                          randomSeed,
                          rescaleFrequency,
                          unrooted,
                          calcderivs,
                          logscalers,
                          eigenCount,
                          eigencomplex,
                          ievectrans,
                          setmatrix,
                          opencl,
                          partitions,
                          sitelikes,
                          newDataPerRep,
                          randomTree,
                          rerootTrees,
                          pectinate);
            }
        }
    } else {
        abort("no BEAGLE resources found");
    }

//#ifdef _WIN32
//    std::cout << "\nPress ENTER to exit...\n";
//    fflush( stdout);
//    fflush( stderr);
//    getchar();
//#endif
}
