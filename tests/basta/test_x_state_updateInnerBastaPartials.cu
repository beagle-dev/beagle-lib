//
// Created by gkarthik on 4/20/24.
//

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <libhmsbeagle/GPU/kernels/BeagleCUDA_kernels.h>

#define NPATTERNS 1

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a ## _ ## b

#define GET_VAR_NAME(VAR) CONCAT(VAR, STATE_SIZE)

#define KERNEL_STRING CONCAT(KERNELS_STRING_DP, STATE_SIZE)


double partials1Host_16[] = {0.137572756735608, 0.477254124591127, 0.428745979443192, 0.218426858074963, 0.529428300447762, 0.110051316907629, 0.000466231256723404, 0.912552231224254, 0.472219149814919, 0.928942704107612, 0.560180699685588, 0.217537123244256, 0.697977353585884, 0.623952043708414, 0.675538737559691, 0.843222103314474};
double partials2Host_16[] = {0.828149347100407, 0.864858906250447, 0.350309345172718, 0.155412578489631, 0.571178458165377, 0.0642696765717119, 0.767122565768659, 0.989166378509253, 0.807592692552134, 0.629633222240955, 0.759149922290817, 0.947871233569458, 0.514384098816663, 0.985400719335303, 0.437582379207015, 0.961291362997144};
double matrices_16[] = {0.318755098618567, 0.740407602395862, 0.874102377332747, 0.690571578918025, 0.701961038634181, 0.795170336496085, 0.348821813939139, 0.719763568369672, 0.895593842957169, 0.0806103076320142, 0.0329403523355722, 0.594381110742688, 0.359542798716575, 0.0881143372971565, 0.649336193455383, 0.699423802085221, 0.194982040207833, 0.0281000500544906, 0.771264882525429, 0.364646208938211, 0.0918229892849922, 0.936870703473687, 0.37309667118825, 0.0921725274529308, 0.159810511628166, 0.626376076601446, 0.692677799146622, 0.207317036110908, 0.970800938550383, 0.101070829667151, 0.944739747559652, 0.842726089060307, 0.207157449331135, 0.9937439779751, 0.821067644748837, 0.698300351854414, 0.782907243119553, 0.973109148675576, 0.77274006837979, 0.618896092521027, 0.300635955762118, 0.499129123752937, 0.656451256712899, 0.900395354721695, 0.56438325252384, 0.860600786982104, 0.49170877225697, 0.889088666066527, 0.0156871858052909, 0.81213018624112, 0.977085781516507, 0.295148188248277, 0.726798894582316, 0.916058843955398, 0.608332335250452, 0.93036122340709, 0.740293311886489, 0.477501145098358, 0.528508115326986, 0.359984828624874, 0.115733818151057, 0.162383440183476, 0.599637272302061, 0.727882955456153, 0.674989679362625, 0.862724364968017, 0.736116931773722, 0.934994351351634, 0.225678282091394, 0.0690643971320242, 0.642801651265472, 0.0847296256106347, 0.11830615834333, 0.203358805505559, 0.326086097629741, 0.730495651019737, 0.29963503475301, 0.561327135190368, 0.0538779601920396, 0.904581440845504, 0.807041098363698, 0.0369221286382526, 0.676678945310414, 0.814023691928014, 0.651314357761294, 0.901343589415774, 0.429449937539175, 0.628918903879821, 0.0323536104988307, 0.718082484090701, 0.4828297609929, 0.284520726883784, 0.997801034012809, 0.577156581915915, 0.48068075068295, 0.720765815814957, 0.403965384466574, 0.170624090125784, 0.876439346699044, 0.593427885789424, 0.769376888172701, 0.647851618006825, 0.337206641677767, 0.408452160190791, 0.0607802064623684, 0.676360174780712, 0.462037036195397, 0.347813759464771, 0.946434140671045, 0.604583449661732, 0.742167289135978, 0.0429284807760268, 0.626676751766354, 0.64561746106483, 0.673989933682606, 0.662993856705725, 0.856125113321468, 0.870688743656501, 0.308341189753264, 0.538020107662305, 0.538775310153142, 0.258829150116071, 0.407576418248937, 0.0590927223674953, 0.520942375296727, 0.434115749085322, 0.340059822658077, 0.17571780225262, 0.53029696107842, 0.975732607766986, 0.831993991043419, 0.983791405102238, 0.55990690854378, 0.710094312205911, 0.0787766603752971, 0.589558490784839, 0.673550385283306, 0.673261155607179, 0.974599071079865, 0.340651801321656, 0.0923955119214952, 0.194945273222402, 0.141258440678939, 0.639831262873486, 0.339301661355421, 0.884930776199326, 0.289516024757177, 0.126225155545399, 0.446420934982598, 0.0850053902249783, 0.395572019508108, 0.157721624244004, 0.929064428433776, 0.875803815200925, 0.36426582490094, 0.161793374456465, 0.387928539188579, 0.59461420099251, 0.174750925740227, 0.115827345522121, 0.152487984858453, 0.035088024334982, 0.983126569306478, 0.575697488617152, 0.209149486385286, 0.201207815203816, 0.110956045333296, 0.0174690247513354, 0.775130218127742, 0.497412796830758, 0.443602121900767, 0.0095906644128263, 0.198237520409748, 0.304748482536525, 0.518023134209216, 0.547018625540659, 0.526118801673874, 0.493327406002209, 0.913669281871989, 0.077805265551433, 0.929841819219291, 0.29471311555244, 0.118532110471278, 0.14501806977205, 0.436429607449099, 0.205097107682377, 0.708817657083273, 0.452689809259027, 0.0368115562014282, 0.239382016705349, 0.471294831018895, 0.290052818134427, 0.141418197192252, 0.406050314893946, 0.123278650688007, 0.793885040329769, 0.803798166336492, 0.653874792391434, 0.570221147732809, 0.238239917438477, 0.29717841069214, 0.753295891918242, 0.86618359782733, 0.0173538792878389, 0.262981676263735, 0.870678049977869, 0.235948005225509, 0.518082186812535, 0.80484891217202, 0.0546407853253186, 0.849502737633884, 0.749514716677368, 0.744097977643833, 0.543421939015388, 0.416765965521336, 0.725398944690824, 0.33459063549526, 0.478168275207281, 0.308823636500165, 0.518112531863153, 0.0382656576111913, 0.569260101066902, 0.542835577391088, 0.786958181066439, 0.191143112257123, 0.756994087249041, 0.930127918720245, 0.0538773911539465, 0.108583583729342, 0.0136260520666838, 0.0493806877639145, 0.0609419206157327, 0.157695269677788, 0.351834069937468, 0.396392825525254, 0.179786366643384, 0.822698512114584, 0.391177944373339, 0.0950508427340537, 0.0245800712145865, 0.28290240559727, 0.772152968682349, 0.14831333537586, 0.672689846949652, 0.826513405656442, 0.395902422023937, 0.837914811447263, 0.927084936527535, 0.0562630859203637, 0.810627286788076, 0.67597354692407, 0.622439709492028, 0.0950460624881089, 0.0380097457673401, 0.166366128250957, 0.867308233864605};
// double expectedPartials_16[] = {12.9769783870594, 25.8647148645348, 37.1028089862958, 26.553395775949, 28.2498807522581, 21.9503211762829, 12.9202534609119, 13.9146869398328, 14.9317655768219, 22.3371483393775, 22.9624904769671, 9.58813665982792, 11.4421566478761, 13.2079833456249, 11.7415865195237, 23.4516262593406};
// double sizes_16[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
double expectedPartials_16[] = {0.0305735894520103, 0.0609369261453154, 0.0437068636330068, 0.0417062995096076, 0.133112690863239, 0.103429332771742, 0.0608798925524519, 0.0655656368012336, 0.0703580844358418, 0.105252052132227, 0.10819864774354, 0.0451790463240147, 0.0539151394667653, 0.0311178340792188, 0.018442015033903, 0.0276258606529718};
double expectedPartials0_16[] = {2.98143795102797, 4.65854349159416, 4.91424944421801, 4.36942401534397, 4.40172642450781, 3.8340440457869, 3.24990363502535, 3.1248633106177, 3.32838756850764, 4.25163890981238, 4.16627182696517, 2.51423509758302, 2.89070331372981, 3.34836030429601, 2.60373247077971, 4.13703860672182};
double expectedPartials1_16[] = {4.35259046145336, 5.55210333684872, 7.55004592409327, 6.07709292636792, 6.4179092537349, 5.72510928777757, 3.97558048234593, 4.45289459303814, 4.48618595926221, 5.25377361840994, 5.51151999453036, 3.8135402170804, 3.95826046676252, 3.94461233119948, 4.50952110145465, 5.66869891454158};
double sizes_16[] = {2, 2, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4};
double coalescent_16[] = {2};


//double partials1Host_4[] = {1, 0, 0, 0};
//double partials2Host_4[] = {0, 0, 1, 0};
//double matrices_4[] = {0.9974538211778671, 5.991008993253103E-4, 0.0010484265738193166, 8.986513489883663E-4, 4.493256744940543E-4, 0.9976035964026984, 0.0010484265738197607, 8.986513489877002E-4, 4.49325674494082E-4, 5.991008993254314E-4, 0.9980529220771925, 8.986513489880332E-4, 4.49325674494082E-4, 5.991008993254456E-4, 0.0010484265738197607, 0.9979031468523611};
//double sizes_4[] = {2, 2, 4, 3};
//double expectedPartials_4[] = {0.8229333698706953, 3.7070898284225927E-4, 0.1764487818245675, 2.4713932189485474E-4};
//double expectedPartials0_4[] = {0.9974538211778671, 4.493256744940543E-4, 4.49325674494082E-4, 4.49325674494082E-4};
//double expectedPartials1_4[] = {0.0010484265738193166, 0.0010484265738197607, 0.9980529220771925, 0.0010484265738197607};
double expectedProb_16[] = {212.2253};

int main() {
    char functionName[] = "kernelInnerBastaPartialsCoalescent";
    CUmodule cudaModule;
    CUresult  res;
    cudaError_t err;

    cuInit(0);

    int numDevices;
    cuDeviceGetCount(&numDevices);
    std::cout << "Number of devices " << numDevices << std::endl;

    CUdevice cuDevice;
    res = cuDeviceGet(&cuDevice, 0);

    CUcontext context;
    res = cuCtxCreate(&context, 0, cuDevice);

    res = cuModuleLoadData(&cudaModule, KERNEL_STRING);

    CUfunction cudaFunction;

    res = cuModuleGetFunction(&cudaFunction, cudaModule, functionName);

    if(res != CUDA_SUCCESS) {
        std::cerr << "Error loading function " << functionName <<std::endl;
    } else {
        std::cerr << "Successfully loaded function " << functionName << " for state size " << STATE_SIZE << std::endl;
    }



    double *partials1, *partials2, *partials3, *matrices1, *matrices2, *accumulation1, *accumulation2, *sizes, *coalescent, *tmpAcc;
    int totalPatterns = NPATTERNS;
    int intervalNumber = 1;

    double *resHost = (double*) malloc(sizeof(double) * STATE_SIZE * NPATTERNS);
    double *accumHost1 = (double*) malloc(sizeof(double) * STATE_SIZE * NPATTERNS);
    double *accumHost2 = (double*) malloc(sizeof(double) * STATE_SIZE * NPATTERNS);
    float *coal = static_cast<float *>(malloc(sizeof(double) * intervalNumber));

    cuMemAlloc((CUdeviceptr*)&partials1, sizeof(double) * STATE_SIZE * NPATTERNS);
    cuMemAlloc((CUdeviceptr*)&partials2, sizeof(double) * STATE_SIZE * NPATTERNS);
    cuMemAlloc((CUdeviceptr*)&partials3, sizeof(double) * STATE_SIZE * NPATTERNS);
    cuMemAlloc((CUdeviceptr*)&accumulation1, sizeof(double) * STATE_SIZE * NPATTERNS);
    cuMemAlloc((CUdeviceptr*)&accumulation2, sizeof(double) * STATE_SIZE * NPATTERNS);
    cuMemAlloc((CUdeviceptr*)&sizes, sizeof(double) * STATE_SIZE * NPATTERNS);
    cuMemAlloc((CUdeviceptr*)&coalescent, sizeof(float) * intervalNumber);
    cuMemAlloc((CUdeviceptr*)&partials3, sizeof(double) * STATE_SIZE * NPATTERNS);
    cuMemAlloc((CUdeviceptr*)&tmpAcc, sizeof(double) * 1);

    cuMemAlloc((CUdeviceptr*)&matrices1, sizeof(double) * STATE_SIZE * STATE_SIZE);
    cuMemAlloc((CUdeviceptr*)&matrices2, sizeof(double) * STATE_SIZE * STATE_SIZE);

    cudaMemcpy(matrices1, GET_VAR_NAME(matrices), sizeof(double) * STATE_SIZE * STATE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(matrices2, GET_VAR_NAME(matrices), sizeof(double) * STATE_SIZE * STATE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(partials1, GET_VAR_NAME(partials1Host), sizeof(double) * STATE_SIZE * NPATTERNS, cudaMemcpyHostToDevice);
    cudaMemcpy(partials2, GET_VAR_NAME(partials2Host), sizeof(double) * STATE_SIZE * NPATTERNS, cudaMemcpyHostToDevice);
    cudaMemcpy(sizes, GET_VAR_NAME(sizes), sizeof(double) * STATE_SIZE * NPATTERNS, cudaMemcpyHostToDevice);
    cudaMemcpy(coalescent, GET_VAR_NAME(coalescent), sizeof(float) * intervalNumber, cudaMemcpyHostToDevice);
    void *params[] = {
            &partials1,  &partials2, &partials3, &matrices1, &matrices2, &accumulation1, &accumulation2, &sizes, &coalescent, &intervalNumber, &totalPatterns
    };

    res = cuLaunchKernel(cudaFunction, 1, 1, 1, STATE_SIZE, 8, 1, 0, NULL, params, NULL);

    cudaDeviceSynchronize();

    if(res != CUDA_SUCCESS) {
        std::cerr << "Error launching kernel " << functionName << ". Error code: " << res << std::endl;
    } else {
        std::cerr << "Successfully launched kernel " << functionName << " for state size " << STATE_SIZE << std::endl;
    }

    err = cudaMemcpy(resHost, partials3, NPATTERNS * STATE_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    std::cerr << "Memcpy partials3 " << err << std::endl;

    err = cudaMemcpy(accumHost1, accumulation1, NPATTERNS * STATE_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    std::cerr << "Memcpy accum1 " << err << std::endl;

    err = cudaMemcpy(accumHost2, accumulation2, NPATTERNS * STATE_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    std::cerr << "Memcpy accum2 " << err << std::endl;

    err = cudaMemcpy(coal, coalescent, intervalNumber * sizeof(float), cudaMemcpyDeviceToHost);
    std::cerr << "Memcpy coal " << err << std::endl;

    double tolerance=1E-5;
    int pass = 0;
    for(int i = 0; i < STATE_SIZE; i++){
        if(abs(GET_VAR_NAME(expectedPartials)[i] - resHost[i]) > tolerance) {
            if(pass == 0)
                std::cerr << "Unequal values at positions " << std::endl;
            pass = -1;
            std::cerr << i << " Expected: " << GET_VAR_NAME(expectedPartials)[i] << " Observed: " << resHost[i] << std::endl;
        }
    }

    if(pass != 0)
        std::cerr << std::endl << std::endl;

    if(pass == 0) {
        std::cerr << "PASS: Test for post-order kernel for state size " << STATE_SIZE << std::endl;
    } else {
        std::cerr << "FAIL: Test for post-order kernel for state size " << STATE_SIZE << std::endl;
    }

    int pass0 = 0;
    for(int i = 0; i < STATE_SIZE; i++){
        if(abs(GET_VAR_NAME(expectedPartials0)[i] - accumHost1[i]) > tolerance) {
            if(pass0 == 0)
                std::cerr << "Unequal values at positions " << std::endl;
            pass0 = -1;
            std::cerr << i << " Expected: " << GET_VAR_NAME(expectedPartials0)[i] << " Observed: " << accumHost1[i] << std::endl;
        }
    }

    if(pass0 != 0)
        std::cerr << std::endl << std::endl;

    if(pass0 == 0) {
        std::cerr << "PASS0: Test for post-order kernel for state size " << STATE_SIZE << std::endl;
    } else {
        std::cerr << "FAIL0: Test for post-order kernel for state size " << STATE_SIZE << std::endl;
    }


    int pass1 = 0;
    for(int i = 0; i < STATE_SIZE; i++){
        if(abs(GET_VAR_NAME(expectedPartials1)[i] - accumHost2[i]) > tolerance) {
            if(pass1 == 0)
                std::cerr << "Unequal values at positions " << std::endl;
            pass1 = -1;
            std::cerr << i << " Expected: " << GET_VAR_NAME(expectedPartials1)[i] << " Observed: " << accumHost2[i] << std::endl;
        }
    }

    if(pass1 != 0)
        std::cerr << std::endl << std::endl;

    if(pass1 == 0) {
        std::cerr << "PASS1: Test for post-order kernel for state size " << STATE_SIZE << std::endl;
    } else {
        std::cerr << "FAIL1: Test for post-order kernel for state size " << STATE_SIZE << std::endl;
    }


    int pass2 = 0;
    for(int i = 0; i < intervalNumber; i++){
        if(abs(GET_VAR_NAME(expectedProb)[i] - coal[i]) > tolerance) {
            if(pass2 == 0)
                std::cerr << "Unequal values at positions " << std::endl;
            pass2 = -1;
            std::cerr << i << " Expected: " << GET_VAR_NAME(expectedProb)[i] << " Observed: " << coal[i] << std::endl;
        }
    }

    if(pass2 != 0)
        std::cerr << std::endl << std::endl;

    if(pass2 == 0) {
        std::cerr << "PASS2: Test for post-order kernel for state size " << STATE_SIZE << std::endl;
    } else {
        std::cerr << "FAIL2: Test for post-order kernel for state size " << STATE_SIZE << std::endl;
    }

    cuModuleUnload(cudaModule);

    cudaFree(matrices1);
    cudaFree(matrices2);
    cudaFree(partials1);
    cudaFree(partials2);
    cudaFree(partials3);
    cudaFree(accumulation1);
    cudaFree(accumulation2);
    cudaFree(sizes);
    cudaFree(coalescent);
    free(resHost);
    free(accumHost1);
    free(accumHost2);
    free(coal);

    return pass;
}