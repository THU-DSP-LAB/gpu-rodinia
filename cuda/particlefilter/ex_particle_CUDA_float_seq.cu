#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <sys/time.h>
#include <time.h>
#include "../../common/rodinia_verify.h"
#define BLOCK_X 16
#define BLOCK_Y 16
constexpr int threads_per_block = 512;
constexpr int normal_uniform_samples = 12;
constexpr float normal_uniform_mean = 6.0F;

#define PARTICLEFILTER_FLOAT_REFERENCE_MAGIC "GPIDL_RODINIA_PARTICLEFILTER_FLOAT_REFERENCE"
#define PARTICLEFILTER_FLOAT_REFERENCE_VERSION 2
#define PARTICLEFILTER_FLOAT_ABS_TOLERANCE 1.0e-6F
#define PARTICLEFILTER_FLOAT_REL_TOLERANCE 1.0e-6F

typedef struct {
    int seed_base;
    int seed_set;
    const char *save_reference_path;
    const char *verify_reference_path;
} ParticlefilterFloatOptions;

typedef struct {
    int dim_x;
    int dim_y;
    int frames;
    int particles;
    int seed_base;
    float xe;
    float ye;
    float distance;
} ParticlefilterFloatReference;

static int parse_int_argument(const char *text, const char *name, int *value) {
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX) {
        fprintf(stderr, "Invalid %s: %s\n", name, text);
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static int parse_reference_options(int argc, char **argv, ParticlefilterFloatOptions *options) {
    options->seed_base = 0;
    options->seed_set = 0;
    options->save_reference_path = NULL;
    options->verify_reference_path = NULL;

    for (int arg = 9; arg < argc; arg++) {
        if (strcmp(argv[arg], "--seed") == 0) {
            if (++arg >= argc || parse_int_argument(argv[arg], "seed", &options->seed_base) != 0) {
                return -1;
            }
            options->seed_set = 1;
            continue;
        }
        if (strcmp(argv[arg], "--save-reference") == 0) {
            if (++arg >= argc) {
                fprintf(stderr, "Missing path after --save-reference\n");
                return -1;
            }
            if (options->verify_reference_path != NULL) {
                fprintf(stderr, "Only one particlefilter_float reference mode may be specified\n");
                return -1;
            }
            options->save_reference_path = argv[arg];
            continue;
        }
        if (strcmp(argv[arg], "--verify-reference") == 0) {
            if (++arg >= argc) {
                fprintf(stderr, "Missing path after --verify-reference\n");
                return -1;
            }
            if (options->save_reference_path != NULL) {
                fprintf(stderr, "Only one particlefilter_float reference mode may be specified\n");
                return -1;
            }
            options->verify_reference_path = argv[arg];
            continue;
        }
        fprintf(stderr, "Unknown option: %s\n", argv[arg]);
        return -1;
    }

    if ((options->save_reference_path != NULL || options->verify_reference_path != NULL) &&
        !options->seed_set) {
        fprintf(stderr, "particlefilter_float reference modes require --seed for reproducible input\n");
        return -1;
    }
    return 0;
}

static int save_reference(const char *path, const ParticlefilterFloatReference *reference) {
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        fprintf(stderr, "Cannot open particlefilter_float reference for write: %s\n", path);
        return -1;
    }
    fprintf(file, "%s %d\n", PARTICLEFILTER_FLOAT_REFERENCE_MAGIC, PARTICLEFILTER_FLOAT_REFERENCE_VERSION);
    fprintf(file, "dim_x %d\n", reference->dim_x);
    fprintf(file, "dim_y %d\n", reference->dim_y);
    fprintf(file, "frames %d\n", reference->frames);
    fprintf(file, "particles %d\n", reference->particles);
    fprintf(file, "seed_base %d\n", reference->seed_base);
    fprintf(file, "xe %.9g\n", reference->xe);
    fprintf(file, "ye %.9g\n", reference->ye);
    fprintf(file, "distance %.9g\n", reference->distance);
    if (fclose(file) != 0) {
        fprintf(stderr, "Failed closing particlefilter_float reference: %s\n", path);
        return -1;
    }
    printf("Saved particlefilter_float reference to '%s'\n", path);
    return 0;
}

static int scan_reference(FILE *file, const char *label, const char *format, void *value) {
    char actual_label[64];
    if (fscanf(file, "%63s", actual_label) != 1 || strcmp(actual_label, label) != 0 ||
        fscanf(file, format, value) != 1) {
        fprintf(stderr, "Invalid particlefilter_float reference field: %s\n", label);
        return -1;
    }
    return 0;
}

static int read_reference(const char *path, ParticlefilterFloatReference *reference) {
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        fprintf(stderr, "Cannot open particlefilter_float reference for read: %s\n", path);
        return -1;
    }
    char magic[128];
    int version = 0;
    if (fscanf(file, "%127s %d", magic, &version) != 2 ||
        strcmp(magic, PARTICLEFILTER_FLOAT_REFERENCE_MAGIC) != 0 ||
        version != PARTICLEFILTER_FLOAT_REFERENCE_VERSION) {
        fprintf(stderr, "Invalid particlefilter_float reference header: %s\n", path);
        fclose(file);
        return -1;
    }
    int failed = 0;
    failed |= scan_reference(file, "dim_x", "%d", &reference->dim_x);
    failed |= scan_reference(file, "dim_y", "%d", &reference->dim_y);
    failed |= scan_reference(file, "frames", "%d", &reference->frames);
    failed |= scan_reference(file, "particles", "%d", &reference->particles);
    failed |= scan_reference(file, "seed_base", "%d", &reference->seed_base);
    failed |= scan_reference(file, "xe", "%f", &reference->xe);
    failed |= scan_reference(file, "ye", "%f", &reference->ye);
    failed |= scan_reference(file, "distance", "%f", &reference->distance);
    fclose(file);
    return failed == 0 ? 0 : -1;
}

static int compare_float_field(const char *field, float actual, float expected) {
    float diff = fabsf(actual - expected);
    float tolerance = PARTICLEFILTER_FLOAT_ABS_TOLERANCE +
        PARTICLEFILTER_FLOAT_REL_TOLERANCE * fabsf(expected);
    if (isfinite(actual) && isfinite(expected) && diff <= tolerance) {
        return 0;
    }
    fprintf(stderr,
            "particlefilter_float reference mismatch for %s: actual=%.9g expected=%.9g diff=%.9g tolerance=%.9g\n",
            field,
            actual,
            expected,
            diff,
            tolerance);
    return -1;
}

static int verify_reference(const char *path, const ParticlefilterFloatReference *actual) {
    ParticlefilterFloatReference expected;
    if (read_reference(path, &expected) != 0) {
        rodinia_print_fail("Particlefilter float reference verification");
        return -1;
    }
    int failed = 0;
    if (actual->dim_x != expected.dim_x || actual->dim_y != expected.dim_y ||
        actual->frames != expected.frames || actual->particles != expected.particles ||
        actual->seed_base != expected.seed_base) {
        fprintf(stderr,
                "particlefilter_float reference task mismatch: actual=%dx%dx%d np=%d seed=%d expected=%dx%dx%d np=%d seed=%d\n",
                actual->dim_x,
                actual->dim_y,
                actual->frames,
                actual->particles,
                actual->seed_base,
                expected.dim_x,
                expected.dim_y,
                expected.frames,
                expected.particles,
                expected.seed_base);
        failed = -1;
    }
    failed |= compare_float_field("xe", actual->xe, expected.xe);
    failed |= compare_float_field("ye", actual->ye, expected.ye);
    failed |= compare_float_field("distance", actual->distance, expected.distance);
    if (failed != 0) {
        rodinia_print_fail("Particlefilter float reference verification");
        return -1;
    }
    rodinia_print_pass("Particlefilter float reference verification");
    return 0;
}

/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
 */
long M = INT_MAX;
/**
@var A value for LCG
 */
int A = 1103515245;
/**
@var C value for LCG
 */
int C = 12345;

/*****************************
 *GET_TIME
 *returns a long int representing the time
 *****************************/
long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) +tv.tv_usec;
}
// Returns the number of seconds elapsed between the two specified times

float elapsed_time(long long start_time, long long end_time) {
    return (float) (end_time - start_time) / 1000000.0F;
}

/*****************************
 * CHECK_ERROR
 * Checks for CUDA errors and prints them to the screen to help with
 * debugging of CUDA related programming
 *****************************/
void check_error(cudaError e) {
    if (e != cudaSuccess) {
        printf("\nCUDA error: %s\n", cudaGetErrorString(e));
        exit(1);
    }
}

void cuda_print_float_array(float *array_GPU, size_t size) {
    //allocate temporary array for printing
    float* mem = (float*) malloc(sizeof (float) *size);

    //transfer data from device
    cudaMemcpy(mem, array_GPU, sizeof (float) *size, cudaMemcpyDeviceToHost);


    printf("PRINTING ARRAY VALUES\n");
    //print values in memory
    for (size_t i = 0; i < size; ++i) {
        printf("[%zu]:%0.6f\n", i, mem[i]);
    }
    printf("FINISHED PRINTING ARRAY VALUES\n");

    //clean up memory
    free(mem);
    mem = NULL;
}

/********************************
 * CALC LIKELIHOOD SUM
 * DETERMINES THE LIKELIHOOD SUM BASED ON THE FORMULA: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
 * param 1 I 3D matrix
 * param 2 current ind array
 * param 3 length of ind array
 * returns a float representing the sum
 ********************************/
__device__ float calcLikelihoodSum(unsigned char * I, int * ind, int numOnes, int index) {
    float likelihoodSum = 0.0F;
    int x;
    for (x = 0; x < numOnes; x++) {
        const float foreground_delta = (float) (I[ind[index * numOnes + x]] - 100);
        const float background_delta = (float) (I[ind[index * numOnes + x]] - 228);
        likelihoodSum +=
            (foreground_delta * foreground_delta - background_delta * background_delta) / 50.0F;
    }
    return likelihoodSum;
}

/****************************
CDF CALCULATE
CALCULATES CDF
param1 CDF
param2 weights
param3 Nparticles
 *****************************/
__device__ void cdfCalc(float * CDF, float * weights, int Nparticles) {
    int x;
    CDF[0] = weights[0];
    for (x = 1; x < Nparticles; x++) {
        CDF[x] = weights[x] + CDF[x - 1];
    }
}

/*****************************
 * RANDU
 * GENERATES A UNIFORM DISTRIBUTION
 * returns a float representing a randomily generated number from a uniform distribution with range [0, 1)
 ******************************/
__device__ float d_randu(int * seed, int index) {

    int M = INT_MAX;
    int A = 1103515245;
    int C = 12345;
    int num = A * seed[index] + C;
    seed[index] = num % M;

    return fabsf(seed[index] / ((float) M));
}/**
* Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
* @see http://en.wikipedia.org/wiki/Linear_congruential_generator
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a uniformly distributed number [0, 1)
*/

float randu(int * seed, int index) {
    int num = A * seed[index] + C;
    seed[index] = num % M;
    return fabsf(seed[index] / ((float) M));
}

/**
 * Approximates a standard normal variate with the centered sum of 12 uniform variates.
 * The fixed FP32 addition path is reproducible across CUDA and functional-simulator backends.
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a float representing random number generated using the Box-Muller algorithm
 * @see https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution
 */
float randn(int * seed, int index) {
    float sample = 0.0F;
    for (int i = 0; i < normal_uniform_samples; ++i) {
        sample += randu(seed, index);
    }
    return sample - normal_uniform_mean;
}

float test_randn(int * seed, int index) {
    return randn(seed, index);
}

__device__ float d_randn(int * seed, int index) {
    float sample = 0.0F;
    for (int i = 0; i < normal_uniform_samples; ++i) {
        sample += d_randu(seed, index);
    }
    return sample - normal_uniform_mean;
}

/****************************
UPDATE WEIGHTS
UPDATES WEIGHTS
param1 weights
param2 likelihood
param3 Nparcitles
 ****************************/
__device__ float updateWeights(float * weights, float * likelihood, int Nparticles) {
    int x;
    float sum = 0;
    for (x = 0; x < Nparticles; x++) {
        weights[x] = weights[x] * expf(likelihood[x]);
        sum += weights[x];
    }
    return sum;
}

__device__ int findIndexBin(float * CDF, int beginIndex, int endIndex, float value) {
    if (endIndex < beginIndex)
        return -1;
    int middleIndex;
    while (endIndex > beginIndex) {
        middleIndex = beginIndex + ((endIndex - beginIndex) / 2);
        if (CDF[middleIndex] >= value) {
            if (middleIndex == 0)
                return middleIndex;
            else if (CDF[middleIndex - 1] < value)
                return middleIndex;
            else if (CDF[middleIndex - 1] == value) {
                while (CDF[middleIndex] == value && middleIndex >= 0)
                    middleIndex--;
                middleIndex++;
                return middleIndex;
            }
        }
        if (CDF[middleIndex] > value)
            endIndex = middleIndex - 1;
        else
            beginIndex = middleIndex + 1;
    }
    return -1;
}

/** added this function. was missing in original float version.
 * Takes in a float and returns an integer that approximates to that float
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
__device__ float dev_round_float(float value) {
    int newValue = (int) (value);
    if (value - newValue < .5f)
        return newValue;
    else
        return newValue++;
}

/*****************************
 * CUDA Find Index Kernel Function to replace FindIndex
 * param1: arrayX
 * param2: arrayY
 * param3: CDF
 * param4: u
 * param5: xj
 * param6: yj
 * param7: weights
 * param8: Nparticles
 *****************************/
__global__ void find_index_kernel(float * arrayX, float * arrayY, float * CDF, float * u, float * xj, float * yj, float * weights, int Nparticles) {
    int block_id = blockIdx.x;
    int i = blockDim.x * block_id + threadIdx.x;

    if (i < Nparticles) {

        int index = -1;
        int x;

        for (x = 0; x < Nparticles; x++) {
            if (CDF[x] >= u[i]) {
                index = x;
                break;
            }
        }
        if (index == -1) {
            index = Nparticles - 1;
        }

        xj[i] = arrayX[index];
        yj[i] = arrayY[index];

        //weights[i] = 1 / ((float) (Nparticles)); //moved this code to the beginning of likelihood kernel

    }
    __syncthreads();
}

__global__ void normalize_weights_kernel(
        float * weights,
        const float * likelihood,
        int Nparticles) {
    __shared__ float reduction[threads_per_block];
    const int tid = threadIdx.x;

    float max_likelihood = -FLT_MAX;
    for (int i = tid; i < Nparticles; i += blockDim.x) {
        max_likelihood = fmaxf(max_likelihood, likelihood[i]);
    }
    reduction[tid] = max_likelihood;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
        }
        __syncthreads();
    }
    max_likelihood = reduction[0];

    float sum = 0.0F;
    for (int i = tid; i < Nparticles; i += blockDim.x) {
        weights[i] = expf(likelihood[i] - max_likelihood);
        sum += weights[i];
    }
    reduction[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
    }

    const float weight_sum = reduction[0];
    for (int i = tid; i < Nparticles; i += blockDim.x) {
        weights[i] /= weight_sum;
    }
    __syncthreads();
}

__global__ void prepare_resampling_kernel(float * weights, int Nparticles, float * CDF, float * u, int * seed) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        cdfCalc(CDF, weights, Nparticles);
        u[0] = (1 / ((float) (Nparticles))) * d_randu(seed, 0);
    }
}

__global__ void initialize_resampling_offsets_kernel(float * u, int Nparticles) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < Nparticles) {
        u[i] = u[0] + i / ((float) (Nparticles));
    }
}

/*****************************
 * CUDA Likelihood Kernel Function to replace FindIndex
 * param1: arrayX
 * param2: arrayY
 * param3: ind
 * param4: objxy
 * param5: likelihood
 * param6: I
 * param7: Nparticles
 * param8: countOnes
 * param9: max_size
 * param10: k
 * param11: IszY
 * param12: Nfr
 *****************************/
__global__ void likelihood_kernel(
        float * arrayX,
        float * arrayY,
        float * xj,
        float * yj,
        int * ind,
        int * objxy,
        float * likelihood,
        unsigned char * I,
        int Nparticles,
        int countOnes,
        int max_size,
        int k,
        int IszY,
        int Nfr,
        int * seed) {
    int block_id = blockIdx.x;
    int i = blockDim.x * block_id + threadIdx.x;
    int y;

    int indX, indY;
    if (i < Nparticles) {
        arrayX[i] = xj[i];
        arrayY[i] = yj[i];
        arrayX[i] = arrayX[i] + 1.0F + 5.0F * d_randn(seed, i);
        arrayY[i] = arrayY[i] - 2.0F + 2.0F * d_randn(seed, i);
        for (y = 0; y < countOnes; y++) {
            //added dev_round_float() to be consistent with roundFloat
            indX = dev_round_float(arrayX[i]) + objxy[y * 2 + 1];
            indY = dev_round_float(arrayY[i]) + objxy[y * 2];

            ind[i * countOnes + y] = abs(indX * IszY * Nfr + indY * Nfr + k);
            if (ind[i * countOnes + y] >= max_size)
                ind[i * countOnes + y] = 0;
        }
        likelihood[i] = calcLikelihoodSum(I, ind, countOnes, i);

        likelihood[i] = likelihood[i] / countOnes;
    }
}

/**
 * Takes in a float and returns an integer that approximates to that float
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
float roundFloat(float value) {
    int newValue = (int) (value);
    if (value - newValue < .5F)
        return newValue;
    else
        return newValue++;
}

/**
 * Set values of the 3D array to a newValue if that value is equal to the testValue
 * @param testValue The value to be replaced
 * @param newValue The value to replace testValue with
 * @param array3D The image vector
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 */
void setIf(int testValue, int newValue, unsigned char * array3D, int * dimX, int * dimY, int * dimZ) {
    int x, y, z;
    for (x = 0; x < *dimX; x++) {
        for (y = 0; y < *dimY; y++) {
            for (z = 0; z < *dimZ; z++) {
                if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
                    array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
            }
        }
    }
}

/**
 * Sets values of 3D matrix using randomly generated numbers from a normal distribution
 * @param array3D The video to be modified
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param seed The seed array
 */
void addNoise(unsigned char * array3D, int * dimX, int * dimY, int * dimZ, int * seed) {
    int x, y, z;
    for (x = 0; x < *dimX; x++) {
        for (y = 0; y < *dimY; y++) {
            for (z = 0; z < *dimZ; z++) {
                array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (unsigned char) (5 * randn(seed, 0));
            }
        }
    }
}

/**
 * Fills a radius x radius matrix representing the disk
 * @param disk The pointer to the disk to be made
 * @param radius  The radius of the disk to be made
 */
void strelDisk(int * disk, int radius) {
    int diameter = radius * 2 - 1;
    int x, y;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            float distance = sqrtf(powf((float) (x - radius + 1), 2) + powf((float) (y - radius + 1), 2));
            if (distance < radius)
                disk[x * diameter + y] = 1;
            else
                disk[x * diameter + y] = 0;
        }
    }
}

/**
 * Dilates the provided video
 * @param matrix The video to be dilated
 * @param posX The x location of the pixel to be dilated
 * @param posY The y location of the pixel to be dilated
 * @param poxZ The z location of the pixel to be dilated
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param error The error radius
 */
void dilate_matrix(unsigned char * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
    int startX = posX - error;
    while (startX < 0)
        startX++;
    int startY = posY - error;
    while (startY < 0)
        startY++;
    int endX = posX + error;
    while (endX > dimX)
        endX--;
    int endY = posY + error;
    while (endY > dimY)
        endY--;
    int x, y;
    for (x = startX; x < endX; x++) {
        for (y = startY; y < endY; y++) {
            float distance = sqrtf(powf((float) (x - posX), 2) + powf((float) (y - posY), 2));
            if (distance < error)
                matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
        }
    }
}

/**
 * Dilates the target matrix using the radius as a guide
 * @param matrix The reference matrix
 * @param dimX The x dimension of the video
 * @param dimY The y dimension of the video
 * @param dimZ The z dimension of the video
 * @param error The error radius to be dilated
 * @param newMatrix The target matrix
 */
void imdilate_disk(unsigned char * matrix, int dimX, int dimY, int dimZ, int error, unsigned char * newMatrix) {
    int x, y, z;
    for (z = 0; z < dimZ; z++) {
        for (x = 0; x < dimX; x++) {
            for (y = 0; y < dimY; y++) {
                if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
                    dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
                }
            }
        }
    }
}

/**
 * Fills a 2D array describing the offsets of the disk object
 * @param se The disk object
 * @param numOnes The number of ones in the disk
 * @param neighbors The array that will contain the offsets
 * @param radius The radius used for dilation
 */
void getneighbors(int * se, int numOnes, int * neighbors, int radius) {
    int x, y;
    int neighY = 0;
    int center = radius - 1;
    int diameter = radius * 2 - 1;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            if (se[x * diameter + y]) {
                neighbors[neighY * 2] = (int) (y - center);
                neighbors[neighY * 2 + 1] = (int) (x - center);
                neighY++;
            }
        }
    }
}

/**
 * The synthetic video sequence we will work with here is composed of a
 * single moving object, circular in shape (fixed radius)
 * The motion here is a linear motion
 * the foreground intensity and the background intensity is known
 * the image is corrupted with zero mean Gaussian noise
 * @param I The video itself
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames of the video
 * @param seed The seed array used for number generation
 */
void videoSequence(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) {
    int k;
    int max_size = IszX * IszY * Nfr;
    /*get object centers*/
    int x0 = (int) roundFloat(IszY / 2.0F);
    int y0 = (int) roundFloat(IszX / 2.0F);
    I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

    /*move point*/
    int xk, yk, pos;
    for (k = 1; k < Nfr; k++) {
        xk = abs(x0 + (k-1));
        yk = abs(y0 - 2 * (k-1));
        pos = yk * IszY * Nfr + xk * Nfr + k;
        if (pos >= max_size)
            pos = 0;
        I[pos] = 1;
    }

    /*dilate matrix*/
    unsigned char * newMatrix = (unsigned char *) malloc(sizeof (unsigned char) * IszX * IszY * Nfr);
    imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
    int x, y;
    for (x = 0; x < IszX; x++) {
        for (y = 0; y < IszY; y++) {
            for (k = 0; k < Nfr; k++) {
                I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
            }
        }
    }
    free(newMatrix);

    /*define background, add noise*/
    setIf(0, 100, I, &IszX, &IszY, &Nfr);
    setIf(1, 228, I, &IszX, &IszY, &Nfr);
    /*add noise*/
    addNoise(I, &IszX, &IszY, &Nfr, seed);

}

/**
 * Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
 * @note This function uses sequential search
 * @param CDF The CDF
 * @param lengthCDF The length of CDF
 * @param value The value to be found
 * @return The index of value in the CDF; if value is never found, returns the last index
 */
int findIndex(float * CDF, int lengthCDF, float value) {
    int index = -1;
    int x;
    for (x = 0; x < lengthCDF; x++) {
        if (CDF[x] >= value) {
            index = x;
            break;
        }
    }
    if (index == -1) {
        return lengthCDF - 1;
    }
    return index;
}

/**
 * The implementation of the particle filter using OpenMP for many frames
 * @see http://openmp.org/wp/
 * @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
 * @param I The video to be run
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames
 * @param seed The seed array used for random number generation
 * @param Nparticles The number of particles to be used
 */
void particleFilter(
        unsigned char * I,
        int IszX,
        int IszY,
        int Nfr,
        int * seed,
        int Nparticles,
        ParticlefilterFloatReference *reference) {
    int max_size = IszX * IszY*Nfr;
    //original particle centroid
    float xe = roundFloat(IszY / 2.0F);
    float ye = roundFloat(IszX / 2.0F);

    //expected object locations, compared to center
    int radius = 5;
    int diameter = radius * 2 - 1;
    int * disk = (int*) malloc(diameter * diameter * sizeof (int));
    strelDisk(disk, radius);
    int countOnes = 0;
    int x, y;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            if (disk[x * diameter + y] == 1)
                countOnes++;
        }
    }
    int * objxy = (int *) malloc(countOnes * 2 * sizeof (int));
    getneighbors(disk, countOnes, objxy, radius);
    //initial weights are all equal (1/Nparticles)
    float * weights = (float *) malloc(sizeof (float) *Nparticles);
    for (x = 0; x < Nparticles; x++) {
        weights[x] = 1 / ((float) (Nparticles));
    }

    //initial likelihood to 0.0
    float * likelihood = (float *) malloc(sizeof (float) *Nparticles);
    float * arrayX = (float *) malloc(sizeof (float) *Nparticles);
    float * arrayY = (float *) malloc(sizeof (float) *Nparticles);
    float * xj = (float *) malloc(sizeof (float) *Nparticles);
    float * yj = (float *) malloc(sizeof (float) *Nparticles);
    float * CDF = (float *) malloc(sizeof (float) *Nparticles);

    //GPU copies of arrays
    float * arrayX_GPU;
    float * arrayY_GPU;
    float * xj_GPU;
    float * yj_GPU;
    float * CDF_GPU;
    float * likelihood_GPU;
    unsigned char * I_GPU;
    float * weights_GPU;
    int * objxy_GPU;

    int * ind = (int*) malloc(sizeof (int) *countOnes * Nparticles);
    int * ind_GPU;
    float * u = (float *) malloc(sizeof (float) *Nparticles);
    float * u_GPU;
    int * seed_GPU;

    //CUDA memory allocation
    check_error(cudaMalloc((void **) &arrayX_GPU, sizeof (float) *Nparticles));
    check_error(cudaMalloc((void **) &arrayY_GPU, sizeof (float) *Nparticles));
    check_error(cudaMalloc((void **) &xj_GPU, sizeof (float) *Nparticles));
    check_error(cudaMalloc((void **) &yj_GPU, sizeof (float) *Nparticles));
    check_error(cudaMalloc((void **) &CDF_GPU, sizeof (float) *Nparticles));
    check_error(cudaMalloc((void **) &u_GPU, sizeof (float) *Nparticles));
    check_error(cudaMalloc((void **) &likelihood_GPU, sizeof (float) *Nparticles));
    //set likelihood to zero
    check_error(cudaMemset((void *) likelihood_GPU, 0, sizeof (float) *Nparticles));
    check_error(cudaMalloc((void **) &weights_GPU, sizeof (float) *Nparticles));
    check_error(cudaMalloc((void **) &I_GPU, sizeof (unsigned char) *IszX * IszY * Nfr));
    check_error(cudaMalloc((void **) &objxy_GPU, sizeof (int) *2 * countOnes));
    check_error(cudaMalloc((void **) &ind_GPU, sizeof (int) *countOnes * Nparticles));
    check_error(cudaMalloc((void **) &seed_GPU, sizeof (int) *Nparticles));


    //Donnie - this loop is different because in this kernel, arrayX and arrayY
    //  are set equal to xj before every iteration, so effectively, arrayX and
    //  arrayY will be set to xe and ye before the first iteration.
    for (x = 0; x < Nparticles; x++) {

        xj[x] = xe;
        yj[x] = ye;

    }

    int k;
    //start send
    long long send_start = get_time();
    check_error(cudaMemcpy(I_GPU, I, sizeof (unsigned char) *IszX * IszY*Nfr, cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(objxy_GPU, objxy, sizeof (int) *2 * countOnes, cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(weights_GPU, weights, sizeof (float) *Nparticles, cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(xj_GPU, xj, sizeof (float) *Nparticles, cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(yj_GPU, yj, sizeof (float) *Nparticles, cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(seed_GPU, seed, sizeof (int) *Nparticles, cudaMemcpyHostToDevice));
    long long send_end = get_time();
    printf("TIME TO SEND TO GPU: %f\n", elapsed_time(send_start, send_end));
    int num_blocks = (Nparticles + threads_per_block - 1) / threads_per_block;


    for (k = 1; k < Nfr; k++) {

        likelihood_kernel << < num_blocks, threads_per_block >> > (
            arrayX_GPU,
            arrayY_GPU,
            xj_GPU,
            yj_GPU,
            ind_GPU,
            objxy_GPU,
            likelihood_GPU,
            I_GPU,
            Nparticles,
            countOnes,
            max_size,
            k,
            IszY,
            Nfr,
            seed_GPU);

        normalize_weights_kernel << < 1, threads_per_block >> > (weights_GPU, likelihood_GPU, Nparticles);

        prepare_resampling_kernel << < 1, 1 >> > (weights_GPU, Nparticles, CDF_GPU, u_GPU, seed_GPU);

        initialize_resampling_offsets_kernel << < num_blocks, threads_per_block >> > (u_GPU, Nparticles);

        find_index_kernel << < num_blocks, threads_per_block >> > (arrayX_GPU, arrayY_GPU, CDF_GPU, u_GPU, xj_GPU, yj_GPU, weights_GPU, Nparticles);

    }//end loop

    //block till kernels are finished
    cudaDeviceSynchronize();
    long long back_time = get_time();

    cudaFree(xj_GPU);
    cudaFree(yj_GPU);
    cudaFree(CDF_GPU);
    cudaFree(u_GPU);
    cudaFree(likelihood_GPU);
    cudaFree(I_GPU);
    cudaFree(objxy_GPU);
    cudaFree(ind_GPU);
    cudaFree(seed_GPU);

    long long free_time = get_time();
    check_error(cudaMemcpy(arrayX, arrayX_GPU, sizeof (float) *Nparticles, cudaMemcpyDeviceToHost));
    long long arrayX_time = get_time();
    check_error(cudaMemcpy(arrayY, arrayY_GPU, sizeof (float) *Nparticles, cudaMemcpyDeviceToHost));
    long long arrayY_time = get_time();
    check_error(cudaMemcpy(weights, weights_GPU, sizeof (float) *Nparticles, cudaMemcpyDeviceToHost));
    long long back_end_time = get_time();
    printf("GPU Execution: %f\n", elapsed_time(send_end, back_time));
    printf("FREE TIME: %f\n", elapsed_time(back_time, free_time));
    printf("TIME TO SEND BACK: %f\n", elapsed_time(back_time, back_end_time));
    printf("SEND ARRAY X BACK: %f\n", elapsed_time(free_time, arrayX_time));
    printf("SEND ARRAY Y BACK: %f\n", elapsed_time(arrayX_time, arrayY_time));
    printf("SEND WEIGHTS BACK: %f\n", elapsed_time(arrayY_time, back_end_time));

    xe = 0;
    ye = 0;
    // estimate the object location by expected values
    for (x = 0; x < Nparticles; x++) {
        xe += arrayX[x] * weights[x];
        ye += arrayY[x] * weights[x];
    }
    printf("XE: %f\n", xe);
    printf("YE: %f\n", ye);
    const float center_x_delta = xe - (int) roundFloat(IszY / 2.0F);
    const float center_y_delta = ye - (int) roundFloat(IszX / 2.0F);
    float distance = sqrtf(center_x_delta * center_x_delta + center_y_delta * center_y_delta);
    printf("%f\n", distance);
    reference->dim_x = IszX;
    reference->dim_y = IszY;
    reference->frames = Nfr;
    reference->particles = Nparticles;
    reference->xe = xe;
    reference->ye = ye;
    reference->distance = distance;

    //CUDA freeing of memory
    cudaFree(weights_GPU);
    cudaFree(arrayY_GPU);
    cudaFree(arrayX_GPU);

    //free regular memory
    free(likelihood);
    free(arrayX);
    free(arrayY);
    free(xj);
    free(yj);
    free(CDF);
    free(ind);
    free(u);
}

int main(int argc, char * argv[]) {

    const char* usage = "float.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles> [--seed <int>] [--save-reference <path>|--verify-reference <path>]";
    //check number of arguments
    if (argc < 9) {
        printf("%s\n", usage);
        return EXIT_FAILURE;
    }
    ParticlefilterFloatOptions options;
    if (parse_reference_options(argc, argv, &options) != 0) {
        printf("%s\n", usage);
        return EXIT_FAILURE;
    }
    //check args deliminators
    if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") || strcmp(argv[7], "-np")) {
        printf("%s\n", usage);
        return EXIT_FAILURE;
    }

    int IszX, IszY, Nfr, Nparticles;

    //converting a string to a integer
    if (sscanf(argv[2], "%d", &IszX) == EOF) {
        printf("ERROR: dimX input is incorrect");
        return EXIT_FAILURE;
    }

    if (IszX <= 0) {
        printf("dimX must be > 0\n");
        return EXIT_FAILURE;
    }

    //converting a string to a integer
    if (sscanf(argv[4], "%d", &IszY) == EOF) {
        printf("ERROR: dimY input is incorrect");
        return EXIT_FAILURE;
    }

    if (IszY <= 0) {
        printf("dimY must be > 0\n");
        return EXIT_FAILURE;
    }

    //converting a string to a integer
    if (sscanf(argv[6], "%d", &Nfr) == EOF) {
        printf("ERROR: Number of frames input is incorrect");
        return EXIT_FAILURE;
    }

    if (Nfr <= 0) {
        printf("number of frames must be > 0\n");
        return EXIT_FAILURE;
    }

    //converting a string to a integer
    if (sscanf(argv[8], "%d", &Nparticles) == EOF) {
        printf("ERROR: Number of particles input is incorrect");
        return EXIT_FAILURE;
    }

    if (Nparticles <= 0) {
        printf("Number of particles must be > 0\n");
        return EXIT_FAILURE;
    }
    //establish seed
    int * seed = (int *) malloc(sizeof (int) *Nparticles);
    int i;
    int seed_base = options.seed_set ? options.seed_base : (int)time(0);
    for (i = 0; i < Nparticles; i++)
        seed[i] = seed_base * i;
    //malloc matrix
    unsigned char * I = (unsigned char *) malloc(sizeof (unsigned char) *IszX * IszY * Nfr);
    long long start = get_time();
    //call video sequence
    videoSequence(I, IszX, IszY, Nfr, seed);
    long long endVideoSequence = get_time();
    printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));
    //call particle filter
    ParticlefilterFloatReference reference;
    memset(&reference, 0, sizeof(reference));
    reference.seed_base = seed_base;
    particleFilter(I, IszX, IszY, Nfr, seed, Nparticles, &reference);
    int status = EXIT_SUCCESS;
    if (options.save_reference_path != NULL &&
        save_reference(options.save_reference_path, &reference) != 0) {
        status = EXIT_FAILURE;
    }
    if (status == EXIT_SUCCESS && options.verify_reference_path != NULL &&
        verify_reference(options.verify_reference_path, &reference) != 0) {
        status = EXIT_FAILURE;
    }
    long long endParticleFilter = get_time();
    printf("PARTICLE FILTER TOOK %f\n", elapsed_time(endVideoSequence, endParticleFilter));
    printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));

    free(seed);
    free(I);
    return status;
}
