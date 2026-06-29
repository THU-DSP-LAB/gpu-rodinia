#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include "../../common/rodinia_verify.h"

#ifdef TIMING
#include "timing.h"

struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0;
#endif

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

//#define BENCH_PRINT

int run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;

static void usage(const char *program)
{
        fprintf(stderr, "Usage: %s <cols> <rows> <pyramid_height> [--verify-cpu]\n", program);
}

static int parse_positive_int(const char *text, const char *name, int *value)
{
        char *end = NULL;
        errno = 0;
        long parsed = strtol(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
                fprintf(stderr, "Invalid %s: %s\n", name, text);
                return -1;
        }
        *value = (int)parsed;
        return 0;
}

static int parse_options(int argc, char** argv, int *verify_cpu)
{
	if(argc < 4){
                usage(argv[0]);
                return -1;
        }
        if (parse_positive_int(argv[1], "cols", &cols) != 0 ||
            parse_positive_int(argv[2], "rows", &rows) != 0 ||
            parse_positive_int(argv[3], "pyramid_height", &pyramid_height) != 0) {
                usage(argv[0]);
                return -1;
        }
        if (pyramid_height * HALO * 2 >= BLOCK_SIZE) {
                fprintf(stderr, "pyramid_height is too large for block size %d\n", BLOCK_SIZE);
                return -1;
        }
        *verify_cpu = 0;
        for (int arg = 4; arg < argc; arg++) {
                if (strcmp(argv[arg], "--verify-cpu") == 0) {
                        *verify_cpu = 1;
                        continue;
                }
                fprintf(stderr, "Unknown option: %s\n", argv[arg]);
                usage(argv[0]);
                return -1;
        }
        return 0;
}

void
init(void)
{
	data = new int[rows*cols];
	wall = new int*[rows];
	for(int n=0; n<rows; n++)
		wall[n]=data+cols*n;
	result = new int[cols];
	
	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
#ifdef BENCH_PRINT
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ",wall[i][j]) ;
        }
        printf("\n") ;
    }
#endif
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

static int verify_cpu_reference(const int *actual)
{
        int *previous = (int *)malloc(sizeof(int) * cols);
        int *current = (int *)malloc(sizeof(int) * cols);
        if (previous == NULL || current == NULL) {
                fprintf(stderr, "Cannot allocate Pathfinder CPU reference buffers\n");
                free(previous);
                free(current);
                return -1;
        }

        memcpy(previous, data, sizeof(int) * cols);
        for (int row = 1; row < rows; row++) {
                int *wall_row = data + row * cols;
                for (int col = 0; col < cols; col++) {
                        int left = col == 0 ? previous[col] : previous[col - 1];
                        int up = previous[col];
                        int right = col == cols - 1 ? previous[col] : previous[col + 1];
                        int shortest = MIN(MIN(left, up), right);
                        current[col] = wall_row[col] + shortest;
                }
                int *temp = previous;
                previous = current;
                current = temp;
        }

        for (int col = 0; col < cols; col++) {
                if (actual[col] != previous[col]) {
                        fprintf(stderr,
                                "Pathfinder CPU reference mismatch at column %d: actual=%d expected=%d\n",
                                col,
                                actual[col],
                                previous[col]);
                        free(previous);
                        free(current);
                        return -1;
                }
        }

        free(previous);
        free(current);
        return rodinia_print_pass("Pathfinder CPU reference verification");
}

__global__ void dynproc_kernel(
                int iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                int cols, 
                int rows,
                int startStep,
                int border)
{

        __shared__ int prev[BLOCK_SIZE];
        __shared__ int result[BLOCK_SIZE];

	int bx = blockIdx.x;
	int tx=threadIdx.x;
	
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
	int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkX = small_block_cols*bx-border;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int xidx = blkX+tx;
       
        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

        int W = tx-1;
        int E = tx+1;
        
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  isValid){
                  computed = true;
                  int left = prev[W];
                  int up = prev[tx];
                  int right = prev[E];
                  int shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  int index = cols*(startStep+i)+xidx;
                  result[tx] = shortest + gpuWall[index];
	
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                prev[tx]= result[tx];
	    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          gpuResults[xidx]=result[tx];		
      }
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols, \
	 int pyramid_height, int blockCols, int borderCols)
{
        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(blockCols);  
	
        int src = 1, dst = 0;
	for (int t = 0; t < rows-1; t+=pyramid_height) {
            int temp = src;
            src = dst;
            dst = temp;
            dynproc_kernel<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, rows-t-1), 
                gpuWall, gpuResult[src], gpuResult[dst],
                cols,rows, t, borderCols);

            // for the measurement fairness
            cudaDeviceSynchronize();
	}
        return dst;
}

int main(int argc, char** argv)
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(DEVICE);

    return run(argc,argv);
}

int run(int argc, char** argv)
{
    int verify_cpu;
    if (parse_options(argc, argv, &verify_cpu) != 0) {
        return EXIT_FAILURE;
    }
    init();

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
	
    int *gpuWall, *gpuResult[2];
    int size = rows*cols;

    cudaMalloc((void**)&gpuResult[0], sizeof(int)*cols);
    cudaMalloc((void**)&gpuResult[1], sizeof(int)*cols);
    cudaMemcpy(gpuResult[0], data, sizeof(int)*cols, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpuWall, sizeof(int)*(size-cols));
    cudaMemcpy(gpuWall, data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice);

#ifdef  TIMING
    gettimeofday(&tv_kernel_start, NULL);
#endif

    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, \
	 pyramid_height, blockCols, borderCols);

#ifdef  TIMING
    gettimeofday(&tv_kernel_end, NULL);
    tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
    kernel_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    cudaMemcpy(result, gpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost);
    int status = EXIT_SUCCESS;
    if (verify_cpu && verify_cpu_reference(result) != 0) {
        rodinia_print_fail("Pathfinder CPU reference verification");
        status = EXIT_FAILURE;
    }

#ifdef BENCH_PRINT
    for (int i = 0; i < cols; i++)
            printf("%d ",data[i]) ;
    printf("\n") ;
    for (int i = 0; i < cols; i++)
            printf("%d ",result[i]) ;
    printf("\n") ;
#endif

    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);

    delete [] data;
    delete [] wall;
    delete [] result;

#ifdef  TIMING
    printf("Exec: %f\n", kernel_time);
#endif
    return status;
}
