/*
 * nn.cu
 * Nearest Neighbor
 *
 */

#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <vector>
#include "cuda.h"
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

#define min( a, b )			a > b ? b : a
#define ceilDiv( a, b )		( a + b - 1 ) / b
#define print( x )			printf( #x ": %lu\n", (unsigned long) x )
#define DEBUG				false

#define DEFAULT_THREADS_PER_BLOCK 256

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28	// character position of the latitude value in each record
#define OPEN 10000	// initial value of nearest neighbors
#define NN_ABS_TOLERANCE 1.0e-4f
#define NN_REL_TOLERANCE 1.0e-6f
#define PARSE_ERROR -1
#define PARSE_OK 0
#define PARSE_HELP 1


typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;

typedef struct options
{
  char filename[100];
  int resultsCount;
  float lat;
  float lng;
  int quiet;
  int timing;
  int platform;
  int device;
  int verify_cpu;
} ProgramOptions;

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations);
void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN);
void printUsage();
int parseCommandline(int argc, char *argv[], ProgramOptions *options);
int verifyCpuReference(const std::vector<LatLong> &locations, const float *distances, int numRecords, float lat, float lng);

/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position
*/
__global__ void euclid(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng)
{
	//int globalId = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
	int globalId = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x; // more efficient
    LatLong *latLong = d_locations+globalId;
    if (globalId < numRecords) {
        float *dist=d_distances+globalId;
        *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	}
}

/**
* This program finds the k-nearest neighbors
**/

int main(int argc, char* argv[])
{
	int    i=0;
	int status = EXIT_SUCCESS;
	ProgramOptions options;
	memset(&options, 0, sizeof(options));
	options.resultsCount = 10;

    std::vector<Record> records;
    std::vector<LatLong> locations;

    // parse command line
    int parseStatus = parseCommandline(argc, argv, &options);
    if (parseStatus != PARSE_OK) {
      printUsage();
      return parseStatus == PARSE_HELP ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    int numRecords = loadData(options.filename,records,locations);
    if (options.resultsCount > numRecords) options.resultsCount = numRecords;

    //for(i=0;i<numRecords;i++)
    //  printf("%s, %f, %f\n",(records[i].recString),locations[i].lat,locations[i].lng);


    //Pointers to host memory
	float *distances;
	//Pointers to device memory
	LatLong *d_locations;
	float *d_distances;


	// Scaling calculations - added by Sam Kauffman
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties( &deviceProp, 0 );
	cudaDeviceSynchronize();
	unsigned long maxGridX = deviceProp.maxGridSize[0];
	unsigned long threadsPerBlock = min( deviceProp.maxThreadsPerBlock, DEFAULT_THREADS_PER_BLOCK );
	size_t totalDeviceMemory;
	size_t freeDeviceMemory;
	cudaMemGetInfo(  &freeDeviceMemory, &totalDeviceMemory );
	cudaDeviceSynchronize();
	unsigned long usableDeviceMemory = freeDeviceMemory * 85 / 100; // 85% arbitrary throttle to compensate for known CUDA bug
	unsigned long maxThreads = usableDeviceMemory / 12; // 4 bytes in 3 vectors per thread
	if ( numRecords > maxThreads )
	{
		fprintf( stderr, "Error: Input too large.\n" );
		exit( 1 );
	}
	unsigned long blocks = ceilDiv( numRecords, threadsPerBlock ); // extra threads will do nothing
	unsigned long gridY = ceilDiv( blocks, maxGridX );
	unsigned long gridX = ceilDiv( blocks, gridY );
	// There will be no more than (gridY - 1) extra blocks
	dim3 gridDim( gridX, gridY );

	if ( DEBUG )
	{
		print( totalDeviceMemory ); // 804454400
		print( freeDeviceMemory );
		print( usableDeviceMemory );
		print( maxGridX ); // 65535
		print( deviceProp.maxThreadsPerBlock ); // 1024
		print( threadsPerBlock );
		print( maxThreads );
		print( blocks ); // 130933
		print( gridY );
		print( gridX );
	}

	/**
	* Allocate memory on host and device
	*/
	distances = (float *)malloc(sizeof(float) * numRecords);
	cudaMalloc((void **) &d_locations,sizeof(LatLong) * numRecords);
	cudaMalloc((void **) &d_distances,sizeof(float) * numRecords);

   /**
    * Transfer data from host to device
    */
    cudaMemcpy( d_locations, &locations[0], sizeof(LatLong) * numRecords, cudaMemcpyHostToDevice);

    /**
    * Execute kernel
    */

#ifdef  TIMING
    gettimeofday(&tv_kernel_start, NULL);
#endif

    euclid<<< gridDim, threadsPerBlock >>>(d_locations,d_distances,numRecords,options.lat,options.lng);
    cudaDeviceSynchronize();

#ifdef  TIMING
    gettimeofday(&tv_kernel_end, NULL);
    tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
    kernel_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    //Copy data from device memory to host memory
    cudaMemcpy( distances, d_distances, sizeof(float)*numRecords, cudaMemcpyDeviceToHost );

	if (options.verify_cpu &&
	    verifyCpuReference(locations, distances, numRecords, options.lat, options.lng) != 0) {
	  rodinia_print_fail("NN CPU reference verification");
	  status = EXIT_FAILURE;
	}

	// find the resultsCount least distances
    if (status == EXIT_SUCCESS) {
      findLowest(records,distances,numRecords,options.resultsCount);
    }

    // print out results
    if (!options.quiet && status == EXIT_SUCCESS)
    for(i=0;i<options.resultsCount;i++) {
      printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    }
    free(distances);
    //Free memory
	cudaFree(d_locations);
	cudaFree(d_distances);

#ifdef  TIMING
    printf("Exec: %f\n", kernel_time);
#endif
	return status;
}

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
    FILE   *flist,*fp;
	int    i=0;
	char dbname[64];
	int recNum=0;

    /**Main processing **/

    flist = fopen(filename, "r");
    if (!flist) {
        fprintf(stderr, "error opening filelist %s\n", filename);
        exit(1);
    }
	while(!feof(flist)) {
		/**
		* Read in all records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
		if(fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(1);
        }
        fp = fopen(dbname, "r");
        if(!fp) {
            printf("error opening a db\n");
            exit(1);
        }
        // read each record
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            fgets(record.recString,49,fp);
            fgetc(fp); // newline
            if (feof(fp)) break;

            // parse for lat and long
            char substr[6];

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);

            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
//    for(i=0;i<rec_count*REC_LENGTH;i++) printf("%c",sandbox[i]);
    return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN){
  int i,j;
  float val;
  int minLoc;
  Record tempRec;
  float tempDist;

  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = records[i];
    records[i] = records[minLoc];
    records[minLoc] = tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int verifyCpuReference(const std::vector<LatLong> &locations, const float *distances, int numRecords, float lat, float lng)
{
  for (int i = 0; i < numRecords; i++) {
    float deltaLat = lat - locations[i].lat;
    float deltaLng = lng - locations[i].lng;
    float expected = sqrtf(deltaLat * deltaLat + deltaLng * deltaLng);
    float diff = fabsf(distances[i] - expected);
    float tolerance = NN_ABS_TOLERANCE + NN_REL_TOLERANCE * fabsf(expected);
    if (!isfinite(distances[i]) || !isfinite(expected) || diff > tolerance) {
      fprintf(stderr,
              "NN CPU reference mismatch at record %d: actual=%g expected=%g diff=%g tolerance=%g\n",
              i,
              distances[i],
              expected,
              diff,
              tolerance);
      return -1;
    }
  }

  return rodinia_print_pass("NN CPU reference verification");
}

int parseCommandline(int argc, char *argv[], ProgramOptions *options){
    int i;
    if (argc < 2) return PARSE_ERROR; // error
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
      return PARSE_HELP;
    }
    if (argv[1][0] == '-') {
      fprintf(stderr, "Missing filename before option: %s\n", argv[1]);
      return PARSE_ERROR;
    }
    strncpy(options->filename,argv[1],sizeof(options->filename) - 1);
    options->filename[sizeof(options->filename) - 1] = '\0';

    for(i=2;i<argc;i++) {
      if (strcmp(argv[i], "--verify-cpu") == 0) {
        options->verify_cpu = 1;
      }
      else if (strcmp(argv[i], "-r") == 0) {
        if (++i >= argc) return PARSE_ERROR;
        options->resultsCount = atoi(argv[i]);
      }
      else if (strcmp(argv[i], "-lat") == 0) {
        if (++i >= argc) return PARSE_ERROR;
        options->lat = atof(argv[i]);
      }
      else if (strcmp(argv[i], "-lng") == 0) {
        if (++i >= argc) return PARSE_ERROR;
        options->lng = atof(argv[i]);
      }
      else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
        return PARSE_HELP;
      }
      else if (strcmp(argv[i], "-q") == 0) {
        options->quiet = 1;
      }
      else if (strcmp(argv[i], "-t") == 0) {
        options->timing = 1;
      }
      else if (strcmp(argv[i], "-p") == 0) {
        if (++i >= argc) return PARSE_ERROR;
        options->platform = atoi(argv[i]);
      }
      else if (strcmp(argv[i], "-d") == 0) {
        if (++i >= argc) return PARSE_ERROR;
        options->device = atoi(argv[i]);
      }
      else {
        fprintf(stderr, "Unknown option: %s\n", argv[i]);
        return PARSE_ERROR;
      }
    }
    if ((options->device >= 0 && options->platform < 0) ||
        (options->platform >= 0 && options->device < 0)) // both p and d must be specified if either are specified
      return PARSE_ERROR;
    return PARSE_OK;
}

void printUsage(){
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] [--verify-cpu] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}
