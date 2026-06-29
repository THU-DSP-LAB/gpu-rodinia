/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <stdint.h>
#include "../../common/rodinia_verify.h"

#ifdef TIMING
#include "timing.h"
#endif

#define MAX_THREADS_PER_BLOCK 512
#define BFS_RESULT_FILE "result.txt"
#define BFS_REFERENCE_MAGIC "GPIDL_RODINIA_BFS_REFERENCE"
#define BFS_REFERENCE_VERSION 1

typedef enum {
	REFERENCE_MODE_RUN = 0,
	REFERENCE_MODE_SAVE,
	REFERENCE_MODE_VERIFY
} ReferenceMode;

typedef struct {
	ReferenceMode mode;
	const char *path;
	bool verify_cpu;
} ReferenceOptions;

int no_of_nodes;
int edge_list_size;
FILE *fp;

#ifdef TIMING
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

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

#include "kernel.cu"
#include "kernel2.cu"

int BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
	return BFSGraph( argc, argv);
}

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);
fprintf(stderr,"       %s <input_file> --verify-cpu\n", argv[0]);
fprintf(stderr,"       %s <input_file> --save-reference <path>\n", argv[0]);
fprintf(stderr,"       %s <input_file> --verify-reference <path>\n", argv[0]);

}

int parse_reference_options(int argc, char **argv, ReferenceOptions *options)
{
	options->mode = REFERENCE_MODE_RUN;
	options->path = NULL;
	options->verify_cpu = false;
	for (int index = 2; index < argc; index++) {
		if (strcmp(argv[index], "--verify-cpu") == 0) {
			options->verify_cpu = true;
			continue;
		}
		if (strcmp(argv[index], "--save-reference") == 0 ||
			strcmp(argv[index], "--verify-reference") == 0) {
			if (index + 1 >= argc) {
				fprintf(stderr, "Missing reference path for %s\n", argv[index]);
				Usage(argc, argv);
				return -1;
			}
			if (options->mode != REFERENCE_MODE_RUN) {
				fprintf(stderr, "Only one reference file mode may be specified\n");
				return -1;
			}
			options->mode = strcmp(argv[index], "--save-reference") == 0 ?
				REFERENCE_MODE_SAVE :
				REFERENCE_MODE_VERIFY;
			options->path = argv[index + 1];
			index++;
			continue;
		}
		fprintf(stderr, "Unknown option: '%s'\n", argv[index]);
		Usage(argc, argv);
		return -1;
	}
	if (argc < 2) {
		Usage(argc, argv);
		return -1;
	}
	return 0;
}

uint64_t fnv1a_file_hash(const char *path)
{
	const uint64_t offset = 1469598103934665603ULL;
	const uint64_t prime = 1099511628211ULL;
	uint64_t hash = offset;
	FILE *file = fopen(path, "rb");
	if (file == NULL) {
		fprintf(stderr, "Cannot open input for hashing: %s\n", path);
		return 0;
	}
	for (;;) {
		int byte = fgetc(file);
		if (byte == EOF) {
			break;
		}
		hash ^= (uint8_t)byte;
		hash *= prime;
	}
	fclose(file);
	return hash;
}

int write_result_file(const int *costs, int count)
{
	FILE *fpo = fopen(BFS_RESULT_FILE, "w");
	if (fpo == NULL) {
		fprintf(stderr, "Cannot open %s for write\n", BFS_RESULT_FILE);
		return -1;
	}
	for (int i = 0; i < count; i++) {
		fprintf(fpo, "%d) cost:%d\n", i, costs[i]);
	}
	fclose(fpo);
	return 0;
}

int save_reference(
	const char *path,
	uint64_t input_hash,
	const int *costs,
	int node_count,
	int edge_count)
{
	FILE *file = fopen(path, "w");
	if (file == NULL) {
		fprintf(stderr, "Cannot open reference for write: %s\n", path);
		return -1;
	}
	fprintf(file, "%s %d\n", BFS_REFERENCE_MAGIC, BFS_REFERENCE_VERSION);
	fprintf(file, "input_hash %016llx\n", (unsigned long long)input_hash);
	fprintf(file, "nodes %d\n", node_count);
	fprintf(file, "edges %d\n", edge_count);
	fprintf(file, "costs %d\n", node_count);
	for (int i = 0; i < node_count; i++) {
		fprintf(file, "%d\n", costs[i]);
	}
	fclose(file);
	printf("Saved BFS reference to '%s'\n", path);
	return 0;
}

int verify_reference_header(
	FILE *file,
	uint64_t input_hash,
	int node_count,
	int edge_count,
	int *cost_count)
{
	char magic[64];
	char label[64];
	int version;
	unsigned long long reference_hash;
	int reference_nodes;
	int reference_edges;
	if (fscanf(file, "%63s %d", magic, &version) != 2 ||
		strcmp(magic, BFS_REFERENCE_MAGIC) != 0 ||
		version != BFS_REFERENCE_VERSION) {
		fprintf(stderr, "Invalid BFS reference header\n");
		return -1;
	}
	if (fscanf(file, "%63s %llx", label, &reference_hash) != 2 ||
		strcmp(label, "input_hash") != 0 ||
		reference_hash != (unsigned long long)input_hash) {
		fprintf(stderr, "BFS reference input hash mismatch\n");
		return -1;
	}
	if (fscanf(file, "%63s %d", label, &reference_nodes) != 2 ||
		strcmp(label, "nodes") != 0 ||
		reference_nodes != node_count) {
		fprintf(stderr, "BFS reference node count mismatch\n");
		return -1;
	}
	if (fscanf(file, "%63s %d", label, &reference_edges) != 2 ||
		strcmp(label, "edges") != 0 ||
		reference_edges != edge_count) {
		fprintf(stderr, "BFS reference edge count mismatch\n");
		return -1;
	}
	if (fscanf(file, "%63s %d", label, cost_count) != 2 ||
		strcmp(label, "costs") != 0 ||
		*cost_count != node_count) {
		fprintf(stderr, "BFS reference cost count mismatch\n");
		return -1;
	}
	return 0;
}

int verify_reference(
	const char *path,
	uint64_t input_hash,
	const int *costs,
	int node_count,
	int edge_count)
{
	FILE *file = fopen(path, "r");
	int cost_count = 0;
	if (file == NULL) {
		fprintf(stderr, "Cannot open reference for read: %s\n", path);
		return -1;
	}
	if (verify_reference_header(file, input_hash, node_count, edge_count, &cost_count) != 0) {
		fclose(file);
		return -1;
	}
	for (int i = 0; i < cost_count; i++) {
		int expected;
		if (fscanf(file, "%d", &expected) != 1) {
			fprintf(stderr, "BFS reference ended before cost[%d]\n", i);
			fclose(file);
			return -1;
		}
		if (costs[i] != expected) {
			fprintf(stderr, "BFS reference mismatch for cost[%d]: actual=%d expected=%d\n", i, costs[i], expected);
			fclose(file);
			return -1;
		}
	}
	fclose(file);
	printf("BFS reference verification matched '%s'\n", path);
	return rodinia_print_pass("BFS reference verification");
}

int handle_reference_mode(
	const ReferenceOptions *options,
	const char *input_path,
	const int *costs,
	int node_count,
	int edge_count)
{
	uint64_t input_hash = 0;
	if (options->mode == REFERENCE_MODE_RUN) {
		return 0;
	}
	input_hash = fnv1a_file_hash(input_path);
	if (input_hash == 0) {
		return -1;
	}
	if (options->mode == REFERENCE_MODE_SAVE) {
		return save_reference(options->path, input_hash, costs, node_count, edge_count);
	}
	return verify_reference(options->path, input_hash, costs, node_count, edge_count);
}

int verify_cpu_reference(const Node *nodes, const int *edges, const int *actual_costs, int node_count, int source)
{
	int *expected_costs = (int*) malloc(sizeof(int) * node_count);
	int *queue = (int*) malloc(sizeof(int) * node_count);
	int head = 0;
	int tail = 0;
	if (expected_costs == NULL || queue == NULL) {
		fprintf(stderr, "Cannot allocate BFS CPU reference buffers\n");
		free(expected_costs);
		free(queue);
		return -1;
	}
	for (int index = 0; index < node_count; index++) {
		expected_costs[index] = -1;
	}
	expected_costs[source] = 0;
	queue[tail++] = source;
	while (head < tail) {
		int node = queue[head++];
		for (int edge = nodes[node].starting; edge < nodes[node].starting + nodes[node].no_of_edges; edge++) {
			int neighbor = edges[edge];
			if (neighbor < 0 || neighbor >= node_count) {
				fprintf(stderr, "BFS CPU reference saw invalid edge target %d\n", neighbor);
				free(expected_costs);
				free(queue);
				return -1;
			}
			if (expected_costs[neighbor] != -1) {
				continue;
			}
			expected_costs[neighbor] = expected_costs[node] + 1;
			queue[tail++] = neighbor;
		}
	}
	for (int index = 0; index < node_count; index++) {
		if (actual_costs[index] != expected_costs[index]) {
			fprintf(stderr, "BFS CPU reference mismatch for cost[%d]: actual=%d expected=%d\n",
				index, actual_costs[index], expected_costs[index]);
			free(expected_costs);
			free(queue);
			return -1;
		}
	}
	free(expected_costs);
	free(queue);
	return rodinia_print_pass("BFS CPU reference verification");
}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
int BFSGraph( int argc, char** argv) 
{

    char *input_f;
	ReferenceOptions reference_options;
	int status = EXIT_SUCCESS;
	if(parse_reference_options(argc, argv, &reference_options) != 0){
	return EXIT_FAILURE;
	}

	input_f = argv[1];
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return EXIT_FAILURE;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    

	printf("Read File\n");

#ifdef  TIMING
    gettimeofday(&tv_total_start, NULL);
#endif
	//Copy the Node list to device memory
	Node* d_graph_nodes;
	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
	cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice) ;

	//Copy the Edge List to device Memory
	int* d_graph_edges;
	cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) ;
	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice) ;

	//Copy the Mask to device memory
	bool* d_graph_mask;
	cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

	bool* d_updating_graph_mask;
	cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

	//Copy the Visited nodes array to device memory
	bool* d_graph_visited;
	cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

	// allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;
	
	// allocate device memory for result
	int* d_cost;
	cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);
	cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice) ;

	//make a bool to check if the execution is over
	bool *d_over;
	cudaMalloc( (void**) &d_over, sizeof(bool));
#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_end, NULL);
    tvsub(&tv_mem_alloc_end, &tv_total_start, &tv);
    h2d_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	printf("Copied Everything to GPU memory\n");

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	int k=0;
	printf("Start traversing the tree\n");
	bool stop;
	//Call the Kernel untill all the elements of Frontier are not false
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;
#ifdef  TIMING
		gettimeofday(&tv_h2d_start, NULL);
#endif
		cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) ;
#ifdef  TIMING
		gettimeofday(&tv_h2d_end, NULL);
		tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
		h2d_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

		Kernel<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);
		// check if kernel execution generated and error

		Kernel2<<< grid, threads, 0 >>>( d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);
		// check if kernel execution generated and error

#ifdef  TIMING
		cudaDeviceSynchronize();
		gettimeofday(&tv_kernel_end, NULL);
		tvsub(&tv_kernel_end, &tv_h2d_end, &tv);
		kernel_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

		cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost) ;
#ifdef  TIMING
		gettimeofday(&tv_d2h_end, NULL);
		tvsub(&tv_d2h_end, &tv_kernel_end, &tv);
		d2h_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

		k++;
	}
	while(stop);


	printf("Kernel Executed %d times\n",k);

	// copy result from device to host
#ifdef  TIMING
	gettimeofday(&tv_d2h_start, NULL);
#endif
	cudaMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) ;
#ifdef  TIMING
	gettimeofday(&tv_d2h_end, NULL);
	tvsub(&tv_d2h_end, &tv_d2h_start, &tv);
	d2h_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	//Store the result into a file
	if (write_result_file(h_cost, no_of_nodes) != 0 ||
		(reference_options.verify_cpu &&
			verify_cpu_reference(h_graph_nodes, h_graph_edges, h_cost, no_of_nodes, source) != 0) ||
		handle_reference_mode(&reference_options, input_f, h_cost, no_of_nodes, edge_list_size) != 0) {
		if (reference_options.verify_cpu) {
			rodinia_print_fail("BFS CPU reference verification");
		}
		if (reference_options.mode == REFERENCE_MODE_VERIFY) {
			rodinia_print_fail("BFS reference verification");
		}
		status = EXIT_FAILURE;
	}
	if (status == EXIT_SUCCESS) {
		printf("Result stored in %s\n", BFS_RESULT_FILE);
	}


	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
#ifdef  TIMING
    gettimeofday(&tv_close_start, NULL);
#endif
	cudaFree(d_graph_nodes);
	cudaFree(d_graph_edges);
	cudaFree(d_graph_mask);
	cudaFree(d_updating_graph_mask);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);
	cudaFree(d_over);

#ifdef  TIMING
	gettimeofday(&tv_close_end, NULL);
	tvsub(&tv_close_end, &tv_close_start, &tv);
	close_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
	tvsub(&tv_close_end, &tv_total_start, &tv);
	total_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	printf("Init: %f\n", init_time);
	printf("MemAlloc: %f\n", mem_alloc_time);
	printf("HtoD: %f\n", h2d_time);
	printf("Exec: %f\n", kernel_time);
	printf("DtoH: %f\n", d2h_time);
	printf("Close: %f\n", close_time);
	printf("Total: %f\n", total_time);
#endif
	return status;
}
