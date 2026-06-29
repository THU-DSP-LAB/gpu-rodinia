/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

/*************************************************************************/
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 binary file: first int is the number of objects     **/
/**                              2nd int is the no. of features of each **/
/**                              object                                 **/
/**                 This example performs a fuzzy c-means clustering    **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department Northwestern University                   **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee					**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					No longer performs "validity" function to analyze	**/
/**					compactness and separation crietria; instead		**/
/**					calculate root mean squared error.					**/
/**                                                                     **/
/*************************************************************************/
#define _CRT_SECURE_NO_DEPRECATE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include "kmeans.h"
#include "../../common/rodinia_verify.h"

extern double wtime(void);

#define KMEANS_ABS_TOLERANCE 0.01f
#define KMEANS_REL_TOLERANCE 0.00001f

typedef struct {
	int npoints;
	int nfeatures;
	int nclusters;
	float threshold;
	float **features;
} CpuKmeansConfig;

static float **alloc_2d_float(int rows, int cols)
{
	float **values = (float **)malloc(rows * sizeof(float *));
	if (values == NULL) {
		return NULL;
	}
	values[0] = (float *)calloc(rows * cols, sizeof(float));
	if (values[0] == NULL) {
		free(values);
		return NULL;
	}
	for (int row = 1; row < rows; row++) {
		values[row] = values[row - 1] + cols;
	}
	return values;
}

static void free_2d_float(float **values)
{
	if (values == NULL) {
		return;
	}
	free(values[0]);
	free(values);
}

static int filter_verify_cpu_option(int argc, char **argv, char ***filtered_argv, int *verify_cpu)
{
	char **filtered = (char **)malloc((argc + 1) * sizeof(char *));
	if (filtered == NULL) {
		fprintf(stderr, "Cannot allocate Kmeans argument list\n");
		return -1;
	}

	int filtered_argc = 1;
	*verify_cpu = 0;
	filtered[0] = argv[0];
	for (int arg = 1; arg < argc; arg++) {
		if (strcmp(argv[arg], "--verify-cpu") == 0) {
			*verify_cpu = 1;
			continue;
		}
		filtered[filtered_argc] = argv[arg];
		filtered_argc++;
	}
	filtered[filtered_argc] = NULL;
	*filtered_argv = filtered;
	return filtered_argc;
}

static int initialize_cpu_clusters(const CpuKmeansConfig *config, float **clusters)
{
	int *initial = (int *)malloc(config->npoints * sizeof(int));
	if (initial == NULL) {
		fprintf(stderr, "Cannot allocate Kmeans CPU reference initial centers\n");
		return -1;
	}
	for (int point = 0; point < config->npoints; point++) {
		initial[point] = point;
	}

	int next = 0;
	int initial_points = config->npoints;
	for (int cluster = 0; cluster < config->nclusters && initial_points >= 0; cluster++) {
		for (int feature = 0; feature < config->nfeatures; feature++) {
			clusters[cluster][feature] = config->features[initial[next]][feature];
		}
		int temp = initial[next];
		initial[next] = initial[initial_points - 1];
		initial[initial_points - 1] = temp;
		initial_points--;
		next++;
	}

	free(initial);
	return 0;
}

static int nearest_cpu_cluster(const CpuKmeansConfig *config, float **clusters, int point)
{
	int nearest = -1;
	float min_dist = FLT_MAX;
	for (int cluster = 0; cluster < config->nclusters; cluster++) {
		float dist = 0.0f;
		for (int feature = 0; feature < config->nfeatures; feature++) {
			float diff = config->features[point][feature] - clusters[cluster][feature];
			dist += diff * diff;
		}
		if (dist < min_dist) {
			min_dist = dist;
			nearest = cluster;
		}
	}
	return nearest;
}

static int assign_cpu_memberships(
	const CpuKmeansConfig *config,
	float **clusters,
	int *membership,
	int *new_centers_len,
	float **new_centers)
{
	int delta = 0;
	for (int point = 0; point < config->npoints; point++) {
		int cluster_id = nearest_cpu_cluster(config, clusters, point);
		new_centers_len[cluster_id]++;
		if (cluster_id != membership[point]) {
			delta++;
			membership[point] = cluster_id;
		}
		for (int feature = 0; feature < config->nfeatures; feature++) {
			new_centers[cluster_id][feature] += config->features[point][feature];
		}
	}
	return delta;
}

static void update_cpu_clusters(
	const CpuKmeansConfig *config,
	float **clusters,
	int *new_centers_len,
	float **new_centers)
{
	for (int cluster = 0; cluster < config->nclusters; cluster++) {
		for (int feature = 0; feature < config->nfeatures; feature++) {
			if (new_centers_len[cluster] > 0) {
				clusters[cluster][feature] =
					new_centers[cluster][feature] / new_centers_len[cluster];
			}
			new_centers[cluster][feature] = 0.0f;
		}
		new_centers_len[cluster] = 0;
	}
}

static float **run_cpu_kmeans_reference(const CpuKmeansConfig *config)
{
	float **clusters = alloc_2d_float(config->nclusters, config->nfeatures);
	float **new_centers = alloc_2d_float(config->nclusters, config->nfeatures);
	int *new_centers_len = (int *)calloc(config->nclusters, sizeof(int));
	int *membership = (int *)malloc(config->npoints * sizeof(int));
	if (clusters == NULL || new_centers == NULL || new_centers_len == NULL || membership == NULL) {
		fprintf(stderr, "Cannot allocate Kmeans CPU reference buffers\n");
		free_2d_float(clusters);
		free_2d_float(new_centers);
		free(new_centers_len);
		free(membership);
		return NULL;
	}

	if (initialize_cpu_clusters(config, clusters) != 0) {
		free_2d_float(clusters);
		free_2d_float(new_centers);
		free(new_centers_len);
		free(membership);
		return NULL;
	}

	for (int point = 0; point < config->npoints; point++) {
		membership[point] = -1;
	}

	int loop = 0;
	int delta;
	do {
		delta = assign_cpu_memberships(config, clusters, membership, new_centers_len, new_centers);
		update_cpu_clusters(config, clusters, new_centers_len, new_centers);
	} while (((float)delta > config->threshold) && (loop++ < 500));

	free_2d_float(new_centers);
	free(new_centers_len);
	free(membership);
	return clusters;
}

static int verify_cpu_reference(
	float **actual,
	float **features,
	int nfeatures,
	int npoints,
	int nclusters,
	float threshold)
{
	CpuKmeansConfig config;
	config.npoints = npoints;
	config.nfeatures = nfeatures;
	config.nclusters = nclusters > npoints ? npoints : nclusters;
	config.threshold = threshold;
	config.features = features;

	float **expected = run_cpu_kmeans_reference(&config);
	if (expected == NULL) {
		return -1;
	}

	for (int cluster = 0; cluster < config.nclusters; cluster++) {
		for (int feature = 0; feature < config.nfeatures; feature++) {
			float diff = fabsf(actual[cluster][feature] - expected[cluster][feature]);
			float tolerance =
				KMEANS_ABS_TOLERANCE + KMEANS_REL_TOLERANCE * fabsf(expected[cluster][feature]);
			if (!isfinite(actual[cluster][feature]) ||
				!isfinite(expected[cluster][feature]) ||
				diff > tolerance) {
				fprintf(stderr,
					"Kmeans CPU reference mismatch at cluster %d feature %d: actual=%g expected=%g diff=%g tolerance=%g\n",
					cluster,
					feature,
					actual[cluster][feature],
					expected[cluster][feature],
					diff,
					tolerance);
				free_2d_float(expected);
				return -1;
			}
		}
	}

	free_2d_float(expected);
	return rodinia_print_pass("Kmeans CPU reference verification");
}



/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
    char *help =
        "\nUsage: %s [switches] -i filename\n\n"
		"    -i filename      :file containing data to be clustered\n"		
		"    -m max_nclusters :maximum number of clusters allowed    [default=5]\n"
        "    -n min_nclusters :minimum number of clusters allowed    [default=5]\n"
		"    -t threshold     :threshold value                       [default=0.001]\n"
		"    -l nloops        :iteration for each number of clusters [default=1]\n"
		"    -b               :input file is in binary format\n"
        "    -r               :calculate RMSE                        [default=off]\n"
		"    -o               :output cluster center coordinates     [default=off]\n"
		"    --verify-cpu     :compare final cluster centers with CPU reference\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int setup(int argc, char **argv) {
		int		opt;
 extern char   *optarg;
 extern int    optind;
		char   *filename = 0;
		char  **filtered_argv = NULL;
		int     filtered_argc;
		int     verify_cpu = 0;
		float  *buf;
		char	line[1024];
		int		isBinaryFile = 0;

		float	threshold = 0.001;		/* default value */
		int		max_nclusters=5;		/* default value */
		int		min_nclusters=5;		/* default value */
		int		best_nclusters = 0;
		int		nfeatures = 0;
		int		npoints = 0;
		float	len;
		         
		float **features;
		float **cluster_centres=NULL;
		int		i, j, index;
		int		nloops = 1;				/* default value */
				
		int		isRMSE = 0;		
		float	rmse;
		
		int		isOutput = 0;
		//float	cluster_timing, io_timing;		

		filtered_argc = filter_verify_cpu_option(argc, argv, &filtered_argv, &verify_cpu);
		if (filtered_argc < 0) {
			return 1;
		}
		optind = 1;

		/* obtain command line arguments and change appropriate options */
		while ( (opt=getopt(filtered_argc,filtered_argv,"i:t:m:n:l:bro"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;            
            case 't': threshold=atof(optarg);
                      break;
            case 'm': max_nclusters = atoi(optarg);
                      break;
            case 'n': min_nclusters = atoi(optarg);
                      break;
			case 'r': isRMSE = 1;
                      break;
			case 'o': isOutput = 1;
					  break;
		    case 'l': nloops = atoi(optarg);
					  break;
            case '?': usage(filtered_argv[0]);
                      break;
            default: usage(filtered_argv[0]);
                      break;
        }
    }

    if (filename == 0) usage(filtered_argv[0]);
		
	/* ============== I/O begin ==============*/
    /* get nfeatures and npoints */
    //io_timing = omp_get_wtime();
    if (isBinaryFile) {		//Binary file input
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        read(infile, &npoints,   sizeof(int));
        read(infile, &nfeatures, sizeof(int));        

        /* allocate space for features[][] and read attributes of all objects */
        buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
        features    = (float**)malloc(npoints*          sizeof(float*));
        features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
        for (i=1; i<npoints; i++)
            features[i] = features[i-1] + nfeatures;

        read(infile, buf, npoints*nfeatures*sizeof(float));

        close(infile);
    }
    else {
        FILE *infile;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
		}		
        while (fgets(line, 1024, infile) != NULL)
			if (strtok(line, " \t\n") != 0)
                npoints++;			
        rewind(infile);
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first attribute): nfeatures = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) nfeatures++;
                break;
            }
        }        

        /* allocate space for features[] and read attributes of all objects */
        buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
        features    = (float**)malloc(npoints*          sizeof(float*));
        features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
        for (i=1; i<npoints; i++)
            features[i] = features[i-1] + nfeatures;
        rewind(infile);
        i = 0;
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;            
            for (j=0; j<nfeatures; j++) {
                buf[i] = atof(strtok(NULL, " ,\t\n"));             
                i++;
            }            
        }
        fclose(infile);
    }
    //io_timing = omp_get_wtime() - io_timing;
	
	printf("\nI/O completed\n");
	printf("\nNumber of objects: %d\n", npoints);
	printf("Number of features: %d\n", nfeatures);	
	/* ============== I/O end ==============*/

	// error check for clusters
	if (npoints < min_nclusters)
	{
		printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n", min_nclusters, npoints);
		exit(0);
	}

	srand(7);												/* seed for future random number generator */	
	memcpy(features[0], buf, npoints*nfeatures*sizeof(float)); /* now features holds 2-dimensional array of features */
	free(buf);

	/* ======================= core of the clustering ===================*/

    //cluster_timing = omp_get_wtime();		/* Total clustering time */
	cluster_centres = NULL;
    index = cluster(npoints,				/* number of data points */
					nfeatures,				/* number of features for each point */
					features,				/* array: [npoints][nfeatures] */
					min_nclusters,			/* range of min to max number of clusters */
					max_nclusters,
					threshold,				/* loop termination factor */
				   &best_nclusters,			/* return: number between min and max */
				   &cluster_centres,		/* return: [best_nclusters][nfeatures] */  
				   &rmse,					/* Root Mean Squared Error */
					isRMSE,					/* calculate RMSE */
					nloops);				/* number of iteration for each number of clusters */		
    
	//cluster_timing = omp_get_wtime() - cluster_timing;


	/* =============== Command Line Output =============== */

	/* cluster center coordinates
	   :displayed only for when k=1*/
	if((min_nclusters == max_nclusters) && (isOutput == 1)) {
		printf("\n================= Centroid Coordinates =================\n");
		for(i = 0; i < max_nclusters; i++){
			printf("%d:", i);
			for(j = 0; j < nfeatures; j++){
				printf(" %.2f", cluster_centres[i][j]);
			}
			printf("\n\n");
		}
	}

	if (verify_cpu &&
		verify_cpu_reference(
			cluster_centres,
			features,
			nfeatures,
			npoints,
			max_nclusters,
			threshold) != 0) {
		rodinia_print_fail("Kmeans CPU reference verification");
		free(features[0]);
		free(features);
		if (cluster_centres != NULL) {
			free(cluster_centres[0]);
			free(cluster_centres);
		}
		free(filtered_argv);
		return 1;
	}
	
	len = (float) ((max_nclusters - min_nclusters + 1)*nloops);

	printf("Number of Iteration: %d\n", nloops);
	//printf("Time for I/O: %.5fsec\n", io_timing);
	//printf("Time for Entire Clustering: %.5fsec\n", cluster_timing);
	
	if(min_nclusters != max_nclusters){
		if(nloops != 1){									//range of k, multiple iteration
			//printf("Average Clustering Time: %fsec\n",
			//		cluster_timing / len);
			printf("Best number of clusters is %d\n", best_nclusters);				
		}
		else{												//range of k, single iteration
			//printf("Average Clustering Time: %fsec\n",
			//		cluster_timing / len);
			printf("Best number of clusters is %d\n", best_nclusters);				
		}
	}
	else{
		if(nloops != 1){									// single k, multiple iteration
			//printf("Average Clustering Time: %.5fsec\n",
			//		cluster_timing / nloops);
			if(isRMSE)										// if calculated RMSE
				printf("Number of trials to approach the best RMSE of %.3f is %d\n", rmse, index + 1);
		}
		else{												// single k, single iteration				
			if(isRMSE)										// if calculated RMSE
				printf("Root Mean Squared Error: %.3f\n", rmse);
		}
	}
	

	/* free up memory */
	free(features[0]);
	free(features);
	if (cluster_centres != NULL) {
		free(cluster_centres[0]);
		free(cluster_centres);
	}
	free(filtered_argv);
    return(0);
}
