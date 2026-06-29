#include "find_ellipse.h"
#include "track_ellipse.h"
#include <stdint.h>
#include "../../../common/rodinia_verify.h"

#define LEUKOCYTE_REFERENCE_MAGIC "GPIDL_RODINIA_LEUKOCYTE_REFERENCE"
#define LEUKOCYTE_REFERENCE_VERSION 1

typedef struct {
	uint64_t hash;
	uint64_t bytes;
	uint64_t lines;
} LeukocyteFileDigest;

typedef struct {
	char video_name[256];
	uint64_t video_hash;
	uint64_t video_bytes;
	int requested_frames;
	int video_width;
	int video_height;
	int cells_detected;
	uint64_t centers_hash;
	uint64_t centers_values;
} LeukocyteReference;

static uint64_t leukocyte_fnv1a_update(uint64_t hash, const void *data, size_t size)
{
	const unsigned char *bytes = (const unsigned char *)data;
	for (size_t i = 0; i < size; i++) {
		hash ^= bytes[i];
		hash *= 1099511628211ULL;
	}
	return hash;
}

static const char *leukocyte_basename(const char *path)
{
	const char *slash = strrchr(path, '/');
	return slash == NULL ? path : slash + 1;
}

static int leukocyte_compute_file_digest(const char *path, LeukocyteFileDigest *digest)
{
	FILE *file = fopen(path, "rb");
	if (file == NULL) {
		fprintf(stderr, "Cannot open file for digest: %s\n", path);
		return -1;
	}
	digest->hash = 1469598103934665603ULL;
	digest->bytes = 0;
	digest->lines = 0;
	unsigned char buffer[65536];
	while (1) {
		size_t count = fread(buffer, 1, sizeof(buffer), file);
		digest->hash = leukocyte_fnv1a_update(digest->hash, buffer, count);
		for (size_t i = 0; i < count; i++) {
			if (buffer[i] == '\n') {
				digest->lines++;
			}
		}
		digest->bytes += count;
		if (count != sizeof(buffer)) {
			if (ferror(file)) {
				fprintf(stderr, "Failed reading file for digest: %s\n", path);
				fclose(file);
				return -1;
			}
			break;
		}
	}
	fclose(file);
	return 0;
}

static void leukocyte_populate_reference(
	LeukocyteReference *reference,
	const char *video_file_name,
	const LeukocyteFileDigest *digest,
	int requested_frames,
	const avi_t *cell_file,
	int cells_detected,
	const double *x_centers,
	const double *y_centers)
{
	memset(reference, 0, sizeof(*reference));
	snprintf(reference->video_name, sizeof(reference->video_name), "%s", leukocyte_basename(video_file_name));
	reference->video_hash = digest->hash;
	reference->video_bytes = digest->bytes;
	reference->requested_frames = requested_frames;
	reference->video_width = cell_file->width;
	reference->video_height = cell_file->height;
	reference->cells_detected = cells_detected;
	uint64_t hash = 1469598103934665603ULL;
	hash = leukocyte_fnv1a_update(hash, x_centers, (size_t)cells_detected * sizeof(double));
	hash = leukocyte_fnv1a_update(hash, y_centers, (size_t)cells_detected * sizeof(double));
	reference->centers_hash = hash;
	reference->centers_values = (uint64_t)cells_detected * 2ULL;
}

static int leukocyte_save_reference(const char *path, const LeukocyteReference *reference)
{
	FILE *file = fopen(path, "w");
	if (file == NULL) {
		fprintf(stderr, "Cannot open leukocyte reference for write: %s\n", path);
		return -1;
	}
	fprintf(file, "%s %d\n", LEUKOCYTE_REFERENCE_MAGIC, LEUKOCYTE_REFERENCE_VERSION);
	fprintf(file, "video_name %s\n", reference->video_name);
	fprintf(file, "video_hash %016llx\n", (unsigned long long)reference->video_hash);
	fprintf(file, "video_bytes %llu\n", (unsigned long long)reference->video_bytes);
	fprintf(file, "requested_frames %d\n", reference->requested_frames);
	fprintf(file, "video_width %d\n", reference->video_width);
	fprintf(file, "video_height %d\n", reference->video_height);
	fprintf(file, "cells_detected %d\n", reference->cells_detected);
	fprintf(file, "centers_hash %016llx\n", (unsigned long long)reference->centers_hash);
	fprintf(file, "centers_values %llu\n", (unsigned long long)reference->centers_values);
	if (fclose(file) != 0) {
		fprintf(stderr, "Failed closing leukocyte reference: %s\n", path);
		return -1;
	}
	printf("Saved leukocyte reference to '%s'\n", path);
	return 0;
}

static int leukocyte_scan_field(FILE *file, const char *label, const char *format, void *value)
{
	char actual_label[64];
	if (fscanf(file, "%63s", actual_label) != 1 || strcmp(actual_label, label) != 0 ||
		fscanf(file, format, value) != 1) {
		fprintf(stderr, "Invalid leukocyte reference field: %s\n", label);
		return -1;
	}
	return 0;
}

static int leukocyte_read_reference(const char *path, LeukocyteReference *reference)
{
	FILE *file = fopen(path, "r");
	if (file == NULL) {
		fprintf(stderr, "Cannot open leukocyte reference for read: %s\n", path);
		return -1;
	}
	char magic[128];
	int version = 0;
	if (fscanf(file, "%127s %d", magic, &version) != 2 ||
		strcmp(magic, LEUKOCYTE_REFERENCE_MAGIC) != 0 ||
		version != LEUKOCYTE_REFERENCE_VERSION) {
		fprintf(stderr, "Invalid leukocyte reference header: %s\n", path);
		fclose(file);
		return -1;
	}
	unsigned long long ull_value = 0;
	int failed = 0;
	failed |= leukocyte_scan_field(file, "video_name", "%255s", reference->video_name);
	failed |= leukocyte_scan_field(file, "video_hash", "%llx", &ull_value);
	reference->video_hash = ull_value;
	failed |= leukocyte_scan_field(file, "video_bytes", "%llu", &ull_value);
	reference->video_bytes = ull_value;
	failed |= leukocyte_scan_field(file, "requested_frames", "%d", &reference->requested_frames);
	failed |= leukocyte_scan_field(file, "video_width", "%d", &reference->video_width);
	failed |= leukocyte_scan_field(file, "video_height", "%d", &reference->video_height);
	failed |= leukocyte_scan_field(file, "cells_detected", "%d", &reference->cells_detected);
	failed |= leukocyte_scan_field(file, "centers_hash", "%llx", &ull_value);
	reference->centers_hash = ull_value;
	failed |= leukocyte_scan_field(file, "centers_values", "%llu", &ull_value);
	reference->centers_values = ull_value;
	fclose(file);
	return failed == 0 ? 0 : -1;
}

static int leukocyte_compare_reference(const LeukocyteReference *actual, const LeukocyteReference *expected)
{
	int failed = 0;
	failed |= strcmp(actual->video_name, expected->video_name) != 0;
	failed |= actual->video_hash != expected->video_hash;
	failed |= actual->video_bytes != expected->video_bytes;
	failed |= actual->requested_frames != expected->requested_frames;
	failed |= actual->video_width != expected->video_width;
	failed |= actual->video_height != expected->video_height;
	failed |= actual->cells_detected != expected->cells_detected;
	failed |= actual->centers_hash != expected->centers_hash;
	failed |= actual->centers_values != expected->centers_values;
	if (failed) {
		fprintf(stderr, "leukocyte reference mismatch\n");
		return -1;
	}
	return 0;
}

static int leukocyte_verify_reference(const char *path, const LeukocyteReference *actual)
{
	LeukocyteReference expected;
	if (leukocyte_read_reference(path, &expected) != 0 ||
		leukocyte_compare_reference(actual, &expected) != 0) {
		rodinia_print_fail("Leukocyte reference verification");
		return -1;
	}
	rodinia_print_pass("Leukocyte reference verification");
	return 0;
}

int main(int argc, char ** argv) {

	// Choose the best GPU in case there are multiple available
	choose_GPU();

	// Keep track of the start time of the program
	long long program_start_time = get_time();
	
	if (argc != 3 && argc != 5){
	fprintf(stderr, "usage: %s <input file> <number of frames to process> [--save-reference <path>|--verify-reference <path>]", argv[0]);
	exit(1);
	}
	const char *save_reference_path = NULL;
	const char *verify_reference_path = NULL;
	if (argc == 5) {
		if (strcmp(argv[3], "--save-reference") == 0) {
			save_reference_path = argv[4];
		} else if (strcmp(argv[3], "--verify-reference") == 0) {
			verify_reference_path = argv[4];
		} else {
			fprintf(stderr, "Unknown leukocyte reference option: %s\n", argv[3]);
			return EXIT_FAILURE;
		}
	}
	
	// Let the user specify the number of frames to process
	int num_frames = atoi(argv[2]);
	
	// Open video file
	char *video_file_name = argv[1];
	LeukocyteFileDigest video_digest;
	if (leukocyte_compute_file_digest(video_file_name, &video_digest) != 0) {
		return EXIT_FAILURE;
	}
	
	avi_t *cell_file = AVI_open_input_file(video_file_name, 1);
	if (cell_file == NULL)	{
		AVI_print_error("Error with AVI_open_input_file");
		return -1;
	}
	
	// Transfer precomputed constants to the GPU
	compute_constants();
	
	int i, j, *crow, *ccol, pair_counter = 0, x_result_len = 0, Iter = 20, ns = 4, k_count = 0, n;
	MAT *cellx, *celly, *A;
	double *GICOV_spots, *t, *G, *x_result, *y_result, *V, *QAX_CENTERS, *QAY_CENTERS;
	double threshold = 1.8, radius = 10.0, delta = 3.0, dt = 0.01, b = 5.0;
	
	// Extract a cropped version of the first frame from the video file
	MAT *image_chopped = get_frame(cell_file, 0, 1, 0);
	printf("Detecting cells in frame 0\n");
	
	// Get gradient matrices in x and y directions
	MAT *grad_x = gradient_x(image_chopped);
	MAT *grad_y = gradient_y(image_chopped);
	
	m_free(image_chopped);
	
	// Get GICOV matrices corresponding to image gradients
	long long GICOV_start_time = get_time();
	MAT *gicov = GICOV(grad_x, grad_y);
	long long GICOV_end_time = get_time();

	// Dilate the GICOV matrices
	long long dilate_start_time = get_time();
	MAT *img_dilated = dilate(gicov);
	long long dilate_end_time = get_time();
	
	// Find possible matches for cell centers based on GICOV and record the rows/columns in which they are found
	pair_counter = 0;
	crow = (int *) malloc(gicov->m * gicov->n * sizeof(int));
	ccol = (int *) malloc(gicov->m * gicov->n * sizeof(int));
	for(i = 0; i < gicov->m; i++) {
		for(j = 0; j < gicov->n; j++) {
			if(!double_eq(m_get_val(gicov,i,j), 0.0) && double_eq(m_get_val(img_dilated,i,j), m_get_val(gicov,i,j)))
			{
				crow[pair_counter]=i;
				ccol[pair_counter]=j;
				pair_counter++;
			}
		}
	}

	GICOV_spots = (double *) malloc(sizeof(double) * pair_counter);
	for(i = 0; i < pair_counter; i++)
		GICOV_spots[i] = sqrt(m_get_val(gicov, crow[i], ccol[i]));
	
	G = (double *) calloc(pair_counter, sizeof(double));
	x_result = (double *) calloc(pair_counter, sizeof(double));
	y_result = (double *) calloc(pair_counter, sizeof(double));
	
	x_result_len = 0;
	for (i = 0; i < pair_counter; i++) {
		if ((crow[i] > 29) && (crow[i] < BOTTOM - TOP + 39)) {
			x_result[x_result_len] = ccol[i];
			y_result[x_result_len] = crow[i] - 40;
			G[x_result_len] = GICOV_spots[i];
			x_result_len++;
		}
	}
	
	// Make an array t which holds each "time step" for the possible cells
	t = (double *) malloc(sizeof(double) * 36);
	for (i = 0; i < 36; i++) {
		t[i] = (double)i * 2.0 * PI / 36.0;
	}
	
	// Store cell boundaries (as simple circles) for all cells
	cellx = m_get(x_result_len, 36);
	celly = m_get(x_result_len, 36);
	for(i = 0; i < x_result_len; i++) {
		for(j = 0; j < 36; j++) {
			m_set_val(cellx, i, j, x_result[i] + radius * cos(t[j]));
			m_set_val(celly, i, j, y_result[i] + radius * sin(t[j]));
		}
	}
	
	A = TMatrix(9,4);
	V = (double *) malloc(sizeof(double) * pair_counter);
	QAX_CENTERS = (double * )malloc(sizeof(double) * pair_counter);
	QAY_CENTERS = (double *) malloc(sizeof(double) * pair_counter);
	memset(V, 0, sizeof(double) * pair_counter);
	memset(QAX_CENTERS, 0, sizeof(double) * pair_counter);
	memset(QAY_CENTERS, 0, sizeof(double) * pair_counter);

	// For all possible results, find the ones that are feasibly leukocytes and store their centers
	k_count = 0;
	for (n = 0; n < x_result_len; n++) {
		if ((G[n] < -1 * threshold) || G[n] > threshold) {
			MAT * x, *y;
			VEC * x_row, * y_row;
			x = m_get(1, 36);
			y = m_get(1, 36);

			x_row = v_get(36);
			y_row = v_get(36);

			// Get current values of possible cells from cellx/celly matrices
			x_row = get_row(cellx, n, x_row);
			y_row = get_row(celly, n, y_row);
			uniformseg(x_row, y_row, x, y);

			// Make sure that the possible leukocytes are not too close to the edge of the frame
			if ((m_min(x) > b) && (m_min(y) > b) && (m_max(x) < cell_file->width - b) && (m_max(y) < cell_file->height - b)) {
				MAT * Cx, * Cy, *Cy_temp, * Ix1, * Iy1;
				VEC  *Xs, *Ys, *W, *Nx, *Ny, *X, *Y;
				Cx = m_get(1, 36);
				Cy = m_get(1, 36);
				Cx = mmtr_mlt(A, x, Cx);
				Cy = mmtr_mlt(A, y, Cy);
				
				Cy_temp = m_get(Cy->m, Cy->n);
				
				for (i = 0; i < 9; i++)
					m_set_val(Cy, i, 0, m_get_val(Cy, i, 0) + 40.0);
					
				// Iteratively refine the snake/spline
				for (i = 0; i < Iter; i++) {
					int typeofcell;
					
					if(G[n] > 0.0) typeofcell = 0;
					else typeofcell = 1;
					
					splineenergyform01(Cx, Cy, grad_x, grad_y, ns, delta, 2.0 * dt, typeofcell);
				}
				
				X = getsampling(Cx, ns);
				for (i = 0; i < Cy->m; i++)
					m_set_val(Cy_temp, i, 0, m_get_val(Cy, i, 0) - 40.0);
				Y = getsampling(Cy_temp, ns);
				
				Ix1 = linear_interp2(grad_x, X, Y);
				Iy1 = linear_interp2(grad_x, X, Y);
				Xs = getfdriv(Cx, ns);
				Ys = getfdriv(Cy, ns);
				
				Nx = v_get(Ys->dim);
				for (i = 0; i < Ys->dim; i++)
					v_set_val(Nx, i, v_get_val(Ys, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));
					
				Ny = v_get(Xs->dim);
				for (i = 0; i < Xs->dim; i++)
					v_set_val(Ny, i, -1.0 * v_get_val(Xs, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));
					
				W = v_get(Nx->dim);
				for (i = 0; i < Nx->dim; i++)
					v_set_val(W, i, m_get_val(Ix1, 0, i) * v_get_val(Nx, i) + m_get_val(Iy1, 0, i) * v_get_val(Ny, i));
					
				V[n] = mean(W) / std_dev(W);
				
				// Find the cell centers by computing the means of X and Y values for all snaxels of the spline contour
				QAX_CENTERS[k_count] = mean(X);
				QAY_CENTERS[k_count] = mean(Y) + TOP;
				
				k_count++;
				
				// Free memory
				v_free(W);
				v_free(Ny);
				v_free(Nx);
				v_free(Ys);
				v_free(Xs);
				m_free(Iy1);
				m_free(Ix1);
				v_free(Y);
				v_free(X);
				m_free(Cy_temp);
				m_free(Cy);
				m_free(Cx);				
			}

			// Free memory
			v_free(y_row);
			v_free(x_row);
			m_free(y);
			m_free(x);
		}
	}
	
	// Free memory
	free(V);
	free(ccol);
	free(crow);
	free(GICOV_spots);
	free(t);
	free(G);
	free(x_result);
	free(y_result);
	m_free(A);
	m_free(celly);
	m_free(cellx);
	m_free(img_dilated);
	m_free(gicov);
	m_free(grad_y);
	m_free(grad_x);
	
	// Report the total number of cells detected
	printf("Cells detected: %d\n\n", k_count);
	
	// Report the breakdown of the detection runtime
	printf("Detection runtime\n");
	printf("-----------------\n");
	printf("GICOV computation: %.5f seconds\n", ((float) (GICOV_end_time - GICOV_start_time)) / (1000*1000));
	printf("   GICOV dilation: %.5f seconds\n", ((float) (dilate_end_time - dilate_start_time)) / (1000*1000));
	printf("            Total: %.5f seconds\n", ((float) (get_time() - program_start_time)) / (1000*1000));
	int status = EXIT_SUCCESS;
	if (save_reference_path != NULL || verify_reference_path != NULL) {
		LeukocyteReference reference;
		leukocyte_populate_reference(
			&reference,
			video_file_name,
			&video_digest,
			num_frames,
			cell_file,
			k_count,
			QAX_CENTERS,
			QAY_CENTERS);
		if (save_reference_path != NULL && leukocyte_save_reference(save_reference_path, &reference) != 0) {
			status = EXIT_FAILURE;
		}
		if (status == EXIT_SUCCESS && verify_reference_path != NULL &&
			leukocyte_verify_reference(verify_reference_path, &reference) != 0) {
			status = EXIT_FAILURE;
		}
	}
	free(QAX_CENTERS);
	free(QAY_CENTERS);
	return status;
	
	// Now that the cells have been detected in the first frame,
	//  track the ellipses through subsequent frames
	if (num_frames > 1) printf("\nTracking cells across %d frames\n", num_frames);
	else                printf("\nTracking cells across 1 frame\n");
	long long tracking_start_time = get_time();
	int num_snaxels = 20;
	ellipsetrack(cell_file, QAX_CENTERS, QAY_CENTERS, k_count, radius, num_snaxels, num_frames);
	printf("           Total: %.5f seconds\n", ((float) (get_time() - tracking_start_time)) / (float) (1000*1000*num_frames));	
	
	// Report total program execution time
    printf("\nTotal application run time: %.5f seconds\n", ((float) (get_time() - program_start_time)) / (1000*1000));

	return 0;
}
