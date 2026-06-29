/* 
 * Copyright (c) 2009, Jiri Matela
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <unistd.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <getopt.h>
#include <vector>

#include "common.h"
#include "components.h"
#include "dwt.h"
#include "dwt_reference.h"
#include "../../common/rodinia_verify.h"

struct dwt {
    char * srcFilename;
    char * outFilename;
    unsigned char *srcImg;
    int pixWidth;
    int pixHeight;
    int components;
    int dwtLvls;
};

static int divRndUpInt(int value, int divisor)
{
    return (value / divisor) + ((value % divisor) ? 1 : 0);
}

static int mirrorIndex(int index, int size)
{
    if (index >= size) {
        return 2 * size - 2 - index;
    }
    if (index < 0) {
        return -index;
    }
    return index;
}

static unsigned char sampleToChar(int sample)
{
    int value = sample + 128;
    if (value > 255) {
        return 255;
    }
    if (value < 0) {
        return 0;
    }
    return (unsigned char)value;
}

static void cpuDwt53Line(const std::vector<int>& input, std::vector<int>& output)
{
    std::vector<int> transformed = input;
    int count = (int)transformed.size();

    for (int index = 1; index < count; index += 2) {
        int previous = transformed[mirrorIndex(index - 1, count)];
        int next = transformed[mirrorIndex(index + 1, count)];
        transformed[index] -= (previous + next) / 2;
    }

    for (int index = 0; index < count; index += 2) {
        int previous = transformed[mirrorIndex(index - 1, count)];
        int next = transformed[mirrorIndex(index + 1, count)];
        transformed[index] += (previous + next + 2) / 4;
    }

    output.clear();
    output.reserve(count);
    for (int index = 0; index < count; index += 2) {
        output.push_back(transformed[index]);
    }
    for (int index = 1; index < count; index += 2) {
        output.push_back(transformed[index]);
    }
}

static void cpuForwardDwt53Level(
    const std::vector<int>& input,
    int width,
    int height,
    std::vector<int>& output)
{
    std::vector<int> vertical(width * height);
    std::vector<int> column(height);
    std::vector<int> transformedColumn;
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            column[y] = input[y * width + x];
        }
        cpuDwt53Line(column, transformedColumn);
        for (int y = 0; y < height; y++) {
            vertical[y * width + x] = transformedColumn[y];
        }
    }

    std::vector<int> transformed(width * height);
    std::vector<int> row(width);
    std::vector<int> transformedRow;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            row[x] = vertical[y * width + x];
        }
        cpuDwt53Line(row, transformedRow);
        for (int x = 0; x < width; x++) {
            transformed[y * width + x] = transformedRow[x];
        }
    }

    int lowWidth = divRndUpInt(width, 2);
    int highWidth = width / 2;
    int lowHeight = divRndUpInt(height, 2);
    int highHeight = height / 2;
    output.assign(width * height, 0);

    for (int y = 0; y < lowHeight; y++) {
        for (int x = 0; x < lowWidth; x++) {
            output[y * lowWidth + x] = transformed[y * width + x];
        }
    }

    int offset = lowWidth * lowHeight;
    for (int y = 0; y < lowHeight; y++) {
        for (int x = 0; x < highWidth; x++) {
            output[offset + y * highWidth + x] = transformed[y * width + lowWidth + x];
        }
    }

    offset += highWidth * lowHeight;
    for (int y = 0; y < highHeight; y++) {
        for (int x = 0; x < lowWidth; x++) {
            output[offset + y * lowWidth + x] = transformed[(lowHeight + y) * width + x];
        }
    }

    offset += lowWidth * highHeight;
    for (int y = 0; y < highHeight; y++) {
        for (int x = 0; x < highWidth; x++) {
            output[offset + y * highWidth + x] =
                transformed[(lowHeight + y) * width + lowWidth + x];
        }
    }
}

static std::vector<int> cpuForwardDwt53Reference(const struct dwt *d)
{
    int width = d->pixWidth;
    int height = d->pixHeight;
    std::vector<int> current(width * height);
    for (int index = 0; index < width * height; index++) {
        current[index] = (int)d->srcImg[index] - 128;
    }

    std::vector<int> reference(width * height, 0);
    for (int level = 0; level < d->dwtLvls; level++) {
        std::vector<int> levelOutput;
        cpuForwardDwt53Level(current, width, height, levelOutput);
        for (int index = 0; index < width * height; index++) {
            reference[index] = levelOutput[index];
        }

        int nextWidth = divRndUpInt(width, 2);
        int nextHeight = divRndUpInt(height, 2);
        if (level + 1 == d->dwtLvls) {
            break;
        }
        current.assign(levelOutput.begin(), levelOutput.begin() + nextWidth * nextHeight);
        width = nextWidth;
        height = nextHeight;
    }
    return reference;
}

template <typename T>
int verifyDWTCPUReference(T *component_cuda, const struct dwt *d, int forward)
{
    (void)component_cuda;
    (void)d;
    (void)forward;
    fprintf(stderr, "DWT2D CPU verification currently supports forward 5/3 integer data only\n");
    return -1;
}

template <>
int verifyDWTCPUReference<int>(int *component_cuda, const struct dwt *d, int forward)
{
    if (!forward) {
        fprintf(stderr, "DWT2D CPU verification currently supports forward transforms only\n");
        return -1;
    }
    if (d->components != 1) {
        fprintf(stderr, "DWT2D CPU verification currently supports one component only\n");
        return -1;
    }

    int samples = d->pixWidth * d->pixHeight;
    std::vector<int> actual(samples);
    cudaMemcpy(actual.data(), component_cuda, samples * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError("Copy DWT2D verification output to host");

    std::vector<int> expected = cpuForwardDwt53Reference(d);
    for (int index = 0; index < samples; index++) {
        if (actual[index] != expected[index]) {
            fprintf(stderr,
                    "DWT2D CPU reference mismatch at sample %d: actual=%d expected=%d actual_byte=%u expected_byte=%u\n",
                    index,
                    actual[index],
                    expected[index],
                    sampleToChar(actual[index]),
                    sampleToChar(expected[index]));
            return -1;
        }
    }

    return rodinia_print_pass("DWT2D CPU reference verification");
}

int getImg(char * srcFilename, unsigned char *srcImg, int inputSize)
{
    // printf("Loading ipnput: %s\n", srcFilename);
    char *path = "";
    char *newSrc = NULL;
    
    if((newSrc = (char *)malloc(strlen(srcFilename)+strlen(path)+1)) != NULL)
    {
        newSrc[0] = '\0';
        strcat(newSrc, path);
        strcat(newSrc, srcFilename);
        srcFilename= newSrc;
    }
    printf("Loading ipnput: %s\n", srcFilename);

    //srcFilename = strcat("../../data/dwt2d/",srcFilename);
    //read image
    int i = open(srcFilename, O_RDONLY, 0644);
    if (i == -1) { 
        error(0,errno,"cannot access %s", srcFilename);
        return -1;
    }
    int ret = read(i, srcImg, inputSize);
    printf("precteno %d, inputsize %d\n", ret, inputSize);
    close(i);

    return 0;
}


void usage() {
    printf("dwt [otpions] src_img.rgb <out_img.dwt>\n\
  -d, --dimension\t\tdimensions of src img, e.g. 1920x1080\n\
  -c, --components\t\tnumber of color components, default 3\n\
  -b, --depth\t\t\tbit depth, default 8\n\
  -l, --level\t\t\tDWT level, default 3\n\
  -D, --device\t\t\tcuda device\n\
  -f, --forward\t\t\tforward transform\n\
  -r, --reverse\t\t\treverse transform\n\
  -9, --97\t\t\t9/7 transform\n\
  -5, --53\t\t\t5/3 transform\n\
  -w  --write-visual\t\twrite output in visual (tiled) fashion instead of the linear\n\
      --verify-cpu\t\tcompare output coefficients with CPU reference\n\
      --save-reference <file>\tsave output summary reference\n\
      --verify-reference <file>\tverify output summary reference\n");
}

template <typename T>
int processDWT(
    struct dwt *d,
    int forward,
    int writeVisual,
    int verifyCpu,
    Dwt2DReferenceContext *referenceContext)
{
    int componentSize = d->pixWidth*d->pixHeight*sizeof(T);
    int status = 0;
    
    T *c_r_out, *backup ;
    cudaMalloc((void**)&c_r_out, componentSize); //< aligned component size
    cudaCheckError("Alloc device memory");
    cudaMemset(c_r_out, 0, componentSize);
    cudaCheckError("Memset device memory");
    
    cudaMalloc((void**)&backup, componentSize); //< aligned component size
    cudaCheckError("Alloc device memory");
    cudaMemset(backup, 0, componentSize);
    cudaCheckError("Memset device memory");
	
    if (d->components == 3) {
        /* Alloc two more buffers for G and B */
        T *c_g_out, *c_b_out;
        cudaMalloc((void**)&c_g_out, componentSize); //< aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_g_out, 0, componentSize);
        cudaCheckError("Memset device memory");
        
        cudaMalloc((void**)&c_b_out, componentSize); //< aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_b_out, 0, componentSize);
        cudaCheckError("Memset device memory");
        
        /* Load components */
        T *c_r, *c_g, *c_b;
        cudaMalloc((void**)&c_r, componentSize); //< R, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_r, 0, componentSize);
        cudaCheckError("Memset device memory");

        cudaMalloc((void**)&c_g, componentSize); //< G, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_g, 0, componentSize);
        cudaCheckError("Memset device memory");

        cudaMalloc((void**)&c_b, componentSize); //< B, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_b, 0, componentSize);
        cudaCheckError("Memset device memory");

        rgbToComponents(c_r, c_g, c_b, d->srcImg, d->pixWidth, d->pixHeight);
		

        /* Compute DWT and always store into file */

        nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        nStage2dDWT(c_g, c_g_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        nStage2dDWT(c_b, c_b_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        status |= dwt2d_record_component_hash(referenceContext, 0, c_r_out, d->pixWidth * d->pixHeight);
        status |= dwt2d_record_component_hash(referenceContext, 1, c_g_out, d->pixWidth * d->pixHeight);
        status |= dwt2d_record_component_hash(referenceContext, 2, c_b_out, d->pixWidth * d->pixHeight);
        if (verifyCpu) {
            fprintf(stderr, "DWT2D CPU verification currently supports one component only\n");
            rodinia_print_fail("DWT2D CPU reference verification");
            status = 1;
        }
     
        // -------test----------
        // T *h_r_out=(T*)malloc(componentSize);
		// cudaMemcpy(h_r_out, c_g_out, componentSize, cudaMemcpyDeviceToHost);
        // int ii;
		// for(ii=0;ii<componentSize/sizeof(T);ii++) {
			// fprintf(stderr, "%d ", h_r_out[ii]);
			// if((ii+1) % (d->pixWidth) == 0) fprintf(stderr, "\n");
        // }
        // -------test----------
        
		
        /* Store DWT to file */
#ifdef OUTPUT        
        if (writeVisual) {
            writeNStage2DDWT(c_r_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".r");
            writeNStage2DDWT(c_g_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".g");
            writeNStage2DDWT(c_b_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".b");
        } else {
            writeLinear(c_r_out, d->pixWidth, d->pixHeight, d->outFilename, ".r");
            writeLinear(c_g_out, d->pixWidth, d->pixHeight, d->outFilename, ".g");
            writeLinear(c_b_out, d->pixWidth, d->pixHeight, d->outFilename, ".b");
        }
#endif


        cudaFree(c_r);
        cudaCheckError("Cuda free");
        cudaFree(c_g);
        cudaCheckError("Cuda free");
        cudaFree(c_b);
        cudaCheckError("Cuda free");
        cudaFree(c_g_out);
        cudaCheckError("Cuda free");
        cudaFree(c_b_out);
        cudaCheckError("Cuda free");

    } 
    else if (d->components == 1) {
        //Load component
        T *c_r;
        cudaMalloc((void**)&(c_r), componentSize); //< R, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_r, 0, componentSize);
        cudaCheckError("Memset device memory");

        bwToComponent(c_r, d->srcImg, d->pixWidth, d->pixHeight);

        // Compute DWT 
        nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        status |= dwt2d_record_component_hash(referenceContext, 0, c_r_out, d->pixWidth * d->pixHeight);

        // Store DWT to file 
// #ifdef OUTPUT        
        if (writeVisual) {
            writeNStage2DDWT(c_r_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".out");
        } else {
            writeLinear(c_r_out, d->pixWidth, d->pixHeight, d->outFilename, ".lin.out");
        }
// #endif
        if (verifyCpu && verifyDWTCPUReference<T>(c_r_out, d, forward) != 0) {
            rodinia_print_fail("DWT2D CPU reference verification");
            status = 1;
        }
        cudaFree(c_r);
        cudaCheckError("Cuda free");
    }

    cudaFree(c_r_out);
    cudaCheckError("Cuda free device");
    cudaFree(backup);
    cudaCheckError("Cuda free device");
    return status;
}

int main(int argc, char **argv) 
{
    int optindex = 0;
    int ch;
    struct option longopts[] = {
        {"dimension",   required_argument, 0, 'd'}, //dimensions of src img
        {"components",  required_argument, 0, 'c'}, //numger of components of src img
        {"depth",       required_argument, 0, 'b'}, //bit depth of src img
        {"level",       required_argument, 0, 'l'}, //level of dwt
        {"device",      required_argument, 0, 'D'}, //cuda device
        {"forward",     no_argument,       0, 'f'}, //forward transform
        {"reverse",     no_argument,       0, 'r'}, //reverse transform
        {"97",          no_argument,       0, '9'}, //9/7 transform
        {"53",          no_argument,       0, '5' }, //5/3transform
        {"write-visual",no_argument,       0, 'w' }, //write output (subbands) in visual (tiled) order instead of linear
        {"verify-cpu",  no_argument,       0, 'V' }, //verify output coefficients against CPU reference
        {"save-reference", required_argument, 0, 1000},
        {"verify-reference", required_argument, 0, 1001},
        {"help",        no_argument,       0, 'h'},
        {0,             0,                 0,  0 }
    };
    
    int pixWidth    = 0; //<real pixWidth
    int pixHeight   = 0; //<real pixHeight
    int compCount   = 3; //number of components; 3 for RGB or YUV, 4 for RGBA
    int bitDepth    = 8; 
    int dwtLvls     = 3; //default numuber of DWT levels
    int device      = 0;
    int forward     = 1; //forward transform
    int dwt97       = 1; //1=dwt9/7, 0=dwt5/3 transform
    int writeVisual = 0; //write output (subbands) in visual (tiled) order instead of linear
    int verifyCpu   = 0;
    const char *saveReferencePath = NULL;
    const char *verifyReferencePath = NULL;
    char * pos;

    while ((ch = getopt_long(argc, argv, "d:c:b:l:D:fr95whV", longopts, &optindex)) != -1) {
        switch (ch) {
        case 'd':
            pixWidth = atoi(optarg);
            pos = strstr(optarg, "x");
            if (pos == NULL || pixWidth == 0 || (strlen(pos) >= strlen(optarg))) {
                usage();
                return -1;
            }
            pixHeight = atoi(pos+1);
            break;
        case 'c':
            compCount = atoi(optarg);
            break;
        case 'b':
            bitDepth = atoi(optarg);
            break;
        case 'l':
            dwtLvls = atoi(optarg);
            break;
        case 'D':
            device = atoi(optarg);
            break;
        case 'f':
            forward = 1;
            break;
        case 'r':
            forward = 0;
            break;
        case '9':
            dwt97 = 1;
            break;
        case '5':
            dwt97 = 0;
            break;
        case 'w':
            writeVisual = 1;
            break;
        case 'V':
            verifyCpu = 1;
            break;
        case 1000:
            if (verifyReferencePath != NULL) {
                fprintf(stderr, "Only one DWT2D reference mode may be specified\n");
                return -1;
            }
            saveReferencePath = optarg;
            break;
        case 1001:
            if (saveReferencePath != NULL) {
                fprintf(stderr, "Only one DWT2D reference mode may be specified\n");
                return -1;
            }
            verifyReferencePath = optarg;
            break;
        case 'h':
            usage();
            return 0;
        case '?':
            return -1;
        default :
            usage();
            return -1;
        }
    }
	argc -= optind;
	argv += optind;

    if (argc == 0) { // at least one filename is expected
        printf("Please supply src file name\n");
        usage();
        return -1;
    }

    if (pixWidth <= 0 || pixHeight <=0) {
        printf("Wrong or missing dimensions\n");
        usage();
        return -1;
    }

    if (forward == 0) {
        writeVisual = 0; //do not write visual when RDWT
    }

    // device init
    int devCount;
    cudaGetDeviceCount(&devCount);
    cudaCheckError("Get device count");
    if (devCount == 0) {
        printf("No CUDA enabled device\n");
        return -1;
    } 
    if (device < 0 || device > devCount -1) {
        printf("Selected device %d is out of bound. Devices on your system are in range %d - %d\n", 
               device, 0, devCount -1);
        return -1;
    }
    cudaDeviceProp devProp;                                          
    cudaGetDeviceProperties(&devProp, device);  
    cudaCheckError("Get device properties");
    if (devProp.major < 1) {                                         
        printf("Device %d does not support CUDA\n", device);
        return -1;
    }                                                                   
    printf("Using device %d: %s\n", device, devProp.name);
    cudaSetDevice(device);
    cudaCheckError("Set selected device");

    struct dwt *d;
    d = (struct dwt *)malloc(sizeof(struct dwt));
    d->srcImg = NULL;
    d->pixWidth = pixWidth;
    d->pixHeight = pixHeight;
    d->components = compCount;
    d->dwtLvls  = dwtLvls;

    // file names
    d->srcFilename = (char *)malloc(strlen(argv[0]) + 1);
    strcpy(d->srcFilename, argv[0]);
    if (argc == 1) { // only one filename supplyed
        d->outFilename = (char *)malloc(strlen(d->srcFilename) + strlen(".dwt") + 1);
        strcpy(d->outFilename, d->srcFilename);
        strcpy(d->outFilename+strlen(d->srcFilename), ".dwt");
    } else {
        d->outFilename = strdup(argv[1]);
    }

    //Input review
    printf("Source file:\t\t%s\n", d->srcFilename);
    printf(" Dimensions:\t\t%dx%d\n", pixWidth, pixHeight);
    printf(" Components count:\t%d\n", compCount);
    printf(" Bit depth:\t\t%d\n", bitDepth);
    printf(" DWT levels:\t\t%d\n", dwtLvls);
    printf(" Forward transform:\t%d\n", forward);
    printf(" 9/7 transform:\t\t%d\n", dwt97);
    
    //data sizes
    int inputSize = pixWidth*pixHeight*compCount; //<amount of data (in bytes) to proccess

    //load img source image
    cudaMallocHost((void **)&d->srcImg, inputSize);
    cudaCheckError("Alloc host memory");
    if (getImg(d->srcFilename, d->srcImg, inputSize) == -1) 
        return -1;

    Dwt2DReferenceContext referenceContext;
    Dwt2DReferenceContext *referenceContextPtr = NULL;
    if (saveReferencePath != NULL || verifyReferencePath != NULL) {
        if (compCount <= 0 || compCount > DWT2D_MAX_COMPONENTS) {
            fprintf(stderr, "DWT2D reference mode supports 1-%d components\n", DWT2D_MAX_COMPONENTS);
            return -1;
        }
        Dwt2DFileDigest digest;
        if (dwt2d_compute_file_digest(d->srcFilename, &digest) != 0) {
            return -1;
        }
        referenceContext.save_path = saveReferencePath;
        referenceContext.verify_path = verifyReferencePath;
        dwt2d_init_reference(
            &referenceContext.reference,
            d->srcFilename,
            &digest,
            pixWidth,
            pixHeight,
            compCount,
            bitDepth,
            dwtLvls,
            forward,
            dwt97,
            writeVisual,
            dwt97 ? (int)sizeof(float) : (int)sizeof(int),
            dwt97 ? "float" : "int");
        referenceContextPtr = &referenceContext;
    }

    /* DWT */
    int status = 0;
    if (forward == 1) {
        if(dwt97 == 1 )
            status = processDWT<float>(d, forward, writeVisual, verifyCpu, referenceContextPtr);
        else // 5/3
            status = processDWT<int>(d, forward, writeVisual, verifyCpu, referenceContextPtr);
    }
    else { // reverse
        if(dwt97 == 1 )
            status = processDWT<float>(d, forward, writeVisual, verifyCpu, referenceContextPtr);
        else // 5/3
            status = processDWT<int>(d, forward, writeVisual, verifyCpu, referenceContextPtr);
    }
    if (status == 0 && dwt2d_finish_reference(referenceContextPtr) != 0) {
        status = 1;
    }

    //writeComponent(r_cuda, pixWidth, pixHeight, srcFilename, ".g");
    //writeComponent(g_wave_cuda, 512000, ".g");
    //writeComponent(g_cuda, componentSize, ".g");
    //writeComponent(b_wave_cuda, componentSize, ".b");
    cudaFreeHost(d->srcImg);
    cudaCheckError("Cuda free host");

    return status;
}
