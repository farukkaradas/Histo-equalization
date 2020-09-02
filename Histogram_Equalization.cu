#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>


typedef struct{
    int w;  // image weight
    int h;  // image height
    unsigned int * img; // image matrix
} PGM_IMG;    


/*
    This function basicaly does round process with a threshold 0.5
    input : a float number 
    output : a rounded integer 
*/
//      
__device__ int Round(float num) { 
    return int(num)+0.5 < num ? int(num) + 1: int(num) + 0.5;
} 


/*
    This function finds cumulative distribution function of given pointer array
    input X: is 256 length pointer array  
    output Y: is cumulative version of X array
*/
__global__ void CDF(unsigned int *X, unsigned int *Y)
{ 
    __shared__ unsigned int XY[256];  // 256 bin size histogram array
    int n = 256; // histogram of 256 bin
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize XY array 
    if (idx < n) {
        XY[idx] = X[idx]; 
    }

    // reduction
    int index; 
    for (int stride = 1; stride < n; stride *= 2){
        __syncthreads();
        index = 2 * (idx + 1) *  stride - 1;
        if (index < 256) {
            XY[index] += XY[index - stride];
        }
    }
    // post reduction
    for (int stride = n/4; stride > 0; stride /= 2){
        __syncthreads();
        index = 2 * (idx + 1) * stride - 1;
        if(index + stride < n) {
            XY[index + stride] += XY[index];
        }
    }
    // writing result to output matrix
    if (idx < n) 
        Y[idx] = XY[idx];
}


/*
    This function calculate the histogram of given array
    X is input pointer array
    histo is histogram of X array
    n is size of X array
*/
__global__ void histogram(unsigned int *X, unsigned int *histo, long int n){
    // shared histogram array for each thread
    //The private histogram size needs to be small, to fit into shared memory
    __shared__ unsigned int private_histo[256];

    // initialize histogram arrays
    if (threadIdx.x < 256) 
        private_histo[threadIdx.x] = 0;
    __syncthreads();   
        // each thread add 1 histogram calculation with atomic add
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        atomicAdd(&private_histo[X[idx]], 1);
    }
    // barrier
    __syncthreads();
    // merge all shared array into output array (local memory)
    if (threadIdx.x < 256) {
        atomicAdd(&(histo[threadIdx.x]), private_histo[threadIdx.x]);
    }
}


/*
    This kernel makes equalization of given histogram 
    histo is input histogram array
    histo_out is output histogram array
    dcdfhisto is device cumulative histogram array
    cdfmin is minimun nonzero value of dcdf array
    n is size of image 
*/
__global__ void histogram_equalization(unsigned int *histo, unsigned int *histo_out,
                                       unsigned int *d_cdf_histo, int cdfmin, long int img_size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;                                        
    // Equalization  old histogram values with new ones
    // 255 is maximun number in the gray scale 
    if(idx < 256)
        histo_out[idx] = Round(((float)d_cdf_histo[idx] - cdfmin) / (img_size - cdfmin) * 255); 
    }


/*
    This kernel makes equalization of given image over histogram 
    img_in is input array
    img_out is output array
    histo is equalized histogram arrayy
    n is size of image 
*/
__global__ void image_equalization(unsigned int *img_in, unsigned int *img_out,
                                   unsigned int *histo, int n){
    /* Get the result image */
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n){
        if(histo[img_in[idx]] > 255){
            img_out[idx] = 255;
        }
        else{
            img_out[idx] = (unsigned int)histo[img_in[idx]];
        }
    }
}


/*
    This function reads PGM B/W images and returns a struct
    path is path of image
    returns a struct that keeps important informations of image
*/
PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    while(getc(in_file) != '\n');             /* skip to end of line */
    while (getc(in_file) == '#'){             /* skip comment lines */
        while (getc(in_file) != '\n');}       /* skip to end of comment line */
    fseek(in_file, -1, SEEK_CUR);             /* backup one character */
    fscanf(in_file, "%d", &result.w);
    fscanf(in_file, "%d", &result.h);
    fscanf(in_file, "%d\n", &v_max);
    printf("Input Image size: %d x %d\n", result.w, result.h);
    result.img = (unsigned int *)malloc(result.w * result.h * sizeof(unsigned int));     
    for(int i=0; i < result.w * result.h; i++)
        fscanf(in_file, "%d ",&result.img[i]);   
    fclose(in_file);
    return result;
}


/*
    This function write PGM (B/W) images and 
    path is output path of image
    img is a struct that keeps important informations of image
*/
void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P2\n%d %d\n255\n",img.w, img.h);
    for (int i = 0; i < img.h*img.w; ++i)
    {
        fprintf(out_file, " %d ", (img.img[i]));
        if (i%17 == 0) fprintf(out_file, "\n\r");
    }
    fclose(out_file);}


/*
    This function deallocate the structure of image
*/
void free_pgm(PGM_IMG img){
    free(img.img);
}


int main() 
{
    PGM_IMG img_in;
    PGM_IMG img_out;
    //Enter file name for the both char array for ease of review
    const char filename[] = "marcie.pgm";
    const char filename_out[] = "marcie_out.pgm";
    printf("%s is loading\n", filename);
    //====================================================================
    //                     DATA I/O and CPU Allocations.
    //==================================================================== 
    /* Reading image and neccessary details */
    img_in = read_pgm(filename);
    /* Preaparing output image details*/
    img_out.w = img_in.w;
    img_out.h = img_in.h;  
    long int img_size = img_in.w * img_in.h;        // 1D lenght of input image
    size_t size = img_size * sizeof(unsigned int);  // neccessary size for input image 
    size_t size_hist = 256 * sizeof(unsigned int);  // neccessary size for histogram arrays

    // d means devices 
    unsigned int *cdf_histo, *dev_img_in,
                 *d_histo, *d_histo_out, *d_cdf_histo, *dev_img_out;
    // Memory Allocations
    img_out.img = (unsigned int *)malloc(size);
    cdf_histo = (unsigned int *)malloc(size_hist);  
    //====================================================================
    //                              GPU
    //====================================================================    
    printf("GPU processes started\n");
    /* Declarations of  GPU Eximination time counters */
    cudaEvent_t start1, stop1;
    cudaEvent_t start2, stop2;
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);
    cudaEventCreate(&stop3);

    /* Making necessary allocations */
    cudaMalloc((void **) &dev_img_in, size);
    cudaMalloc((void **) &dev_img_out, size);
    cudaMalloc((void **) &d_histo, size_hist);
    cudaMalloc((void **) &d_histo_out, size_hist);
    cudaMalloc((void **) &d_cdf_histo, size_hist);

    cudaMemcpy(dev_img_in, img_in.img, size, cudaMemcpyHostToDevice);

    //==================================================================== 
    int blockSize;   // The launch configurator returned block size 
    int gridSize;    // The actual grid size needed, based on input size 

    // Round up according to array size 
    blockSize = 1024;
    gridSize = (img_size + blockSize - 1) / blockSize; 
    printf("Grid and Block number for histogram [%d,%d]\n", gridSize, blockSize);
    
	cudaEventRecord(start1);
    histogram<<<gridSize, blockSize, size_hist>>>(dev_img_in, d_histo, img_size);
    cudaEventRecord(stop1); // private_histo_kernel
    //====================================================================     
    
    cudaDeviceSynchronize();
    
    //====================================================================
    // Round up according to array size 
    blockSize = 256;
    gridSize = (256 + blockSize - 1) / blockSize; 
    printf("Grid and Block number for CDF kernel [%d,%d]\n", gridSize, blockSize);
    
	cudaEventRecord(start2);
    CDF<<<gridSize, blockSize, size_hist>>>(d_histo, d_cdf_histo);
    cudaEventRecord(stop2);
    //====================================================================	
    
    cudaDeviceSynchronize();    
    
    //====================================================================
    // Round up according to array size 
    blockSize = 256;
    gridSize = (256 + blockSize - 1) / blockSize; 
    printf("Grid and Block number for histogram_equalization [%d,%d]\n"
                                                , gridSize, blockSize);

    cudaEventRecord(start3); 

    cudaMemcpy(cdf_histo, d_cdf_histo, size_hist, cudaMemcpyDeviceToHost);
    // if this process can be done in gpu time of process will be decreased
    /* While loop for finding first nonzero element of CDF */
	int cdfmin = 0, i = 0; 
	while(cdfmin == 0){
        cdfmin = cdf_histo[i++];
    }
    histogram_equalization<<<gridSize, blockSize>>>(d_histo, d_histo_out,
                                            d_cdf_histo, cdfmin, img_size);  
    cudaEventRecord(stop3);
    //====================================================================
    
    cudaDeviceSynchronize(); 
    
    //====================================================================
    // Round up according to array size 
    blockSize = 1024;
    gridSize = (img_size + blockSize - 1) / blockSize;
    printf("Grid and Block number for image_equalization kernel [%d,%d]\n"
                                                    , gridSize, blockSize);
    
    image_equalization<<<gridSize, blockSize>>>(dev_img_in, dev_img_out,
                                                d_histo_out, img_size);
    //====================================================================
	cudaMemcpy(img_out.img, dev_img_out, size, cudaMemcpyDeviceToHost);
    //====================================================================
    //                              Timings
    //==================================================================== 

    cudaEventSynchronize(stop1);
    cudaEventSynchronize(stop2);
    cudaEventSynchronize(stop3);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    
	printf("GPU Execution time of histogram kernel is %.2f milliseconds\n", milliseconds);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start2, stop2);
    printf("GPU Execution time of CDF kernel is %.2f milliseconds\n", milliseconds);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start2, stop2);
    printf("GPU Execution time of histogram_equalization kernel is %.2f milliseconds\n", milliseconds);

    //====================================================================
    //                   DATA Output and Memory Deallocations
    //====================================================================     
    write_pgm(img_out, filename_out);    

    free(cdf_histo);
    free_pgm(img_in);
    free_pgm(img_out);    
	cudaFree(dev_img_in);
    cudaFree(dev_img_out);
    cudaFree(d_histo);
	cudaFree(d_histo_out);
    cudaFree(d_cdf_histo);    
    return 0;
}