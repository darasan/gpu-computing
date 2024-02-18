// -----------------------------------------------------------------------------
// * Name:       main.cu
// * Purpose:    Main CUDA program for SIFT algorithm implementation on GPU
// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#include <iostream>
#include <string>
#include <stdlib.h>
#include <cuda.h>
#include "SIFT_CUDA.hxx"

#define RUN_ON_CPU 1 //Enable to run computation on host CPU, disable to run on GPU
#define NUM_THREADS_1D 16 //Number of threads in 1 dimension of thread block

const char* filename = "../img/landscape512.jpg";
//const char* filename = "../img/UGA.jpg";


__device__ unsigned char getPixelColour(int x, int y, int width, int height, int numChannels, pxChannel colour, unsigned char *data)
{
  if (x < 0 || x >= width || y < 0 ||  y >= height || colour < 0 || colour > 3){
    return 0;
  }

  else{
    return *(data + ((x + y*width) * numChannels) + (int) colour);
  }
}

__device__ void setPixelColour(int x, int y, int width, int height, int numChannels, pxChannel colour, unsigned char *data, unsigned char value)
{
  if (x < 0 || x >= width || y < 0 ||  y >= height || colour < 0 || colour > 3 ){
  }

  else{
    *(data + ((x + y*width) * numChannels) + (int) colour) = value;
  }
}

__device__ bool CheckForLocalMaxInNeighbourScales(unsigned char *imgScale1, unsigned char *imgScale2, unsigned char *imgScale3, unsigned char curPxVal, int inputWidth, int inputHeight, int x, int y)
{
  int inputChannels = 1;
  bool is_min = true, is_max = true;
  unsigned char neighbor = 0;

  for (int dx : {-1,0,1}) {
      for (int dy : {-1,0,1}) {
          
          neighbor = getPixelColour(x+dx, y+dy, inputWidth, inputHeight, inputChannels, RED, imgScale1);
          if (neighbor > curPxVal) is_max = false;
          if (neighbor < curPxVal) is_min = false;
          //printf("neighbor GPU: %d\n", neighbor);

          neighbor = getPixelColour(x+dx, y+dy, inputWidth, inputHeight, inputChannels, RED, imgScale2);
          if (neighbor > curPxVal) is_max = false;
          if (neighbor < curPxVal) is_min = false;

          neighbor = getPixelColour(x+dx, y+dy, inputWidth, inputHeight, inputChannels, RED, imgScale3);
          // std::cout << "curPxVal: " <<  curPxVal<< " neighbor: " << neighbor << std::endl; 
          if (neighbor > curPxVal) is_max = false;
          if (neighbor < curPxVal) is_min = false;

          if (!is_min && !is_max) return false;
      }
  }
  return true;
}

__global__ void FINDMAX_CUDA(int inputWidth, int inputHeight, int inputChannels, unsigned char *cudaDeviceInputData, int *cudaDeviceResult)
{
  int max = 0, min = 0;
  int contrastThreshold = (int) 255.0 * 0.8; //Max level for detection
  int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
  int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
  int blockSum = 0;

  extern __shared__ int blockResult[NUM_THREADS_1D*NUM_THREADS_1D];

  //Init shared mem (once per block)
  if((threadIdx.x == 0) && (threadIdx.y == 0)){
    for(int i = 0; i<(NUM_THREADS_1D*NUM_THREADS_1D); i++){
      blockResult[i] = 0;
    }
  }
  __syncthreads();

  unsigned char *imgScale1 = cudaDeviceInputData;
  unsigned char curPxVal = getPixelColour(tid_x, tid_y, inputWidth, inputHeight, 1, RED, imgScale1);

  if (curPxVal < contrastThreshold) {
      if (CheckForLocalMaxInNeighbourScales(imgScale1, imgScale1, imgScale1, curPxVal, inputWidth, inputHeight, tid_x, tid_y)) {
          max++;
      }
      else{
          min++;
      }
  }

  blockResult[(tid_x+(tid_y*blockDim.x))] = max;
  __syncthreads();

  //Calculate partial sum for block
  for(int j = 0; j<(NUM_THREADS_1D*NUM_THREADS_1D); j++){
    blockSum +=  blockResult[j];
  }
  __syncthreads();
  
  //Write to global mem
  if((threadIdx.x == 0) && (threadIdx.y == 0)){
    cudaDeviceResult[blockIdx.x] = blockSum;
  }
}

int main(int argc, char **argv) {

    //Init CUDA device
    cudaError_t error;
    cudaDeviceProp deviceProp;
    int devID = 0;
    error = cudaGetDevice(&devID);
    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties returned error code " << error << "line " << __LINE__  << std::endl;
        exit(0);
    } else {
        std::cout << "GPU Device " << devID << deviceProp.name << " compute capability " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "totalGlobalMem: " << deviceProp.totalGlobalMem << " bytes" << " sharedMemPerBlock: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "maxGridSize x:" << deviceProp.maxGridSize[0] << " y:" <<  deviceProp.maxGridSize[1] << " maxThreadsPerBlock: " << deviceProp.maxThreadsPerBlock << std::endl << std::endl;
    }
    
    //Build Difference of Gaussians pyramid from base image
    SIFT_CUDA sift;
    Image img = Image(filename);
    sift.BuildGaussianPyramid(img);
    sift.BuildDoGPyramid(sift.gPyramid);  //note g for now, not DoG

#ifdef RUN_ON_CPU
    //Timer
    cudaEvent_t start;
    cudaEvent_t stop;
    float msecTotal;

    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);

    int keypointsFound = 0;
    int numOctaves = sift.gPyramid.octaves.size();

    for(int i=0; i<numOctaves;i++)
    {
        int numScales = sift.gPyramid.octaves[i].size(); //note g for now, not DoG
        for(int j=0; j<(numScales-2); j++)
        {
          int max = 0;
          Image currImg = sift.gPyramid.octaves[i][j+0];
          currImg.ConvertToGrayscale();

          Image prevImg = sift.gPyramid.octaves[i][j+1]; 
          prevImg.ConvertToGrayscale();

          Image nextImg = sift.gPyramid.octaves[i][j+2];
          nextImg.ConvertToGrayscale();

          max = sift.FindLocalMaxima(currImg, prevImg, nextImg);
          keypointsFound += max;
        }
    }
    std::cout << "Total keypoints found: " << keypointsFound << std::endl;

    //Stop timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    std::cout << "Processing time: " << msecTotal << " (ms)" << std::endl;
    
    exit(0);
#else

    //Configure kernel
    dim3 threadsPerBlock, numBlocks;
    threadsPerBlock = dim3(NUM_THREADS_1D,NUM_THREADS_1D);
    //dim3 numBlocks(ceil(currImg.width() / threadsPerBlock.x), ceil(currImg.height() / threadsPerBlock.y));
    numBlocks = dim3(16,1); 
    printf("numBlocks.x / y : %d total threadsPerBlock: %d\n", numBlocks.x, threadsPerBlock.x * threadsPerBlock.x);

    //Allocate device memory for input image
    unsigned char *cudaDeviceInputData;
    cudaMalloc((void **)&cudaDeviceInputData, currImg.size());

    //Allocate and init device memory for result. Array of ints, each thread writes num keypoints found
    int resultSize = numBlocks.x * numBlocks.y; //Calculate sum for each block, then store global result for all blocks
    printf("resultSize: %d blocks\n", resultSize);
    int *cudaDeviceResult;
    cudaMalloc((void **)&cudaDeviceResult, resultSize* sizeof(int));
    cudaMemset(cudaDeviceResult, 0, resultSize * sizeof(int));
    
    //Allocate and init host memory for result. One element for each block sum
    int hostResultData[resultSize] = {0};

    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);

    //Copy image from host to device
    cudaMemcpy(cudaDeviceInputData, currImg.data(), currImg.size(), cudaMemcpyHostToDevice);

    //Run kernel
    std::cout << "Run kernel" << std::endl;
    FINDMAX_CUDA<<<numBlocks, threadsPerBlock>>>(currImg.width(), currImg.height(), currImg.numChannels(), cudaDeviceInputData, cudaDeviceResult);

    //Copy back to host 
    std::cout << "Done. Copy result to host" << std::endl << std::endl;
    cudaMemcpy(hostResultData, cudaDeviceResult, resultSize * sizeof(int), cudaMemcpyDeviceToHost);

    //Stop timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    //Check num keypoints in results
    int total = 0;
    for (int i = 0; i < resultSize; i++){
      total += hostResultData[i];
      //printf("hostResultData: %d i:%d\n", hostResultData[i], i);
    }

    printf("Total kps: %d\n", total);
    printf("numBlocks.x / y : %d total threadsPerBlock: %d\n", numBlocks.x, threadsPerBlock.x * threadsPerBlock.x);
    std::cout << "Processing time: " << msecTotal << " (ms)" << std::endl;


    sift.FreePyramidMemory();
    exit(0);
#endif
}
