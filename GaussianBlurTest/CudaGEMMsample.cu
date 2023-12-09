// -----------------------------------------------------------------------------
// * Name:       main_gpu.cxx
// * Purpose:    Driver for matrix multiplication on GPU
// * History:    Christophe Picard, Fall 2021
// -----------------------------------------------------------------------------

// includes, system
#include <cmath>
#include <iostream>
#include <string>

#include <cuda.h>

// Parsing command line options using cxxopts 
// https://github.com/jarro2783/cxxopts.git
#include "args.hxx"

// Matrix manipulation function
#include "matrix_utils.h"

// Define different gemm kernel
#include <gemm_kernel.cuh>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
//Single file image lib (header and implem in one)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define REAL float
#define BLOCK_SIZE 32


//New approach - use working CUDA config and add kernel for Gaussian from Github sample. Add STB image also
///
/// Top level driver
///
int main(int argc, char **argv) {

  std::cout << "\nGaussian Blur Test\n" << std::endl;

  // Define parser 
  args::ArgumentParser parser("gemm_cuda", "Gaussian Blur Test");

  // Set parser value
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::ValueFlag<int> widthA(parser, "widthA", "Width of matrix A", {"wA"}, 256);
  args::ValueFlag<int> widthB(parser, "widthB", "Width of matrix B", {"wB"}, 256);
  args::ValueFlag<int> heightA(parser, "heightA", "Height of matrix A", {"hA"},256);
  args::ValueFlag<int> heightB(parser, "heightB", "Height of matrix B", {"hB"}, 256);
  
  // Invoke parser
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  } catch (args::ValidationError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  // Initialize matrix dimensions
  int WA = args::get(widthA);
  int WB = args::get(widthB);
  int HA = args::get(heightA);
  int HB = args::get(heightB);
  int WC = WA;
  int HC = HB;

  int success = 0;

  //Query image for info
  const int numChannels = 0; //Fix to 4 chans (RGBA?), to have same static alloc for input/output
  int inputWidth, inputHeight, inputChannels; //set by lib on load, but forced to 4 above
  success = stbi_info("../Val.jpg", &inputWidth, &inputHeight, &inputChannels); //Note we are in build dir, files are 1 above
  printf("STB image info:  loadOK:%d x:%d y:%d inputChannels:%d\n", success, inputWidth, inputHeight, inputChannels);

  //Load it into memory. malloc done internally, returns ptr to data
  unsigned char *inputData = stbi_load("../Val.jpg", &inputWidth, &inputHeight, &inputChannels, numChannels);

  //Try resize
  //From docs: If you pass NULL or zero for the output_pixels, we will allocate the output buffer
  //for you and return it from the function (free with free() or STBIR_FREE)

  //STBIRDEF unsigned char * stbir_resize_uint8_srgb( const unsigned char *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
  //unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes, stbir_pixel_layout pixel_type );
  
  //TODO try flip vertical

  //Output buf
  //char *outputData = new char[inputWidth * inputHeight * numChannels];
  //unsigned char *outputData = inputData; //just point to input data, use for output


  //Write out
  success = stbi_write_jpg("outputFile.jpg", inputWidth, inputHeight, inputChannels, inputData, 80); //last is quality, 1-100. 0 default?
  if(success){
    printf("Wrote file OK!\n");
    printf("x:%d y:%d inputChannels:%d\n", inputWidth, inputHeight, inputChannels);
  }

  // Setup CUDA environnement 
  cudaError_t error;

  cudaDeviceProp deviceProp;
  int devID = 0;
  error = cudaGetDevice(&devID);

  if (error != cudaSuccess) {
    printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
  }

  error = cudaGetDeviceProperties(&deviceProp, devID);

  if (deviceProp.computeMode == cudaComputeModeProhibited) {
    std::cerr << "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice() ." <<std::endl;
    exit(EXIT_SUCCESS);
  }

  if (error != cudaSuccess) {
    printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
  } else {
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
  }

  // utilities
  cudaEvent_t start;
  cudaEvent_t stop;
  float msecTotal;

  // allocate host memory for matrices A and B
  unsigned int size_A = WA * HA;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A = (float *)malloc(mem_size_A);
  unsigned int size_B = WB * HB;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B = (float *)malloc(mem_size_B);
  
  // initialize host memory
  fill_random<REAL>(h_A, WA, HA);
  fill_random<REAL>(h_B, WB, HB);
 
  // allocate device memory
  float *d_A;
  cudaMalloc((void **)&d_A, mem_size_A);
  float *d_B;
  cudaMalloc((void **)&d_B, mem_size_B);

  // allocate device memory for result
  unsigned int size_C = WA * HB;
  unsigned int mem_size_C = sizeof(float) * size_C;
  float *d_C;
  cudaMalloc((void **)&d_C, mem_size_C);

  // allocate host memory for the result
  float *h_C = (float *)malloc(mem_size_C);

  dim3 threads, grid;

  // create and start timer
  cudaEventCreate(&start);
  cudaEventRecord(start, NULL);
 
  // copy host memory to device
  cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

  // setup execution parameters
  threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
  grid = dim3(WC / threads.x, HC / threads.y);
  
  // execute the kernel
  gemm_naive<<<grid, threads>>>(d_C, d_A, d_B, WA, WB);

  // copy result from device to host
  cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

  // stop and destroy timer
  cudaEventCreate(&stop);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);

  /* Performance computation, results and performance printing ------------ */
  auto flop = 2 * (float)WC * (float)HC * (float)WA;

  std::cout << " == Performances " << std::endl;
  std::cout << "\t Processing time: " << msecTotal << " (ms)"
            << std::endl;
  std::cout << "\t GFLOPS: " << flop / msecTotal / 1e+6 << std::endl;

  return (EXIT_SUCCESS);
}
