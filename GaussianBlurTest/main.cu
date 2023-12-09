// -----------------------------------------------------------------------------
// * Name:       STB_readWriteTest.cxx
// * Purpose:    Testing STB image libary functions

// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#include <cmath>
#include <iostream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
//Single file image lib (header and implem in one)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//Note we are in build dir, files are 1 dir above
const char* filename = "../crossRatio.jpg";

int main(int argc, char **argv) {

  std::cout << "\nGaussian Blur Test\n" << std::endl;

  //Query image for info
  const int numChannels = 0; //Fix to 4 chans (RGBA?), to have same static alloc for input/output
  int inputWidth, inputHeight, inputChannels; //set by lib on load, but forced to 4 above
  int success = stbi_info(filename, &inputWidth, &inputHeight, &inputChannels);
  if(success){
    printf("Read file OK. x:%d y:%d inputChannels:%d\n", inputWidth, inputHeight, inputChannels);
  }
  else{
    printf("Error reading file\n");
  }

  //Load image into memory. malloc done internally, returns ptr to data
  unsigned char *inputData = stbi_load(filename, &inputWidth, &inputHeight, &inputChannels, numChannels);
  
  //Output buf
  //char *outputData = new char[inputWidth * inputHeight * numChannels];
  //unsigned char *outputData = inputData; //just point to input data, use for output

  //Flip vertically
  //stbi_flip_vertically_on_write(1);

  //Write same data (inputData) back out
  success = stbi_write_jpg("outputFile.jpg", inputWidth, inputHeight, inputChannels, inputData, 80); //last arg is quality, 1-100
  if(success){
    printf("Wrote file OK! x:%d y:%d inputChannels:%d\n", inputWidth, inputHeight, inputChannels);
  }

  else{
    printf("Error writing file\n");
  }

  return (EXIT_SUCCESS);
}
