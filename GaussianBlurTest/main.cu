// -----------------------------------------------------------------------------
// * Name:       STB_readWriteTest.cxx
// * Purpose:    Testing STB image libary functions

// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#include <cmath>
#include <iostream>
#include <string>
#include <stdlib.h> 

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
//Single file image lib (header and implem in one)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define GRAYSCALE 0
#define SEPIA 0
#define NO_FILTER 0
#define RED_FILTER 1
#define GAUSSIAN 0

const char* filename = "../LivingRoom.jpg";

int main(int argc, char **argv) {

  std::cout << "\nGaussian Blur Test\n" << std::endl;

  //Query image for info
  int inputWidth, inputHeight, inputChannels; //set by lib on load, but forced to 4 above
  int success = stbi_info(filename, &inputWidth, &inputHeight, &inputChannels);
  if(success){
    printf("Read file OK. x:%d y:%d inputChannels:%d Size:%d\n", inputWidth, inputHeight, inputChannels, (inputWidth*inputHeight*inputChannels));

  }
  else{
    printf("Error reading file\n");
  }

  //Load image into memory. malloc done internally, returns ptr to data
  unsigned char *inputData = stbi_load(filename, &inputWidth, &inputHeight, &inputChannels, 0); //last arg is num chans to load, set 0 to load all
  int inputImageSize = (inputWidth * inputHeight * inputChannels);

  //Calculate num channels and size for output file
#if GRAYSCALE
  int outputChannels = inputChannels == 4 ? 2 : 1;
#else
  int outputChannels = inputChannels;
#endif

  int outputWidth = inputWidth;
  int outputHeight = inputHeight; //Assume we always write image of same size as input
  int outputImageSize = (outputWidth * outputHeight * outputChannels);

  printf("Allocate memory\n");
  unsigned char *outputData = new unsigned char[outputImageSize]; 
  if(outputData == NULL) {
    printf("Unable to allocate memory for output image\n");
    exit(1);
  }

#if GRAYSCALE
  printf("Convert to gray\n"); //calc average of rgb pixels
  for(unsigned char *p = inputData, *pg = outputData; p != inputData + inputImageSize; p += inputChannels, pg += outputChannels)
  {
    *pg = (uint8_t)((*p + *(p + 1) + *(p + 2))/3.0);
    if(inputChannels == 4) {
      *(pg + 1) = *(p + 3);
    }
  }
#elif SEPIA
  printf("Sepia filter\n");
  //Sepia filter coefficients from https://stackoverflow.com/questions/1061093/how-is-a-sepia-tone-created
  for(unsigned char *p = inputData, *pg = outputData; p != inputData + inputImageSize; p += inputChannels, pg += outputChannels)
  {
    *pg       = (uint8_t)fmin(0.393 * *p + 0.769 * *(p + 1) + 0.189 * *(p + 2), 255.0);         // writing to red pixel in target image (pg)
    *(pg + 1) = (uint8_t)fmin(0.349 * *p + 0.686 * *(p + 1) + 0.168 * *(p + 2), 255.0);         // ""         green
    *(pg + 2) = (uint8_t)fmin(0.272 * *p + 0.534 * *(p + 1) + 0.131 * *(p + 2), 255.0);         // ""         blue  
    //Not very well written, note its R= (0.272 * *p), G = (0.534 * *(p + 1)), B = (0.131 * *(p + 2)). So getting each RGB component for each pixel, mult by coeff and write to output
    if(inputChannels == 4) {
      //Just copy alpha channel as is
      *(pg + 3) = *(p + 3);
    }
  }
#elif NO_FILTER
  printf("No filter\n");
  for(unsigned char *p = inputData, *pg = outputData; p != inputData + inputImageSize; p += inputChannels, pg += outputChannels)
  {
    *pg       = *p;
    *(pg + 1) = *(p+1);
    *(pg + 2) = *(p+2);
    if(inputChannels == 4) {
      //Just copy alpha channel as is
      *(pg + 3) = *(p + 3);
    }
    //printf("Orig values: R: %d G: %d B: %d \n", *pg, *(pg + 1), *(pg + 2));
  }

#elif RED_FILTER
  printf("Red filter\n");
  float redBalance = 0.0; //range 0.0 - 255.0
  for(unsigned char *p = inputData, *pg = outputData; p != inputData + inputImageSize; p += inputChannels, pg += outputChannels)
  {
    *pg       = (uint8_t) *p * (redBalance/255.0); //Need float else any value less than 255 just goes to 0
    *(pg + 1) = *(p+1);
    *(pg + 2) = *(p+2);
    if(inputChannels == 4) {
      //Just copy alpha channel as is
      *(pg + 3) = *(p + 3);
    }
  }

#endif

  //Write result to file
  printf("Write to file\n");
  success = stbi_write_jpg("outputFile.jpg", outputWidth, outputHeight, outputChannels, outputData, 80); //last arg is quality, 1-100
  if(success){
    printf("Wrote file OK! x:%d y:%d channels:%d\n", inputWidth, inputHeight, outputChannels);
  }

  else{
    printf("Error writing file\n");
  }

  stbi_image_free(inputData);
  delete(outputData);

  return (EXIT_SUCCESS);
}
