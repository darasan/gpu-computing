// -----------------------------------------------------------------------------
// * Name:       DoG.cu
// * Purpose:    Testing Difference of Gaussians (DoG) algorithm

// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#include <cmath>
#include <iostream>
#include <string>
#include <stdlib.h> 

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" 

const char* filename = "../LivingRoom.jpg";

enum pxColour{
  RED = 0,
  GREEN,
  BLUE
};

enum pxColour colour;

float kernel[5] = {0.0f};

unsigned char getPixelColour(int x, int y, int width, int height, int numChannels, pxColour colour, unsigned char *data)
{
  if (x < 0 || x >= width || y < 0 ||  y >= height || numChannels<0 || colour < 0 || colour > 3){
    //printf("Error getPixelColour: out of bounds\n");
    return 0;
  }

  else{
    return *(data + ((x + y*width) * numChannels) + (int) colour);
  }
}

void setPixelColour(int x, int y, int width, int height, int numChannels, pxColour colour, unsigned char *data, unsigned char value)
{
  if (x < 0 || x >= width || y < 0 ||  y >= height || numChannels<0 || colour < 0 || colour > 3 ){
    //printf("Error setPixelColour: out of bounds x:%d y:%d chans:%d colour%d\n", x,y,numChannels,colour);
  }

  else{
    *(data + ((x + y*width) * numChannels) + (int) colour) = value;
  }
}

void CreateGaussianKernel(float sigma)
{
  //float kernel[5];
  float sum = 0.0f;
  float mean = 8/2.0f; //8 = 1-radius. Dont include centre pixel
  float x = 0.0f;

  for(int i = 0; i<9;i++)
  {
    x = i - mean;
    kernel[i] = std::exp(-(x*x) / (2*sigma*sigma)); //No need for 1 over 2*pi*sigma squared term, as will normalise after
    sum += kernel[i];
  }

  //Normalise
  for(int i= 0; i<9; i++)
  {
    kernel[i] /= sum;
    printf("Normalised kernel: i = %f\n", kernel[i]);
  }
}

void ApplyGaussianBlur(unsigned char *inputData, int kernelRadius, int inputHeight, int inputWidth, int inputChannels)
{
  unsigned char output_red_h = 0, output_green_h = 0, output_blue_h = 0;
  unsigned char output_red_v = 0, output_green_v = 0, output_blue_v = 0;

  int imageSize = inputHeight*inputWidth*inputChannels;
  int rowIdx = 0, colIdx = 0;

  unsigned char *tempData = new unsigned char[imageSize]; 
  if(tempData == NULL ) {
    printf("ApplyGaussianBlur: unable to allocate memory for output image\n");
    return;
  }

  while(rowIdx<inputHeight){ //Scan rows top to bottom

    //Get middle (current) pixel colour
    output_red_h = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, inputData) * kernel[0];
    output_green_h = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, GREEN, inputData) * kernel[0];
    output_blue_h = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, BLUE, inputData) * kernel[0];

    //Blur horizontal pixels
    for(int k = 1; k<kernelRadius; k++)
    {
      output_red_h += getPixelColour(colIdx+k, rowIdx, inputWidth, inputHeight, inputChannels, RED, inputData) * kernel[k];
      output_green_h += getPixelColour(colIdx+k, rowIdx, inputWidth, inputHeight, inputChannels, GREEN, inputData) * kernel[k];
      output_blue_h += getPixelColour(colIdx+k, rowIdx, inputWidth, inputHeight, inputChannels, BLUE, inputData) * kernel[k];
    }

    //Write result
    setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, tempData, (output_red_h));
    setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, GREEN, tempData, (output_green_h));
    setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, BLUE, tempData, (output_blue_h));

    colIdx++; //Move to next pixel in row
    if(colIdx>=inputWidth)
    {
      rowIdx++;
      colIdx = 0; //new row, reset cols
    }
  }

    //Clear old data, then repeat blur for vertical component. Use blurred horiz data as starting point
    memset(inputData,0,imageSize);
    rowIdx = 0;
    colIdx = 0;

    while(rowIdx<inputHeight)
    {
      output_red_v = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, tempData) * kernel[0];
      output_green_v = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, GREEN, tempData) * kernel[0];
      output_blue_v = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, BLUE, tempData) * kernel[0];

      for(int k = 1; k<kernelRadius;k++)
      {
        output_red_v += getPixelColour(colIdx, rowIdx+k, inputWidth, inputHeight, inputChannels, RED, tempData) * kernel[k];
        output_green_v += getPixelColour(colIdx, rowIdx+k, inputWidth, inputHeight, inputChannels, GREEN, tempData) * kernel[k];
        output_blue_v += getPixelColour(colIdx, rowIdx+k, inputWidth, inputHeight, inputChannels, BLUE, tempData) * kernel[k];
      }

      setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, inputData, (output_red_v));
      setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, GREEN, inputData, (output_green_v));
      setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, BLUE, inputData, (output_blue_v));

      colIdx++;
      if(colIdx>=inputWidth)
      {
        rowIdx++;
        colIdx = 0;
      }
    }
    delete(tempData);
}

void CalculateDoG(unsigned char *inputData, int inputWidth, int inputHeight, int inputChannels, unsigned char *outputDoG)
{
  int rowIdx = 0, colIdx = 0;
  unsigned char image1_px = 0, image2_px = 0;
  int imageSize = inputHeight*inputWidth*inputChannels;

  unsigned char *imageGaussian1 = new unsigned char[imageSize]; 
  unsigned char *imageGaussian2 = new unsigned char[imageSize]; 
  if(imageGaussian1 == NULL || imageGaussian2 == NULL) {
    printf("CalculateDoG: unable to allocate memory for output image\n");
    return;
  }

  memcpy(imageGaussian1, inputData, imageSize);
  CreateGaussianKernel(0.2f);
  ApplyGaussianBlur(imageGaussian1, 9, inputHeight, inputWidth, inputChannels);

  CreateGaussianKernel(0.1f);
  memcpy(imageGaussian2, inputData, imageSize);
  ApplyGaussianBlur(imageGaussian2, 9, inputHeight, inputWidth, inputChannels);

  while(rowIdx<inputHeight) //Scan rows top to bottom
  {
      image1_px = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, imageGaussian1);
      image2_px = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, imageGaussian2);
      //Subtract and copy to new image
      setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, outputDoG, (image1_px - image2_px));
      //printf("image1_px %d image2_px %d \n", image1_px, image2_px);

      image1_px = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, GREEN, imageGaussian1);
      image2_px = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, GREEN, imageGaussian2);
      setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, GREEN, outputDoG, (image1_px - image2_px));

      image1_px = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, BLUE, imageGaussian1);
      image2_px = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, BLUE, imageGaussian2);
      setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, BLUE, outputDoG, (image1_px - image2_px));

      colIdx++;
      if(colIdx>=inputWidth)
      {
          rowIdx++;
          colIdx = 0;
      }
  }
  delete(imageGaussian1);
  delete(imageGaussian2);
}

int main(int argc, char **argv) {

    std::cout << "\nDoG Test\n" << std::endl;

    //Load 2 blurred images
    int inputWidth, inputHeight, inputChannels;
    int inputWidth2, inputHeight2, inputChannels2;
    int success = 0;
  
    unsigned char *inputData = stbi_load(filename, &inputWidth, &inputHeight, &inputChannels, 0);
    int inputImageSize = (inputWidth * inputHeight * inputChannels);

    if(inputData == NULL){
        printf("Error loading images\n");
        return 0;
    }

    //Calculate num channels and size for output file
    int outputChannels = inputChannels = 3;
    int outputWidth = inputWidth;
    int outputHeight = inputHeight;
    int outputImageSize = (outputWidth * outputHeight * outputChannels);

    printf("Allocate memory: width %d height %d chans %d size %d\n", outputWidth, outputHeight, outputChannels, outputImageSize);
    unsigned char *outputData = new unsigned char[outputImageSize];
    if(outputData == NULL) {
        printf("Unable to allocate memory for output image\n");
        exit(1);
    }

    //test
    CreateGaussianKernel(3.0f);
    ApplyGaussianBlur(inputData, 9, inputHeight, inputWidth, inputChannels);
    
    //CalculateDoG(inputData,inputWidth, inputHeight, inputChannels, outputData);

    //Write result to file
    printf("Write to file\n");
    success = stbi_write_jpg("DoGoutput.jpg", outputWidth, outputHeight, outputChannels, inputData, 100);
    if(success){
        printf("Wrote file OK! x:%d y:%d channels:%d\n", outputWidth, outputHeight, outputChannels);
    }

    else{
        printf("Error writing file\n");
    }

    stbi_image_free(inputData);
    delete(outputData);

    return (EXIT_SUCCESS);
}
