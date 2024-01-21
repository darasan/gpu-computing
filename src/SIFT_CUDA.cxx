// -----------------------------------------------------------------------------
// * Name:       SIFT_CUDA.cxx
// * Purpose:    Implementation of SIFT algorithm for CUDA
// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#include <iostream>
#include "SIFT_CUDA.h"
#include <cmath>
#include <cstring>

using namespace std;

unsigned char SIFT_CUDA::getPixelColour(int x, int y, int width, int height, int numChannels, pxColour colour, unsigned char *data)
{
  if (x < 0 || x >= width || y < 0 ||  y >= height || numChannels<0 ){
    //printf("Error getPixelColour: out of bounds\n");
    return 0;
  }

  else{
    return *(data + ((x + y*width) * numChannels) + (int) colour);
  }
}

void SIFT_CUDA::setPixelColour(int x, int y, int width, int height, int numChannels, pxColour colour, unsigned char *data, unsigned char value)
{
  if (x < 0 || x >= width || y < 0 ||  y >= height || numChannels<0 ){
    //printf("Error setPixelColour: out of bounds x:%d y:%d chans:%d colour%d\n", x,y,numChannels,colour);
  }

  else{
    *(data + ((x + y*width) * numChannels) + (int) colour) = value;
  }
}

void SIFT_CUDA::CreateGaussianKernel(float sigma)
{
  float sum = 0.0f;
  float mean = (kernelSize-1)/2.0f;
  float x = 0.0f;

  for(int i = 0; i<kernelSize;i++)
  {
    x = i - mean;
    this->kernel[i] = exp(-(x*x) / (2*sigma*sigma)); //No need for 1 over 2*pi*sigma squared term, as will normalise after
    sum += this->kernel[i];
  }

  //Normalise
  for(int i= 0; i<kernelSize; i++)
  {
    this->kernel[i] /= sum;
    std::cout << "Normalised kernel: i = " << this->kernel[i] << std::endl;
  }
}

void SIFT_CUDA::ApplyGaussianBlur(Image img)
{
  unsigned char output_red_h = 0, output_green_h = 0, output_blue_h = 0;
  unsigned char output_red_v = 0, output_green_v = 0, output_blue_v = 0;

  int imageSize = img.size();
  int width = img.width();
  int height = img.height();
  int chans = img.numChannels();
  unsigned char *data = img.data();
  int rowIdx = 0, colIdx = 0;

  unsigned char *tempData = new unsigned char[imageSize]; 
  if(tempData == NULL ) {
        std::cout << "ApplyGaussianBlur: unable to allocate memory for output image\n" << std::endl;
    return;
  }

  while(rowIdx<height){ //Scan rows top to bottom

    //Get middle pixel value
    output_red_h    = getPixelColour(colIdx, rowIdx, width, height, chans, RED, data) * kernel[0];
    output_green_h  = getPixelColour(colIdx, rowIdx, width, height, chans, GREEN, data) * kernel[0];
    output_blue_h   = getPixelColour(colIdx, rowIdx, width, height, chans, BLUE, data) * kernel[0];

    //Blur horizontal pixels
    for(int k = 1; k<kernelSize; k++)
    {
      output_red_h    += getPixelColour(colIdx+k, rowIdx, width, height, chans, RED, data) * kernel[k];
      output_green_h  += getPixelColour(colIdx+k, rowIdx, width, height, chans, GREEN, data) * kernel[k];
      output_blue_h   += getPixelColour(colIdx+k, rowIdx, width, height, chans, BLUE, data) * kernel[k];
    }

    //Write result
    setPixelColour(colIdx, rowIdx, width, height, chans, RED, tempData, (output_red_h));
    setPixelColour(colIdx, rowIdx, width, height, chans, GREEN, tempData, (output_green_h));
    setPixelColour(colIdx, rowIdx, width, height, chans, BLUE, tempData, (output_blue_h));


    colIdx++;
    if(colIdx>=width)
    {
      rowIdx++;
      colIdx = 0; //new row, reset cols
    }
  }

    //Repeat for vertical component. Start from blurred horiz data
    memset(data,0,imageSize);
    rowIdx = 0;
    colIdx = 0;

    while(rowIdx<height)
    {
      output_red_v    = getPixelColour(colIdx, rowIdx, width, height, chans, RED, tempData) * kernel[0];
      output_green_v  = getPixelColour(colIdx, rowIdx, width, height, chans, GREEN, tempData) * kernel[0];
      output_blue_v   = getPixelColour(colIdx, rowIdx, width, height, chans, BLUE, tempData) * kernel[0];

      for(int k = 1; k<kernelSize;k++)
      {
        output_red_v    += getPixelColour(colIdx, rowIdx+k, width, height, chans, RED, tempData) * kernel[k];
        output_green_v  += getPixelColour(colIdx, rowIdx+k, width, height, chans, GREEN, tempData) * kernel[k];
        output_blue_v   += getPixelColour(colIdx, rowIdx+k, width, height, chans, BLUE, tempData) * kernel[k];
      }

      //Write final result to original image
      setPixelColour(colIdx, rowIdx, width, height, chans, RED, data, (output_red_v));
      setPixelColour(colIdx, rowIdx, width, height, chans, GREEN, data, (output_green_v));
      setPixelColour(colIdx, rowIdx, width, height, chans, BLUE, data, (output_blue_v));

      colIdx++;
      if(colIdx>=width)
      {
        rowIdx++;
        colIdx = 0;
      }
    }
    delete(tempData);
}



