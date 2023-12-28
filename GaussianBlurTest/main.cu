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
#define RED_FILTER 0
#define BORDER 0
#define GAUSSIAN 1

const char* filename = "../LivingRoom.jpg";

float offset[5] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f };
float weight[5] = {0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162 };

enum pxColour{ 
  RED = 0, 
  GREEN, 
  BLUE 
};

enum pxColour colour; 

unsigned char getPixelColour(int x, int y, int width, int height, int numChannels, pxColour colour, unsigned char *data)
{
  if (x < 0 || x >= width || y < 0 ||  y >= height || numChannels!=3 || colour < 0 || colour > 3){
    printf("Error getPixelColour: out of bounds\n");
    return 0;
  }

  else{
    return *(data + ((x + y*width) * numChannels) + (int) colour);
  }
}

void setPixelColour(int x, int y, int width, int height, int numChannels, pxColour colour, unsigned char *data, unsigned char value)
{
  if (x < 0 || x >= width || y < 0 ||  y >= height || numChannels!=3 || colour < 0 || colour > 3 ){
    printf("Error setPixelColour: out of bounds\n");
  }

  else{
    *(data + ((x + y*width) * numChannels) + (int) colour) = value;
  }
}

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

  if(inputChannels!=3){ //Accept 3 channels only, to simplify processing
    printf("Error: expect image with 3 input channels\n");
    return 0;
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
  }
#elif SEPIA
  printf("Sepia filter\n");
  //Sepia filter coefficients from https://stackoverflow.com/questions/1061093/how-is-a-sepia-tone-created
  for(unsigned char *p = inputData, *pg = outputData; p != inputData + inputImageSize; p += inputChannels, pg += outputChannels)
  {
    *pg       = (uint8_t)fmin(0.393 * *p + 0.769 * *(p + 1) + 0.189 * *(p + 2), 255.0);         // writing to red pixel in target image (pg)
    *(pg + 1) = (uint8_t)fmin(0.349 * *p + 0.686 * *(p + 1) + 0.168 * *(p + 2), 255.0);         // ""         green
    *(pg + 2) = (uint8_t)fmin(0.272 * *p + 0.534 * *(p + 1) + 0.131 * *(p + 2), 255.0);         // ""         blue  
    //Note its R= (0.272 * *p), G = (0.534 * *(p + 1)), B = (0.131 * *(p + 2)). So getting each RGB component for each pixel, mult by coeff and write to output
  }
#elif NO_FILTER
  printf("No filter\n");

  unsigned char *in = inputData;
  unsigned char *out = outputData;

  for(int i = 0; i<inputImageSize; i+=inputChannels) //inputImageSize is in pixels (3 elements, RGB). But data ptr is per element (should be 3x more)
  {
    *out       = *in; 
    *(out + 1) = *(in + 1);
    *(out + 2) = *(in + 2);

    //printf("Read values: R: %d G: %d B: %d \n", *in, *(in + 1), *(in + 2));
    //printf("Wrote values: R: %d G: %d B: %d \n", *out, *(out + 1), *(out + 2));

    in += inputChannels; //move to next pixel
    out += outputChannels;
  }

#elif RED_FILTER
  printf("Red filter\n");
  float redBalance = 0.0; //range 0.0 - 255.0
  for(unsigned char *p = inputData, *pg = outputData; p != inputData + inputImageSize; p += inputChannels, pg += outputChannels)
  {
    *pg       = (uint8_t) *p * (redBalance/255.0); //Need float else any value less than 255 just goes to 0
    *(pg + 1) = *(p+1);
    *(pg + 2) = *(p+2);
  }

#elif BORDER
  printf("Draw border\n");

  unsigned char *in = inputData;
  unsigned char *out = outputData;
  int border_width = 5;
  int rowIdx = 0;
  int colIdx = 0;

  for(int i = 0; i<inputImageSize; i+=inputChannels)
  {
    /* //test - draw top half of image only
    if(i<=inputWidth*(inputHeight/2)*inputChannels){    
      *out       = *in; 
      *(out + 1) = *(in + 1);
      *(out + 2) = *(in + 2);
    } */

    //Draw top and bottom borders
    if((rowIdx<border_width) || (rowIdx>(inputHeight-border_width)))
    {
      *out       = 255;
      *(out + 1) = 0;
      *(out + 2) = 0;
    }

    //Draw side borders
    else if((colIdx < border_width) || (colIdx >= (inputWidth-border_width)))
    {
      *out       = 255;
      *(out + 1) = 0;
      *(out + 2) = 0;
    }

    //Write image
    else
    {
      *out       = *in;
      *(out + 1) = *(in + 1);
      *(out + 2) = *(in + 2);
    }

    colIdx++;  //already counting cols each loop iteration (1 col = 1 px = 3 chans, i+=inputChans)

    if(i%(inputWidth*inputChannels) == 0) //Count rows
    {
      //printf("rowIdx: %d\n", rowIdx);
      rowIdx++;
      colIdx = 0; //new row, reset cols
    }

    in += inputChannels; //move to next pixel
    out += outputChannels;
  }

  //Target test
  for (int val=0;val<100;val++)
  {
    setPixelColour(270+val,240,inputWidth,inputHeight,inputChannels,RED,outputData,255); //horiz: 320-50 = 270
    setPixelColour(320,190+val,inputWidth,inputHeight,inputChannels,GREEN,outputData,255); //vert:  240-50 = 190
  }

#elif GAUSSIAN
  unsigned char output_red_h = 0, output_green_h = 0, output_blue_h = 0;
  unsigned char output_red_v = 0, output_green_v = 0, output_blue_v = 0;
  int kernelSize = 5;

  unsigned char *in = inputData;
  unsigned char *out = outputData;
  int border_width = 5;
  int rowIdx = 0;
  int colIdx = 0;

  while(rowIdx<inputHeight) //Scan rows top to bottom
  {
    if((rowIdx < border_width) || (rowIdx>(inputHeight-border_width))) //top and bottom borders
    {
      setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, out, 255);
    }

    else if((colIdx < border_width) || (colIdx >= (inputWidth-border_width))) //side borders
    {
      setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, out, 255);
    }

    else //blur
    { 
      //Get middle (current) pixel colour
      output_red_h = output_red_v = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, in) * weight[0];
      output_green_h = output_green_v = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, GREEN, in) * weight[0];
      output_blue_h = output_blue_v = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, BLUE, in) * weight[0];

      //Add horizontal px
      for(int k = 1; k<kernelSize; k++)
      {
        output_red_h += getPixelColour(colIdx+k, rowIdx, inputWidth, inputHeight, inputChannels, RED, in) * weight[k]; //px to the right 
        output_red_h += getPixelColour(colIdx-k, rowIdx, inputWidth, inputHeight, inputChannels, RED, in) * weight[k]; //px to the left

        output_green_h += getPixelColour(colIdx+k, rowIdx, inputWidth, inputHeight, inputChannels, GREEN, in) * weight[k];
        output_green_h += getPixelColour(colIdx-k, rowIdx, inputWidth, inputHeight, inputChannels, GREEN, in) * weight[k];

        output_blue_h += getPixelColour(colIdx+k, rowIdx, inputWidth, inputHeight, inputChannels, BLUE, in) * weight[k];
        output_blue_h += getPixelColour(colIdx-k, rowIdx, inputWidth, inputHeight, inputChannels, BLUE, in) * weight[k];
      }

      //Add vertical px
      for(int k = 1; k<kernelSize;k++)
      {
        output_red_v += getPixelColour(colIdx, rowIdx+k, inputWidth, inputHeight, inputChannels, RED, in) * weight[k]; //px above
        output_red_v += getPixelColour(colIdx, rowIdx-k, inputWidth, inputHeight, inputChannels, RED, in) * weight[k]; //px below

        output_green_v += getPixelColour(colIdx, rowIdx+k, inputWidth, inputHeight, inputChannels, GREEN, in) * weight[k];
        output_green_v += getPixelColour(colIdx, rowIdx-k, inputWidth, inputHeight, inputChannels, GREEN, in) * weight[k];

        output_blue_v += getPixelColour(colIdx, rowIdx+k, inputWidth, inputHeight, inputChannels, BLUE, in) * weight[k];
        output_blue_v += getPixelColour(colIdx, rowIdx-k, inputWidth, inputHeight, inputChannels, BLUE, in) * weight[k];
      }

      //Write new pixel
      setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, out, (output_red_h + output_red_v)/2);
      setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, GREEN, out, (output_green_h + output_green_v)/2);
      setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, BLUE, out, (output_blue_h + output_blue_v)/2);
    }

    colIdx++; //Move to next pixel in row
    if(colIdx>=inputWidth) 
    {
      rowIdx++;
      colIdx = 0; //new row, reset cols
    }
  }
#endif

  //Write result to file
  printf("Write to file\n");
  success = stbi_write_jpg("outputFile.jpg", outputWidth, outputHeight, outputChannels, outputData, 100); //last arg is quality, 1-100
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
