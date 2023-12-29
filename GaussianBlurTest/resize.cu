// -----------------------------------------------------------------------------
// * Name:       resize.cu
// * Purpose:    Testing image resizing

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
  if (x < 0 || x >= width || y < 0 ||  y >= height || numChannels<0 || colour < 0 || colour > 3){
    printf("Error getPixelColour: out of bounds\n");
    return 0;
  }

  else{
    return *(data + ((x + y*width) * numChannels) + (int) colour);
  }
}

void setPixelColour(int x, int y, int width, int height, int numChannels, pxColour colour, unsigned char *data, unsigned char value)
{
  if (x < 0 || x >= width || y < 0 ||  y >= height || numChannels<0 || colour < 0 || colour > 3 ){
    printf("Error setPixelColour: out of bounds x:%d y:%d chans:%d colour%d\n", x,y,numChannels,colour);
  }

  else{
    *(data + ((x + y*width) * numChannels) + (int) colour) = value;
  }
}

int main(int argc, char **argv) {

  std::cout << "\nResize Test\n" << std::endl;

  //Query image for info
  int inputWidth, inputHeight, inputChannels;
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
  unsigned char *inputData = stbi_load(filename, &inputWidth, &inputHeight, &inputChannels, 1); //last arg is num chans to load, set 0 to load all
  int inputImageSize = (inputWidth * inputHeight * inputChannels);

  //Calculate num channels and size for output file
  int outputChannels = inputChannels = 1; //need to assume we loaded 1 chan cos requested, lib wont set inputChannels to 1 itself
  int outputWidth = inputWidth*2;
  int outputHeight = inputHeight*2; //Assume we always write image of same size as input
  int outputImageSize = (outputWidth * outputHeight * outputChannels);

  printf("Allocate memory\n");
  unsigned char *outputData = new unsigned char[outputImageSize]; 
  if(outputData == NULL) {
    printf("Unable to allocate memory for output image\n");
    exit(1);
  }

  int border_width = 5;
  int rowIdx = 0;
  int colIdx = 0;
  int out_px = 0;
  int writeCol = 0;
  int writeRow = 0;

  while(rowIdx<inputHeight) //Scan rows top to bottom
  {
    /*
    if((rowIdx < border_width) || (rowIdx>(inputHeight-border_width))) //top and bottom borders
    {
      setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, out, 255);
    }

    else if((colIdx < border_width) || (colIdx >= (inputWidth-border_width))) //side borders
    {
      setPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, out, 255);
    } */

    out_px = getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, inputData);

    //copy to new image
    setPixelColour(writeCol, writeRow, outputWidth, outputHeight, outputChannels, RED, outputData, out_px);

    //copy to px on right and below
    setPixelColour(writeCol+1, writeRow, outputWidth, outputHeight, outputChannels, RED, outputData, out_px);
    setPixelColour(writeCol, writeRow+1, outputWidth, outputHeight, outputChannels, RED, outputData, out_px);

    //bottom diagonal right
    setPixelColour(writeCol+1, writeRow+1, outputWidth, outputHeight, outputChannels, RED, outputData, out_px);

    writeCol+=2;

    colIdx++; //Move to next pixel in row
    if(colIdx>=inputWidth) 
    {
        rowIdx++;
        colIdx = 0; //new row, reset cols
    }

    if(writeCol>=outputWidth)
    {
        writeRow+=2;
        writeCol = 0; //new row, reset cols
    }
  }

  //Write result to file
  printf("Write to file\n");
  success = stbi_write_jpg("resizedFile.jpg", outputWidth, outputHeight, outputChannels, outputData, 100); //last arg is quality, 1-100
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
