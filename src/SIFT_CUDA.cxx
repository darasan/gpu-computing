// -----------------------------------------------------------------------------
// * Name:       SIFT_CUDA.cxx
// * Purpose:    Implementation of SIFT algorithm for CUDA
// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#include "SIFT_CUDA.hxx"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" 
#endif

using namespace std;

void GaussianPyramid::WriteAllImagesToFile(void)
{
  int success = 0;
  std::cout <<"WriteAllImagesToFile: " << this->octaves.size() << " octaves" << std::endl;
  std::cout <<"scales array size (num images) : " <<  this->octaves[0].size() << std::endl;

  for(int i=0; i<this->octaves[0].size();i++)
  {
      //only writing first octave now, TODO add inner loop for all
      string filename {"image"+std::to_string(i) + ".jpg"};
      success = stbi_write_jpg(filename.c_str(), this->octaves[0][i].width(), this->octaves[0][i].height(), this->octaves[0][i].numChannels(), this->octaves[0][i].data(), 100);

      //std::cout <<"Wrote image " + std::to_string(i) + " to file" << std::endl;
      if(success){
          std::cout <<"Wrote file: " << filename << " w:" << this->octaves[0][i].width() << " h: " << this->octaves[0][i].height() << " chans: " << this->octaves[0][i].numChannels() << std::endl;
      }

      else{
          std::cout <<"Error writing file: " << filename << std::endl;
      }
  }
}

void SIFT_CUDA::BuildGaussianPyramid(Image baseImg)
{
  float base_blur = 1.6f;

  //Double base image size and blur
  baseImg.Resize((baseImg.width())*2, (baseImg.height())*2, InterpType::BILINEAR);
  this->CreateGaussianKernel(base_blur);
  this->ApplyGaussianBlur(baseImg); 

  //Add to image array
  std::vector<Image> scales;
  scales.push_back(baseImg);

  int numImages = this->gPyramid.numScalesPerOctave();
  for(int i= 1; i<numImages; i++){
    Image img = Image(baseImg.width(), baseImg.height(), baseImg.numChannels(), baseImg.data());
    this->CreateGaussianKernel(base_blur+(i*0.35f));
    this->ApplyGaussianBlur(img);
    scales.push_back(img);
  }
  this->gPyramid.octaves.push_back(scales);
  std::cout << "Pushed octave to pyramid, pyr size" << this->gPyramid.octaves.size() << std::endl;
}

Image::Image(std::string filename)
{
  //std::cout << "Load image from file" << std::endl;

  int inputWidth, inputHeight, numChannels;
  unsigned char *inputData = stbi_load(filename.c_str(), &inputWidth, &inputHeight, &numChannels, 0);

  if(inputData == NULL){
      std::cout << "Error loading image" << std::endl;
      exit(1);
  }

  else{
      std::cout << "Loaded image: " << filename << " w:" << inputWidth << " h:" << inputHeight << "chans:" << numChannels << std::endl;
  }

  this->_width = inputWidth;
  this->_height = inputHeight;
  this->_numChannels = numChannels;
  this->_size = (inputWidth*inputHeight*numChannels);
  this->_data =  new unsigned char[_size];
  std::memcpy(this->_data,inputData,_size);

  stbi_image_free(inputData); //have own copy, don't need original

};

unsigned char Image::getPixelValue(int x, int y, pxChannel channel)
{
  if (x < 0 || x >= this->_width || y < 0 ||  y >= this->_height || channel > this->_numChannels){
    //printf("Error getPixelValue: out of bounds\n");
    return 0;
  }

  else{
    return *(this->_data + ((x + y * (this->_width)) * (this->_numChannels)) + (int) channel);
  }
}

void Image::setPixelValue(int x, int y, pxChannel channel, unsigned char value)
{
  if (x < 0 || x >= this->_width || y < 0 ||  y >= this->_height || channel > this->_numChannels){
    //printf("Error setPixelValue: out of bounds x:%d y:%d chans:%d colour:%d\n", x, y, this->_numChannels, channel);
  }

  else{
    *(this->_data + ((x + y*(this->_width)) * (this->_numChannels)) + (int) channel) = value;
  }
}

void SIFT_CUDA::CreateGaussianKernel(float sigma)
{
  float sum = 0.0f;
  float mean = (kernelSize-1)/2.0f;
  float x = 0.0f;

  std::cout << "Create kernel with sigma = " << sigma << std::endl;

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
    //std::cout << "Normalised kernel: i = " << this->kernel[i] << std::endl;
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


  Image tempImg = Image(width, height, chans, img.data());
  if(tempImg.data() == NULL ) {
        std::cout << "Unable to create temp image\n" << std::endl;
    return;
  }

  while(rowIdx<height){ //Scan rows top to bottom

    //Get middle pixel value
    output_red_h    = img.getPixelValue(colIdx, rowIdx, RED) * kernel[0];
    output_green_h  = img.getPixelValue(colIdx, rowIdx, GREEN) * kernel[0];
    output_blue_h   = img.getPixelValue(colIdx, rowIdx, BLUE) * kernel[0];

    //Blur horizontal pixels
    for(int k = 1; k<kernelSize; k++)
    {
      output_red_h    += img.getPixelValue(colIdx+k, rowIdx, RED) * kernel[k];
      output_green_h  += img.getPixelValue(colIdx+k, rowIdx, GREEN) * kernel[k];
      output_blue_h   += img.getPixelValue(colIdx+k, rowIdx, BLUE) * kernel[k];
    }

    //Write result to temp image
    tempImg.setPixelValue(colIdx, rowIdx, RED, (output_red_h));
    tempImg.setPixelValue(colIdx, rowIdx, GREEN, (output_green_h));
    tempImg.setPixelValue(colIdx, rowIdx, BLUE, (output_blue_h));

    colIdx++;
    if(colIdx>=width)
    {
      rowIdx++;
      colIdx = 0; //new row, reset cols
    }
  }

    //Repeat for vertical component. Start from blurred horiz data in tempImage
    memset(data,0,imageSize);
    rowIdx = 0;
    colIdx = 0;

    while(rowIdx<height)
    {
      output_red_v    = tempImg.getPixelValue(colIdx, rowIdx, RED) * kernel[0];
      output_green_v  = tempImg.getPixelValue(colIdx, rowIdx, GREEN) * kernel[0];
      output_blue_v   = tempImg.getPixelValue(colIdx, rowIdx, BLUE) * kernel[0];

      for(int k = 1; k<kernelSize;k++)
      {
        output_red_v    += tempImg.getPixelValue(colIdx, rowIdx+k, RED) * kernel[k];
        output_green_v  += tempImg.getPixelValue(colIdx, rowIdx+k, GREEN) * kernel[k];
        output_blue_v   += tempImg.getPixelValue(colIdx, rowIdx+k, BLUE) * kernel[k];
      }

      //Write final result to original image
      img.setPixelValue(colIdx, rowIdx, RED, (output_red_v));
      img.setPixelValue(colIdx, rowIdx, GREEN, (output_green_v));
      img.setPixelValue(colIdx, rowIdx, BLUE, (output_blue_v));

      colIdx++;
      if(colIdx>=width)
      {
        rowIdx++;
        colIdx = 0;
      }
    }

    tempImg.FreeImageData();
    std::cout << "Finish gaussian\n" << std::endl;
}



