#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

#define MASK_WIDTH 3


__constant__ char M[MASK_WIDTH*MASK_WIDTH];

using namespace cv;


int main(int argc, char **argv){
		cudaError_t error = cudaSuccess;
    clock_t start, end, startGPU, endGPU;
    double cpu_time_used, gpu_time_used;
    char h_M[] = {-1,0,1,-2,0,2,-1,0,1};
    
  	const char imgName[20]= "./inputs/img1.jpg";
  	
    unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput, *d_sobelOutput;
    Mat image;
    image = imread(&imgName[0], 1);
	  if(!image.data){
        printf("No image Data \n");
        return -1;
    }
  
   	Size s = image.size();

    int width = s.width;
    int height = s.height;
    int size = sizeof(unsigned char)*width*height*image.channels();
    int sizeGray = sizeof(unsigned char)*width*height;
  
  	Mat gray_image;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imageOutput;

    start = clock();
    Mat gray_image_opencv, grad_x, abs_grad_x;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
    Sobel(gray_image_opencv,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    end = clock();


    
  //IMAGEN FUNCION SECUENCIAL
  	imwrite("./Sobel_Image.jpg",gray_image_opencv); 
}
