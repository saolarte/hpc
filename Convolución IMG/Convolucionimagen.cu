#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

#define TILE_SIZE 32
#define MASK_WIDTH 3


__constant__ char M[MASK_WIDTH*MASK_WIDTH];

using namespace cv;



__device__ unsigned char clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}


__global__ void sobelFilter_const(unsigned char *imageInput, int width, int height, \
        unsigned int maskWidth,unsigned char *imageOutput){

    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

    int Pvalue = 0;

    int N_start_point_row = row - (maskWidth/2);
    int N_start_point_col = col - (maskWidth/2);

    for(int i = 0; i < maskWidth; i++){
        for(int j = 0; j < maskWidth; j++ ){
            if((N_start_point_col + j >=0 && N_start_point_col + j < width) \
                    &&(N_start_point_row + i >=0 && N_start_point_row + i < height)){
                Pvalue += imageInput[(N_start_point_row + i)*width+(N_start_point_col + j)] * M[i*maskWidth+j];
            }
        }
    }
    imageOutput[row*width+col] = clamp(Pvalue);
}




__global__ void sobelFilter_glob(unsigned char *imageInput, int width, int height, unsigned int maskWidth,\
        char *M,unsigned char *imageOutput){
    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

    int Pvalue = 0;

    int N_start_point_row = row - (maskWidth/2);
    int N_start_point_col = col - (maskWidth/2);

    for(int i = 0; i < maskWidth; i++){
        for(int j = 0; j < maskWidth; j++ ){
            if((N_start_point_col + j >=0 && N_start_point_col + j < width) \
                    &&(N_start_point_row + i >=0 && N_start_point_row + i < height)){
                Pvalue += imageInput[(N_start_point_row + i)*width+(N_start_point_col + j)] * M[i*maskWidth+j];
            }
        }
    }
    imageOutput[row*width+col] = clamp(Pvalue);
}




__global__ void sobelFilter_shared(unsigned char *imageInput, int width, int height, \
        unsigned int maskWidth,unsigned char *imageOutput){
    __shared__ float N_ds[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE+ MASK_WIDTH - 1];
    int n = maskWidth/2;
    int dest = threadIdx.y*TILE_SIZE+threadIdx.x, destY = dest / (TILE_SIZE+MASK_WIDTH-1), destX = dest % (TILE_SIZE+MASK_WIDTH-1),
        srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
        src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = imageInput[src];
    else
        N_ds[destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest /(TILE_SIZE + MASK_WIDTH - 1), destX = dest % (TILE_SIZE + MASK_WIDTH - 1);
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * width + srcX);
    if (destY < TILE_SIZE + MASK_WIDTH - 1) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imageInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    __syncthreads();

    int accum = 0;
    int y, x;
    for (y = 0; y < maskWidth; y++)
        for (x = 0; x < maskWidth; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * maskWidth + x];
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (y < height && x < width)
        imageOutput[(y * width + x)] = clamp(accum);
    __syncthreads();
}









__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
        imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 \
                                     + imageInput[(row*width+col)*3+BLUE]*0.114;
    }
}







int main(int argc, char **argv){
  	cudaSetDevice(0);//GTX980
		cudaError_t error = cudaSuccess;
    clock_t assi_start, assi_end, sec_start, sec_end, const_start, const_end, glob_start, glob_end, shar_start, shar_end;
    double sec_time_used, const_time_used, assi_time_used, glob_time_used, shar_time_used;
    char h_M[] = {-1,0,1,-2,0,2,-1,0,1}, *d_M;
    
  	const char imgName[20]= "./inputs/img2.jpg";
  	
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
  
  
  	
    
  	dataRawImage = (unsigned char*)malloc(size);
    error = cudaMalloc((void**)&d_dataRawImage,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_dataRawImage\n");
        exit(-1);
    }

    h_imageOutput = (unsigned char *)malloc(sizeGray);
    error = cudaMalloc((void**)&d_imageOutput,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_imageOutput\n");
        exit(-1);
    }

    error = cudaMalloc((void**)&d_sobelOutput,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_sobelOutput\n");
        exit(-1);
    }

    dataRawImage = image.data;

    assi_start = clock();

    error = cudaMemcpy(d_dataRawImage,dataRawImage,size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de dataRawImage a d_dataRawImage \n");
        exit(-1);
    }

    error = cudaMemcpyToSymbol(M,h_M,sizeof(char)*MASK_WIDTH*MASK_WIDTH);
    if(error != cudaSuccess){
        printf("Error copiando los datos de h_M a d_M \n");
        exit(-1);
    }
  
  
   	
  	error = cudaMalloc((void**)&d_M,sizeof(char)*9); error = cudaMalloc((void**)&d_M,sizeof(char)*9);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_M\n");
        exit(-1);
    }
  
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_M\n");
        exit(-1);
    
   }
  
  
  
  
  
  error = cudaMemcpy(d_M,h_M,sizeof(char)*9, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de h_M a d_M \n");
        exit(-1);
    }
  
    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
  
  	assi_end= clock();
  
  	  
    assi_time_used= ((double) (assi_end - assi_start)) / CLOCKS_PER_SEC;
		printf ("La reserva de memoria tardó %f\n", assi_time_used); 
  
  
  ////////////////////////////////MEMORIA CONSTANTE /////////////////////////////////////////
    
  	const_start= clock();
  	img2gray<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);
    cudaDeviceSynchronize();
    sobelFilter_const<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,3,d_sobelOutput);
    cudaMemcpy(h_imageOutput,d_sobelOutput,sizeGray,cudaMemcpyDeviceToHost);

    const_end = clock();	
    const_time_used= ((double) (const_end - const_start)) / CLOCKS_PER_SEC;
		printf ("El algoritmo usando memoria constante tardó %f\n", const_time_used);
//******************************************************************************************
  
  
  ////////////////////////////////MEMORIA GLOBAL/////////////////////////////////////////////////
	  glob_start=clock();
  	img2gray<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);
  	cudaDeviceSynchronize();
    sobelFilter_glob<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,3,d_M,d_sobelOutput);
  	cudaMemcpy(h_imageOutput,d_sobelOutput,sizeGray,cudaMemcpyDeviceToHost);
  
  	glob_end= clock();  
    glob_time_used= ((double) (glob_end - glob_start)) / CLOCKS_PER_SEC;
		printf ("El algoritmo usando memoria global tardó %f\n", glob_time_used); 
  
   
  
  //********************************************************************************************
  
  
  
  
  
  
  /////////////////////////////////MEMORIA COMPARTIDA///////////////////////////////////////////////
  
		shar_start=clock();
  	img2gray<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);
    cudaDeviceSynchronize();
    sobelFilter_shared<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,MASK_WIDTH,d_sobelOutput);
    cudaMemcpy(h_imageOutput,d_sobelOutput,sizeGray,cudaMemcpyDeviceToHost);  
  	shar_end=clock();
  
  
  	shar_end= clock();  
    shar_time_used= ((double) (shar_end - shar_start)) / CLOCKS_PER_SEC;
		printf ("El algoritmo usando memoria compartida tardó %f\n", shar_time_used); 
  
  
  //***********************************************************************************************
  
  	Mat gray_image;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imageOutput;

///////////////////////SECUENCIAL////////////////////////////////////
    sec_start = clock();
    Mat gray_image_opencv, grad_x, abs_grad_x;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
    Sobel(gray_image_opencv,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    sec_end = clock();
  
  	sec_time_used= ((double) (sec_end - sec_start)) / CLOCKS_PER_SEC;
		printf ("El algoritmo secuencial tardó %f", sec_time_used);
  	imwrite("./outputs/1088317736.png",grad_x); 
//**********************************************************************  
}
