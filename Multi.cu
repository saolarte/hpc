#include <cuda.h>
#include <stdio.h>
#include<time.h>

#define TILE_WIDTH 32



///////////////////////////////////////////FUNCION INICIALIZAR  LAS MATRICES//////////////////////////////////
//////////////////////////Todas la posiciones de la matriz son inicializadas en 1/////////////////////////////
void inic_matriz(int* matriz, int tam){
  for(int i=0; i<tam*tam; i++){
    matriz[i] = 1;
  }

}



/////////////////////////////////////MULTIPLICACION DE MATRICES SECUENCIAL///////////////////////////////////
int mult_matrices(int* A, int* B, int* C, int tam){
  int acum=0;
  for (int fila=0; fila<tam; ++fila){
    for (int col=0;col<tam; ++col){
      acum=0;
      for (int k=0; k<tam; ++k){
        acum += A[fila*col + k]*B[k*tam+col]; 
        
      }
     C[fila*fila+col]=acum;
      
      
    }
  
  
  }
  return 0;
    
  
}

////////////////////////////KERNEL MULTIPLICACION DE MATRICES SIN TILING//////////////////////////
__global__ void matrixMulKernel(int *d_A, int *d_B, int *d_C, int tam){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int Pvalue;
    if((row < tam)&&(col < tam)){
        Pvalue = 0;
        for (int k = 0; k < tam ; ++k){
            Pvalue += d_A[row*tam+k] * d_B[k*tam+col];
        }
        d_C[row*tam+col] = Pvalue;
    }
}


__global__ void matrixMulKernelTiled(int *d_M, int *d_N, int *d_P, int width){
    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for(int m = 0; m < width / TILE_WIDTH; ++m){
  Mds[ty][tx] = d_M[row*width + m*TILE_WIDTH + tx];
  Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty) * width + col];
  __syncthreads();

  for(int k = 0; k < TILE_WIDTH; ++k){
    Pvalue += Mds[ty][k] * Nds[k][tx];      
  }
  __syncthreads();
    }
    d_P[row*width+col] = Pvalue;
}



int main(){
  clock_t sec_ini, sec_fin, par_ini, par_fin, tile_ini, tile_fin;
  double tiempo_sec, tiempo_par, tiempo_tile;
   
  int tam= 1024;
  int size= tam*tam*sizeof (int);
  
  
  
  //////////////////////////////VARIABLES EN HOST/////////////////////////////////
  int *h_A, *h_B, *h_Cp, *h_Ct, *sec_C;
  ////////////////////////////////Reservar memoria///////////////////////////////
  h_A = (int * ) malloc (size);
  h_B = (int * ) malloc (size);
  h_Cp = (int * ) malloc (size);
  h_Ct = (int * ) malloc (size);
  sec_C = (int * ) malloc (size);
  /////////////////////////Inicializar variables en host/////////////////////////
  inic_matriz(h_A, tam);
  inic_matriz(h_B, tam);
  inic_matriz(h_Cp, tam);
  inic_matriz(h_Ct, tam);
  ////////////////////////////VARIABLES EN HOST////////////////////////////////
   
  
  ///////////////////////////////REALIZAR MULTIPLICACION SECUENCIAL////////////////////////////
  
  sec_ini=clock();
  
  mult_matrices(h_A,h_B,sec_C, tam);
  
  sec_fin= clock();
  tiempo_sec= ((double) (sec_fin - sec_ini)) / CLOCKS_PER_SEC;
  printf("EL ALGORITMO SECUENCIAL TARDO: %.10f\n", tiempo_sec);
//////////////////////////////////////////////////////////////////////////////////////////// 
  
   
  
  
/////////////////////////////EJECUCUCION ALGORITMO PARALELO/////////////////////////////////////////
 /////////////////////////////Variables en device////////////////////////////
  int *d_A, *d_B, *d_Cp, *d_Ct;
///////////////////////////////Reserva de memoria////////////////////////////////
  
  cudaMalloc((void**)&d_A,size);
  cudaMalloc((void**)&d_B,size);
  cudaMalloc((void**)&d_Cp,size);
  cudaMalloc((void**)&d_Ct,size);
  

  
  
  
  
  cudaMemcpy(d_A,h_A,size,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_B,h_B,size,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_Cp,h_Cp,size,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ct,h_Ct,size,  cudaMemcpyHostToDevice);
  
  
  int blockSize = 32;
  dim3 dimBlock(blockSize,blockSize,1);
  dim3 dimGrid(ceil(tam/float(blockSize)),ceil(tam/float(blockSize)),1);
  
  
  par_ini=clock();
  matrixMulKernel<<<dimGrid,dimBlock>>>(d_A,d_B,d_Cp,tam);
  cudaDeviceSynchronize();
  cudaMemcpy(h_Cp,d_Cp,size,cudaMemcpyDeviceToHost);
  
  
  par_fin=clock();
  tiempo_par= ((double) (par_fin - par_ini)) / CLOCKS_PER_SEC;
  printf("EL ALGORITMO PARALELO TARDO: %.10f\n", tiempo_par);
    
  
  cudaFree(d_A); cudaFree(d_B);  cudaFree(d_Cp);
  
  
  ///////////////////////////////////////////////////////////////////////////////
  
  
  tile_ini=clock();
  
      
  matrixMulKernelTiled<<<dimGrid,dimBlock>>>(d_A,d_B,d_Ct,tam);
  cudaDeviceSynchronize();
  cudaMemcpy(h_Ct,d_Ct,size,cudaMemcpyDeviceToHost);
   
  
  tile_fin=clock();
  
  tiempo_tile= ((double) (tile_fin - tile_ini)) / CLOCKS_PER_SEC;
  printf("EL ALGORITMO PARALELO USANDO TILING TARDO: %.10f\n", tiempo_tile);
    
  
  cudaFree(d_A); cudaFree(d_B);  cudaFree(d_Cp);
  
  
}


