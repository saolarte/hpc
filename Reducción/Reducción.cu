#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#define TILE_WIDTH 4


using namespace std;


__global__ void reduce(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
  

	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
  
  // write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}




void llenar(float *MA,int n,int m){
	int tam=n*m;
	for(int  i = 0; i < tam; i++ )
        MA[i] = i;
}


void comparar(float *C1,float *C2,int rC,int cC){

    int w=rC*cC;
    for(int i=0;i<w;i++){
      if(C1[i]!=C2[i]){
        cout<<"No son iguales"<<endl;
        break;
      }
    }
    cout<<"Son iguales"<<endl;
}




void imprimir(float *MA,int n,int m){
    int tam=n*m,cont=0;
    for(int i = 0; i < tam; i++ ) {
        printf("%f ",MA[i]);
       if(cont==m-1){
            printf("\n");
            cont=-1;
        }
        cont++;
    }
    printf("\n");
}

int main(){
  
    size_t tam=(SIZE)*sizeof(int);
    int *vect_in=(int*)malloc(bytesA);
    int *vect_out=(int*)malloc(bytesA);
    int *d_vector_in, d_vector_out;
	
    //bloques
    dim3 dimGrid(ceil(SIZE/BLOCKSIZE),ceil(SIZE/BLOCKSIZE),1);
	//hilos
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE,1);
    cudaDeviceSynchronize();
    
  
    llenar(vec, tam);
  
    //ALGORITMO PARALELO
    clock_t start = clock(); 
  
    cudaMalloc(&d_vect_in,SIZE);
    cudaMalloc(&d_vect_oput,SIZE);
  
    reduce<<<dimGrid, dimBlock>>>(vect_in vect_out);
  
    clock_t end= clock();
     
  
    //LIBERAR MEMORIA
    free(vec);
    cudaFree(d_vect);
  

}
