#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 32


using namespace std;

__global__ void matrixMulKernel(float *d_A, float *d_B, float *d_C,int rA,int cA,int rB,int cB){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int Pvalue;
    if((row < rA)&&(col < cB)){
        Pvalue = 0;
        for (int k = 0; k < rB ; ++k){
            Pvalue += d_A[row*cA+k] * d_B[k*cB+col];
        }
        d_C[row*cB+col] = Pvalue;
    }
}



void multMxN(float *A,float *B, float *C, float rA,int cA,int rB,int cB){
	//El numero de col de A, debe ser igual al numero de row de B;
	//Matriz C= FilasAxColumnasB


	for(int i=0;i<rA;i++){
      	for(int j=0;j<cB;j++){
      		int value=0;
          	for(int k=0;k<rB;k++){
             	//h_c[n*i+j]+=h_a[n*i+k]*h_b[n*k+j];
             	value+=A[cA*i+k]*B[cB*k+j];
          	}
          	C[cB*i+j]=value;
      	}
    }
}

void llenar(float *MA,int n,int m){
	int tam=n*m;
	for(int  i = 0; i < tam; i++ )
        MA[i] = i;
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

int main(){
	int rA=128;
	int cA=1024;
	int cB=512;
	float blockSize = TILE_WIDTH;

	int rB=cA;

	size_t bytesA=(rA*cA)*sizeof(float);
	size_t bytesB=(rB*cB)*sizeof(float);
	size_t bytesC=(rA*cB)*sizeof(float);

	float *A=(float*)malloc(bytesA);
	float *B=(float*)malloc(bytesB);
	float *C1=(float*)malloc(bytesC);
	float *C2=(float*)malloc(bytesC);
	
	//Se inicializan las matrices
	llenar(A,rA,cA);
	llenar(B,rB,cB);

	
	//MULTIPLICACION SECUENCIAL DE MATRICES
	clock_t start = clock();      
    multMxN(A,B,C1,rA,cA,rB,cB);
    clock_t end= clock(); 
  	double elapsed_seconds=end-start;
    printf("Tiempo algoritmo secuencial Secuencial: %lf\n", (elapsed_seconds / CLOCKS_PER_SEC));
    
	//MULTIPLICACION EN PARALELO SIN TILES
    float *d_A;
	float *d_B;
    float *d_C;
	// Allocate memory for each vector on GPU
    cudaMalloc(&d_A,bytesA);
    cudaMalloc(&d_B,bytesB);
    cudaMalloc(&d_C,bytesC);
	
    

	// Copy host vectors to device
    cudaMemcpy( d_A, A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy( d_B, B, bytesB, cudaMemcpyHostToDevice);

  
	//bloques
	dim3 dimGrid(ceil(cB/blockSize),ceil(rA/blockSize),1);
	//hilos
	dim3 dimBlock(blockSize,blockSize,1);
  //Tiempo de ejecucion Paralelo
	clock_t start2 = clock();   
	 
    // Execute the kernel
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,rA,cA,rB,cB);
    //multTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,rA,cA,rB,cB);
    cudaDeviceSynchronize();
    // Copy array back to host
    cudaMemcpy( C2, d_C, bytesC, cudaMemcpyDeviceToHost );
    clock_t end2= clock(); 
  	double elapsed_seconds2=end2-start2;
    printf("Tiempo transcurrido Paralelo SinTile: %lf\n", (elapsed_seconds2 / CLOCKS_PER_SEC));  
      
   
	comparar(C1,C2,rA,cB);
	
	// Release device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
 
    // Release host memory
    free(A);
    free(B);
    free(C1);
    free(C2);
	
	return 0;
}