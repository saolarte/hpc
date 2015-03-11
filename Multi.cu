#include <cuda.h>
#include <stdio.h>
#include<time.h>

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




int main(){
	clock_t sec_ini, sec_fin, par_ini, par_fin;
  double tiempo_sec, tiempo_par;
   
  int tam= 1024;
  int size= tam*tam*sizeof (int);
  
  
  
  //////////////////////////////VARIABLES EN HOST/////////////////////////////////
  int *h_A, *h_B, *h_C, *sec_C;
  ////////////////////////////////Reservar memoria///////////////////////////////
  h_A = (int * ) malloc (size);
  h_B = (int * ) malloc (size);
  h_C = (int * ) malloc (size);
  sec_C = (int * ) malloc (size);
  /////////////////////////Inicializar variables en host/////////////////////////
  inic_matriz(h_A, tam);
  inic_matriz(h_B, tam);
  inic_matriz(h_C, tam);
  ////////////////////////////VARIABLES EN HOST ¡////////////////////////////////
    
     
  
  
  
  //Variables en device
  int *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A,size);
  cudaMemcpy(d_A,h_A,size,  cudaMemcpyHostToDevice);
  
  cudaMalloc((void **)&d_B, size);
  cudaMemcpy(d_B,h_B,size,  cudaMemcpyHostToDevice);
  
  
  cudaMalloc((void **)&d_C, size);
  cudaMemcpy(d_C,h_C,size,  cudaMemcpyHostToDevice);
//////////////////////////////////////////////////////////////////////////////////
  
  
  ///////////////////////////////REALIZAR MULTIPLICACION SECUENCIAL////////////////////////////
  
  sec_ini=clock();
  
  mult_matrices(h_A,h_B,sec_C, tam);
  
  sec_fin= clock();
  tiempo_sec= ((double) (sec_fin - sec_ini)) / CLOCKS_PER_SEC;
  printf("EL ALGORITMO SECUENCIAL TARDO: %.10f\n", tiempo_sec);
//////////////////////////////////////////////////////////////////////////////////////////// 
  
  
  
  par_ini=clock();
  par_fin=clock();
  tiempo_par= ((double) (par_fin - par_ini)) / CLOCKS_PER_SEC;
  printf("EL ALGORITMO PARALELO TARDO: %.10f\n", tiempo_par);
  
  
  
  
  
  
  cudaFree(d_A); cudaFree(d_B);  cudaFree(d_C);
  
}


