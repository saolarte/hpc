#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 32


void multMxN(int *A,int *B, int *C, int rA,int cA,int rB,int cB){
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

void llenar(int *MA,int n,int m){
	int tam=n*m;
	for(int  i = 0; i < tam; i++ )
        MA[i] = i;
}

void imprimir(int *MA,int n,int m){
	int tam=n*m,cont=0;
	for(int i = 0; i < tam; i++ ) {
        printf("%d ",MA[i]);
        if(cont==m-1){
            printf("\n");
            cont=-1;
        }
        cont++;  
    }
	printf("\n");
}

void comparar(int *C1,int *C2,int rC,int cC){

    int w=rC*cC;
    for(int i=0;i<w;i++){
      if(C1[i]!=C2[i]){
        printf ("No son iguales\n");
        break;
      }
    }
    printf("Son iguales\n");
}

int main(){
	int rA=128;
	int cA=1024;
	int cB=512;
	int rB=cA;

	size_t bytesA=(rA*cA)*sizeof(int);
	size_t bytesB=(rB*cB)*sizeof(int);
	size_t bytesC=(rA*cB)*sizeof(int);

	int *A=(int*)malloc(bytesA);
	int *B=(int*)malloc(bytesB);
	int *C1=(int*)malloc(bytesC);
	int *C2=(int*)malloc(bytesC);
	int *C3=(int*)malloc(bytesC);


	//1.Lleno las matrices
	llenar(A,rA,cA);
	llenar(B,rB,cB);

	//2.imprimo las matrices
	//cout<<"Matriz A:"<<endl;
	//imprimir(A,rA,cA);

	//cout<<"Matriz B:"<<endl;
	//imprimir(B,rB,cB);

	//3.Multiplico matrices NxM Secuencial
	clock_t start = clock();      
    multMxN(A,B,C1,rA,cA,rB,cB);
    clock_t end= clock(); 
  	double elapsed_seconds=end-start;
    printf("Tiempo transcurrido Secuencial: %lf\n", (elapsed_seconds / CLOCKS_PER_SEC));

 
    // Release host memory
    free(A);
    free(B);
    free(C1);
    free(C2);
	
	return 0;
}