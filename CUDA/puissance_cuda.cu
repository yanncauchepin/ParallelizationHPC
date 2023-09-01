//puissance_cuda.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "defs.h"

#define TAILLE_BLOC_X 16

//nb de bloc : gridDim.x
//indice de parcours des blocs : blockIdx.x

//nb de thread par bloc : blockDim.x
//indice de parcours des threads : threadIdx.x

__global__ void matmulKernel(REAL_T* d_A, REAL_T* d_B, REAL_T* d_C, int n) {
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i < n){
    REAL_T temp=0;
    for(int k=0; k<n; k++)
      temp = temp + d_A[i * n + k] * d_B[i];
    d_C[i] = temp;
  }
}

__global__ void norm_to_Kernel(REAL_T* d_C, int n,REAL_T *norm) {
  *norm=0;
  for(int k=0; k<n; k++){
      *norm += d_C[k]*d_C[k];
  }
  *norm = sqrt(*norm);

}


__global__ void normKernel(REAL_T* d_Y,REAL_T *norm, int n) {
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i < n){
    d_Y[i] = d_Y[i]/(*norm);
  }
}

__global__ void errorKernel(REAL_T* d_X, REAL_T* d_Y,int n,REAL_T *erreur) {
  *erreur=0;
  for(int k=0; k<n; k++){
      REAL_T inter = d_X[k]-d_Y[k];
      *erreur += inter*inter;
  }
  *erreur = sqrt(*erreur);
}


//Création de la matrice sur CPU pour accès GPU
int main(int argc, char **argv){

    long i, n;
    long long size;
    REAL_T *norm, *error, error_cpu,norm_cpu;
    error_cpu = 99;
    REAL_T *A, *A_i, *X, *Y, *d_A, *d_X,*d_Y;
    double start_time, total_time;
    int n_iterations;
    FILE *output;



    if (argc < 2){
        printf("USAGE: %s [n]\n", argv[0]);
        exit(1);
    }
    n = atoll(argv[1]);
    size = n * n * sizeof(REAL_T);
    printf("taille de la matrice : %.1f G\n", size / 1073741824.);

    /*** allocation de la matrice et des vecteurs ***/
    A = (REAL_T *)malloc(size);
    if (A == NULL) {
        perror("impossible d'allouer la matrice");
        exit(1);
    }
    X = (REAL_T *)malloc(n * sizeof(REAL_T));
    Y = (REAL_T *)malloc(n * sizeof(REAL_T));
    if ((X == NULL) || (Y == NULL)) {
        perror("impossible d'allouer les vecteur");
        exit(1);
    }

    /*** initialisation de la matrice et de x ***/
    A_i = A;
    for (i = 0; i < n; i++) {
        init_ligne(A_i, i, n);
        A_i += n;
    }

    for (i = 0; i < n; i++) {
        X[i] = 1.0 / n;
    }

    //Initialisation de variables
    start_time = my_gettimeofday();
    n_iterations = 0;

    // Allocation GPU
    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_X, n * sizeof(REAL_T));
    cudaMalloc((void **) &d_Y, n * sizeof(REAL_T));
    cudaMalloc((void **) &error, 1 * sizeof(REAL_T));
    cudaMalloc((void **) &norm, 1 * sizeof(REAL_T));

    //Transfert CPU → GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, n * sizeof(REAL_T), cudaMemcpyHostToDevice);

    //BOUCLE DE CALCUL
    int nb =0;
    while (error_cpu > 0.00005) {
      nb++;

      //KERNEL 1 : Multipication matrice
      //Lancement de kernel (asynchrone) :
      //Définition des variables GPU
      dim3 threadsParBloc(TAILLE_BLOC_X);
      dim3 tailleGrille(ceil(n/(float) TAILLE_BLOC_X));
      dim3 threadsParBloc2(1);
      dim3 tailleGrille2(1);

      matmulKernel<<< tailleGrille, threadsParBloc>>>(d_A, d_X, d_Y, n);

      //KERNEL 2 : Norme euclidienne total
      //Définition des variables GPU
      norm_to_Kernel<<< threadsParBloc2, tailleGrille2>>>(d_Y, n, norm);

      //KERNEL 3 : Applique la norme
      normKernel<<< tailleGrille, threadsParBloc>>>(d_Y, norm, n);

      // KERNEL 4 : Ecart quadratique
      errorKernel<<< threadsParBloc2, tailleGrille2>>>(d_X, d_Y, n, error);

      //COMMUNICATION GPU -> CPU
      cudaMemcpy(&error_cpu, error, 1*sizeof(REAL_T), cudaMemcpyDeviceToHost);
      n_iterations ++ ;

      cudaMemcpy(X, d_Y, n*sizeof(REAL_T), cudaMemcpyDeviceToHost);
      cudaMemcpy(d_X, X, n * sizeof(REAL_T), cudaMemcpyHostToDevice);


    }
    cudaMemcpy(Y, d_Y, n, cudaMemcpyDeviceToHost);
    cudaMemcpy(&norm_cpu, norm, 1*sizeof(REAL_T), cudaMemcpyDeviceToHost);

    total_time = my_gettimeofday() - start_time;

    printf("erreur finale après %4d iterations : %g (|VP| = %g)\n", n_iterations, error_cpu, norm_cpu);
    printf("temps : %.1f s      Mflop/s : %.1f \n", total_time, (2.0 * n * n + 7.0 * n) * n_iterations / 1048576. / total_time);
    /*** stocke le vecteur propre dans un fichier ***/
    output = fopen("result.out", "w");
    if (output == NULL) {
        perror("impossible d'ouvrir result.out en écriture");
        exit(1);
    }
    fprintf(output, "%ld\n", n);
    for (i = 0; i < n; i++) {
        fprintf(output, "%.17g\n", Y[i]);
    }
    fclose(output);

    /* Libération mémoire GPU et CPU : */

    cudaFree(d_A); cudaFree(d_X); cudaFree(d_Y);
    free(A); free(X); free(Y);

  }
