//puissance_cuda.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "defs.h"

#define TAILLE_BLOC_X 256
#define NB_ELEM 16

//nb de bloc : gridDim.x
//indice de parcours des blocs : blockIdx.x

//nb de thread par bloc : blockDim.x
//indice de parcours des threads : threadIdx.x

__global__ void matmulKernel(REAL_T* d_A, REAL_T* d_B, REAL_T* d_C, int n,REAL_T *norm) {
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i*NB_ELEM < n)
  {
    for(int line = 0; line<NB_ELEM; line++)
    {
      int line_n = i*NB_ELEM+line;
      REAL_T temp = 0;
      for(int k=0; k<n; k++)
         temp += d_A[line_n * n + k] * d_B[line_n];
      d_C[line_n] = temp;
      REAL_T inter = temp*temp;
      atomicAdd(norm,inter);
      }
  }
}

__global__ void norm_to_Kernel(REAL_T *norm) {
  *norm = sqrt(*norm);
}


__global__ void normKernel(REAL_T* d_Y,REAL_T* d_X, REAL_T *norm, int n, REAL_T* erreur) {
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;

  if (i*NB_ELEM < n)
  {
    REAL_T temp;
    REAL_T inter;
    for(int line = 0; line<NB_ELEM; line++)
    {
      int line_n = i*NB_ELEM+line;
      temp = d_Y[line_n]/(*norm);
      d_Y[line_n] = temp;
      inter =  d_X[line_n] - temp;
      inter = inter*inter;
      atomicAdd(erreur,inter);
    }
  }
}

__global__ void errorKernel(REAL_T *erreur) {
  *erreur = sqrt(*erreur);
}


//Création de la matrice sur CPU pour accès GPU
int main(int argc, char **argv){

    long i, n;
    long long size;
    REAL_T *norm, *error, error_cpu,norm_cpu,zero=0;
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

    dim3 threadsParBloc(ceil(TAILLE_BLOC_X/(float)NB_ELEM));
    dim3 tailleGrille(ceil(n/(float) TAILLE_BLOC_X));
    dim3 threadsParBloc2(1);
    dim3 tailleGrille2(1);

    //BOUCLE DE CALCUL
    while (error_cpu > 0.00005) {

      //KERNEL 1 : Multipication matrice
      //Lancement de kernel (asynchrone) :
      //Définition des variables GPU
      cudaMemcpy(norm, &zero, 1*sizeof(REAL_T), cudaMemcpyHostToDevice);
      cudaMemcpy(error, &zero, 1*sizeof(REAL_T), cudaMemcpyHostToDevice);

      matmulKernel<<< tailleGrille, threadsParBloc>>>(d_A, d_X, d_Y, n, norm);
      //KERNEL 2 : Norme euclidienne total
      //Définition des variables GPU
      norm_to_Kernel<<< threadsParBloc2, tailleGrille2>>>(norm);

      //KERNEL 3 : Applique la norme
      normKernel<<< tailleGrille, threadsParBloc>>>(d_Y, d_X, norm, n, error);

      // KERNEL 4 : Ecart quadratique
      errorKernel<<< threadsParBloc2, tailleGrille2>>>(error);

      //COMMUNICATION GPU -> CPU
      cudaMemcpy(&error_cpu, error, 1*sizeof(REAL_T), cudaMemcpyDeviceToHost);
      cudaMemcpy(X, d_Y, n * sizeof(REAL_T), cudaMemcpyDeviceToHost);
      cudaMemcpy(d_X, X, n * sizeof(REAL_T), cudaMemcpyHostToDevice);

      n_iterations ++ ;

    }
    cudaMemcpy(Y, d_Y, n, cudaMemcpyDeviceToHost);
    cudaMemcpy(&norm_cpu, norm, 1*sizeof(REAL_T), cudaMemcpyDeviceToHost);

    total_time = my_gettimeofday() - start_time;

    printf("Nombre d'op/thread : %d, Nombre de threads: %d, Nombre de blocs: %d\n",n*NB_ELEM,threadsParBloc.x,tailleGrille.x);
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
