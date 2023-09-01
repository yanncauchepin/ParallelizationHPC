/* indent -nfbs -i4 -nip -npsl -di0 -nut iterated_seq.c */
/* Auteur: C. Bouillaguet et P. Fortin (Univ. Lille) */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "defs.h"

int main(int argc, char **argv){
    long i, j, k, n;
    long long size;
    REAL_T norm, inv_norm, error, delta;
    REAL_T *A, *A_i, *X, *Y;
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

    start_time = my_gettimeofday();
    error = INFINITY;
    n_iterations = 0;
    while (error > ERROR_THRESHOLD) {
        printf("iteration %4d, erreur actuelle %g\n", n_iterations, error);

        /*** y <--- M.x ***/
        A_i = A;
        for (i = 0; i < n; i++) {
            Y[i] = 0;
            for (j = 0; j < n; j++) {
                Y[i] += A_i[j] * X[j];
            }
            A_i += n;
        }

        /*** norm <--- ||y|| ***/
        norm = 0;
        for (i = 0; i < n; i++) {
            norm += Y[i] * Y[i];
        }
        norm = sqrt(norm);

        /*** y <--- y / ||y|| ***/
        inv_norm = 1.0 / norm;
        for (i = 0; i < n; i++) {
            Y[i] *= inv_norm;
        }

        /*** error <--- ||x - y|| ***/
        error = 0;
        for (i = 0; i < n; i++) {
            delta = X[i] - Y[i];
            error += delta * delta;
        }
        error = sqrt(error);

	/*** x <--> y ***/
 	REAL_T *tmp = X; X = Y ; Y = tmp;

        n_iterations++;
    }

    total_time = my_gettimeofday() - start_time;
    printf("erreur finale après %4d iterations : %g (|VP| = %g)\n", n_iterations, error, norm);
    printf("temps : %.1f s      Mflop/s : %.1f \n", total_time, (2.0 * n * n + 7.0 * n) * n_iterations / 1048576. / total_time);

    /*** stocke le vecteur propre dans un fichier ***/
    output = fopen("result.out", "w");
    if (output == NULL) {
        perror("impossible d'ouvrir result.out en écriture");
        exit(1);
    }
    fprintf(output, "%ld\n", n);
    for (i = 0; i < n; i++) {
        fprintf(output, "%.17g\n", X[i]);
    }
    fclose(output);

    free(A);
    free(X);
    free(Y);
}
