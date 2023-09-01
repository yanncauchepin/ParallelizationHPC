/* indent -nfbs -i4 -nip -npsl -di0 -nut iterated_seq.c */
/* Auteur: C. Bouillaguet et P. Fortin (Univ. Lille) */ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>



#define PRNG_MAX 0x0007FFFFFFFll
#define PRNG_1   0x00473EE661Dll
#define PRNG_2   0x024719D0275ll
#define RANGE    101

double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

// Initialise la ligne N°i (pointée par A_i) de la matrice de taille n x n : 
void init_ligne(double *A_i, long i, long n){
  for (long j = 0; j < n; j++) {
    A_i[j] = (((double)((i * i * PRNG_1 + j * j * PRNG_2) & PRNG_MAX)) / PRNG_MAX) / n;
  }
  for (long k = 1; k < n; k *= 2) {
    if (i + k < n) {
      A_i[i + k] = ((i - k) * PRNG_2 + i * PRNG_1) % RANGE;
    }
    if (i - k >= 0) {
      A_i[i - k] = ((i + k) * PRNG_2 + i * PRNG_1) % RANGE;
    }
  }
}

int main(int argc, char **argv){
	
	
	long n;
	double start_time, total_time;
	
	
	/*
	if (argc < 2)
	{
		printf("USAGE: %s [n]\n", argv[0]);
		exit(1);
	}*/
	/*n = atoll(argv[1]);*/
	
	n = 16376;

	int p;
	int my_rank;

	MPI_Init(&argc,&argv);

	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&p);

	int nb_processus=p;
	int taille_bloc,source;

	taille_bloc = n/nb_processus;
	
	printf("PROCESSUS : %d G\n", my_rank);
	printf("taille de la matrice : %d G\n", n*n*sizeof(double));
	printf("taille bloc : %d G\n", taille_bloc);
	
	double* X = (double*)malloc(n*sizeof(double));
	double* Y = (double*)malloc(taille_bloc*sizeof(double));
	
	double norme;
	double norme_totale;
	double erreur;
	double erreur_totale;
	double delta;
	int fini=0;
	
	if(my_rank!=0)
	{
		double *bloc=(double*)malloc(taille_bloc*n*sizeof(double));
		
		MPI_Recv(bloc,taille_bloc*n,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&status);


		while(fini==0)
		{
			//printf("PROCESSUS : %d RECEPTION X\n", my_rank);
			MPI_Bcast(X,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
			//printf("PROCESSUS : %d RECEPTION X FAITE\n", my_rank);
			
			norme = 0;
			for(int i=0;i<taille_bloc;i++)
			{
				Y[i] = 0;
				
				for(int j=0;j<n;j++)
				{
					Y[i] += bloc[i*n + j] * X[j];
				}
				norme += Y[i]*Y[i];
			}
			
			//printf("PROCESSUS : %d ENVOIE NORME %lf\n", my_rank, norme);
			MPI_Reduce(&norme,&norme_totale,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
			//printf("PROCESSUS : %d FIN ENVOIE NORME %lf\n", my_rank, norme);
			
			
			MPI_Bcast(&norme,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
			//printf("PROCESSUS : %d RECEPTION NORME\n", my_rank);
			
			erreur=0;
			
			for(int i =0; i <taille_bloc;i++)
			{
				Y[i] = Y[i]/norme;
				delta = X[i+my_rank*taille_bloc] - Y[i];
				erreur += delta*delta;
			}
			
			MPI_Reduce(&erreur,&erreur_totale,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
			MPI_Gather(Y,taille_bloc,MPI_DOUBLE,X,taille_bloc,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Bcast(&fini,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
			
		}
		
		
		free(bloc);
		
	}
	else
	{
		srandom (123);
		double *A, *A_i;
		
		int n_iterations;
		FILE *output;
		
		double *bloc=(double*)malloc(taille_bloc*n*sizeof(double));
		
		
		/*** allocation de la matrice et des vecteurs ***/
		A = (double *)malloc(n*taille_bloc*sizeof(double));
		
		if (A == NULL) {
			perror("impossible d'allouer la sous-matrice");
			exit(1);
		}
		if (bloc == NULL) {
			perror("impossible d'allouer la matrice");
			exit(1);
		}
		if (X == NULL)
		{
			perror("impossible d'allouer les vecteur");
			exit(1);
		}
		
		for (int i = 0; i < n; i++)
		{
			X[i] = 1.0/n;
		}
			
		/* BLOC MATRICE PROCESSUS 0 */
		A_i = A;
		long nb_ligne =0;
		for(long j=0; j<taille_bloc;j++)
		{
			init_ligne(A_i, j, n);
			nb_ligne++;
			A_i += n;
		}
		/*FIN BLOC MATRICE PROCESSUS 0*/
		
		
		/* initialisation des blocs matrice */
		start_time = my_gettimeofday();
		
		for (int i = 1; i < nb_processus; i++)
		{
			A_i = bloc;
			for(int j=0; j<taille_bloc;j++)
			{
				init_ligne(A_i, nb_ligne, n);
				nb_ligne++;
				A_i += n;
			}
			MPI_Send(bloc,taille_bloc*n,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
		}
		free(bloc);
		MPI_Bcast(X,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		
		printf("MATRICE ET VECTEUR ENVOYES \n");
		
		/*fin initialisation des blocs matrice*/

		start_time = my_gettimeofday();
		n_iterations = 0;
		
		
		erreur = INFINITY;
		while(fini ==0)
		{
			printf("iteration %4d, erreur actuelle %lf \n", n_iterations, erreur);
			
			/*TRAVAIL CALCUL PARTIE Y*/
			
			norme=0;
			
			for(int i=0;i<taille_bloc;i++)
			{
				Y[i] = 0;
				
				for(int j=0;j<n;j++)
				{
					Y[i] += A[i*n + j] * X[j];
				}
				norme += Y[i]*Y[i];
			}
			
			
			/*FIN TRAVAIL CALCUL PARTIE Y*/
			MPI_Reduce(&norme,&norme_totale,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
			
			norme = sqrt(norme_totale);
			
			MPI_Bcast(&norme,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
			
			/*TRAVAIL CALCUL erreur*/
			
			erreur=0;
			for(int i =0; i <taille_bloc;i++)
			{
				Y[i] = Y[i]/norme;
				delta = X[i] - Y[i];
				erreur += delta*delta;
				
				
				X[i] = Y[i]; //X[0:taille_bloc]=Y
			}
			
			
			/*FIN TRAVAIL CALCUL erreur*/
			
			MPI_Reduce(&erreur,&erreur_totale,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
			MPI_Gather(Y,taille_bloc,MPI_DOUBLE,X,taille_bloc,MPI_DOUBLE,0,MPI_COMM_WORLD);
			
			
			if(erreur <1e-9)
			{
				fini = 1;
				MPI_Bcast(&fini,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
			}
			else
			{
				fini = 0;
				MPI_Bcast(&fini,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
				MPI_Bcast(X,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
			}
			
			n_iterations++;
		}
		
		total_time = my_gettimeofday() - start_time;
		printf("erreur finale après %4d iterations : %g (|VP| = %g)\n", n_iterations, erreur, norme);
		printf("temps : %.1f s      Mflop/s : %.1f \n", total_time, (2.0 * n * n + 7.0 * n) * n_iterations / 1048576. / total_time);
		
		
		/*** stocke le vecteur propre dans un fichier ***/
		output = fopen("result.out", "w");
		if (output == NULL) {
			perror("impossible d'ouvrir result.out en écriture");
			exit(1);
		}
		fprintf(output, "%ld\n", n);
		for (int i = 0; i < n; i++) {
			fprintf(output, "%.17g\n", X[i]);
		}
		fclose(output);

		free(A);
	}
	
	free(Y);
	free(X);
	
	
	MPI_Finalize();
	return 0;
}
