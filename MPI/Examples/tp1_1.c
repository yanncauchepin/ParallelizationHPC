#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>


#define SIZE_H_N 50

int main (int argc, char* argv [])
{
    int my_rank;
    int p;
    int source;
    int dest;
    int tag;
    char message[100];
    MPI_Status status;
    char hostname[SIZE_H_N];

    printf("Tous le monde me reçoit ! \n");
    gethostname(hostname, SIZE_H_N);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);


    if (my_rank != 0) {
        sprintf(message, "\n HELLO du processus #%d depuis %s ! \n", my_rank, hostname);
        dest = 0;
        MPI_Send(message, strlen(message)+1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    } else {
        for (source = 1; source < p ; source++)
        {
            printf("HELLO du processus #%d, source %d \n", my_rank, source);
            MPI_Recv(message, 100, MPI_CHAR, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
            printf("Sur %s, le processus #%d a recu le message: %s \n", hostname, my_rank, message);
        }
    }
    MPI_Finalize();
}

//mpicc -o test exercice2.c
//mpiexec -n 4 ./test
