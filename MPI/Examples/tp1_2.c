#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>


#define SIZE_H_N 50

int main (int argc, char* argv [])
{
    int my_rank,p,source,dest,tag=0;
    char message[100];
    MPI_Status status;
    char hostname[SIZE_H_N];


    gethostname(hostname, SIZE_H_N);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);


    sprintf(message, "Coucou du processus #%d depuis %s !", my_rank, hostname);

    MPI_Send(message, strlen(message)+1, MPI_CHAR,(my_rank+1)%p,tag,MPI_COMM_WORLD);


    if(my_rank == 0) {
        source = p-1;
    } else {
        source = my_rank-1;
    }

    MPI_Recv(message, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
    printf("Sur %s, le processus #%d a recu le message: %s \n", hostname, my_rank,message);

    MPI_Finalize();
}
