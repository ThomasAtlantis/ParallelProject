#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#define size 100
#define root 0
#define maxp 10

void MPI_Allgatherv_mine(
        const void * send_buff, int send_count, MPI_Datatype send_type,
        void * recv_buff, const int * recv_counts, MPI_Datatype recv_type) {
    int i, id, pn, recv_count = 0; MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &pn);
    if (id != root) {
        for (i = 0; i < pn; ++ i) recv_count += recv_counts[i];
        MPI_Send(send_buff, send_count, send_type, root, 0, MPI_COMM_WORLD);
        MPI_Recv(recv_buff, recv_count, recv_type, root, 0, MPI_COMM_WORLD, &status);
    } else {
        strncpy((char *)recv_buff, (char *)send_buff, send_count * sizeof(send_type));
        recv_count = send_count;
        for (i = root + 1; i < pn; ++ i) {
            MPI_Recv((char *)recv_buff + recv_count * sizeof(recv_type), recv_counts[i], recv_type, i, 0, MPI_COMM_WORLD, &status);
            recv_count += recv_counts[i];
        }
        for (i = root + 1; i < pn; ++ i) {
            MPI_Send(recv_buff, recv_count, send_type, i, 0, MPI_COMM_WORLD);
        }
    }
}

void display(double pp, const double * recv_buff, const int * recv_counts, double elapsed_time) {
    int i, pn, id, offset = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &pn);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    printf("Process [%d] sent: ", id);
    printf("%7.4lf x%d", pp, size);
    printf(", received: ");
    for (i = 0; i < pn; ++ i) {
        printf("%7.4lf x%d, ", recv_buff[offset], size);
        offset += recv_counts[i];
    }
    printf("Time: %10.6fs\n", elapsed_time);
}

int main(int argc, char* argv[]) {
    double elapsed_time;
    int i, id, pn, recv_counts[maxp];
    double send_buff[size], recv_buff[size * maxp];

    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &pn);

    const double pi = 3.1415926535;
    const double pp = (id + 1) * pi;
    for (int i = 0; i < size; ++ i) {
        send_buff[i] = pp;
    }
    for (int i = 0; i < pn; ++ i) {
        recv_counts[i] = size;
    }
    printf("MPI_DOUBLE: %lu, double: %lu\n", sizeof(MPI_DOUBLE), sizeof(double));

    elapsed_time = -MPI_Wtime();
    MPI_Allgatherv_mine(send_buff, size, MPI_DOUBLE, recv_buff, recv_counts, MPI_DOUBLE);
    elapsed_time += MPI_Wtime();

    display(pp, recv_buff, recv_counts, elapsed_time);

    MPI_Finalize();
    return 0;
}