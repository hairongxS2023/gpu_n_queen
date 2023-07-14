#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <typeinfo>
#include <stdio.h>
#include <cmath>
#include <set>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand_kernel.h>
#include "CycleTimer.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
using namespace std;

#define NUM_THREADS 256
#define N 20
#define weak_prob 0.2
#define prob 0.5
#define gen_max 50
// Initialize board
// Returns a random vector that represents row (i coordinate) of the queens
int *initialize()
{
    int *board = new int[N];
    for (int i = 0; i < N; i++)
    {
        board[i] = rand() % N;
    }
    return board;
}

void print_vec(const int *board)
{
    // Print board array
    cout << "[";
    for (int k = 0; k < N; k++)
    {
        cout << board[k] << " ";
    }
    cout << "]" << endl;
}

// Print board
void print_board(const int *board)
{
    // Print board array
    print_vec(board);

    // Print 2D board
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (board[j] == i)
            {
                cout << 'Q' << " ";
            }
            else
            {
                cout << '.' << " ";
            }
        }
        cout << endl;
    }
}

// Fitness function (no. of pairs of non-attacking queens)
int fitness(const int *board)
{
    int fitness = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            int queen1 = board[i];
            int queen2 = board[j];
            bool attack;
            // check same row
            if (queen1 == queen2)
            {
                attack = true;
            }
            // check diagonal
            else if (abs(queen1 - queen2) == abs(i - j))
            {
                attack = true;
            }
            // by construction, guaranteed to be different column
            else
            {
                attack = false;
            }

            if (attack == false)
            {
                fitness++;
            }
        }
    }
    return fitness;
} // end fitness()

__device__ int fitness_kernel(const int *board)
{
    int fitness = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            int queen1 = board[i];
            int queen2 = board[j];
            bool attack;
            // check same row
            if (queen1 == queen2)
            {
                attack = true;
            }
            // check diagonal
            else if (abs(queen1 - queen2) == abs(i - j))
            {
                attack = true;
            }
            // by construction, guaranteed to be different column
            else
            {
                attack = false;
            }

            if (attack == false)
            {
                fitness++;
            }
        }
    }
    return fitness;
}

__device__ void selection(int **population, int **selected_pop, int sel_size, int *fitness_vector, int pop_size, curandState_t *state)
{
    // int **selected_pop = new int*[sel_size];
    // Create an array of indices from 0 to len-1
    // int *indices = new int[pop_size];
    // int *local_fitness_vector = new int[pop_size];
    // for (int i = 0; i < pop_size; i++)
    // {
    //     indices[i] = i;
    //     local_fitness_vector[i] = fitness_vector[i];
    // }

    // thrust::device_ptr<int> d_fitness_vector = local_fitness_vector;
    // thrust::device_ptr<int> d_indices(indices);
    // thrust::sort(d_indices, d_indices + pop_size, 
    //              [d_fitness_vector] __device__(int a, int b)
    //              { return d_fitness_vector[a] < d_fitness_vector[b]; });
    for (int i = 0; i < sel_size; i++)
    {
        float random_num = curand_uniform(state);
        int sel_idx = int(random_num * (pop_size - 1));
        // if (threadIdx.x == 0) {
        //     printf("sel_idx = %d\n", sel_idx);
        // }
        selected_pop[i] = population[sel_idx];
        // float r = curand_uniform(state);
        // if (r < weak_prob)
        // {
        //     // Randomly select a weak offspring
        //     int weak_idx = d_indices[int(curand_uniform(state) * chunk_size)];
        //     selected_pop[i] = population[weak_idx];
        // }
        // else
        // {
        //     // Select a strong offspring
        //     int strong_idx = d_indices[chunk_size - i];
        //     selected_pop[i] = population[strong_idx];
        // }
    }
    // delete[] local_fitness_vector;
    // delete[] indices;
    // return selected_pop;
} // end selection

__device__ void cross_over(int **selected_pop, int **chunk_pop, int pop_size, int new_sel_size, curandState_t *state)
{
    int threadId = threadIdx.x;
    // int **population_crossover = new int *[pop_size];
    
    // Copy selected population to the crossover population
    for (int i = 0; i < new_sel_size; i++)
    {
        chunk_pop[i] = selected_pop[i];
    }

    for (int i = new_sel_size; i < pop_size; i++)
    {
        int *pair1 = selected_pop[int(curand_uniform(state) * (new_sel_size - 1))];
        int *pair2 = selected_pop[int(curand_uniform(state) * (new_sel_size - 1))];
        int cross_loc = int(curand_uniform(state) * (N - 1));
        int *new_ind = new int[N];
        for (int j = 0; j < N; j++)
        {
            if (j <= cross_loc)
            {
                new_ind[j] = pair1[j];
            }
            else
            {
                new_ind[j] = pair2[j];
            }
        }
        chunk_pop[i] = new_ind;
        // if (threadId == 0) {
        //     printf("in crossover pop -> [");
        //     for (int j=0; j<N; j++) {
        //         printf("%d ", pair1[j]);
        //     }
        //     printf("]\n");
        // }
        // free(pair1);
        // free(pair2);
        free(new_ind);
    }
    // if (threadId == 127) {
    //     printf("cross over end -----------------------\n");
    // }
    // __syncthreads();
    // return population_crossover;
}
//  // end cross_over()

__device__ void mutate(int **population, int pop_size, curandState_t *state)
{
    // int **mutate_pop = (int **)malloc(pop_size * sizeof(int *));

    // Copy the original population to the mutated population
    // for (int i = 0; i < pop_size; i++)
    // {
    //     int *board_copy = (int *)malloc(N * sizeof(int));
    //     for (int j = 0; j < N; j++)
    //     {
    //         board_copy[j] = population[i][j];
    //     }
    //     mutate_pop[i] = board_copy;
    // }

    for (int i = 0; i < pop_size; i++)
    {
        float r = curand_uniform(state);
        if (r < prob)
        {
            int col = int(curand_uniform(state) * (N - 1));
            int row = int(curand_uniform(state) * (N - 1));

            // int temp = mutate_pop[i][rand1];
            population[i][col] = row;//mutate_pop[i][rand2];
            // mutate_pop[i][rand2] = temp;
        }
    }
    // return mutate_pop;
}

static int **initPopulation(int **hostPopulation, int pop_size) {
    int **devicePopulation;

    // Allocate memory for device population
    cudaMalloc(&devicePopulation, pop_size * sizeof(int *));

    // Allocate and copy the individual boards to the device
    for (int i = 0; i < pop_size; i++) {
        int *deviceBoard;
        cudaMalloc(&deviceBoard, N * N * sizeof(int));
        cudaMemcpy(deviceBoard, hostPopulation[i], N * N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&(devicePopulation[i]), &deviceBoard, sizeof(int *), cudaMemcpyHostToDevice);
    }

    // Return the pointer to the device population
    return devicePopulation;
}

__global__ void curandSetup(curandState_t *state, unsigned long long seed_offset) {
    int id = threadIdx.x;
    curand_init(id + seed_offset, 0, 0, &state[id]);
}

__global__ void run_genetic(curandState_t *state, int **d_population, int *fitness_vector, int pop_size) {
    int f_max = (N * (N - 1)) / 2;
    int threadId = threadIdx.x;
    int chunk_pop_size = pop_size / NUM_THREADS; // (pop_size + NUM_THREADS - 1) / NUM_THREADS;
    int chunk_sel_size = chunk_pop_size / 10;
    int start_idx = threadId * chunk_pop_size;
    int end_idx = min(start_idx + chunk_pop_size, pop_size);

    curandState_t threadState = state[threadId];
    int **d_chunk_pop = new int*[chunk_pop_size];
    int **d_chunk_pop_selected = new int*[chunk_sel_size];
    for (int i = 0; i < chunk_pop_size; i++) {
        int pos = min(start_idx + i, pop_size - 1);
        d_chunk_pop[i] = d_population[pos];
    }

    // if (threadId == 127) {
    //     printf("start_idx = %d, end_idx = %d, chunk_pop_size = %d, chunk_sel_size = %d\n", start_idx, end_idx, chunk_pop_size, chunk_sel_size);
    //     for (int i=0; i<chunk_pop_size; i++) {
    //         printf("%d pop -> [", i);
    //         for (int j=0; j<N; j++) {
    //             printf("%d ", d_chunk_pop[i][j]);
    //         }
    //         printf("]\n");
    //     }
    //     printf("-----------starting generation------------\n");
    // }

    for (int gen = 2; gen <= gen_max; gen++) {
        int f_curr = 0;
        selection(d_chunk_pop, d_chunk_pop_selected, chunk_sel_size, fitness_vector, chunk_pop_size, &threadState);
        cross_over(d_chunk_pop_selected, d_chunk_pop, chunk_pop_size, chunk_sel_size, &threadState);
        // if (threadId == 127) {
        //     printf("2. start_idx = %d, end_idx = %d, chunk_pop_size = %d, chunk_sel_size = %d\n", start_idx, end_idx, chunk_pop_size, chunk_sel_size);
        // }
        mutate(d_chunk_pop, chunk_pop_size, &threadState);
        // if (threadId == 127) {
        //     printf("3. start_idx = %d, end_idx = %d, chunk_pop_size = %d, chunk_sel_size = %d\n", start_idx, end_idx, chunk_pop_size, chunk_sel_size);
        // }
        
        for (int i=0; i < chunk_pop_size; i++) {
            int* board = d_chunk_pop[i];
            int f_score = fitness_kernel(board);
            fitness_vector[i] = f_score;
            if (f_score > f_curr)
            {
                f_curr = f_score;
            }
            if (f_score == f_max)
            {
                printf("Thread %d: Solution found\n", threadId);
                break;
            }
        }
        __syncthreads();
        if (threadId == 127) {
            printf("Thread %d: Generation %d: f_curr=%d\n", threadId, gen, f_curr);
        }
        // if (threadId == 127) {
        //     printf("Thread %d: Generation %d: f_curr=%d\n", threadId, gen, f_curr);
        //     for (int i=0; i<=chunk_sel_size; i++) {
        //         printf("%d pop -> [", i);
        //         for (int j=0; j<N; j++) {
        //             printf("%d ", d_chunk_pop_selected[i][j]);
        //         }
        //         printf("]\n");
        //     }
        // }
    }
    __syncthreads();
    // if (threadId == 0) {
    //     printf("start_idx = %d, end_idx = %d, chunk_sel_size = %d\n", start_idx, end_idx, chunk_sel_size);
    //     for (int i=start_idx; i<end_idx; i++) {
    //         printf("%d pop -> [", i);
    //         for (int j=0; j<N; j++) {
    //             printf("%d ", d_population_2[i][j]);
    //         }
    //         printf("]\n");
    //     }
    //     printf("pop -> [");
    //     for (int j=0; j<N; j++) {
    //         printf("%d ", d_selected_pop[chunk_sel_size - 1][j]);
    //     }
    //     printf("]\n");
    //     printf("------after cross over -------\n");
    //     printf("pop -> [");
    //     for (int k=0; k<N; k++) {
    //         printf("%d ", d_population_crossover[chunk_sel_size][k]);
    //     }
    //     printf("]\n");
    // }

    // for (int gen = 0; gen < 1000; gen++) {
    //     d_population = selection(d_population, fitness_vector, weak_prob, pop_size, &threadState);
    //     d_population = cross_over(d_population, pop_size, sel_size, N, &threadState);



    //     __syncthreads();
    //     int f_curr = 0;
    //     for (int i = start_idx; i < end_idx; i++)
    //     {
    //         int* board = d_population[i];
    //         int f_score = fitness_kernel(board, N);
    //         fitness_vector[i] = f_score;
    //         if (f_score > f_curr)
    //         {
    //             f_curr = f_score;
    //         }
    //         if (f_score == f_max)
    //         {
    //             printf("Thread %d: Solution found\n", threadId);
    //             break;
    //         }
    //         __syncthreads();
    //     }

    //     printf("Thread %d: Generation %d: f_curr=%d\n", threadId, gen, f_curr);
    //     __syncthreads();
    // }
}

static int *initFitnessVector(int *hostFitnessVector, int pop_size) {
    int *deviceFitnessVector;
    cudaMalloc(&deviceFitnessVector, pop_size * sizeof(int));
    cudaMemcpy(deviceFitnessVector, hostFitnessVector, pop_size * sizeof(int), cudaMemcpyHostToDevice);
    return deviceFitnessVector;
}



int main()
{
    // // measure CPU time
    // clock_t begin = clock();
    // Seed random generator
    srand(time(0));
    double startTime = CycleTimer::currentSeconds();

    //////// Parameters ////////
    // Set dimension of board NxN
    // int N = 20;
    cout << "N = " << N << endl;
    // Fixed population size
    // 20, 50, 100, 200
    int pop_size = N * 20 * 128;
    cout << "pop_size = " << pop_size << endl;
    // Selection size
    // int sel_size = pop_size / 10;
    // Probability of randomly including weak offspring in selection
    // float weak_prob = 0.2;
    // Mutation probability
    // float prob = 0.5;
    // Maximum generations to iterate
    // int gen_max = 1000;
    ////////////////////////////

    // Maximum theoretical value of fitness (N choose 2)
    int f_max = (N * (N - 1)) / 2;
    cout << "f_max=" << f_max << endl;
    // Current best fitness
    int f_curr = 0;
    // Population
    int **population = new int *[pop_size];
    // Fitness vector
    int *fitness_vector = new int[pop_size];
    // Generation number
    int gen = 1;

    curandState_t *state;
    int curandState_byte = NUM_THREADS * sizeof(curandState_t);
    cudaMalloc(&state, curandState_byte);
    unsigned long long seed = CycleTimer::currentTicks();
    curandSetup<<<1, NUM_THREADS>>>(state, seed);
    cudaDeviceSynchronize();

    // Initialize Population
    for (int i = 0; i < pop_size; i++)
    {
        int *board = initialize();
        population[i] = board;
        // print_board(board);
        // cout << fitness(board) << endl;
        int f_score = fitness(board);
        fitness_vector[i] = f_score;
        if (f_score > f_curr)
        {
            f_curr = f_score;
        }
        if (f_score == f_max)
        {
            cout << "Solution found:" << endl;
            // print_board(board);
            break;
        }
    } // end for
    cout << "Generation " << gen << ": f_curr=" << f_curr << endl;
    gen++;

    double copyStartTime = CycleTimer::currentSeconds();
    int **d_population = initPopulation(population, pop_size);
    int *d_fitness_vector = initFitnessVector(fitness_vector, pop_size);
    double copyEndTime = CycleTimer::currentSeconds();
    double copyTime = copyEndTime - copyStartTime;
    printf("Time elapsed to transfer data (seconds): %.2fs\n", copyTime);

    run_genetic<<<1, NUM_THREADS>>>(state, d_population, d_fitness_vector, pop_size);
    cudaDeviceSynchronize();
    // Time elapsed
    double endTime = CycleTimer::currentSeconds();
    double elapsed_secs = endTime - startTime;
    printf("Time elapsed (seconds): %.4fs\n", elapsed_secs);
    return 0;
} // end main()