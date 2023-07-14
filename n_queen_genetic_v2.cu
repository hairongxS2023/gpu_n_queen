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
#define weak_prob 0.4
#define prob 0.3
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

__device__ void initialize_kernel(int* board, curandState_t *state)
{
    for (int i = 0; i < N; i++)
    {
        board[i] = int(curand_uniform(state) * (N - 1));
    }
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

__device__ void print_vec(int *board)
{
    // Print board array
    printf("[");
    for (int k = 0; k < N; k++)
    {
        printf("%d ", board[k]);
    }
    printf("]\n");
}

// Print board
__device__ void print_board_kernel(int *board)
{
    print_vec(board);
    // Print 2D board
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (board[j] == i)
            {
                printf("Q ");
            }
            else
            {
                printf(". ");
            }
        }
        printf("\n");
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

__device__ void merge(int *indices, int *fitness_vector, int left, int mid, int right)
{
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = new int[n1];
    int *R = new int[n2];

    for (i = 0; i < n1; i++)
        L[i] = indices[left + i];
    for (j = 0; j < n2; j++)
        R[j] = indices[mid + 1 + j];

    i = 0;
    j = 0;
    k = left;
    while (i < n1 && j < n2)
    {
        if (fitness_vector[L[i]] <= fitness_vector[R[j]])
        {
            indices[k] = L[i];
            i++;
        }
        else
        {
            indices[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        indices[k] = L[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        indices[k] = R[j];
        j++;
        k++;
    }

    delete[] L;
    delete[] R;
}

// Iterative Merge Sort function
__device__ void mergeSort(int *indices, int *fitness_vector, int n)
{
    int curr_size;
    int left_start;

    for (curr_size = 1; curr_size <= n - 1; curr_size = 2 * curr_size)
    {
        for (left_start = 0; left_start < n - 1; left_start += 2 * curr_size)
        {
            int mid = min(left_start + curr_size - 1, n - 1);
            int right_end = min(left_start + 2 * curr_size - 1, n - 1);

            merge(indices, fitness_vector, left_start, mid, right_end);
        }
    }
}

__device__ void selection(int **population, int **selected_pop, int sel_size, int *fitness_vector, int pop_size, curandState_t *state)
{
    int threadId = threadIdx.x;
    int *indices = new int[pop_size];
    for (int i = 0; i < pop_size; i++)
    {
        indices[i] = i;
    }

    mergeSort(indices, fitness_vector, pop_size);

    // if (threadId == 0) {
    //     for (int i=0; i<pop_size; i++) {
    //         printf("pop %d -> fitness: %d\n", i, fitness_vector[indices[i]]);
    //     }
    // }

    for (int i = 0; i < sel_size; i++)
    {
        // float random_num = curand_uniform(state);
        // int sel_idx = int(random_num * (pop_size - 1));
        // selected_pop[i] = population[sel_idx];
        float r = curand_uniform(state);
        int idx;
        if (r < weak_prob)
        {
            // Randomly select a weak offspring
            idx = indices[curand(state) % pop_size];
        }
        else
        {
            // Select a strong offspring
            idx = indices[pop_size - i - 1];
        }

        for (int j=0; j<N; j++) {
            selected_pop[i][j] = population[idx][j];
        }
    }
    delete[] indices;
} // end selection

__device__ void cross_over(int **selected_pop, int **chunk_pop, int pop_size, int new_sel_size, curandState_t *state)
{
    int threadId = threadIdx.x;
    // int **population_crossover = new int *[pop_size];
    
    // Copy selected population to the crossover population
    for (int i = 0; i < new_sel_size; i++)
    {
        for (int j=0; j < N; j++) {
            chunk_pop[i][j] = selected_pop[i][j];
        }
    }

    for (int i = new_sel_size; i < pop_size; i++)
    {
        int *pair1 = selected_pop[curand(state) % new_sel_size];
        int *pair2 = selected_pop[curand(state) % new_sel_size];
        int cross_loc = curand(state) % N;
        for (int j = 0; j < N; j++)
        {
            if (j <= cross_loc)
            {
                chunk_pop[i][j] = pair1[j];
            }
            else
            {
                chunk_pop[i][j] = pair2[j];
            }
        }
    }
}
//  // end cross_over()

__device__ void mutate(int **population, int pop_size, curandState_t *state)
{
    for (int i = 0; i < pop_size; i++)
    {
        float r = curand_uniform(state);
        if (r < prob)
        {
            int col = curand(state) % N;
            int row = curand(state) % N;
            population[i][col] = row;
        }
    }
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
    curand_init(id + seed_offset, id, 0, &state[id]);
}

__device__ int solution_found = 0;

__global__ void run_genetic(curandState_t *state, int *d_chunk_pop, int **d_chunk_pop_ptrs, int *d_chunk_pop_selected, int **d_chunk_pop_selected_ptrs, int *fitness_vector, int pop_size) {

    int threadId = threadIdx.x;
    curandState_t threadState = state[threadId];
    int chunk_sel_size = pop_size / 10;

    for (int i = 0; i < pop_size; i++) {
        int* board = d_chunk_pop + i * N;
        initialize_kernel(board, &threadState);
        d_chunk_pop_ptrs[i] = board;
    }

    for (int i = 0; i < chunk_sel_size; i++) {
        int* board = d_chunk_pop_selected + i * N;
        for (int j = 0; j < N; j++)
        {
            board[j] = 0;
        }
        d_chunk_pop_selected_ptrs[i] = board;
    }

    // if (threadId == 0) {
    //     for (int i=0; i<chunk_sel_size; i++) {
    //         printf("%d chunk_sel_pop -> [", i);
    //         for (int j=0; j<N; j++) {
    //              printf("%d ", d_chunk_pop_selected_ptrs[i][j]);
    //         }
    //         printf("]\n");
    //     }
    // }
    int f_curr = 0;
    int f_max = (N * (N - 1)) / 2;
    for (int i=0; i < pop_size; i++) {
        int* board = d_chunk_pop_ptrs[i];
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

    // // __syncthreads();
    for (int gen = 0; gen <= 50; gen++) {
        if (solution_found) {
            break;
        }
        selection(d_chunk_pop_ptrs, d_chunk_pop_selected_ptrs, chunk_sel_size, fitness_vector, pop_size, &threadState);
        cross_over(d_chunk_pop_selected_ptrs, d_chunk_pop_ptrs, pop_size, chunk_sel_size, &threadState);
        mutate(d_chunk_pop_ptrs, pop_size, &threadState);
        
        for (int i=0; i < pop_size; i++) {
            int* board = d_chunk_pop_ptrs[i];
            int f_score = fitness_kernel(board);
            fitness_vector[i] = f_score;
            if (f_score > f_curr)
            {
                f_curr = f_score;
            }
            // if (f_score == f_max)
            // {
            //     printf("Thread %d: Solution found\n", threadId);
            //     // if (threadId == 0) {
            //     //     print_board_kernel(board);
            //     // }
            //     atomicExch(&solution_found, 1);
            //     break;
            // }
        }
        __syncthreads();
        if (threadId == 0) {
            printf("Thread %d: Generation %d: f_curr=%d\n", threadId, gen, f_curr);
        }
    }
    __syncthreads();
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
    int pop_size = N * 200 * 128;
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
    // for (int i = 0; i < pop_size; i++)
    // {
    //     int *board = initialize();
    //     population[i] = board;
    //     // print_board(board);
    //     // cout << fitness(board) << endl;
    //     int f_score = fitness(board);
    //     fitness_vector[i] = f_score;
    //     if (f_score > f_curr)
    //     {
    //         f_curr = f_score;
    //     }
    //     if (f_score == f_max)
    //     {
    //         cout << "Solution found:" << endl;
    //         // print_board(board);
    //         break;
    //     }
    // } // end for
    // cout << "Generation " << gen << ": f_curr=" << f_curr << endl;
    // gen++;

    // double copyStartTime = CycleTimer::currentSeconds();
    // int **d_population = initPopulation(population, pop_size);
    // int *d_fitness_vector = initFitnessVector(fitness_vector, pop_size);
    // double copyEndTime = CycleTimer::currentSeconds();
    // double copyTime = copyEndTime - copyStartTime;
    // printf("Time elapsed to transfer data (seconds): %.2fs\n", copyTime);
    int chunk_pop_size = pop_size / NUM_THREADS;
    int chunk_sel_size = chunk_pop_size / 10;

    int *h_chunk_pop_selected = new int[chunk_sel_size * N];
    int **h_chunk_pop_selected_ptrs = new int*[chunk_sel_size];

    int *h_chunk_pop = new int[pop_size * N];
    int **h_chunk_pop_ptrs = new int*[pop_size];

    int *h_fitness_vector = new int[pop_size];

    int *d_chunk_pop;
    int **d_chunk_pop_ptrs;
    int *d_chunk_pop_selected;
    int **d_chunk_pop_selected_ptrs;
    int *d_fitness_vector;

    cudaMalloc(&d_chunk_pop_selected, chunk_sel_size * N * sizeof(int));
    cudaMalloc(&d_chunk_pop_selected_ptrs, chunk_sel_size * sizeof(int*));

    cudaMemcpy(d_chunk_pop_selected, h_chunk_pop_selected, chunk_sel_size * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chunk_pop_selected_ptrs, h_chunk_pop_selected_ptrs, chunk_sel_size * sizeof(int*), cudaMemcpyHostToDevice);

    cudaMalloc(&d_chunk_pop, pop_size * N * sizeof(int));
    cudaMalloc(&d_chunk_pop_ptrs, pop_size * sizeof(int*));

    cudaMemcpy(d_chunk_pop, h_chunk_pop, pop_size * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chunk_pop_ptrs, h_chunk_pop_ptrs, pop_size * sizeof(int*), cudaMemcpyHostToDevice);

    cudaMalloc(&d_fitness_vector, pop_size * sizeof(int));
    cudaMemcpy(d_fitness_vector, h_fitness_vector, pop_size * sizeof(int), cudaMemcpyHostToDevice);


    run_genetic<<<1, NUM_THREADS>>>(state, 
    d_chunk_pop, d_chunk_pop_ptrs, d_chunk_pop_selected, d_chunk_pop_selected_ptrs, d_fitness_vector, chunk_pop_size);
    cudaDeviceSynchronize();
    cudaFree(d_chunk_pop);
    cudaFree(d_chunk_pop_selected);
    cudaFree(d_fitness_vector);
    // Time elapsed
    double endTime = CycleTimer::currentSeconds();
    double elapsed_secs = endTime - startTime;
    printf("Time elapsed (seconds): %.4fs\n", elapsed_secs);
    return 0;
} // end main()