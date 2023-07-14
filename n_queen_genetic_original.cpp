#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <typeinfo>
#include <cmath>
#include <set>
using namespace std;

// Initialize board
// Returns a random vector that represents row (i coordinate) of the queens
int *initialize(const int &N)
{
    int *board = new int[N];
    for (int i = 0; i < N; i++)
    {
        board[i] = rand() % N;
    }
    return board;
}

void print_vec(const int *board, int N)
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
void print_board(const int *board, int N)
{
    // Print board array
    print_vec(board, N);

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
int fitness(const int *board, int N)
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

int **selection(int **population, int *fitness_vector, int sel_size, int weak_prob, int pop_size)
{
    int **selected_pop = new int *[sel_size];

    // Create an array of indices from 0 to len-1
    int *indices = new int[pop_size];
    for (int i = 0; i < pop_size; i++)
    {
        indices[i] = i;
    }

    // Sort the indices based on the fitness values
    std::sort(indices, indices + pop_size, [fitness_vector](int a, int b)
              { return fitness_vector[a] < fitness_vector[b]; });

    // Remove duplicates
    // int unique_len = 1;
    // for (int i = 1; i < pop_size; i++) {
    //     if (fitness_vector[indices[i]] != fitness_vector[indices[i-1]]) {
    //         indices[unique_len++] = indices[i];
    //     }
    // }

    for (int i = 0; i < sel_size; i++)
    {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (r < weak_prob)
        {
            // Randomly select a weak offspring
            int weak_idx = indices[rand() % pop_size];
            selected_pop[i] = population[weak_idx];
        }
        else
        {
            // Select a strong offspring
            int strong_idx = indices[pop_size - i];
            selected_pop[i] = population[strong_idx];
        }
    }
    delete[] indices;
    return selected_pop;
} // end selection

int **cross_over(int **population, int pop_size, int sel_size, int N)
{
    int **cross_pop = new int *[pop_size];
    // Copy selected population to the crossover population
    for (int i = 0; i < sel_size; i++)
    {
        cross_pop[i] = population[i];
    }

    for (int i = sel_size; i < pop_size; i++)
    {
        int *pair1 = population[rand() % sel_size];
        int *pair2 = population[rand() % sel_size];
        int cross_loc = rand() % (N - 1);

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
        cross_pop[i] = new_ind;
    }
    return cross_pop;
} // end cross_over()

int **mutate(int **population, int pop_size, int N, float prob)
{
    int **mutate_pop = new int *[pop_size];

    // Copy the original population to the mutated population
    for (int i = 0; i < pop_size; i++)
    {
        int *board_copy = new int[N];
        std::copy(population[i], population[i] + N, board_copy);
        mutate_pop[i] = board_copy;
    }

    for (int i = 0; i < pop_size; i++)
    {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (r < prob)
        {
            // Swap 2 indices randomly
            int rand1 = rand() % N;
            int rand2 = rand() % N;

            int temp = mutate_pop[i][rand1];
            mutate_pop[i][rand1] = mutate_pop[i][rand2];
            mutate_pop[i][rand2] = temp;
        }
    }
    return mutate_pop;
} // end mutate

int main()
{
    // measure CPU time
    clock_t begin = clock();
    // Seed random generator
    srand(time(0));

    //////// Parameters ////////
    // Set dimension of board NxN
    int N = 20;
    cout << "N = " << N << endl;
    // Fixed population size
    int pop_size = N * 100 * 128;
    // Selection size
    int sel_size = pop_size / 10;
    // Probability of randomly including weak offspring in selection
    float weak_prob = 0.3;
    // Mutation probability
    float prob = 0.3;
    // Maximum generations to iterate
    int gen_max = 100;
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

    // Initialize Population
    for (int i = 0; i < pop_size; i++)
    {
        int *board = initialize(N);
        population[i] = board;
        // print_board(board);
        // cout << fitness(board) << endl;
        int f_score = fitness(board, N);
        fitness_vector[i] = f_score;
        if (f_score > f_curr)
        {
            f_curr = f_score;
        }
        if (f_score == f_max)
        {
            cout << "Solution found:" << endl;
            print_board(board, N);
            break;
        }
    } // end for
    cout << "Generation " << gen << ": f_curr=" << f_curr << endl;
    gen++;

    while (gen < gen_max) // f_curr < f_max && 
    {
        // Selection
        population = selection(population, fitness_vector, sel_size, weak_prob, pop_size);
        // cout << population.size() << endl;
        /*
        cout << "index 0 board:" << endl;
        print_board(population[0]);
        cout << fitness(population[0]) << endl;
        cout << "index last board:" << endl;
        print_board(population[population.size()-1]);
        cout << fitness(population[population.size()-1]) << endl;
        */

        // Cross Over
        population = cross_over(population, pop_size, sel_size, N);
        //     //cout << population.size() << endl;

        // Mutation
        population = mutate(population, pop_size, N, prob);

        //     // Calculate Fitness
        std::fill(fitness_vector, fitness_vector + pop_size, 0);
        for (int i = 0; i < pop_size; i++)
        {
            int* board = population[i];
            int f_score = fitness(board, N);
            fitness_vector[i] = f_score;
            if (f_score > f_curr)
            {
                f_curr = f_score;
                // print_board(board);
            }
            // if (f_score == f_max)
            // {
            //     cout << "Solution found:" << endl;
            //     print_board(board, N);
            //     break;
            // }
        }

        //     // Increment generation
        cout << "Generation " << gen << ": f_curr=" << f_curr << endl;
        gen++;
    } // end while

    // Time elapsed
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time elapsed (seconds): " << elapsed_secs << endl;
    return 0;
} // end main()