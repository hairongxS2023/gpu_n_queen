# gpu_n_queen

In this project, we realized the application of Genetic Algorithm (GA) to solve the N-Queens problem, which involves placing N chess queens on an NxN chessboard such that no two queens threaten each other. Brute force algorithms become time- consuming for larger values of N, and hence GA is used to find a valid solution. GA involves creating a population of possible solutions represented as strings of numbers, evaluating their fitness based on the number of threatening pairs of queens, and then breeding the best solutions through crossover to generate the next generation. This research suggests implementing a parallelized GA on NVIDIA GPU to enhance performance by assigning a portion of the population to distinct threads. In comparison to the sequential genetic algorithm, the proposed approach achieved speedups of 4.4 and 5.7 for 128 and 256 threads, respectively, within 50 generations and a search space with size of 512,000 candidate solutions.
