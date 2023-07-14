run_array:
	g++ -o n_queen_genetic n_queen_genetic_original.cpp -std=c++11

compile_vector:
	g++ -o n_queen_genetic n_queen_genetic.cpp -std=c++11 -DP=20

compile_gpu_v2:
	nvcc -o n_queen_genetic n_queen_genetic_v2.cu -std=c++11

run_vector: compile_vector
	./n_queen_genetic

run_gpu:
	nvcc -o n_queen_genetic n_queen_genetic.cu -std=c++11 --extended-lambda

run_gpu_v2: compile_gpu_v2
	./n_queen_genetic