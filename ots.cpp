/**
 * PRL 2020 | Odd-even transposition sort
 * Author: Andrej Nano (xnanao00)
 */
#include <iostream>
#include <fstream>
#include <iomanip>
#include <mpi.h>

#define TAG 0

/**
 * Loads numbers from a specified file
 * @param file_name name of the input file
 * @param count number of expected values in the file
 */
int * loadNumbersFromFile(std::string file_name, int count) {

  int invariant = 0;
  int * numbers = new int [count];

  std::fstream file_input;
  file_input.open(file_name, std::ios::in);

  while(file_input.good()) {
    numbers[invariant] = file_input.get();
    if (!file_input.good()) { break; }
    std::cout << numbers[invariant] << " ";
    invariant++;
  }

  std::cout << std::endl;
  file_input.close();

  return numbers;
}

int main(int argc, char *argv[]) {

  // root process rank
  const int root_rank = 0;

  // number of processors/processes
  int number_of_processors;

  // this process rank/id
  int my_rank;

  // neighbour value
  int neightbour_value;

  // this process value (number assigned to this processor)
  int my_value;

  // unsorted list of numbers
  int* unsorted_numbers;

  // sorted list of numbers
  int* sorted_numbers = new int [number_of_processors];

  // time measurement
  double t1, t2, time_difference, max_time;

  // MPI Setup and initialization
  MPI_Status mpi_status;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &number_of_processors);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // Algorithm limits & ranges
  int even_limit_max = 2*(number_of_processors/2)-1;
  int odd_limit_max = 2*((number_of_processors - 1)/2);
  int half_steps_max = number_of_processors/2; //todo: should be ceiled

  // Processor w/ rank 0 loads all values
  // Sequentally dispatches values to all other processors and to itself as well
  if (my_rank == root_rank) {
    unsorted_numbers = loadNumbersFromFile("numbers", number_of_processors);
  }

  // Distribute values from the "unsorted_numbers" array to every processor, assigning 1 number per processor
  MPI_Scatter(unsorted_numbers, 1, MPI_INT, &my_value, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* START MEASURING TIME from this point */
  t1 = MPI_Wtime();

  /*** START OF THE SORTING ALGORITHM ***/
  for (int k = 1; k <= half_steps_max; k++) {

    /************************/
    /* 1. STEP  |  ODD PASS */
    /************************/

    // odd ranks dispatch their numbers to neighbouring even ranks (on the right side),
    // then receive either:
    //    1. smaller value from neighbour  OR
    //    2. the same number that was dispatched
    if ((my_rank % 2) && (my_rank < odd_limit_max)) {
      // --> DISPATCH my_value to adjecent process
      MPI_Send(&my_value, 1, MPI_INT, my_rank+1, TAG, MPI_COMM_WORLD);
      // <-- RECEIVE swapped or unchanged value and store in my_value
      MPI_Recv(&my_value, 1, MPI_INT, my_rank+1, TAG, MPI_COMM_WORLD, &mpi_status);
    }
    // neighbouring even ranks receive numbers from odd ranks (from the left side),
    // then comparison occurs, upon which numbers are either:
    //    1. swapped, odd neighbours (from the left side) receive values from even ranks
    //    2. not swapped, odd neighbours (from the left side) receive values they previously sent
    else if (my_rank <= odd_limit_max && my_rank != 0) {

      // <-- RECEIVE value to be compared against
      MPI_Recv(&neightbour_value, 1, MPI_INT, my_rank-1, TAG, MPI_COMM_WORLD, &mpi_status);

      // COMPARE
      if(neightbour_value > my_value) {
        // SWAP
        MPI_Send(&my_value, 1, MPI_INT, my_rank-1, TAG, MPI_COMM_WORLD);
        my_value = neightbour_value;
      } else {
        // DON'T SWAP, return the same number back
        MPI_Send(&neightbour_value, 1, MPI_INT, my_rank-1, TAG, MPI_COMM_WORLD);
      }
    }

    /*************************/
    /* 2. STEP  |  EVEN PASS */
    /*************************/

    // even ranks dispatch their numbers to neighbouring odd ranks (on the right side),
    // then receive either:
    //    1. smaller value from neighbour  OR
    //    2. the same number that was dispatched
    if ((!(my_rank % 2) || my_rank == 0) && (my_rank < even_limit_max)) {
      // --> DISPATCH my_value to adjecent process
      MPI_Send(&my_value, 1, MPI_INT, my_rank+1, TAG, MPI_COMM_WORLD);
      // <-- RECEIVE swapped or unchanged value and store in my_value
      MPI_Recv(&my_value, 1, MPI_INT, my_rank+1, TAG, MPI_COMM_WORLD, &mpi_status);
    }
    // neighbouring odd ranks receive numbers from even ranks (from the left side),
    // then comparison occurs, upon which numbers are either:
    //    1. swapped, even neighbours (from the left side) receive values from odd ranks
    //    2. not swapped, even neighbours (from the left side) receive values they previously sent
    else if (my_rank <= even_limit_max) {

      // <-- RECEIVE value to be compared against
      MPI_Recv(&neightbour_value, 1, MPI_INT, my_rank-1, TAG, MPI_COMM_WORLD, &mpi_status);

      // COMPARE
      if (neightbour_value > my_value) {
        // SWAP
        MPI_Send(&my_value, 1, MPI_INT, my_rank-1, TAG, MPI_COMM_WORLD);
        my_value = neightbour_value;
      } else {
        // DON'T SWAP, return the same number back
        MPI_Send(&neightbour_value, 1, MPI_INT, my_rank-1, TAG, MPI_COMM_WORLD);
      }

    }
  }
  /*** END OF THE SORTING ALGORITHM ***/

  /* STOP MEASURING TIME at this point */
  t2 = MPI_Wtime();

  // calculate the interval between the two time points
  time_difference = t2 - t1;

  // find the maximum processing time from all processes and store it in a 'max_time' variable of the root processor
  MPI_Reduce(&time_difference, &max_time, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD);

  // collect all numbers in the new sorted order and store in a corresponding 'sorted_numbers' array
  MPI_Gather(&my_value, 1, MPI_INT, sorted_numbers, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // print sorted numbers
  if(my_rank == root_rank) {
    for (int i = 0; i < number_of_processors; i++) {
      std::cout << sorted_numbers[i] << std::endl;
    }
  }

  /* ONLY FOR MEASUREMENT */
  if (my_rank == root_rank) {
    std::cout << "Execution time: " << max_time << "s" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
