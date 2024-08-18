"""
solutions.py

This script addresses the Resource-Constrained Project Scheduling Problem (RCPSP) using a Mixed-Integer Linear Programming (MILP) model implemented with CPLEX.
It processes instances from the RG30 and RG300 datasets and outputs the solutions obtained from CPLEX.

Functions and Methods:
- Imports from main_code.py:
  - read_instance
  - read_instance_300
  - process_and_solve_instances_save
  - solve_with_cplex

Usage:
- Define the paths to the folders containing the RG30 and RG300 instances.
- Specify the number of instances to be solved.
- Utilize the solve_with_cplex function as the solver method.
- Specify the text file where the solutions will be saved.
- Run the script to obtain solutions for the specified instances.

Example:
    folder_path = "RG30_301_320"
    folder_path_300 = "RG300_1_5"
    activity_start_times_30 = process_and_solve_instances(folder_path, num_instances=0, solver_method=solve_with_cplex, read_instance_fn=read_instance)
    activity_start_times_300 = process_and_solve_instances(folder_path_300, num_instances=1, solver_method=solve_with_cplex, read_instance_fn=read_instance_300)


Date: June 13, 2024
"""

import os
import time
from cplex import Cplex
from docplex.mp.model import Model


# Import functions from main_code.py
from main_code import (
    read_instance, # read RG30
    read_instance_300, # read RG300
    solve_with_cplex,
    process_and_solve_instances_save
)

def main():
    # ---------------------------------- Solve RG30 instance ----------------------------------
    
    try:
        # Define the path to the folder containing the RG30 instances
        folder_path = "RG30_301_320"

        # Specify the text file path where the RG30 solutions will be saved
        solutions_file_30 = "solutions\RG30_solutions_temp.txt"
        
        # Run the solver function to find solutions for the RG30 instances
        # Arguments:
        #   folder_path: The path to the folder containing instance files.
        #   num_instances: The number of instances to solve.
        #   solve_with_cplex: The solve_with_cplex function as the solver method.
        #   read_instance_fn: The function to read instance files.
        #   output_filename: The file to save the solutions.
        # Returns:
        #   activity_start_times: Start times for each activity in the solved instances
        
        print(f"Processing RG30 instances from folder: {folder_path}")
        activity_start_times_30 = process_and_solve_instances_save(folder_path, num_instances=1, solver_method=solve_with_cplex, read_instance_fn=read_instance, output_filename=solutions_file_30)
        print(f"\n\nRG30 activity start times: {activity_start_times_30}")
        
    except Exception as e:
        print(f"An error occurred while processing RG30 instances: {e}")
    
    # ---------------------------------- Solve RG300 instance ----------------------------------
    
    try:
        # Define the path to the folder containing the RG300 instances
        folder_path_300 = "RG300_1_5"

        # Specify the text file path where the RG300 solutions will be saved
        solutions_file_300 = "solutions\RG300_solutions_temp.txt"
        
        # Run the solver function to find solutions for the RG300 instances
        print(f"\nProcessing RG300 instances from folder: {folder_path_300}")
        activity_start_times_300 = process_and_solve_instances_save(folder_path_300, num_instances=0, solver_method=solve_with_cplex, read_instance_fn=read_instance_300, output_filename=solutions_file_300)
        print(f"\n\nRG300 activity start times: {activity_start_times_300}")
        
    except Exception as e:
        print(f"An error occurred while processing RG300 instances: {e}")

if __name__ == "__main__":
    main()
