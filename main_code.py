"""
main_code.py

This script defines functions for solving instances of the Resource-Constrained Project Scheduling Problem (RCPSP) using CPLEX.

Functions defined:
- read_instance: Reads RCPSP instance data for 30-activity problems (RG30 dataset).
- read_instance_300: Reads RCPSP instance data for 300-activity problems (RG300 dataset).
- forward_pass: Performs the forward pass in Critical Path Method (CPM) to calculate earliest start (ES) and finish times (EF).
- backward_pass: Performs the backward pass in CPM to calculate latest start (LS) and finish times (LF).
- calculate_early_and_late_start_times: Calculates ES, EF, LS, and LF for activities based on given instance data.
- print_activity_times: Displays ES, EF, LS, and LF along with activity durations.
- solve_with_cplex: Formulates and solves RCPSP instances using CPLEX.
- process_and_solve_instances: Processes and solves multiple RCPSP instances from a specified folder using a selected solver.
- process_and_solve_instances_save: Processes and solves multiple RCPSP instances from a specified folder using a selected solver and save the solution to a specific text file.
- save_all_solutions: Saves all solved solutions to a text file.


These functions are designed to be imported and used in other scripts, such as `solutions.py`, to process and solve RCPSP instances.

Usage:
- Import this script into another Python file where RCPSP instances need to be processed and solved.
- Utilize the functions provided based on specific requirements and configurations.


Date: June 13, 2024
"""


# import required libraries
import os
import cplex
from cplex import Cplex
from docplex.mp.model import Model
import time


# read RG30 data file
def read_instance(filename):
    """
    Reads a resource-constrained project scheduling problem (RCPSP) instance from a file.

    This function parses the RCPSP data from a specified file, extracting the number of activities,
    the number of resources, the capacity of each resource, activity durations, the number of successors
    for each activity, successor relationships, and resource requirements for each activity.

    Args:
        filename (str): The path to the file containing the RCPSP instance data.

    Returns:
        tuple: A tuple containing the following elements:
            - nb_activities (int): The number of activities in the project.
            - nb_resources (int): The number of resources available.
            - capacity (list of int): The capacity of each resource.
            - duration (list of int): The duration of each activity.
            - nb_successors (list of int): The number of successors for each activity.
            - successors (list of list of int): The list of successor activities for each activity.
            - horizon (int): The total time required to complete all activities.
            - resources (list of list of int): The resource requirements for each activity.

    References:
        Hexaly Documentation: https://www.hexaly.com/docs/last/exampletour/resource-constrained-project-scheduling-problem-rcpsp.html
    """
    # Read the file and extract the data
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[1].split()
    nb_activities = int(first_line[0])
    nb_resources = int(first_line[1])
    capacity = [int(lines[2].split()[r]) for r in range(nb_resources)]

    duration = [0 for i in range(nb_activities)]
    nb_successors = [0 for i in range(nb_activities)]
    successors = [[] for i in range(nb_activities)]
    resources = [[0 for r in range(nb_resources)] for i in range(nb_activities)]

    for i in range(nb_activities):
        line = lines[i + 4].split()
        duration[i] = int(line[0])
        for r in range(nb_resources):
            resources[i][r] = int(line[1 + r])
        nb_successors[i] = int(line[nb_resources + 1])
        successors[i] = [int(line[nb_resources + 2 + s]) - 1 for s in range(nb_successors[i])]

    horizon = sum(duration[i] for i in range(nb_activities))

    return nb_activities, nb_resources, capacity, duration, nb_successors, successors, horizon, resources

# read RG300 data
def read_instance_300(filename):
    """
    Reads a resource-constrained project scheduling problem (RCPSP) instance from a file.

    This function parses the RCPSP data from a specified file, extracting the number of activities,
    the number of resources, the capacity of each resource, activity durations, the number of successors
    for each activity, successor relationships, and resource requirements for each activity.

    Args:
        filename (str): The path to the file containing the RCPSP instance data.

    Returns:
        tuple: A tuple containing the following elements:
            - num_activities (int): The number of activities in the project.
            - num_resources (int): The number of resources available.
            - capacity (list of int): The capacity of each resource.
            - durations (list of int): The duration of each activity.
            - num_successors_list (list of int): The number of successors for each activity.
            - successors_list (list of list of int): The list of successor activities for each activity.
            - horizon (int): The total time required to complete all activities.
            - resources (list of list of int): The resource requirements for each activity.
    """
    # Read the file and extract the data
    with open(filename) as f:
        content = f.readlines()

    # Step 1: Read the header line
    header = content[0].strip().split()
    num_activities = int(header[0])
    num_resources = int(header[1])

    # Step 2: Read the resource availability
    capacity = list(map(int, content[1].strip().split()))

    # Step 3: Read activity details
    resources = []
    durations = []
    num_successors_list = []
    successors_list = []
    current_line = 2

    while current_line < len(content):
        parts = content[current_line].strip().split()
        if len(parts) < num_resources + 2:
            current_line += 1
            continue  # Skip improperly formatted lines

        duration = int(parts[0])
        resource_requirements = list(map(int, parts[1:num_resources + 1]))
        num_successors = int(parts[num_resources + 1])

        # Read successors from the current line and subsequent lines
        successors = parts[num_resources + 2:]
        current_line += 1

        while len(successors) < num_successors:
            if current_line < len(content):
                successors.extend(content[current_line].strip().split())
                current_line += 1
            else:
                break  # Exit the loop if current_line is out of range

        # Convert successors to integers and adjust indices to be 0-indexed
        successors = [int(s) - 1 for s in successors]

        # Store activity details
        resources.append(resource_requirements)
        durations.append(duration)
        num_successors_list.append(num_successors)
        successors_list.append(successors)


    # Calculate the horizon
    horizon = sum(durations[i] for i in range(num_activities))

    return num_activities, num_resources, capacity, durations, num_successors_list, successors_list, horizon, resources



# calculate ES values
def forward_pass(nb_activities, duration, successors):
    """
    Performs the forward pass in the Critical Path Method (CPM) to calculate
    the earliest start (ES) and earliest finish (EF) times for each task.

    Parameters:
    nb_activities (int): The number of activities in the project.
    duration (list): A list of activity durations.
    successors (list): A list of lists containing successors for each activity.

    Returns:
    tuple: A tuple containing:
        - ES (list): Earliest start times for each activity.
        - EF (list): Earliest finish times for each activity.
    """
    ES = [0] * nb_activities
    EF = [0] * nb_activities

    for i in range(nb_activities):
        EF[i] = ES[i] + duration[i]
        for succ in successors[i]:
            ES[succ] = max(ES[succ], EF[i])
            EF[succ] = ES[succ] + duration[succ]

    return ES, EF


# calculate LS values
def backward_pass(nb_activities, duration, successors, EF):
    """
    Performs the backward pass in the Critical Path Method (CPM) to calculate
    the latest start (LS) and latest finish (LF) times for each activity.

    Parameters:
    nb_activities (int): The number of activities in the project.
    duration (list): A list of activity durations.
    successors (list): A list of lists containing successors for each activity.
    EF (list): Earliest finish times for each activity.

    Returns:
    tuple: A tuple containing:
        - LS (list): Latest start times for each task.
        - LF (list): Latest finish times for each task.
    """
    horizon = max(EF)
    LS = [horizon] * nb_activities
    LF = [horizon] * nb_activities

    for i in reversed(range(nb_activities)):
        if not successors[i]:
            LS[i] = horizon - duration[i]
            LF[i] = horizon
        else:
            for succ in successors[i]:
                LS[i] = min(LS[i], LS[succ] - duration[i])
                LF[i] = LS[i] + duration[i]

    return LS, LF


def calculate_early_and_late_start_times(filename, read_instance_fn):
    """
    Calculates the earliest and latest start and finish times for activities
    in a project scheduling problem using the Critical Path Method (CPM).

    Parameters:
    filename (str): The name of the file containing the instance data.

    Returns:
    tuple: A tuple containing:
        - ES (list): Earliest start times for each activity.
        - EF (list): Earliest finish times for each activity.
        - LS (list): Latest start times for each activity.
        - LF (list): Latest finish times for each activity.
    """
    nb_activities, nb_resources, capacity, duration, nb_successors, successors, horizon, resources = read_instance_fn(filename)
    ES, EF = forward_pass(nb_activities, duration, successors)
    LS, LF = backward_pass(nb_activities, duration, successors, EF)
    return ES, EF, LS, LF


def print_activity_times(ES, EF, LS, LF, duration):
    """
    Prints the early start (ES), early finish (EF), late start (LS), and late finish (LF) times
    for each activity along with their durations.

    Parameters:
    ES (list): Earliest start times for each activity.
    EF (list): Earliest finish times for each activity.
    LS (list): Latest start times for each activity.
    LF (list): Latest finish times for each activity.
    duration (list): Durations of each activity.

    Returns:
    None
    """
    print("Activity\tDuration\tES\tEF\tLS\tLF")
    for i in range(len(duration)):
        print(f"{i}\t\t{duration[i]}\t\t{ES[i]}\t{EF[i]}\t{LS[i]}\t{LF[i]}")

# model
def solve_with_cplex(nb_activities, duration, successors, capacity, ES, LS, resources, nb_resources):
    """
    Solves the Resource-Constrained Project Scheduling Problem (RCPSP) using CPLEX.

    This function formulates the RCPSP as a Mixed-Integer Linear Programming (MILP) problem and solves it
    using the CPLEX solver. It handles the constraints related to task scheduling, precedence, and resource
    availability.

    Args:
        nb_activities (int): Number of activities in the project.
        duration (list of int): Duration of each activity.
        successors (list of list of int): List of successor activities for each activity.
        capacity (list of int): Capacity of each resource.
        ES (list of int): Earliest start time for each activity.
        LS (list of int): Latest start time for each activity.
        resources (list of list of int): Resource requirements for each activity.
        nb_resources (int): Number of resources available.

    Returns:
        list of tuple: List of tuples where each tuple contains the activity index and its start time.
    """
    max_LS = max(LS)
    H = list(range(max_LS + 1))

    mdl = Model('RCPSP')

    # Decision variables: x[i+1, t] is 1 if task (i+1) starts at time t, 0 otherwise
    x = {(i+1, t): mdl.binary_var(name=f'x_{i+1}_{t}') for i in range(nb_activities) for t in H}

    # Each task must start exactly once within its ES and LS window
    for i in range(nb_activities):
        mdl.add_constraint(mdl.sum(x[i+1, t] for t in range(ES[i], LS[i] + 1)) == 1)

    # Precedence constraints
    for i in range(nb_activities):
        for succ in successors[i]:
            mdl.add_constraint(
                mdl.sum(t * x[succ+1, t] for t in range(ES[succ], LS[succ] + 1)) >=
                mdl.sum(t * x[i+1, t] for t in range(ES[i], LS[i] + 1)) + duration[i]
            )

    # Resource constraints for each resource and each time slot
    nb_resources = len(capacity)
    for k in range(nb_resources):
        for s in H:
            mdl.add_constraint(
                mdl.sum(resources[i][k] * mdl.sum(
                    x[i, t] for t in range(max(ES[i], s - duration[i] + 1), min(LS[i], s) + 1)
                ) for i in range(nb_activities)) <= capacity[k]
            )

    # Objective: Minimize the total schedule time
    last_task_start_times = [x[nb_activities, t] * t for t in range(ES[nb_activities-1], LS[nb_activities-1] + 1)]
    objective = mdl.sum(last_task_start_times)
    mdl.minimize(objective)

    # Print model information
    mdl.print_information()

    # Print the objective function
    print("\nObjective Function:")
    print(objective)


    solution = mdl.solve(log_output=True)

    task_start_times = []
    if solution:
        print("Solution found")
        for i in range(nb_activities):
            for t in H:
                if x[i+1, t].solution_value > 0:
                    print(f'activity {i} starts at time {t}')
                    task_start_times.append((i, t))
        print(f"Objective value = {mdl.objective_value}")
    else:
        print("No solution found")

    return task_start_times




def process_and_solve_instances(folder_path, num_instances, solver_method, read_instance_fn):
    """
    Process and solve multiple instances of the Resource-Constrained Project Scheduling Problem (RCPSP).

    This function reads RCPSP instance files from a specified folder, processes them to extract relevant
    data, calculates early and late start times for each activity, and solves the instances using a given
    solver method. It stops after solving the specified number of instances.

    Args:
        folder_path (str): Path to the folder containing the RCPSP instance files.
        num_instances (int): Number of instances to solve. If 0, the function exits immediately.
        solver_method (function): Function to use for solving the RCPSP instance. 
                                  Expected to be either solve_with_cplex or solve_with_cplex_open.
        read_instance_fn (function): Function to read an RCPSP instance file and return the instance data.
                                     Expected to be read_instance or read_instance_300.

    Returns:
        list of tuple: A list of tuples where each tuple contains an activity index and its start time
                       for the last solved instance.
    """
    
    if num_instances == 0:
        print("Number of instances to solve is 0. Exiting function.")
        return
    
    # Counter for the number of instances solved
    solved_instances = 0
    
    all_solutions = []

    
    # Loop over all instance files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".rcp"):
            full_path = os.path.join(folder_path, filename)
            print(f"\nProcessing instance: {full_path}\n")

            # Read instance data
            nb_activities, nb_resources, capacity, duration, nb_successors, successors, horizon, resources = read_instance_fn(full_path)
            print(f" nb_activities: {nb_activities}\n nb_resources: {nb_resources}\n resources: {resources}\n capacity: {capacity}\n duration: {duration}\n nb_successors: {nb_successors}\n successors: {successors}\n horizon - Total duration: {horizon}")

            # Calculate ES, EF, LS, LF of the instance
            ES, EF, LS, LF = calculate_early_and_late_start_times(full_path, read_instance_fn)
            print_activity_times(ES, EF, LS, LF, duration)

            # Solve the model
            if solver_method == solve_with_cplex:
                activity_start_times = solve_with_cplex(nb_activities, duration, successors, capacity, ES, LS, resources, nb_resources)
            elif solver_method == solve_with_cplex_open:
                activity_start_times = solve_with_cplex_open(nb_activities, duration, successors, capacity, ES, LS, resources)
            else:
                raise ValueError("Unknown solver method")

            # save solution
            all_solutions.append((filename, activity_start_times))

            # Increment the counter for solved instances
            solved_instances += 1
            
            # Check if the desired number of instances has been solved
            if solved_instances == num_instances:
                break
    """
    # Print all solutions obtained
    for filename, solution in all_solutions:
        print(f"filename: {filename}\nSolution: {solution}\n")
    """


    return all_solutions  # Return all solutions for all instances processed


# save solutions to a text file
def save_all_solutions(all_solutions, output_filename):
    """
    Saves all solutions to a text file.

    Args:
        all_solutions (list of tuple): A list of tuples where each tuple contains the instance name
                                       and a list of tuples with activity index and start time.
        output_filename (str): The name of the output file to save the solutions.
    """
    with open(output_filename, 'w') as file:
        for instance_name, solutions in all_solutions:
            file.write(f"Instance: {instance_name}\n")
            file.write("Solutions (activity, start time):\n")
            for activity, start_time in solutions:
                file.write(f"  Activity {activity} starts at time {start_time}\n")
            file.write("\n")


def process_and_solve_instances_save(folder_path, num_instances, solver_method, read_instance_fn, output_filename):
    """
    Process and solve multiple instances of the Resource-Constrained Project Scheduling Problem (RCPSP).

    This function reads RCPSP instance files from a specified folder, processes them to extract relevant
    data, calculates early and late start times for each activity, and solves the instances using a given
    solver method. It stops after solving the specified number of instances.

    Args:
        folder_path (str): Path to the folder containing the RCPSP instance files.
        num_instances (int): Number of instances to solve. If 0, the function exits immediately.
        solver_method (function): Function to use for solving the RCPSP instance. 
                                  Expected to be either solve_with_cplex or solve_with_cplex_open.
        read_instance_fn (function): Function to read an RCPSP instance file and return the instance data.
                                     Expected to be read_instance or read_instance_300.
        output_filename (str): The name of the output file to save all solutions.

    Returns:
        list of tuple: A list of tuples where each tuple contains an activity index and its start time
                       for the last solved instance.
    """
    
    if num_instances == 0:
        print("Number of instances to solve is 0. Exiting function.")
        return
    
    # Counter for the number of instances solved
    solved_instances = 0
    
    all_solutions = []

    # Loop over all instance files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".rcp"):
            full_path = os.path.join(folder_path, filename)
            print(f"\nProcessing instance: {full_path}\n")

            # Read instance data
            nb_activities, nb_resources, capacity, duration, nb_successors, successors, horizon, resources = read_instance_fn(full_path)
            print(f" nb_activities: {nb_activities}\n nb_resources: {nb_resources}\n resources: {resources}\n capacity: {capacity}\n duration: {duration}\n nb_successors: {nb_successors}\n successors: {successors}\n horizon - Total duration: {horizon}")

            # Calculate ES, EF, LS, LF of the instance
            ES, EF, LS, LF = calculate_early_and_late_start_times(full_path, read_instance_fn)
            print_activity_times(ES, EF, LS, LF, duration)

            # Solve the model
            if solver_method == solve_with_cplex:
                activity_start_times = solve_with_cplex(nb_activities, duration, successors, capacity, ES, LS, resources, nb_resources)
            elif solver_method == solve_with_cplex_open:
                activity_start_times = solve_with_cplex_open(nb_activities, duration, successors, capacity, ES, LS, resources)
            else:
                raise ValueError("Unknown solver method")

            # Save solution
            all_solutions.append((filename, activity_start_times))

            # Increment the counter for solved instances
            solved_instances += 1
            
            # Check if the desired number of instances has been solved
            if solved_instances == num_instances:
                break

    # Save all solutions to the specified output file
    save_all_solutions(all_solutions, output_filename)

    return all_solutions  # Return all solutions for all instances processed








