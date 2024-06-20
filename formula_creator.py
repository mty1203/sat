import numpy as np
#from pysat.solvers import Glucose4
import os
import multiprocessing as mp
from multiprocessing import Pool
import functools
import re

def create_solvable_formulas(num_variables, num_clauses, type, amount, output_directory):

    mp.set_start_method("spawn")
    with Pool(8, maxtasksperchild=10) as p:
        worker_partial = functools.partial(worker, num_variables=num_variables, num_clauses=num_clauses, type=type, output_directory=output_directory)
        p.map(worker_partial, list(range(0, amount)))


def worker(index, num_variables, num_clauses, type, output_directory):

    formula = []

    if type == "random":
        formula = create_random_uniform_formula(num_variables, num_clauses)

        while not is_satisfiable(formula):
            formula = create_random_uniform_formula(num_variables, num_clauses)


    export_formula_to_dimacs(formula, os.path.join(output_directory, str(index) + ".dimacs"))




def is_satisfiable(formula):
    with Glucose4(bootstrap_with=formula, use_timer=True) as solver:

        if solver.solve():
            stats = solver.accum_stats()

            print("SAT")
            print('Solution:', solver.get_model())
            print("TIME: ", solver.time_accum())
            print("Stats: ", stats)

            return True

def create_random_uniform_formula(num_vars, num_clauses):
    formula = []
    while len(formula) < num_clauses:
        clause_vars = np.random.choice(range(1, num_vars+1), size=3, replace=False)
        signs = np.random.choice([-1, +1], size=3, replace=True)
        # convert numpy integers -> python 'int': python sat solvers cannot handle numpy integers
        formula.append([x.item() for x in clause_vars * signs])

    return formula







# ---------------------------------- Helpers ---------------------------------- #

def export_formula_to_dimacs(sat_formula, file_path):
    num_clauses = len(sat_formula)
    num_vars = len(set([abs(literal) for clause in sat_formula for literal in clause]))

    with open(file_path, "w") as sat_file:
        sat_file.write("p cnf {vars} {clauses} \n".format(vars=num_vars, clauses=num_clauses))
        for clause in sat_formula:
            for literal in clause:
                sat_file.write(str(literal) + " ")
            sat_file.write("0\n")

def load_formula_from_dimacs_file(file_path):
    formula = []
    with open(file_path, "r") as file:
        for line in file:
            # discard first dimacs line
            if not line.startswith("p"):
                # discard dimacs 0 at the end of each line ([:-1])
                formula.append(list(map(lambda x: int(x), line.split(" ")))[:-1])
    return formula

def load_file(file_path):
    formula = []
    number_of_vars = 0
    number_of_clauses = 0
    with open(file_path, "r") as file:
        for line in file:
            # read first line info
            numbers = re.findall(r'\d+', line)
            number_of_vars= int(numbers[0])
            number_of_clauses = int(numbers[1])
            if not line.startswith("p"):
                formula.append(list(map(lambda x: int(x), line.split(" ")))[:-1])
    return number_of_vars, number_of_clauses,formula

def evaluate_cnf(cnf_formula, assignments):
    """
    Evaluate a CNF formula based on the given variable assignments.

    :param cnf_formula: List of lists, where each inner list represents a clause.
                        Variables are represented by integers, where positive
                        integers indicate the variable itself and negative
                        integers indicate the negation of the variable.
    :param assignments: Dictionary of variable assignments where keys are variable
                        indices (positive integers) and values are Boolean (True or False).
    :return: True if the CNF formula is satisfied, False otherwise.
    """
    # Evaluate each clause in the formula
    for clause in cnf_formula:
        clause_satisfied = False
        for variable in clause:
            var_index = abs(variable)
            value = assignments[var_index]
            # Determine if the variable or its negation is True
            if (variable > 0 and value) or (variable < 0 and not value):
                clause_satisfied = True
                break
        if not clause_satisfied:
            return False  # If any clause is unsatisfied, the formula is False
    return True  # All clauses are satisfied

