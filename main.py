# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 01:37:15 2024

@author: ludon
"""
import numpy as np
import matplotlib.pyplot as plt
import utils
import numpy as np
from models.SATNMBase import SATBase
from qubosearcher import *
from formula_creator import *
import os

class ChancellorSAT(SATBase):
    def __init__(self, formula):
        super().__init__(formula)
        self.auxiliary_vars = {}  # 存储辅助变量的字典
        self.variable_clause_map = {}  # 存储变量和子句之间的关系

    def create_qubo(self):
        for clause_index, clause in enumerate(self.formula):
            aux_var = self.num_variables + clause_index + 1  # 辅助变量的索引
            self.auxiliary_vars[clause_index] = aux_var  # 记录辅助变量
            self.variable_clause_map[aux_var] = clause  # 记录辅助变量和子句的关系

            if list(np.sign(clause)) == [1, 1, 1]:
                self.add(clause[0], clause[0], -2)
                self.add(clause[1], clause[1], -2)
                self.add(clause[2], clause[2], -2)
                self.add(aux_var, aux_var, -2)

                self.add(clause[0], clause[1], 1)
                self.add(clause[0], clause[2], 1)
                self.add(clause[0], aux_var, 1)

                self.add(clause[1], clause[2], 1)
                self.add(clause[1], aux_var, 1)

                self.add(clause[2], aux_var, 1)

            elif list(np.sign(clause)) == [1, 1, -1]:
                self.add(clause[0], clause[0], -1)
                self.add(clause[1], clause[1], -1)
                self.add(clause[2], clause[2], 0)
                self.add(aux_var, aux_var, -1)

                self.add(clause[0], clause[1], 1)
                self.add(clause[0], clause[2], 0)
                self.add(clause[0], aux_var, 1)

                self.add(clause[1], clause[2], 0)
                self.add(clause[1], aux_var, 1)

                self.add(clause[2], aux_var, 1)

            elif list(np.sign(clause)) == [1, -1, -1]:
                self.add(clause[0], clause[0], -1)
                self.add(clause[1], clause[1], -1)
                self.add(clause[2], clause[2], -1)
                self.add(aux_var, aux_var, -2)

                self.add(clause[0], clause[1], 0)
                self.add(clause[0], clause[2], 0)
                self.add(clause[0], aux_var, 1)

                self.add(clause[1], clause[2], 1)
                self.add(clause[1], aux_var, 1)

                self.add(clause[2], aux_var, 1)

            else:
                self.add(clause[0], clause[0], -1)
                self.add(clause[1], clause[1], -1)
                self.add(clause[2], clause[2], -1)
                self.add(aux_var, aux_var, -1)

                self.add(clause[0], clause[1], 1)
                self.add(clause[0], clause[2], 1)
                self.add(clause[0], aux_var, 1)

                self.add(clause[1], clause[2], 1)
                self.add(clause[1], aux_var, 1)

                self.add(clause[2], aux_var, 1)

    def get_auxiliary_vars(self):
        return self.auxiliary_vars

    def get_variable_clause_map(self):
        return self.variable_clause_map

def qubo_to_ising(Q):
    """ Convert QUBO matrix Q to Ising model J and h """
    n = Q.shape[0]
    J = np.zeros((n, n))
    h = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                J[i, j] = Q[i, j] / 4.0
        h[i] = (np.sum(Q[i, :]) + np.sum(Q[:, i]) - 2 * Q[i, i]) / 4.0
    return J, h

class nuesslein1:

    def __init__(self, formula, V):
        # sort the formula (i.e. all negative literals are at the back of the clause)
        self.formula = [sorted(c, reverse=True) for c in formula]
        self.L = []
        for i in range(V):
            self.L.append(i+1)
            self.L.append(-(i+1))
        self.V = V
        self.Q = {}

    # new values are added to the QUBO-Matrix Q via this monitor
    def add(self, x, y, value):
        if x > y:
            x,y = y,x
        if (x,y) in self.Q.keys():
            self.Q[(x,y)] += value
        else:
            self.Q[(x,y)] = value

    def R1(self, x):
        n = 0
        for c in self.formula:
            if x in c:
                n += 1
        return n

    def R2(self, x, y):
        n = 0
        for c in self.formula:
            if x in c and y in c:
                n += 1
        return n

    # this function creates the QUBO-Matrix Q
    def fillQ(self):
        for i in range(2*self.V + len(self.formula)):
            for j in range(2*self.V + len(self.formula)):
                if i > j:
                    continue
                if i == j and j < 2*self.V:
                    self.add(i, j, -self.R1(self.L[i]))
                elif i == j and j >= 2*self.V:
                    self.add(i, j, 2)
                elif j < 2*self.V and j-i == 1 and i%2 == 0:
                    self.add(i, j, len(self.formula)+1)
                elif i < 2*self.V and j < 2*self.V:
                    self.add(i, j, self.R2(self.L[i], self.L[j]))
                elif j >= 2*self.V and i < 2*self.V and self.L[i] in self.formula[j-2*self.V]:
                    self.add(i, j, -1)




def initialize_vector_pairwise(num_variables, variable_clause_map):
    """
    Initialize the state vector with num_variables variables and m auxiliary variables.
    The variables are initialized in pairs to ensure they are opposite (-1 and 1).
    The auxiliary variables are initialized based on the OR of their corresponding variables.
    
    Args:
    - num_variables: The number of original variables (n).
    - variable_clause_map: A dictionary mapping each auxiliary variable to a clause (list of variables).

    Returns:
    - variables: The initialized variables vector.
    - auxiliary_vars: The initialized auxiliary variables vector.
    """
    assert num_variables % 2 == 0, "The number of variables must be even for pairwise initialization."
    
    # Initialize the first num_variables elements in pairs to be opposite
    variables = np.zeros(num_variables, dtype=int)
    for i in range(0, num_variables, 2):
        value = np.random.choice([-1, 1])
        variables[i] = value
        variables[i + 1] = -value
    
    # Initialize the auxiliary variables based on the OR of the corresponding variables
    auxiliary_vars = []
    for aux_var, clause in variable_clause_map.items():
        clause_satisfied = -1  # Initialize as -1 (False)
        for var in clause:
            index = abs(var) - 1  # Get the index of the variable (convert 1-based to 0-based)
            if (var > 0 and variables[index] == 1) or (var < 0 and variables[index] == -1):
                clause_satisfied = 1  # If any variable in the clause is satisfied, set to 1 (True)
                break
        auxiliary_vars.append(clause_satisfied)
    
    return variables, np.array(auxiliary_vars)


def flip_and_update_vector(variables, auxiliary_vars, flip_indices, num_variables, variable_clause_map):
    """
    Flip the spins at flip_indices and update the auxiliary variables based on the variable-clause map.
    
    Args:
    - variables: The current variables vector.
    - auxiliary_vars: The current auxiliary variables vector.
    - flip_indices: The indices of the spins to be flipped.
    - num_variables: The number of original variables (n).
    - variable_clause_map: A dictionary mapping each auxiliary variable to a clause (list of variables).
    
    Returns:
    - variables: The updated variables vector.
    - auxiliary_vars: The updated auxiliary variables vector.
    """
    # Flip the spins at the specified indices
    for flip_index in flip_indices:
        variables[flip_index] *= -1
#         if flip_index % 2 == 0:
#             variables[flip_index + 1] = -variables[flip_index]  # Ensure the pair is opposite
#         else:
#             variables[flip_index - 1] = -variables[flip_index]  # Ensure the pair is opposite
    
    # Update the auxiliary variables if any variable is flipped
    if any(flip_index < num_variables for flip_index in flip_indices):
        for aux_var, clause in variable_clause_map.items():
            clause_satisfied = -1  # Initialize as -1 (False)
            for var in clause:
                index = abs(var) - 1  # Get the index of the variable (convert 1-based to 0-based)
                if (var > 0 and variables[index] == 1) or (var < 0 and variables[index] == -1):
                    clause_satisfied = 1  # If any variable in the clause is satisfied, set to 1 (True)
                    break
            aux_index = aux_var - num_variables - 1  # Get the index of the auxiliary variable in auxiliary_vars
            auxiliary_vars[aux_index] = clause_satisfied
    
    return variables, auxiliary_vars



def metropolis_algorithm(J, h, num_variables, variable_clause_map, max_iterations=400, initial_temperature=1.0, final_temperature=0.01):
    # Initialize the variables and auxiliary variables
    variables, auxiliary_vars = initialize_vector_pairwise(num_variables, variable_clause_map)
    assert check_pairwise_opposite(variables), "Initial variables do not satisfy pairwise opposite condition."

    # Store the best energy and state found
    best_energy = float('inf')
    best_variables = None
    best_auxiliary_vars = None

    # List to record energy changes
    energy_list = []

    # Temperature factor for perturbation
    temperature = initial_temperature

    # Temperature decay factor
    decay = (final_temperature / initial_temperature) ** (1 / max_iterations)

    def compute_energy(variables, auxiliary_vars):
        S = np.concatenate([variables, auxiliary_vars])
        return -0.5 * np.sum(J * np.outer(S, S)) - np.dot(h, S)

    index = 0
    # Perform the iterative process
    for iteration in range(max_iterations):
        # Calculate the energy of the current state
        energy = compute_energy(variables, auxiliary_vars)

        # Record the energy
        energy_list.append(energy)

        # Update the best energy and state if the current one is better
        if energy < best_energy:
            best_energy = energy
            best_variables = variables.copy()
            best_auxiliary_vars = auxiliary_vars.copy()

        # Update the state vector S with a chance of random perturbation
        for i in range(0, num_variables, 2):
            # Copy variables and auxiliary_vars for temporary flip
            variables_flipped = variables.copy()
            auxiliary_vars_flipped = auxiliary_vars.copy()
            # Flip the spins in pairs and update auxiliary variables
            variables_flipped, auxiliary_vars_flipped = flip_and_update_vector(variables_flipped, auxiliary_vars_flipped, [i, i + 1], num_variables, variable_clause_map)
            energy_flipped = compute_energy(variables_flipped, auxiliary_vars_flipped)
            
            # Calculate the energy difference
            delta_energy = energy_flipped - energy

            # Decide whether to accept the new state
<<<<<<< HEAD
            if delta_energy <0 or np.exp(-delta_energy / temperature) > 50* np.random.rand():
=======
            if delta_energy < 0: or np.exp(-delta_energy / temperature) > 50* np.random.rand():
>>>>>>> 4d0247ce08b6fe9101a38e5d7d510356665587c6
                # Accept the flip
                variables = variables_flipped
                auxiliary_vars = auxiliary_vars_flipped
                variable_states = best_variables == 1
                assignments = {i // 2 + 1: variable_states[i] for i in range(0,num_variables, 2)}
                result = evaluate_cnf(formula, assignments)
                print(result)
                energy = energy_flipped
                if energy < best_energy:
                    best_energy = energy
                    best_variables = variables.copy()
                    best_auxiliary_vars = auxiliary_vars.copy()

        # Decrease the temperature (reduce the perturbation over time)
        temperature *= decay

        # Check pairwise opposite condition
        assert check_pairwise_opposite(variables), f"Iteration {iteration}: variables do not satisfy pairwise opposite condition."

    print('Max iterations:', iteration)

    # Convert best_variables to True/False for variables
    variable_states = best_variables == 1
    print('Variable states (True for 1, False for -1):', variable_states)

    # Create a dictionary for variable assignments
    #assignments = {i + 1: variable_states[i] for i in range(num_variables)}
    assignments = {i // 2 + 1: variable_states[i] for i in range(0,num_variables, 2)}
    # Plot the energy changes
    plt.plot(energy_list)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Energy changes over iterations')
    plt.show()

    return best_variables, best_auxiliary_vars, best_energy, assignments

def check_pairwise_opposite(variables):
    """
    Check if the variables vector satisfies the condition that every pair of variables are opposite.
    
    Args:
    - variables: The variables vector to be checked.
    
    Returns:
    - True if every pair of variables are opposite, False otherwise.
    """
    num_variables = len(variables)
    assert num_variables % 2 == 0, "The number of variables must be even for pairwise checking."
    
    for i in range(0, num_variables, 2):
        if variables[i] != -variables[i + 1]:
            return False
    return True

if __name__ == '__main__':
    # Example QUBO matrix from Chancellor transformation
    V = 16
    num_variables =2*V
    formula = load_formula_from_dimacs_file(os.path.join(os.getcwd(), "formulas", "sat_k3_v16_c48_s431343124.cnf"))
    nuesslein_sat = ChancellorSAT(formula)
    nuesslein_sat.create_qubo()
    qubo_dict = nuesslein_sat.qubo
    max_index = max(max(pair) for pair in qubo_dict.keys())
    size = max_index + V+1
    variable_clause_map = nuesslein_sat.get_variable_clause_map()
    qubo_matrix = np.zeros((size, size))
    a = nuesslein1(formula, V)
    a.fillQ()
    qubo_dict = a.Q
    for (i, j), value in qubo_dict.items():
        qubo_matrix[i, j] = value
        if i != j:  # Assume symmetry for the QUBO matrix
            qubo_matrix[j, i] = value
    qubo_matrix = np.random.randint(-2, 3, size=(num_variables + 48, num_variables + 48))
    qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2  # 确保矩阵对称
    np.fill_diagonal(qubo_matrix, np.random.randint(-1, 2, size=num_variables + 48))
    J, h = qubo_to_ising(qubo_matrix)
    
    # Run Metropolis Monte Carlo algorithm to solve the Ising problem
    best_variables, best_auxiliary_vars, best_energy, assignments = metropolis_algorithm(J, h, num_variables, variable_clause_map)
    print(assignments)
    result = evaluate_cnf(formula, assignments)
    print('Evaluation result:', result)
    print('Best state:', best_variables)
    print('Best energy:', best_energy)
