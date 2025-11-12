import math
import shutil
import time
from cpmpy import intvar, boolvar, Model, all, sum
from sklearn.utils import class_weight
import numpy as np
import cpmpy
import re
from cpmpy.expressions.utils import all_pairs
from itertools import chain, combinations
import networkx as nx
from collections import defaultdict
import json
import matplotlib.pyplot as plt

import networkx as nx
import re


def build_instance_mapping(variables):
    
    import cpmpy as cp
    mapping = {}

    max_vm = 0
    max_pm = 0

    for var in variables:
        var_name = var.name
        if var_name.startswith('assign_VM_'):
            try:
                vm_num = int(var_name[len('assign_VM_'):])
                max_vm = max(max_vm, vm_num)
            except ValueError:
                continue
        elif var_name.startswith('active_PM_'):
            try:
                pm_num = int(var_name[len('active_PM_'):])
                max_pm = max(max_pm, pm_num)
            except ValueError:
                continue

    try:
        with open('output/vm_allocation/learned_global_constraints.txt', 'r') as f:
            for line in f:

                vm_matches = re.findall(r'assign_VM_(\d+)', line)
                for vm_num in vm_matches:
                    max_vm = max(max_vm, int(vm_num))

                pm_matches = re.findall(r'active_PM_(\d+)', line)
                for pm_num in pm_matches:
                    max_pm = max(max_pm, int(pm_num))
    except FileNotFoundError:
        pass  

    try:
        with open('output/vm_allocation/bias_constraints.txt', 'r') as f:
            for line in f:

                vm_matches = re.findall(r'assign_VM_(\d+)', line)
                for vm_num in vm_matches:
                    max_vm = max(max_vm, int(vm_num))

                pm_matches = re.findall(r'active_PM_(\d+)', line)
                for pm_num in pm_matches:
                    max_pm = max(max_pm, int(pm_num))
    except FileNotFoundError:
        pass  

    max_vm = max(max_vm, 12)  
    max_pm = max(max_pm, 8)   

    for i in range(1, max_vm + 1):
        var_name = f'assign_VM_{i}'
        if var_name not in mapping:
            var = cp.intvar(1, 5, name=var_name)  
            mapping[var_name] = var

    for i in range(1, max_pm + 1):
        var_name = f'active_PM_{i}'
        if var_name not in mapping:
            var = cp.intvar(0, 1, name=var_name)  
            mapping[var_name] = var

    for var in variables:
        var_name = var.name
        if var_name not in mapping:
            mapping[var_name] = var

    grid_dims = (max_vm, max_pm)
    return mapping, grid_dims

def parse_learned_constraint(constraint_str, instance_mapping=None):
    
    import cpmpy as cp
    from cpmpy.expressions.variables import _IntVarImpl
    from cpmpy.expressions.globalconstraints import AllDifferent

    if instance_mapping is None:
        instance_mapping = {}

    def get_var(var_name):
        

        if var_name in instance_mapping:
            return instance_mapping[var_name]

        if var_name.startswith('assign_VM_'):
            try:
                vm_num = int(var_name[len('assign_VM_'):])
                var = cp.intvar(1, 5, name=var_name)  
                instance_mapping[var_name] = var
                return var
            except ValueError as e:
                print(f"Error parsing VM number from {var_name}: {e}")
                return None
        elif var_name.startswith('active_PM_'):
            try:
                pm_num = int(var_name[len('active_PM_'):])
                var = cp.intvar(0, 1, name=var_name)  
                instance_mapping[var_name] = var
                return var
            except ValueError as e:
                print(f"Error parsing PM number from {var_name}: {e}")
                return None

        print(f"Warning: Creating default variable for unknown type: {var_name}")
        var = cp.intvar(0, 100, name=var_name)  
        instance_mapping[var_name] = var
        return var

    def extract_inside_sum(sum_str):
        
        start_idx = sum_str.find('Sum(') + 4
        paren_count = 1
        end_idx = start_idx
        
        while paren_count > 0 and end_idx < len(sum_str):
            if sum_str[end_idx] == '(':
                    paren_count += 1
            elif sum_str[end_idx] == ')':
                    paren_count -= 1
            end_idx += 1
            
        return sum_str[start_idx:end_idx-1].strip()

    def parse_sum_expression(sum_str):
        
        if not sum_str:
            return None

        terms = []
        current_term = ""
        paren_count = 0
        
        try:

            terms_str = sum_str.strip('sum([]).').strip()

            for char in terms_str:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char == ',' and paren_count == 0:
                    if current_term.strip():
                        terms.append(current_term.strip())
                    current_term = ""
                else:
                    current_term += char
            
            if current_term.strip():
                terms.append(current_term.strip())

            coefficients = []
            conditions = []
            
            for term in terms:
                term = term.strip('() ')
                if '*' in term:

                    coeff_part, cond_part = term.split('*', 1)

                    if coeff_part.startswith('[') and coeff_part.endswith(']'):
                        coefficients.extend([int(x.strip()) for x in coeff_part[1:-1].split(',')])
                    else:
                        coefficients.append(int(coeff_part.strip('() ')))

                    if cond_part.startswith('[') and cond_part.endswith(']'):
                        conds = cond_part[1:-1].split(',')
                        for cond in conds:
                            var_name, value = [x.strip() for x in cond.split('==')]
                            var = get_var(var_name)
                            if var is not None:
                                value = int(value)
                                conditions.append(var == value)
                    else:
                        var_name = cond_part.split('==')[0].strip()
                        value = int(cond_part.split('==')[1].strip())
                        var = get_var(var_name)
                        if var is not None:
                            conditions.append(var == value)
                else:

                    if '==' in term:
                        var_name, value = [x.strip() for x in term.split('==')]
                        var = get_var(var_name)
                        if var is not None:
                            value = int(value)
                            conditions.append(var == value)
                            coefficients.append(1)  

            if len(coefficients) == len(conditions):

                return cp.wsum(coefficients, conditions)
            print(f"Warning: Mismatch in coefficients ({len(coefficients)}) and conditions ({len(conditions)})")
            return None
            
        except Exception as e:
            print(f"Error parsing sum expression: {e}")
            return None

    def extract_comparison(expr_str):
        
        if '<=' in expr_str:
            parts = expr_str.split('<=')
            return '<=', int(parts[1].strip('() '))
        elif '>=' in expr_str:
            parts = expr_str.split('>=')
            return '>=', int(parts[1].strip('() '))
        elif '==' in expr_str:
            parts = expr_str.split('==')
            return '==', int(parts[1].strip('() '))
        elif '>' in expr_str:
            parts = expr_str.split('>')
            return '>', int(parts[1].strip('() '))
        return None, None

    if constraint_str.startswith('AllDifferent('):
        vars_str = constraint_str[len('AllDifferent('):-1]
        var_names = [name.strip() for name in vars_str.split(',')]
        vars_list = []
        for name in var_names:
            var = get_var(name)
            if var is not None:
                vars_list.append(var)
        if vars_list:
            return ('ALLDIFFERENT', AllDifferent(vars_list), vars_list)

    if constraint_str.startswith('Count('):
        count_expr = constraint_str[len('Count('):constraint_str.rindex(')')]
        conditions = [cond.strip('() ') for cond in count_expr.split(',')]
        vars_list = []
        count_conditions = []
        
        for cond in conditions:
            if '==' in cond:
                var_name, value = [x.strip() for x in cond.split('==')]
                var = get_var(var_name)
                if var is not None:
                    value = int(value)
                    vars_list.append(var)
                    count_conditions.append(var == value)
        
        if count_conditions:

            remaining = constraint_str[constraint_str.rindex(')')+1:].strip()
            op, bound = extract_comparison(remaining)
            
            if bound is not None:
                if op == '<=':
                    return ('COUNT', cp.sum(count_conditions) <= bound, vars_list)
                elif op == '>=':
                    return ('COUNT', cp.sum(count_conditions) >= bound, vars_list)
                elif op == '==':
                    return ('COUNT', cp.sum(count_conditions) == bound, vars_list)

    if constraint_str.startswith('active_PM_'):
        parts = constraint_str.split('==', 1)
        if len(parts) == 2:
            pm_var_name = parts[0].strip()
            rest = parts[1].strip('() ')
            pm_var = get_var(pm_var_name)
            
            if pm_var is not None and rest.startswith('(Sum('):
                sum_expr = parse_sum_expression(rest)
                if sum_expr is not None:

                    remaining = rest[rest.rindex(')')+1:].strip('() ')
                    op, bound = extract_comparison(remaining)
                    
                    if op == '>':
                        return ('ACTIVE_PM', pm_var == (sum_expr > bound), [pm_var])
                    elif op == '<':
                        return ('ACTIVE_PM', pm_var == (sum_expr < bound), [pm_var])

    if '=>' in constraint_str:
        antecedent_str, consequent_str = [x.strip() for x in constraint_str.split('=>')]

        antecedent_str = antecedent_str.strip('() ')
        antecedent_var = None
        if '==' in antecedent_str:
            var_name, value = [x.strip() for x in antecedent_str.split('==')]
            antecedent_var = get_var(var_name)
            if antecedent_var is not None:
                antecedent_value = int(value)
                antecedent = (antecedent_var == antecedent_value)
        else:
            antecedent_var = get_var(antecedent_str)
            if antecedent_var is not None:
                antecedent = antecedent_var

        if antecedent_var is not None and consequent_str.startswith('(Sum('):
            sum_expr = parse_sum_expression(consequent_str)
            if sum_expr is not None:
                op, bound = extract_comparison(consequent_str)
                
                if bound is not None:
                    if op == '>=':
                        consequent = (sum_expr >= bound)
                    elif op == '<=':
                        consequent = (sum_expr <= bound)
                    return ('IMPLICATION', cp.any([~antecedent, consequent]), [antecedent_var])

    if constraint_str.startswith('Sum('):
        try:
            sum_expr = parse_sum_expression(constraint_str)
            if sum_expr is not None:
                op, bound = extract_comparison(constraint_str)
                
                if bound is not None:
                    if op == '<=':
                        return ('SUM', sum_expr <= bound, [])
                    elif op == '>=':
                        return ('SUM', sum_expr >= bound, [])
                    elif op == '==':
                        return ('SUM', sum_expr == bound, [])
        except Exception as e:
            print(f"Error parsing sum constraint: {str(e)}")

    return None

def save_valid_constraints(global_constraints, invalid_constraints, output_file_path):
    
    with open(output_file_path, 'w') as f:
        for gc in global_constraints:

            key = tuple(gc[:2] + (tuple(var.name for var in gc[2]),))

            if key in invalid_constraints:
                continue

            constraint_type = gc[0]
            vars_in_scope = [var.name for var in gc[2]]

            if constraint_type == 'ALLDIFFERENT':

                line = f"ALLDIFFERENT\t{' '.join(vars_in_scope)}\n"
            elif constraint_type == 'SUM':


                if len(gc) >= 4:
                    total_sum = gc[3]
                    line = f"SUM\t{' '.join(vars_in_scope)}\t{total_sum}\n"
                else:
                    continue  
            elif constraint_type == 'COUNT':


                if len(gc) >= 5:
                    count_value = gc[3]
                    count_times = gc[4]
                    line = f"COUNT\t{' '.join(vars_in_scope)}\t{count_value}\t{count_times}\n"
                else:
                    continue  
            elif constraint_type == 'ALTERNATING':

                line = f"ALTERNATING\t{' '.join(vars_in_scope)}\n"
            elif constraint_type == 'MINDAYSBETWEEN':

                if len(gc) >= 4:
                    days = gc[3]
                    line = f"MINDAYSBETWEEN\t{' '.join(vars_in_scope)}\t{days}\n"
                else:
                    continue  
            else:
                continue  

            f.write(line)


def visualize_dependency_graph(dependency_graph, output_path='constraint_dependency_graph.png'):
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(dependency_graph, k=0.15, iterations=20)
    nx.draw(dependency_graph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_weight='bold')
    plt.title("Constraint Dependency Graph")
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_constraint_dependency_graph(global_constraints, variables):
    
    G = nx.Graph()

    for idx, constraint_info in enumerate(global_constraints):
        constraint_type = constraint_info[0]
        vars_in_scope = constraint_info[2]
        key = tuple(constraint_info[:2] + (tuple(var.name for var in vars_in_scope),))
        G.add_node(key, type=constraint_type)

    for (c1, c2) in combinations(global_constraints, 2):
        key1 = tuple(c1[:2] + (tuple(var.name for var in c1[2]),))
        key2 = tuple(c2[:2] + (tuple(var.name for var in c2[2]),))
        vars1 = set(var.name for var in c1[2])
        vars2 = set(var.name for var in c2[2])
        if vars1.intersection(vars2):
            G.add_edge(key1, key2)

    return G

def compute_centrality_measures(dependency_graph):
    
    degree_centrality = nx.degree_centrality(dependency_graph)
    betweenness_centrality = nx.betweenness_centrality(dependency_graph)
    closeness_centrality = nx.closeness_centrality(dependency_graph)

    combined_centrality = {}
    for node in dependency_graph.nodes():
        combined_centrality[node] = (
            degree_centrality.get(node, 0) * 0.5 +
            betweenness_centrality.get(node, 0) * 0.3 +
            closeness_centrality.get(node, 0) * 0.2
        )

    return combined_centrality

def append_missing_vars(model_file, vars_file, default_value=0, backup=False):
    

    model_vars = parse_model_file_vars(model_file)
    print(f"Variables in _model file: {sorted(model_vars)}")

    existing_vars, total_vars = parse_vars_file_existing(vars_file)
    print(f"Existing variables in _vars file: {sorted(existing_vars)}")
    print(f"Total variables before update: {total_vars}")

    missing_vars = model_vars - existing_vars
    if not missing_vars:

        return
    print(f"Missing variables to append: {sorted(missing_vars)}")

    if backup:
        backup_file = vars_file + ".bak"
        shutil.copy(vars_file, backup_file)
        print(f"Backup created at {backup_file}")

    with open(vars_file, 'a') as file:
        for var in sorted(missing_vars):
            file.write(f"{var} {default_value}\n")
            print(f"Appended variable {var} with default value {default_value}.")

    new_total_vars = total_vars + len(missing_vars)

    with open(vars_file, 'r') as file:
        lines = file.readlines()

    lines[0] = f"{new_total_vars}\n"

    with open(vars_file, 'w') as file:
        file.writelines(lines)
    print(f"Updated total variables in _vars file to: {new_total_vars}")


def parse_model_file_vars(model_file_path):
    
    variable_indices = set()
    with open(model_file_path, 'r') as file:
        for line in file:

            indices = re.findall(r'\[(.*?)\]', line)
            for group in indices:
                nums = re.findall(r'\d+', group)
                variable_indices.update(map(int, nums))
    return variable_indices


def parse_vars_file_existing(vars_file_path):
    
    existing_vars = set()
    with open(vars_file_path, 'r') as file:
        total_vars = int(file.readline().strip())
        for line in file:
            parts = line.strip().split()
            if parts:
                var_index = int(parts[0])
                existing_vars.add(var_index)
    return existing_vars, total_vars


def backup_file(original_path, backup_path):
    
    shutil.copy(original_path, backup_path)
    print(f"Backup created at {backup_path}")

def is_full_row(vars_in_scope):
    
    indices = [int(var.name[3:]) for var in vars_in_scope]
    n = 9  
    if len(indices) != n:
        return False
    rows = [idx // n for idx in indices]

    if len(set(rows)) != 1:
        return False
    cols = [idx % n for idx in indices]

    if set(cols) == set(range(n)):
        return True
    return False

def is_full_column(vars_in_scope):
    
    indices = [int(var.name[3:]) for var in vars_in_scope]
    n = 9  
    if len(indices) != n:
        return False
    cols = [idx % n for idx in indices]

    if len(set(cols)) != 1:
        return False
    rows = [idx // n for idx in indices]

    if set(rows) == set(range(n)):
        return True
    return False

import re

def parse_var_name_to_1d(var_name: str, cols: int = 9) -> int:
    

    match_2d = re.match(r"var\[(\d+),(\d+)\]", var_name)
    if match_2d:
        row = int(match_2d.group(1))
        col = int(match_2d.group(2))
        return row * cols + col

    match_1d = re.match(r"var(\d+)", var_name)
    if match_1d:
        return int(match_1d.group(1))

    match_game = re.match(r"(home|away)_(\d+)_(\d+)_(\d+)", var_name)
    if match_game:
        is_home = match_game.group(1) == "home"
        round_num = int(match_game.group(2))
        group_num = int(match_game.group(3))
        match_num = int(match_game.group(4))


        base_idx = 3000  
        matches_per_round = 8 * 2  
        round_offset = round_num * matches_per_round
        group_offset = group_num * 2
        match_idx = match_num
        total_idx = base_idx + round_offset + group_offset + match_idx
        return total_idx

    if var_name.startswith("group_"):

        return var_name

    match_day = re.match(r"match_day_(\d+)", var_name)
    if match_day:
        return int(match_day.group(1))

    vm_assign = re.match(r"assign_VM_(\d+)", var_name)
    if vm_assign:
        return int(vm_assign.group(1))

    pm_active = re.match(r"active_PM_(\d+)", var_name)
    if pm_active:
        return int(pm_active.group(1))

    resource = re.match(r"(cpu|memory|disk)_PM_(\d+)", var_name)
    if resource:
        pm_num = int(resource.group(2))
        resource_type = resource.group(1)

        resource_offset = {"cpu": 0, "memory": 1000, "disk": 2000}
        return resource_offset[resource_type] + pm_num


    return var_name

def constraint_to_vector(constraint_info, total_vars, cols=9):
    

    if hasattr(constraint_info, '_variables'):

        vars_in_scope = list(constraint_info._variables)
    elif isinstance(constraint_info, tuple) and len(constraint_info) >= 3:

        vars_in_scope = constraint_info[2]
    else:

        return [0] * total_vars

    vector = [0] * total_vars

    for var in vars_in_scope:
        try:

            var_name = var.name if hasattr(var, 'name') else str(var)

            if var_name.startswith('assign_'):

                vm_match = re.search(r'assign_VM_(\d+)', var_name)
                if vm_match:
                    idx = int(vm_match.group(1)) - 1
                else:

                    vm_num = var_name[len('assign_'):]
                    idx = int(vm_num) - 1
            elif var_name.startswith('active_'):

                pm_match = re.search(r'active_PM_(\d+)', var_name)
                if pm_match:
                    idx = int(pm_match.group(1)) - 1 + total_vars // 2  
                else:

                    pm_num = var_name[len('active_'):]
                    idx = int(pm_num) - 1 + total_vars // 2
            else:

                idx = parse_var_name_to_1d(var_name, cols=cols)
                if isinstance(idx, str):
                    print(f"Warning: Could not convert {var_name} to valid index")
                    continue
            
            if 0 <= idx < total_vars:
                vector[idx] = 1
            else:
                print(f"Warning: Index {idx} out of bounds for variable {var_name}")
        except Exception as e:
            print(f"Warning: Failed to process variable {var}: {e}")
            continue
    
    return vector

def is_full_block(vars_in_scope, n=9):
    


    indices = []
    for var in vars_in_scope:
        idx = parse_var_name_to_1d(var.name, n)
        indices.append(idx)

    if len(indices) != n:
        return False

    rows = [idx // n for idx in indices]
    cols = [idx % n  for idx in indices]


    block_rows = [r // 3 for r in rows]  
    block_cols = [c // 3 for c in cols]

    if len(set(block_rows)) != 1 or len(set(block_cols)) != 1:
        return False

    br = block_rows[0]  
    bc = block_cols[0]  

    expected_indices = [
        r * n + c
        for r in range(br * 3, br * 3 + 3)
        for c in range(bc * 3, bc * 3 + 3)
    ]

    return set(indices) == set(expected_indices)

def is_diagonal(vars_in_scope, n=9):
    
    indices = [int(var.name[3:]) for var in vars_in_scope]

    if len(indices) != n:
        return False
    main_diagonal = [i * n + i for i in range(n)]
    anti_diagonal = [i * n + (n - 1 - i) for i in range(n)]
    if set(indices) == set(main_diagonal) or set(indices) == set(anti_diagonal):
        return True
    return False

import re


def parse_dom_file(file_path):
    domain_constraints = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 3:
                var_index = int(parts[0])
                lower_bound = int(parts[2])
                upper_bound = int(parts[-1])
                domain_constraints[var_index] = (lower_bound, upper_bound)
    return domain_constraints

def parse_con_file(file_path):
    biases = []

    with open(file_path, 'r') as file:
        for line in file:
            con_type, var1, var2 = map(int, line.strip().split())
            biases.append((con_type, var1, var2))

    return biases


def constraint_type_to_string(con_type):
    return {
        0: "!=",
        1: "==",
        2: ">",
        3: "<",
        4: ">=",
        5: "<="
    }.get(con_type, "Unknown")


def parse_vars_file(file_path):
    with open(file_path, 'r') as file:
        total_vars = int(file.readline().strip())
        vars_values = [0] * total_vars

        for i, line in enumerate(file):
            value, _ = map(int, line.split())
            vars_values[i] = value

    return vars_values


def parse_model_file(file_path):
    max_index = -1
    constraints = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            constraint_type, indices_part = parts[0], parts[1]
            indices = re.findall(r'\d+', indices_part)
            indices = [int(i) for i in indices]

            max_index_in_line = max(indices)
            if max_index_in_line > max_index:
                max_index = max_index_in_line

            if constraint_type == 'ALLDIFFERENT':
                constraints.append((constraint_type, indices))

    return constraints, max_index


def are_comparisons_equal(comp1, comp2):
    
    if comp1.name != comp2.name:
        return False

    if comp1.args[0] != comp2.args[0]:
        return False

    if comp1.args[1] != comp2.args[1]:
        return False

    return True


def check_value(c):
    return bool(c.value())


def get_con_subset(B, Y):
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_scope(c)).issubset(Y)]


def get_kappa(B, Y):
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_scope(c)).issubset(Y) and c.value() is False]


def get_lambda(B, Y):
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_scope(c)).issubset(Y) and c.value() is True]


def gen_pairwise(v1, v2):
    return [v1 == v2, v1 != v2, v1 < v2, v1 > v2]

def gen_pairwise_ineq(v1, v2):
    return [v1 != v2]


def alldiff_binary(grid):
    for v1, v2 in all_pairs(grid):
        for c in gen_pairwise_ineq(v1, v2):
            yield c


def gen_scoped_cons(grid):

    for row in grid:
        for v1, v2 in all_pairs(row):
            for c in gen_pairwise_ineq(v1, v2):
                yield c

    for col in grid.T:
        for v1, v2 in all_pairs(col):
            for c in gen_pairwise_ineq(v1, v2):
                yield c


    for i1 in range(0, 4, 2):
        for i2 in range(i1, i1 + 2):
            for j1 in range(0, 4, 2):
                for j2 in range(j1, j1 + 2):
                    if (i1 != i2 or j1 != j2):
                        for c in gen_pairwise_ineq(grid[i1, j1], grid[i2, j2]):
                            yield c


def gen_all_cons(grid):

    for v1, v2 in all_pairs(grid.flat):
        for c in gen_pairwise(v1, v2):
            yield c


def construct_bias(X, gamma):
    all_cons = []

    X = list(X)

    for relation in gamma:

        if relation.count("var") == 2:

            for v1, v2 in all_pairs(X):
                constraint = relation.replace("var1", "v1")
                constraint = constraint.replace("var2", "v2")
                constraint = eval(constraint)

                all_cons.append(constraint)

        elif relation.count("var") == 4:

            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    for x in range(j + 1, len(X) - 1):
                        for y in range(x + 1, len(X)):
                            if (y != i and x != j and x != i and y != j):


                                constraint = relation.replace("var1", "X[i]")
                                constraint = relation.replace("var2", "X[j]")
                                constraint = relation.replace("var3", "X[x]")
                                constraint = relation.replace("var4", "X[y]")
                                constraint = eval(constraint)

                                all_cons.append(constraint)

    return all_cons


def construct_bias_for_var(X, gamma, v1):
    all_cons = []

    for relation in gamma:
        if relation.count("var") == 2:
            for v2 in X:
                if not (v1 is v2):
                    constraint = relation.replace("var1", "v1")
                    constraint = constraint.replace("var2", "v2")
                    constraint = eval(constraint)

                    all_cons.append(constraint)

        elif relation.count("var") == 4:
            X = X.copy()
            X.reverse()
            print(X)
            for j in range(0, len(X)):
                for x in range(j + 1, len(X) - 1):
                    for y in range(x + 1, len(X)):



                        constraint = relation.replace("var1", "v1")
                        constraint = relation.replace("var2", "X[j]")
                        constraint = relation.replace("var3", "X[x]")
                        constraint = relation.replace("var4", "X[y]")
                        constraint = eval(constraint)

                        all_cons.append(constraint)

    return all_cons


def get_scopes_vars(C):
    return set([x for scope in [get_scope(c) for c in C] for x in scope])


def get_scopes(C):
    return list(set([tuple(get_scope(c)) for c in C]))


def get_scope(constraint):

    if isinstance(constraint, cpmpy.expressions.variables._IntVarImpl):
        return [constraint]
    elif isinstance(constraint, cpmpy.expressions.core.Expression):
        all_variables = []
        for argument in constraint.args:
            if isinstance(argument, cpmpy.expressions.variables._IntVarImpl):

                all_variables.append(argument)
            else:
                all_variables.extend(get_scope(argument))
        return all_variables
    else:
        return []


def get_arity(constraint):
    return len(get_scope(constraint))


def get_min_arity(C):
    if len(C) > 0:
        return min([get_arity(c) for c in C])
    return 0


def get_max_arity(C):
    if len(C) > 0:
        return max([get_arity(c) for c in C])
    return 0


def get_relation(c, gamma):
    scope = get_scope(c)

    for i in range(len(gamma)):
        relation = gamma[i]

        if relation.count("var") != len(scope):
            continue

        constraint = relation.replace("var1", "scope[0]")
        for j in range(1, len(scope)):
            constraint = constraint.replace("var" + str(j + 1), "scope[" + str(j) + "]")

        constraint = eval(constraint)

        if hash(constraint) == hash(c):
            return i

    return -1




def get_divisors(n):
    divisors = list()
    for i in range(2, int(n / 2) + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


def join_con_net(C1, C2):
    C3 = [[c1 & c2 if c1 is not c2 else c1 for c2 in C2] for c1 in C1]
    C3 = list(chain.from_iterable(C3))
    C3 = remove_redundant_conj(C3)
    return C3


def remove_redundant_conj(C1):
    C2 = list()

    for c in C1:
        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        flag_eq = False
        flag_neq = False
        flag_geq = False
        flag_leq = False
        flag_ge = False
        flag_le = False

        for c1 in conj_args:
            print(c1.name)

            if c1.name == "==":
                flag_eq = True
            elif c1.name == "!=":
                flag_neq = True
            elif c1.name == "<=":
                flag_leq = True
            elif c1.name == ">=":
                flag_geq = True
            elif c1.name == "<":
                flag_le = True
            elif c1.name == ">":
                flag_ge = True
            else:
                raise Exception("constraint name is not recognised")

            if not ((flag_eq and (flag_neq or flag_le or flag_ge)) or (
                    (flag_leq or flag_le) and ((flag_geq or flag_ge)))):
                C2.append(c)
    return C2


def get_max_conjunction_size(C1):
    max_conj_size = 0

    for c in C1:
        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        max_conj_size = max(len(conj_args), max_conj_size)

    return max_conj_size


def get_delta_p(C1):
    max_conj_size = get_max_conjunction_size(C1)

    Delta_p = [[] for _ in range(max_conj_size)]

    for c in C1:

        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        Delta_p[len(conj_args) - 1].append(c)

    return Delta_p


def compute_sample_weights(Y):
    c_w = class_weight.compute_class_weight('balanced', classes=np.unique(Y), y=Y)
    sw = []

    for i in range(len(Y)):
        if Y[i] == False:
            sw.append(c_w[0])
        else:
            sw.append(c_w[1])

    return sw


class Metrics:

    def __init__(self):
        self.gen_queries_count = 0
        self.queries_count = 0
        self.top_lvl_queries = 0
        self.generated_queries = 0
        self.findscope_queries = 0
        self.findc_queries = 0

        self.average_size_queries = 0

        self.start_time_query = time.time()
        self.max_waiting_time = 0
        self.generation_time = 0

        self.converged = 1
        self.N_egativeQ = set()
        self.gen_no_answers = 0
        self.gen_yes_answers = 0

    def increase_gen_queries_count(self, amount=1):
        self.gen_queries_count += amount

    def increase_queries_count(self, amount=1):
        self.queries_count += amount

    def increase_top_queries(self, amount=1):
        self.top_lvl_queries += amount

    def increase_generated_queries(self, amount=1):
        self.generated_queries += amount

    def increase_findscope_queries(self, amount=1):
        self.findscope_queries += amount

    def increase_findc_queries(self, amount=1):
        self.findc_queries += amount

    def increase_generation_time(self, amount):
        self.generation_time += self.generation_time

    def increase_queries_size(self, amount):
        self.average_size_queries += 1

    def aggreagate_max_waiting_time(self, max2):
        if self.max_waiting_time < max2:
            self.max_waiting_time = max2

    def aggregate_convergence(self, converged2):
        if self.converged + converged2 < 2:
            self.converged = 0

    def __add__(self, other):

        new = self

        new.increase_queries_count(other.queries_count)
        new.increase_top_queries(other.top_lvl_queries)
        new.increase_generated_queries(other.generated_queries)
        new.increase_findscope_queries(other.findscope_queries)
        new.increase_findc_queries(other.findc_queries)
        new.increase_generation_time(other.generation_time)
        new.increase_queries_size(other.average_size_queries)

        new.aggreagate_max_waiting_time(other.max_waiting_time)
        new.aggregate_convergence(other.converged)

        return new


def find_suitable_vars_subset2(l, B, Y):
    if len(Y) <= get_min_arity(B) or len(B) < 1:
        return Y

    scope = get_scope(B[0])
    Y_prime = list(set(Y) - set(scope))

    l2 = int(l) - len(scope)

    if l2 > 0:
        Y1 = Y_prime[:l2]
    else:
        Y1 = []

    [Y1.append(y) for y in scope]

    return Y1


def calculate_modularity(G, communities):
    
    m = G.size(weight='weight')
    degrees = dict(G.degree(weight='weight'))
    Q = 0
    for community in communities:
        Lc = 0
        Dc = 0
        for u in community:
            Dc += degrees[u]
            for v in community:
                if G.has_edge(u, v):
                    Lc += G[u][v].get('weight', 1)
        Q += (Lc / (2 * m)) - (Dc / (2 * m)) ** 2
    return Q


def get_communities(partition):
    
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    return list(communities.values())


def optimize_modularity(G, max_iterations=1000, min_modularity_improvement=0.0001):
    partition = {node: i for i, node in enumerate(G.nodes())}
    best_modularity = -1
    best_partition = partition.copy()
    improvement = True
    iteration = 0

    while improvement and iteration < max_iterations:
        improvement = False
        current_modularity = calculate_modularity(G, get_communities(partition))

        for node in G.nodes():
            best_community = partition[node]
            best_increase = 0
            current_community = partition[node]

            partition[node] = -1

            for neighbor in G.neighbors(node):
                if partition[neighbor] != -1:
                    community = partition[neighbor]
                    partition[node] = community
                    new_modularity = calculate_modularity(G, get_communities(partition))
                    increase = new_modularity - current_modularity

                    if increase > best_increase:
                        best_increase = increase
                        best_community = community

                    partition[node] = -1

            partition[node] = best_community

            if best_increase > min_modularity_improvement:
                improvement = True

        new_modularity = calculate_modularity(G, get_communities(partition))

        if new_modularity > best_modularity:
            best_modularity = new_modularity
            best_partition = partition.copy()

        communities = get_communities(partition)
        new_G = nx.Graph()

        for i, community in enumerate(communities):
            new_node = i
            new_G.add_node(new_node)

            for node in community:
                for neighbor in G.neighbors(node):
                    if partition[neighbor] == partition[node]:
                        if new_G.has_edge(new_node, new_node):
                            new_G[new_node][new_node]['weight'] += G[node][neighbor].get('weight', 1)
                        else:
                            new_G.add_edge(new_node, new_node, weight=G[node][neighbor].get('weight', 1))
                    else:
                        neighbor_comm = partition[neighbor]

                        if new_G.has_edge(new_node, neighbor_comm):
                            new_G[new_node][neighbor_comm]['weight'] += G[node][neighbor].get('weight', 1)
                        else:
                            new_G.add_edge(new_node, neighbor_comm, weight=G[node][neighbor].get('weight', 1))

        G = new_G
        partition = {node: i for i, node in enumerate(G.nodes())}
        iteration += 1

    return get_communities(best_partition)

def transform_bias_constraints_pl_mapping(biases, instance_mapping, instance):
    
    import cpmpy as cp
    
    constraints = []

    max_vm = 0
    max_pm = 0

    for bias in biases:
        bias = bias.strip()
        vm_matches = re.findall(r'assign_VM_(\d+)', bias)
        for vm_num in vm_matches:
            max_vm = max(max_vm, int(vm_num))
        
        pm_matches = re.findall(r'active_PM_(\d+)', bias)
        for pm_num in pm_matches:
            max_pm = max(max_pm, int(pm_num))

    for i in range(1, max_vm + 1):
        var_name = f'assign_VM_{i}'
        if var_name not in instance_mapping:
            var = cp.intvar(1, 5, name=var_name)  
            instance_mapping[var_name] = var
    
    for i in range(1, max_pm + 1):
        var_name = f'active_PM_{i}'
        if var_name not in instance_mapping:
            var = cp.intvar(0, 1, name=var_name)  
            instance_mapping[var_name] = var

    for bias in biases:
        bias = bias.strip()
        if bias.startswith('Sum('):

            try:

                sum_expr = bias[4:bias.rindex(')')+1]
                comparison = bias[bias.rindex(')')+1:].strip()

                terms = []
                for term in sum_expr.strip('()').split(','):
                    term = term.strip()
                    if '*' in term:

                        coeff_part, var_part = term.split('*', 1)
                        coeff = int(coeff_part.strip('() '))
                        var_expr = var_part.strip('() ')
                        var_name = var_expr.split('==')[0].strip()
                        value = int(var_expr.split('==')[1].strip())
                        
                        if var_name in instance_mapping:
                            var = instance_mapping[var_name]
                            terms.append(coeff * (var == value))

                sum_constraint = cp.sum(terms)

                if '<=' in comparison:
                    bound = int(comparison.split('<=')[1].strip())
                    constraints.append(sum_constraint <= bound)
                elif '>=' in comparison:
                    bound = int(comparison.split('>=')[1].strip())
                    constraints.append(sum_constraint >= bound)
                elif '==' in comparison:
                    bound = int(comparison.split('==')[1].strip())
                    constraints.append(sum_constraint == bound)
            except Exception as e:
                print(f"Error parsing Sum constraint: {bias}")
                print(f"Error: {str(e)}")
                continue
        elif 'alldifferent' in bias.lower():

            try:
                var_list = bias[bias.find('(')+1:bias.find(')')].split(',')
                vars_to_diff = []
                for var_name in var_list:
                    var_name = var_name.strip()
                    if var_name in instance_mapping:
                        vars_to_diff.append(instance_mapping[var_name])
                if vars_to_diff:
                    constraints.append(cp.AllDifferent(vars_to_diff))
            except Exception as e:
                print(f"Error parsing AllDifferent constraint: {bias}")
                print(f"Error: {str(e)}")
                continue
        else:

            tokens = bias.split()
            if len(tokens) != 3:
                continue
            
            var1_name = tokens[0]
            operator = tokens[1]
            var2_name = tokens[2]

            if var1_name not in instance_mapping or var2_name not in instance_mapping:
                continue
            
            var1 = instance_mapping[var1_name]
            var2 = instance_mapping[var2_name]

            if operator == '==':
                constraints.append(var1 == var2)
            elif operator == '!=':
                constraints.append(var1 != var2)
            elif operator == '<':
                constraints.append(var1 < var2)
            elif operator == '>':
                constraints.append(var1 > var2)
            elif operator == '<=':
                constraints.append(var1 <= var2)
            elif operator == '>=':
                constraints.append(var1 >= var2)
    
    return constraints

def get_var_name(var):
    
    print(var.name)

    indices = re.findall(r"\[\d+[,\d+]*\]", var.name)
    if indices:

        name = var.name.replace(indices[0], '')
    else:

        name = var.name
    return name


def get_var_ndims(var):
    
    dims = re.findall(r"\[\d+[,\d+]*\]", var.name)
    if not dims:
        return 0  
    dims_str = "".join(dims)
    ndims = len(re.split(",", dims_str))
    return ndims


def get_var_dims(var):
    
    dims = re.findall(r"\[\d+[,\d+]*\]", var.name)
    if not dims:
        return []  
    dims_str = "".join(dims)
    dims = re.split(r"[\[\]]", dims_str)[1]
    dims = [int(dim) for dim in re.split(",", dims)]
    return dims

def extract_features(constraint):
    
    scope = get_scope(constraint)
    arity = len(scope)

    features = {
        'relation': constraint.name,
        'arity': arity,
        'has_constant': 0,
        'constant_value': 0
    }

    if hasattr(constraint, 'args'):
        for arg in constraint.args:
            if isinstance(arg, int):
                features['has_constant'] = 1
                features['constant_value'] = arg
                break

    var_names = [v.name for v in scope]
    base_names = [re.sub(r'\d+', '', name) for name in var_names]
    features['var_name_same'] = 1 if len(set(base_names)) == 1 else 0

    var_ndims = [get_var_ndims(v) for v in scope]
    features['var_ndims_same'] = 1 if len(set(var_ndims)) == 1 else 0
    features['var_ndims_max'] = max(var_ndims) if var_ndims else 0
    features['var_ndims_min'] = min(var_ndims) if var_ndims else 0

    if features['var_ndims_same'] and features['var_ndims_max'] > 0:
        for d in range(features['var_ndims_max']):
            dim_indices = []
            all_have_dim = True
            for v in scope:
                dims = get_var_dims(v)
                if len(dims) > d:
                    dim_indices.append(dims[d])
                else:
                    all_have_dim = False
                    break
            
            features[f'var_dim{d}_has'] = 1 if all_have_dim else 0
            if all_have_dim:
                features[f'var_dim{d}_same'] = 1 if len(set(dim_indices)) == 1 else 0
                features[f'var_dim{d}_max'] = np.max(dim_indices)
                features[f'var_dim{d}_min'] = np.min(dim_indices)
                features[f'var_dim{d}_avg'] = np.mean(dim_indices)
                features[f'var_dim{d}_spread'] = np.max(dim_indices) - np.min(dim_indices)
            else:
                features[f'var_dim{d}_same'] = 0
                features[f'var_dim{d}_max'] = 0
                features[f'var_dim{d}_min'] = 0
                features[f'var_dim{d}_avg'] = 0
                features[f'var_dim{d}_spread'] = 0

    feature_vector = [
        features['relation'], features['arity'], features['has_constant'], features['constant_value'],
        features['var_name_same'], features['var_ndims_same'], features['var_ndims_max'], features['var_ndims_min']
    ]
    for d in range(features.get('var_ndims_max', 0)):
        feature_vector.extend([
            features.get(f'var_dim{d}_has', 0), features.get(f'var_dim{d}_same', 0),
            features.get(f'var_dim{d}_max', 0), features.get(f'var_dim{d}_min', 0),
            features.get(f'var_dim{d}_avg', 0), features.get(f'var_dim{d}_spread', 0)
        ])



    feature_vector[0] = hash(features['relation']) % 1000  

    final_vector = []
    for val in feature_vector:
        final_vector.append(float(val))



    expected_feature_count = 30 
    while len(final_vector) < expected_feature_count:
        final_vector.append(0.0)

    return final_vector[:expected_feature_count]

from pycona import ConstraintOracle

def get_positive_examples(problem_class, n_examples):
    return problem_class.get_positive_examples(n_examples)

def get_oracles(problem_class):
    return ConstraintOracle(problem_class.C_T)