import argparse
import os
import pickle
import time
import sys
from cpmpy import *
from cpmpy import cpm_array
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.globalconstraints import AllDifferent
from pycona import ProblemInstance
from pycona.utils import get_kappa
from bayesian_quacq import BayesianQuAcq
from bayesian_ca_env import BayesianActiveCAEnv
from enhanced_bayesian_pqgen import EnhancedBayesianPQGen
from benchmarks_global import construct_sudoku, construct_jsudoku, construct_latin_square
from benchmarks_global import construct_graph_coloring_register, construct_graph_coloring_scheduling
from benchmarks_global import construct_sudoku_greater_than
from benchmarks_global import construct_examtt_simple as ces_global
from benchmarks_global import construct_examtt_variant1, construct_examtt_variant2
from benchmarks_global import construct_nurse_rostering as nr_global


def load_phase1_data(pickle_path):
    
    print(f"\nLoading Phase 1 data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded Phase 1 data:")
    print(f"  CG (global constraints): {len(data['CG'])}")
    print(f"  B_fixed (pruned bias): {len(data['B_fixed'])}")
    print(f"  E+ (positive examples): {len(data['E+'])}")
    print(f"  Initial probabilities: {len(data.get('initial_probabilities', {}))}")

    return data


def display_sudoku_grid(variables, title="Sudoku Grid", debug=False):
    
    print(f"\n{title}")
    print("  " + "-" * 37)

    grid = [[None for _ in range(9)] for _ in range(9)]

    if debug and len(variables) > 0:
        print(f"\n  DEBUG: Checking first 3 variables...")
        for i, var in enumerate(variables[:3]):
            print(f"    Var {i}: name={var.name if hasattr(var, 'name') else 'NO NAME'}, "
                  f"value={var.value() if hasattr(var, 'value') else 'NO VALUE METHOD'}, "
                  f"type={type(var)}")

    for var in variables:
        if hasattr(var, 'name') and 'grid[' in str(var.name):

            try:
                var_name = str(var.name)
                parts = var_name.split('[')[1].split(']')[0].split(',')
                row = int(parts[0])
                col = int(parts[1])

                if callable(getattr(var, 'value', None)):
                    val = var.value()
                elif hasattr(var, '_value'):
                    val = var._value
                else:
                    val = None
                
                if val is not None:
                    grid[row][col] = val
            except Exception as e:
                if debug:
                    print(f"  DEBUG: Error parsing {var.name}: {e}")
                continue

    for i in range(9):
        if i > 0 and i % 3 == 0:
            print("  " + "-" * 37)
        
        row_str = "  |"
        for j in range(9):
            if j > 0 and j % 3 == 0:
                row_str += " |"
            val = grid[i][j]
            if val is None:
                row_str += " . "
            else:
                row_str += f" {val} "
        row_str += "|"
        print(row_str)
    
    print("  " + "-" * 37)

    filled = sum(1 for row in grid for cell in row if cell is not None)
    print(f"  Filled cells: {filled}/81")


def extract_alldifferent_constraints(oracle):
    
    alldiff_constraints = []
    for c in oracle.constraints:
        if isinstance(c, AllDifferent) or "alldifferent" in str(c).lower():
            alldiff_constraints.append(c)
    return alldiff_constraints


def initialize_probabilities(constraints, prior=0.5):
    
    probabilities = {}
    for c in constraints:
        probabilities[c] = prior
    return probabilities


def update_supporting_evidence(P_c, alpha):
    
    return P_c + (1 - P_c) * (1 - alpha)


def generate_violation_query(CG, C_validated, probabilities, all_variables):
    
    import cpmpy as cp
    import time
    
    print(f"  Building COP model: {len(CG)} candidates, {len(C_validated)} validated, {len(all_variables)} variables")

    model = cp.Model()

    for c in C_validated:
        model += c

    gamma = {str(c): cp.boolvar(name=f"gamma_{i}") for i, c in enumerate(CG)}

    for c in CG:
        c_str = str(c)
        model += (gamma[c_str] == ~c)
    
    gamma_list = list(gamma.values())
    model += (cp.sum(gamma_list) >= 1)  



    violation_count = cp.sum(gamma_list)
    weighted_preference = cp.sum([
        (1.0 - probabilities[c]) * gamma[str(c)]
        for c in CG
    ])
    
    epsilon = 0.01
    objective = violation_count - epsilon * weighted_preference




    model.minimize(objective)

    print(f"  Solving COP...")
    solve_start = time.time()

    result = model.solve(time_limit=30)
    solve_time = time.time() - solve_start
    if not result:
        print("UNSAT")
    else:
        violated = []
        for i, c in enumerate(CG):
            gi = gamma[str(c)].value()
            if gi is None:
                print(f"gamma_{i} has no value (solver didn’t assign).")
            elif gi:  
                violated.append((i, c))
        print(f"Violated {len(violated)}/{len(CG)} constraints:")
        for i, c in violated:
            print(f" - gamma_{i} -> VIOLATED: {c}")

    
    if result:
        print(f"  Solved in {solve_time:.2f}s - found violation query")
        Y = get_variables(model.constraints)

        values_set = sum(1 for v in Y if v.value() is not None)
        print(f"  Variables with values: {values_set}/{len(Y)}")

        Viol_e = get_kappa(CG, Y)
        print(f"  Violating {len(Viol_e)}/{len(CG)} constraints")

        
        return Y, Viol_e, "SAT"
    else:
        print(f"  UNSAT after {solve_time:.2f}s - cannot find violation query")
        return None, [], "UNSAT"



def disambiguate_violated_constraints(Viol_e, C_validated, CG, oracle, probabilities, all_variables, 
                                      alpha, theta_max, theta_min, max_queries_per_constraint=10):
    
    updated_probs = probabilities.copy()
    to_remove = set()
    total_disambiguation_queries = 0
    
    for c_target in Viol_e:
        print(f"\n  Disambiguating constraint: {c_target}")
        print(f"  Current P(c) = {probabilities[c_target]:.3f}")


        init_cl = list(C_validated)

        remaining_cg = [c for c in CG if c not in Viol_e]
        init_cl.extend(remaining_cg)
        print(f"  Init CL: {len(C_validated)} validated + {len(remaining_cg)} remaining CG candidates")

        all_vars = get_variables([c_target] + init_cl)

        instance = ProblemInstance(
            variables=cpm_array(all_vars),
            init_cl=init_cl,
            bias=[c_target],  
            name="isolation_learning"
        )

        env = BayesianActiveCAEnv(
            qgen=EnhancedBayesianPQGen(),
            theta_max=theta_max,
            theta_min=theta_min,
            prior=probabilities[c_target],  
            alpha=alpha
        )

        env.constraint_probs = {c_target: probabilities[c_target]}
        env.max_queries = max_queries_per_constraint

        ca_system = BayesianQuAcq(ca_env=env)
        learned_instance = ca_system.learn(instance, oracle=oracle, verbose=2)

        if hasattr(env, 'metrics') and env.metrics is not None:
            queries_used_for_this_constraint = env.metrics.membership_queries_count
        else:
            queries_used_for_this_constraint = 1  
        
        total_disambiguation_queries += queries_used_for_this_constraint
        print(f"  [Queries for this constraint: {queries_used_for_this_constraint}]")

        if c_target in learned_instance.cl:


            updated_probs[c_target] = env.constraint_probs.get(c_target, probabilities[c_target])
            print(f"  Result: Kept (P={updated_probs[c_target]:.3f})")
        
        elif c_target not in learned_instance.bias:

            updated_probs[c_target] = env.constraint_probs.get(c_target, probabilities[c_target] * alpha)
            print(f"  Result: Rejected (P={updated_probs[c_target]:.3f})")
            
            if updated_probs[c_target] <= theta_min:
                to_remove.add(c_target)
        else:

            updated_probs[c_target] = env.constraint_probs.get(c_target, probabilities[c_target])
            print(f"  Result: Uncertain (P={updated_probs[c_target]:.3f})")
    
    print(f"\n[DISAMBIGUATION] Total queries used: {total_disambiguation_queries}")
    return updated_probs, to_remove, total_disambiguation_queries


def cop_based_refinement(experiment_name, oracle, candidate_constraints, initial_probabilities,
                         variables, alpha=0.42, theta_max=0.9, theta_min=0.1, 
                         max_queries=500, timeout=600):
    
    start_time = time.time()
    queries_used = 0


    CG = set(candidate_constraints) if not isinstance(candidate_constraints, set) else candidate_constraints.copy()
    probabilities = initial_probabilities.copy()
    C_validated = []
    
    print(f"\n{'='*60}")
    print(f"COP-Based Refinement for {experiment_name}")
    print(f"{'='*60}")
    print(f"Initial candidate constraints: {len(CG)}")
    print(f"Parameters: alpha={alpha}, theta_max={theta_max}, theta_min={theta_min}")
    print(f"Budget: {max_queries} queries, {timeout}s timeout\n")
    
    iteration = 0
    consecutive_unsat = 0  
    
    while True:

        iteration += 1
        print(f"\n{'-'*60}")
        print(f"Iteration {iteration}")
        print(f"{'-'*60}")

        if queries_used >= max_queries:
            print(f" Reached maximum query budget ({max_queries})")
            break
        
        if time.time() - start_time > timeout:
            print(f" Timeout ({timeout}s) reached")
            break
        
        if not CG:
            print(f" No more candidate constraints")
            break
        
        if len(CG) > 0 and min(probabilities[c] for c in CG) > theta_max:
            print(f" All remaining constraints have P(c) > {theta_max}")

            for c in CG:
                C_validated.append(c)
                print(f"  Accepted: {c} (P={probabilities[c]:.3f})")
            CG = set()  
            break
        
        print(f"Status: {len(C_validated)} validated, {len(CG)} candidates, {queries_used} queries used")

        print(f"\n[QUERY] Generating violation query...")
        Y, Viol_e, status = generate_violation_query(CG, C_validated, probabilities, variables)
        
        if status == "UNSAT":
            consecutive_unsat += 1
            print(f"[UNSAT] Cannot generate violation query for remaining {len(CG)} constraints")
            print(f"[DECISION] Accepting remaining constraints as likely correct or implied")

            for c in list(CG):
                if probabilities[c] >= 0.7:  
                    C_validated.append(c)
                    print(f"  [ACCEPT] {c} (P={probabilities[c]:.3f})")
                else:
                    print(f"  [UNCERTAIN] {c} (P={probabilities[c]:.3f}) - keeping in candidates")
            
            CG = {c for c in CG if probabilities[c] < 0.7}  
            
            if not CG:
                break

            if consecutive_unsat >= 2:
                print(f" Multiple consecutive UNSAT results - accepting/rejecting remaining constraints")
                for c in list(CG):
                    if probabilities[c] >= 0.5:  
                        C_validated.append(c)
                        print(f"  [FINAL ACCEPT] {c} (P={probabilities[c]:.3f})")
                    else:
                        print(f"  [FINAL REJECT] {c} (P={probabilities[c]:.3f}) - too uncertain")
                break
            else:
                print(f"[CONTINUE] {len(CG)} uncertain constraints remaining (UNSAT count: {consecutive_unsat})")

                continue  

        consecutive_unsat = 0
        
        print(f"Generated query violating {len(Viol_e)} constraints")
        for c in Viol_e:
            print(f"  - {c} (P={probabilities[c]:.3f})")

        if 'sudoku' in experiment_name.lower() and len(variables) == 81:
            try:
                display_sudoku_grid(Y, title="Violation Query Assignment", debug=False)
            except Exception as e:
                print(f"Error displaying Sudoku grid: {e}")
                print(Y)

        print(f"\n[ORACLE] Asking oracle...")
        answer = oracle.answer_membership_query(Y)
        queries_used += 1
        
        if answer == False:  
            print(f"Oracle: Yes (valid assignment)")
            print(f"{len(Viol_e)} constraints violated by valid solution - entering disambiguation")


            probabilities, to_remove, disambiguation_queries = disambiguate_violated_constraints(
                Viol_e, C_validated, CG, oracle, probabilities, variables, 
                alpha, theta_max, theta_min,
                max_queries_per_constraint=10  
            )

            queries_used += disambiguation_queries
            print(f"[QUERIES] Main loop: {queries_used - disambiguation_queries}, Disambiguation: {disambiguation_queries}, Total so far: {queries_used}")

            CG = set(CG)
            for c in to_remove:
                if c in CG:
                    CG.remove(c)
                    print(f"  [REMOVE] Removed: {c} (P={probabilities[c]:.3f})")


            for c in Viol_e:
                if c not in to_remove and c in CG and probabilities[c] >= theta_max:
                    C_validated.append(c)
                    CG.remove(c)
                    print(f"  [ACCEPT] Accepted: {c} (P={probabilities[c]:.3f} >= {theta_max})")
        
        else:  
            print(f"[Oracle: No (invalid assignment)")
            print(f"Supporting {len(Viol_e)} constraints")

            for c in Viol_e:
                old_prob = probabilities[c]
                probabilities[c] = update_supporting_evidence(probabilities[c], alpha)
                print(f"  [UPDATE] {c}: P={old_prob:.3f} -> {probabilities[c]:.3f}")

                if probabilities[c] >= theta_max:
                    C_validated.append(c)
                    CG.remove(c)
                    print(f"    [ACCEPT] Accepted (P >= {theta_max})")
    
    end_time = time.time()
    total_duration = end_time - start_time

    print(f"\n{'='*60}")
    print(f"Refinement Complete")
    print(f"{'='*60}")
    print(f"Validated constraints: {len(C_validated)}")
    print(f"Rejected constraints: {len(candidate_constraints) - len(C_validated)}")
    print(f"Total queries: {queries_used}")
    print(f"Total time: {total_duration:.2f}s")
    print(f"\nValidated constraints:")
    for c in C_validated:
        print(f"  [OK] {c}")
    
    stats = {
        'queries': queries_used,
        'time': total_duration,
        'validated': len(C_validated),
        'rejected': len(candidate_constraints) - len(C_validated)
    }
    
    return C_validated, stats


def construct_instance(experiment_name):
    
    if 'graph_coloring_register' in experiment_name.lower() or experiment_name.lower() == 'register':
        result = construct_graph_coloring_register()

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'graph_coloring_scheduling' in experiment_name.lower() or experiment_name.lower() == 'scheduling':
        result = construct_graph_coloring_scheduling()

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'latin_square' in experiment_name.lower() or 'latin' in experiment_name.lower():
        result = construct_latin_square(n=9)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'jsudoku' in experiment_name.lower():
        result = construct_jsudoku(grid_size=9)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'sudoku_gt' in experiment_name.lower() or 'sudoku_greater' in experiment_name.lower():
        result = construct_sudoku_greater_than(3, 3, 9)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'sudoku' in experiment_name.lower():
        result = construct_sudoku(3, 3, 9)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'examtt_v1' in experiment_name.lower() or 'examtt_variant1' in experiment_name.lower():
        result = construct_examtt_variant1(nsemesters=6, courses_per_semester=5, 
                                           slots_per_day=6, days_for_exams=10)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'examtt_v2' in experiment_name.lower() or 'examtt_variant2' in experiment_name.lower():
        result = construct_examtt_variant2(nsemesters=8, courses_per_semester=7, 
                                           slots_per_day=8, days_for_exams=12)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'examtt' in experiment_name.lower():
        result = ces_global(nsemesters=9, courses_per_semester=6, 
                           slots_per_day=9, days_for_exams=14)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'nurse' in experiment_name.lower():
        result = nr_global()
        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'uefa' in experiment_name.lower():
        from benchmarks_global.uefa import construct_uefa as construct_uefa_instance
        
        teams_data = {
            "RealMadrid": {"country": "ESP", "coefficient": 134000},
            "BayernMunich": {"country": "GER", "coefficient": 129000},
            "ManchesterCity": {"country": "ENG", "coefficient": 128000},
            "PSG": {"country": "FRA", "coefficient": 112000},
            "Liverpool": {"country": "ENG", "coefficient": 109000},
            "Barcelona": {"country": "ESP", "coefficient": 98000},
            "Juventus": {"country": "ITA", "coefficient": 95000},
            "AtleticoMadrid": {"country": "ESP", "coefficient": 94000},
            "ManchesterUnited": {"country": "ENG", "coefficient": 92000},
            "Chelsea": {"country": "ENG", "coefficient": 91000},
            "BorussiaDortmund": {"country": "GER", "coefficient": 88000},
            "Ajax": {"country": "NED", "coefficient": 82000},
            "RB Leipzig": {"country": "GER", "coefficient": 79000},
            "InterMilan": {"country": "ITA", "coefficient": 76000},
            "Sevilla": {"country": "ESP", "coefficient": 75000},
            "Napoli": {"country": "ITA", "coefficient": 74000},
            "Benfica": {"country": "POR", "coefficient": 73000},
            "Porto": {"country": "POR", "coefficient": 72000},
            "Arsenal": {"country": "ENG", "coefficient": 71000},
            "ACMilan": {"country": "ITA", "coefficient": 70000},
            "RedBullSalzburg": {"country": "AUT", "coefficient": 69000},
            "ShakhtarDonetsk": {"country": "UKR", "coefficient": 68000},
            "BayerLeverkusen": {"country": "GER", "coefficient": 67000},
            "Olympiacos": {"country": "GRE", "coefficient": 66000},
            "Celtic": {"country": "SCO", "coefficient": 65000},
            "Rangers": {"country": "SCO", "coefficient": 64000},
            "PSVEindhoven": {"country": "NED", "coefficient": 63000},
            "SportingCP": {"country": "POR", "coefficient": 62000},
            "Marseille": {"country": "FRA", "coefficient": 61000},
            "ClubBrugge": {"country": "BEL", "coefficient": 60000},
            "Galatasaray": {"country": "TUR", "coefficient": 59000},
            "Feyenoord": {"country": "NED", "coefficient": 58000}
        }
        
        instance, oracle = construct_uefa_instance(teams_data)
    
    elif 'vm_allocation' in experiment_name.lower():
        print("Constructing VM Allocation...")
        from benchmarks_global.vm_allocation import construct_vm_allocation as construct_vm_instance
        from vm_allocation_model import PM_DATA, VM_DATA
        
        instance, oracle = construct_vm_instance(PM_DATA, VM_DATA)
    
    else:
        print(f"Unknown experiment: {experiment_name}")
        sys.exit(1)
    
    return instance, oracle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='HCAR AllDifferent Phase 2'
    )
    parser.add_argument('--experiment', type=str, default='sudoku',
                       help='Benchmark name (sudoku, examtt, nurse, uefa, vm_allocation)')
    parser.add_argument('--phase1_pickle', type=str, default=None,
                       help='Path to Phase 1 pickle file (optional)')
    parser.add_argument('--alpha', type=float, default=0.42,
                       help='Bayesian learning rate (default: 0.42)')
    parser.add_argument('--theta_max', type=float, default=0.9,
                       help='Acceptance threshold (default: 0.9)')
    parser.add_argument('--theta_min', type=float, default=0.1,
                       help='Rejection threshold (default: 0.1)')
    parser.add_argument('--max_queries', type=int, default=500,
                       help='Maximum total queries (default: 500)')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout in seconds (default: 600)')
    parser.add_argument('--prior', type=float, default=0.5)
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"HCAR AllDifferent COP Experiment")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment}")
    print(f"Alpha: {args.alpha}")
    print(f"Theta_max: {args.theta_max}")
    print(f"Theta_min: {args.theta_min}")
    print(f"Max queries: {args.max_queries}")
    print(f"Timeout: {args.timeout}s")
    print(f"Prior: {args.prior}")
    print(f"{'='*60}\n")

    instance, oracle = construct_instance(args.experiment)

    oracle.variables_list = cpm_array(instance.X)

    phase1_data = None  
    if args.phase1_pickle:

        phase1_data = load_phase1_data(args.phase1_pickle)
        CG = phase1_data['CG']

        if 'initial_probabilities' in phase1_data:
            probabilities = phase1_data['initial_probabilities']
        else:

            probabilities = initialize_probabilities(CG, prior=args.prior)
            print(f"\n No initial_probabilities in pickle, using uniform prior={args.prior}")
    else:

        CG = extract_alldifferent_constraints(oracle)
        
        # print(f"\nExtracted {len(CG)} AllDifferent constraints from oracle:")
        # for i, c in enumerate(CG, 1):
        #     print(f"  {i}. {c}")

        probabilities = initialize_probabilities(CG, prior=args.prior)
    
    if len(CG) == 0:
        print(f"\n No AllDifferent constraints found")
        sys.exit(0)

    C_validated, stats = cop_based_refinement(
        experiment_name=args.experiment,
        oracle=oracle,
        candidate_constraints=CG,
        initial_probabilities=probabilities,
        variables=instance.X,
        alpha=args.alpha,
        theta_max=args.theta_max,
        theta_min=args.theta_min,
        max_queries=args.max_queries,
        timeout=args.timeout
    )

    print(f"\n{'='*60}")
    print(f"Comparison with Target Model")
    print(f"{'='*60}")
    
    target_alldiff = extract_alldifferent_constraints(oracle)
    target_strs = set(str(c) for c in target_alldiff)
    learned_strs = set(str(c) for c in C_validated)
    
    correct = len(learned_strs & target_strs)
    missing = len(target_strs - learned_strs)
    spurious = len(learned_strs - target_strs)
    
    print(f"Target AllDifferent constraints: {len(target_alldiff)}")
    print(f"Learned AllDifferent constraints: {len(C_validated)}")
    print(f"Correct: {correct}")
    print(f"Missing: {missing}")
    print(f"Spurious: {spurious}")
    
    if correct == len(target_alldiff) and spurious == 0:
        print(f"\n[SUCCESS] Perfect learning!")
    else:
        if missing > 0:
            print(f"\n[ERROR] Missing constraints:")
            for c in target_alldiff:
                if str(c) not in learned_strs:
                    print(f"  - {c}")
        
        if spurious > 0:
            print(f"\n[ERROR] Spurious constraints:")
            for c in C_validated:
                if str(c) not in target_strs:
                    print(f"  - {c}")
    
    print(f"\n{'='*60}")
    print(f"Final Statistics")
    print(f"{'='*60}")
    print(f"Total queries: {stats['queries']}")
    print(f"Total time: {stats['time']:.2f}s")
    print(f"Queries per second: {stats['queries']/stats['time']:.2f}")
    print(f"{'='*60}\n")

    phase2_output = {
        'C_validated': C_validated,  
        'C_validated_strs': [str(c) for c in C_validated],  
        'probabilities': probabilities,  
        'experiment_name': args.experiment,
        'phase2_stats': stats,

        'phase1_data': phase1_data if args.phase1_pickle else None,
        'E_plus': phase1_data['E_plus'] if args.phase1_pickle and 'E_plus' in phase1_data else None,
        'B_fixed': phase1_data['B_fixed'] if args.phase1_pickle and 'B_fixed' in phase1_data else None,
        'all_variables': list(instance.X),
        'metadata': {
            'alpha': args.alpha,
            'theta_max': args.theta_max,
            'theta_min': args.theta_min,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_queries': stats['queries'],
            'total_time': stats['time']
        }
    }

    phase2_output_dir = "phase2_output"
    os.makedirs(phase2_output_dir, exist_ok=True)
    phase2_pickle_path = os.path.join(phase2_output_dir, f"{args.experiment}_phase2.pkl")
    
    with open(phase2_pickle_path, 'wb') as f:
        pickle.dump(phase2_output, f)
    
    print(f"\n Phase 2 outputs saved to: {phase2_pickle_path}")
    print(f"  - Validated constraints: {len(C_validated)}")
