import time
from pycona.active_algorithms.quacq import QuAcq
from pycona.problem_instance import ProblemInstance
from pycona.answering_queries import Oracle, UserOracle
from pycona.utils import get_kappa
from pycona import Metrics
from bayesian_ca_env import BayesianActiveCAEnv
from enhanced_bayesian_pqgen import EnhancedBayesianPQGen


class BayesianQuAcq(QuAcq):
    def __init__(self, ca_env=None, theta_max=0.9, 
                theta_min=0.1, prior=0.5, alpha=0.2):

        if ca_env is None:
            ca_env = BayesianActiveCAEnv(
                qgen=EnhancedBayesianPQGen(), 
                theta_max=theta_max,
                theta_min=theta_min,
                prior=prior,
                alpha=alpha
            )
        
        super().__init__(ca_env)
        
        if not isinstance(self.env, BayesianActiveCAEnv):
            self.env = BayesianActiveCAEnv(
                qgen=EnhancedBayesianPQGen(),
                find_scope=self.env.find_scope if hasattr(self.env, 'find_scope') else None,
                findc=self.env.findc if hasattr(self.env, 'findc') else None,
                theta_max=theta_max,
                theta_min=theta_min,
                prior=prior,
                alpha=alpha
            )

    def learn(self, instance: ProblemInstance, oracle: Oracle = UserOracle(), verbose=0, metrics: Metrics = None):
        self.env.init_state(instance, oracle, verbose, metrics)

        if len(self.env.instance.bias) == 0:
            self.env.instance.construct_bias()
            
        max_attempts_without_query = 3
        attempts_without_query = 0
        
        if not hasattr(self.env.qgen, 'previous_assignments'):
            if self.env.verbose >= 1:
                print("Warning: Query generator does not support tracking previous assignments.")
                print("Duplicate queries may be generated.")

        if self.env.verbose >= 1:
            print("\n===== Constraint Budget Information =====")
            print(f"Total query budget for current constraint: {self.env.max_queries}")
            for i, c in enumerate(self.env.instance.bias):
                print(f"Constraint {i+1}/{len(self.env.instance.bias)}: {c}")
                # print(f"  Initial probability: {self.env.constraint_probs.get(c, 0.5):.3f}")
            print("========================================\n")

        while True:
            if self.env.verbose > 2:
                print("Size of CL: ", len(self.env.instance.cl))
                print("Size of B: ", len(self.env.instance.bias))
                print("Number of Queries: ", self.env.metrics.membership_queries_count)
                
                if hasattr(self.env.qgen, 'print_assignment_history') and len(self.env.instance.bias) == 1:
                    self.env.qgen.print_assignment_history(self.env.instance.bias[0])

            gen_start = time.time()
            Y = self.env.run_query_generation()
            gen_end = time.time()

            if Y is None:
                if self.env.verbose >= 1:
                    print("Query budget exceeded, forcing decision based on current probabilities.")

                for c in list(self.env.instance.bias):
                    if self.env.constraint_probs[c] >= 0.8: 
                        if self.env.verbose >= 1:
                            print(f"Adding {c} to CL (p={self.env.constraint_probs[c]:.3f})")
                        self.env.instance.cl.append(c)
                    else:
                        if self.env.verbose >= 1:
                            print(f"Removing {c} from bias (p={self.env.constraint_probs[c]:.3f})")
                    self.env.instance.bias.remove(c)

                self.env.metrics.finalize_statistics()
                if self.env.verbose >= 1:
                    print(f"\nLearned {len(self.env.instance.cl)} constraints with budget exhausted.")
                return self.env.instance
            
            if len(Y) == 0:
                attempts_without_query += 1


                if attempts_without_query >= max_attempts_without_query or len(self.env.instance.bias) == 0:
                    self.env.metrics.finalize_statistics()
                    if self.env.verbose >= 1:
                        print(f"\nLearned {self.env.metrics.cl} constraints in "
                            f"{self.env.metrics.membership_queries_count} queries.")
                        print("Active learning stops because no more queries can be generated.")
                    return self.env.instance

                if len(self.env.instance.bias) > 0:
                    if self.env.verbose >= 1:
                        print("No query could be generated, but constraints remain in bias.")
                        print("Forcing decision based on current probabilities.")

                    for c in list(self.env.instance.bias):
                        if self.env.constraint_probs[c] >= 0.7: 
                            if self.env.verbose >= 1:
                                print(f"Adding {c} to CL (p={self.env.constraint_probs[c]:.3f})")
                            self.env.instance.cl.append(c)
                        else:
                            if self.env.verbose >= 1:
                                print(f"Removing {c} from bias (p={self.env.constraint_probs[c]:.3f})")
                        self.env.instance.bias.remove(c)
                    
                continue

            attempts_without_query = 0

            self.env.metrics.increase_generation_time(gen_end - gen_start)
            self.env.metrics.increase_generated_queries()
            self.env.metrics.increase_top_queries()
            kappaB = get_kappa(self.env.instance.bias, Y)
            
            if self.env.verbose >= 2:
                print(f"Query{self.env.metrics.membership_queries_count}: is this a solution?")
                print(Y)
                print(f"violated from B: {kappaB}")

            answer = self.env.ask_membership_query(Y)
            if answer:    
                print(f"Removing {kappaB} from bias")
                self.env.remove_from_bias(kappaB)
            else:  
                print(f"Adding {kappaB} to CL")
                self.env.bayesian_add_to_cl(kappaB)

        return self.env.instance 