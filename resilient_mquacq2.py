

import time
import numpy as np
from pycona import MQuAcq2
from pycona.utils import get_kappa


class ResilientMQuAcq2(MQuAcq2):
    
    
    def __init__(self, ca_env=None, **kwargs):
        super().__init__(ca_env, **kwargs)
        self.skipped_scopes = []
        self.invalid_cl_constraints = []  
    
    def learn(self, instance, oracle=None, verbose=0, X=None, metrics=None):
        
        from pycona import UserOracle
        
        if oracle is None:
            oracle = UserOracle()
        
        if X is None:
            X = instance.X
        assert isinstance(X, list) and set(X).issubset(set(instance.X)), \
            "When using .learn(), set parameter X must be a list of variables"
        
        self.env.init_state(instance, oracle, verbose, metrics)

        self.hashX = [hash(x) for x in self.env.instance.X]
        self.cl_neighbours = np.zeros((len(self.env.instance.X), len(self.env.instance.X)), dtype=bool)
        
        if len(self.env.instance.bias) == 0:
            self.env.instance.construct_bias()
        
        while True:
            gen_start = time.time()
            Y = self.env.run_query_generation(X)
            gen_end = time.time()
            self.env.metrics.increase_generation_time(gen_end - gen_start)
            
            if len(Y) == 0:

                self.env.metrics.finalize_statistics()
                if self.env.verbose >= 1:
                    print(f"\nLearned {self.env.metrics.cl} constraints in "
                          f"{self.env.metrics.membership_queries_count} queries.")
                    if len(self.skipped_scopes) > 0:
                        print(f"[RESILIENT] Skipped {len(self.skipped_scopes)} scope(s) due to missing constraints in bias")
                return self.env.instance
            
            self.env.metrics.increase_generated_queries()
            kappaB = get_kappa(self.env.instance.bias, Y)
            
            while len(kappaB) > 0:
                
                if self.env.verbose >= 3:
                    print("Size of CL: ", len(self.env.instance.cl))
                    print("Size of B: ", len(self.env.instance.bias))
                    print("Number of queries: ", self.env.metrics.total_queries)
                    print("MQuAcq-2 Queries: ", self.env.metrics.top_lvl_queries)
                    print("FindScope Queries: ", self.env.metrics.findscope_queries)
                    print("FindC Queries: ", self.env.metrics.findc_queries)
                
                self.env.metrics.increase_top_queries()
                
                if self.env.ask_membership_query(Y):

                    self.env.remove_from_bias(kappaB)
                    kappaB = set()
                else:  
                    
                    scope = self.env.run_find_scope(Y)
                    c = self.env.run_findc(scope)

                    if c is None:

                        scope_str = f"[{', '.join(str(v) for v in scope)}]"
                        self.skipped_scopes.append(scope_str)



                        Y = [y for y in Y if y not in scope]
                        kappaB = get_kappa(self.env.instance.bias, Y)
                        continue

                    self.env.add_to_cl(c)
                    
                    NScopes = set()
                    NScopes.add(tuple(scope))
                    
                    if self.perform_analyzeAndLearn:
                        try:
                            # Validate CL before calling analyze_and_learn
                            self._validate_cl_for_mquacq2()
                            NScopes = NScopes.union(self.analyze_and_learn(Y))
                        except IndexError as e:
                            # Handle the case where find_neighbours fails due to invalid scope
                            print(f"\n[WARNING] analyze_and_learn failed: {e}")                    
                    Y = [y2 for y2 in Y if not any(y2 in set(nscope) for nscope in NScopes)]
                    
                    kappaB = get_kappa(self.env.instance.bias, Y)
    
    @property
    def perform_analyzeAndLearn(self):
        
        return self._perform_analyzeAndLearn
    
    def _validate_cl_for_mquacq2(self):
        from utils import get_scope
        
        valid_cl = []
        for c in self.env.instance.cl:
            try:
                scope = get_scope(c)
                
                # Check if all variables in scope are in instance.X
                scope_hashes = [hash(v) for v in scope]
                all_in_X = all(h in self.hashX for h in scope_hashes)
                
                if len(scope) < 2:
                    print(f"  [VALIDATION] Removing constraint with invalid scope (arity {len(scope)}): {c}")
                    self.invalid_cl_constraints.append(c)
                elif not all_in_X:
                    print(f"  [VALIDATION] Removing constraint with variables not in instance.X: {c}")
                    self.invalid_cl_constraints.append(c)
                else:
                    valid_cl.append(c)
            except Exception as e:
                print(f"  [VALIDATION] Failed to validate constraint {c}: {e}")
                self.invalid_cl_constraints.append(c)
        
        # Update the CL with only valid constraints
        if len(valid_cl) < len(self.env.instance.cl):
            removed_count = len(self.env.instance.cl) - len(valid_cl)
            print(f"  [VALIDATION] Removed {removed_count} invalid constraints from CL")
            self.env.instance.cl = valid_cl
    
    def get_resilience_report(self):
        
        return {
            'skipped_scopes_count': len(self.skipped_scopes),
            'skipped_scopes': self.skipped_scopes,
            'invalid_cl_constraints_count': len(self.invalid_cl_constraints),
            'invalid_cl_constraints': [str(c) for c in self.invalid_cl_constraints]
        }

