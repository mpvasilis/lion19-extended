import numpy as np
from cpmpy.expressions.core import Expression
from pycona.ca_environment.active_ca import ActiveCAEnv
from pycona.utils import get_kappa


class BayesianActiveCAEnv(ActiveCAEnv):
    def __init__(self, qgen=None, find_scope=None, findc=None,
                theta_max=0.9, theta_min=0.1, prior=0.5, alpha=0.42):

        super().__init__(qgen, find_scope, findc)
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.prior = prior
        self.alpha = alpha
        self.constraint_probs = {}
        self.current_constraint = None
        self.max_queries = None  
        self.queries_executed = 0  
        
    def init_state(self, instance, oracle, verbose, metrics=None):
        super().init_state(instance, oracle, verbose, metrics)

        self.queries_executed = 0
        
        for c in self.instance.bias:
            if c not in self.constraint_probs:
                self.constraint_probs[c] = self.prior
                
    def update_probabilities(self, constraints, is_positive_example):
        for c in constraints:
            if c not in self.constraint_probs:
                self.constraint_probs[c] = self.prior

            p_c_in_Ct = self.constraint_probs[c]
            
            if is_positive_example:

                p_E_given_c_in_Ct     = 1 - self.alpha

                p_E_given_not_c_in_Ct = self.alpha
            else:

                p_E_given_c_in_Ct     = 1 - self.alpha

                p_E_given_not_c_in_Ct = self.alpha

            p_c_in_Ct_and_E      = p_E_given_c_in_Ct * p_c_in_Ct

            p_E                  = p_c_in_Ct_and_E + p_E_given_not_c_in_Ct * (1 - p_c_in_Ct)

            p_c_in_Ct_given_E    = p_c_in_Ct_and_E / p_E
            
            self.constraint_probs[c] = p_c_in_Ct_given_E
            
            if self.verbose >= 3:
                print(f"Updated P({c} ∈ Cₜ | E): {p_c_in_Ct:.3f} -> {p_c_in_Ct_given_E:.3f}")
    
    def run_query_generation(self):

        if self.max_queries is not None and self.queries_executed >= self.max_queries:
            if self.verbose >= 2:
                print(f"Maximum queries ({self.max_queries}) reached, stopping query generation")
            return None

        if len(self.instance.bias) == 1:
            self.current_constraint = self.instance.bias[0]
        else:
            self.current_constraint = None
            
        result = super().run_query_generation()

        if result is not None:
            self.queries_executed += 1
            
        return result
    
    def check_budget_exceeded(self):
        
        if self.max_queries is not None and self.queries_executed >= self.max_queries:
            if self.verbose >= 2:
                print(f"Budget exceeded: {self.queries_executed}/{self.max_queries} queries used")
            return True
        return False
    
    def bayesian_add_to_cl(self, kappaB):
        self.update_probabilities(kappaB, is_positive_example=False)
        
        constraints_to_add = []
        constraints_to_remove = []
        
        for c in kappaB:

            if self.constraint_probs[c] >= self.theta_max:
                constraints_to_add.append(c)
                if self.verbose >= 2:
                    print(f"Constraint {c} probability {self.constraint_probs[c]:.3f} exceeds threshold {self.theta_max}, adding to CL")

            elif self.constraint_probs[c] <= self.theta_min:
                constraints_to_remove.append(c)
                if self.verbose >= 2:
                    print(f"Constraint {c} probability {self.constraint_probs[c]:.3f} below threshold {self.theta_min}, removing from bias")

            elif self.check_budget_exceeded():
                if self.constraint_probs[c] >= 0.5:  
                    constraints_to_add.append(c)
                    if self.verbose >= 2:
                        print(f"Budget exceeded with probability {self.constraint_probs[c]:.3f} > 0.5, adding to CL")
                else:
                    constraints_to_remove.append(c)
                    if self.verbose >= 2:
                        print(f"Budget exceeded with probability {self.constraint_probs[c]:.3f} < 0.5, removing from bias")
        
        if constraints_to_add:
            for c in constraints_to_add:
                if c in self.instance.bias:
                    self.instance.cl.append(c)
                    self.instance.bias.remove(c)
        
        if constraints_to_remove:
            for c in constraints_to_remove:
                if c in self.instance.bias:
                    self.instance.bias.remove(c)
                    if c in self.constraint_probs:
                        del self.constraint_probs[c]
    
    def bayesian_remove_from_bias(self, kappaB):
        for c in kappaB:
            if c in self.instance.bias:
                if self.verbose >= 2:
                    print(f"Immediately removing constraint {c} from bias due to positive example")
                self.instance.bias.remove(c)
                if c in self.constraint_probs:
                    del self.constraint_probs[c]
    
    def add_to_cl(self, kappaB):
        self.bayesian_add_to_cl(kappaB)
        
    def remove_from_bias(self, kappaB):
        self.bayesian_remove_from_bias(kappaB) 