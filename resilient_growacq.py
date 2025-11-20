
import copy
import time
from pycona import GrowAcq
from pycona.problem_instance import ProblemInstance
from pycona.answering_queries import Oracle, UserOracle
from pycona import Metrics


class ResilientGrowAcq(GrowAcq):

    def __init__(self, ca_env=None, inner_algorithm=None, **kwargs):
        
        if inner_algorithm is None:
            from resilient_mquacq2 import ResilientMQuAcq2
            inner_algorithm = ResilientMQuAcq2(ca_env=ca_env)
        
        super().__init__(ca_env=ca_env, inner_algorithm=inner_algorithm, **kwargs)
        self.skipped_scopes = []
        self.invalid_cl_constraints = []
        self.collapse_warnings = 0
    
    def learn(self, instance: ProblemInstance, oracle: Oracle = UserOracle(), verbose=0, X=None, metrics: Metrics = None):
 
        if X is None:
            X = instance.X
        assert isinstance(X, list) and set(X).issubset(set(instance.X)), \
            "When using .learn(), set parameter X must be a list of variables"

        self.env.init_state(instance, oracle, verbose, metrics)

        if verbose >= 1:
            print(f"Running ResilientGrowAcq with {self.inner_algorithm.__class__.__name__} as inner algorithm")

        Y = []
        n_vars = len(X)
        
        for x in X:
            try:
                
                Y.append(x)
                
                
                if len(self.env.instance.bias) == 0:
                    self.env.instance.construct_bias_for_var(x, Y)
                    
                if verbose >= 3:
                    print(f"Added variable {x} in GrowAcq")
                    print("size of B in growacq: ", len(self.env.instance.bias))

                if verbose >= 2:
                    print(f"\nGrowAcq: calling inner_algorithm for {len(Y)}/{n_vars} variables")
                
                
                self.inner_algorithm.env = copy.copy(self.env)
                
                
                try:
                    self.env.instance = self.inner_algorithm.learn(
                        self.env.instance, 
                        oracle, 
                        verbose=verbose, 
                        X=Y, 
                        metrics=self.env.metrics
                    )
                except Exception as e:
                    
                    print(f"\n[WARNING] Inner algorithm failed for variable {x}: {e}")
                    self.collapse_warnings += 1
                    
                    
                    
                    if hasattr(self.inner_algorithm, 'get_resilience_report'):
                        inner_report = self.inner_algorithm.get_resilience_report()
                        if inner_report.get('skipped_scopes_count', 0) > 0:
                            self.skipped_scopes.extend(inner_report.get('skipped_scopes', []))
                        if inner_report.get('invalid_cl_constraints_count', 0) > 0:
                            self.invalid_cl_constraints.extend(inner_report.get('invalid_cl_constraints', []))
                    
                    
                    continue

                if verbose >= 3:
                    print("C_L: ", len(self.env.instance.cl))
                    print("B: ", len(self.env.instance.bias))
                    print("Number of queries: ", self.env.metrics.membership_queries_count)
                    print("Top level Queries: ", self.env.metrics.top_lvl_queries)
                    print("FindScope Queries: ", self.env.metrics.findscope_queries)
                    print("FindC Queries: ", self.env.metrics.findc_queries)
                    
            except Exception as e:
                
                print(f"\n[WARNING] Error processing variable {x} in GrowAcq: {e}")
                self.collapse_warnings += 1
                
                continue
        
        
        if hasattr(self.inner_algorithm, 'get_resilience_report'):
            inner_report = self.inner_algorithm.get_resilience_report()
            if inner_report.get('skipped_scopes_count', 0) > 0:
                self.skipped_scopes.extend(inner_report.get('skipped_scopes', []))
            if inner_report.get('invalid_cl_constraints_count', 0) > 0:
                self.invalid_cl_constraints.extend(inner_report.get('invalid_cl_constraints', []))

        if verbose >= 3:
            print("Number of queries: ", self.env.metrics.membership_queries_count)
            print("Number of recommendation queries: ", self.env.metrics.recommendation_queries_count)
            print("Number of generalization queries: ", self.env.metrics.generalization_queries_count)
            print("Top level Queries: ", self.env.metrics.top_lvl_queries)
            print("FindScope Queries: ", self.env.metrics.findscope_queries)
            print("FindC Queries: ", self.env.metrics.findc_queries)
        
        if verbose >= 1:
            if len(self.skipped_scopes) > 0:
                print(f"[RESILIENT] GrowAcq skipped {len(self.skipped_scopes)} scope(s) due to missing constraints in bias")
            if self.collapse_warnings > 0:
                print(f"[RESILIENT] GrowAcq handled {self.collapse_warnings} collapse(s) during learning")

        self.env.metrics.finalize_statistics()
        return self.env.instance
    
    def get_resilience_report(self):
        """
        Get resilience report including skipped scopes and invalid constraints.
        Also aggregates reports from inner algorithm if available.
        """
        report = {
            'skipped_scopes_count': len(self.skipped_scopes),
            'skipped_scopes': self.skipped_scopes,
            'invalid_cl_constraints_count': len(self.invalid_cl_constraints),
            'invalid_cl_constraints': [str(c) for c in self.invalid_cl_constraints],
            'collapse_warnings': self.collapse_warnings
        }
        
        
        if hasattr(self.inner_algorithm, 'get_resilience_report'):
            inner_report = self.inner_algorithm.get_resilience_report()
            report['inner_algorithm'] = inner_report
        
        return report

