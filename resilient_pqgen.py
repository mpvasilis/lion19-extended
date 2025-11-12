"""
Resilient PQGen that handles None values in solution hints.
"""

import time
import cpmpy as cp
from pycona.query_generation.pqgen import PQGen
from cpmpy.transformations.get_variables import get_variables
from pycona.utils import get_con_subset


class ResilientPQGen(PQGen):
    """
    Extension of PQGen that handles None values in solution hints gracefully.
    """
    
    def generate(self, Y=None):
        """
        Generate a partial query that violates at least one constraint in the bias.
        Handles None values in solution hints to prevent crashes.
        """
        if Y is None:
            Y = self.env.instance.X
        assert isinstance(Y, list), "When generating a query, Y must be a list of variables"

        t0 = time.time()
        
        # Get variables that appear in bias
        Y2 = frozenset(get_variables(self.env.instance.bias))
        
        if len(Y2) < len(Y):
            Y = Y2
        
        lY = list(Y)
        
        # Get constraints from bias that involve Y
        B = get_con_subset(self.env.instance.bias, Y)
        
        # Get constraints from CL that involve Y
        Cl = get_con_subset(self.env.instance.cl, Y)
        
        if len(B) == 0:
            return set()
        
        if len(Cl) == 0:
            # Use all of CL if no constraints found on Y
            Cl = self.env.instance.cl
        
        # First check if we should switch to partial queries
        if not self.partial and len(B) > self.blimit:
            m = cp.Model(Cl)
            flag = m.solve()
            
            if flag and not all([c.value() for c in B]):
                return lY
            else:
                self.partial = True
        
        # Build model with CL and negation of at least one constraint in B
        m = cp.Model(Cl)
        m += ~cp.all(B)
        
        # Solve to find a partial query
        flag = m.solve()
        
        t1 = time.time() - t0
        if not flag or (t1 > self.time_limit):
            return lY if flag else set()
        
        # Extract values, filtering out None values
        values = [x.value() for x in lY]
        
        # Filter out variables with None values before creating solution hint
        valid_vars = []
        valid_values = []
        none_count = 0
        
        for i, (var, val) in enumerate(zip(lY, values)):
            if val is not None:
                valid_vars.append(var)
                valid_values.append(val)
            else:
                none_count += 1
        
        if none_count > 0 and self.env and hasattr(self.env, 'verbose') and self.env.verbose >= 2:
            print(f"  [PQGen] Filtered {none_count} variables with None values from solution hint")
        
        # Create new model with valid solution hint
        if valid_vars and valid_values:
            m2 = cp.Model(Cl)
            m2 += ~cp.all(B)
            
            # Add solution hint only for variables with valid values
            try:
                if hasattr(m2, 'solution_hint'):
                    m2.solution_hint(valid_vars, valid_values)
            except Exception as e:
                if self.env and hasattr(self.env, 'verbose') and self.env.verbose >= 1:
                    print(f"  [WARNING] Failed to add solution hint: {e}")
                # Continue without hint
            
            # Re-solve with hint
            flag2 = m2.solve()
            
            if flag2:
                # Use the new solution
                return lY
        
        # Return the original result
        return lY

