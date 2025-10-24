import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_graph_coloring(graph_type="queen_5x5", num_colors=None):
    
    
    if graph_type == "queen_5x5":



        n = 5
        num_nodes = n * n
        if num_colors is None:
            num_colors = 5

        nodes = cp.intvar(1, num_colors, shape=(n, n), name="color")
        
        model = cp.Model()

        for row in nodes:
            model += cp.AllDifferent(row)

        for col in nodes.T:
            model += cp.AllDifferent(col)


        for k in range(-(n-1), n):
            diag = [nodes[i, i-k] for i in range(n) if 0 <= i-k < n]
            if len(diag) > 1:
                model += cp.AllDifferent(diag)

        for k in range(2*n-1):
            diag = [nodes[i, k-i] for i in range(n) if 0 <= k-i < n]
            if len(diag) > 1:
                model += cp.AllDifferent(diag)
        
        variables = nodes
        graph_name = "queen_5x5"
        
    elif graph_type == "queen_6x6":

        n = 6
        num_nodes = n * n
        if num_colors is None:
            num_colors = 6
        
        nodes = cp.intvar(1, num_colors, shape=(n, n), name="color")
        model = cp.Model()

        for row in nodes:
            model += cp.AllDifferent(row)
        for col in nodes.T:
            model += cp.AllDifferent(col)
        for k in range(-(n-1), n):
            diag = [nodes[i, i-k] for i in range(n) if 0 <= i-k < n]
            if len(diag) > 1:
                model += cp.AllDifferent(diag)
        for k in range(2*n-1):
            diag = [nodes[i, k-i] for i in range(n) if 0 <= k-i < n]
            if len(diag) > 1:
                model += cp.AllDifferent(diag)
        
        variables = nodes
        graph_name = "queen_6x6"
        
    elif graph_type == "register":



        if num_colors is None:
            num_colors = 4  
        
        num_nodes = 12
        nodes = cp.intvar(1, num_colors, shape=num_nodes, name="register")
        
        model = cp.Model()


        cliques = [
            [0, 1, 2],           
            [1, 2, 3, 4],        
            [3, 4, 5],           
            [4, 5, 6, 7],        
            [6, 7, 8],           
            [8, 9, 10, 11],      
            [0, 9, 10],          
            [2, 5, 8, 11],       
        ]
        
        for clique in cliques:
            if len(clique) > 1:
                model += cp.AllDifferent([nodes[i] for i in clique])
        
        variables = nodes
        graph_name = "register_allocation"
        
    elif graph_type == "scheduling":



        if num_colors is None:
            num_colors = 5  
        
        num_courses = 15
        courses = cp.intvar(1, num_colors, shape=num_courses, name="timeslot")
        
        model = cp.Model()


        conflict_groups = [
            [0, 1, 2, 3],        
            [2, 3, 4, 5],        
            [4, 5, 6, 7, 8],     
            [6, 7, 9, 10],       
            [9, 10, 11, 12],     
            [11, 12, 13, 14],    
            [0, 5, 10, 14],      
            [1, 6, 11],          
            [3, 8, 13],          
        ]
        
        for group in conflict_groups:
            if len(group) > 1:
                model += cp.AllDifferent([courses[i] for i in group])
        
        variables = courses
        graph_name = "course_scheduling"
        
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    C_T = list(set(toplevel_list(model.constraints)))


    mock_constraints = []



    
    if graph_type == "register" and num_nodes >= 8:

        mock_c1 = cp.AllDifferent([nodes[0], nodes[3], nodes[6]])
        mock_constraints.append(mock_c1)

        mock_c2 = cp.AllDifferent([nodes[1], nodes[4], nodes[7]])
        mock_constraints.append(mock_c2)
    
    if graph_type == "queen_5x5":

        n = 5
        mock_c2 = cp.AllDifferent([nodes[0,0], nodes[0,n-1], nodes[n-1,0], nodes[n-1,n-1]])
        mock_constraints.append(mock_c2)
    
    if graph_type == "scheduling":

        mock_c1 = cp.AllDifferent([courses[0], courses[5], courses[10]])
        mock_constraints.append(mock_c1)
        mock_c2 = cp.AllDifferent([courses[2], courses[7], courses[12]])
        mock_constraints.append(mock_c2)

    AV = absvar(2)
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]
    
    parameters = {"graph_type": graph_type, "num_colors": num_colors}
    instance = ProblemInstance(variables=variables, params=parameters, language=lang, name=graph_name)
    oracle = ConstraintOracle(C_T)
    
    return instance, oracle, mock_constraints


def construct_graph_coloring_queen5():
    
    return construct_graph_coloring("queen_5x5", num_colors=5)


def construct_graph_coloring_register():
    
    return construct_graph_coloring("register", num_colors=4)


def construct_graph_coloring_scheduling():
    
    return construct_graph_coloring("scheduling", num_colors=5)

