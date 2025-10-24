import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_graph_coloring_binary(graph_type="queen_5x5", num_colors=None):
    
    
    if graph_type == "queen_5x5":
        n = 5
        if num_colors is None:
            num_colors = 5
        
        nodes = cp.intvar(1, num_colors, shape=(n, n), name="color")
        model = cp.Model()

        for row in nodes:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    model += (row[i] != row[j])

        for col in nodes.T:
            for i in range(len(col)):
                for j in range(i + 1, len(col)):
                    model += (col[i] != col[j])

        for k in range(-(n-1), n):
            diag = [nodes[i, i-k] for i in range(n) if 0 <= i-k < n]
            if len(diag) > 1:
                for i in range(len(diag)):
                    for j in range(i + 1, len(diag)):
                        model += (diag[i] != diag[j])
        
        for k in range(2*n-1):
            diag = [nodes[i, k-i] for i in range(n) if 0 <= k-i < n]
            if len(diag) > 1:
                for i in range(len(diag)):
                    for j in range(i + 1, len(diag)):
                        model += (diag[i] != diag[j])
        
        variables = nodes
        graph_name = "queen_5x5"
        
    elif graph_type == "queen_6x6":
        n = 6
        if num_colors is None:
            num_colors = 6
        
        nodes = cp.intvar(1, num_colors, shape=(n, n), name="color")
        model = cp.Model()

        for row in nodes:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    model += (row[i] != row[j])
        for col in nodes.T:
            for i in range(len(col)):
                for j in range(i + 1, len(col)):
                    model += (col[i] != col[j])
        for k in range(-(n-1), n):
            diag = [nodes[i, i-k] for i in range(n) if 0 <= i-k < n]
            if len(diag) > 1:
                for i in range(len(diag)):
                    for j in range(i + 1, len(diag)):
                        model += (diag[i] != diag[j])
        for k in range(2*n-1):
            diag = [nodes[i, k-i] for i in range(n) if 0 <= k-i < n]
            if len(diag) > 1:
                for i in range(len(diag)):
                    for j in range(i + 1, len(diag)):
                        model += (diag[i] != diag[j])
        
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
                clique_nodes = [nodes[i] for i in clique]
                for i in range(len(clique_nodes)):
                    for j in range(i + 1, len(clique_nodes)):
                        model += (clique_nodes[i] != clique_nodes[j])
        
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
                group_courses = [courses[i] for i in group]
                for i in range(len(group_courses)):
                    for j in range(i + 1, len(group_courses)):
                        model += (group_courses[i] != group_courses[j])
        
        variables = courses
        graph_name = "course_scheduling"
        
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    C_T = list(set(toplevel_list(model.constraints)))
    
    AV = absvar(2)
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]
    
    parameters = {"graph_type": graph_type, "num_colors": num_colors}
    instance = ProblemInstance(variables=variables, params=parameters, language=lang, name=graph_name)
    oracle = ConstraintOracle(C_T)
    
    return instance, oracle


def construct_graph_coloring_binary_queen5():
    
    return construct_graph_coloring_binary("queen_5x5", num_colors=5)


def construct_graph_coloring_binary_register():
    
    return construct_graph_coloring_binary("register", num_colors=4)


def construct_graph_coloring_binary_scheduling():
    
    return construct_graph_coloring_binary("scheduling", num_colors=5)

