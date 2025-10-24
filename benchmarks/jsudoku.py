import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_jsudoku_binary(grid_size=9, regions=None):
    
    
    parameters = {"grid_size": grid_size}

    if regions is None and grid_size == 9:
        regions = [

            [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)],

            [(0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)],

            [(0,6), (0,7), (0,8), (1,6), (1,7), (1,8), (2,6), (2,7), (2,8)],

            [(3,0), (3,1), (3,2), (4,0), (4,1), (4,2), (5,0), (5,1), (5,2)],

            [(3,3), (3,4), (3,5), (4,3), (4,4), (4,5), (5,3), (5,4), (5,5)],

            [(3,6), (3,7), (3,8), (4,6), (4,7), (4,8), (5,6), (5,7), (5,8)],

            [(6,0), (6,1), (6,2), (7,0), (7,1), (7,2), (8,0), (8,1), (8,2)],

            [(6,3), (6,4), (6,5), (7,3), (7,4), (7,5), (8,3), (8,4), (8,5)],

            [(6,6), (6,7), (6,8), (7,6), (7,7), (7,8), (8,6), (8,7), (8,8)],
        ]
    elif regions is None:
        raise ValueError(f"No default regions for grid_size={grid_size}. Please provide custom regions.")
    
    parameters["num_regions"] = len(regions)

    grid = cp.intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")
    
    model = cp.Model()

    for row in grid:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                model += (row[i] != row[j])

    for col in grid.T:
        for i in range(len(col)):
            for j in range(i + 1, len(col)):
                model += (col[i] != col[j])

    for region_idx, region in enumerate(regions):
        region_cells = [grid[r, c] for r, c in region]
        for i in range(len(region_cells)):
            for j in range(i + 1, len(region_cells)):
                model += (region_cells[i] != region_cells[j])
    
    C_T = list(set(toplevel_list(model.constraints)))

    AV = absvar(2)
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]
    
    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name="jsudoku")
    oracle = ConstraintOracle(C_T)
    
    return instance, oracle


def construct_jsudoku_binary_4x4():
    
    regions = [

        [(0,0), (0,1), (1,0), (1,1)],

        [(0,2), (0,3), (1,2), (1,3)],

        [(2,0), (2,1), (3,0), (3,1)],

        [(2,2), (2,3), (3,2), (3,3)],
    ]
    return construct_jsudoku_binary(grid_size=4, regions=regions)


def construct_jsudoku_binary_6x6():
    
    regions = [

        [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)],

        [(0,2), (0,3), (1,2), (1,3), (2,2), (2,3)],

        [(0,4), (0,5), (1,4), (1,5), (2,4), (2,5)],

        [(3,0), (3,1), (4,0), (4,1), (5,0), (5,1)],

        [(3,2), (3,3), (4,2), (4,3), (5,2), (5,3)],

        [(3,4), (3,5), (4,4), (4,5), (5,4), (5,5)],
    ]
    return construct_jsudoku_binary(grid_size=6, regions=regions)

