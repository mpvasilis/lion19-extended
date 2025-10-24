import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_sudoku_binary(block_size_row, block_size_col, grid_size):

    parameters = {"block_size_row": block_size_row, "block_size_col": block_size_col, "grid_size": grid_size}

    grid = cp.intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")

    model = cp.Model()

    for row in grid:
        model += cp.AllDifferent(row)

    for col in grid.T:
        model += cp.AllDifferent(col)

    for i in range(0, grid_size, block_size_row):
        for j in range(0, grid_size, block_size_col):
            model += cp.AllDifferent(grid[i:i + block_size_row, j:j + block_size_col]) 

    C_T = list(set(toplevel_list(model.constraints)))

    AV = absvar(2)  

    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name="sudoku")

    oracle = ConstraintOracle(C_T)

    return instance, oracle