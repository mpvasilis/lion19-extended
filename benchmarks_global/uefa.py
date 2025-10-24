import cpmpy as cp
import numpy as np
from cpmpy import cpm_array
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar

def construct_uefa(teams_data, n_groups=8, teams_per_group=4):
    parameters = {
        "n_groups": n_groups,
        "teams_per_group": teams_per_group,
        "n_teams": len(teams_data)
    }

    group_assignments = {}
    for team in teams_data:
        group_assignments[team] = cp.intvar(0, n_groups-1, name=f"group_{team}")
 

    model = cp.Model()
    
    all_constraints = []
    global_constraints = []

    for g in range(n_groups):
        group_size = cp.Count([group_assignments[team] for team in teams_data], g)
        constraint = (group_size == teams_per_group)
        all_constraints.append(constraint)

        global_constraints.append(constraint) 

    countries = {}
    for team, data in teams_data.items():
        country = data['country']
        if country not in countries:
            countries[country] = []
        countries[country].append(team)

    for country, country_teams in countries.items():
        if 1 < len(country_teams) <= 4:  
            country_vars = [group_assignments[team] for team in country_teams]
            constraint = cp.AllDifferent(country_vars)
            all_constraints.append(constraint)

            global_constraints.append(constraint) 

    teams_by_coefficient = sorted(teams_data.items(), key=lambda x: x[1]['coefficient'], reverse=True)
    for pot_idx in range(0, len(teams_data), n_groups):
        pot_end = min(pot_idx + n_groups, len(teams_data))
        pot_teams = teams_by_coefficient[pot_idx:pot_end]
        if len(pot_teams) > 1:
            pot_vars = [group_assignments[team[0]] for team in pot_teams]
            constraint = cp.AllDifferent(pot_vars)
            all_constraints.append(constraint)


            global_constraints.append(constraint)  
































































    oracle_constraints = list(global_constraints)



    mock_constraints = []


    mock_size = min(n_groups, 4)

    if len(teams_data) >= mock_size * 2:

        first_teams = sorted(teams_data.keys())[:mock_size]
        first_vars = [group_assignments[team] for team in first_teams]
        mock_c1 = cp.AllDifferent(first_vars)
        mock_constraints.append(mock_c1)
        all_constraints.append(mock_c1)
        global_constraints.append(mock_c1)

        last_teams = sorted(teams_data.keys())[-mock_size:]
        last_vars = [group_assignments[team] for team in last_teams]
        mock_c2 = cp.AllDifferent(last_vars)
        mock_constraints.append(mock_c2)
        all_constraints.append(mock_c2)
        global_constraints.append(mock_c2)

    for constraint in all_constraints:
        model += constraint

    AV = absvar(2)  

    lang = [
        sum(AV) == AV[0],    
        sum(AV) <= AV[0],   
        sum(AV) >= AV[0],     
    ]

    variables = []
    for team in sorted(teams_data.keys()):
        variables.append(group_assignments[team])








    instance = ProblemInstance(variables=cpm_array(variables), params=parameters, language=lang, name="uefa")
    oracle = ConstraintOracle(oracle_constraints)  

    return instance, oracle 

def generate_uefa_instance(instance_params=None):
  
    if instance_params is None:
        instance_params = {}
    
    n_groups = instance_params.get('n_groups', 8)
    teams_per_group = instance_params.get('teams_per_group', 4)
    n_teams = n_groups * teams_per_group
    
    teams_data = {}
    countries = ['England', 'Spain', 'Germany', 'Italy', 'France', 'Portugal', 'Netherlands', 'Russia']
    
    while len(countries) < n_teams // 4 + 1:
        countries.append(f"Country_{len(countries) + 1}")
    
    for i in range(n_teams):
        country_idx = i % len(countries)
        team_name = f"Team_{i+1}"
        teams_data[team_name] = {
            'country': countries[country_idx],
            'coefficient': 100 - i
        }
    
    return construct_uefa(teams_data, n_groups, teams_per_group)