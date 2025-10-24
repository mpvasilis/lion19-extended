import cpmpy as cp
import numpy as np

from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar
from cpmpy.transformations.get_variables import get_variables

def construct_vm_allocation(pm_data, vm_data):

    parameters = {
        "n_pms": len(pm_data),
        "n_vms": len(vm_data),
        "pm_capacities": {pm: {k: v for k, v in data.items() if k.startswith('capacity')} 
                         for pm, data in pm_data.items()},
        "vm_demands": {vm: {k: v for k, v in data.items() if k.startswith('demand')}
                      for vm, data in vm_data.items()}
    }

    model = cp.Model()
    
    vm_assignments = {
        vm: cp.intvar(0, len(pm_data), name=f"assign_{vm}") 
        for vm in vm_data
    }
    
    gpu_assignments = {
        vm: cp.intvar(0, pm_data[pm].get("capacity_gpu", 0), name=f"gpu_assign_{vm}")
        for vm in vm_data
        for pm in pm_data
        if vm_data[vm].get("demand_gpu", 0) > 0 and "capacity_gpu" in pm_data[pm]
    }
    
    resource_types = ['cpu', 'memory', 'disk']
    utilization = {
        resource: {
            pm: cp.intvar(0, pm_data[pm][f"capacity_{resource}"], name=f"{resource}_{pm}")
            for pm in pm_data
        }
        for resource in resource_types
    }

    gpu_utilization = {
        pm: cp.intvar(0, pm_data[pm].get("capacity_gpu", 0), name=f"gpu_{pm}")
        for pm in pm_data if "capacity_gpu" in pm_data[pm]
    }

    for i, pm in enumerate(pm_data):
        for resource in resource_types:
            resource_sum = sum(vm_data[vm][f"demand_{resource}"] * (vm_assignments[vm] == i + 1) 
                             for vm in vm_data)
            model += [utilization[resource][pm] == resource_sum]
            model += [resource_sum <= pm_data[pm][f"capacity_{resource}"]]

    for az in ["AZ1", "AZ2"]:
        high_priority_vms = [
            vm for vm in vm_data 
            if vm_data[vm]["availability_zone"] == az and vm_data[vm]["priority"] == 1
        ]
        if len(high_priority_vms) > 1:
            for i in range(len(high_priority_vms)):
                for j in range(i + 1, len(high_priority_vms)):
                    model += [vm_assignments[high_priority_vms[i]] != vm_assignments[high_priority_vms[j]]]

   
    for vm in vm_data:
        model += [vm_assignments[vm] >= 1]  
        model += [vm_assignments[vm] <= len(pm_data)]  

   
    for i, pm in enumerate(pm_data):
        if "capacity_gpu" in pm_data[pm]:
            pm_idx = i + 1

            for gpu_demand in set(vm_data[vm].get("demand_gpu", 0) for vm in vm_data if vm_data[vm].get("demand_gpu", 0) > 0):
                relevant_vms = [vm for vm in vm_data if vm_data[vm].get("demand_gpu", 0) == gpu_demand]
                
                if relevant_vms:

                    vms_with_this_gpu = cp.Count([vm_assignments[vm] for vm in relevant_vms], pm_idx)
                    model += [vms_with_this_gpu * gpu_demand <= pm_data[pm]["capacity_gpu"]]




    C_T = list(model.constraints)

    AV = absvar(2)
    lang = [
        AV[0] == AV[1],
        AV[0] != AV[1],
        AV[0] < AV[1],
        AV[0] > AV[1],
        AV[0] >= AV[1],
        AV[0] <= AV[1]
    ]

   
    parameters = {
        "num_vms": len(vm_data),
        "num_pms": len(pm_data),
        "gpu_requirements": {vm: vm_data[vm].get("demand_gpu", 0) for vm in vm_data}
    }

    variables = []

    for vm in sorted(vm_data.keys()):
        variables.append(vm_assignments[vm])

    for resource in sorted(resource_types):
        for pm in sorted(pm_data.keys()):
            variables.append(utilization[resource][pm])

    for vm in sorted(gpu_assignments.keys()):
        variables.append(gpu_assignments[vm])

    for pm in sorted(gpu_utilization.keys()):
        variables.append(gpu_utilization[pm])

    instance = ProblemInstance(
        variables=cp.cpm_array(variables), 
        params=parameters, 
        language=lang, 
        name="vm_allocation"
    )
    oracle = ConstraintOracle(C_T)





    return instance, oracle
