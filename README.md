# MARL-HeteroUAVs
Code for the project: The dynamic collaborative awareness and path planning of Heterogeneous UAV swarm under telecommunication limitations based on MARL.  

This is an ongoing research project. Code and experimental results are **subject to change**.

## Current progress

- We have established a 2.5D heterogeneous UAV simulation environment based on the second-order integral **dynamic physics engine**.

- We have respectively implemented the **MAPPO** and the **MASAC** algorithm, both of which incorporated the **multi-head attention** mechanism.



# Project structure

```
MARL-HeteroUAVSwarm/
├── coverage_calcukate_test/    # Initial coverage calculation code
│	├── coverage_calculate.py
│	├── position_and_distance.py
│	├── main.py
│	├── READMA.md
├── Singal_UAV 					# A test singal-uav simulation enviroment.
│	├── main.py
│	├── env/
│	│	├── env.py
│	├── requirements.txt
│	├── README.md
├── marl_uav.py  				# Main code of our project
	├── env/
	│	├── config.py
	│	├── core.py
	│	├── cover_scan.py
	│	├── test_env.py
	│	├── scenarios/
	│		├── uav_mission.py
	├── algorithms/
	│	├── modules.py
    │   ├── masac.py
    │   ├── mappo.py
    │   ├── buffer.py
    ├── scripts/
    │	├── wariting to be developed……
    ├── results/
    │	├── wariting to be developed……
    ├── requirements.txt
```



## Installation

First, create a conda environment using Python=3.9

```
conda create -n marl_uav python=3.9
conda activate marl_uav
```

Then, install the dependencies

```
pip install -r requirements
```

