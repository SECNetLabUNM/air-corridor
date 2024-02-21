# Transformer-based Multi-agent Reinforcement Learning for Multiple Unmanned Aerial Vehicle Coordination in Air Corridors
## Animation on cylinder-torus-torus-cylinder


![cylinder-torus-torus-cylinder.gif](test%20and%20visualization%2Fmd_present%2Fcylinder-torus-torus-cylinder.gif)

[D3MOVE_v4.py](test%20and%20visualization%2FD3MOVE_v4.py) for visualization of UAVs coordination in air corridors. 

##  Air Corridor Modeling
UAVs need to traverse several air corridors to reach their destinations.
Air corridors are modelled as cylinder and partial torus.
### Cylinder and Torus
![Air_corridor.jpg](test%20and%20visualization%2Fmd_present%2FAir_corridor.jpg)
[corridor.py](air_corridor%2Fd3%2Fcorridor%2Fcorridor.py)
## RL Training
### Network Structure
- H(), embedding layer, normalizes the input values and standardize the input dimensions.
- G(), transformer layer, deals with stochastic neighbors information
- F(), actor-critic network combined.
![TransRL.jpg](test%20and%20visualization%2Fmd_present%2FTransRL.jpg)


![network function.png](test%20and%20visualization%2Fmd_present%2Fnetwork%20function.png)

### Training
related package can be found in [environment.yml](environment.yml)

[main.py](rl_multi_3d_trans%2Fmain.py)


