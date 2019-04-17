This project has been realised by Juliette Rengot and Yonatan Deloro, from November, 2018 to January, 2019, within the scope of the Reinforcement Learning class of the MVA Master.

This project explores reinforcement learning problems where sub-goals can be identified to speed up the learning of the main task. It aims at beating the performances of the Q-learning algorithm on the flat MDP formulation of the Taxi Domain with two different approaches : Macro Q-Learning with Diverse-Density-based options and HEXQ algorithm. 

The folder "doc" contains some relevant articles related with this project.
A complete description of the project is available in the document project_report.pdf

This repository contains the implementation of three different appoaches :

### Definition of the Environment ###
* gridworld.py      -> definition of an environment for the original taxi problem, and for the simplified taxi problem (cf report for description)
* gridrender.py     -> visualisation for such environments

### "Flat Q-learning" : an approach without any sub-goal discovery strategy ###
* Qlearning.py      -> classic Qlearning implementation
* main_Qlearning.py -> running the Qlearning on the flat MDP formulation over the Original/Simplified Taxi Domain, and testing

###  DDMQ : MacroQlearning with Diverse Density-based options ###
(main contributor : Yonatan Deloro)

* gridworld.py      -> definition of an environment for the original taxi problem, and for the simplified taxi problem (cf report for description)
* gridrender.py     -> visualisation for such environments
* Qlearning.py      -> classic Qlearning implementation
* MacroQLearning.py -> MacroQLearning implementation
* DiverseDensity.py -> Functions useful for Diverse Density based Subgoal Discovery over the Simplified Taxi Domain
* main_DDMQ.py -> running the MacroQlearning with Diverse Density-based options over the Simplified Taxi Domain, and comparing to the Qlearning on the flat MDP formulation

### HEXQ Algorithm ###
(main contributor : Juliette Rengot)

* The HEXQ code is availabled at https://github.com/rengotj/sub_goal_RL
