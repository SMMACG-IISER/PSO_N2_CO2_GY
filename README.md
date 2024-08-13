# PSO for N2/CO2 Adsortion on Graphyne
The Python code executing global optimization of adsorbed molecular cluster configurations using particle swarm optimization. 

Requirements:
1. The code requires the Numpy, SciPy and Multiprocessing libraries to function.
2. The user has to provide a .txt file containing the Cartesian coordinates of the Graphyne (GY) carbons in comma separated format (CSV).
3. The code returns three files:
   (a) A text file containing the PSO output of all the trials.
   (b) A text file containing the PSO + L-BFGS-B output of all the trials.
   (c) A text file containing the best PSO + L-BFGS-B output among all the trials.
