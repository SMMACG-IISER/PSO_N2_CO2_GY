# PSO for N2/CO2 Adsorption on Graphyne (GY)

The Python code for executing global optimization of bare and adsorbed molecular cluster configurations using particle swarm optimization.

## Prerequisites

- Python 3.x
- Required Python libraries: `numpy`, `scipy`, `math`, `multiprocessing`

Install the required libraries using:
```bash
pip install numpy scipy
```
## Input File

- **GY.txt**: This file should contain the atomic coordinates of the GY sheet in a comma-separated format. Each line should represent an atom with its corresponding x, y, z coordinates. 

Example format:
```
C, 0.0000, 0.0000, 0.0000
C, 1.2300, 0.0000, 0.0000
C, 2.4600, 0.0000, 0.0000
```
- **GY-COORDS/**: This folder contains the coordinates of γ-GY used in the study.
## Variables to Specify

Before running the script, specify the following variables in the script:

- **Number of Molecules**:
  - `m`: Number of CO2 molecules
  - `n`: Number of N2 molecules
  - Example values:
    ```python
    m = 2  # Number of CO2 molecules
    n = 2  # Number of N2 molecules
    ```

- **PSO Algorithm Parameters**:
  - `pop`: Population size
  - `maxtrial`: Number of trials
  - `maxit`: Maximum number of iterations
  - Example values:
    ```python
    pop = 2000      # Population size
    maxtrial = 25   # Number of trials
    maxit = 1000    # Maximum number of iterations
    ```

## How to Run the Script

The script reads the atomic coordinates from `GY.txt` using `numpy.loadtxt()` to process the atomic species and coordinates. Ensure that your `GY.txt` file is correctly formatted and available in the script's directory. Update the script with the appropriate values for the number of molecules and PSO parameters as outlined above.

To execute the script, simply run from the terminal:

```bash
python GY_N2_CO2_PSO.py
```
