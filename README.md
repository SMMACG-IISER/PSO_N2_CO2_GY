# PSO for N2/CO2 Adsortion on Graphyne (GY)

The Python code executing global optimization of adsorbed molecular cluster configurations using particle swarm optimization.

## Prerequisites

- Python 3.x
- Required Python libraries: `numpy`, `scipy`, `math`, `multiprocessing`

Install the required libraries using:
```bash
pip install numpy scipy multiprocessing
```

## Input File

- **GY.txt**: This file should contain the atomic coordinates of the Graphyne sheet in a comma-separated format. Each line should represent an atom with its corresponding x, y, z coordinates.

Example format:
```
C, 0.0000, 0.0000, 0.0000
C, 1.2300, 0.0000, 0.0000
C, 2.4600, 0.0000, 0.0000
```

## How to Run the Script

The script reads the atomic coordinates from `GY.txt` using `numpy.loadtxt()` to process the atomic species and coordinates. Ensure that your `GY.txt` file is correctly formatted and available in the script's directory.

To execute the script, simply run:

```bash
python GY_N2_CO2_PSO.py
```
