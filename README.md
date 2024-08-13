# PSO for $ \text{N}_2 $ / $ \text{CO}_2 $ Adsortion on Graphyne

The Python code executing global optimization of adsorbed molecular cluster configurations using particle swarm optimization.

## Prerequisites

- Python 3.x
- Required Python libraries: `numpy`, `scipy`, `math`, `multiprocessing`

Install the required libraries using:
```bash
pip install numpy scipy multiprocessing
```

## Input File

- **GDY.txt**: This file should contain the atomic coordinates of the Graphyne sheet in a comma-separated format. Each line should represent an atom with its corresponding x, y, z coordinates.

Example format:
```
C, 0.0000, 0.0000, 0.0000
C, 1.2300, 0.0000, 0.0000
C, 2.4600, 0.0000, 0.0000
```

## How to Run the Script

The script reads the atomic coordinates from `GDY.txt` using `numpy.loadtxt()` to process the atomic species and coordinates. Ensure that your `GDY.txt` file is correctly formatted and available in the script's directory.

To execute the script, simply run:

```bash
python GY_N2_CO2_PSO.py
```
