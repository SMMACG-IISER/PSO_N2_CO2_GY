import numpy as np
from numpy import random
import math
import scipy
from scipy import optimize
from scipy.spatial.transform import Rotation as R
from multiprocessing import Pool


input_sheet="GDY.txt"  #File containing the coordinates of sheet provided in a comma-separated format

s_atom=np.loadtxt(input_sheet,delimiter=',',usecols=(0),dtype=str,unpack=True) #Reading atoms
s_cord=np.loadtxt(input_sheet,delimiter=',',usecols=(1,2,3),dtype=float,unpack=True) #Reading coordinates:  A 3xN array representing the coordinates of the GY atoms
s_natoms=int(sum(1 for line in open(input_sheet))) #Counting number of lines/atoms


   
def charge_pos(coord):
    """Calculates the positions of four charges around a given N2 molecule.

    Args:
        coord (numpy.ndarray): A 2x3 array representing the coordinates of N2 molecule.

    Returns:
        numpy.ndarray: A 4x3 array containing the positions of the four charges.
    """

    # Calculate the vector between the two N atoms in N2 molecule
    r_vec = coord[1] - coord[0]

    # Calculate the magnitude and unit vector of the vector
    magnitude = np.linalg.norm(r_vec)
    unit_vec = r_vec / magnitude

    # Calculate the midpoint between the two N atoms in N2 molecule
    midpoint = (coord[1] + coord[0]) / 2

    # Calculate the positions of the four charges around the midpoint
    C1 = midpoint - 1.044 * unit_vec
    C2 = midpoint - 0.847 * unit_vec
    C3 = midpoint + 0.847 * unit_vec
    C4 = midpoint + 1.044 * unit_vec

    # Return the positions of the four charges
    return np.array([C1, C2, C3, C4])


def N2_N2(pos_int, n):
    """Calculates the total interaction energy between n N2 molecules.

    Args:
        pos_int (list): A list of n 2x3 arrays, each representing the coordinates of the two atoms in an N2 molecule.
        n (int): The number of N2 molecules.

    Returns:
        float: The total interaction energy between the N2 molecules.
    """

    # Buckingham-type potential parameters for noble gases
    A = 29995.9  # kcal/mol
    C6 = 392.274  # kcal/mol*angstrom^6
    alpha = 3.46136  # angstrom^-1
    Q_list = [-0.373, 0.373, 0.373, -0.373]  # Charges on N atoms

    E = 0  # Initialize total energy

    for i in range(n):
        # Calculate charge positions for the i-th N2 molecule
        ch_pos1 = charge_pos(pos_int[i])

        for j in range(i + 1, n):
            # Calculate charge positions for the j-th N2 molecule
            ch_pos2 = charge_pos(pos_int[j])

            # Calculate non-bonded interactions
            for k in range(2): # Iterate over atoms in the N2 molecule
                for u in range(2):
                    # Calculate distance between atoms k and u
                    r_s = np.linalg.norm(pos_int[i][k] - pos_int[j][u])

                    # Ensure distance is not too small to avoid division by zero
                    if r_s <= 1:
                        Ei_nonel = math.inf
                    else:
                        # Calculate non-bonded interaction energy by Buckingham-type potential
                        Ei_nonel = A * math.exp(-alpha * r_s) - C6 / (r_s**6)
                    E += Ei_nonel

            # Calculate electrostatic interactions
            for g in range(4): # Iterate over charges in the N2 molecule
                for m in range(4):
                    # Calculate distance between charges g and m
                    ch_dist = np.linalg.norm(ch_pos1[g] - ch_pos2[m])

                    # Ensure distance is not too small to avoid division by zero
                    if ch_dist <= 1:
                        Ei_el = math.inf
                    else:
                        # Calculate electrostatic interaction energy
                        Q = Q_list[g] * Q_list[m]
                        Ei_el = 332 * (Q / ch_dist)
                    E += Ei_el

    return E
    
    
def N2_GY(pos_int, n):
    """Calculates the total interaction energy between n N2 molecules and GY model.

    Args:
        pos_int (list): A list of n 2x3 arrays, each representing the coordinates of the two atoms in an N2 molecule.
        n (int): The number of N2 molecules.

    Returns:
        float: The total interaction energy between the N2 molecules and GY model.
    """

    # Lennard-Jones potential parameters for N2-GY interaction
    epsilon_sh = 0.0915734121  # kcal/mol
    rm_sh = 3.828  # angstrom
    beta = 7.5
    m = 6

    E = 0  # Initialize total energy

    for i in range(n):  # Iterate over N2 molecules
        for h in range(s_natoms):  # Iterate over GY atoms
            for d in range(2):  # Iterate over atoms in the N2 molecule
                # Calculate distance between the N2 atom and the GY atom
                r = np.linalg.norm(pos_int[i][d] - s_cord[:, h])

                # Calculate the Lennard-Jones potential energy
                n_r = beta + 4 * (r / rm_sh)**2
                Ei_sh1 = epsilon_sh * ((m / (n_r - m)) * (rm_sh / r)**n_r - (n_r / (n_r - m)) * (rm_sh / r)**m)
                E += Ei_sh1

    return E

    
def CO2_CO2(pos_int, m):
    """Calculates the total interaction energy between m CO2 molecules.

    Args:
        pos_int (list): A list of m 3x3 arrays, each representing the coordinates of the three atoms in a CO2 molecule in the order C,O,O.
        m (int): The number of CO2 molecules.

    Returns:
        float: The total interaction energy between the CO2 molecules.
    """

    # Lennard-Jones potential parameters for C-C, C-O, and O-O interactions
    epsilon_CC = 0.0535373  # kcal/mol
    epsilon_OO = 0.157027  # kcal/mol
    sigma_CC = 2.80  # angstrom
    sigma_OO = 3.05  # angstrom

    # Charges on C and O atoms
    q_C = 0.70
    q_O = -0.35

    E = 0  # Initialize total energy

    for i in range(m): 
        for j in range(i + 1, m):
            for k in range(3): # Iterate over atoms in the CO2 molecule
                for u in range(3):
                    # Calculate the distance between atoms k and u
                    r = np.linalg.norm(pos_int[i][k] - pos_int[j][u])

                    # Determine interaction type and parameters
                    if k == u and k == 0:  # C-C interaction
                        epsilon = epsilon_CC
                        sigma = sigma_CC
                        Q = q_C**2
                    elif k == u and (k == 1 or k == 2):  # O-O interaction
                        epsilon = epsilon_OO
                        sigma = sigma_OO
                        Q = q_O**2
                    elif k != u and (k == 0 or u == 0):  # C-O interaction
                        epsilon = (epsilon_CC * epsilon_OO)**0.5 # Applying Lorentz–Berthelot mixing rule
                        sigma = (sigma_OO + sigma_CC) * 0.5 # Applying Lorentz–Berthelot mixing rule
                        Q = q_C * q_O
                    else:  # O-O interaction
                        epsilon = epsilon_OO
                        sigma = sigma_OO
                        Q = q_O**2

                    # Calculate Lennard-Jones and Coulombic interaction energies
                    Ei_nonel = 4 * epsilon * ((sigma/ r)**12 - (sigma / r)**6)
                    Ei_el = 332 * (Q / r)
                    E += (Ei_nonel + Ei_el)

    return E

    
    
def CO2_GY(pos_int, n):
    """Calculates the total interaction energy between n CO2 molecules and GY model.

    Args:
        pos_int (list): A list of n 3x3 arrays, each representing the coordinates of the three atoms in a CO2 molecule in the order C,O,O.
        n (int): The number of CO2 molecules.

    Returns:
        float: The total interaction energy between the CO2 molecules and GY model.
    """

    # Improved Lennard-Jones potential parameters for CO2-GY interactions
    epsilon_C_GY = 0.0820494082  # kcal/mol (C(CO2)-C(GY) interaction)
    epsilon_O_GY = 0.115648618  # kcal/mol (O(CO2) -C(GY) interaction)
    rm_C_GY = 3.564  # angstrom (C(CO2)-C(GY) equilibrium distance)
    rm_O_GY = 3.676  # angstrom (O(CO2)-C(GY) equilibrium distance)
    beta = 6.75
    m=6

    E = 0  # Initialize total energy

    for i in range(n):  # Iterate over CO2 molecules
        for h in range(s_natoms):  # Iterate over GY atoms
            for d in range(3):  # Iterate over atoms in the CO2 molecule
                # Determine atom type and corresponding parameters
                if d == 0:  # Central carbon atom
                    epsilon_sh = epsilon_C_GY
                    rm_sh = rm_C_GY
                else:  # Oxygen atom
                    epsilon_sh = epsilon_O_GY
                    rm_sh = rm_O_GY

                # Calculate distance between the CO2 atom and the GY atom
                r = np.linalg.norm(pos_int[i][d] - s_cord[:, h])

                # Calculate the Improved Lennard-Jones potential energy
                n_r = beta + 4 * (r / rm_sh)**2
                Ei_sh2 = epsilon_sh * ((m / (n_r - m)) * (rm_sh / r)**n_r- (n_r / (n_r - m)) * (rm_sh / r)**m)
                E += Ei_sh2

    return E

    
def N2_CO2(pos_int1, pos_int2, n, k):
    """Calculates the total interaction energy between n N2 molecules and k CO2 molecules.

    Args:
        pos_int1 (list): A list of n 2x3 arrays, each representing the coordinates of the two atoms in an N2 molecule.
        pos_int2 (list): A list of k 3x3 arrays, each representing the coordinates of the three atoms in a CO2 molecule in the order C,O,O.
        n (int): The number of N2 molecules.
        k (int): The number of CO2 molecules.

    Returns:
        float: The total interaction energy between the N2 and CO2 molecules.
    """

    # Improved Lennard-Jones potential parameters for N2-CO2 interactions
    epsilon_N_C = 0.07840584  # kcal/mol (N-C interaction)
    epsilon_N_O = 0.1037724  # kcal/mol (N-O interaction)
    rm_N_C = 3.548  # angstrom (N-C equilibrium distance)
    rm_N_O = 3.699  # angstrom (N-O equilibrium distance)
    beta = 9
    m=6

    # Charges on N and O atoms
    Q_list = [-0.373, 0.373, 0.373, -0.373]  # Charges on N2 molecule
    q_C = 0.70
    q_O = -0.35

    E = 0  # Initialize total energy

    for i in range(n):  # Iterate over N2 molecules
        ch_pos = charge_pos(pos_int1[i])  # Calculate charge positions for the i-th N2 molecule

        for j in range(k):  # Iterate over CO2 molecules
            for w in range(2):  # Iterate over N atoms in the N2 molecule
                for d in range(3):  # Iterate over atoms in the CO2 molecule
                    # Determine atom type and corresponding parameters
                    if d == 0:  # Central carbon atom
                        epsilon_mix = epsilon_N_C
                        rm_mix = rm_N_C
                    else:  # Oxygen atom
                        epsilon_mix = epsilon_N_O
                        rm_mix = rm_N_O

                    # Calculate distance between the N atom and the CO2 atom
                    r_s = np.linalg.norm(pos_int1[i][w] - pos_int2[j][d])

                    # Calculate the improved Lennard-Jones potential energy
                    n_r = beta + 4 * (r_s / rm_mix)**2
                    Ei_mix_nonel = epsilon_mix * ((m / (n_r - m)) * (rm_mix / r_s)**n_r - (n_r / (n_r - m)) * (rm_mix / r_s)**m)
                    E += Ei_mix_nonel

            for u in range(4):  # Iterate over charges in the N2 molecule
                for v in range(3):  # Iterate over atoms in the CO2 molecule
                    # Calculate distance between the charge and the CO2 atom
                    r_q = np.linalg.norm(ch_pos[u] - pos_int2[j][v])

                    # Ensure distance is not too small to avoid division by zero
                    if r_q <= 1:
                        Ei_mix_el = math.inf
                    else:
                        # Determine charge product
                        if v == 0:
                            Q = q_C * Q_list[u]
                        else:
                            Q = q_O * Q_list[u]

                        # Calculate the Coulombic interaction energy
                        Ei_mix_el = 332* (Q / r_q)
                        E += Ei_mix_el

    return E

 
 
def total_energy(pos_int1, pos_int2, n, m):
    """Calculates the total interaction energy between a system of n N2 molecules and m CO2 molecules.

    Args:
        pos_int1 (list): A list of n 2x3 arrays, each representing the coordinates of the two atoms in an N2 molecule.
        pos_int2 (list): A list of m 3x3 arrays, each representing the coordinates of the three atoms in a CO2 molecule.
        n (int): The number of N2 molecules.
        m (int): The number of CO2 molecules.

    Returns:
        float: The total interaction energy of the system.
    """

    # Calculate individual energy contributions
    E_N2_N2 = N2_N2(pos_int1, n)  # N2-N2 interactions
    E_N2_GY = N2_GY(pos_int1, n)  # N2-GY interactions
    E_CO2_CO2 = CO2_CO2(pos_int2, m)  # CO2-CO2 interactions
    E_CO2_GY = CO2_GY(pos_int2, m)  # CO2-GY interactions
    E_N2_CO2 = N2_CO2(pos_int1, pos_int2, n, m)  # N2-CO2 interactions

    # Calculate total energy
    E = E_N2_N2 + E_N2_GY + E_CO2_CO2 + E_CO2_GY + E_N2_CO2

    return E


def conversion_N2(pos, eul_angl, n):
    """Converts N2 molecule centre of mass positions and orientations to Cartesian coordinates.

    Args:
        pos (numpy.ndarray): An n x 3 array representing the position of the center of mass of each N2 molecule.
        eul_angl (numpy.ndarray): An n x 3 array representing the Euler angles (ZYX convention) for the orientation of each N2 molecule.
        n (int): The number of N2 molecules.

    Returns:
        numpy.ndarray: An n x 2 x 3 array representing the Cartesian coordinates of the two atoms in each N2 molecule.
    """

    # Define the relative coordinates of the two N atoms in an N2 molecule
    N2_coord = np.array([[0.0, 0.0, -0.547], [0.0, 0.0, 0.547]])

    # Initialize an array to store the Cartesian coordinates of the N atoms
    pos_int = np.empty((n, 2, 3))

    for i in range(n):
        # Create a rotation matrix from the Euler angles
        r = R.from_euler('ZYX', eul_angl[i], degrees=False)
        R_M = r.as_matrix()

        # Calculate the inverse of the rotation matrix
        RM_inv = np.linalg.inv(R_M)

        for k in range(2):
            # Apply the rotation and translation to the N2 coordinates
            product = np.matmul(RM_inv, N2_coord[k])
            pos_int[i][k] = pos[i] + product

    return pos_int


def conversion_CO2(pos, eul_angl, n):
    """Converts CO2 molecule centre of mass positions and orientations to Cartesian coordinates.

    Args:
        pos (numpy.ndarray): An n x 3 array representing the position of the center of mass of each CO2 molecule.
        eul_angl (numpy.ndarray): An n x 3 array representing the Euler angles (ZYX convention) for the orientation of each CO2 molecule.
        n (int): The number of CO2 molecules.

    Returns:
        numpy.ndarray: An n x 3 x 3 array representing the Cartesian coordinates of the three atoms in each CO2 molecule.
    """

    # Define the relative coordinates of the three atoms in a CO2 molecule
    CO2_coord = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.16037], [0.0, 0.0, -1.16037]])

    # Initialize an array to store the Cartesian coordinates of the CO2 atoms
    pos_int = np.empty((n, 3, 3))

    for i in range(n):
        # Create a rotation matrix from the Euler angles
        r = R.from_euler('ZYX', eul_angl[i], degrees=False)
        R_M = r.as_matrix()

        # Calculate the inverse of the rotation matrix
        RM_inv = np.linalg.inv(R_M)

        for k in range(3):
            # Apply the rotation and translation to the CO2 coordinates
            product = np.matmul(RM_inv, CO2_coord[k])
            pos_int[i][k] = pos[i] + product

    return pos_int



def lbfgs_function(np_array, n, m):
    """Calculates the total interaction energy of a system containing n N2 molecules and m CO2 molecules.

    Args:
        np_array (numpy.ndarray): A 1D array containing the centre of mass positions and orientations of the molecules.
        n (int): The number of N2 molecules.
        m (int): The number of CO2 molecules.

    Returns:
        float: The total interaction energy of the system.
    """

    # Split the input array into subarrays for N2 and CO2 molecules
    new_array = np.array(np.split(np_array, [n * 3, n * 3 + m * 3, n * 3 + m * 3 + n * 3, n * 3 + m * 3 + n * 3 + m * 3]))

    # Extract positions and orientations for N2 and CO2 molecules
    pos1 = np.array(np.split(new_array[0], n))
    eul_angl1 = np.array(np.split(new_array[2], n))
    pos2 = np.array(np.split(new_array[1], m))
    eul_angl2 = np.array(np.split(new_array[3], m))

    # Convert positions and orientations to Cartesian coordinates
    pos_int1 = conversion_N2(pos1, eul_angl1, n)
    pos_int2 = conversion_CO2(pos2, eul_angl2, m)

    # Calculate the total interaction energy
    E = total_energy(pos_int1, pos_int2, n, m)

    return E
 
    
        
# Define the search space boundaries
x_min = -18.47184; x_max = 18.47184
y_min = -21.32945; y_max = 21.32945
z_min = 0; z_max = 5 
φ_min = 0; φ_max = (2*np.pi)
θ_min = 0; θ_max = np.pi
Ψ_min = 0; Ψ_max = (2*np.pi)


# Define the maximum velocity limits in each dimension
velmax_x = 0.5*(x_max-x_min)
velmax_y = 0.5*(y_max-y_min)
velmax_z = 0.5*(z_max-z_min)
velmax_φ = 0.5*(φ_max-φ_min)
velmax_θ = 0.5*(θ_max-θ_min)
velmax_Ψ = 0.5*(Ψ_max-Ψ_min) 


# PSO algorithm parameters
pop = 2000  # Population size
maxtrial = 25  # Number of trials
maxit = 1000  # Maximum number of iterations
c1 = 2.05  # Cognitive coefficient
c2 = 2.05  # Social coefficient
chi = 0.729  # Constriction factor


def PSO_init(n):
    """Initializes the particle swarm optimization (PSO) algorithm.

    Args:
        n (int): The number of particles in the swarm.

    Returns:
        tuple: A tuple containing:
            - pos (numpy.ndarray): An n x 3 array representing the initial positions of the particles corresponding to centre of mass of the molecules.
            - eul_angl (numpy.ndarray): An n x 3 array representing the initial positions of the particles corresponding to Euler angles of the molecules.
            - vel (numpy.ndarray): An n x 3 array representing the initial velocities of the particles corresponding to centre of mass of the molecules.
            - vel_angl (numpy.ndarray): An n x 3 array representing the initial velocities of the particles corresponding to Euler angles of the molecules.
    """

    np.random.seed()  # Set a random seed for reproducibility

    # Initialize particle positions and orientations randomly within the specified boundaries
    x = np.random.uniform(x_min, x_max, n)
    y = np.random.uniform(y_min, y_max, n)
    z = np.random.uniform(z_min, z_max, n)
    φ = np.random.uniform(φ_min, φ_max, n)
    θ = np.random.uniform(θ_min, θ_max, n)
    Ψ = np.random.uniform(Ψ_min, Ψ_max, n)

    # Combine position and orientation coordinates into matrices
    pos = np.column_stack((x, y, z))
    eul_angl = np.column_stack((φ, θ, Ψ))

    # Initialize particle velocities to zero
    vel_x = np.zeros(n, dtype=float)
    vel_y = np.zeros(n, dtype=float)
    vel_z = np.zeros(n, dtype=float)
    vel_φ = np.zeros(n, dtype=float)
    vel_θ = np.zeros(n, dtype=float)
    vel_Ψ = np.zeros(n, dtype=float)

    # Combine velocity components into matrices
    vel = np.column_stack((vel_x, vel_y, vel_z))
    vel_angl = np.column_stack((vel_φ, vel_θ, vel_Ψ))

    return pos, eul_angl, vel, vel_angl

    
    
def PSO(velocity, position, pbest_position, gbest_pos, vel_angle, eul_angle, pbest_angle, gbest_angle, n, velmax, velmin, velmax_agl, velmin_agl):
    """
    Perform one iteration of a single particle (representing n adsorbed CO2 or N2 molecules) in the Particle Swarm Optimization (PSO) algorithm.

    Parameters:
    -----------
    velocity : ndarray
        Current velocity of the particle in the search space (n x 3 array for x, y, z components).
    position : ndarray
        Current position of the particle in the search space (n x 3 array for x, y, z components).
    pbest_position : ndarray
        Best known position of the particle (personal best) (n x 3 array for x, y, z components).
    gbest_pos : ndarray
        Global best known position among all particles (n x 3 array for x, y, z components).
    vel_angle : ndarray
        Current velocity of the particle in the search space (n x 3 array for φ, θ, Ψ components).
    eul_angle : ndarray
        Current position of the particle in the search space (n x 3 array for φ, θ, Ψ components).
    pbest_angle : ndarray
        Best known position of the particle (personal best) (n x 3 array for φ, θ, Ψ components).
    gbest_angle : ndarray
        Global best known position among all particles (n x 3 array for φ, θ, Ψ components).
    n : int
        Number of molecules (CO2 or N2).
    velmax : ndarray
        Maximum allowable velocity for each component (1 x 3 array for x, y, z components).
    velmin : ndarray
        Minimum allowable velocity for each component (1 x 3 array for x, y, z components).
    velmax_agl : ndarray
        Maximum allowable velocity for each component (1 x 3 array for φ, θ, Ψ components).
    velmin_agl : ndarray
        Minimum allowable velocity for each component (1 x 3 array for φ, θ, Ψ components).

    Returns:
    --------
    position : ndarray
        Updated positions of the particle for x, y, z components after one PSO iteration.
    eul_angle : ndarray
        Updated positions of the particle for φ, θ, Ψ components after one PSO iteration.
    velocity : ndarray
        Updated velocities of the particle for x, y, z components after one PSO iteration.
    vel_angle : ndarray
        Updated velocities of the particle for φ, θ, Ψ components after one PSO iteration.
    """

    # Random numbers for stochastic component of velocity update
    r1 = np.random.rand(n, 3)  # Random numbers for position update (x, y, z)
    r2 = np.random.rand(n, 3)
    r11 = np.random.rand(n, 3)  # Random numbers for angular update (φ, θ, Ψ)
    r22 = np.random.rand(n, 3)

    # Update velocity based on personal best, global best, and inertia
    velocity = chi * (velocity + c1 * r1 * (pbest_position - position) + c2 * r2 * (gbest_pos - position))
    vel_angle = chi * (vel_angle + c1 * r11 * (pbest_angle - eul_angle) + c2 * r22 * (gbest_angle - eul_angle))

    # Velocity clamping to ensure velocities remain within specified limits
    for i in range(n):
        velocity[i] = np.minimum(np.maximum(velocity[i], velmin), velmax)
        vel_angle[i] = np.minimum(np.maximum(vel_angle[i], velmin_agl), velmax_agl)

    # Update positions based on the new velocities
    position = position + velocity
    eul_angle = eul_angle + vel_angle

    # Position boundary enforcement to ensure positions corresponding to centre of mass coordinates remain within the search space
    for m in range(n):
        min_p = np.array([x_min + m * 0.00001, y_min + m * 0.00001, z_min + m * 0.00001])
        max_p = np.array([x_max + m * 0.00001, y_max + m * 0.00001, z_max + m * 0.00001])
        position[m] = np.minimum(np.maximum(position[m], min_p), max_p)

    # Boundary enforcement to ensure positions corresponding to Euler angles remain within specified limits
    min_agl = np.array([φ_min, θ_min, Ψ_min])
    max_agl = np.array([φ_max, θ_max, Ψ_max])
    for y in range(n):
        eul_angle[y] = np.minimum(np.maximum(eul_angle[y], min_agl), max_agl)
        
    # Return the updated positions, Euler angles, velocities, and angular velocities
    return position, eul_angle, velocity, vel_angle

# Specify the number of molecules to study
# Specify n/m = 0 for the case of unary cluster adsorption
m = 2 # Number of CO2 molecules
n = 2 # Number of N2 molecules

def parallel(a, b):
    """
    Executes a parallel particle swarm optimization (PSO) algorithm for 
    optimizing the positions and orientations of N2 and CO2 molecules 
    to minimize their total energy. This is followed by a local optimization 
    using the L-BFGS-B algorithm.

    Parameters:
    a (int): Trial identifier.
    b (int): Second parameter, but not used in the function.

    Returns:
    tuple: Contains the optimized energy and the final Cartesian coordinates of
           N2 and CO2 molecules.
    """

    trial = a

    # Open a file to log PSO results for N2 and CO2 adsorption on GY
    f = open("GY-N2-CO2_PSO", "a")
    f.write("\ntrial: %s" % trial)        
    f.write("\nNo. of N2 molecules: %s" % n)
    f.write("\nNo. of CO2 molecules: %s" % m)

    # Initialize variables for PSO
    gbest_energy = math.inf  # Global best energy initialized to infinity
    positions1, eul_angles1 = [], []  # Positions corresponding to center of mass positions and Euler angles for N2 molecules
    velocities1, vel_angles1 = [], []  # Velocity corresponding to center of mass positions and Euler angles for N2 molecules
    positions2, eul_angles2 = [], []  # Position corresponding to center of mass positions and Euler angles for CO2 molecules
    velocities2, vel_angles2 = [], []  # Velocity corresponding to center of mass positions and Euler angles for CO2 molecules
    pbest_energies = np.zeros(pop)  # Best energies for each particle

    # Initialization of particles
    for i in range(pop):
        # N2 molecules
        N2_init = PSO_init(n)
        velocities1.append(N2_init[2])
        vel_angles1.append(N2_init[3])
        pos1 = N2_init[0]
        eul_angl1 = N2_init[1]
        pos_int1 = conversion_N2(pos1, eul_angl1, n)

        # CO2 molecules
        CO2_init = PSO_init(m)
        velocities2.append(CO2_init[2])
        vel_angles2.append(CO2_init[3])
        pos2 = CO2_init[0]
        eul_angl2 = CO2_init[1]
        pos_int2 = conversion_CO2(pos2, eul_angl2, m)
        
        # Evaluate the total energy for the current positions
        energy = total_energy(pos_int1, pos_int2, n, m)

        # Store initial positions corresponding to center of mass positions and Euler angles
        positions1.append(pos1)
        positions2.append(pos2)
        eul_angles1.append(eul_angl1)
        eul_angles2.append(eul_angl2)

        # Update personal best (pbest) for each particle
        pbest_energies[i] = energy
        pbest_positions1 = np.copy(positions1)
        pbest_angles1 = np.copy(eul_angles1)
        pbest_positions2 = np.copy(positions2)
        pbest_angles2 = np.copy(eul_angles2)

        # Update global best (gbest) if current energy is lower
        if energy < gbest_energy:
            gbest_energy = energy
            gbest_pos1 = pos1
            gbest_angle1 = eul_angl1
            gbest_pos2 = pos2
            gbest_angle2 = eul_angl2
                
    # PSO iterations
    for t in range(maxit):
        # Linearly reduce the velocity limits over time
        velmax = np.array([velmax_x, velmax_y, velmax_z]) * (t / maxit)
        velmin = np.array([-velmax_x, -velmax_y, -velmax_z]) * (t / maxit)
        velmax_agl = np.array([velmax_φ, velmax_θ, velmax_Ψ]) * (t / maxit)
        velmin_agl = np.array([-velmax_φ, -velmax_θ, -velmax_Ψ]) * (t / maxit)

        # Update particles
        for k in range(pop):
            # Update position and velocity corresponding to center of mass positions and Euler angles of N2 molecules
            pos_vel_k_1 = PSO(velocities1[k], positions1[k], pbest_positions1[k], gbest_pos1, vel_angles1[k], eul_angles1[k], pbest_angles1[k], gbest_angle1, n, velmax, velmin, velmax_agl, velmin_agl)

            # Update position and velocity corresponding to center of mass positions and Euler angles of CO2 molecules
            pos_vel_k_2 = PSO(velocities2[k], positions2[k], pbest_positions2[k], gbest_pos2, vel_angles2[k], eul_angles2[k], pbest_angles2[k], gbest_angle2, m, velmax, velmin, velmax_agl, velmin_agl)

            # Convert center of mass positions and Euler angles to Cartesian coordinates
            pos_int1 = conversion_N2(pos_vel_k_1[0], pos_vel_k_1[1], n)
            pos_int2 = conversion_CO2(pos_vel_k_2[0], pos_vel_k_2[1], m)
            
            # Calculate total energy for current positions
            energy = total_energy(pos_int1, pos_int2, n, m)

            # Update positions and velocities
            positions1[k] = pos_vel_k_1[0]
            eul_angles1[k] = pos_vel_k_1[1]
            velocities1[k] = pos_vel_k_1[2]
            vel_angles1[k] = pos_vel_k_1[3]
            positions2[k] = pos_vel_k_2[0]
            eul_angles2[k] = pos_vel_k_2[1]
            velocities2[k] = pos_vel_k_2[2]
            vel_angles2[k] = pos_vel_k_2[3]           

            # Update personal best if current energy is lower
            if energy < pbest_energies[k]:
                pbest_energies[k] = energy
                pbest_positions1[k] = pos_vel_k_1[0]
                pbest_angles1[k] = pos_vel_k_1[1]
                pbest_positions2[k] = pos_vel_k_2[0]
                pbest_angles2[k] = pos_vel_k_2[1]                

            # Update global best if current personal best is lower
            if pbest_energies[k] < gbest_energy:
                gbest_energy = pbest_energies[k]
                gbest_pos1 = pbest_positions1[k]
                gbest_angle1 = pbest_angles1[k]
                gbest_pos2 = pbest_positions2[k]
                gbest_angle2 = pbest_angles2[k]

    # Convert the global best positions to Cartesian coordinates
    gbest_pos_int1 = conversion_N2(gbest_pos1, gbest_angle1, n)
    gbest_pos_int2 = conversion_CO2(gbest_pos2, gbest_angle2, m)

    # Log the global best energy
    f.write("\nGlobal Best Energy: %s" % gbest_energy)
    
    # Log the final positions of N2 and CO2 molecules in the output file
    for i in range(n):
        B = np.array(gbest_pos_int1[i])
        f.write('\n N      %s      %s      %s' % (B[0][0], B[0][1], B[0][2]))
        f.write('\n N      %s      %s      %s' % (B[1][0], B[1][1], B[1][2])) 

    for j in range(m):
        B = np.array(gbest_pos_int2[j])
        f.write('\n C      %s      %s      %s' % (B[0][0], B[0][1], B[0][2]))
        f.write('\n O      %s      %s      %s' % (B[1][0], B[1][1], B[1][2]))
        f.write('\n O      %s      %s      %s' % (B[2][0], B[2][1], B[2][2]))

    f.write("\n************************************************************************************")
    f.close()
    
    ###############################
    # Local Optimization using L-BFGS-B
    
    # Flatten and concatenate global best positions to a 1D array for local optimization
    array1 = np.ndarray.flatten(gbest_pos1)
    array2 = np.ndarray.flatten(gbest_pos2)
    array3 = np.ndarray.flatten(gbest_angle1)
    array4 = np.ndarray.flatten(gbest_angle2)
    np_array = np.concatenate((array1, array2, array3, array4))

    # Open a new file to log results of local optimization
    g = open("GY-N2-CO2_loc", "a")
    g.write("\ntrial: %s" % trial)        
    g.write("\nNo. of N2 molecules: %s" % n)
    g.write("\nNo. of CO2 molecules: %s" % m)
    
    # Perform the local optimization
    res = scipy.optimize.minimize(lbfgs_function, np_array, args=(n, m), method='L-BFGS-B', options={'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 100000, 'maxiter': 100000, 'iprint': -1, 'maxls': 20})

    # Extract optimized values
    r_array = res.x # optimized array
    opt_energy = res.fun

    # Log the optimized energy
    g.write("\nGlobal Best Energy: %s" % opt_energy)

    # Split the optimized array into center of mass positions and Euler angles for N2 and CO2
    new_array1 = np.split(r_array, [n*3, n*3+m*3, n*3+m*3+n*3, n*3+m*3+n*3+m*3])        
    pos_final1 = np.array(np.split(new_array1[0], n))
    eul_angl_final1 = np.array(np.split(new_array1[2], n))
    pos_final2 = np.array(np.split(new_array1[1], m))
    eul_angl_final2 = np.array(np.split(new_array1[3], m))
    
    # Convert the final optimized positions to Cartesian coordinates
    pos_int_final1 = conversion_N2(pos_final1, eul_angl_final1, n)
    pos_int_final2 = conversion_CO2(pos_final2, eul_angl_final2, m)
    
    # Calculate individual energy components
    E_N2_N2 = N2_N2(pos_int_final1, n)
    E_N2_GY = N2_GY(pos_int_final1, n)
    E_CO2_CO2 = CO2_CO2(pos_int_final2, m)
    E_CO2_GY = CO2_GY(pos_int_final2, m)
    E_N2_CO2 = N2_CO2(pos_int_final1, pos_int_final2, n, m)

    # Log individual energy components
    g.write("\nN2-N2 Energy: %s" % E_N2_N2)
    g.write("\nN2-GY Energy: %s" % E_N2_GY)
    g.write("\nCO2-CO2 Energy: %s" % E_CO2_CO2)
    g.write("\nCO2-GY Energy: %s" % E_CO2_GY) 
    g.write("\nCO2-N2 Energy: %s" % E_N2_CO2)
        
    # Log the final positions of N2 and CO2 molecules in the output file
    for i in range(n):
        B = np.array(pos_int_final1[i])
        g.write('\n N      %s      %s      %s' % (B[0][0], B[0][1], B[0][2]))
        g.write('\n N      %s      %s      %s' % (B[1][0], B[1][1], B[1][2]))                              
    for j in range(m):
        B = np.array(pos_int_final2[j])
        g.write('\n C      %s      %s      %s' % (B[0][0], B[0][1], B[0][2]))
        g.write('\n O      %s      %s      %s' % (B[1][0], B[1][1], B[1][2])) 
        g.write('\n O      %s      %s      %s' % (B[2][0], B[2][1], B[2][2]))                      
        
    g.write("\n************************************************************************************")
    g.close()

    # Return the optimized energy and final Cartesian coordinates for N2 and CO2
    return opt_energy, pos_int_final1, pos_int_final2
    
        
# Create an array of tuples, where each tuple is (x, x+1), for x in the range from 0 to maxtrial
array_of_numbers = [(x, x + 1) for x in range(0, maxtrial)]

# Initialize a pool of workers, specifying 25 parallel processes
p = Pool(25)

# Use starmap to apply the 'parallel' function to each tuple in 'array_of_numbers'
# The result is a list of outputs from the 'parallel' function for each tuple
output = p.starmap(parallel, array_of_numbers)

# Initialize an empty list to store the energies from each run of the 'parallel' function
Energies = []
for i in range(maxtrial):
    # Append the first element of each output tuple (the energy) to the Energies list
    Energies.append(output[i][0])

# Convert the list of energies to a NumPy array for easier processing
Energies = np.array(Energies)

# Find the minimum energy in the array
gbest_min = np.amin(Energies)

# Find the index of the minimum energy in the array
# The result is an array containing the index of the minimum energy
result = np.array(np.where(Energies == np.amin(Energies)))

# Extract the positions corresponding to the minimum energy from the output
gbest_min_pos1 = output[result[0][0]][1]
gbest_min_pos2 = output[result[0][0]][2]

# Open a file to log the global best results corresponding to the minimum of 25 trials
h = open("GY-N2-CO2_PSO_min", "a")
h.write("\nZ maximum: %s" % z_max)
h.write("\nSwarmsize: %s" % pop)
h.write("\nNo. of N2 molecules: %s" % n)
h.write("\nNo. of CO2 molecules: %s" % m)
h.write("\nGlobal best energy: %s" % gbest_min)

# Log the positions of the N2 molecules with the minimum energy
for i in range(n):
    B = np.array(gbest_min_pos1[i])
    h.write('\n N      %s      %s      %s' % (B[0][0], B[0][1], B[0][2]))
    h.write('\n N      %s      %s      %s' % (B[1][0], B[1][1], B[1][2]))

# Log the positions of the CO2 molecules with the minimum energy
for i in range(m):
    B = np.array(gbest_min_pos2[i])
    h.write('\n C      %s      %s      %s' % (B[0][0], B[0][1], B[0][2]))
    h.write('\n O      %s      %s      %s' % (B[1][0], B[1][1], B[1][2]))
    h.write('\n O      %s      %s      %s' % (B[2][0], B[2][1], B[2][2]))

# Open the files used for logging PSO and local optimization results to append a separator line
f = open("GY-N2-CO2_PSO", "a")  
g = open("GY-N2-CO2_loc", "a")

# Write a separator line in both files to distinguish different runs
f.write("\n```````````````````````````````````````````````````````````````````````````````````````````````")
g.write("\n```````````````````````````````````````````````````````````````````````````````````````````````")

# Close the files to ensure data is written and resources are freed
f.close()
g.close()

# Write a separator line in the global best results file and close it
h.write("\n```````````````````````````````````````````````````````````````````````````````````````````````")
h.close()

		

		
		

		
