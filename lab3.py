# CLIMATE 410 Lab 3 = Energy Balance Atmosphere Model
# Due Date = 10/17/24
# Name = Lars Andersland

'''
A set of tools and routines for solving the N-layer atmosphere energy
balance problem and perform some useful analysis.

'''

import numpy as np
import matplotlib.pyplot as plt

#  Define some useful constants here.
sigma = 5.67E-8  # Steffan-Boltzman constant.

def n_layer_atmos(N, epsilon, S0=1350, albedo=0.33, nuclear_winter = False, debug=False):
    '''
    NOTE: the vast majority of this function was written by Dr. Welling in lecture

    Solve the n-layer atmosphere problem and return temperature at each layer.

    Parameters
    ----------
    N : int
        Set the number of layers.
    epsilon : float, default=1.0
        Set the emissivity of the atmospheric layers.
    albedo : float, default=0.33
        Set the planetary albedo from 0 to 1.
    S0 : float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2.
    nuclear_winter : bool, optional
        Set to True if simulating a nuclear winter scenario. Defaults to false.
    debug : boolean, default=False
        Turn on debug output.

    Returns
    -------
    temps : Numpy array of size N+1
        Array of temperatures from the Earth's surface (element 0) through
        each of the N atmospheric layers, from lowest to highest.
    '''

    # Create matrices:
    A = np.zeros([N+1, N+1])
    b = np.zeros(N+1)
    # If not a nuclear winter, solar flux goes to the surface
    if not nuclear_winter:
        b[0] = -S0 / 4 * (1 - albedo)
    # Top layer absorbs all incoming solar flux
    else:
        b[N] = -S0 / 4  

    if debug:
        print(f"Populating N+1 x N+1 matrix (N = {N})")

    # Populate our A matrix piece-by-piece.
    for i in range(N+1):
        for j in range(N+1):
            if debug:
                print(f"Calculating point i={i}, j={j}")
            # Diagonal elements are always -2 ('cept at Earth's surface.)
            if i == j:
                A[i, j] = -1*(i > 0) - 1
                # print(f"Result: A[i,j]={A[i,j]}")
            else:
                # This is the pattern we solved for in class!
                m = np.abs(j-i) - 1
                A[i, j] = epsilon * (1-epsilon)**m
    # At Earth's surface, epsilon =1, breaking our pattern.
    # Divide by epsilon along surface to get correct results.
    A[0, 1:] /= epsilon

    # Verify our A matrix.
    if debug:
        print(A)

    # Get the inverse of our A matrix.
    Ainv = np.linalg.inv(A)

    # Multiply Ainv by b to get Fluxes.
    fluxes = np.matmul(Ainv, b)

    # Convert fluxes to temperatures.
    # Fluxes for all atmospheric layers
    temps = (fluxes/epsilon/sigma)**0.25
    temps[0] = (fluxes[0]/sigma)**0.25  # Flux at ground: epsilon=1.

    return temps

def q_3():
    '''
    Answers Question 3 via performing experiments to analyze surface temperature 
    in relation to emissivity and the number of atmospheric layers.

    Experiment 1: 
        - Vary the emissivity from 0.025 to 1 for a single-layer atmosphere and
          calculate the corresponding surface temperature. Approximate the emissivity 
          for a surface temperature of 288 K.

    Experiment 2: 
        - Vary the number of atmospheric layers from 1 to 100 with a fixed emissivity 
          and calculate the corresponding surface temperature. Approximate the number of 
          layers for a surface temperature of 288 K.

    Both experiments generate and save plots to visualize the relationship between 
    surface temperature and emissivity, and surface temperature and the number of layers.

    Prints
    ------
    - Approximate emissivity for 288 K in Experiment 1.
    - Approximate number of layers for 288 K in Experiment 2.
    '''
    
    # EXPERIMENT 3.1: Range of Emissivities
    # Constructing emissivity datapoints from .025 through 1
    emissivities = np.linspace(0.025, 1, 21)
    surface_temps_emiss = []

    # Populating corresponding surface temperature array for each point
    for emmis in emissivities:
        temp_surface = n_layer_atmos(1, emmis, 1350, 0.33)[0]
        surface_temps_emiss.append(temp_surface)

    # Approximating the emissivity value via the numpy interpolation function
    # Documentation = https://numpy.org/doc/stable/reference/generated/numpy.interp.html
    approx_emissivity = np.interp(288, surface_temps_emiss, emissivities)
    print(f"Approximate emissivity for 288 K: {approx_emissivity}")

    # Sets up emissivity plot and saves it as a .png file
    plt.figure()
    plt.plot(emissivities, surface_temps_emiss)
    plt.xlabel('Emissivity', fontsize=14)
    plt.ylabel('Surface Temperature (K)', fontsize=14)
    plt.title('Surface Temperature vs. Emissivity (Single Layer Atmosphere)')
    # NOTE: this just puts grid lines behind the graph to make it look nicer
    plt.grid(True) 
    plt.savefig('temp_vs_emiss.png')

    # EXPERIMENT 2: Varying Number of Layers
    # Constructing layer datapoints from 1 to 100
    num_layers = np.arange(1, 100)
    surface_temps_layers = []

    # Populating corresponding surface temperature array for each point
    for N in num_layers:
        temp_surface = n_layer_atmos(N, 0.255, 1350, 0.33)[0]
        surface_temps_layers.append(temp_surface)

    # Approximating the layer value via the numpy interpolation function
    # Documentation = https://numpy.org/doc/stable/reference/generated/numpy.interp.html
    approx_layers = np.interp(288, surface_temps_layers, num_layers)
    print(f"Approximate number of layers for 288 K: {approx_layers}")

    plt.figure()
    plt.plot(num_layers, surface_temps_layers)
    plt.xlabel('Number of Layers', fontsize=14)
    plt.ylabel('Surface Temperature (K)', fontsize=14)
    plt.title('Surface Temperature vs. Number of Atmospheric Layers')
    plt.grid(True)
    plt.savefig('temp_vs_layers.png')

def q_4():
    '''
    Answers Question 4 via simulating the temperature on Venus by incrementing the 
    number of atmospheric layers until the surface temperature exceeds 700 K.

    Prints
    ------
    - The surface temperature at the point of stopping.
    - The number of layers required for Venus to reach a surface temperature of 700 K.
    '''
    num_layers_venus = 1
    while True:
        temp_surface = n_layer_atmos(num_layers_venus, 1, 2600, .7)[0]
        if temp_surface > 700:
            print(f"Surface temperature at this point: {temp_surface:.2f} K")
            break
        num_layers_venus += 1

    print(f"Number of layers for Venus to reach 700 K: {num_layers_venus}")
    # Testing Earth
    temp_surface = n_layer_atmos(num_layers_venus, 1, 2600, .33)[0]
    print(f"Earth with 69 layers: {temp_surface}")


def q_5():
    '''
    Answers Question 5 via simulating a nuclear winter scenario and plotting the temperature
    at various altitudes for a five-layer atmosphere.

    This function calculates the temperature at each atmospheric layer under a nuclear winter scenario 
    and generates a plot of altitude vs. temperature.

    Prints
    ------
    - Surface temperature at the Earth's surface in the nuclear winter scenario.
    
    Generates
    ---------
    - A plot of altitude vs. temperature, saved as 'nuclear_winter_temp.png'.
    '''
    temps_nuclear_winter = n_layer_atmos(5, .5, 1350, .33, nuclear_winter=True)
    altitude = np.arange(0, 5 + 1) 
    # Setting up plot and axes
    plt.figure()
    plt.plot(altitude, temps_nuclear_winter)
    plt.xlabel('Altitude', fontsize=14)
    plt.ylabel('Temperature (K)', fontsize=14)
    plt.title('Nuclear Winter: Altitude vs. Temperature')
    plt.grid(True)
    plt.savefig('nuclear_winter_temp.png')
    # Printing surface temperature with two decimal places
    print(f"Surface temperature = {temps_nuclear_winter[0]:.2f} K")
    plt.show()

# Main function for the questions
def main():
    
    # DATA VERIFICATION
    print("Verifying data for the two-atmosphere and 1/3 emissivity case")
    temperature_1 = n_layer_atmos(2, 1/3, 1350, .33, debug=True);
    # Printing temps in order of online model
    reversed_temps = temperature_1[::-1]
    for temp in reversed_temps:
        print(temp)
    print("Verifying data for the three-atmosphere and .5 emissivity case")
    temperature_2 = n_layer_atmos(3, .5, 1350, .33, debug=True);
    # Printing temps in order of online model
    reversed_temps = temperature_2[::-1]
    for temp in reversed_temps:
        print(temp)

    # QUESTION 3
    q_3()
   
   # QUESTION 4
    q_4()

   # QUESTION 5
    q_5()

# Run the main function
if __name__ == "__main__":
    main()
