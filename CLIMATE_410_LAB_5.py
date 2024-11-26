#!/usr/bin/env python3

'''
Draft code for Lab 5: SNOWBALL EARTH!!!
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Some constants:
radearth = 6357000.  # Earth radius in meters.
mxdlyr = 50.         # depth of mixed layer (m)
sigma = 5.67e-8      # Steffan-Boltzman constant
C = 4.2e6            # Heat capacity of water
rho = 1020           # Density of sea-water (kg/m^3)


def gen_grid(nbins=18):
    '''
    Generate a grid from 0 to 180 lat (where 0 is south pole, 180 is north)
    where each returned point represents the cell center.

    Parameters
    ----------
    nbins : int, defaults to 18
        Set the number of latitude bins.

    Returns
    -------
    dlat : float
        Grid spacing in degrees.
    lats : Numpy array
        Array of cell center latitudes.
    '''

    dlat = 180 / nbins  # Latitude spacing. (NOTE: edges = #bins + 1)
    lats = np.arange(0, 180, dlat) + dlat/2. # (NOTE: "dlat/2" shifts location to cell center)

    # Alternative way to obtain grid:
    # lats = np.linspace(dlat/2., 180-dlat/2, nbins) <-- POTENTIALLY DO IT THIS WAY???

    return dlat, lats

# NOTE: directly from lab manual
def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Get base grid:
    dlat, lats = gen_grid()

    # Set initial temperature curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    coeffs = np.polyfit(lats, T_warm, 2) # (NOTE: = creates
    # polynomial curve for the given temperature values, enabling
    # prediction of temperature output for given latitude input)
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2 
    
    return temp

# NOTE: = directly from the lab manual
def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes. (i.e. S(y))
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation


def snowball_earth(nbins=18, dt=1., tstop=10000, lam=100., spherecorr=True,
                   debug=False, albedo=0.3, emiss=1, S0=1370,
                   insolation_multiplier = 1, init = None, dynalb = False):
    '''
    Simulates the latitudinal temperature distribution on Earth over time 
    using a numerical climate model based on heat diffusion, spherical 
    corrections, and radiative forcing.

    Parameters
    ----------
    nbins : int, optional (default=18)
        Number of latitude bins. Each bin corresponds to a latitudinal 
        "ring" where temperatures are calculated.
    dt : float, optional (default=1.0)
        Timestep in units of years. Determines the temporal resolution of the simulation.
    tstop : float, optional (default=10000)
        Total simulation time in years.
    lam : float, optional (default=100)
        Diffusion coefficient (m²/s) representing the efficiency of heat transfer between latitude bands.
    spherecorr : bool, optional (default=True)
        If True, includes the spherical coordinate correction term to account 
        for Earth's curvature in the diffusion process.
    debug : bool, optional (default=False)
        If True, enables debug print statements for diagnostic purposes.
    albedo : float, optional (default=0.3)
        Initial value for Earth's surface albedo. Ignored if `dynalb` is True.
    emiss : float, optional (default=1.0)
        Emissivity of the Earth's surface, which determines its efficiency 
        as a blackbody radiator. Affects radiative cooling.
    S0 : float, optional (default=1370)
        Solar constant in W/m². Represents the total incoming solar energy.
    insolation_multiplier : float, optional (default=1.0)
        Multiplier applied to the insolation term, allowing exploration of 
        varying solar forcing conditions.
    init : float or array-like, optional (default=None)
        Initial temperature(s) in °C for each latitude band. If None, 
        defaults to a pre-defined "warm Earth" condition.
    dynalb : bool, optional (default=False)
        If True, enables dynamic albedo changes based on temperature. Albedo 
        transitions between frozen (0.6) and unfrozen (0.3) ground values 
        at a temperature threshold of -10°C.

    Returns
    -------
    lats : numpy.ndarray
        Array of latitude values in degrees, where 0° represents the south pole 
        and 180° represents the north pole.
    Temp : numpy.ndarray
        Final equilibrium temperatures in °C for each latitude band.

    Notes
    -----
    - The model assumes a uniform mixed layer depth of 50 m for simplicity.
    - The implicit numerical solver ensures stability for large timesteps.
    - Dynamic albedo functionality introduces a feedback mechanism 
      where frozen regions increase cooling due to higher reflectivity.
    '''
    # Get time step in seconds:
    dt_sec = 365 * 24 * 3600 * dt  # Years to seconds.

    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get grid spacing in meters.
    dy = radearth * np.pi * dlat / 180.

    # Generate insolation:
    insol = insolation(S0, lats)

    # Initialize temp
    if init is None:
        Temp = temp_warm(lats)
    else:
        Temp = init

    # Initialize albedo
    if dynalb:
        albedo = np.zeros_like(Temp)
        albedo[Temp <= -10] = 0.6
        albedo[Temp > -10] = 0.3
    else:
        albedo = np.full_like(Temp, albedo)

    # Get number of timesteps:
    nstep = int(tstop / dt)

    # Build matrix A
    A = np.identity(nbins) * -2  # Set diagonal elements to -2
    A[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    A[np.arange(nbins-1)+1, np.arange(nbins-1)] = 1  # Set off-diag elements
    # Set boundary conditions:
    A[0, 1], A[-1, -2] = 2, 2

    # Build "B" matrix for applying spherical correction:
    B = np.zeros((nbins, nbins))
    B[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    B[np.arange(nbins-1)+1, np.arange(nbins-1)] = -1  # Set off-diag elements
    # Set boundary conditions:
    B[0, :], B[-1, :] = 0, 0

    # Set the surface area of the "side" of each latitude ring at bin center.
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2) 

    # Set units of A derp
    A /= dy**2

    # Get our "L" matrix:
    L = np.identity(nbins) - dt_sec * lam * A
    L_inv = np.linalg.inv(L) 

    for i in range(nstep):
        # Add spherical correction term:
        if spherecorr:
            Temp += dt_sec * lam * dAxz * np.matmul(B, Temp) # Second Term

        if(dynalb):
            loc_ice = Temp <= -10
            albedo = np.zeros_like(Temp)
            albedo[loc_ice] = .6
            albedo[~loc_ice] = .3

        # Apply insolation and radiative losses:
        # print('T before insolation:', Temp)
        radiative = (1-albedo) * insol * insolation_multiplier - emiss*sigma*(Temp+273.15)**4
        Temp += dt_sec * radiative / (rho*C*mxdlyr) # Last Term

        Temp = np.matmul(L_inv, Temp) # Applying inverse multiplier to all terms

    return lats, Temp

def function_plotter(nbins, temp_prof_array, title_array, title):
    '''
    Generalized function to plot temperature profiles.

    Parameters
    ----------
    nbins: int
        Number of latitude bins for the grid.
    temp_prof_array: list of arrays
        List of temperature profiles to be plotted.
    title_array: list of str
        List of titles corresponding to each temperature profile.
    title: str
        Title for the plot and saved PNG file.
    '''
    # Generate very simple grid
    dlat, lats = gen_grid(nbins)
    # Transforming lats to match desired x axis, centered at 0
    lats = lats - 90
    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # Plot each temperature profile
    for i in range(len(temp_prof_array)):
        ax.plot(lats, temp_prof_array[i], label= title_array[i])
    # Set axes and legend
    ax.set_xlabel('Latitude (0 = Equator)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')
    ax.legend(loc='best')
    ax.set_title(title)
    # Adjust layout
    fig.tight_layout()
    # Save the plot as a file, with specified name
    filename = f"{title}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')

def driver_function(plotting_condition, tstop=10000, nbins = 18):
    '''
    Executes and plots results for various climate scenarios based on the 
    Snowball Earth model.

    Parameters
    ----------
    plotting_condition : str
        Specifies the scenario to execute:
        - "validation_case": Validates model with basic diffusion, spherical 
          correction, and radiative forcing cases.
        - "emmis-diffus_variance": Explores the effect of varying emissivity (ϵ) 
          and diffusivity (λ) on equilibrium temperature profiles.
        - "albedo_variance": Investigates the impact of initial conditions 
          ("hot," "cold," and "freeze-thaw") on equilibrium with dynamic albedo.
        - "solar_insolation_variance": Simulates temperature responses to 
          time-varying solar insolation under cold initial conditions.
    tstop : float, optional (default=10000)
        Total simulation time in years for each trial.
    nbins : int, optional (default=18)
        Number of latitude bins for the model.

    Returns
    -------
    None
        Generates and saves plots for the specified scenario.

    Notes
    -----
    - Results are saved as images and displayed for visualization.
    - Uses helper functions (`snowball_earth` and `function_plotter`) 
      to perform simulations and create plots.
    '''
    # Create initial condition:
    dlat, lats = gen_grid(nbins)
    initial = temp_warm(lats)
    # Q1 ------------------------------------------------------------
    if plotting_condition == "validation_case":
        # Get simple diffusion solution:
        lats, t_diff = snowball_earth(tstop=tstop, spherecorr=False, S0=0, emiss=0)
        # Get diffusion + spherical correction:
        lats, t_sphe = snowball_earth(tstop=tstop, S0=0, emiss=0)
        # Get diffusion + sphercorr + radiative terms:
        lats, t_rad = snowball_earth(tstop=tstop)

        # Creating lists for function_plotter() input
        temps = [initial, t_diff, t_sphe, t_rad]
        names = ["Warm Earth Init. Cond.", "Simple Diffusion",
                 "Diffusion + Sphere. Corr.", 
                 "Diffusion + Sphere. Corr. + Radiative"]
        function_plotter(nbins, temps, names, "Validation Plot")

    # Q2 ------------------------------------------------------------
    if plotting_condition == "emmis-diffus_variance":
        # Epsilon general behavior, holding lambda constant at 100
        epsilons = [0, .05, .15, .5, 1]
        temp_profiles_epsilon = []

        for epsilon in epsilons:
            temp_profile = snowball_earth(emiss = epsilon)[1]
            temp_profiles_epsilon.append(temp_profile)

        # Call plotter for epsilon variation
        function_plotter(nbins, temp_profiles_epsilon, 
                        [f"Epsilon = {round(eps, 2)}" for eps in epsilons], 
                        "Emissivity Variance")

        # Lambda general behavior, holding epsilon constant at .03
        lambdas = [0, 7.5, 15, 50, 150]
        temp_profiles_lambda = []

        for lamb in lambdas:
            temp_profile = snowball_earth(lam = lamb)[1]
            temp_profiles_lambda.append(temp_profile)

        # Call plotter for lambda variation
        function_plotter(nbins, temp_profiles_lambda, 
                        [f"Lambda = {round(lamb, 2)}" for lamb in lambdas], 
                        "Diffusivity Variance")
        
        # Visually looking for close combination to warm Earth
        # NOTE: this plot is NOT included in the report; it was just a visual means
        # to find a close fit
        # Trialing Diffusivity-Epsilon Combinations
        lambdas = [10, 15, 20, 22, 24, 30, 35, 40]
        epsilons = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9]
        # Store temperature profiles and labels
        temp_profiles = []
        labels = []
        for lam, epsilon in zip(lambdas, epsilons):
            lats, temp_profile = snowball_earth(nbins=nbins, lam=lam, emiss=epsilon)
            temp_profiles.append(temp_profile)
            labels.append(f"$\\lambda$={lam}, $\\epsilon$={epsilon}")
        # Putting the warm one on for comparison too
        temp_profiles.append(initial)
        labels.append("Warm Earth")
        # Plot all combinations
        function_plotter(
            nbins,
            temp_profiles,
            labels,
            "Diffusivity and Emissivity Combinations"
        )

    # Q3 ------------------------------------------------------------
    if plotting_condition == "albedo_variance":
        current_temp = np.full(18, 60, dtype=float)
        hot_earth = snowball_earth(tstop = 40000, albedo = .3, lam = 24, emiss = .7, init = current_temp, dynalb = True)[1]
        current_temp = np.full(18, -60, dtype=float)  
        cold_earth = snowball_earth(tstop = 40000, albedo = .6, lam = 24, emiss = .7, init = current_temp, dynalb = True)[1]
        flash_freeze = snowball_earth(tstop = 40000, albedo = .6, lam = 24, emiss = .7)[1]
        # Plot the best-matching profile against the warm Earth target
        function_plotter(
            nbins, 
            [hot_earth, cold_earth, flash_freeze, initial], 
            ["Hot Earth", "Cold Earth", "Flash Freeze", "Warm Earth"], 
            "Comparing Hot, Cold, Flash Freeze, and Warm Scenarios"
        )
    # Q4 ------------------------------------------------------------
    if plotting_condition == "solar_insolation_variance":
        gamma_values = np.arange(0.4, 1.45, 0.05).tolist() + np.arange(1.35, 0.35, -0.05).tolist()
        avg_temps = [] 
        # Start with a "cold Earth" initial condition
        current_temp = np.full(18, -60, dtype=float)
        # Iterate through increasing and decreasing gamma values
        for gamma in gamma_values:
            lats, current_temp = snowball_earth(tstop=20000,albedo = .6,lam=24,
            emiss=.7,insolation_multiplier=gamma, init= current_temp,  dynalb=True,
            )
            avg_temps.append(np.mean(current_temp)) 

        # Plot average global temperature vs gamma
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(gamma_values, avg_temps, marker='o', label='Average Global Temperature')
        ax.set_xlabel("Solar Multiplier Factor ($\\gamma$)")
        ax.set_ylabel("Average Global Temperature ($^{\circ}C$)")
        ax.set_title("Impact of Solar Forcing on Snowball Earth Stability")
        ax.axhline(y=0, color='r', linestyle='--', label='Freezing Point')
        ax.legend()
        plt.tight_layout()
        plt.savefig("solar_insolation_variance.png", dpi=300, bbox_inches="tight")
        plt.show()

# main definition
def main():

    # Q1 = Data Validation
    driver_function("validation_case")

    # Q2 = Diffusivity and Emissivity Variance
    driver_function("emmis-diffus_variance")
    # Best Match Observed: Lambda = 24 Epsilon = .7

    # Q3 = Albedo Variance
    driver_function("albedo_variance")

    # Q4 = Solar Insolation Variance
    driver_function("solar_insolation_variance")

# main driver
if __name__ == "__main__":
    main()

