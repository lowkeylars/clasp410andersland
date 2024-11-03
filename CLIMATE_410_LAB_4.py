#!/bin/bash/env python3
# CLIMATE 410
# LAB #4 - HEAT DIFFUSION & PERMAFROST
# DUE 10/31/2024
# STUDENT NAME - LARS ANDERSLAND

# improvements - docstring, break up code to take plotting out of main, add a convergence check
'''
Tools and methods for solving our heat equation/diffusion

CHANGES
-------
- [x] added shebang at top of file
- [x] redefine the map variable so it doesn't overwrite built in map function
- [ ] move unit tests and plotting to external function
- [ ] add convergence check to reduce compute time then run each experiment once instead
- [ ] improve the temp_kanger docstring
- [ ] improve the docstring for script
'''
import numpy as np
import matplotlib.pyplot as plt
import sys #enables exiting 

# Kangerlussuaq average temperature:
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
10.7, 8.5, 3.1, -6.0, -12.0, -16.9])

def temp_kanger(t, offset_temp):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.

    Parameters
    ----------
    t : np.ndarray
        array of times to interpolate temperatures to
    offset_temp : float 
        how much to offset the temperature by

    Return
    ------
        : np.ndarray
        array of temperatures interpolated to the unit in the solver
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean() + offset_temp

def heatdiff(xmax, tmax, dx, dt, offset_temp, file_prefix, c2=.0216, debug=False, validation_case = False):
    '''
    Simulates heat diffusion through the ground using the forward difference method.

    Parameters:
    -----------
    xmax : float
        Maximum depth in meters
    tmax : float
        Maximum time in days
    dx : float
        Step size in meters
    dt : float
        Time step in seconds
    offset_temp : float
        Offset value for the temperature at the surface boundary, which is applied in the 
        surface boundary condition using the Kangerlussuaq heating equation.
     file_prefix : str, optional
        Prefix for the filenames of the output .png files.
    c2 : float, optional
        Thermal diffusivity constant (default = 1). It defines how quickly heat diffuses 
        through the medium.
    debug : bool, optional
        If True, prints information about the grid and numerical details (default is False).
    validation_case : bool, optional
        If True, uses a specific set of initial and boundary conditions for validation purposes 
        (default is False). 

    Returns:
    --------
    xgrid : ndarray
        Array of spatial points in the ground
    tgrid : ndarray
        Array of time values
    U : ndarray
        2D array of calculated temperatures at each spatial and temporal point, representing 
        the temperature profile over time and depth.

    MISC Notes:
    ------
    - The function checks for numerical stability based on checking `dt <= dx^2 / (2 * c2)`. 
      If the not met, the function prompts the user to retry with smaller time or depth steps.
    - Enables doing either the validation or nonvalidation case via parameter input
    - The function also generates and saves two plots:
        1. A space-time heat map showing the temperature variation over time and depth.
        2. Seasonal temperature profiles at the maximum time step, displaying the temperature distribution 
        in winter and summer. Active and permafrost layer depths are estimated and visualized.
    '''
    # Checking for numerical stability
    if(dt > (dx**2 / (2 * c2))):
        # TO DO: raise a warning instead of just exiting with a print statement
        print('Your selected time and depth steps result in numerically unstable '
             'behavior. Please retry with steps so that dt <= (dx^2) / (2 * c2).')
        sys.exit(1)

    # Start by calculating size of array: MxN
    M = int(np.round(xmax / dx + 1))
    N = int(np.round(tmax / dt + 1))

    # IDEA = creates arrays for the axes
    xgrid, tgrid = np.arange(0, xmax+dx, dx), np.arange(0, tmax+dt, dt)

    # Initialize our data array:
    U = np.zeros((M,N))

    # Setting Initial and Boundary Conditions
    # TO DO: make this an external test case and remove if statement
    if(validation_case):
        U[:, 0] = 4*xgrid - 4*xgrid**2
        U[0, :] = 0
        U[-1, :] = 0
    else:
        U[0, :] = temp_kanger(tgrid, offset_temp) # Surface, due to Kangerlussuaq heating equation
        U[-1, :] = 5 # Lowest depth, due to geothermal 

    # Set our "r" constant.
    r = c2 * dt / dx**2

    # Solve! Forward differnce ahoy.
    # IDEA = for each time step, solve for all of the corresponding 
    # spatial values (except the bounday conditions)
    for j in range(N-1):
        U[1:-1, j+1] = (1-2*r) * U[1:-1, j] + \
            r*(U[2:, j] + U[:-2, j])

    # Debug or Validation Case Condition Printing Out Grids
    if debug or validation_case:
        print(f'There are {M} points in space and {N} points in time.')
        print('Temperature array U in grid format:')
        # array2string to make the precision exactly match what's in the lab manual
        # documentation: https://numpy.org/devdocs/reference/generated/numpy.array2string.html
        print(np.array2string(U, formatter={'float_kind': lambda x: f"{x:.7f}" if x < 0.1 else f"{x:.6f}"
                                         if x < 0.2 else f"{x:.5f}" if x < 0.5 else f"{x:.3f}"}))

    # No plots generated for the validation case. Just printing the table.
    if not validation_case:
        # TO DO: move outside of the function
        # PLOT #1 = Space-Time Heat Map
        # Setting up plot, axes, and labels
        fig, ax1 = plt.subplots(1, 1)
        # TO DO: change map so not redefining python map function
        colormap = ax1.pcolor(tgrid / 365, xgrid, U, cmap='seismic', vmin=-25, vmax=25)
        ax1.set_title('Ground Temperature: Kangerlussuaq, Greenland')
        ax1.set_xlabel('Time (Years)', fontsize = 14)
        ax1.invert_yaxis()
        ax1.set_ylabel('Depth (m)', fontsize = 14)
        plt.figtext(0.5, -0.04, f"Temperature Offset: {offset_temp}°C", ha="center", fontsize=12)
        plt.colorbar(colormap, ax=ax1, label='Temperature ($C$)')
        fig.savefig(f"{file_prefix}_heat_map.png", dpi=300, bbox_inches='tight')
        
        # PLOT #2 = Seasonal Temperature Profiles
        # Set indexing for final year (i.e. finding how many timesteps are in a year)
        loc = int(-365 / dt) 
        # Extracting min values as winter
        winter = U[:, loc:].min(axis=1) 
        # Extracting max values as summer
        summer = U[:, loc:].max(axis=1) 

        # Identify the active layer depth (first point where summer temperature transitions from positive to negative)
        active_layer_depth = None
        for i in range(1, len(summer)):  
            if summer[i] < 0 and summer[i-1] > 0:  
                # Linear interpolation required for active layer due to significant change in point
                depth1, depth2 = xgrid[i-1], xgrid[i]
                temp1, temp2 = summer[i-1], summer[i]
                active_layer_depth = depth1 + (0 - temp1) * (depth2 - depth1) / (temp2 - temp1)
                # Stop after finding the first transition
                break  

        # Identify the permafrost depth (first point where summer temperature reaches 0°C from the bottom)
        permafrost_depth = None
        for i in range(len(summer) - 1, -1, -1): 
            # Method without usig linear interpolation has shown to be sufficiently accurate
            if abs(summer[i]) < 1e-2:  
                permafrost_depth = xgrid[i]
                permafrost_index = i  # i think this is redundant variable?
                # Stop after finding the first crossing from the bottom
                break  

        # Setting up plot, axes, and labels
        fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        ax2.plot(winter, xgrid, label='Winter')
        ax2.plot(summer, xgrid, label='Summer', linestyle='--')
        plt.gca().invert_yaxis()  
        ax2.set_xlim(-8, 6)
        ax2.set_ylim(100, 0)
        ax2.set_xlabel('Temperature (°C)', fontsize = 14)
        ax2.set_ylabel('Depth (m)', fontsize = 14)
        plt.figtext(0.5, 0.01, f"Time Elapsed: {tmax/365:.2f} Years | Temperature Offset: {offset_temp}°C", 
                ha="center", fontsize=12)
        ax2.set_title('Ground Temperature: Kangerlussuaq')
        # Enabling the gridlines shown in the lab manual
        ax2.set_xticks(np.arange(-8, 8, 2))
        ax2.set_yticks(np.arange(0, 110, 10))  
        ax2.grid(True)  

        # Plot the active and permafrost layer depths if found (NOTE: there were edge cases (extremely small
        # or large tmax)where my algorithm here didn't find one, which is why I have the conditions
        if active_layer_depth is not None:
            ax2.plot(0, active_layer_depth, 'ro', label='Active Layer Depth')
            ax2.text(0.2, active_layer_depth, f'{active_layer_depth:.2f} m', 
                    ha='left', va='center', color='red', fontsize=14)
        if permafrost_depth is not None:
            ax2.plot(0, permafrost_depth, 'bo', label='Permafrost Depth')
            ax2.text(0.2, permafrost_depth, f'{permafrost_depth:.2f} m', 
                    ha='left', va='center', color='blue', fontsize=14)
         
        ax2.legend()
        plt.show()
        fig.savefig(f"{file_prefix}_seasonal_profile.png", dpi=300, bbox_inches='tight')

    # Return grid and result:
    return xgrid, tgrid, U

def unit_test():
    """
    Function completes a unit test to verify if the solver
    is working correctly. Does this using the solution given 
    in the lab manual.
    """
    # Solution to problem 10.3 from fink/matthews as a nested list:
    sol10p3 = [[0.000000, 0.640000, 0.960000, 0.960000, 0.640000, 0.000000],
        [0.000000, 0.480000, 0.800000, 0.800000, 0.480000, 0.000000],
        [0.000000, 0.400000, 0.640000, 0.640000, 0.400000, 0.000000],
        [0.000000, 0.320000, 0.520000, 0.520000, 0.320000, 0.000000],
        [0.000000, 0.260000, 0.420000, 0.420000, 0.260000, 0.000000],
        [0.000000, 0.210000, 0.340000, 0.340000, 0.210000, 0.000000],
        [0.000000, 0.170000, 0.275000, 0.275000, 0.170000, 0.000000],
        [0.000000, 0.137500, 0.222500, 0.222500, 0.137500, 0.000000],
        [0.000000, 0.111250, 0.180000, 0.180000, 0.111250, 0.000000],
        [0.000000, 0.090000, 0.145625, 0.145625, 0.090000, 0.000000],
        [0.000000, 0.072812, 0.117813, 0.117813, 0.072812, 0.000000]]
    
    # Convert to an array and transpose it to get correct ordering:
    sol10p3 = np.array(sol10p3).transpose()

    
    
# main definition
def main():
# NOTE: all plots were deleted before uploading to Github
# NOTE: due to computation time, do NOT try running all lines here at once.
# Instead, comment out all the ones you don't want to run, running one or two at a time

# could simplify this a lot to reduce the compute time, instead do a convergence check then exit

### QUESTION 1 ###
    heatdiff(1, .2, .2, .02, 0, "NA", c2 = 1, validation_case=True) # Figure 2

    ## Stability Testing - uncomment each line to test; correct behavior = system exiting
    #heatdiff(100, 1825, 1, 23.15, 0, "NA")
    #heatdiff(100, 1825, .5, 6, 0, "NA")
    #heatdiff(100, 1825, 2, 100, 0, "NA")
    
### QUESTION 2 ###
    heatdiff(100, 1825, .5, .2, 0, "Q2.1") # 5 Year
    heatdiff(100, 3650, .5, .2, 0, "Q2.2") # 10 Years
    heatdiff(100, 7300, .5, .2, 0, "Q2.3") # 20 Years
    heatdiff(100, 10950, .5, .2, 0, "Q2.4") # 30 Years
    heatdiff(100, 5475, .5, .2, 0, "Q2.5") # 15 Years

### QUESTION 3 ###
    heatdiff(100, 3650, .5, .2, .5, "Q3.1") # 10 Years, .5 Shift
    heatdiff(100, 10950, .5, .2, .5, "Q3.2") # 30 Years, .5 Shift
    heatdiff(100, 3650, .5, .2, 1, "Q3.3") # 10 Years, 1 Shift
    heatdiff(100, 10950, .5, .2, 1, "Q3.4") # 30 Years, 1 Shift
    heatdiff(100, 3650, .5, .2, 3, "Q3.5") # 10 Years, 3 Shift
    heatdiff(100, 10950, .5, .2, 3, "Q3.6") # 30 Years, 3 Shift
    heatdiff(100, 3650, .5, .2, 5, "Q3.7") # 30 Years, 3 Shift

# main() driver
if __name__ == "__main__":
    main()
    