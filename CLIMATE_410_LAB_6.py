import numpy as np
import matplotlib.pyplot as plt


def calculate_rainfall_energy(rainfall_intensity):
    '''
    Calculate unit rainfall energy based on rainfall intensity.

    Parameters
    ----------
    rainfall_intensity : float
        Rainfall intensity for the event.

    Returns
    -------
    float
        rainfall_energy

    Notes
    -----
    - Uses empirical relationship between rainfall intensity and energy
    '''
    # Calculate energy using the empirical relationship with intensity
    exponential_term = np.exp(-0.082 * rainfall_intensity)
    rainfall_energy = 0.29 * (1 - 0.72 * exponential_term)
    
    return rainfall_energy

def generate_rainfall_event(climate_factor=1.0):
    '''
    Generate a rainfall event based on the given probability distribution.

    Parameters
    ----------
    climate_factor : float, optional (default=1.0)
        Multiplier for intense rainfall probabilities to simulate climate change.

    Returns
    -------
    float
        Selected rainfall intensity based on probability distribution.

    Notes
    -----
    - Default distribution represents typical Midwest rainfall patterns
    - Climate factor modifies probabilities of intense events (>3.5)
    '''
    # Given rainfall data
    rainfall_data = [
        (0.25, 0.30),  # Light rain
        (1.0, 0.25),   # Moderate rain
        (2.0, 0.20),   # Heavy rain
        (3.5, 0.15),   # Very heavy rain
        (5.0, 0.07),   # Extreme rain
        (7.0, 0.03)    # Severe storm
    ]
    
    # Split up this lab manual list into two separate ones
    rainfall_intensities = [data[0] for data in rainfall_data]
    rainfall_probabilities = [data[1] for data in rainfall_data]
    
    # Adjust probabilities if climate change is being modeled (i.e. Q4)
    if climate_factor > 1.0:
        # Loop through each rainfall intensity and adjust its probability
        for index in range(len(rainfall_probabilities)):
            if rainfall_intensities[index] > 3.5:
                rainfall_probabilities[index] = rainfall_probabilities[index] * climate_factor
        
        # Renormalize probabilities to sum to 1
        probability_sum = sum(rainfall_probabilities)
        for index in range(len(rainfall_probabilities)):
            rainfall_probabilities[index] = rainfall_probabilities[index] / probability_sum
    
    # Generate a random value and find corresponding rainfall intensity
    random_value = np.random.random()
    running_probability_sum = 0
    # LOGIC = find if random number generated is less than given probability; if not, add next
    # probability and see if it's less. First time finding it's less indicates return
    for current_intensity, current_probability in zip(rainfall_intensities, rainfall_probabilities):
        running_probability_sum += current_probability
        if random_value <= running_probability_sum:
            return current_intensity

def calculate_k_factor(time_years):
    '''
    Calculate time-varying soil erodibility factor using exponential grow.

    Parameters
    ----------
    time_years : float
        Time in years since start of simulation.

    Returns
    -------
    float
        Current soil erodibility factor (K) based on exponential progression.

    Notes
    -----
    - Models increasing soil erodibility as protective topsoil erodes away
    '''
    # Calculate how soil erodibility changes over time using exponential approach
    exponential_term = np.exp(.15 * time_years)
    current_k_factor = .45 + (.15) * exponential_term
    
    return current_k_factor

def calculate_c_factor(time_years):
    '''
    Calculate seasonal cover management factor using a sinusoidal function.

    Parameters
    ----------
    time_years : float
        Time in years since start of simulation.

    Returns
    -------
    float
        Current cover management factor (C) based on seasonal cycle.

    Notes
    -----
    - Models seasonal vegetation changes affecting soil protection
    - Minimum C (most protection) occurs in summer
    - Maximum C (least protection) occurs in winter
    - Complete cycle occurs each year
    '''
    # Calculate seasonal variation using sine wave
    yearly_cycle = 2 * np.pi * time_years
    phase_shift = np.pi/2
    seasonal_variation = np.sin(yearly_cycle - phase_shift)

    current_c_factor = .4 + .2 * seasonal_variation
    
    return current_c_factor

def calculate_erosion(years, ls, p, timestep_years=1/12, num_trials=1, static=False, climate_factor=1.0,
        fixed_rainfall=None):
    '''
    Calculate soil erosion over time using the Universal Soil Loss Equation (USLE).

    Parameters
    ----------
    years : float
        Total simulation time in years.
    ls : float
        Topographic factor combining slope length and steepness.
    p : float
        Support practice factor representing erosion control measures.
    timestep_years : float
        Time step for calculations in years.
    num_trials : int, optional (default=1)
        Number of trials to average for probabilistic calculations.
    static : bool, optional (default=False)
        If True, uses fixed parameters and simple time multiplication.
    climate_factor : float, optional (default=1.0)
        Factor to increase probability of intense rainfall events.
    fixed_rainfall : float, optional (default=None)
        If provided, uses this fixed intensity instead of random generation.

    Returns
    -------
    array
        (time_points, erosion_values) where:
        - time_points : array of simulation times in years
        - erosion_values : array of cumulative erosion values in t/ha

    Notes
    -----
    - Static mode uses constant parameters multiplied by time
    - Dynamic mode updates K and C factors at each timestep
    - Rainfall can be fixed or generated probabilistically (via generate_rainfall_event())
    - Multiple trials are averaged to handle random variation
    '''
    # Creating numbe time points in accordance with given time_step
    time_points = np.arange(0, years + timestep_years, timestep_years)
    number_of_steps = len(time_points)
    
    # Create array to store results from
    all_trials_erosion = np.zeros((num_trials, number_of_steps))
    
    # Run the specified number of trials you want
    for trial_number in range(num_trials):
        # STATIC CASE: simply calculate erosion once and then multiply over time
        if static:
            rainfall_intensity = fixed_rainfall # i.e. anytime running static, will have this fixed rain
    
            # Calculate erosion rate via equation
            rainfall_energy = calculate_rainfall_energy(rainfall_intensity)
            rainfall_factor = rainfall_energy * rainfall_intensity
            erosion_rate = rainfall_factor * .3 * ls * .4 * p
            
            # Applying rate to all time points... i.e. fixed fixed erosion rate * fixed time points
            all_trials_erosion[trial_number] = erosion_rate * time_points
        # DYANMIC CASE: calculate erosion at each time step
        else:
            # For dynamic case, calculate erosion at each time step
            cumulative_erosion = 0
            
            for time_step in range(number_of_steps):
                current_time = time_points[time_step]

                # Generate rainfall event (NOTE: climate factor possibility included here)          
                rainfall_intensity = generate_rainfall_event(climate_factor=climate_factor)
                
                # Calculating rainfall factor via random weather event
                rainfall_energy = calculate_rainfall_energy(rainfall_intensity)
                rainfall_factor = rainfall_energy * rainfall_intensity
                
                # Geting current k and c factors via their function
                current_k = calculate_k_factor(current_time)
                current_c = calculate_c_factor(current_time)
                
                # Calculate erosion for this timestep using total equation
                timestep_erosion = rainfall_factor * current_k * ls * current_c * p * timestep_years
                
                # Add to cumulative erosion total
                cumulative_erosion += timestep_erosion
                all_trials_erosion[trial_number, time_step] = cumulative_erosion
    
    # Calculating average erosion via all trials ran
    average_erosion = np.mean(all_trials_erosion, axis=0)
    
    return time_points, average_erosion

def plot_erosion_comparison(time_points, results, labels, title=None, filename= "no_file_name_provided1"):
    '''
    Create and save a plot comparing multiple erosion scenarios.

    Parameters
    ----------
    time_points : array-like
        Array of time values for x-axis.
    results : list of array-like
        List of erosion results to plot, one array per scenario.
    labels : list of str
        List of labels corresponding to each erosion result.
    title : str, optional
        Plot title. If None, uses default title.
    filename : str, optional
        Filename to save the plot. If None, plot is not saved.

    Returns
    -------
    None
        Saves plot to file if filename is provided.
    '''
    # Setting up figure
    plt.figure(figsize=(12, 7))
    number_of_scenarios = len(results)
    line_colors = plt.cm.viridis(np.linspace(0, 1, number_of_scenarios))
    
    # Plotting
    for erosion_result, scenario_label, line_color in zip(results, labels, line_colors):
        plt.plot(time_points, erosion_result, '-', 
                color=line_color, 
                label=scenario_label, 
                linewidth=2)
    
    # Add labels and title
    plt.xlabel('Time (years)')
    plt.ylabel('Cumulative Soil Loss (t/ha)')
    if title is None:
        title = 'Cumulative Soil Erosion Over Time'
    plt.title(title)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    plt.close()

def main():    
    # !!!!Q2 SOLUTION!!!!
    time_points, static_erosion = calculate_erosion(years=10, ls=1.5, p=1.0, static=True, fixed_rainfall=2.0)
    
    plot_erosion_comparison(time_points, [static_erosion], ['Static Implementation'],
                            'Static Erosion Model over 10 Years','figure1_static_erosion.png')
    
    # !!!!Q3 SOLUTION!!!!

    # For 10-year
    time_points_10, erosion_10 = calculate_erosion(years=10, ls=1.5, p=1.0, static=False, num_trials=50)
    plot_erosion_comparison(time_points_10, [erosion_10], ['10-year Evolution'],'Short-term Erosion Patterns',
                            'figure2a_10year.png')
    # For 50-year
    time_points_50, erosion_50 = calculate_erosion(years=50, ls=1.5, p=1.0, static=False, num_trials=50)

    plot_erosion_comparison(time_points_50, [erosion_50], ['50-year Evolution'], 'Long-term Erosion Patterns', 'figure2b_50year.png')

    # !!!!Q4 SOLUTION!!!!

    # Baseline
    time_points, baseline = calculate_erosion(years=10,ls=1.5, static=False, num_trials=50, climate_factor=1.0, p=1.0)
    # Climate change
    _, climate_change = calculate_erosion(years=10,ls=1.5,static=False, num_trials=50,climate_factor=1.5,p=1.0)
    # Climate change with conservation
    _, conservation = calculate_erosion(years=10,ls=1.5,static=False,num_trials=50,climate_factor=1.5,p=0.6)
    # Plotting them toether
    plot_erosion_comparison(time_points,[baseline, climate_change, conservation],
                            ['Baseline', 'Climate Change', 'Climate Change + Conservation'],
                            'Impact of Climate Change and Conservation Practices',
                            'figure3_climate_conservation.png'
    )

if __name__ == "__main__":
    main()
