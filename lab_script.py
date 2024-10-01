"""
COURSE: Climate 410
ASSIGNMENT: Lab 2
DUE DATE: 10-01-2024 at 9:30 AM
NAME: Lars Andersland

Introduction Notes:
-------------------
General Introduction:
I have used object-oriented programming for this lab. Most notably, I have changed
my commenting convention to be more docstring oriented.

Commenting Convention:
I have docstrings for every function (as well as a summary of them all at the beginning
of the class). I have also commented individual lines of code within the functions, when appropriate.

User Input Capability:
I implemented code to both allow and error check for the user to specify parameters
and program functionality.

Code Ordering:
I defined my class first with functions in roughly the order they're used (but the driver function is last).
Then, at the very bottom, is my main() where the code executes.
"""

# Importing libraries
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
from matplotlib.colors import ListedColormap


class CompetitionModel:
    """
    A class used to represent the Lotka-Volterra competition and predator-prey models.
    
    Methods
    -------
    end_check():
        Exits the program when an invalid input is detected.
    isfloat(num):
        Checks if the provided input can be converted to a float.
    input_check():
        Parses and validates user input for model parameters and initial conditions.
    derivative_calc_comp(t, N):
        Calculates the derivatives for the competition model at a given time step.
    derivative_calc_pred(t, N):
        Calculates the derivatives for the predator-prey model at a given time step.
    euler_solve(func, time_step):
        Solves the system of equations using the Euler method for numerical integration.
    rk8_solve(func):
        Solves the system of equations using the RK8 method for numerical integration.
    display_results(t_euler_c, N1_euler_c, N2_euler_c, t_RK8_c, N1_RK8_c, N2_RK8_c, t_euler_p, N1_euler_p, N2_euler_p, t_RK8_p, N1_RK8_p, N2_RK8_p):
        Generates and saves plots for the competition and predator-prey models.
    delete_png_files():
        Deletes all existing PNG files from the directory to avoid confusion with previous runs.
    model_driver():
        Executes the entire simulation process from input validation to result display.
    """

    ### INPUT PARSING AND ERROR CHECKING ###

    def end_check(self):
        """
        Terminates the program when an invalid input is detected and prints an error message.
        """
        print("Invalid input detected. EXITING PROGRAM")
        sys.exit(1)

    # NOTE: I pulled this from online (https://www.programiz.com/python-programming/examples/check-string-number)
    # because Python doesn't have a built-in "is-float" function and I wasn't sure how to make it
    def isfloat(self, num):
        """
        Determines if the provided input can be converted to a floating-point number.
        
        Parameters
        ----------
        num : str
            The string input to check.
        
        Returns
        -------
        bool
            True if the input can be converted to a float, False otherwise.
        """
        try:
            float(num)
            return True
        except ValueError:
            return False

    def input_check(self):
        """
        Parses and validates user input for model parameters and initial conditions. Ensures that all inputs are valid floats and 
        time steps are non-negative.
        """

        # Time Length (can't be negative)
        self.time_length = input("Time length (years): ")
        if not self.isfloat(self.time_length): self.end_check()
        self.time_length = float(self.time_length)
        if not self.time_length >= 0: self.end_check()

        # Time Step Competition (can't be negative)
        self.time_step_comp = input("Euler time step competition equation (years): ")
        if not self.isfloat(self.time_step_comp): self.end_check()
        self.time_step_comp = float(self.time_step_comp)
        if not self.time_step_comp >= 0: self.end_check()

        # Time Step Predator-Prey (can't be negative)
        self.time_step_pred = input("Euler time step predator-prey equation (years): ")
        if not self.isfloat(self.time_step_pred): self.end_check()
        self.time_step_pred = float(self.time_step_pred)
        if not self.time_step_pred >= 0: self.end_check()

        # Time Step RK* (can't be negative)
        self.max_time_step = input("RK8 max time step (years): ")
        if not self.isfloat(self.max_time_step): self.end_check()
        self.max_time_step = float(self.max_time_step)
        if not self.max_time_step >= 0: self.end_check()

        # Parameter a
        self.a = input("Parameter a: ")
        if not self.isfloat(self.a): self.end_check()
        self.a = float(self.a)

        # Parameter b
        self.b = input("Parameter b: ")
        if not self.isfloat(self.b): self.end_check()
        self.b = float(self.b)

        # Parameter c
        self.c = input("Parameter c: ")
        if not self.isfloat(self.c): self.end_check()
        self.c = float(self.c)

        # Parameter d
        self.d = input("Parameter d: ")
        if not self.isfloat(self.d): self.end_check()
        self.d = float(self.d)

        # N1 Initial Condition
        self.N1_init = input("Species N1 Initial Population: ")
        if not self.isfloat(self.N1_init): self.end_check()
        self.N1_init = float(self.N1_init)

        # N2 Initial Condition
        self.N2_init = input("Species N2 Initial Population: ")
        if not self.isfloat(self.N2_init): self.end_check()
        self.N2_init = float(self.N2_init)

    ### EULER AND RK8 SOLVING ###

    def derivative_calc_comp(self, t, N):
        """
        Computes the rate of change of species populations for the competition model using the provided parameters.
        
        Parameters
        ----------
        t : float
            The current time step.
        N : list
            A list containing the population sizes for species N1 and N2.
        
        Returns
        -------
        tuple
            The derivatives for species N1 and N2 at the current time step.
        """

        # Calculating and returning the derivatives
        dN1dt = self.a*N[0]*(1-N[0]) - self.b*N[0]*N[1]
        dN2dt = self.c*N[1]*(1-N[1]) - self.d*N[1]*N[0]
        return dN1dt, dN2dt
    
    def derivative_calc_pred(self, t, N):
        """
        Computes the rate of change of species populations for the predator-prey model using the provided parameters.
        
        Parameters
        ----------
        t : float
            The current time step.
        N : list
            A list containing the population sizes for prey (N1) and predator (N2).
        
        Returns
        -------
        tuple
            The derivatives for prey (N1) and predator (N2) at the current time step.
        """
        # Calculating and returning the derivatives
        dN1dt = self.a*N[0] - self.b*N[0]*N[1]
        dN2dt = -self.c*N[1] + self.d*N[1]*N[0]
        return dN1dt, dN2dt

# NOTE: I added in a condition that, rather than crashing due to overflow, Euler would stop
# calculating after reaching a value of 10000
    def euler_solve(self, func, time_step):
            """
            Solves the system of two functions via Euler's Method with overflow detection.
            
            Parameters
            ----------
            func : function
                The function that calculates the derivatives for the model.
            time_step : float
                The step size for the Euler method.
                
            Returns
            -------
            tuple
                Time array, and arrays of population sizes for species N1 and N2.
            """
            
            # Initializing time and solution arrays
            t = np.arange(0.0, self.time_length + time_step, time_step)
            N1 = np.zeros(t.size)
            N2 = np.zeros(t.size)
            N1[0] = self.N1_init
            N2[0] = self.N2_init
            
            # Integration step with overflow check
            for i in range(t.size - 1):
                dN1, dN2 = func(t[i], [N1[i], N2[i]])
                N1[i + 1] = N1[i] + time_step * dN1
                N2[i + 1] = N2[i] + time_step * dN2

                # Check for overflow
                if abs(N1[i + 1]) > 10000 or abs(N2[i + 1]) > 10000:
                    print(f"Euler overflow detected at time {t[i+1]:.2f}. Prematurely stopping Euler calculation.")
                    return t[:i+2], N1[:i+2], N2[:i+2]
            
            return t, N1, N2

    def rk8_solve(self, func):
        """
        Solves the system of differential equations using the RK8 method for numerical integration.
        
        Parameters
        ----------
        func : function
            The function that calculates the derivatives for the model.
        
        Returns
        -------
        tuple
            Arrays representing the time steps and population sizes for species N1 and N2.
        """
        from scipy.integrate import solve_ivp
        # Configure the initial value problem solver
        result = solve_ivp(func, [0, self.time_length], [self.N1_init, self.N2_init],
        method='DOP853', max_step= self.max_time_step)
        # Perform the integration
        time, N1, N2 = result.t, result.y[0, :], result.y[1, :]
        # Return values to caller.
        return time, N1, N2
    
    def display_results(self, t_euler_c, N1_euler_c, N2_euler_c, t_RK8_c, N1_RK8_c, N2_RK8_c,
                        t_euler_p, N1_euler_p, N2_euler_p, t_RK8_p, N1_RK8_p, N2_RK8_p):
        """
        Generates and saves plots for both the competition and predator-prey models, including side-by-side comparison,
        individual competition plot, and predator-prey phase plot.

        Parameters
        ----------
        t_euler_c : array
            Time steps for the Euler competition model.
        N1_euler_c : array
            Population sizes for species N1 using the Euler method in the competition model.
        N2_euler_c : array
            Population sizes for species N2 using the Euler method in the competition model.
        t_RK8_c : array
            Time steps for the RK8 competition model.
        N1_RK8_c : array
            Population sizes for species N1 using the RK8 method in the competition model.
        N2_RK8_c : array
            Population sizes for species N2 using the RK8 method in the competition model.
        t_euler_p : array
            Time steps for the Euler predator-prey model.
        N1_euler_p : array
            Population sizes for prey (N1) using the Euler method in the predator-prey model.
        N2_euler_p : array
            Population sizes for predator (N2) using the Euler method in the predator-prey model.
        t_RK8_p : array
            Time steps for the RK8 predator-prey model.
        N1_RK8_p : array
            Population sizes for prey (N1) using the RK8 method in the predator-prey model.
        N2_RK8_p : array
            Population sizes for predator (N2) using the RK8 method in the predator-prey model.
        
        Returns
        -------
        None
            This function saves three plot files: 
            1. Side-by-side plots for both competition and predator-prey models.
            2. An individual plot for the competition model.
            3. Phase plots for the predator-prey model.
        """
        # NOTE: there's apparent code duplication here, but it was difficult to condense
        # due to saving three different plots (i.e. each plot needed its own separate code)

        # 1. Side-By-Side Competition And Predator-Prey Plots
        # (following structure shown in lab report)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Competition Model
        ax1.set_title('Lotka-Volterra Competition Model')
        ax1.plot(t_euler_c, N1_euler_c, label='N1 Euler', color='C0')
        ax1.plot(t_euler_c, N2_euler_c, label='N2 Euler', color='C1')
        ax1.plot(t_RK8_c, N1_RK8_c, label='N1 RK8', linestyle='--', color='C2')
        ax1.plot(t_RK8_c, N2_RK8_c, label='N2 RK8', linestyle='--', color='C3')
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('Population/Carrying Cap.')
        ax1.legend(loc='best')

        # Predator-Prey Model
        ax2.set_title('Lotka-Volterra Predator-Prey Model')
        ax2.plot(t_euler_p, N1_euler_p, label='Prey (Euler)', color='C0')
        ax2.plot(t_euler_p, N2_euler_p, label='Predator (Euler)', color='C1')
        ax2.plot(t_RK8_p, N1_RK8_p, label='Prey (RK8)', linestyle='--', color='C2')
        ax2.plot(t_RK8_p, N2_RK8_p, label='Predator (RK8)', linestyle='--', color='C3')
        ax2.set_xlabel('Time (years)')
        ax2.set_ylabel('Population/Carrying Cap.')
        ax2.legend(loc='best')

        # Coefficients and Time Steps
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.figtext(0.5, 0.04, f'Coefficients: a = {self.a}, b = {self.b}, c = {self.c}, d = {self.d}', 
                    ha="center", fontsize=10, style='italic')
        plt.figtext(0.5, 0.01, 
            f'Time Steps (years): Competition = {self.time_step_comp}, Predator-Prey = {self.time_step_pred}, RK8 Max = {self.max_time_step}', 
            ha="center", fontsize=10, style='italic')

        # Saving and showing the plot
        fig.savefig('lotka_volterra_solution.png')
        plt.show()

        # 2. Competition graph alone
        fig_comp, comp_mod_alone = plt.subplots(figsize=(6, 6))  # New figure for the single subplot

        # Plot only the Competition Model
        comp_mod_alone .set_title('Lotka-Volterra Competition Model')
        comp_mod_alone .plot(t_euler_c, N1_euler_c, label='N1 Euler', color='C0')
        comp_mod_alone .plot(t_euler_c, N2_euler_c, label='N2 Euler', color='C1')
        comp_mod_alone .plot(t_RK8_c, N1_RK8_c, label='N1 RK8', linestyle='--', color='C2')
        comp_mod_alone .plot(t_RK8_c, N2_RK8_c, label='N2 RK8', linestyle='--', color='C3')
        comp_mod_alone .set_xlabel('Time (years)')
        comp_mod_alone .set_ylabel('Population/Carrying Cap.')
        comp_mod_alone .legend(loc='best')

        # Coefficients and Time Steps
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.figtext(0.5, 0.04, f'Coefficients: a = {self.a}, b = {self.b}, c = {self.c}, d = {self.d}', 
                    ha="center", fontsize=10, style='italic')
        plt.figtext(0.5, 0.01, f'Time Steps (years): Euler = {self.time_step_comp}, RK8 Max = {self.max_time_step}', 
                    ha="center", fontsize=10, style='italic')

        # Saving Plot
        fig_comp.savefig('lotka_volterra_competition_model.png')

        # 3. Phase Plots For The Predator-Prey Equation

        # Euler Phase Plot
        fig_euler, ax_euler = plt.subplots(figsize=(6, 6))
        ax_euler.set_title('Predator-Prey Model Phase Plot (Euler)')
        ax_euler.plot(N1_euler_p, N2_euler_p, label='Prey vs Predator (Euler)', color='C0')
        ax_euler.set_xlabel('Prey Population (N1)')
        ax_euler.set_ylabel('Predator Population (N2)')
        ax_euler.legend(loc='best')

        # Coefficients and Time Step for Euler
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.figtext(0.5, 0.04, f'Coefficients: a = {self.a}, b = {self.b}, c = {self.c}, d = {self.d}', 
                    ha="center", fontsize=10, style='italic')
        plt.figtext(0.5, 0.01, f'Euler Time Step (years): {self.time_step_pred}, Simulation Duration: {self.time_length} years', 
            ha="center", fontsize=10, style='italic')

        # Saving plot
        fig_euler.savefig('predator_prey_phase_plot_euler.png')
        plt.show()

        # RK8 Phase Plot
        fig_rk8, ax_rk8 = plt.subplots(figsize=(6, 6))
        ax_rk8.set_title('Predator-Prey Model Phase Plot (RK8)')
        ax_rk8.plot(N1_RK8_p, N2_RK8_p, label='Prey vs Predator (RK8)', linestyle='--', color='C1')
        ax_rk8.set_xlabel('Prey Population (N1)')
        ax_rk8.set_ylabel('Predator Population (N2)')
        ax_rk8.legend(loc='best')

        # Coefficients and Time Step for RK8
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.figtext(0.5, 0.04, f'Coefficients: a = {self.a}, b = {self.b}, c = {self.c}, d = {self.d}', 
                    ha="center", fontsize=10, style='italic')
        plt.figtext(0.5, 0.01, f'RK8 Max Time Step (years): {self.max_time_step}, Simulation Duration: {self.time_length} years', 
            ha="center", fontsize=10, style='italic')
        

        # Saving plot
        fig_rk8.savefig('predator_prey_phase_plot_rk8.png')
        plt.show()

        # Combined Phase Plot
        fig_combined, ax_combined = plt.subplots(figsize=(6, 6))
        ax_combined.set_title('Combined Predator-Prey Phase Plot (Euler & RK8)')
        ax_combined.plot(N1_euler_p, N2_euler_p, label='Prey vs Predator (Euler)', color='C0')
        ax_combined.plot(N1_RK8_p, N2_RK8_p, label='Prey vs Predator (RK8)', linestyle='--', color='C1')
        ax_combined.set_xlabel('Prey Population (N1)')
        ax_combined.set_ylabel('Predator Population (N2)')
        ax_combined.legend(loc='best')

        # Coefficients and Time Step for Combined Plot
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.figtext(0.5, 0.04, f'Coefficients: a = {self.a}, b = {self.b}, c = {self.c}, d = {self.d}', 
                    ha="center", fontsize=10, style='italic')
        plt.figtext(0.5, 0.01, f'Time Steps (Euler: {self.time_step_pred}, RK8: {self.max_time_step}), Simulation Duration: {self.time_length} years', 
                    ha="center", fontsize=10, style='italic')

        # Saving the combined plot
        fig_combined.savefig('predator_prey_phase_plot_combined.png')
        plt.show()
    
    ### DRIVER AND MISC FUNCTIONS ###

    # NOTE = I had no idea how to do this in Python so I got help online...
    # here's a place online where the functionality is documented:
    # https://favtutor.com/blogs/delete-file-python
    def delete_png_files(self):
        """
        Deletes all .png files in the current directory to avoid confusion with previous results.
        """
        for file in os.listdir():
            if file.endswith(".png"):
                os.remove(file)

    def model_driver(self):
        """
        Executes the main workflow of the model, including input validation, solving the competition and predator-prey models,
        and displaying results.
        """
        # Deleting past .png files
        self.delete_png_files()

        # Processing and error-checking input
        self.input_check()

        # Numerically Solving Competition and Predator-Prey Functions
        t_RK8_p, N1_RK8_p, N2_RK8_p = self.rk8_solve(self.derivative_calc_pred)
        t_Euler_p, N1_Euler_p, N2_Euler_p = self.euler_solve(self.derivative_calc_pred, self.time_step_pred)
        t_RK8_c, N1_RK8_c, N2_RK8_c = self.rk8_solve(self.derivative_calc_comp)
        t_Euler_c, N1_Euler_c, N2_Euler_c = self.euler_solve(self.derivative_calc_comp, self.time_step_comp)
        
        

        # Displaying Results
        self.display_results(t_Euler_c, N1_Euler_c, N2_Euler_c, t_RK8_c, N1_RK8_c, N2_RK8_c,
                        t_Euler_p, N1_Euler_p, N2_Euler_p, t_RK8_p, N1_RK8_p, N2_RK8_p)

### MAIN() EXECUTING ENTIRE PROGRAM ###

if __name__ == "__main__":
        # Print welcome message
        print("Welcome to the CLIMATE 410 model for simulating competition with differential equations and numerical methods!")
        print("Please specify your model parameters:")
        # Declare model member
        model = CompetitionModel()
        # Run driver function
        model.model_driver()
