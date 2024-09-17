# COURSE: Climate 410 
# ASSIGNMENT: Lab 1
# DUE DATE: 09-17-2024
# NAME: Lars Andersland

#
### Introduction Notes###
#

# Commenting Convention
# WHAT = for code block titles that are not not self-explanatory, I write "WHAT =" on the following line and
# explain what it refers to in more detail (ex. as I have done here)

# User Input Capability
# WHAT = even though it wasn't assigned in the project, I implemented code to both allow and error check
# for the user to specify parameters (ex. spread probability) and program functionality (ex. disease vs. fire)

# Code Ordering
# WHAT = I defined my class first with functions in roughly the order they're used (but driver = last one)
# then, at the very bottom, is my main() where the code executes

#
### Code Implementation
#

# Importing libraries
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
from matplotlib.colors import ListedColormap

# SpreadModel Class
# WHAT = class encompassing the entirety of model spread code. I prefer this object-oriented approach
# because it provides a very clean implementation-interface separation
class SpreadModel:

    # Error Exit
    # WHAT = exits program when an invalid input is given
    def end_check(self):
        print("Invalid input detected. EXITING PROGRAM")
        sys.exit(1)

    # Float Checking
    # WHAT = checks if value is float
    # NOTE: I pulled this from online (https://www.programiz.com/python-programming/examples/check-string-number)
    # because Python doesn't have a built-in "is-float" function and I wasn't sure how to make it
    def isfloat(self, num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    # Input Parsing
    # WHAT = Parses user input from the command line and error checks for every class variable
    # Error checks = is number for grid sizing and probability and is letter possibilities for mode
    # Error check = ensure all grid sizing and probability >= 0 and > 0 for num_iterations
    def input_check(self):
        # Model Type
        self.model_type = input("Model type, W (Wildfire) or D (Disease): ")
        if(self.model_type != "W" and self.model_type != "D"): self.end_check()

        # "Center case" for proving the model works (i.e. question one)
        if self.model_type == "W":
            self.center_case = input("Center cell on fire test case, Y (Yes) or N (No): ")
            if(self.center_case != "Y" and self.center_case != "N"): self.end_check()

        # X dimension
        self.nx = input("X dimension: ")
        if not self.nx.isdigit(): self.end_check()
        self.nx = int(self.nx)
        if not self.nx >= 0: self.end_check()

        # Y dimension
        self.ny = input("Y dimension: ")
        if not self.ny.isdigit(): self.end_check()
        self.ny = int(self.ny)
        if not self.ny >= 0: self.end_check()

        # If not the center case, specify probability parameters
        if self.model_type == 'D' or self.center_case == 'N':
            # Spread probability
            self.prob_spread = input("Spread probability: ")
            if not self.isfloat(self.prob_spread): self.end_check()
            self.prob_spread = float(self.prob_spread)
            if not self.prob_spread >= 0: self.end_check()

            # Bare or immune probability 
            if self.model_type == 'W': self.prob_bare = input("Bare probability: ")
            else: self.prob_bare = input("Immune probability: ")
            if not self.isfloat(self.prob_bare): self.end_check()
            self.prob_bare = float(self.prob_bare)
            if not self.prob_bare >= 0: self.end_check()

            # Start probability
            self.prob_start = input("Start probability: ")
            if not self.isfloat(self.prob_start): self.end_check()
            self.prob_start = float(self.prob_start)
            if not self.prob_start >= 0: self.end_check()

            # Fatal probability
            if self.model_type == 'D':
                self.prob_fatal = input("Fatal probability: ")
                if not self.isfloat(self.prob_fatal): self.end_check()
                self.prob_fatal = float(self.prob_fatal)
                if not self.prob_fatal >= 0: self.end_check()
            else: self.prob_fatal = 0
        # If center case, set parameters accordingly
        else:
            self.prob_spread = 1
            self.prob_bare = 0
            self.prob_start = 0

        # Number of iterations
        self.num_iterations = input("Number of iterations: ")
        if not self.num_iterations.isdigit(): self.end_check()
        self.num_iterations = int(self.num_iterations)
        if not self.num_iterations > 0: self.end_check()

        # Figure display setting
        self.figure_display = input("Display figure for every iteration or one once finished, E (Every) or F (Finished): ")
        if(self.figure_display != "E" and self.figure_display != "F"): self.end_check()
    
    # Grid Initialization
    # WHAT = initially sets grid full of trees, then sets fire cells, and then sets bare cells
    def grid_initialization(self):
        # Create an initial grid, set all values to "2". dtype sets the value type in our array to integers only.
        self.forest = np.zeros([self.ny, self.nx], dtype=int) + 2

        # NOTE: for the two code blocks below, I set fire cells first and then set bare, allowing fir
        # cells that were set to be overidden by bare cells. I do this because, intuitively, a bare
        # cell shouldn't be able to catch on fire; thus, only cells that are not initially bare
        # should be able to catch on fire

        # If center case mode, only set the center cell (via int division) to burning 
        if self.model_type == 'W':
            if self.center_case == 'Y':
                self.forest[self.ny // 2, self.nx // 2] = 3

        # Setting fire spots
        # Create an array of randomly generated numbers of range [0, 1):
        isfire = np.random.rand(self.ny, self.nx)
        # Turn it into an array of True/False values:
        isfire = isfire < self.prob_start
        # Logical indexing to change "true" spots to bare
        self.forest[isfire] = 3

        # Setting bare spots
        # Create an array of randomly generated numbers of range [0, 1):
        isbare = np.random.rand(self.ny, self.nx)
        # Turn it into an array of True/False values:
        isbare = isbare < self.prob_bare
        # Logical indexing to change "true" spots to bare
        self.forest[isbare] = 1

    # Spread Function
    # WHAT = applies probability for disease or fire spread
    def spread(self, j_neighbor, i_neighbor):
        if np.random.rand() < self.prob_spread:
            self.forest[j_neighbor, i_neighbor] = 3

    # Model Running
    # WHAT = implements model logic, running 
    def run_model(self):
        # Starting with the first iteration
        current_iterations = 0
        # Looping through all iterations of the model
        while current_iterations < self.num_iterations:

            # If desired, show iteration figure
            if(self.figure_display == 'E'):
                self.display_results(current_iterations)

            # Increment current iteration
            current_iterations += 1

            # Create grid of bools showing where fire is, 
            # with 1 = fire and 0 = no fire via logical indexing
            fire_grid = np.zeros((self.ny, self.nx))
            fire_grid[self.forest == 3] = 1

            # Applying spread logic to each cell of the grid
            for i in range(self.nx):
                for j in range(self.ny):
                    # Do nothing if spot isn't on fire (i.e. there's no spread)
                    if fire_grid[j, i] == 0: 
                        pass
                    # Otherwise, apply the spread logic to each neighbor
                    else: 
                        # If Disease Mode, check if they died...
                        # i.e. if they die they can't infect others
                        # If Forest Mode, will always evaluate to false b/c prob_fatal = 0
                        if np.random.rand() < self.prob_fatal:
                            self.forest[j, i] = 0
                        # Otherwise, apply the spread logic
                        else:
                            # Analogous logic for each of the four neighbors ensures cell
                            # "off the grid" is never attempted to be accessed
                            # Checking up
                            if j != 0 and self.forest[j - 1, i] == 2:
                                self.spread(j - 1, i)
                            # Checking down
                            if j != self.ny - 1 and self.forest[j + 1, i] == 2:
                                self.spread(j + 1, i)
                            # Checking left
                            if i != 0 and self.forest[j, i - 1] == 2:
                                self.spread(j, i - 1)
                            # Checking right
                            if i != self.nx - 1 and self.forest[j, i + 1] == 2:
                                self.spread(j, i + 1)
                            # Finally, set the current cell to bare/immune
                            self.forest[j, i] = 1
                        
                        

    # Display Results
    # WHAT = displays a color map signifying the gird
    def display_results(self, iteration_number):
        # If fire, three colors tan = bare; darkgreen = forest; crimson = fire
        if self.model_type == 'W':
            forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
            # Create figure and set of axes:
            fig, ax = plt.subplots(1,1)
            ax.pcolor(self.forest, cmap=forest_cmap, vmin=1, vmax=3)
            ax.set_title(f"Forest Fire Spread - Iteration {iteration_number}", fontsize = 20)
            ax.set_xlabel("X Coordinate", fontsize = 16)
            ax.set_ylabel("Y Coordinate", fontsize = 16)
            # Setting legend (documentation: https://stackoverflow.com/questions/39500265/how-to-manually-create-a-legend)
            legend_labels = ['Bare', 'Forested', 'Fire']
            legend_colors = ['tan', 'darkgreen', 'crimson']
            patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
            ax.legend(handles=patches, loc='upper right')  # You can change the location with the 'loc' argument
             # Set x and y ticks to only show integers
            ax.set_xticks(range(self.forest.shape[1] + 1))  # Adjust based on array dimensions
            ax.tick_params(axis='x', labelsize=12)
            ax.set_yticks(range(self.forest.shape[0] + 1))  # Adjust based on array dimensions
            ax.tick_params(axis='y', labelsize=12)
            # show the plot
            plt.savefig(f'output_fire_{iteration_number}.png')
            plt.show()
        # Otherwise, it's disease and four colors black = dead; tan = immune
        # darkgreen = uninfected; crimson = infected
        else:
            forest_cmap = ListedColormap(['black', 'tan', 'darkgreen', 'crimson'])
            # Create figure and set of axes:
            fig, ax = plt.subplots(1,1)
            ax.pcolor(self.forest, cmap=forest_cmap, vmin=0, vmax=3)
            ax.set_title(f"Disease Spread - Iteration {iteration_number}", fontsize = 20)
            ax.set_xlabel("X Coordinate", fontsize = 16)
            ax.set_ylabel("Y Coordinate", fontsize = 16)
            # Setting legend (documentation: https://stackoverflow.com/questions/39500265/how-to-manually-create-a-legend)
            legend_labels = ['Deceased', 'Immune', 'Healthy', 'Infected']
            legend_colors = ['black', 'tan', 'darkgreen', 'crimson']
            patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
            ax.legend(handles=patches, loc='upper right')  # You can change the location with the 'loc' argument
             # Set x and y ticks to only show integers
            ax.set_xticks(range(self.forest.shape[1] + 1))  # Adjust based on array dimensions
            ax.set_yticks(range(self.forest.shape[0] + 1))  # Adjust based on array dimensions
            # show the plot
            plt.savefig(f'output_disease_{iteration_number}.png')
            plt.show()
    
    # Delete Files Function
    # WHAT = deletes all .png files from previous runs to avoid confusion
    # NOTE = I had no idea how to do this in Python so I got help online...
    # here's a place online where the functionality is documented:
    # https://favtutor.com/blogs/delete-file-python
    def delete_png_files(self):
        for file in os.listdir():
            if file.endswith(".png"):
                os.remove(file)

    # Driver Function
    # WHAT = runs class functions in correct ordering
    def model_driver(self):
        # Deleting past .png files
        self.delete_png_files()
        # Processing and error-checking input
        self.input_check()
        # Initializing grid
        self.grid_initialization()
        # Running model
        self.run_model()
        # Displaying model results
        self.display_results(self.num_iterations)

# Executing Entire Program
if __name__ == "__main__":
        # Print welcome message
        print("Welcome to CLIMATE model for simulating disease and wildfire spread! Please specify your model parameters:")
        # Declare model member
        model = SpreadModel()
        # Run driver function
        model.model_driver()