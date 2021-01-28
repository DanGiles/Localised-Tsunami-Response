# Localised-Tsunami-Response

Extended Green's Law
=====

Python code with C bindings which calculates the tsunami amplification factor (beta)
for various sites.

## Compiling the C bindings

The gradient descent approach which optimises for the beta values is written in C.
In order for these to work with the python code, one must first compile the C code
into a shared library. Please run the following command in the src folder:

cc -fPIC -shared -o cfuncs.so gradient_descent.c

## File Structures

All the data; Bathymetry, Coarse Grid Simulations and Fine Grid Simulations, must be stored in
a ../Data folder. Please ensure that you set up the paths correctly.

Each site has its own main.py file located in the main folder.
The various parameters associated with the optimisation are specified in the params.py file.

## Workflow

- coarse_forecast.py : Takes the maximum wave heights from the coarse forecasts and uses
Green's Law to forecast to a specified highlim.

- coarse_to_fine.py : Interpolates the maximum wave height on the coarse grid to the
deepest point in a fine grid.

- beta_calculate.py : Optimises for the beta parameter and forecasts for the maximum
wave heights in a fine grid.

- gauges.py : Calculates the wave heights at the gauge locations in the fine grids,
calculates the relevant errors and then plots for each site.

## Example
To carry out the above for the Villefranche Sur Mer site one simply runs
the following in the main folder:

python main_vlfr.py

## Outputs
The code will output the optimised beta values for each site along with the forecasted
maximum wave heights.
