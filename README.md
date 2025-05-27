# VoroParaSense 
[![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--4930-d45815.svg)](https://doi.org/10.18419/darus-4930)

A Voronoi cell interface-based parameter sensitivity analysis technique.
Reference implementation for the paper [Voronoi Cell Interface-Based Parameter Sensitivity Analysis for Labeled Samples](https://doi.org/10.1111/cgf.70122).

## Install
Tested with [Python 3.12.3](https://www.python.org/downloads/release/python-3123/).

1. Setup virtual environment with Python: `python -m venv .venv`
2. Activate the environment, then install the requirements via: `pip install -r requirements.txt`
3. Run `python ./main.py` to run the main example.

## Figures
Some of the figures contained in the Jupyter Notebooks were used to generate the [CCY BY 4.0](https://creativecommons.org/licenses/by/4.0/) licensed figures in [https://doi.org/10.1111/cgf.70122](https://doi.org/10.1111/cgf.70122).
The figures here are the raw `.png` figures. 
The published figures are mostly vector graphics and were slightly modified compared to the figures rendered in the Jupyter notebooks. 

- Author: Ruben Bauer
- Source: [https://doi.org/10.1111/cgf.70122](https://doi.org/10.1111/cgf.70122)
- License: [CCY BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  
The figures of the main paper are:
- [examples\notebooks\plot_examples\two_d_clipping_and_distribution.ipynb](examples\notebooks\plot_examples\two_d_clipping_and_distribution.ipynb) used to generate [Figure 2](https://doi.org/10.1111/cgf.70122)
- [examples\notebooks\plot_examples\three_dimensional_case.ipynb](examples\notebooks\plot_examples\three_dimensional_case.ipynb) used to generate [Figure 3](https://doi.org/10.1111/cgf.70122)
- [examples\notebooks\plot_examples\two_d_clipping_and_distribution.ipynb](examples\notebooks\plot_examples\two_d_clipping_and_distribution.ipynb) used to generate [Figure 4](https://doi.org/10.1111/cgf.70122)
- [examples\notebooks\plot_examples\two_d_major_transition_directions.ipynb](examples\notebooks\plot_examples\two_d_major_transition_directions.ipynb) used to generate [Figure 5](https://doi.org/10.1111/cgf.70122)
- [examples\notebooks\single_dataset_convenience_plots\iris_conv_plot.ipynb](examples\notebooks\plot_examples\two_d_major_transition_directions.ipynb) used to generate [Figure 6, 7, 8, 9, 10, 12](https://doi.org/10.1111/cgf.70122)
- [examples\notebooks\plot_examples\space_dividing_line.ipynb](examples\notebooks\plot_examples\two_d_major_transition_directions.ipynb) used to generate [Figure 11](https://doi.org/10.1111/cgf.70122)
- [examples\notebooks\single_dataset_convenience_plots\semiconductor_conv_plot.ipynb](examples\notebooks\single_dataset_convenience_plots\semiconductor_conv_plot.ipynb) used to generate [Figure 14](https://doi.org/10.1111/cgf.70122)
- [examples\notebooks\single_dataset_convenience_plots\droplet_impact_conv_plot.ipynb](examples\notebooks\single_dataset_convenience_plots\droplet_impact_conv_plot.ipynb) used to generate [Figure 15, 16](https://doi.org/10.1111/cgf.70122)

For the supplemental material:
- [examples\notebooks\plot_examples\normal_vectors_angle_vis.ipynb](examples\notebooks\plot_examples\normal_vectors_angle_vis.ipynb) used to generate [Figure 1](https://doi.org/10.1111/cgf.70122)
- [examples\notebooks\plot_examples\plane_plane_distance.ipynb](examples\notebooks\plot_examples\plane_plane_distance.ipynb) used to generate [Figure 2](https://doi.org/10.1111/cgf.70122)
- [examples\notebooks\plot_examples\runtime_experiments_2.ipynb](examples\notebooks\plot_examples\runtime_experiments_2.ipynb) used to generate [Figure 3, 4](https://doi.org/10.1111/cgf.70122)
- [examples\notebooks\plot_examples\bandwidth_experiments.ipynb](examples\notebooks\plot_examples\bandwidth_experiments.ipynb) used to generate [Figure 5 - 15](https://doi.org/10.1111/cgf.70122)

## Warning
This package uses a fork of the pyclustering library with a [customized cluster update](https://github.com/rbnbr/pyclustering/releases/tag/0.10.1.2-custom-cluster-update) method.
Do not use that version for regular k-Means but refer to the original [pyclustering](https://github.com/annoviko/pyclustering) implementation instead. 

## Examples
- To run the Python files example in the ./examples/ directory, go to the directory via `cd examples` and then run `python ./the_example_to_run.py`
- To run the Jupyter Notebooks in the ./examples/notebooks/ directory, select the virtual environment as Python Interpreter first, then run them.

# Cite
[paper](https://doi.org/10.1111/cgf.70122)

tbd.
