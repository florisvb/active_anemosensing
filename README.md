
# Overview

This repository provides all of the code used to analyze the data, run the simulations, and generate the figures, described in the manuscript titled "Active Anemosensing Hypothesis: How Flying Insects Could Estimate Ambient Wind Direction Through Sensory Integration & Active Movement". Authors: Floris van Breugel, Renan Jewell, and Jaleesa Houle. 


# Directory structure

    ├── data_simulations
    │   ├── .download_from_data_dryad
    ├── data_experiments_clean
    │   ├── .download_from_data_dryad
    ├── data_experiments_raw
    │   ├── .download_from_data_dryad
    ├── data_experiments_preprocessed
    │   ├── .download_from_data_dryad
    ├── data_tmp
    │   ├── .temporary_storage_for_intermediate_processing_steps
    ├── code
    │   ├── convex_solve_tan_cot.py
    │   ├── direct_trajectory_simulator.py
    │   ├── run_ALL_analysis.py
    │   ├── run_analysis.py
    │   ├── utility.py
    │   ├── plot_utility.py
    ├── figures_png
    ├── figures_svg
    │   ├── fig_1
    │   │   ├── fly_cartoon_2panel.svg
    │   │   └── fly_coordinates.ai
    │   ├── fig_2_anemotaxis.svg
    │   ├── fig_3_algorithm_flow.svg
    │   ├── fig_4_wind.svg
    │   ├── fig_5_fly_trajectories.svg
    │   ├── fig_6_botfly.svg
    │   ├── fig_7_cvx_estimate_overview.svg
    │   ├── fig_8_time_constants.svg
    │   ├── fig_9_absine_aligngamma_realdynamic_Tmult1.svg
    │   ├── fig_9_absine_alignpsi_realconstant_Tmult1.svg
    │   ├── fig_9_absine_alignpsi_realdynamic_Tmult1.svg
    │   ├── fig_9_constant_alignpsi_realdynamic_Tmult1.svg
    │   ├── fig_10_summary.svg
    │   ├── fig_11_gamma_analysis.svg
    │   ├── fig_S1_bestomega.svg
    │   └── fig_S2_bestT.svg
    ├── notebooks_to_generate_figures
    │   ├── fig_2_anemotaxis.ipynb
    │   ├── fig_3_algorithm_flow_figure.ipynb
    │   ├── fig_4B_wind.ipynb
    │   ├── fig_4CD_wind.ipynb
    │   ├── fig_5_fly_trajectories.ipynb
    │   ├── fig_6_botfly.ipynb
    │   ├── fig_7_cvx_estimate_overview.ipynb
    │   ├── fig_8_time_constants.ipynb
    │   ├── fig_9_3noise_heatmaps.ipynb
    │   ├── fig_10_summary.ipynb
    │   ├── fig_11_gamma_analysis.ipynb
    │   ├── fig_S1_best_omega.ipynb
    │   └── fig_S2_bestT.ipynb
    ├── preprocess_raw_botfly_data
    │   ├── step_1_read_interpolate_and_merge_raw_botfly_data.ipynb
    │   └── step_2_convert_botfly_data_to_compatible_dataframe.ipynb
    ├── preprocess_raw_wind_data
    │   ├── load_windgps_data_to_pandas.py
    │   ├── process_windgps_data.py
    │   ├── step_1_process_windgps_data_notebook.ipynb
    │   └── step_2_merge_and_convert_wind_data.ipynb
    └── README.md

# Data format

All of the cleaned data and simulation data is provided as hdf files. Raw data is provided in either .csv or .bin files. See preprocessing notebooks for instructions on how to read these raw data formats.  

# Downloading the data

Raw and cleaned experimental data, as well as the simulation results, are available from the Data Dryad repository: LINK. Download this data and place into the appropriate directories for this repository (which are empty).  

# Processing raw data

First download the raw data from Data Dryad as described above. To prepare the raw data for analysis, you will need to run the following notebooks. 

If you only wish to regenerate the figures, you can skip this step and just use the provided cleaned data.

### Preprocess raw wind data

1. Run the notebook `preprocess_raw_wind_data/step_1_process_windgps_data_notebook.ipynb`. This reads the binary data files for the horizontal and vertical wind stations and saves each as a hdf file in the `data_experiments_preprocessed` directory.
2. Run the notebook `preprocess_raw_wind_data/step_2_merge_and_convert_wind_data.ipynb`. This reads the two hdf files in the `data_experiments_preprocessed` directory and interpolates them to a common time base, and converts the coordinates from magnetic north to standard mathematical coordinates (where an angle of zero degrees corresponds to True East). The resulting hdf file is saved in `data_experiments_preprocessed`, and is then manually copied over to the `data_experiments_clean` directory for further use.

### Preprocess raw botfly data

1. Run the notebook `preprocess_raw_botfly_data/step_1_read_interpolate_and_merge_raw_botfly_data.ipynb`. This reads all the individual .csv files for the botfly experiment, as well as the stationary wind station data, and interpolates them all to a common time base. The resulting hdf file is saved to `data_experiments_preprocessed`.
2. Run the notebook `preprocess_raw_botfly_data/step_2_convert_botfly_data_to_compatible_dataframe.ipynb`. This reads the interpolated hdf file for the botfly data and converts the coordinate frames from latitude/longitude, magnetic north, and navigation standards to a the standard mathematical framework where zero degrees corresponds to True East, and positive angles are defined as counter clockwise. The resulting hdf file is saved in `data_experiments_preprocessed`, and is then manually copied over to the `data_experiments_clean` directory for further use.

### Fly trajectory data

The fly trajectory data is only provided as a cleaned hdf file. See manuscript for processing details. The raw data can be acquired from this data dryad repository: https://datadryad.org/stash/dataset/doi:10.5061/dryad.n0b8m. 

# Running the simulations

First download the data from Data Dryad as described above (the wind station data is required for the simulations). Then run the python script `code/run_ALL_analysis.py`. This will run all of the simulations that are discussed in the paper. Running this script may take one or more weeks on a powerful desktop. It can be run more efficiently by running different conditions in parallel using separate scripts, see `code/run_analysis.py` for an example of how to run a single condition (which may take a few days). 

If you only wish to regenerate the figures, you can skip this step and just use the data provided in the Data Dryad link.


# Generating the figures

Each figure is generated by a jupyter notebook found in the directory `notebooks_to_generate_figures`. The notebook will read the appropriate clean experimental data and/or simulation data and plot the results into the svg templates found in the directory `figures_svg`. It is important to run the notebooks in order the first time, as some of the later notebooks rely on temporary data files generated by the `fig_4CD_wind.ipynb` notebook. 

