# CRRT Effect Simulation and Visualization
## Project Overview
This project uses the R language to simulate patient data under different effects (beneficial, detrimental, and neutral) of Continuous Renal Replacement Therapy (CRRT). It then employs a random forest model for prediction and finally generates heatmap visualizations. The code aims to explore the relationship between treatment recommendations and actual and counterfactual outcomes under different CRRT effects.

## Features
### Data Generation: 
Generate patient covariate data based on preset distributions and parameters.
### Data Cleaning: 
Clean and transform the generated covariate data, including handling outliers and converting numerical variables to appropriate ranges and types.
### Simulation of Different CRRT Effects: 
Simulate patient treatment and outcome data for three scenarios: beneficial, detrimental, and neutral effects of CRRT.
### Model Building: 
Use a random forest model to predict patient treatment outcomes.
### Visualization: 
Generate heatmaps based on different prediction thresholds to show the relationship between treatment recommendations and actual and counterfactual outcomes. Save the heatmaps as JPEG and PDF files.

## Notes
The random seed in the code is set to 42 to ensure the reproducibility of the results. If you need different random results, you can modify the value in set.seed(42).
The patient data generated in the code is simulated data and does not represent real clinical data.
