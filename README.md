# Investigating the Structure-Function Relationships of Fluorinated polymeric interfaces for PFAS Capture and Release

This repository contains the Python code used to generate the calculations and the figures in the manuscript.

## MD simulation data 
Simulations were conducted for 9 systems : 1CF_copol_TFA, 1CF_copol_PFBA, 1CF_copol_PFOA, 3CF_copol_TFA, 3CF_copol_PFBA, 3CF_copol_PFOA,
7CF_copol_TFA, 7CF_copol_PFBA, 7CF_copol_PFOA. All the MD trajectory files with descriptions for the file structure
can be found at https://uofi.box.com/s/frus9cjl93m3fcn1drcgx0npkbq2thcp .

##  Annealing Script
Annealing.py : OpenMM script for a seven-step compression-relaxation equilibriation process for fabrication of the polymer matrix.

## Preferential Interaction coefficient calculation:
* PFAS_water_count.py is used to extract the time evolution of PFAS and water counts as a function of z-coordinate the MD trajectories
* PIC_vs_Z.ipynb contains example code used to generate the plots for PIC(z) at a constant time .
* PIC_vs_time.ipynb contains example code used to generate the plots for PIC(t) at a fixed z.

## PFAS agglomeration and penetration depth 
* PFAS_density.py is used to extract the time evolution of the number of PFAS at discrete z intervals from the MD trajectories.
* AgglomerationPlots.ipynb contains code to generate the plots for discrete PFAS number density  and Cumulative PFAS density 
