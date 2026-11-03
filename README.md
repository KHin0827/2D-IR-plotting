# 2DIR Peak Tilt Analysis

A Python script that automates the analysis of 2D infrared (2DIR) spectra:

1. Loads the parallel and perpendicular 2DIR datasets.
2. Identifies the strongest positive (ESA) and negative (GSB/SE) features.
3. Refines each feature along by a quadratic fit.
4. Groups the refined positions into clusters by k-means.
5. Fits a straight line to each cluster, giving its slope, intercept, and $R^2$.
6. Writes a small CSV summary of every cluster.

The slope of each cluster quantifies a 2DIR band in the $(\omega_1,\omega_3)$ plane. It is a data-driven readout of the coupling signature/ frequency correlation over time of the bands.
