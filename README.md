# DigitalTwins

Cardiac Mesh Deformation

Research project on patient-specific cardiac mesh deformation, conducted remotely with GSSI (Gran Sasso Science Institute), Italy.

What this does:
Given a known deformation of the endocardium (inner heart wall) fitted to a patient’s CT scan, the goal is to extend that deformation to the full cardiac template, including the epicardium (outer wall), using spatial interpolation methods.

Three methods are compared:

RBF (Radial Basis Function interpolation)
IDW (Inverse Distance Weighting)
IDW-KNN (IDW with K Nearest Neighbors)

Quality is evaluated using wall thickness preservation and surface mesh quality metrics.
