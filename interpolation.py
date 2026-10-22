"""Conceptual example: interpolate a template-subset deformation to a full mesh."""

from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree


def interpolate_displacement(source_points, source_displacement, query_points):
  """Estimate displacement at query_points from known source point displacements."""
  rbf = RBFInterpolator(
    source_points,
    source_displacement,
    kernel="thin_plate_spline",
    smoothing=1e-3,
    neighbors=80,
    #epsilon=1.0,
  )
  return rbf(query_points)

def interpolate_displacement_nn(source_points, source_displacement, query_points):
    tree = KDTree(source_points)
    _, idx = tree.query(query_points)
    return source_displacement[idx]
from scipy.spatial.distance import cdist

def dist_matrix(x0, x1):
    return cdist(x0, x1, metric="euclidean")

def interpolate_displacement_idw(source_points, source_displacement, query_points, p=4):
    dist = dist_matrix(source_points, query_points)
    weights = 1.0 / dist ** p
    weights /= weights.sum(axis=0)
    u = np.dot(weights.T, source_displacement)
    return u

def interpolate_displacement_idw_knn(source_points, source_displacement, query_points, p=4, k=10):
    tree = KDTree(source_points)
    distances, indices = tree.query(query_points, k=k)
    weights = 1.0 / distances ** p
    weights /= weights.sum(axis=1, keepdims=True)
    u = np.einsum('ij,ijk->ik', weights, source_displacement[indices])
    return u

#run_dir = Path("~/Downloads/fix_compenetration_all_real/pat000_smoothed_alignTemplate_fix_compenetration_it0").expanduser()
#reg_dir = run_dir / "template_transformed_to_combined_geometry_std"
out_dir = Path("./interpolation_concept_output")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Complete template: all anatomical structures, before patient deformation.
whole_template = pv.read("whole_template.vtk")

# 2. Bring the complete template to the same frame used by the registration.
std_tr = np.loadtxt("std_tr.txt")
#umeyama_tr = np.loadtxt(run_dir / "umemaya_tr.txt")
umeyama_tr = np.loadtxt("./umemaya_tr.txt")
whole_template.points = whole_template.points * (1.0 / 21.0)
whole_template.points = whole_template.points @ std_tr[:3, :3].T + std_tr[:3, 3]
whole_template.points = whole_template.points @ umeyama_tr[:3, :3].T + umeyama_tr[:3, 3]

# 3. Registration template at the beginning and at the end of deformation.
#template_it0 = pv.read(reg_dir / "template_transformed_source.vtp")
template_it0 = pv.read("template.vtp")
#template_deformed = pv.read(run_dir / "template_transformed_to_combined_geometry_std.vtp")
template_deformed = pv.read("./template_deformed.vtp")

if template_it0.n_points != template_deformed.n_points:
  raise ValueError("template_it0 and template_deformed must have matching points.")

# 4. Check that the registration template is really a subset of the full template.
dist, _ = KDTree(whole_template.points).query(template_it0.points)
print(f"subset check: mean distance = {dist.mean():.3e}, max = {dist.max():.3e}")

# 5. Known displacement on the registered template subset.
known_displacement = template_deformed.points - template_it0.points
template_it0["known_displacement"] = known_displacement
template_it0["known_displacement_mag"] = np.linalg.norm(known_displacement, axis=1)

# 6. Use fewer control points to keep the first experiment fast and readable.
rng = np.random.default_rng(0)
n_control = 200
control_ids = rng.choice(template_it0.n_points, size=n_control, replace=False)
control_points = template_it0.points[control_ids]
control_displacement = known_displacement[control_ids]

# 7. Interpolate / extrapolate the displacement to the complete anatomical mesh.
whole_displacement = interpolate_displacement(control_points, control_displacement, whole_template.points)
whole_deformed = whole_template.copy()
whole_deformed.points = whole_template.points + whole_displacement
whole_deformed["interpolated_displacement"] = whole_displacement
whole_deformed["interpolated_displacement_mag"] = np.linalg.norm(whole_displacement, axis=1)

whole_displacement_nn = interpolate_displacement_nn(control_points, control_displacement, whole_template.points)
whole_deformed_nn = whole_template.copy()
whole_deformed_nn.points = whole_template.points + whole_displacement_nn

# compute both
whole_displacement_idw2 = interpolate_displacement_idw(control_points, control_displacement, whole_template.points, p=4)
whole_deformed_idw2 = whole_template.copy()
whole_deformed_idw2.points = whole_template.points + whole_displacement_idw2

# whole_displacement_idw4 = interpolate_displacement_idw(control_points, control_displacement, whole_template.points, p=8)
# whole_deformed_idw4 = whole_template.copy()
# whole_deformed_idw4.points = whole_template.points + whole_displacement_idw4

# whole_displacement_idw4 = interpolate_displacement_idw(control_points, control_displacement, whole_template.points, p=8)
# whole_deformed_idw4 = whole_template.copy()
# whole_deformed_idw4.points = whole_template.points + whole_displacement_idw4

whole_displacement_knn50 = interpolate_displacement_idw_knn(control_points, control_displacement, whole_template.points, p=4, k=200)
whole_deformed_knn = whole_template.copy()
whole_deformed_knn.points = whole_template.points + whole_displacement_knn50
# 8. Save everything important for inspection in ParaView/PyVista.
whole_template.save(out_dir / "01_complete_template_patient_frame.vtk")
template_it0.save(out_dir / "02_template_it0_subset.vtp")
template_deformed.save(out_dir / "03_template_deformed_to_patient.vtp")
whole_deformed.save(out_dir / "04_complete_template_deformed_by_interpolation.vtk")
print(f"saved results in: {out_dir}")

whole_deformed.save(out_dir / "result_rbf.vtk")
whole_deformed_idw2.save(out_dir / "result_idw_p4.vtk")
whole_deformed_knn.save(out_dir / "result_idw_knn_p4_k200.vtk")

# 9. Simple visualization: original full template, known subset deformation, result.
pl = pv.Plotter(shape=(1, 5), window_size=(3840, 800))
pl.subplot(0, 0); pl.add_text("full template", font_size=10)
pl.add_mesh(whole_template, color="#BDBDBD", opacity=0.35)
pl.subplot(0, 1); pl.add_text("known deformation", font_size=10)
pl.add_mesh(template_it0, color="#1E90FF", style="wireframe", opacity=0.8)
pl.add_mesh(template_deformed, color="#DC143C", style="wireframe", opacity=0.8)

pl.subplot(0, 2); pl.add_text("RBF", font_size=10)
pl.add_mesh(whole_deformed, color="#32CD32", opacity=0.45)
pl.add_mesh(template_deformed, color="#DC143C", style="wireframe", opacity=0.8)

pl.subplot(0, 3); pl.add_text("IDW p=4", font_size=10)
pl.add_mesh(whole_deformed_idw2, color="#FF6600", opacity=0.45)
pl.add_mesh(template_deformed, color="#DC143C", style="wireframe", opacity=0.8)

pl.subplot(0, 4); pl.add_text("IDW_knn p=4, k=10", font_size=10)
pl.add_mesh(whole_deformed_knn, color="#FF6600", opacity=0.45)
pl.add_mesh(template_deformed, color="#DC143C", style="wireframe", opacity=0.8)
pl.link_views(); pl.show()
