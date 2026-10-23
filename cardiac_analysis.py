from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree


def interpolate_displacement(source_points, source_displacement, query_points):
  """Estimate displacement at query_points from known source point displacements."""
  rbf = RBFInterpolator(
    source_points,
    source_displacement,
    kernel="linear",
    smoothing=5e-1,
    neighbors=250,
  )
  return rbf(query_points)


def measure_thickness_udf(endocardium, epicardium):
  """Measure endocardium-to-epicardium thickness using unsigned distance field."""
  endocardium = endocardium.copy()
  _ = endocardium.compute_implicit_distance(epicardium, inplace=True)
  dist = endocardium['implicit_distance']
  return np.abs(dist)

from scipy.spatial.distance import cdist

def interpolate_displacement_idw(source_points, source_displacement, query_points, p=4):
    dist = cdist(source_points, query_points, metric="euclidean")
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

THICKNESS_THRESHOLD = 0.46 # endocardium points with thickness > than this are discarded (thickness cannot be defined on these points)

#run_dir = Path("~/Downloads/fix_compenetration_all_real/pat000_smoothed_alignTemplate_fix_compenetration_it0").expanduser()
#reg_dir = run_dir / "template_transformed_to_combined_geometry_std"
out_dir = Path("./interpolation_concept_output")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Complete template: all anatomical structures, before patient deformation.
whole_template = pv.read("whole_template.vtk")
septum_tag = whole_template.cell_data["Septum"].astype(int)
ids = np.unique(septum_tag)
ids_endo = ids[ids < 100]
ids_epi = ids[ids > 100] - 100
assert np.array_equal(ids_endo, ids_epi)
whole_template.cell_data["GroupIds"] = np.where(septum_tag > 100, septum_tag - 101, septum_tag - 1)

# 2. Bring the complete template to the same frame used by the registration (std + umemaya)
std_tr = np.loadtxt("std_tr.txt")
#umeyama_tr = np.loadtxt(run_dir / "umemaya_tr.txt")
umeyama_tr = np.loadtxt("./umemaya_tr.txt")
whole_template.points = whole_template.points * (1.0 / 21.0)
whole_template = whole_template.transform(std_tr, inplace = False)
whole_template = whole_template.transform(umeyama_tr, inplace = False)

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
n_control = min(1000, template_it0.n_points)
control_ids = rng.choice(template_it0.n_points, size=n_control, replace=False)
control_points = template_it0.points[control_ids]
control_displacement = known_displacement[control_ids]


# optional: measure initial whole template thickness
# (from each point of the endocardium measure the distance wrt epicardium)
template_it0_epicardium = whole_template.extract_cells(whole_template.cell_data["Septum"] > 100).extract_surface()
thickness = measure_thickness_udf(template_it0, template_it0_epicardium)
nonmeasured_point_ids = np.nonzero(thickness > THICKNESS_THRESHOLD)[0]
thickness[nonmeasured_point_ids] = 0.0
template_it0['thickness'] = thickness

#pl = pv.Plotter(window_size=(800, 800))
#pl.add_mesh(whole_template, color="#BDBDBD", opacity=0.35)
#pl.add_mesh(template_it0, scalars="thickness", opacity=0.9)
#pl.show()
#template_it0_epicardium.save('template_it0_epicardium.vtk')
#template_it0.save('template_it0_with_thickness.vtk')
#exit(0)


# 7. Interpolate / extrapolate the displacement to the complete anatomical mesh.
whole_displacement = interpolate_displacement(control_points, control_displacement, whole_template.points)
whole_deformed = whole_template.copy()
whole_deformed.points = whole_template.points + whole_displacement
whole_deformed["interpolated_displacement"] = whole_displacement
whole_deformed["interpolated_displacement_mag"] = np.linalg.norm(whole_displacement, axis=1)


# IDW interpolation
whole_displacement_idw = interpolate_displacement_idw(control_points, control_displacement, whole_template.points, p=4)
whole_deformed_idw = whole_template.copy()
whole_deformed_idw.points = whole_template.points + whole_displacement_idw

# thickness measurement for IDW
template_deformed_epicardium_idw = whole_deformed_idw.extract_cells(whole_deformed_idw.cell_data["Septum"] > 100).extract_surface()
thickness_deformed_idw = measure_thickness_udf(template_deformed, template_deformed_epicardium_idw)
thickness_deformed_idw[nonmeasured_point_ids] = 0.0
thickness_abs_error_idw = np.abs(thickness_deformed_idw - thickness)

# knn interpolation
whole_displacement_knn = interpolate_displacement_idw_knn(control_points, control_displacement, whole_template.points, p=4, k=10)
whole_deformed_knn = whole_template.copy()
whole_deformed_knn.points = whole_template.points + whole_displacement_knn

# thickness measurement for knn
template_deformed_epicardium_knn = whole_deformed_knn.extract_cells(whole_deformed_knn.cell_data["Septum"] > 100).extract_surface()
thickness_deformed_knn = measure_thickness_udf(template_deformed, template_deformed_epicardium_knn)
thickness_deformed_knn[nonmeasured_point_ids] = 0.0
thickness_abs_error_knn = np.abs(thickness_deformed_knn - thickness)


# same thickness measurement after the extrapolated deformation
template_deformed_epicardium = whole_deformed.extract_cells(whole_deformed.cell_data["Septum"] > 100).extract_surface()
thickness_deformed = measure_thickness_udf(template_deformed, template_deformed_epicardium)
thickness_deformed[nonmeasured_point_ids] = 0.0
template_deformed['thickness'] = thickness_deformed
thickness_abs_error = np.abs(thickness_deformed - thickness)
template_deformed['thickness_abs_error'] = thickness_abs_error

pdf_y_knn, _ = np.histogram(thickness_abs_error_knn/thickness.std()*100, bins=60, density=False)
pdf_y_knn = pdf_y_knn / pdf_y_knn.sum() * 100
pdf_y, pdf_edges = np.histogram(thickness_abs_error/thickness.std()*100, bins=60, density=False)
pdf_x = 0.5 * (pdf_edges[:-1] + pdf_edges[1:])
pdf_y = pdf_y / pdf_y.sum() * 100

fig, axs = plt.subplots(1, 3, figsize=(11, 4))

axs[0].plot(thickness_deformed_knn, label="IDW KNN p=4 k=10", linewidth=1.0, color="#54A24B")
axs[0].plot(thickness_deformed, label="RBF", linewidth=1.0, color="#4C78A8")
axs[0].plot(thickness_deformed_idw, label="IDW p=4", linewidth=1.0, color="#E45756")
axs[0].plot(thickness, label="initial", linewidth=1.0, color="black")
axs[0].set_xlabel("endocardium point id")
axs[0].set_ylabel("thickness")
axs[0].set_title("Point-wise thickness")
axs[0].legend()

axs[1].plot(thickness_abs_error_knn, label="IDW KNN error", linewidth=1.0, color="#54A24B")
axs[1].plot(thickness_abs_error, label="RBF error", linewidth=1.0, color="#4C78A8")
axs[1].plot(thickness_abs_error_idw, label="IDW p=4 error", linewidth=1.0, color="#E45756")
axs[1].set_xlabel("endocardium point id")
axs[1].set_ylabel("thickness error")
axs[1].set_title("Point-wise thickness error")
axs[1].legend()

axs[2].plot(pdf_x, pdf_y_knn, color="#54A24B", linewidth=2.0, label="IDW KNN p=4 k=10")
axs[2].fill_between(pdf_x, pdf_y_knn, color="#54A24B", alpha=0.25)
axs[2].plot(pdf_x, pdf_y, color="#4C78A8", linewidth=2.0, label="RBF")
axs[2].fill_between(pdf_x, pdf_y, color="#4C78A8", alpha=0.25)
axs[2].set_xlabel("abs(deformed thickness - initial thickness)/std(initial thickness) %")
axs[2].set_ylabel("Histogram [occ. frequency %]")
axs[2].set_title("Absolute error histogram (Frequency)")

pdf_y_idw, _ = np.histogram(thickness_abs_error_idw/thickness.std()*100, bins=60, density=False)
pdf_y_idw = pdf_y_idw / pdf_y_idw.sum() * 100
axs[2].plot(pdf_x, pdf_y_idw, color="#E45756", linewidth=2.0, label="IDW p=4")
axs[2].fill_between(pdf_x, pdf_y_idw, color="#E45756", alpha=0.25)
axs[2].legend()

fig.tight_layout()
fig.savefig(out_dir / "05_thickness_comparison.png", dpi=200)
plt.show()

# mesh quality comparison
template_epicardium = whole_template.extract_cells(whole_template.cell_data["Septum"] > 100).extract_surface().triangulate()
sq_template = template_epicardium.compute_cell_sizes().cell_data['Area']


rbf_epicardium = whole_deformed.extract_cells(whole_deformed.cell_data["Septum"] > 100).extract_surface().triangulate()
sq_rbf = rbf_epicardium.compute_cell_sizes().cell_data['Area']

idw_epicardium = whole_deformed_idw.extract_cells(whole_deformed_idw.cell_data["Septum"] > 100).extract_surface().triangulate()
sq_idw = idw_epicardium.compute_cell_sizes().cell_data['Area']

knn_epicardium = whole_deformed_knn.extract_cells(whole_deformed_knn.cell_data["Septum"] > 100).extract_surface().triangulate()
sq_knn = knn_epicardium.compute_cell_sizes().cell_data['Area']

def mesh_quality_summary(label, mesh, threshold_angle=5.0):
    ma = mesh.cell_quality(quality_measure='min_angle').cell_data['min_angle']
    n_bad = (ma < threshold_angle).sum()
    print(f"\n[{label}]")
    print(f"  Min angle : min={ma.min():.2f}°  mean={ma.mean():.2f}°  "
          f"cells < {threshold_angle}°: {n_bad} ({100*n_bad/len(ma):.1f}%)")
    return ma

meshes = [
    ("template", template_epicardium),
    ("RBF",      rbf_epicardium),
    ("IDW p=4",  idw_epicardium),
    ("IDW KNN",  knn_epicardium),
]

colors = {"template": "black", "RBF": "#4C78A8", "IDW p=4": "#E45756", "IDW KNN": "#54A24B"}

fig, ax = plt.subplots(figsize=(8, 4))
THRESHOLD = 5.0

for label, mesh in meshes:
    ma = mesh_quality_summary(label, mesh, threshold_angle=THRESHOLD)
    counts, edges = np.histogram(ma, bins=60, range=(0, 61))
    x = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(x, counts / counts.sum() * 100, label=label, color=colors[label], linewidth=2)

ax.axvline(THRESHOLD, color="gray", linestyle="--", linewidth=1.5, label=f"threshold ({THRESHOLD}°)")
ax.set_xlabel("Min angle (degrees)")
ax.set_ylabel("Frequency %")
ax.set_title("Epicardium mesh quality — Min angle")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "06_mesh_quality_comparison.png", dpi=200)
plt.show()

# 8. Save everything important for inspection in ParaView/PyVista.
whole_template.save(out_dir / "01_complete_template_patient_frame.vtk")
template_it0.save(out_dir / "02_template_it0_subset.vtp")
template_deformed.save(out_dir / "03_template_deformed_to_patient.vtp")
whole_deformed.save(out_dir / "04_complete_template_deformed_by_interpolation.vtk")
template_it0.save(out_dir / "05_initial_endocardium_with_thickness.vtp")
template_deformed.save(out_dir / "06_deformed_endocardium_with_thickness.vtp")
print(f"saved results in: {out_dir}")

# 9. Simple visualization: original full template, known subset deformation, result.
pl = pv.Plotter(shape=(1, 3), window_size=(1500, 500))
pl.subplot(0, 0); pl.add_text("full template + subset", font_size=10)
pl.add_mesh(whole_template, color="#BDBDBD", opacity=0.35)
pl.add_mesh(template_it0, color="black", style="wireframe", opacity=0.9)
pl.add_mesh(pv.PolyData(control_points), color="green", style='points_gaussian', render_points_as_spheres=True)
pl.subplot(0, 1); pl.add_text("known subset deformation", font_size=10)
pl.add_mesh(template_it0, color="#1E90FF", style="wireframe", opacity=0.8)
pl.add_mesh(template_deformed, color="#DC143C", style="wireframe", opacity=0.8)
pl.subplot(0, 2); pl.add_text("interpolated full template", font_size=10)
pl.add_mesh(whole_deformed, color="#BDBDBD", opacity=0.35)

pl.link_views(); pl.show()