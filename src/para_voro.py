import os
import warnings
import numba
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError
from scipy.optimize import linprog
from tornado.gen import Return
from src.para_voro_util import *
from src.mp_util import *
from src.general_util import *
from itertools import repeat

from src.clustering.kmeans_plus_plus import KMeansClusteringAlgorithm
from pyclustering.utils.metric import distance_metric, type_metric

import multiprocessing as mp
from multiprocessing import shared_memory


@numba.njit()
def _identify_infinity_regions_and_neighbors_nb(vor_point_region: np.ndarray, vor_regions: list[np.ndarray], vor_ridge_points: np.ndarray) -> tuple:
    """
    Finds the regions with infinity edges and returns it as a list of region indices.
    Also returns mappings between point index and region index as well as a mapping from region index to point index.

    Furthermore, returns a mapping from region index to neighboring region indices.
    Also returns a mapping from region index to all ridge indices that belong to this region. 
    """
    # Identify regions that have infinity edges
    point_idx_region_mapping = dict()
    region_idx_point_idx_mapping = dict()

    regions_with_infinity_edges = []
    for point_idx, region_idx in enumerate(vor_point_region):
        if region_idx == -1:
            continue

        point_idx_region_mapping[point_idx] = region_idx
        region_idx_point_idx_mapping[region_idx] = point_idx

        vertices = vor_regions[region_idx]
        if -1 in vertices:
            regions_with_infinity_edges.append(region_idx)


    # Get a list of all neighboring regions for each region index
    region_idx_neighboring_regions_indices_mapping = dict()  # mapping from region idx to list of adjacent region indices
    region_idx_ridge_indices_mapping = dict()  # mapping from region index to list of ridge indices that make up that region
    for region_idx in range(len(vor_regions)):
        if region_idx not in region_idx_point_idx_mapping:
            continue
 
        cur_point_idx = region_idx_point_idx_mapping[region_idx]

        ridges = set()
        neighbors = set()

        # get all ridge indices that belong to the current region
        for ridge_idx in range(len(vor_ridge_points)):
            # get points of ridge
            point_idx_1, point_idx_2 = vor_ridge_points[ridge_idx]
            if point_idx_1 == cur_point_idx:
                # point_idx_2 is the neighboring point
                neighbors.add(point_idx_region_mapping[point_idx_2])
                ridges.add(ridge_idx)
            if point_idx_2 == cur_point_idx:
                # point_idx_1 is the neighboring point
                neighbors.add(point_idx_region_mapping[point_idx_1])
                ridges.add(ridge_idx)

        region_idx_ridge_indices_mapping[region_idx] = np.asarray(np.sort(list(ridges)))
        region_idx_neighboring_regions_indices_mapping[region_idx] = np.asarray(np.sort(list(neighbors)))

    return point_idx_region_mapping, region_idx_point_idx_mapping, regions_with_infinity_edges, region_idx_neighboring_regions_indices_mapping, region_idx_ridge_indices_mapping


def identify_infinity_regions_and_neighbors(vor: Voronoi) -> tuple:
    """
    Finds the regions with infinity edges and returns it as a list of region indices.
    Also returns mappings between point index and region index as well as a mapping from region index to point index.

    Furthermore, returns a mapping from region index to neighboring region indices.
    Also returns a mapping from region index to all ridge indices that belong to this region. 
    """
    return _identify_infinity_regions_and_neighbors_nb(vor_point_region=vor.point_region, 
                                                       vor_regions=[np.array(v, dtype=int) for v in vor.regions],
                                                       vor_ridge_points=vor.ridge_points)


def clip_voronoi_ridge_vertices(vor: Voronoi, clip_equations: np.ndarray, indices_of_ridges_to_be_clipped: list = None,
                                         *, identify_infinity_regions_and_neighbors_results: tuple = None, verbose: bool = False) -> tuple:
    """
    Clip the infinity ridges of an n-dimensional voronoi diagram by an n-dimensional bounding box.
    Returns a tuple with all clipped infinity ridge vertices that are inside the aabbox and corresponding original ridge indices.
    """
    assert vor.ndim == clip_equations.shape[-1] - 1, f"dimension mismatch with clip equations: {vor.ndim} != {clip_equations.shape[-1] - 1}"

    def pf(*args):
        if verbose:
            print(*args)
        else:
            pass

    indices_of_ridges_to_be_clipped = np.asarray(indices_of_ridges_to_be_clipped, dtype=int).flatten()

    pf(f"got {len(indices_of_ridges_to_be_clipped)} ridges to clip")

    if identify_infinity_regions_and_neighbors_results is None:
        identify_infinity_regions_and_neighbors_results = identify_infinity_regions_and_neighbors(vor)

    [point_idx_region_mapping, region_idx_point_idx_mapping, _, _, region_idx_ridge_indices_mapping] = identify_infinity_regions_and_neighbors_results

    ridge_indices_set = set()
    region_indices_set = set()

    indices_of_ridges_to_be_clipped_set = set(indices_of_ridges_to_be_clipped)

    ridge_indices = []
    new_vertices = []
    
    for i, ridge_index_to_be_clipped in enumerate(indices_of_ridges_to_be_clipped):
        pf(f"clip ridge number {i} with idx {ridge_index_to_be_clipped}")
        if ridge_index_to_be_clipped in ridge_indices_set:
            pf(f"skipped ridge number {i} with idx {ridge_index_to_be_clipped} as already clipped as part of other region")
            continue

        # determine region index of ridge 
        # (there are always two regions for a ridge so we can choose either, 
        #   though, it is probably more efficient to choose the region that contains more infinity ridges as all of them are being clipped at once)
        point_index_1, point_index_2 = vor.ridge_points[ridge_index_to_be_clipped]

        region_index_1, region_index_2 = point_idx_region_mapping[point_index_1], point_idx_region_mapping[point_index_2]

        # check which region has more infinity ridges
        n_infinity_ridges_1 = sum([1 if r_idx in indices_of_ridges_to_be_clipped_set else 0 for r_idx in region_idx_ridge_indices_mapping[region_index_1]])
        n_infinity_ridges_2 = sum([1 if r_idx in indices_of_ridges_to_be_clipped_set else 0 for r_idx in region_idx_ridge_indices_mapping[region_index_2]])

        n_ridges_to_be_clipped = n_infinity_ridges_1

        if n_infinity_ridges_1 > n_infinity_ridges_2:
            region_index = region_index_1
        else:
            region_index = region_index_2
            n_ridges_to_be_clipped = n_infinity_ridges_2

        if region_index in region_indices_set:
            pf(f"skipping ridge number {i} as the corresponding region with idx {region_index} was already tried to be clipped")
            continue
        region_indices_set.add(region_index)

        pf(f"clip region idx {region_index} of ridge idx {ridge_index_to_be_clipped}, with {n_ridges_to_be_clipped} expected clipped ridges")

        # perform clipping
        (finite_ridge_vertices, ridge_indices_with_finite_vertices), (infinite_ridge_vertices, infinite_ridge_indices_with_finite_vertices) = clip_ridges_of_voronoi_region(
            region_idx=region_index, vor=vor, clip_equations=clip_equations, 
            region_idx_point_idx_mapping=region_idx_point_idx_mapping, 
            region_idx_ridge_indices_mapping=region_idx_ridge_indices_mapping,
            include_finite_ridge=True,
            include_infinity_ridges=True,
        )

        # add all found clipped finite ridges
        for clipped_vertices, clipped_ridge_index in zip(finite_ridge_vertices, ridge_indices_with_finite_vertices):
            if clipped_ridge_index in ridge_indices_set or clipped_ridge_index not in indices_of_ridges_to_be_clipped_set:
                continue
            ridge_indices.append(clipped_ridge_index)
            new_vertices.append(clipped_vertices)

            ridge_indices_set.add(clipped_ridge_index)

        # add all found clipped infinite ridges
        for clipped_vertices, clipped_ridge_index in zip(infinite_ridge_vertices, infinite_ridge_indices_with_finite_vertices):
            if clipped_ridge_index in ridge_indices_set or clipped_ridge_index not in indices_of_ridges_to_be_clipped_set:
                continue
            ridge_indices.append(clipped_ridge_index)
            new_vertices.append(clipped_vertices)

            ridge_indices_set.add(clipped_ridge_index)
        
    return new_vertices, ridge_indices


def vertices_fully_inside_aabbox(vertices: np.ndarray, aabbox: np.ndarray) -> bool:
    """
    Checks if all vertices are fully inside the aabbox.
    """
    N = len(aabbox)

    for dim in range(N):
        if not (np.all(vertices[:, dim] >= aabbox[dim][0]) and np.all(vertices[:, dim] <= aabbox[dim][1])):
            return False

    return True


@numba.njit()
def vertices_fully_inside_convex_hull(vertices: np.ndarray, equations: np.ndarray) -> bool:
    """
    Checks if all of the provided vertices lie in the convex hull.
    """
    d = (vertices @ equations[:, :-1].T) + equations[:, -1]
    return np.all(d <= 1e-12)


@numba.njit()
def vertices_fully_outside_convex_hull(vertices: np.ndarray, equations: np.ndarray) -> bool:
    """
    Checks if all of the provided vertices lie outside the convex hull.
    """
    d = (vertices @ equations[:, :-1].T) + equations[:, -1]

    return np.all(d > 1e-12)


@numba.njit()
def vertices_fully_inside_hypersphere(vertices: np.ndarray, circle_center: np.ndarray, radius: float) -> bool:
    """
    Returns true if all vertices are inside the given sphere (according to euclidean distance), else false.
    """
    for i in range(len(vertices)):
        if np.linalg.norm(vertices[i] - circle_center) > radius:
            return False

    return True


@numba.njit()
def vertices_fully_outside_hypersphere(vertices: np.ndarray, circle_center: np.ndarray, radius: float) -> bool:
    """
    Returns true if all vertices are outside the given sphere (according to euclidean distance), else false.
    """
    for i in range(len(vertices)):
        if np.linalg.norm(vertices[i] - circle_center) <= radius:
            return True

    return False


@numba.njit()
def vertices_fully_inside_outside_convex_hull(vertices: np.ndarray, equations: np.ndarray) -> tuple:
    """
    Checks if all of the provided vertices lie inside/outside the convex hull.
    Returns a bool for each. 
    @return (bool, bool), (all_inside, all_outside)
    """
    # d = (vertices @ equations[:, :-1].T) + equations[:, -1]
    # eps = 1e-12
    # return np.all(d <= eps), np.all(d > eps)

    all_inside = True
    all_outside = True

    b = False

    eps = 1e-12

    for point in vertices:
        if b:
            break
        for eq in equations:
            d_ = point @ eq[:-1] + eq[-1]

            all_inside = all_inside and d_ <= eps
            all_outside = all_outside and d_ > eps

            if not (all_inside or all_outside):
                b = True
                break
    
    return all_inside, all_outside


@numba.njit()
def vertices_fully_inside_convex_hull2(vertices: np.ndarray, equations: np.ndarray) -> tuple:
    """
    Checks if all of the provided vertices lie inside the convex hull.
    Returns a bool for each. 
    @return (bool, bool), (all_inside, all_outside)
    """
    # d = (vertices @ equations[:, :-1].T) + equations[:, -1]
    # eps = 1e-12
    # return np.all(d <= eps), np.all(d > eps)

    all_inside = True

    b = False

    eps = 1e-12

    for point in vertices:
        if b:
            break
        for eq in equations:
            d_ = point @ eq[:-1] + eq[-1]

            all_inside = all_inside and d_ <= eps

            if not all_inside:
                b = True
                break
    
    return all_inside
    

def vertices_fully_on_one_outer_side_of_aabbox(vertices: np.ndarray, aabbox: np.ndarray) -> bool:
    """
    Check if all vertices are on one outer side of the aabbox.
    In that case the simplex spanned by these vertices cannot cross the aabbox.
    """
    N = len(aabbox)

    for dim in range(N):
        if np.all(vertices[:, dim] > aabbox[dim][1]) or np.all(vertices[:, dim] < aabbox[dim][0]):
            return True

    return False


@numba.njit()
def map_points_to_hyperplanes(points: np.ndarray, hyperplane_equations: np.ndarray) -> tuple:
    # print("points:\n", points)
    # print("hyperplane_equations:\n", hyperplane_equations)
    
    valid_hyperplanes = []
    points_on_hyperplanes = []

    for j, eq in enumerate(hyperplane_equations):
        ridge_halfspace_definition = eq

        simplex_points = []

        for i in range(len(points)):
            d = np.dot(points[i], ridge_halfspace_definition[:-1]) + ridge_halfspace_definition[-1]

            if abs(d) < 1e-12:
                # point is on plane
                simplex_points.append(points[i])

        if len(simplex_points) > 0:
            valid_hyperplanes.append(j)
            points_on_hyperplanes.append(nb_stack(simplex_points))

    return points_on_hyperplanes, valid_hyperplanes


def clip_voronoi_region(region_idx: int, vor: Voronoi, clip_equations: np.ndarray,
                        region_idx_point_idx_mapping: dict,
                        region_idx_ridge_indices_mapping: dict) -> tuple[np.ndarray, tuple]:
    """
    Clip a voronoi region.
    Returns the new region indices after clipping (first return value), plus additional values in second return value.
    """
    # 1. Construct all halfspace definitions for the voronoi-cell.
    # region_vertices_indices = np.asarray(vor.regions[region_idx])
    region_point_idx = region_idx_point_idx_mapping[region_idx]
    cur_point = vor.points[region_point_idx]

    # find neighboring points with which we share an infinity edge
    ridges = region_idx_ridge_indices_mapping[region_idx]

    if len(ridges) == 0:
        return None, ([], [], [], [])  # something went wrong in the voronoi. we should have neighboring ridges

    # find ridges with infinity edges
    ridge_indices_of_region_with_infinity = []
    ridge_indices_of_region_with_finite = []

    for i in range(len(ridges)):
        if -1 in vor.ridge_vertices[ridges[i]]:
            ridge_indices_of_region_with_infinity.append(ridges[i])
        else:
            ridge_indices_of_region_with_finite.append(ridges[i])

    # map ridges of region to points
    infinity_ridges_point_indices = vor.ridge_points[ridge_indices_of_region_with_infinity]
    finite_ridges_point_indices = vor.ridge_points[ridge_indices_of_region_with_finite]

    # get other points
    other_point_indices_of_infinite = infinity_ridges_point_indices.flatten()[infinity_ridges_point_indices.flatten() != region_point_idx]
    other_point_indices_of_finite = finite_ridges_point_indices.flatten()[finite_ridges_point_indices.flatten() != region_point_idx]

    # compute normal per ridge
    other_points_infinite = vor.points[other_point_indices_of_infinite]
    other_points_finite = vor.points[other_point_indices_of_finite]

    normals_not_normed_infinite = other_points_infinite - cur_point
    mid_points_infinite = cur_point + (normals_not_normed_infinite / 2.0)  # points on halfspace
    normals_not_normed_lengths_infinite = np.linalg.norm(normals_not_normed_infinite, axis=-1)
    normals_infinite = (normals_not_normed_infinite.T / normals_not_normed_lengths_infinite).T
    infinity_ridge_halfspaces = np.array([compute_halfspace_equation_from_normal_and_point(normal=n, point=p) 
                                          for n, p in zip(normals_infinite, mid_points_infinite)])

    normals_not_normed_finite = other_points_finite - cur_point
    mid_points_finite = cur_point + (normals_not_normed_finite / 2.0)  # points on halfspace
    normals_not_normed_lengths_finite = np.linalg.norm(normals_not_normed_finite, axis=-1)
    normals_finite = (normals_not_normed_finite.T / normals_not_normed_lengths_finite).T
    finite_ridge_halfspaces = np.array([compute_halfspace_equation_from_normal_and_point(normal=n, point=p) 
                                          for n, p in zip(normals_finite, mid_points_finite)])

    if len(finite_ridge_halfspaces) == 0:
        combined_halfspaces = infinity_ridge_halfspaces
    elif len(infinity_ridge_halfspaces) == 0:
        combined_halfspaces = finite_ridge_halfspaces
    else:
        combined_halfspaces = np.vstack([finite_ridge_halfspaces, infinity_ridge_halfspaces])

    # 2. Add halfspace definitions for aabbox.
    final_halfspaces = np.vstack([combined_halfspaces, clip_equations])

    # 3. Compute intersection points.
    ip = get_interior_point_from_halfspaces(final_halfspaces)[0]

    try:
        hi = HalfspaceIntersection(halfspaces=final_halfspaces, interior_point=ip)
    except QhullError as e:
        if (str(e).startswith("QH6023") or str(e).startswith("QH6347") or 
            str(e).startswith("QH7088") or str(e).startswith("QH6271") or
            str(e).startswith("QH7086") or str(e).startswith("QH7088") or
            str(e).startswith("QH6297")):
            # apparently this region is outside of the aabbox and we cannot clip it at all
            # QH6023 qhull input error: feasible point is not clearly inside halfspace
            # QH6347: qhull precision error (qh_mergefacet): wide merge for facet f25 into f97 for mergetype 3 (concave).
            # QhullError: QH7088 Qhull precision warning: in post-processing (qh_check_maxout)
            # QH6271 qhull topology error (qh_check_dupridge): wide merge (4970471537208.3x wider) due to dupridge
            # QH7086 Qhull precision warning: repartition coplanar point
            # QH7088 Qhull precision warning: in post-processing (qh_check_maxout)
            # QH6297 Qhull precision error (qh_check_maxout): large increase in qh.max_outside during post-processing
            return None, ([], [], [], [])
        else:
            raise e
        
    intersections = hi.intersections

    return intersections, (ridge_indices_of_region_with_finite, ridge_indices_of_region_with_infinity, finite_ridge_halfspaces, infinity_ridge_halfspaces)


def clip_ridges_of_voronoi_region(region_idx: int, vor: Voronoi, clip_equations: np.ndarray,
                        region_idx_point_idx_mapping: dict,
                        region_idx_ridge_indices_mapping: dict, 
                        include_infinity_ridges: bool = True, 
                        include_finite_ridge: bool = False) -> tuple:
    """
    Clip ridges of a voronoi region with a bounding box.
    
    We do this by constructing the clipped Voronoi-Cell and extract the simplex of each ridge (which now should be always finite). 
    1. Construct all halfspace definitions for the voronoi-cell.
    2. Add halfspace definitions for bbox.
    3. Compute intersection points.
    4. Construct convex-hull of intersection points.
    5. Find simplex that belongs to each original ridge.
    6. Replace original ridge vertices with the vertices of the simplex.

    @return: Returns a list with new ridge vertices and a list with ridge indices that they belong to.
    Returns a tuple for originally finite and originally infinite ridges ((finite_vertices, finite_ridge_indices), (infinite_vertices, infinite_ridge_indices))
    For the ones it is False returns None for the tuple entries.

    """
    assert include_finite_ridge or include_infinity_ridges, "include either or both"

    intersections, (ridge_indices_of_region_with_finite, ridge_indices_of_region_with_infinity, finite_ridge_halfspaces, infinity_ridge_halfspaces) = clip_voronoi_region(region_idx, vor, clip_equations, region_idx_point_idx_mapping, region_idx_ridge_indices_mapping)

    # 4. Find simplex that belongs to infinity ridge.
    # They should have the same equation!
    # make sure that at most one equation is equal and matches!
    # The algorithm only works if the aabbox is large enough such that all infinity ridges are inside of it, otherwise the equation cannot be found in the next check as it has been cut off due to the aabbox!
    finite_ridge_vertices = None
    ridge_indices_with_finite_vertices = None
    if include_infinity_ridges:
        if len(infinity_ridge_halfspaces) == 0:
            finite_ridge_vertices = []
            ridge_indices_with_finite_vertices = []
        else:
            finite_ridge_vertices, valid_halfspaces = map_points_to_hyperplanes(intersections, infinity_ridge_halfspaces)
            ridge_indices_with_finite_vertices = [ridge_indices_of_region_with_infinity[v] for v in valid_halfspaces]

    clipped_finite_ridge_vertices = None
    clipped_ridge_indices_with_finite_vertices = None
    if include_finite_ridge:
        if len(finite_ridge_halfspaces) == 0:
            clipped_finite_ridge_vertices = []
            clipped_ridge_indices_with_finite_vertices = []
        else:
            clipped_finite_ridge_vertices, valid_halfspaces = map_points_to_hyperplanes(intersections, finite_ridge_halfspaces)
            clipped_ridge_indices_with_finite_vertices = [ridge_indices_of_region_with_finite[v] for v in valid_halfspaces]
    
    return (clipped_finite_ridge_vertices, clipped_ridge_indices_with_finite_vertices), (finite_ridge_vertices, ridge_indices_with_finite_vertices)


def get_interior_point_from_halfspaces(halfspaces: np.ndarray, return_full: bool = False) -> np.ndarray:
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = - halfspaces[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    assert res.success
    x = res.x[:-1]
    y = res.x[-1]

    if return_full:
        return res, x, y

    return x, y


def make_halfspaces(one_dim_sample_point_min: float, one_dim_sample_point_max: float, n_dim: int, dim: int) -> np.array:
    # create the half space representations that divides the n_dim space at the sample point for dimension at index dim
    # returns the lower half space and the upper half space, such that the sample point is in the center
    # Returns [A; b] with Ax + b <= 0
    lower_hs = np.zeros((n_dim + 1,))
    upper_hs = np.zeros((n_dim + 1,))

    assert n_dim > 0
    assert dim >= 0 and dim < n_dim, "dim is zero-indexed and must be between 0 and n_dim - 1"

    lower_hs[dim] = -1.0
    upper_hs[dim] = 1.0

    lower_hs[-1] = one_dim_sample_point_min
    upper_hs[-1] = -one_dim_sample_point_max

    return lower_hs, upper_hs


def make_lower_halfspace(one_dim_sample_point_min: float, n_dim: int, dim: int) -> np.array:
    # create the lower half space representation that divides the n_dim space at the sample point for dimension at index dim
    # returns the lower half space, such that the sample point is "to the right" of it
    # Returns [A; b] with Ax + b <= 0
    lower_hs = np.zeros((n_dim + 1,))

    assert n_dim > 0
    assert dim >= 0 and dim < n_dim, "dim is zero-indexed and must be between 0 and n_dim - 1"

    lower_hs[dim] = -1.0

    lower_hs[-1] = one_dim_sample_point_min

    return lower_hs


@numba.njit()
def rotate_to_lower_dimensional_hyperplane_nb(vertices: np.ndarray) -> tuple:
    """
    Compute the SVD of the provided vertices and cuts off one dimension with the lowest eigenvalue.
    Returns rotation matrix to be applied: vertices @ R
    If return_full is True, just returns the full svd result.
    """
    U, S, Vh = np.linalg.svd(vertices - nb_mean0(vertices), True)
    return U, S, Vh


def rotate_to_lower_dimensional_hyperplane(vertices: np.ndarray, return_full: bool = False) -> tuple:
    """
    Compute the SVD of the provided vertices and cuts off one dimension with the lowest eigenvalue.
    Returns rotation matrix to be applied: vertices @ R
    If return_full is True, just returns the full svd result.
    """
    U, S, Vh = np.linalg.svd(vertices - vertices.mean(0), full_matrices=False)
    if return_full:
        return U, S, Vh
    return Vh[:, :-1]  # [np.abs(S) > 1e-08]


@numba.njit()
def find_point_in_intersecting_plane(h: np.ndarray, dim: int, d: float) -> tuple:
    """
    Given a hyperplane h defined by a normal vector n and offset d, combined in h = [n; d],
        as well as a second hyperplane define by normal vector in direction of dimension-axis dim and with offset d.
    Computes a trivial intersecting point and returns it.
    Second return value is always False unless no intersection point is found, then it is True.  (Somehow cannot catch exception with numba)
    """
    # check if no intersection occurs
    n = h[:-1]
    o = h[-1]

    N = n.shape[0]

    aw = np.argwhere(n)

    assert len(aw) > 0

    if aw.shape[0] == 1 and aw.item() == dim:
        # check value
        v = o / n[aw.item()]
        if v == d:
            # parallel planes, generate trivial point
            ret = np.zeros(N)
            ret[dim] = v
            return ret, False
        else:
            return np.zeros(0), True  # RuntimeError("cannot compute intersection of parallel hyperplanes that are not identical")
    else:
        # they have to intersect somewhere
        ret = np.zeros(N)
        non_zero_dim = aw.flatten()[aw.flatten() != dim][0]  # want non-zero dimension that is also not the one that we replace
        coefficient = n[non_zero_dim]
        ret[dim] = d
        ret[non_zero_dim] = (o - n[dim] * d) / coefficient
        return ret, False


@numba.njit()
def get_plane_offset_from_normal_and_point_in_plane(n: np.ndarray, point: np.ndarray) -> float:
    """
    Returns the offset for the plane given it's normal and an arbitrary point on the plane.
    """
    return n @ point


@numba.njit()
def compute_lower_dimensional_halfspace(intersecting_point: np.ndarray, vh: np.ndarray,
                                        rotated_normal_of_hs: np.ndarray) -> np.ndarray:
    """
    Compute the lower-dimensional halfspace given the original halfspace and the rotation matrix.
    """
    # project intersecting point on lower dimensional space
    intersecting_point_in_lower_dim = (intersecting_point @ vh.T)[:-1]
    # compute lower dimensional offset of clipping halfspace
    offset_in_lower_dim = get_plane_offset_from_normal_and_point_in_plane(rotated_normal_of_hs, intersecting_point_in_lower_dim)
    # create new halfspaces in lower dimensions
    halfspace = np.append(rotated_normal_of_hs, -offset_in_lower_dim)
    return halfspace


def compute_halfspace_equation_from_normal_and_point(normal: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Compute the halfspace equation given the normal and a point on the halfspace.
    The normal points away from the resulting halfspace.
    """
    # compute offset of halfspace
    offset = get_plane_offset_from_normal_and_point_in_plane(normal, point)
    # create new halfspaces in lower dimensions
    halfspace = np.append(normal, -offset)
    return halfspace


def construct_halfspace_definition_for_aabbox(aabbox: np.ndarray) -> np.ndarray:
    """
    Given an n-dimensional bounding box, returns the halfspace definitions that represent the space 
        of the given axis-aligned bounding box.
    @param aabbox: np.ndarray of shape (N, 2) where N is the dimensionality of the space.
        aabbox[:, 0] are all the minimum values, and aabbox[:, 1] are all the maximum values.
    @return halfspaces: np.ndarray of shape (2 * N, N + 1). 
        Each row represents one halfspace in the form [a_0, a_1, ..., a_{N-1}, b].
        The halfspaces fullfil the equations Ax + b <= 0 such that the aabbox is the space they border.
    """
    assert aabbox.shape[0] > 0 and aabbox.shape[1] == 2
    N = aabbox.shape[0]
    halfspaces = np.concatenate([make_halfspaces(aabbox[i, 0], aabbox[i, 1], N, i) for i in range(N)])
    return halfspaces


@numba.njit()
def _compute_lower_dim_halfspaces(lower_hs: np.ndarray, vh: np.ndarray, n: np.ndarray, o: float, dim: int, sample_points_min_value: float, sample_points_max_value: float, ch_equations: np.ndarray) -> tuple:
    # rotate halfspaces, create convex hulls, compute volumes
    # the halfspace definition are still in original dimensionality, we have to rotate that as well
    rotated_normal_lower_hs, l = normalize_vec_full((lower_hs[:-1] @ vh.T)[:-1])
    if l < 1e-12:
        # the normal is basically zero, the clipping should have no effect
        return True, ch_equations
    
    # rotated_normal_upper_hs = np.array([normalize_vec((upper_hs[:-1] @ vh.T)[:-1]) for vh in relevant_vhs])
    rotated_normal_upper_hs = -rotated_normal_lower_hs  # can optimize: no need to consider sample_idx to rotate normal. also no need to rotate both normals as they are just opposite directions
    
    # compute intersecting points on the planes for each normal normals
    # handle cases where there is no intersection point (we can just exclude that (most likely both hyperplanes then))
    lip, exclude_lower = find_point_in_intersecting_plane(np.append(n, o), dim, sample_points_min_value)
    uip, exclude_upper = find_point_in_intersecting_plane(np.append(n, o), dim, sample_points_max_value)
    
    exclude_both = exclude_lower and exclude_upper
    
    if not exclude_lower:
        lower_halfspace = compute_lower_dimensional_halfspace(lip, vh, rotated_normal_lower_hs).reshape(1, -1)
    
    if not exclude_upper:
        upper_halfspace = compute_lower_dimensional_halfspace(uip, vh, rotated_normal_upper_hs).reshape(1, -1)
    
    # add new halfspaces to form lower-dim convex hull
    if exclude_lower and exclude_upper:
        new_lower_dim_halfspaces = ch_equations
    elif exclude_lower and not exclude_upper:
        new_lower_dim_halfspaces = nb_vstack([ch_equations, upper_halfspace])
    elif not exclude_lower and exclude_upper:
        new_lower_dim_halfspaces = nb_vstack([ch_equations, lower_halfspace])
    else:
        new_lower_dim_halfspaces = nb_vstack([ch_equations, lower_halfspace, upper_halfspace])

    return exclude_both, new_lower_dim_halfspaces


@numba.njit()
def _build_directed_ridge_vertices_selector_matrix(unique_labels: np.ndarray, unqie_labels_inverse: np.ndarray, samples: np.ndarray, labels: np.ndarray, ridge_point_indices: np.ndarray) -> np.ndarray:
    N = samples.shape[-1]
    L = unique_labels.shape[0]
    R = ridge_point_indices.shape[0]

    M = np.full((N, L, L, R), False)

    for rv_i in range(R):
        points_indices_rv_i = ridge_point_indices[rv_i]

        # get original points of rv_i
        points_rv_i = samples[points_indices_rv_i]

        # get original labels of rv_i
        from_label_idx_rv_i, to_label_idx_rv_i = unqie_labels_inverse[points_indices_rv_i]

        for dim_i in range(N):
            is_transition = points_rv_i[0, dim_i] < points_rv_i[1, dim_i]
            
            M[dim_i, from_label_idx_rv_i, to_label_idx_rv_i, rv_i] = is_transition
            M[dim_i, to_label_idx_rv_i, from_label_idx_rv_i, rv_i] = not is_transition
    return M


def build_directed_ridge_vertices_selector_matrix(samples: np.ndarray, labels: np.ndarray, ridge_point_indices: np.ndarray) -> np.ndarray:
    """
    Returns list of arrays with shape (n_dim, n_unique_labels, n_unique_labels, n_gen_ridge_vertices) of dtype bool.
    It can be used as accessor to get all ridge vertices with certain attributes, e.g.,:
        M = build_directed_ridge_vertices_selector_matrix(samples, labels, ridge_point_indices)
        ridge_vertices_from_label_idx_0_to_label_idx_1_for_dim_0 = ridge_vertices[M[0, 0, 1]]  # [dim_i, from_label_idx, to_label_idx]
    """
    unique_labels, unqie_labels_inverse = np.unique(labels, return_inverse=True)

    return _build_directed_ridge_vertices_selector_matrix(unique_labels, unqie_labels_inverse, samples, labels, ridge_point_indices)
    


def generate_geometric_segment_boundaries_via_voronoi(samples: np.ndarray, labels: np.ndarray, clip_bbox: np.ndarray = None, clip_convex_hull: bool = False, 
                                                      label_pair_to_consider: tuple = None, vor: Voronoi = None,
                                                      *, return_original_ridge_point_indices: bool = False, verbose: bool = False) -> tuple:
    """
    Does the following:
    1. Generates the voronoi diagram of the input samples.
    2. Extract ridges that are between two samples with different label.
    3. Clip the potentially contained infinity ridges to a given bounding box if it is provided (tbd.), otherwise ignores them.
    4. Returns the ridge vertices of the resulting ridges.

    @param samples: the input samples (N_Samples, N)
    @param labels: the label for each input sample (N, )
    @clip_bbox: the axis-aligned bounding box to clip with (N, 2)
    @clip_convex_hull: bool, whether to also clip with the convex hull of the sample points.
    @label_pair_to_consider: tuple of ints, (label_1: int, label_2: int): only considers ridges between these two labels
    @vor: Voronoi; Precomputed Voronoi Diagram for the given samples. Especially useful if specifying label_pair_to_consider to avoid recomputation of the diagram.
    
    @return_original_ridge_point_indices: bool, whether to return the list of ridge point indices. shape (N_returned_ridges, 2). 
        The point indices correspond to the input samples (or form the provided voronoi diagram, which should be the same)
    @return: returns the ridge vertices of the resulting ridges. Return type is np.ndarray of dtype=object. The elements of this array are of shape [*, N] where N is the 
        dimensionality of the space and * is the number of vertices per ridge. In general, this is * >= N.
    """
    def pf(*args):
        if verbose:
            print(*args)
        else:
            pass

    labels = np.asarray(labels, dtype=int)
    samples = np.asarray(samples)
    
    orig_clip_bbox = None
    if clip_bbox is not None:
        clip_bbox = np.asarray(clip_bbox)
        orig_clip_bbox = clip_bbox.copy()

    assert len(samples) > 0 and len(labels) == len(samples)

    if len(samples.shape) == 1:
        samples = samples.reshape(-1, 1)

    N = samples.shape[-1]

    pf(f"got {len(samples)} labeled samples with {len(np.unique(labels))} different labels and space dimensionality {N}")
    pf(f"Clipping with aabbox is {clip_bbox is not None}")
    pf(f"Clipping with convex hull is {clip_convex_hull}")

    # custom case for 1D
    if N == 1:
        # sort samples and labels
        samples = samples.flatten()
        sort_selector = np.argsort(samples)
        samples = samples[sort_selector]
        labels = labels[sort_selector]

        # segment vertices just lie in the middle of each sample point
        # there are always exactly two infinity ridges: the very "left" and the very "right" one.

        # get sample indices where the "right" sample has a different label
        if label_pair_to_consider is not None:
            label_1, label_2 = label_pair_to_consider
            relevant_samples_indices = np.argwhere(np.logical_or(
                np.logical_and(labels[:-1] == label_1, labels[1:] == label_2), 
                np.logical_and(labels[:-1] == label_2, labels[1:] == label_1)
            ))
        else:
            relevant_samples_indices = np.argwhere(labels[:-1] != labels[1:])

        ridge_vertices = np.empty((len(relevant_samples_indices), ), dtype=object)  # exclude infinity ridges to the "left" and "right"

        if return_original_ridge_point_indices:
            ridge_point_indices = np.empty((len(relevant_ridges_indices), ), dtype=int)

        # compute ridge vertices (they lie in the middle between each relevant sample to it's right neighbor)
        for i in range(len(relevant_samples_indices)):
            sample_idx = relevant_samples_indices[i]
            rv = samples[sample_idx] + ((samples[sample_idx+1] - samples[sample_idx]) / 2.0)

            ridge_vertices[i] = np.array([[rv]])

            if return_original_ridge_point_indices:
                ridge_point_indices[i] = sort_selector[[sample_idx, sample_idx+1]]

        if clip_convex_hull:
            if clip_bbox is None:
                clip_bbox = np.array([[samples[0], samples[-1]]])
            else:
                clip_bbox = clip_bbox.flatten()
                clip_bbox[0] = max(samples[0], clip_bbox[0])
                clip_bbox[1] = min(samples[-1], clip_bbox[1])

        # clip to bbox
        if clip_bbox is not None:
            clip_bbox = clip_bbox.flatten()
            min_v, max_v = clip_bbox[:2]

            ok_selector = np.array([bool(rv[0] >= min_v and rv[0] <= max_v) for rv in ridge_vertices])
            if return_original_ridge_point_indices:
                return ridge_vertices[ok_selector], ridge_point_indices[ok_selector]
            return ridge_vertices[ok_selector]  # no need for further clipping in 1D case

        if return_original_ridge_point_indices:
            return ridge_vertices, ridge_point_indices
        return ridge_vertices

    # compute voronoi
    pf(f"compute voronoi..")
    if vor is None:
        vor = Voronoi(samples)  # qhull_options="Tv"  (Use option 'Tv' to verify the output from Qhull. It verifies that adjacent facets are clearly convex. It verifies that all points are on or below all facets. )
    pf(f"extract ridges")

    # Extract ridges (faces) that correspond to points with different labels
    face_points = vor.ridge_points

    # map face_points to labels
    face_labels = labels[face_points]

    if label_pair_to_consider is not None:
        label_1, label_2 = label_pair_to_consider
        relevant_ridges_selector = np.logical_or(
                np.logical_and(face_labels[:, 0] == label_1, face_labels[:, 1] == label_2), 
                np.logical_and(face_labels[:, 0] == label_2, face_labels[:, 1] == label_1)
            )
    else:
        relevant_ridges_selector = face_labels[:, 0] != face_labels[:, 1]

    relevant_ridges_indices = np.argwhere(relevant_ridges_selector)

    # TODO: don't need to apply selector afterwards, can directly select ridge vertices by index from vor.*
    dividing_ridges = make_object_array([np.asarray(rv) for rv in vor.ridge_vertices], dtype=object)[relevant_ridges_selector]

    # dividing_points = face_points[relevant_ridges_selector]

    pf(f"got a total of {len(dividing_ridges)} dividing ridges")

    if len(dividing_ridges) == 0:
        pf(f"early return due to zero dividing ridges")
        if return_original_ridge_point_indices:
            return np.empty((0, ), dtype=object), np.empty((0, 2), dtype=object)
        return np.empty((0, ), dtype=object)

    # filter infinity ridges
    selector = np.array(([np.all(dv >= 0) for dv in dividing_ridges]))  # filter infinity ridges

    pf(f"got {selector.sum()} many finite ridges and {len(dividing_ridges) - selector.sum()} infinity ridges")

    if clip_convex_hull:
        pf(f"compute convex hull for clipping...")
        clip_hull = ConvexHull(samples)

        verts = clip_hull.points[clip_hull.vertices]
        clip_hull_bbox = np.vstack([np.min(verts, axis=0), np.max(verts, axis=0)]).T

        if clip_bbox is None:
            clip_bbox = clip_hull_bbox
        else:
            clip_bbox = clip_bbox.copy()
            # increase the range of the clip bbox (for trivial rejects, we won't be able to use it for trivial accepts)
            clip_bbox[:, 0] = np.min(np.vstack([clip_bbox[:, 0], clip_hull_bbox[:, 0]]), axis=0)
            clip_bbox[:, 1] = np.max(np.vstack([clip_bbox[:, 1], clip_hull_bbox[:, 1]]), axis=0)
    else:
        clip_hull = None

    if clip_bbox is not None:
        pf(f"identify infinity regions and adjacencies of cells")
        identify_infinity_regions_and_neighbors_results = identify_infinity_regions_and_neighbors(vor)

        # # clip infinite ridges with provided clip_bbox and add to list of finite ridges (after mapping to vertices). 
        # # -> can be done by using the halfspace intersection of the clip bbox together with the halfspace intersections of the (open) voronoi cell.
        # # dividing_infinity_ridges = dividing_ridges[np.invert(selector)]
        # clipped_infinity_ridge_vertices, _ = clip_voronoi_ridge_vertices(
        #     vor=vor, aabbox=clip_bbox, indices_of_ridges_to_be_clipped=relevant_ridges_indices[np.invert(selector)],
        #     identify_infinity_regions_and_neighbors_results=identify_infinity_regions_and_neighbors_results)

        # also clip finite ridges
        # determine which ridges have to be clipped
        # trivial non-clipping if all points are in the clip_bbox or if all points are on the SAME side OUTSIDE of the bbox
        indices_of_infinite_ridge_to_be_clipped = relevant_ridges_indices[np.invert(selector)].flatten()

        # trivially reject ridges that are fully inside or fully outside (on one side) of the aabbox
        indices_of_finite_ridges_to_be_clipped = relevant_ridges_indices[selector].flatten()
        
        trivial_accepts = []
        trivial_rejects = []
        to_be_clipped = []

        pf(f"compute trivial rejects or accepts for clipping")

        # combine the halfspaces if given to accelerate trivial accept
        if clip_hull is None:
            combined_halfspaces_clip_hull_aabbox = construct_halfspace_definition_for_aabbox(orig_clip_bbox)
        else:
            combined_halfspaces_clip_hull_aabbox = np.vstack([
                construct_halfspace_definition_for_aabbox(orig_clip_bbox),
                clip_hull.equations
            ])
        
        # compute interior point and radius of sphere trivially inside the convex hull and bbox
        interior_point, radius = get_interior_point_from_halfspaces(combined_halfspaces_clip_hull_aabbox)
        pf(f"inside interior_point: {interior_point}, radius: {radius}")

        # out_check_ok = False
        # try:
        #     out_check_intersection_points = HalfspaceIntersection(combined_halfspaces_clip_hull_aabbox, interior_point=interior_point).intersections
        #     
        #     out_radius = np.linalg.norm(out_check_intersection_points - interior_point, axis=1).max()
        #     pf(f"interior_point: {interior_point}, outside radius: {out_radius}")
        #     
        #     out_check_ok = True
        # except Exception as e:
        #     pf(f"not doing outside check due to exception: {str(e)}")
        
        for r_idx in indices_of_finite_ridges_to_be_clipped:
            rv = vor.vertices[vor.ridge_vertices[r_idx]]

            # we can always use the clip bbox to compute trivial rejects (as we extended it to the convex hull if it exists)
            # test for trivial rejects
            if vertices_fully_on_one_outer_side_of_aabbox(rv, clip_bbox):
                trivial_rejects.append(r_idx)
                continue

            if clip_hull is None:
                # in this case, we can also test for trivial accept as there is no given convex hull
                if vertices_fully_inside_aabbox(rv, clip_bbox):  # The convex hull clip box cannot be used as trivial accept for the ridges!
                    trivial_accepts.append(r_idx)
                    continue
            else:
                # test for early trivial accept
                if vertices_fully_inside_hypersphere(rv, interior_point, radius):
                    trivial_accepts.append(r_idx)
                    continue

                # # test for early trivial reject  # cannot trivially reject (going over the edge)
                # if out_check_ok and vertices_fully_outside_hypersphere(rv, interior_point, out_radius):
                #     trivial_rejects.append(r_idx)
                #     continue

                # in the case of a given convex hull, we can use it for trivial reject
                # all_inside, all_outside = vertices_fully_inside_outside_convex_hull(rv, clip_hull.equations)
                # if all_outside: # cannot trivially reject (going over the edge)
                #     # trivial_rejects.append(r_idx)
                #     to_be_clipped.append(r_idx)  
                #     continue

                all_inside = vertices_fully_inside_convex_hull2(rv, clip_hull.equations)
                
                if all_inside:
                    # we have to check if there was also a clip bbox given before accepting it
                    if orig_clip_bbox is None:
                        # there was none, so we can accept
                        trivial_accepts.append(r_idx)
                        continue
                    else:
                        # there was one, so we have to check with that as well
                        if vertices_fully_inside_aabbox(rv, orig_clip_bbox):
                            trivial_accepts.append(r_idx)
                            continue

            to_be_clipped.append(r_idx)

        indices_of_finite_ridges_to_be_clipped = to_be_clipped

        total_to_be_clipped = np.concatenate([indices_of_finite_ridges_to_be_clipped, indices_of_infinite_ridge_to_be_clipped])

        # for actual clipping we can provide the convex hull clip box if no other was given
        if orig_clip_bbox is None:
            orig_clip_bbox = clip_bbox  # at this point, clip_bbox is the same as the convex hull bbox

        # the rest has to be checked and clipped with aabbox
        # we can do the same as we do when we clip for sample points but use the halfspaces of the aabbox instead.
        pf(f"clip {len(total_to_be_clipped)} voronoi ridges")
        aabbox_equations = construct_halfspace_definition_for_aabbox(orig_clip_bbox)

        if clip_convex_hull:
            clip_equations = np.vstack([aabbox_equations, clip_hull.equations])
        else:
            clip_equations = aabbox_equations

        clipped_ridges, clipped_ridge_indices = clip_voronoi_ridge_vertices(
            vor=vor, clip_equations=clip_equations, indices_of_ridges_to_be_clipped=total_to_be_clipped,
            identify_infinity_regions_and_neighbors_results=identify_infinity_regions_and_neighbors_results, verbose=verbose
        )

        trivial_accept_ridges = [vor.vertices[vor.ridge_vertices[ridge_idx]] for ridge_idx in trivial_accepts]

        final_ridges = clipped_ridges + trivial_accept_ridges
        final_ridge_indices = clipped_ridge_indices + trivial_accepts
        pf(f"got a total of {len(trivial_accepts)} trivial accepts, {len(trivial_rejects)} trivial rejects, and {len(total_to_be_clipped)} to be clipped ridges")
        pf(f"of {len(total_to_be_clipped)} to be clipped ridges, {len(clipped_ridges)} are remaining")
    else:
        dividing_finite_ridges = dividing_ridges[selector]
        final_ridges = [vor.vertices[ridge] for ridge in dividing_finite_ridges]
        final_ridge_indices = relevant_ridges_indices[selector]

    ridge_vertices = make_object_array(final_ridges, dtype=object)

    pf(f"returning {len(ridge_vertices)} many ridges")

    if return_original_ridge_point_indices:
        # map ridge indices to face point indices
        final_face_point_indices = vor.ridge_points[final_ridge_indices]
        return ridge_vertices, final_face_point_indices
    return ridge_vertices


@numba.njit()
def compute_sample_points(clip_box: np.ndarray, bandwidths: np.ndarray, step_size: np.ndarray = None) -> tuple:
    N = clip_box.shape[0]

    mins = clip_box[:, 0]
    maxs = clip_box[:, 1]

    # sample frequency should be twice the bandwidth
    if step_size is None:
        step_size = bandwidths
    
    sample_points = [np.arange(mins[dim], maxs[dim], step_size[dim]) + bandwidths[dim] / 2.0 for dim in range(N)]

    sample_points_min = [sample_points[dim] - bandwidths[dim] / 2.0 for dim in range(N)]
    sample_points_max = [sample_points[dim] + bandwidths[dim] / 2.0 for dim in range(N)]

    return sample_points, sample_points_min, sample_points_max


@numba.njit()
def compute_trivial_rejects(ridge_vertices: list, sample_points_min: list, sample_points_max: list) -> tuple:
    N = len(sample_points_min)

    if len(ridge_vertices) == 0:
        # reject all as we don't have any ridges
        return [np.full((len(sample_points_min[dim]), 1), True) for dim in range(N)]

    min_dim_values = nb_stack([
        nb_min0(vertices) for vertices in ridge_vertices
    ])
    max_dim_values = nb_stack([
        nb_max0(vertices) for vertices in ridge_vertices
    ])

    # working with lists instead of pure numpy arrays. Have to do this since we don't have the same step sizes anymore.
    # less_than_min_rejects = np.apply_along_axis(lambda a: a < min_dim_values, 1, sample_points_max)
    # more_than_max_rejects = np.apply_along_axis(lambda a: a > max_dim_values, 1, sample_points_min)

    less_than_min_rejects = [nb_stack([sample_points_max[dim][i] < min_dim_values[:, dim] for i in range(len(sample_points_max[dim]))]) for dim in range(N)]
    more_than_max_rejects = [nb_stack([sample_points_min[dim][i] > max_dim_values[:, dim] for i in range(len(sample_points_min[dim]))]) for dim in range(N)]

    trivial_rejects = [np.logical_or(less_than_min_rejects[dim], more_than_max_rejects[dim]) for dim in range(N)]
    
    return trivial_rejects


def _compute_ridge_based_para_sense_1d(okays: np.ndarray, return_n_ridges: bool = False) -> tuple:
    dim = 0  # just 1D

    sensitivities = np.zeros((okays[dim].shape[0], ))

    for sample_idx in range(okays[dim].shape[0]):
        # get relevant rotation matrices of ridges per sample point
        selector = okays[dim][sample_idx, :]

        # ridge vertices not relevant
        # trivial rejects already removed with selector, so we can just sum up the amount of ridge vertices in the bin at the sample point
        volume = selector.sum()
        sensitivities[sample_idx] += volume

    if return_n_ridges:
        return sensitivities, np.vstack([sensitivities, sensitivities]).T
    return (sensitivities, )


def compute_ridge_based_transitions_2d_for_one_dim(dim: int, ridge_vertices: np.ndarray, okays: np.ndarray, return_n_ridges: bool, 
                                                   sample_points_min_dim: np.ndarray, sample_points_max_dim: np.ndarray) -> tuple:
    sensitivities = np.zeros((okays.shape[0], ))
    n_ridges = np.zeros((okays.shape[0], 2))
    
    for sample_idx in range(okays.shape[0]):
        # get relevant rotation matrices of ridges per sample point
        selector = okays[sample_idx, :]

        _n_ridges = selector.sum()

        if _n_ridges == 0:
            # there is no intersection of any ridge with the space remaining after clipping according to sample point and bandwidth
            continue
        if return_n_ridges:
            n_ridges[sample_idx, :] += _n_ridges

        relevant_ridge_vertices = ridge_vertices[selector]

        for i in range(len(relevant_ridge_vertices)):
            rv = relevant_ridge_vertices[i]
            
            # compute absolute gradient
            dy = np.abs(rv[0, 1-dim] - rv[1, 1-dim])
            dx = np.abs(rv[0, dim] - rv[1, dim])

            if dx == 0:
                volume = dy
            else:
                m = dy / dx

                # clip ridge to sample interval
                sp_min = sample_points_min_dim[sample_idx]
                sp_max = sample_points_max_dim[sample_idx]

                lower_ridge = rv[:, dim].min()
                upper_ridge = rv[:, dim].max()
                
                if lower_ridge < sp_min:
                    lower_ridge = sp_min
                if upper_ridge > sp_max:
                    upper_ridge = sp_max

                new_dx = upper_ridge - lower_ridge

                volume = new_dx * m
            
            sensitivities[sample_idx] += volume
        
    if return_n_ridges:
        return sensitivities, n_ridges
    return (sensitivities, )


def _compute_ridge_based_para_sense_2d(ridge_vertices: np.ndarray, sample_points_min: np.ndarray, sample_points_max: np.ndarray, 
                                       okays: np.ndarray, return_n_ridges: bool = False, pf=lambda x: x) -> tuple:
    N = 2
    sensitivities = []

    if return_n_ridges:
        n_ridges = []
    
    for dim in range(N):
        pf(f"dim {dim} for {okays[dim].shape[0]} samples")
        ret = compute_ridge_based_transitions_2d_for_one_dim(dim=dim, 
                                                             ridge_vertices=ridge_vertices,
                                                             okays=okays[dim],
                                                             return_n_ridges=return_n_ridges,
                                                             sample_points_min_dim=sample_points_min[dim],
                                                             sample_points_max_dim=sample_points_max[dim])
        
        sensitivities.append(ret[0])
        if return_n_ridges:
            n_ridges.append(ret[1])

    if return_n_ridges:
        return sensitivities, n_ridges
    return (sensitivities, )


def _compute_ridge_volume(rv: np.ndarray, lower_hs: np.ndarray, vh: np.ndarray, o: float, rds: float,
                          n: np.ndarray, dim: int, sample_points_min_value: float, 
                          sample_points_max_value: float, ch_equations: np.ndarray, dim_accessor: np.ndarray,
                          *, _debug_skipped_ridges_reasons: dict = None, pf=lambda x: x) -> tuple[float, bool]:
    exclude_both, new_lower_dim_halfspaces = _compute_lower_dim_halfspaces(lower_hs, vh, n, o, dim, sample_points_min_value, sample_points_max_value, ch_equations)

    # print("-")
    # print(f"{lower_hs}, {vh}, {n}, {o}, {dim}, {sample_points_min_value}, {sample_points_max_value}, {ch_equations}, {exclude_both}, {new_lower_dim_halfspaces}")
    if not exclude_both:
        try:
            # compute new interior point
            lower_dim_interior_points = get_interior_point_from_halfspaces(new_lower_dim_halfspaces)[0]
        except AssertionError as e:
            # pf(f"skipping ridge {idx} due to linprog failing to find interior point")
            _debug_skipped_ridges_reasons["linprog_error"] += 1
            return 0.0, True  # could not find interior point, linprog failed

        # add half spaces to convex hull equations of ridges to get clipped convex hulls
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                clipped_lower_dim_halfspaces = HalfspaceIntersection(new_lower_dim_halfspaces, lower_dim_interior_points)
            except QhullError as e:
                if str(e).startswith("QH6154") or str(e).startswith("QH6271") or str(e).startswith("QH6297") \
                    or str(e).startswith("QH7086") or str(e).startswith("QH6347") or str(e).startswith("QH6348"):
                    # QH6154 Qhull precision error: Initial simplex is flat (facet 1 is coplanar with the interior point)
                    # QH6297 Qhull precision error (qh_check_maxout): large increase in qh.max_outside during post-processing
                    # QhullError: QH6271 qhull topology error (qh_check_dupridge): wide merge
                    # QH7086 Qhull precision warning: repartition coplanar point
                    # QhullError: QH6347 qhull precision error (qh_mergefacet): wide merge for facet ..
                    return 0.0, True
                elif str(e).startswith("QH6023"):
                    # QH6023 qhull input error: feasible point is not clearly inside halfspace
                    # pf(f"skipping ridge {idx} due to infeasible point")
                    _debug_skipped_ridges_reasons["infeasible_point"] += 1
                    return 0.0, True
                else:
                    pf(f"ignored error #2 '{str(e)[:5]}': {str(e)}")
                    return 0.0, True
                    # raise e
            except RuntimeWarning as re:
                # pf(f"skipped ridge {idx} due to {str(re)}")
                if str(re).startswith("divide by zero encountered in divide"):
                    _debug_skipped_ridges_reasons["runtime_warning_div_by_zero"] += 1
                    return 0.0, True  # checking for divide by zero encountered in divide
                raise re

        # get new intersection points
        clipped_lower_dim_intersection_points = clipped_lower_dim_halfspaces.intersections
        # rotate intersection points back to original dimension
        intersection_points = np.column_stack([clipped_lower_dim_intersection_points, np.full((len(clipped_lower_dim_intersection_points), ), rds)]) @ vh
    else:
        # they are the same as before
        intersection_points = rv
            # project on n-1 dim without the current axis
    projected_intersection_points = intersection_points[:, dim_accessor]
            # compute clipped convex hulls
    # handle cases where convex hull would be flat (we can just skip those)
    try:
        clipped_convex_hulls = ConvexHull(projected_intersection_points)
    except QhullError as e:
        if not (str(e).startswith("QH6154") or str(e).startswith("QH6013") 
                or str(e).startswith("QH6271") or str(e).startswith("QH7088")
                or str(e).startswith("QH7086") or str(e).startswith("QH6347")
                or str(e).startswith("QH6297") or str(e).startswith("QH6348")):
            pf(f"ignored error #1 '{str(e)[:5]}': {str(e)}")
            # raise e
            return 0.0, True
        else:
            # Error QH6154 means that the simplex is flat, i.e., volume is zero
            # Error QH6013 means basically the same but for 2d: "input is less than 2-dimensional since all points have the same x coordinate"
            # Error QH6271 qhull topology error (qh_check_dupridge): wide facet merge
            # QhullError: QH7088 Qhull precision warning: in post-processing (qh_check_maxout)
            # QhullError: QH7086 Qhull precision warning: repartition coplanar point p18 from f327 as an outside point above hidden facet f482 dist 0.00084 nearest vertices 0.037
            # QH6297 Qhull precision error (qh_check_maxout): large increase in qh.max_outside during post-processing dist 0.00084 (21464063054.1x).  See warning QH0032/QH0033.  Allow with 'Q12' (allow-wide) and 'Pp'
            # QhullError: QH6347 qhull precision error (qh_mergefacet): wide merge
            # pf(f"skipping ridge {idx} due to being flat (volume is zero) after clipping and projecting")
            # QhullError: QH6348 qhull precision error (qh_mergefacet): wide merge for pinched facet
            _debug_skipped_ridges_reasons["flat"] += 1
            return 0.0, True

    volume = clipped_convex_hulls.volume

    return volume, False


def compute_ridge_based_transitions_for_one_dimension_of_nd(dim: int, N: int, okays: np.ndarray, 
                                                            ridge_vertices: np.ndarray, Vhs: np.ndarray, normals: np.ndarray, offsets: np.ndarray, d_s: np.ndarray, 
                                                            lower_dim_convex_hull_equations: np.ndarray, 
                                                            sample_points_min_dim: np.ndarray, 
                                                            sample_points_max_dim: np.ndarray, dim_accessor: np.ndarray,
                                                            return_n_ridges: bool = False, pf=lambda x: x) -> tuple:
    n_ridges = np.zeros((okays.shape[0], 2))
    sensitivities = np.zeros((okays.shape[0], ))
    
    for sample_idx in range(okays.shape[0]):
        _debug_skipped_ridges = 0
        _debug_skipped_ridges_reasons = {
            "linprog_error": 0,
            "runtime_warning_div_by_zero": 0,
            "infeasible_point": 0,
            "flat": 0,
        }

        # pf(sample_idx)
        # get relevant rotation matrices of ridges per sample point
        selector = okays[sample_idx, :]

        _n_ridges = selector.sum()

        if _n_ridges == 0:
            # there is no intersection of any ridge with the space remaining after clipping according to sample point and bandwidth
            continue

        if return_n_ridges:
            n_ridges[sample_idx, 0] += _n_ridges

        relevant_ridge_vertices = ridge_vertices[selector]

        relevant_vhs = Vhs[selector]
        relevant_normals = normals[selector]
        relevant_offsets = offsets[selector]
        # relevant_lower_dim_proj = lower_dim_proj[selector]
        relevant_d_s = d_s[selector]
        relevant_lower_dim_convex_hull_equations = lower_dim_convex_hull_equations[selector]
        
        # clip to sample point half space
        # we can just take the convex hull equations and stack the equations for the dividing half spaces on top of them
        # lower_hs, upper_hs = make_halfspaces(sample_points_min[sample_idx, dim], sample_points_max[sample_idx, dim], N, dim)
        lower_hs = make_lower_halfspace(sample_points_min_dim[sample_idx], N, dim)  # optimized: we don't need to construct the upper hs as we can easily compute it based on the lower hs

        sample_points_min_value = sample_points_min_dim[sample_idx]
        sample_points_max_value = sample_points_max_dim[sample_idx]

        for i in range(len(relevant_ridge_vertices)):
            _debug_skipped_ridges += 1

            vh = relevant_vhs[i]
            n = relevant_normals[i]
            o = relevant_offsets[i]
            ch_equations = relevant_lower_dim_convex_hull_equations[i]
            rv = relevant_ridge_vertices[i]
            rds = relevant_d_s[i]
            
            volume, skip = _compute_ridge_volume(rv=rv, lower_hs=lower_hs, vh=vh, o=o, rds=rds,
                                                 n=n, dim=dim, sample_points_min_value=sample_points_min_value,
                                                 sample_points_max_value=sample_points_max_value,
                                                 ch_equations=ch_equations, dim_accessor=dim_accessor,
                                                 _debug_skipped_ridges_reasons=_debug_skipped_ridges_reasons,
                                                 pf=pf)
            if skip:
                continue

            # compute convex hull volumes and add to sensitivities
            sensitivities[sample_idx] += volume
            if return_n_ridges:
                n_ridges[sample_idx, 1] += 1
            _debug_skipped_ridges -= 1
        
        pf(f"skipped ridges: sample idx: {sample_idx}, ridges: {_debug_skipped_ridges} / {len(relevant_ridge_vertices)}; {_debug_skipped_ridges_reasons}")
    
    if return_n_ridges:
        return sensitivities, n_ridges
    return (sensitivities, )


def _compute_ridge_based_para_sense_nd(ridge_vertices: np.ndarray, Vhs: np.ndarray, normals: np.ndarray, offsets: np.ndarray, d_s: np.ndarray, 
                                       lower_dim_convex_hull_equations: np.ndarray, sample_points_min: np.ndarray, sample_points_max: np.ndarray, 
                                       okays: np.ndarray, N: int, return_n_ridges: bool = False, pf=lambda x: x) -> tuple:
    available_dims = np.arange(N)
    dims_accessor = np.array([
        available_dims != i for i in range(N)
    ])

    sensitivities = []
    if return_n_ridges:
        n_ridges = []

    # for each dimension and sample point, compute local sensitivity across all ridges based on their convex hull
    for dim in range(N):
        pf(f"dim {dim} for {okays[dim].shape[0]} samples")
        da = dims_accessor[dim]
        ret = compute_ridge_based_transitions_for_one_dimension_of_nd(dim=dim, 
                                                                        N=N,
                                                                        okays=okays[dim], 
                                                                        ridge_vertices=ridge_vertices,
                                                                        Vhs=Vhs,
                                                                        normals=normals,
                                                                        offsets=offsets,
                                                                        d_s=d_s,
                                                                        lower_dim_convex_hull_equations=lower_dim_convex_hull_equations,
                                                                        sample_points_min_dim=sample_points_min[dim],
                                                                        sample_points_max_dim=sample_points_max[dim],
                                                                        return_n_ridges=return_n_ridges,
                                                                        dim_accessor=da,
                                                                        pf=pf)
        sensitivities.append(ret[0])
        if return_n_ridges:
            n_ridges.append(ret[1])
    
    if return_n_ridges:
        return sensitivities, n_ridges
    return (sensitivities, )


def compute_ridge_transformations(ridge_vertices: list, *, pf=lambda x: x) -> tuple:
    """
    NOTE: returned ch_equations are useless for N=2.
    """
    N = ridge_vertices[0].shape[-1]
    
    pf("compute projection matrices of ridges")

    # not necessary for 1D or 2D
    VhsFull = [rotate_to_lower_dimensional_hyperplane_nb(vertices) for vertices in ridge_vertices]
    Vhs = np.stack([vhsf[-1] for vhsf in VhsFull])
    normals = Vhs[:, -1, :]
    offsets = (normals * np.vstack([rv[0] for rv in ridge_vertices])).sum(-1)

    lower_dim_proj = make_object_array([rv @ vh.T for rv, vh in zip(ridge_vertices, Vhs)], dtype=object)
    d_s = np.array([ldp[0, -1] for ldp in lower_dim_proj])

    # not possible for 1D or 2D
    ch_equations = []
    selector = []
    skipped = []

    volumes = []

    z = np.zeros(N)
    for i, ldp in enumerate(lower_dim_proj):
        if len(VhsFull[i][1]) != N:  # may happen, don't know how though (TODO: can we check this?)
            skipped.append(i)  # surface will be flat
            continue

        n_close_ev_to_zero = np.isclose(VhsFull[i][1], z, rtol=1e-06, atol=1e-12).sum()

        if n_close_ev_to_zero > 1:
            skipped.append(i)  # surface will be flat
            continue

        if N == 2:
            ch_equations.append([])  # not used for 2D
            selector.append(i)
            volumes.append(np.linalg.norm(ridge_vertices[i][0] - ridge_vertices[i][1]))  # distance between both points, i.e., length of the ridge
            continue

        try:
            ch = ConvexHull(ldp[:, :-1])
            ch_equations.append(ch.equations)
            selector.append(i)
            volumes.append(ch.volume)
        except QhullError as e:
            skipped.append(i)
            # usually happens when projected surface would be flat (at least two eigenvalues are close to zero).. trying to skip this above to avoid error
            pf(f"convex hull error.... #zero-eigenvalues: {n_close_ev_to_zero}; {str(e)[:50]}")
            pass

    # adjust skipped
    ridge_vertices = ridge_vertices[selector]
    Vhs = Vhs[selector]
    normals = normals[selector]
    offsets = offsets[selector]
    # lower_dim_proj = lower_dim_proj[selector]
    d_s = d_s[selector]
    volumes = np.array(volumes)

    lower_dim_convex_hull_equations = make_object_array(ch_equations, dtype=object)

    pf(f"skipped {len(skipped)} ridges due to errors... remaining are {len(ridge_vertices)} ridges")

    return Vhs, normals, offsets, d_s, lower_dim_convex_hull_equations, volumes, np.asarray(selector)


def ridge_based_para_sense_preparations(ridge_vertices: np.ndarray, bandwidths: np.ndarray, clip_box: np.ndarray, 
                           step_size: np.ndarray = None, sample_points_min_max: tuple = None,
                           *, verbose: bool = False) -> tuple:    
    def pf(*args):
        if verbose:
            print(*args)
        else:
            pass
    
    pf(f"got {len(ridge_vertices)} ridges")

    # assert len(ridge_vertices) > 0 and len(ridge_vertices[0]) > 0, "got empty ridges... cannot compute dimensionality or sample points"
    # N = ridge_vertices[0].shape[-1]

    N = len(bandwidths)

    assert N > 0, "must be at least 1-dimensional"

    pf(f"got dimensionality {N}")

    # construct rotation matrices to rotate facets into their hyperplane
    Vhs, normals, offsets, d_s, lower_dim_convex_hull_equations, volumes, selector = None, None, None, None, None, None, np.arange(len(ridge_vertices))
    if N > 1 and len(ridge_vertices) > 0:
        Vhs, normals, offsets, d_s, lower_dim_convex_hull_equations, volumes, selector = compute_ridge_transformations(ridge_vertices=ridge_vertices, pf=pf)

    ridge_vertices = ridge_vertices[selector]

    pf("compute sample points")
    if sample_points_min_max is None:
        sample_points, sample_points_min, sample_points_max = compute_sample_points(clip_box=clip_box, bandwidths=bandwidths, step_size=step_size)
    else:
        sample_points, sample_points_min, sample_points_max = sample_points_min_max

    pf("compute trivial rejects")
    if len(ridge_vertices) == 0:
        # early termination
        okays = [np.full((len(sample_points[dim]), 0), True) for dim in range(N)]            
    else:
        trivial_rejects = compute_trivial_rejects(
            ridge_vertices=ridge_vertices.tolist(), sample_points_min=sample_points_min, sample_points_max=sample_points_max
        )

        okays = [np.invert(trivial_rejects[dim]) for dim in range(N)]

    pf("number of sample points per dimension: ", [len(sn) for sn in sample_points])
    return sample_points, sample_points_min, sample_points_max, okays, Vhs, normals, offsets, d_s, lower_dim_convex_hull_equations, volumes, selector



def ridge_based_para_sense(ridge_vertices: np.ndarray, bandwidths: np.ndarray, clip_box: np.ndarray, 
                           step_size: np.ndarray = None, return_n_ridges: bool = False, sample_points_min_max: tuple = None,
                           ridge_based_para_sense_preparations_result: tuple = None,
                           *, verbose: bool = False) -> tuple:
    """
    The ridge-based parameter sensitivity algorithm.

    Given a set of sample Points S for a given dimension d and bandwidth B, for each sample point s:
        1. Clip each facet in dimension d to [S_d - B, S_d + B].
        2. Project each facet to N-1 dimesnional space by remove the dimension d from the facets vertices.
        3. Compute the N-1-"volume" of each facet and sum them up. -> resulting sensitivity regarding dimension d at sample point S for bandwidth B.

    :params
    @ridge_vertices: A list of ridges, each defined by an array of vertices that define the ridge.
        dtype=object, each element is a np.ndarray of vertices with shape [*, N], where * is the amount of vertices that define each ridge (variable per ridge but generally * >= N).
    @bandwidths: np.ndarray of shape (N, ). The bandwidth per dimension per sample point to compute the local sensitivity
    @clip_box: np.ndarray of shape (N, 2). The axis-aligned bounding box in which to sample. Defines minimum and maximum sample point.
    @step_size: np.ndarray, (N, ). The step size for sampling. If not given, it will be twice the bandwidth. 
        If step_size is equal to twice the bandwidth (2.0 * bandwidths), then there is no overlap between sampled bins and there will be no gap between bins.
        The sum of the volume of all bins should be equal to the global volume of all ridges along the dimension.
    @return_n_ridges: bool. 
        If True, returns an additional list with the amount of total ridges that fall into each sampled bin and which were considered for sensitivity computation.
        NOTE: This does not include degenerate ridges.
        NOTE: In the 1D-case, this is exactly the same as the sensitivity.
        It returns a list of np.ndarrays with the same leading shapes as the sensitivites except that there are two last-dimensions (M, 2)
        The first is the total amount of relevant ridges per sample point, the second is the amount of ridges that contributed to the sensitivity. 
    @return: tuple: (sample_points, sensitivities, (opt.) n_ridges, (opt.) max_volumes)
    """
    def pf(*args):
        if verbose:
            print(*args)
        else:
            pass
    
    N = len(bandwidths)

    if ridge_based_para_sense_preparations_result is None:
        ridge_based_para_sense_preparations_result = ridge_based_para_sense_preparations(ridge_vertices=ridge_vertices, bandwidths=bandwidths, 
                                                                                         clip_box=clip_box, step_size=step_size, 
                                                                                         sample_points_min_max=sample_points_min_max, 
                                                                                         verbose=verbose)

    sample_points, sample_points_min, sample_points_max, okays, Vhs, normals, offsets, d_s, lower_dim_convex_hull_equationss, volumes, _ = \
        ridge_based_para_sense_preparations_result
    
    pf(f"perform {N} dimensional rb-para sense")
    if N == 1:
        return sample_points, *_compute_ridge_based_para_sense_1d(okays, return_n_ridges=return_n_ridges)
    elif N == 2:
        return sample_points, *_compute_ridge_based_para_sense_2d(ridge_vertices, sample_points_min, sample_points_max, okays, return_n_ridges=return_n_ridges, pf=pf)
    else:
        if len(ridge_vertices) == 0:
            sensitivities = [np.zeros((okays[dim].shape[0], )) for dim in range(N)]
            if return_n_ridges:
                n_ridges = [np.zeros((okays[dim].shape[0], 2)) for dim in range(N)]
                return sample_points, sensitivities, n_ridges
            else:
                return sample_points, sensitivities
        else:
            return sample_points, *_compute_ridge_based_para_sense_nd(ridge_vertices, Vhs, normals, offsets, d_s, lower_dim_convex_hull_equationss, sample_points_min, sample_points_max, okays, N, return_n_ridges=return_n_ridges, pf=pf)


def __compute_okays(label_idx_a: int, label_idx_b: int, selector_matrix_dim: np.ndarray, okays_dim: np.ndarray, rv_prep_selector_indices: np.ndarray):
    selector = selector_matrix_dim[label_idx_a, label_idx_b]

    # merge selector with selector from preparation
    selector_indices = np.argwhere(selector)

    # adjust okays
    okays_ = np.full(okays_dim.shape, False)
    # find indices that the selector_indices map to in the okays array
    _, _, okay_indices_ = np.intersect1d(selector_indices, rv_prep_selector_indices, assume_unique=True, return_indices=True)

    if len(okay_indices_) > 0:
        okays_[:, okay_indices_] = okays_dim[:, okay_indices_]
    
    return okays_

# mp_lock = mp.Lock()

def __crbps1d_mp(args: tuple): 
    # label_idx_a: int, label_idx_b: int, M_attr: tuple, R_attr: tuple, okays_dim_attr: tuple, selector_matrix_dim_attr: tuple, rv_prep_selector_indices_attr: tuple
    label_tuples_args, (M_attr, R_attr, okays_dim_attr, selector_matrix_dim_attr, rv_prep_selector_indices_attr) = args

    M, M_shm = get_shared_ndarray(M_attr)
    R, R_shm = get_shared_ndarray(R_attr)
    
    okays_dim, okays_dim_shm = get_shared_ndarray(okays_dim_attr)
    selector_matrix_dim, selector_matrix_dim_shm = get_shared_ndarray(selector_matrix_dim_attr)
    rv_prep_selector_indices, rv_prep_selector_indices_shm = get_shared_ndarray(rv_prep_selector_indices_attr)

    for label_idx_a, label_idx_b in label_tuples_args:
        okays_ = __compute_okays(label_idx_a, label_idx_b, selector_matrix_dim, okays_dim, rv_prep_selector_indices)

        sensitivities, n_ridges = _compute_ridge_based_para_sense_1d(okays_, return_n_ridges=True)

        M[label_idx_a, label_idx_b] = sensitivities
        R[label_idx_a, label_idx_b] = n_ridges

    # close all shared array views
    M_shm.close(), R_shm.close(), okays_dim_shm.close(), selector_matrix_dim_shm.close(), rv_prep_selector_indices_shm.close()


def __crbt2dfod(args: tuple):
    # dim: int, label_idx_a: int, label_idx_b: int, M_attr: tuple, R_attr: tuple, okays_dim_attr: tuple, selector_matrix_dim_attr: tuple, rv_prep_selector_indices_attr: tuple,
    # combined_ridge_vertices_attr: tuple, crv_offsets_attr: tuple, crv_sizes_attr: tuple,
    # sample_points_min_dim_attr: tuple, sample_points_max_dim_attr: tuple
    label_tuples_args, (dim, M_attr, R_attr, okays_dim_attr, selector_matrix_dim_attr, rv_prep_selector_indices_attr, 
        combined_ridge_vertices_attr, crv_offsets_attr, crv_sizes_attr, sample_points_min_dim_attr, sample_points_max_dim_attr) = args

    M, M_shm = get_shared_ndarray(M_attr)
    R, R_shm = get_shared_ndarray(R_attr)

    okays_dim, okays_dim_shm = get_shared_ndarray(okays_dim_attr)
    selector_matrix_dim, selector_matrix_dim_shm = get_shared_ndarray(selector_matrix_dim_attr)
    rv_prep_selector_indices, rv_prep_selector_indices_shm = get_shared_ndarray(rv_prep_selector_indices_attr)

    crv, crv_shm = get_shared_ndarray(combined_ridge_vertices_attr)
    crv_offsets, crv_offsets_shm = get_shared_ndarray(crv_offsets_attr)
    crv_sizes, crv_sizes_shm = get_shared_ndarray(crv_sizes_attr)

    ridge_vertices = uncombine_variable_sized_arrays_from_one(crv, crv_offsets, crv_sizes)

    sample_points_min_dim, sample_points_min_dim_shm = get_shared_ndarray(sample_points_min_dim_attr)
    sample_points_max_dim, sample_points_max_dim_shm = get_shared_ndarray(sample_points_max_dim_attr)

    for label_idx_a, label_idx_b in label_tuples_args:
        okays_ = __compute_okays(label_idx_a, label_idx_b, selector_matrix_dim, okays_dim, rv_prep_selector_indices)

        sensitivities, n_ridges = compute_ridge_based_transitions_2d_for_one_dim(dim=dim, 
                                                                     ridge_vertices=ridge_vertices,
                                                                     okays=okays_,
                                                                     return_n_ridges=True,
                                                                     sample_points_min_dim=sample_points_min_dim,
                                                                     sample_points_max_dim=sample_points_max_dim)

        M[label_idx_a, label_idx_b] = sensitivities
        R[label_idx_a, label_idx_b] = n_ridges

    # close all shared array views
    M_shm.close(), R_shm.close(), okays_dim_shm.close(), selector_matrix_dim_shm.close(), rv_prep_selector_indices_shm.close()
    crv_shm.close(), crv_offsets_shm.close(), crv_sizes_shm.close(), sample_points_min_dim_shm.close(), sample_points_max_dim_shm.close()


def __crbtfodond(args: tuple):
    # N: int, dim: int, label_idx_a: int, label_idx_b: int, M_attr: tuple, R_attr: tuple, okays_dim_attr: tuple, selector_matrix_dim_attr: tuple, rv_prep_selector_indices_attr: tuple,
    # combined_ridge_vertices_attr: tuple, crv_offsets_attr: tuple, crv_sizes_attr: tuple,
    # sample_points_min_dim_attr: tuple, sample_points_max_dim_attr: tuple, 
    # Vhs_attr: tuple, normals_attr: tuple, offsets_attr: tuple, d_s_attr: tuple, 
    # combined_lower_dim_convex_hull_equations_attr: tuple, cldche_offsets_attr: tuple, cldche_sizes_attr: tuple
    label_tuples_args, (N, dim, M_attr, R_attr, okays_dim_attr, selector_matrix_dim_attr, rv_prep_selector_indices_attr,
                        combined_ridge_vertices_attr, crv_offsets_attr, crv_sizes_attr, sample_points_min_dim_attr,
                        sample_points_max_dim_attr,
                        Vhs_attr, normals_attr, offsets_attr, d_s_attr, combined_lower_dim_convex_hull_equations_attr,
                        cldche_offsets_attr, cldche_sizes_attr, verbose) = args

    cp_name = None
    def pf(*args):
        if verbose:
            if cp_name is None:
                print(*args)
            else:
                print(f"{cp_name}: ", *args)
        else:
            pass

    pf(f"current_process: {mp.current_process()}")
    cp_name = mp.current_process().name

    pf(f"get shared arrays...")
    M, M_shm = get_shared_ndarray(M_attr)
    R, R_shm = get_shared_ndarray(R_attr)

    okays_dim, okays_dim_shm = get_shared_ndarray(okays_dim_attr)
    selector_matrix_dim, selector_matrix_dim_shm = get_shared_ndarray(selector_matrix_dim_attr)
    rv_prep_selector_indices, rv_prep_selector_indices_shm = get_shared_ndarray(rv_prep_selector_indices_attr)

    crv, crv_shm = get_shared_ndarray(combined_ridge_vertices_attr)
    crv_offsets, crv_offsets_shm = get_shared_ndarray(crv_offsets_attr)
    crv_sizes, crv_sizes_shm = get_shared_ndarray(crv_sizes_attr)

    ridge_vertices = make_object_array(uncombine_variable_sized_arrays_from_one(crv, crv_offsets, crv_sizes))

    sample_points_min_dim, sample_points_min_dim_shm = get_shared_ndarray(sample_points_min_dim_attr)
    sample_points_max_dim, sample_points_max_dim_shm = get_shared_ndarray(sample_points_max_dim_attr)

    pf(f"get rv preparations")
    Vhs, Vhs_shm = get_shared_ndarray(Vhs_attr)
    normals, normals_shm = get_shared_ndarray(normals_attr)
    offsets, offsets_shm = get_shared_ndarray(offsets_attr)
    d_s, d_s_shm = get_shared_ndarray(d_s_attr)

    pf(f"construct lower dim convex hull equations")
    combined_lower_dim_convex_hull_equations, combined_lower_dim_convex_hull_equations_shm = get_shared_ndarray(combined_lower_dim_convex_hull_equations_attr)
    cldche_offsets, cldche_offsets_shm = get_shared_ndarray(cldche_offsets_attr)
    cldche_sizes, cldche_sizes_shm = get_shared_ndarray(cldche_sizes_attr)

    lower_dim_convex_hull_equations = make_object_array(uncombine_variable_sized_arrays_from_one(combined_lower_dim_convex_hull_equations, cldche_offsets, cldche_sizes))

    pf(f"compute dim accessor")
    da = np.arange(N) != dim

    pf(f"run work for {len(label_tuples_args)} tuple")
    for idx, (label_idx_a, label_idx_b) in enumerate(label_tuples_args):
        pf(f"compute {idx+1}/{len(label_tuples_args)}")
        okays_ = __compute_okays(label_idx_a, label_idx_b, selector_matrix_dim, okays_dim, rv_prep_selector_indices)

        # pf(f"okays_: {okays_}")

        sensitivities, n_ridges = compute_ridge_based_transitions_for_one_dimension_of_nd(dim=dim, 
                                                                                N=N,
                                                                                okays=okays_, 
                                                                                ridge_vertices=ridge_vertices,
                                                                                Vhs=Vhs,
                                                                                normals=normals,
                                                                                offsets=offsets,
                                                                                d_s=d_s,
                                                                                lower_dim_convex_hull_equations=lower_dim_convex_hull_equations,
                                                                                sample_points_min_dim=sample_points_min_dim,
                                                                                sample_points_max_dim=sample_points_max_dim,
                                                                                return_n_ridges=True,
                                                                                dim_accessor=da,
                                                                                pf=pf
                                                                                )

        M[label_idx_a, label_idx_b] = sensitivities
        R[label_idx_a, label_idx_b] = n_ridges

    M_shm.close(), R_shm.close(), okays_dim_shm.close(), selector_matrix_dim_shm.close(), rv_prep_selector_indices_shm.close()
    crv_shm.close(), crv_offsets_shm.close(), crv_sizes_shm.close(), sample_points_min_dim_shm.close(), sample_points_max_dim_shm.close()
    Vhs_shm.close(), normals_shm.close(), offsets_shm.close(), d_s_shm.close(), combined_lower_dim_convex_hull_equations_shm.close(), cldche_offsets_shm.close(), cldche_sizes_shm.close()

    pf("done..")


def _build_directed_transition_cubes_mp(selector_matrix: np.ndarray, ridge_vertices: np.ndarray, bandwidths: np.ndarray, clip_box: np.ndarray, step_size: np.ndarray = None, 
                                        *, n_proc: int = None, 
                                        sample_points_min_max: tuple = None,
                                        ridge_based_para_sense_preparations_result: tuple = None, 
                                        verbose: bool = False) -> list:
    """
    Same as build_directed_transition_cubes but uses multiple processes to speed up the computation.
    Each call of compute_ridge_based_transitions_* will run in its own process.

    @n_proc: number of processes to use
    """
    def pf(*args):
        if verbose:
            print(*args)
        else:
            pass

    # NOTE: we don't need lock as we wait after each dimension and the writes do not overlap in memory adresses
    if n_proc <= 0:
        n_proc = None

    N = selector_matrix.shape[0]
    L = selector_matrix.shape[1]

    if sample_points_min_max is None:
        sample_points_min_max = compute_sample_points(clip_box=clip_box, bandwidths=bandwidths, step_size=step_size)

    ret = []

    for dim in range(N):
        n_samples_dim_i = sample_points_min_max[0][dim].shape[0]
        ret.append((np.zeros((L, L, n_samples_dim_i)), np.zeros((L, L, n_samples_dim_i, 2))))

    if ridge_based_para_sense_preparations_result is None:
        ridge_based_para_sense_preparations_result = ridge_based_para_sense_preparations(ridge_vertices=ridge_vertices,
                                                                                     bandwidths=bandwidths,
                                                                                     clip_box=clip_box,
                                                                                     step_size=step_size,
                                                                                     sample_points_min_max=sample_points_min_max,
                                                                                     verbose=False)
    
    _, sample_points_min, sample_points_max, okays, Vhs, normals, offsets, d_s, lower_dim_convex_hull_equations, volumes, rv_prep_selector_indices = \
        ridge_based_para_sense_preparations_result

    selector_matrix = selector_matrix[..., rv_prep_selector_indices]
    ridge_vertices = ridge_vertices[rv_prep_selector_indices]

    if len(ridge_vertices) == 0:
        return sample_points_min_max, ret

    pf(f"run shared memory manager...")
    with SharedMemoryManager() as smn:
        rv_prep_selector_indices_shared, rv_prep_selector_indices_shm_buffer = create_shared_ndarray(rv_prep_selector_indices, smn=smn)

        #if N > 2:
        #    available_dims = np.arange(N)
        #    dims_accessor = np.array([
        #        available_dims != i for i in range(N)
        #    ])
        #    dims_accessor_shared, dims_accessor_shm_buffer = create_shared_ndarray(dims_accessor)

        combined_ridge_vertices, crv_offsets, crv_sizes  = combine_variable_sized_arrays_to_one(ridge_vertices)

        if N > 1:
            crv_shared, crv_shm_buffer = create_shared_ndarray(combined_ridge_vertices, smn=smn)
            crv_offsets_shared, crv_offsets_shm_buffer = create_shared_ndarray(crv_offsets, smn=smn)
            crv_sizes_shared, crv_sizes_shm_buffer = create_shared_ndarray(crv_sizes, smn=smn)

        if N > 2:
            Vhs_shared, Vhs_shm_buffer = create_shared_ndarray(Vhs, smn=smn)
            normals_shared, normals_shm_buffer = create_shared_ndarray(normals, smn=smn)
            offsets_shared, offsets_shm_buffer = create_shared_ndarray(offsets, smn=smn)
            d_s_shared, d_s_shm_buffer = create_shared_ndarray(d_s, smn=smn)

            combined_lower_dim_convex_hull_equations, cldchw_offs, cldchw_szs = combine_variable_sized_arrays_to_one(lower_dim_convex_hull_equations)
            combined_lower_dim_convex_hull_equations_shared, combined_lower_dim_convex_hull_equations_shm_buffer = create_shared_ndarray(combined_lower_dim_convex_hull_equations, smn=smn)
            cldchw_offs_shared, cldchw_offs_shm_buffer = create_shared_ndarray(cldchw_offs, smn=smn)
            cldchw_szs_shared, cldchw_szs_shm_buffer = create_shared_ndarray(cldchw_szs, smn=smn)

        chunksize = 1

        label_tuples = []
        for label_idx_a in range(L):
            for label_idx_b in range(L):
                label_tuples.append((label_idx_a, label_idx_b))

        label_tuples_args = list(n_chunks(label_tuples, n_proc if n_proc is not None else os.cpu_count()))  # os.process_cpu_count()))

        pf(f"start process pool... n_proc={n_proc if n_proc is not None else os.cpu_count()}")
        with mp.Pool(processes=n_proc) as pool:
            for dim in range(N):
                selector_matrix_dim = selector_matrix[dim]
                M, R = ret[dim]

                okays_dim = okays[dim]
                sample_points_min_dim = sample_points_min[dim]
                sample_points_max_dim = sample_points_max[dim]

                # make relevant arrays shared memory
                pf(f"run second shared memory manager...")
                with SharedMemoryManager() as smm2:
                    M_shared, M_shm_buffer = create_shared_ndarray(M, smn=smm2)
                    R_shared, R_shm_buffer = create_shared_ndarray(R, smn=smm2)
                    selector_matrix_dim_shared, selector_matrix_dim_shm_buffer = create_shared_ndarray(selector_matrix_dim, smn=smm2)
                    sample_points_min_dim_shared, sample_points_min_dim_shm_buffer = create_shared_ndarray(sample_points_min_dim, smn=smm2)
                    sample_points_max_dim_shared, sample_points_max_dim_shm_buffer = create_shared_ndarray(sample_points_max_dim, smn=smm2)
                    okays_dim_shared, okays_dim_shm_buffer = create_shared_ndarray(okays_dim, smn=smm2)

                    if N == 1:
                        pool.map(__crbps1d_mp, zip(label_tuples_args, repeat((
                            (M_shm_buffer.name, M_shared.shape, M_shared.dtype),
                            (R_shm_buffer.name, R_shared.shape, R_shared.dtype),
                            (okays_dim_shm_buffer.name, okays_dim_shared.shape, okays_dim_shared.dtype),
                            (selector_matrix_dim_shm_buffer.name, selector_matrix_dim_shared.shape, selector_matrix_dim_shared.dtype),
                            (rv_prep_selector_indices_shm_buffer.name, rv_prep_selector_indices_shared.shape, rv_prep_selector_indices_shared.dtype)
                        ), len(label_tuples_args))), chunksize)
                    elif N == 2:
                        pool.map(__crbt2dfod, zip(label_tuples_args, repeat((
                            dim,
                            (M_shm_buffer.name, M_shared.shape, M_shared.dtype),
                            (R_shm_buffer.name, R_shared.shape, R_shared.dtype),
                            (okays_dim_shm_buffer.name, okays_dim_shared.shape, okays_dim_shared.dtype),
                            (selector_matrix_dim_shm_buffer.name, selector_matrix_dim_shared.shape, selector_matrix_dim_shared.dtype),
                            (rv_prep_selector_indices_shm_buffer.name, rv_prep_selector_indices_shared.shape, rv_prep_selector_indices_shared.dtype),
                            (crv_shm_buffer.name, crv_shared.shape, crv_shared.dtype),
                            (crv_offsets_shm_buffer.name, crv_offsets_shared.shape, crv_offsets_shared.dtype),
                            (crv_sizes_shm_buffer.name, crv_sizes_shared.shape, crv_sizes_shared.dtype),
                            (sample_points_min_dim_shm_buffer.name, sample_points_min_dim_shared.shape, sample_points_min_dim_shared.dtype),
                            (sample_points_max_dim_shm_buffer.name, sample_points_max_dim_shared.shape, sample_points_max_dim_shared.dtype),
                        ), len(label_tuples_args))), chunksize)
                    else:
                        pool.map(__crbtfodond, zip(label_tuples_args, repeat((
                            N, dim, 
                            (M_shm_buffer.name, M_shared.shape, M_shared.dtype),
                            (R_shm_buffer.name, R_shared.shape, R_shared.dtype),
                            (okays_dim_shm_buffer.name, okays_dim_shared.shape, okays_dim_shared.dtype),
                            (selector_matrix_dim_shm_buffer.name, selector_matrix_dim_shared.shape, selector_matrix_dim_shared.dtype),
                            (rv_prep_selector_indices_shm_buffer.name, rv_prep_selector_indices_shared.shape, rv_prep_selector_indices_shared.dtype),
                            (crv_shm_buffer.name, crv_shared.shape, crv_shared.dtype),
                            (crv_offsets_shm_buffer.name, crv_offsets_shared.shape, crv_offsets_shared.dtype),
                            (crv_sizes_shm_buffer.name, crv_sizes_shared.shape, crv_sizes_shared.dtype),
                            (sample_points_min_dim_shm_buffer.name, sample_points_min_dim_shared.shape, sample_points_min_dim_shared.dtype),
                            (sample_points_max_dim_shm_buffer.name, sample_points_max_dim_shared.shape, sample_points_max_dim_shared.dtype),
                            (Vhs_shm_buffer.name, Vhs_shared.shape, Vhs_shared.dtype),
                            (normals_shm_buffer.name, normals_shared.shape, normals_shared.dtype),
                            (offsets_shm_buffer.name, offsets_shared.shape, offsets_shared.dtype),
                            (d_s_shm_buffer.name, d_s_shared.shape, d_s_shared.dtype),
                            (combined_lower_dim_convex_hull_equations_shm_buffer.name, combined_lower_dim_convex_hull_equations_shared.shape, combined_lower_dim_convex_hull_equations_shared.dtype),
                            (cldchw_offs_shm_buffer.name, cldchw_offs_shared.shape, cldchw_offs_shared.dtype),
                            (cldchw_szs_shm_buffer.name, cldchw_szs_shared.shape, cldchw_szs_shared.dtype),
                            False,
                        ), len(label_tuples_args))), chunksize)

                    M[:] = M_shared[:]
                    R[:] = R_shared[:]

                    pf(f"finished computation for dim {dim+1}/{N}")

    return sample_points_min_max, ret



def build_directed_transition_cubes(selector_matrix: np.ndarray, ridge_vertices: np.ndarray, bandwidths: np.ndarray, clip_box: np.ndarray, 
                           step_size: np.ndarray = None, sample_points_min_max: tuple = None,
                           *, 
                           ridge_based_para_sense_preparations_result: tuple = None, 
                           n_proc: int = None, verbose: bool = False) -> list:
    """
    Build a directed transition matrix.
    For each dimension, builds a cube that holds the computed transitions between pairs of ridges. 
    Returns list of these cubes of length N (dimensionality).
    Each cube c_i is an np.ndarray of shape (L, L, n_samples_for_current_dim_i), where L is the number of unique labels.
    The entry c_i[a, b, c] represents the amount of transitions along dimension i from cells with label_idx a to cells with label_idx b at sample point index c.
    The number of ridges is also returned in the nr_i matrix of shape (L, L, n_samples_for_current_dim_i, 2).

    @n_proc: set to non-None to use multiprocessing; set to negative to use all cores, otherwise, uses the specified number.
        WARNING: using n_proc non-None requires the main-script to be guarded via, e.g., if __name__ == "__main__": (..)
            otherwise, recursive execution of code may happen. Also, the unguarded code in the main script will execute
            once for each process.

    @return (sample_points, sample_points_min, sample_points_mas), [(c_i, nr_i) for i in range(n_dim)]
    """
    def pf(*args):
        if verbose:
            print(*args)
        else:
            pass

    if n_proc is not None:
        pf(f"using multiprocessing...")
        return _build_directed_transition_cubes_mp(selector_matrix, ridge_vertices, bandwidths, clip_box, step_size, 
                                                   sample_points_min_max=sample_points_min_max, 
                                                   ridge_based_para_sense_preparations_result=ridge_based_para_sense_preparations_result, 
                                                   n_proc=n_proc, 
                                                   verbose=verbose)

    N = selector_matrix.shape[0]
    L = selector_matrix.shape[1]

    if sample_points_min_max is None:
        sample_points_min_max = compute_sample_points(clip_box=clip_box, bandwidths=bandwidths, step_size=step_size)

    ret = []

    for dim in range(N):
        n_samples_dim_i = sample_points_min_max[0][dim].shape[0]
        ret.append((np.zeros((L, L, n_samples_dim_i)), np.zeros((L, L, n_samples_dim_i, 2))))

    if ridge_based_para_sense_preparations_result is None:
        ridge_based_para_sense_preparations_result = ridge_based_para_sense_preparations(ridge_vertices=ridge_vertices,
                                                                                     bandwidths=bandwidths,
                                                                                     clip_box=clip_box,
                                                                                     step_size=step_size,
                                                                                     sample_points_min_max=sample_points_min_max,
                                                                                     verbose=verbose)
    
    _, sample_points_min, sample_points_max, okays, Vhs, normals, offsets, d_s, lower_dim_convex_hull_equations, volumes, rv_prep_selector_indices = \
        ridge_based_para_sense_preparations_result

    selector_matrix = selector_matrix[..., rv_prep_selector_indices]
    ridge_vertices = ridge_vertices[rv_prep_selector_indices]
    
    if N > 2:
        available_dims = np.arange(N)
        dims_accessor = np.array([
            available_dims != i for i in range(N)
        ])

    for dim in range(N):
        selector_matrix_dim = selector_matrix[dim]
        M, R = ret[dim]
        if N > 2:
            da = dims_accessor[dim]
        okays_dim = okays[dim]
        sample_points_min_dim = sample_points_min[dim]
        sample_points_max_dim = sample_points_max[dim]

        for label_idx_a in range(L):
            for label_idx_b in range(L):
                selector = selector_matrix_dim[label_idx_a, label_idx_b]

                # merge selector with selector from preparation
                selector_indices = np.argwhere(selector)

                # adjust okays
                okays_ = np.full(okays_dim.shape, False)
                # find indices that the selector_indices map to in the okays array
                _, _, okay_indices_ = np.intersect1d(selector_indices, rv_prep_selector_indices, assume_unique=True, return_indices=True)

                if len(okay_indices_) > 0:
                    okays_[:, okay_indices_] = okays_dim[:, okay_indices_]

                # _, sensitivities, n_ridges = ridge_based_para_sense(relevant_ridge_vertices, bandwidths=bandwidths,
                #                                                     clip_box=clip_box, step_size=step_size, return_n_ridges=True, 
                #                                                     sample_points_min_max=sample_points_min_max, verbose=verbose)
                if N == 1:
                    sensitivities, n_ridges = _compute_ridge_based_para_sense_1d(okays_, return_n_ridges=True)
                elif N == 2:
                    sensitivities, n_ridges = compute_ridge_based_transitions_2d_for_one_dim(dim=dim, 
                                                             ridge_vertices=ridge_vertices,
                                                             okays=okays_,
                                                             return_n_ridges=True,
                                                             sample_points_min_dim=sample_points_min_dim,
                                                             sample_points_max_dim=sample_points_max_dim)
                else:
                    sensitivities, n_ridges = compute_ridge_based_transitions_for_one_dimension_of_nd(dim=dim, 
                                                                        N=N,
                                                                        okays=okays_, 
                                                                        ridge_vertices=ridge_vertices,
                                                                        Vhs=Vhs,
                                                                        normals=normals,
                                                                        offsets=offsets,
                                                                        d_s=d_s,
                                                                        lower_dim_convex_hull_equations=lower_dim_convex_hull_equations,
                                                                        sample_points_min_dim=sample_points_min_dim,
                                                                        sample_points_max_dim=sample_points_max_dim,
                                                                        return_n_ridges=True,
                                                                        dim_accessor=da,
                                                                        # pf=pf
                                                                        )
                    
                M[label_idx_a, label_idx_b] = sensitivities
                R[label_idx_a, label_idx_b] = n_ridges
                
    return sample_points_min_max, ret


def compute_label_distribution(samples: np.ndarray, labels: np.ndarray, bandwidths: np.ndarray, 
                               clip_box: np.ndarray = None, step_size: np.ndarray = None,
                               clip_convex_hull: bool = False, vor: Voronoi = None, *, 
                               sample_points_min_max: tuple = None,
                               identify_infinity_regions_and_neighbors_results: tuple = None,
                               verbose: bool = False) -> tuple:
    """
    Returns the label distribution similar to the sensitivity for each dimension and sample idx.
    For each dimension i, the distribition matrix D_i has shape (n_samples_for_current_dim_i, L),
        where L is the number of unique labels.
    """
    def pf(*args):
        if verbose:
            print(*args)
        else:
            pass

    labels = np.asarray(labels, dtype=int)
    samples = np.asarray(samples)

    unique_labels, unique_labels_inverse = np.unique(labels, return_inverse=True)

    L = len(unique_labels)

    if clip_box is not None:
        clip_box = np.asarray(clip_box)
    else:
        clip_box = np.vstack([np.min(samples, axis=0), np.max(samples, axis=0)]).T

    assert len(samples) > 0 and len(labels) == len(samples)
    
    if len(samples.shape) == 1:
        samples = samples.reshape(-1, 1)
    N = samples.shape[-1]

    pf(f"compute voronoi for {len(samples)} samples in dimensionality {N}...")
    if vor is None:
        vor = Voronoi(samples)

    if identify_infinity_regions_and_neighbors_results is None:
        identify_infinity_regions_and_neighbors_results = identify_infinity_regions_and_neighbors(vor)

    [_, region_idx_point_idx_mapping, _, _, region_idx_ridge_indices_mapping] = identify_infinity_regions_and_neighbors_results

    pf(f"compute sample points...")
    if sample_points_min_max is None:
        sample_points_min_max = compute_sample_points(clip_box=clip_box, bandwidths=bandwidths, step_size=step_size)
    sample_points, sample_points_min, sample_points_max = sample_points_min_max

    sample_min_max_clip_halfspaces = []
    for dim_i in range(N):
        hs = []

        for sample_idx in range(len(sample_points[dim_i])):
            lower_hs = make_lower_halfspace(sample_points_min[dim_i][sample_idx], N, dim_i)
            upper_hs = -lower_hs
            upper_hs[-1] = -sample_points_max[dim_i][sample_idx]

            hs.append(np.stack([lower_hs, upper_hs]))

        sample_min_max_clip_halfspaces.append(np.stack(hs))

    clip_equations = construct_halfspace_definition_for_aabbox(clip_box)

    if clip_convex_hull:
        ch = ConvexHull(samples)
        clip_equations = np.vstack([clip_equations, ch.equations])

    ret = []

    for dim_i in range(N):
        ret.append(np.zeros((sample_points[dim_i].shape[0], L)))

    for region_index in range(len(vor.regions)):
        if region_index not in region_idx_point_idx_mapping:
            pf(f"skipping region with idx {region_index} as it does not map to a point")
            continue
        point_idx = region_idx_point_idx_mapping[region_index]
        label_idx = unique_labels_inverse[point_idx]

        rv, (_, _, r_fhs, r_ihs) = clip_voronoi_region(region_idx=region_index, vor=vor, clip_equations=clip_equations, 
                region_idx_point_idx_mapping=region_idx_point_idx_mapping, 
                region_idx_ridge_indices_mapping=region_idx_ridge_indices_mapping)

        if rv is None:
            pf(f"skipping region with idx {region_index} as clipping was not successful")
            continue

        rv_bbox = np.vstack([np.min(rv, axis=0), np.max(rv, axis=0)]).T

        for dim_i in range(N):
            D_i = ret[dim_i]
            samples_min_i, samples_max_i = sample_points_min[dim_i], sample_points_max[dim_i]

            hs = sample_min_max_clip_halfspaces[dim_i]

            for sample_idx in range(len(samples_min_i)):
                s_min = samples_min_i[sample_idx]
                s_max = samples_max_i[sample_idx]
                
                if rv_bbox[dim_i, 1] < s_min:
                    break
                
                if rv_bbox[dim_i, 0] > s_max:
                    continue

                lower_hs = hs[sample_idx, 0]
                upper_hs = hs[sample_idx, 1]

                # clip cell
                sce_ = [clip_equations, lower_hs, upper_hs, r_fhs, r_ihs]
                sample_clip_equations = np.vstack([sce__ for sce__ in sce_ if len(sce__) > 0])
                
                # interior point
                ip = get_interior_point_from_halfspaces(sample_clip_equations)[0]

                # new intersection points
                try:
                    new_rv = HalfspaceIntersection(sample_clip_equations, ip).intersections
                except Exception:
                    pf(f"failed to clip region with idx {region_index} at sample idx {sample_idx} for dimension {dim_i}")
                    continue

                # compute volume
                try:
                    new_ch = ConvexHull(new_rv)
                except Exception:
                    pf(f"failed to construct convex hull for region with idx {region_index} at sample idx {sample_idx} for dimension {dim_i}")
                    continue

                D_i[sample_idx, label_idx] += new_ch.volume


    return ret, sample_points, sample_points_min, sample_points_max


def normalize_label_distribution(distr_matrix_list: list) -> list:
    """
    Normalizes the distribution such that it is one at each sample point disregarding the original absolute values.
        The only exception is when no normalization is possible, such as the absolute values were in sum 0.
    """
    ret = []

    for dim in range(len(distr_matrix_list)):
        D = distr_matrix_list[dim]

        D_sum = np.sum(D, axis=1)

        selector = (D_sum != 0.)

        D_norm = np.zeros(D.shape)

        D_norm[selector] = D[selector] / D_sum[selector].reshape(-1, 1)

        ret.append(D_norm)

    return ret


def normalize_label_distribution_by_bandwidth(distr_matrix_list: list, bandwidths: np.ndarray) -> list:
    """
    Normalizes the distribution such that it is one at each sample point disregarding the original absolute values.
        The only exception is when no normalization is possible, such as the absolute values were in sum 0.
    """
    ret = []

    for dim in range(len(distr_matrix_list)):
        D = distr_matrix_list[dim]

        D_norm = np.zeros(D.shape)

        D_norm[:] = D[:] / bandwidths[dim]

        ret.append(D_norm)

    return ret


def normalize_transition_matrix_by_distribution(sense_matrix_list: list, distr_matrix_list: list) -> list:
    """
    Divides the transitions by the total distribution unless where the distribution is zero.
    """
    ret = []

    for dim in range(len(distr_matrix_list)):
        D = distr_matrix_list[dim]

        D_sum = np.sum(D, axis=1)

        selector = (D_sum != 0.)

        S_norm = sense_matrix_list[dim].copy()

        S_norm[:, :, selector] = S_norm[:, :, selector] / D_sum[selector]

        ret.append(S_norm)

    return ret


def normalize_transition_matrix_by_bandwidth(sense_matrix_list: list, bandwidths: np.ndarray) -> list:
    """
    Divides the transitions by the total distribution unless where the distribution is zero.
    """
    ret = []

    for dim in range(len(sense_matrix_list)):
        S_norm = sense_matrix_list[dim].copy()

        S_norm[:, :, :] = S_norm[:, :, :] / bandwidths[dim]

        ret.append(S_norm)

    return ret


def compute_in_out_agg_sens_matrix_list(sense_matrix_list: list) -> list:
    """
    Computes the in-out aggregated transition matrices for the given overall matrix list.
    For each dimension i, returns a matrix A_i of shape (L, 2, n_sample_points_dim_i),
        where L is the amount of unique labels, and A_i[l_i, 0, :] are the ingoing transitions for label l_i,
        and A_i[l_i, 1, :] are the outgoing transitions for label l_i.
    """
    N = len(sense_matrix_list)
    ret = []

    for dim in range(N):
        S_i = sense_matrix_list[dim]
        A_in = S_i.sum(axis=0)
        A_out = S_i.sum(axis=1)

        A = np.stack([A_in, A_out], axis=1)
        ret.append(A)

    return ret


def compute_global_param_transitions(trans_matrix_list: list) -> np.ndarray:
    """
    Computes the global parameter transitions.
    Returns a matrix of shape (N, ) that contains the summed transitions for each dimension.
    """
    N = len(trans_matrix_list)
    res = np.zeros((N, ))

    for dim in range(N):
        M_i = trans_matrix_list[dim]
        res[dim] = M_i.sum()

    return res


def compute_global_param_transitions_by_pairs(trans_matrix_list: list) -> np.ndarray:
    """
    Computes the global parameter transitions.
    Returns a matrix of shape (N, L, L) that contains the summed transitions for each dimension and label pair.
    """
    N = len(trans_matrix_list)
    L = trans_matrix_list[0].shape[0]
    res = np.zeros((N, L, L))

    for dim in range(N):
        M_i = trans_matrix_list[dim]
        res[dim] = M_i.sum(axis=2)

    return res


def compute_ridge_volume_2d(ridge_vertices: np.ndarray) -> np.ndarray:
    """
    Compute the ridge volumes for ridges in 2D.
        -> Basically returns their length.

    @ridge_vertices: array of shape (n_ridges, 2)
    """
    return np.linalg.norm(ridge_vertices[:, 0] - ridge_vertices[:, 1], axis=1)


def _compute_volume_from_equations(ch_equations: np.ndarray) -> tuple[float, bool]:
    try:
        # compute new interior point
        ip = get_interior_point_from_halfspaces(ch_equations)[0]
    except AssertionError:
        return 0.0, False
        
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            ldhi = HalfspaceIntersection(halfspaces=ch_equations, interior_point=ip)
        except QhullError:
            return 0.0, False
        except RuntimeWarning as re:
            if str(re).startswith("divide by zero encountered in divide"):
                return 0.0, False
            raise re
        
    ld_ips = ldhi.intersections

    try:
        ld_ch = ConvexHull(ld_ips)
    except QhullError:
        return 0.0, False
    
    return ld_ch.volume, True


def compute_ridge_volumes(ridge_vertices: list, *, compute_ridge_transformations_result: tuple = None) -> tuple[np.ndarray, np.ndarray]:
    """
    For a given list of ridge vertices, compute their volume.
    Returns a list of volumes (float) and a list of bools that describe whether the volume computation failed or not.
    """
    if len(ridge_vertices) == 0:
        return np.empty((0, ), dtype=float), np.empty((0, ), dtype=bool)
    
    N = ridge_vertices[0].shape[-1]

    if N == 1:
        return np.ones((len(ridge_vertices), )), np.full((len(ridge_vertices), ), True)
    
    # N > 2 case:
    volumes, okays = np.zeros((len(ridge_vertices), )), np.full((len(ridge_vertices),), False)

    if compute_ridge_transformations_result is None:
        compute_ridge_transformations_result = compute_ridge_transformations(ridge_vertices=ridge_vertices)
    _, _, _, _, _, volumes_, selector = compute_ridge_transformations_result

    for i, idx in enumerate(selector):
        volumes[idx], okays[idx] = volumes_[i], True
    
    return volumes, okays


@numba.njit()
def transform_plane_normals(normals: np.ndarray, b: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Rotates plane normals such that the dot product between them to b is alsways positive (or zero).
    @normals: array of shape (n_normals, N)
    @b: vector of shape (N, )
    @normalize: if True, normalizes all normals first.
    """
    if normalize:
        nl = nb_linalg_norm_1(normals).reshape(-1, 1)
        assert np.all(nl > 0)
        normals = normals / nl
    else:
        normals = normals.copy()

    for i in range(normals.shape[0]):
        if normals[i] @ b < 0:
            normals[i] = -normals[i]
    
    return normals


@numba.njit()
def average_plane_normals(normals: np.ndarray, b: np.ndarray = None, normalize: bool = True) -> np.ndarray:
    """
    Compute the normalized average plane normal.
    Note, that all normals are transformed such that the dot product with b (or np.ones(N)) is always positive.
    """
    if b is None:
        b = np.ones(normals.shape[-1])
        b = b / np.sqrt(normals.shape[-1])
    else:
        bl = np.linalg.norm(b)
        assert bl > 0
        b = b / bl

    normals = transform_plane_normals(normals=normals, b=b, normalize=normalize)

    avg_normal = nb_sum0(normals)
    avg_normal = avg_normal / np.linalg.norm(avg_normal)
    return avg_normal


@numba.njit()
def plane_angle_metric(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    """
    v1l = np.linalg.norm(v1)
    v2l = np.linalg.norm(v2)
    assert v1l > 0
    assert v2l > 0

    if v1l != 1.0:
        v1 = v1 / v1l

    if v2l != 1.0:
        v2 = v2 / v2l

    v = np.abs(v1 @ v2).item()
    if v < -1.0:
        v = -1.0
    elif v > 1.0:
        v = 1.0

    return np.arccos(v) / (np.pi * 0.5)


def aggregate_directed_transition_matrix(transition_matrix_list: list, n_ridges_matrix_list: list = None) -> tuple:
    """
    Aggregates the directed transititon matrix.
    The result should be the same as the original ridge-based transition computations.
    """
    
    S_i_s = [m.sum(axis=(0, 1)) for m in transition_matrix_list]

    if n_ridges_matrix_list is None:
        return (S_i_s, )

    R_i_s = [m.sum(axis=(0, 1)) for m in n_ridges_matrix_list]

    return S_i_s, R_i_s


def cluster_distance(cluster_data: np.ndarray, cluster_centroid: np.ndarray, distance_fn) -> float:
    return np.sum([distance_fn(cluster_data[i], cluster_centroid) for i in range(len(cluster_data))]) / len(cluster_data)


def compute_clustered_normals(ridge_vertices: list, n_clusters: int = 5, 
                              dim_labels: list = None,
                              kmeans_kwargs: dict = None,
                              ridge_based_para_sense_preparations_result: tuple = None,
                              compute_ridge_transformations_result: tuple = None):
    """
    Computes clusters on the normals of the provided ridge_vertices.
    Returns tuple.
    @return (pd.Dataframe with results, clustering_instance, sorted_normals, sorted_ridge_volumes, sorted_normals_transformed)
    """
    if compute_ridge_transformations_result is not None:
        crtr = compute_ridge_transformations_result
    elif ridge_based_para_sense_preparations_result is not None:
        crtr = ridge_based_para_sense_preparations_result[4:]
    else:
        crtr = compute_ridge_transformations(ridge_vertices=ridge_vertices)

    ridge_volumes, ok = compute_ridge_volumes(ridge_vertices, compute_ridge_transformations_result=crtr)
    normals = crtr[1]
    ridge_volumes = ridge_volumes[ok]

    sort_selector = np.argsort(ridge_volumes)

    normals_sorted = normals[sort_selector]
    rvv_sorted = ridge_volumes[sort_selector]

    # weighted_normals = normals_sorted * rvv_sorted.reshape(-1, 1)

    metric = distance_metric(type_metric.USER_DEFINED, func=plane_angle_metric)
    normals_sorted_tf = transform_plane_normals(normals=normals_sorted, b=np.ones(normals_sorted.shape[-1]), normalize=False)

    _kmeans_kwargs = dict(
        tolerance=1e-7
    )
    if kmeans_kwargs is not None:
        _kmeans_kwargs.update(kmeans_kwargs)

    kcs = KMeansClusteringAlgorithm(n_clusters=n_clusters, metric=metric, kmeans_kwargs=_kmeans_kwargs)

    clusters, centers = kcs.process(data=normals_sorted_tf)

    cluster_volumes = np.array([rvv_sorted[c].sum() for c in clusters])

    cluster_volumes_sort_selector = np.argsort(cluster_volumes)
    cluster_volumes_sorted = cluster_volumes[cluster_volumes_sort_selector]
    cluster_sorted_by_volume = make_object_array(clusters)[cluster_volumes_sort_selector]

    centers_sorted = np.stack(centers)[cluster_volumes_sort_selector]
    centers_sorted_normed = centers_sorted / np.linalg.norm(centers_sorted, axis=1).reshape(-1, 1)

    N = centers_sorted_normed.shape[1]

    if dim_labels is None:
        dim_labels = [f"x{dim}" for dim in range(N)]

    df_knn = pd.DataFrame(data=centers_sorted_normed, columns=dim_labels)
    df_knn["cluster_size"] = np.array([len(c) for c in cluster_sorted_by_volume])
    df_knn["volume"] = cluster_volumes_sorted
    df_knn["cluster_distance"] = [cluster_distance(normals_sorted_tf[cluster_sorted_by_volume[i]], np.array(centers_sorted[i]), metric) for i in range(len(clusters))]

    return df_knn, kcs, normals_sorted, rvv_sorted, normals_sorted_tf
