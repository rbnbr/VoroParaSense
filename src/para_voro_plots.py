import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy.spatial import Voronoi, voronoi_plot_2d, convex_hull_plot_2d
from scipy.spatial import ConvexHull
import pandas as pd
from src.para_voro_util import *
from src.general_util import *
from matplotlib import transforms
from matplotlib.colors import LinearSegmentedColormap


# src: https://spectrum.adobe.com/page/color-for-data-visualization/
def get_12_categorical_colors(hashtag: bool = True) -> list:
    colors = ["0fb5ae", "4046ca", "f68511", "de3d82", "7e84fa", "72e06a",
              "147af3", "7326d3", "e8c600", "cb5d00", "008f5d", "bce931"]

    if not hashtag:
        return colors.copy()
    
    return [f"#{c}" for c in colors]


def plot_simple_plot(x: list, y: list, title: str = None):
    plt.close()
    fig = plt.figure()
    ax = fig.gca()

    ax.plot(x, y)
    if title is not None and title != "":
        ax.set_title(title)

    plt.show()
    return fig


def plot_data(samples: np.ndarray, labels: np.ndarray, clip_box: np.ndarray = None, colors: np.ndarray = None,
              dim_names: list = None, legend_kwargs: dict = None, plot_legend: bool = False):
    if dim_names is None:
        dim_names = ["$x_1$", "$x_2$", "$x_3$"]

    # Plot the data in 3D
    if colors is None:
        colors = labels
    else:
        colors = colors[np.unique(labels, return_inverse=True)[1]]

    _legend_kwargs = {
        "loc": "lower left",
        "ncol": len(np.unique(labels))
    }
    if legend_kwargs is not None:
        _legend_kwargs.update(legend_kwargs)

    # create legend
    legend_patches = []
    for label_idx in range(len(np.unique(labels))):
        l = f"{label_idx}"
        legend_patches.append(mpatches.Patch(color=colors[label_idx], label=l))
    _legend_kwargs["handles"] = legend_patches

    n_d = samples.shape[-1]
    fig = plt.figure()
    if n_d == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=colors, cmap='winter', s=5)
        ax.set_xlabel(dim_names[0])
        ax.set_ylabel(dim_names[1])
        ax.set_zlabel(dim_names[2])
    elif n_d == 2:
        ax = fig.add_subplot(111)
        vor = Voronoi(samples)
        voronoi_plot_2d(vor, show_vertices=False, show_points=False, point_size=5, ax=ax)

        # Extract ridges (faces) that correspond to points with different labels
        face_points = vor.ridge_points

        # map face_points to labels
        face_labels = labels[face_points]

        dividing_ridges = np.array(vor.ridge_vertices)[face_labels[:, 0] != face_labels[:, 1]]

        # filter infinity ridges
        dividing_ridges = dividing_ridges[np.all(dividing_ridges >= 0, 1)]

        ax.scatter(samples[:, 0], samples[:, 1], color=colors, cmap='winter', s=5)

        for ridge in dividing_ridges:
            ridge_vertices = vor.vertices[ridge]
            ax.plot(ridge_vertices[:, 0], ridge_vertices[:, 1], color="blue", )

        ax.set_xlabel(dim_names[0])
        ax.set_ylabel(dim_names[1])
    elif n_d == 1:
        ax = fig.add_subplot(111)

        ax.scatter(samples.flatten(), np.zeros(len(samples)), c=colors)
        ax.set_xlabel(dim_names[0])
        ax.yaxis.set_visible(False)
    else:
        raise RuntimeError("cannot plot dimensionality: " + str(n_d))
    
    if clip_box is not None:
        ax.set_xlim(clip_box[0])
        if n_d > 1:
            ax.set_ylim(clip_box[1])
        if n_d > 2:
            ax.set_zlim(clip_box[2])

    # Show plot
    ax.set_aspect("equal")
    if plot_legend:
        fig.legend(**_legend_kwargs)
    plt.title(f'Labeled Samples in {n_d}D')
    plt.show()
    return fig


def plot_face_3d(points: np.ndarray, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    
    points = np.vstack([points, points[0]])
    ax.plot(points[:, 0], points[:, 1], points[:, 2])
    plt.show()
    return fig


def plot_ridges(ridge_vertices: list, samples: np.ndarray = None, labels: np.ndarray = None, clip_box: np.ndarray = None, clip_convex_hull: bool = False,
                dim_names: list = None, colors: list = None, legend_kwargs: dict = None, plot_legend: bool = False, f: float = 0.1, plot: bool = True, title: str = None):
    if dim_names is None:
        dim_names = ["$x_1$", "$x_2$", "$x_3$"]

    if labels is not None:
        if colors is None:
            colors = labels
            orig_colors = np.arange(np.unique(labels).shape[0])
        else:
            orig_colors = colors.copy()
            colors = colors[np.unique(labels, return_inverse=True)[1]]

    ridge_line_color = "blue"
    ridge_point_color = "gray"

    _legend_kwargs = {
        "loc": "lower left",
        "ncol": len(np.unique(labels))
    }
    if legend_kwargs is not None:
        _legend_kwargs.update(legend_kwargs)

    # create legend
    legend_patches = []
    for label_idx in range(len(np.unique(labels))):
        l = f"{label_idx}"
        legend_patches.append(Line2D([], [], color=orig_colors[label_idx], label=l, marker='o', linewidth=0))
    legend_patches.append(Line2D([], [], color="gray", label="ridge point", marker='o', linewidth=0))
    _legend_kwargs["handles"] = legend_patches

    if len(ridge_vertices) == 0:
        if samples is None:
            return
        else:
            n_d = samples.shape[-1]
    else:
        n_d = ridge_vertices[0].shape[-1]

    fig = plt.figure()
    if n_d == 3:
        ax = fig.add_subplot(111, projection='3d')

        for rv in ridge_vertices:
            ax.scatter(rv[:, 0], rv[:, 1], rv[:, 2], c=ridge_point_color, s=5)

            rv_ = np.vstack([rv, rv[0]])
            ax.plot(rv_[:, 0], rv_[:, 1], rv_[:, 2], c=ridge_line_color)

        if clip_box is not None:
            bbox_points_lower = np.array([
                clip_box[:, 0], [clip_box[0, 0], clip_box[1, 1], clip_box[2, 0]], [clip_box[0, 1], clip_box[1, 1], clip_box[2, 0]], [clip_box[0, 1], clip_box[1, 0], clip_box[2, 0]]
            ])

            bbox_points_upper = np.array([
                [clip_box[0, 0], clip_box[1, 0], clip_box[2, 1]], [clip_box[0, 0], clip_box[1, 1], clip_box[2, 1]], [clip_box[0, 1], clip_box[1, 1], clip_box[2, 1]], [clip_box[0, 1], clip_box[1, 0], clip_box[2, 1]]
            ])

            print(bbox_points_lower)

            bboxp_lower = np.vstack([bbox_points_lower, bbox_points_lower[0]])
            bboxp_upper = np.vstack([bbox_points_upper, bbox_points_upper[0]])
            
            ax.plot(bboxp_lower[:, 0], bboxp_lower[:, 1], bboxp_lower[:, 2], c="red")
            ax.plot(bboxp_upper[:, 0], bboxp_upper[:, 1], bboxp_upper[:, 2], c="red")

            for i in range(len(bbox_points_lower)):
                ax.plot([bbox_points_lower[i][0], bbox_points_upper[i][0]], [bbox_points_lower[i][1], bbox_points_upper[i][1]], [bbox_points_lower[i][2], bbox_points_upper[i][2]], c="red")
        
        ax.set_xlabel(dim_names[0])
        ax.set_ylabel(dim_names[1])
        ax.set_zlabel(dim_names[2])
    elif n_d == 2:
        ax = fig.add_subplot(111)
        
        if samples is not None:
            vor = Voronoi(samples)
            voronoi_plot_2d(vor, show_vertices=False, show_points=labels is None, ax=ax)

            if clip_convex_hull:
                ch = ConvexHull(samples)
                # convex_hull_plot_2d(ch, ax=ax)
                chv = ch.points[ch.vertices]
                chv = np.vstack([chv, chv[:1]])
                for i in range(len(ch.vertices)):
                    pp_ = chv[i:i+2]
                    ax.plot(pp_[:, 0], pp_[:, 1], c="red")

            if labels is not None:
                ax.scatter(samples[:, 0], samples[:, 1], c=colors, cmap='winter', s=2**6)

        for rv in ridge_vertices:
            ax.scatter(rv[:, 0], rv[:, 1], color=ridge_point_color)
            ax.plot(rv[:, 0], rv[:, 1], color=ridge_line_color)

        ax.set_xlabel(dim_names[0])
        ax.set_ylabel(dim_names[1])

        if clip_box is not None:
            bbox_points = np.array([
                clip_box[:, 0], [clip_box[0, 0], clip_box[1, 1]], clip_box[:, 1], [clip_box[0, 1], clip_box[1, 0]]
            ])
            bboxp = np.vstack([bbox_points, bbox_points[0]])
            ax.plot(bboxp[:, 0], bboxp[:, 1], c="red")
    elif n_d == 1:
        ax = fig.add_subplot(111)
        ax.scatter(np.array(ridge_vertices).flatten(), np.zeros(len(ridge_vertices)), c=ridge_point_color)

        if labels is not None:
            ax.scatter(samples.flatten(), np.zeros(len(samples)), c=colors)

        ax.set_xlabel(dim_names[0])
        ax.yaxis.set_visible(False)
    else:
        raise RuntimeError("cannot plot dimensionality: " + str(n_d))
    
    if clip_box is not None:
        offset = f * (clip_box[:, 1] - clip_box[:, 0])
        offset = np.vstack([-offset, offset]).T
        ax.set_xlim(clip_box[0] + offset[0])
        if n_d > 1:
            ax.set_ylim(clip_box[1] + offset[1])
        if n_d > 2:
            ax.set_zlim(clip_box[2] + offset[2])

    # Show plot
    if plot_legend:
        fig.legend(**_legend_kwargs)
    ax.set_aspect("equal")
    fig.tight_layout()
    if title is None:
        plt.title(f'Clipped Ridges in {n_d}D')
    else:
        plt.title(title)

    if plot:
        plt.show()
    return fig


def plot_sensitivities(sample_points: list, sensitivities: list, 
                       *, n_ridges: list = None, bandwidths: np.ndarray = None, dim_labels: list = None,
                       discrete_dimension_ticks_and_labels: dict = None, discrete_dimension_ticks_and_labels_kwargs: dict = None,
                       horizontal_plot: bool = False,
                       share_y_axis: bool = False,
                       fig_kwargs: dict = None,
                       title: str = "Transition and Number of Ridges per Dimension"):
    """
    @discrete_dimension_ticks_and_labels: A dictionary that maps dimension index to a tuple of (ticks, ticklabels) which will be set for that dimension like
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)

        e.g., discrete_dimension_ticks_and_labels = { 0: ([0.0, 0.5, 1.0], ["Appel", "Orange", "Tea"]) }
    @horizontal_plot: If True, plots axes side by side horizontally, else, vertically.
    @share_y_axis: If True, uses shared y axis.
    @fig_kwargs: Arguments for creating the figure such as figsize and dpi etc.
    @title: title for whole figure. not set if empty or None
    """
    plt.close()

    if discrete_dimension_ticks_and_labels is None:
        discrete_dimension_ticks_and_labels = dict()
    if discrete_dimension_ticks_and_labels_kwargs is None:
        discrete_dimension_ticks_and_labels_kwargs = dict()

    N = len(sample_points)

    figsize = (6, N * 3)
    if horizontal_plot:
        figsize = (figsize[1], figsize[0])
    
    _fig_kwargs = dict(figsize=figsize, dpi=300)
    if fig_kwargs is not None:
        _fig_kwargs.update(fig_kwargs)

    fig = plt.figure(**_fig_kwargs)

    axes = []

    for dim in range(N):
        add_subplot_kwargs = dict()
        if share_y_axis and dim > 0:
            add_subplot_kwargs["sharey"] = axes[0]

        if horizontal_plot:
            ax = fig.add_subplot(1, N, dim+1, **add_subplot_kwargs)
        else:
            ax = fig.add_subplot(N, 1, dim+1, **add_subplot_kwargs)

        axes.append(ax)

        if N == 1:
            sens_values = sensitivities
        else:
            sens_values = sensitivities[dim]

        ax.plot(sample_points[dim], sens_values, c="blue", zorder=9)
        ax.scatter(sample_points[dim], sens_values, s=2**3, c="blue", zorder=10)

        ax.set_xlabel(dim_labels[dim] if dim_labels is not None else f"x{dim+1}")
        ax.set_ylabel("transitions per bin", color="blue")

        # plot bandwidths
        if bandwidths is not None:
            bw = bandwidths[dim]
            x = sample_points[dim][sample_points[dim].shape[0] // 2]
            p = Rectangle((x - bw, 0.0), 2 * bw, sens_values.max())
            ax.add_patch(p)

        # ax.set_ylim(0.0, sens_values.max() * 1.1)

        if n_ridges is not None:
            ax2 = ax.twinx()
            if N == 1:
                n_ridges_values = n_ridges
            else:
                n_ridges_values = n_ridges[dim]

            ax2.set_ylabel('#ridges', color='orange')

            ax2.plot(sample_points[dim], n_ridges_values[:, 0], c="orange", zorder=7)
            ax2.scatter(sample_points[dim], n_ridges_values[:, 0], s=2**3, c="orange", zorder=8)

            ax2.plot(sample_points[dim], n_ridges_values[:, 1], c="gray", zorder=5)
            ax2.scatter(sample_points[dim], n_ridges_values[:, 1], s=2**3, c="gray", zorder=6)
            # ax2.set_ylim(0.0, n_ridges_values.max() * 1.1)

            ax.set_zorder(ax2.get_zorder()+1)

        if dim in discrete_dimension_ticks_and_labels:
            x_ticks, x_ticklabels = discrete_dimension_ticks_and_labels[dim]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels, **discrete_dimension_ticks_and_labels_kwargs)
            
    if title is not None or title != "":
        fig.suptitle(title)

    fig.tight_layout()

    plt.show()
    return fig


def plot_sens_matrix_list(sample_points: list, sens_matrix_list: list, *, 
                          stack_plot: bool = True, 
                          stack_plot_kwargs: dict = None, 
                          stack_plot_gradient: bool = False,
                          stack_plot_outline: bool = False,
                          stackplot_fill: int = 0,
                          stackplot_color_hatch: bool = False,
                          stackplot_hatch_linewidth: int = 1,
                          stackplot_divider: bool = False,
                          horizontal_plot: bool = False,
                          n_ridges_matrix_list: list = None, dim_labels: list = None, line_labels: list = None,
                          do_pairwise_labels: bool = False,
                          discrete_dimension_ticks_and_labels: dict = None,
                          discrete_dimension_ticks_and_labels_kwargs: dict = None,
                          legend_kwargs: dict = None,
                          no_legend: bool = False,
                          share_y_axis: bool = False,
                          fig_kwargs: dict = None,
                          title: str = "Pairwise transitions between labels"):
    """
    @discrete_dimension_ticks_and_labels: A dictionary that maps dimension index to a tuple of (ticks, ticklabels) which will be set for that dimension like
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)

        e.g., discrete_dimension_ticks_and_labels = { 0: ([0.0, 0.5, 1.0], ["Appel", "Orange", "Tea"]) }
    @stack_plot_gradient: whether to use the stackplot gradient
    @stackplot_fill: if > 0, will add artificial stackplot regions with a fixed height to add spacings to the stackplots
    @stackplot_color_hatch: fills the plot with colored hatches. The hatch shape is taken from the stack_plot_kwargs["hatch"] arg. If not given, it is "+". 
        The hatch color describes the origin label, and the fill the destination label.
    @do_pairwise_labels: if True, each axis gets its own pairwose labels, otherwise, create a legend for each class label.
    @legend_kwargs: kwargs given either to ax.legend or plt.legend call.
    @share_y_axis: If True, uses shared y axis.
    @fig_kwargs: Arguments for creating the figure such as figsize and dpi etc.
    @stackplot_divider: If True, draws a thin black line between each stackplot layer.
    @stackplot_hatch_linewidth: Sets the linewidth of a hatch, temporarily overwrites the plt.rcParams!
    @title: title for whole figure. not set if empty or None
    """
    plt.close()

    if discrete_dimension_ticks_and_labels is None:
        discrete_dimension_ticks_and_labels = dict()
    if discrete_dimension_ticks_and_labels_kwargs is None:
        discrete_dimension_ticks_and_labels_kwargs = dict()

    if stack_plot_kwargs is None:
        stack_plot_kwargs = dict()
    else:
        stack_plot_kwargs = stack_plot_kwargs.copy()

    old_hatch_width = plt.rcParams["hatch.linewidth"]
    plt.rcParams["hatch.linewidth"] = stackplot_hatch_linewidth

    L = len(sens_matrix_list[0])
    N = len(sample_points)

    _legend_kwargs = {
        "loc": "lower left",
        "ncol": N
    }
    if legend_kwargs is not None:
        _legend_kwargs.update(legend_kwargs)

    figsize = (3, N * 3)
    if horizontal_plot:
        figsize = (figsize[1], figsize[0])
    
    _fig_kwargs = dict(figsize=figsize, dpi=300)

    if fig_kwargs is not None:
        _fig_kwargs.update(fig_kwargs)

    fig = plt.figure(**_fig_kwargs)

    if "colors" not in stack_plot_kwargs:
        colors = plt.color_sequences.get("tab10")[:L]
    else:
        colors = stack_plot_kwargs["colors"][:L]

    hatch = "+"
    if "hatch" in stack_plot_kwargs:
        hatch = stack_plot_kwargs["hatch"]
        if stackplot_color_hatch:
            del stack_plot_kwargs["hatch"]  # to avoid duplicated hatch plotting

    # create legend
    if not do_pairwise_labels:
        legend_patches = []
        for label_idx in range(L):
            l = f"{label_idx}"
            if line_labels is not None:
                l = line_labels[label_idx]
            legend_patches.append(mpatches.Patch(color=colors[label_idx], label=l))

        _legend_kwargs["handles"] = legend_patches

    axes = []

    for dim in range(N):
        add_subplot_kwargs = dict()
        if share_y_axis and dim > 0:
            add_subplot_kwargs["sharey"] = axes[0]

        if horizontal_plot:
            ax = fig.add_subplot(1, N, dim+1, **add_subplot_kwargs)
        else:
            ax = fig.add_subplot(N, 1, dim+1, **add_subplot_kwargs)
        
        if share_y_axis and dim > 0:
            ax.tick_params(labelleft=False)

        axes.append(ax)

        sens_matrix = sens_matrix_list[dim]

        labels = []
        x_values = sample_points[dim]
        y_values = []

        cmaps = []
        poly_fill_color = []
        poly_outline_color = []

        sens_values_max = sens_matrix.sum(axis=(0, 1)).max()

        for label_idx_a in range(sens_matrix.shape[0]):
            for label_idx_b in range(sens_matrix.shape[1]):
                sens_values = sens_matrix[label_idx_a, label_idx_b]

                if sens_values.sum() == 0:
                    continue

                if line_labels is None:
                    label = f"{label_idx_a} -> {label_idx_b}"
                else:
                    label = f"{line_labels[label_idx_a]} -> {line_labels[label_idx_b]}"
                
                labels.append(label)
                y_values.append(sens_values)

                cmaps.append(LinearSegmentedColormap.from_list(label, [colors[label_idx_a], colors[label_idx_b]] * 2, 4))
                poly_fill_color.append(colors[label_idx_b])
                poly_outline_color.append(colors[label_idx_a])

                if stackplot_fill > 0:
                    y_values.append(np.full((len(x_values), ), stackplot_fill * sens_values_max / 100.0))
                    cmaps.append(LinearSegmentedColormap.from_list(label, ["white", "white"], 2))
                    poly_fill_color.append("white")
                    poly_outline_color.append("white")
        
        if stackplot_fill > 0:
            # remove last again
            y_values = y_values[:-1]
            cmaps = cmaps[:-1]
            poly_fill_color = poly_fill_color[:-1]
            poly_outline_color = poly_outline_color[:-1]

        stack_plot_kwargs["colors"] = poly_fill_color

        if len(y_values) > 0:           
            if not stack_plot:
                for label_, y_values_ in zip(labels, y_values):
                    if not do_pairwise_labels:
                        ax.plot(x_values, y_values_)
                    else:
                        ax.plot(x_values, y_values_, label=label_)
            else:
                if do_pairwise_labels:
                    stack_plot_kwargs["labels"] = labels
                p = ax.stackplot(x_values, *y_values, **stack_plot_kwargs)

                old_xlim, old_ylim = ax.get_xlim(), ax.get_ylim()
                
                if stack_plot_gradient:
                    for i, polygon in enumerate(p):
                        verts = np.vstack([p.vertices for p in polygon.get_paths()])
                        cmap = cmaps[i]
                        gradient = ax.imshow(np.linspace(0, 1, 100).reshape(1, -1), cmap=cmap, aspect='auto', 
                                              extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()], 
                                              # transform=transforms.Affine2D().rotate_deg(((i % 2) * 2 - 1) * 45) + ax.transData
                                              )
                        gradient.set_clip_path(polygon.get_paths()[0], transform=ax.transData)
                        # ax.plot(verts[:, 0], verts[:, 1], c="black")
                        polygon.remove()
                    # restore limits as the ax.imshow changes it
                    ax.set_xlim(old_xlim)
                    ax.set_ylim(old_ylim)

                if stack_plot_outline:
                    upper_offset = transforms.ScaledTranslation(0.0, 1/72., fig.dpi_scale_trans)  # the upper line should be a littler lower
                    lower_offset = transforms.ScaledTranslation(0.0, -1/72., fig.dpi_scale_trans)  # the lower line should be a littler higher
                    upper_offset_transform = ax.transData + upper_offset
                    lower_offset_transform = ax.transData + lower_offset
                    for polygon_idx in range(len(p)):
                        if stackplot_fill > 0:
                            # don't draw stackplot_fill polygon outlines
                            if polygon_idx % 2 == 1:
                                continue

                        polygon = p[polygon_idx]
                        
                        assert len(polygon.get_paths()) == 1, "if not 1, then not sure if splitting the path into upper and lower line still works correctly"
                        
                        verts = np.vstack([p.vertices for p in polygon.get_paths()])

                        assert len(verts) % 2 == 1, "if not uneven, not sure whether path splitting still works correctly"

                        # split verts into upper and lower enclosing line
                        mid_floor = len(verts) // 2
                        mid_ceil = mid_floor + 1
                        verts_lower, verts_upper = verts[:mid_ceil], verts[mid_floor:]

                        # plot black divider
                        # ax.plot(verts[:, 0], verts[:, 1], c="black", linewidth=1)

                        # plot lower and upper slightly offsetted
                        # check out where difference is > 1%
                        
                        # diff = np.abs(verts_upper[:, 1] - verts_lower[:, 1])
                        # diff_w = diff > (2.0 * sens_values_max / 100.0)
                        # occtf = get_continuous_occurrences_of_true_and_false(diff_w)
                        # idx = 0
                        # for i in range(len(occtf)):
                        #     v = occtf[i]                            
                        #     rm = 0
                        #     if v < 2:
                        #         rm = 1
                        #         idx = idx - 1  # include one from the left
                        #         v = v + 2  # include one from the right
                        #     # print(idx, v, i % 2 == 0)
                        #     if i % 2 == 0:
                        #         # case True
                        #         ax.plot(verts_lower[idx:idx+v, 0], verts_lower[idx:idx+v, 1], c=poly_outline_color[polygon_idx], linewidth=1, transform=upper_offset_transform)
                        #         ax.plot(verts_upper[idx:idx+v, 0], verts_upper[idx:idx+v, 1], c=poly_outline_color[polygon_idx], linewidth=1, transform=lower_offset_transform)
                        #     else:
                        #         # case False
                        #         ax.plot(verts_lower[idx:idx+v, 0], verts_lower[idx:idx+v, 1], c=poly_outline_color[polygon_idx], linewidth=1)
                        #         ax.plot(verts_upper[idx:idx+v, 0], verts_upper[idx:idx+v, 1], c=poly_outline_color[polygon_idx], linewidth=1)
                        #     idx += v - rm

                        ax.plot(verts_lower[:, 0], verts_lower[:, 1], c=poly_outline_color[polygon_idx], linewidth=1, transform=upper_offset_transform)
                        ax.plot(verts_upper[:, 0], verts_upper[:, 1], c=poly_outline_color[polygon_idx], linewidth=1, transform=lower_offset_transform)

                if stackplot_color_hatch:
                    for polygon_idx, polygon in enumerate(p):
                        if stackplot_fill > 0:
                            # don't draw stackplot_fill polygon outlines
                            if polygon_idx % 2 == 1:
                                continue

                        assert len(polygon.get_paths()) == 1, "if not 1, then not sure if splitting the path into upper and lower line still works correctly"
                        verts = np.vstack([p.vertices for p in polygon.get_paths()])
                        assert len(verts) % 2 == 1, "if not uneven, not sure whether path splitting still works correctly"

                        # split verts into upper and lower enclosing line
                        mid_floor = len(verts) // 2
                        mid_ceil = mid_floor + 1
                        verts_lower, verts_upper = verts[:mid_ceil], verts[mid_floor:]

                        # switch verts_upper
                        verts_upper = verts_upper[::-1]

                        assert np.all(verts_lower[:, 0] == verts_upper[:, 0])

                        # print(verts_lower, verts_upper)
                        # print("-")

                        ax.fill_between(verts_upper[:, 0], verts_lower[:, 1], verts_upper[:, 1], 
                                        hatch=hatch, facecolor="none", linewidth=0, edgecolor=poly_outline_color[polygon_idx])
                
                if stackplot_divider:
                    for polygon in p:
                        verts = np.vstack([p.vertices for p in polygon.get_paths()])
                        ax.plot(verts[:, 0], verts[:, 1], c="black", linewidth=0.5)

        if dim in discrete_dimension_ticks_and_labels:
            x_ticks, x_ticklabels = discrete_dimension_ticks_and_labels[dim]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels, **discrete_dimension_ticks_and_labels_kwargs)

        ax.set_title(dim_labels[dim] if dim_labels is not None else f"x{dim+1}")
        ax.set_ylabel("transitions per bin", color="black")

        if do_pairwise_labels and not no_legend:
            plt.legend(**_legend_kwargs)

        # if not stack_plot:
        #     ax.set_ylim(0.0, sens_matrix.max() * 1.1)
        # else:
        #     ax.set_ylim(0.0, sens_matrix.sum(axis=(0, 1)).max() * 1.1)

    if not do_pairwise_labels and not no_legend:
        fig.legend(**_legend_kwargs)

    if title is not None or title != "":
        fig.suptitle(title)

    fig.tight_layout()
    plt.show()

    plt.rcParams["hatch.linewidth"] = old_hatch_width
    return fig


def plot_scatter_matrix(df: pd.DataFrame, colors: np.ndarray, legend_kwargs: dict = None, scatter_matrix_kwarg: dict = None):
    """
    @df: Dataframe with values as the first n-1 columns and labels as the last column.
    @legend_kwargs: kwargs given either to ax.legend or plt.legend call.
    """
    plt.close()
    _legend_kwargs = {
        "loc": "upper center",
        "ncol": df.shape[1] - 1
    }
    if legend_kwargs is not None:
        _legend_kwargs.update(legend_kwargs)

    _scatter_matrix_kwarg = dict(
        diagonal="hist",
        alpha=1.0
    )

    if scatter_matrix_kwarg is not None:
        _scatter_matrix_kwarg.update(scatter_matrix_kwarg)

    uql, uql_inv = np.unique(df.values[:, -1], return_inverse=True)

    # create legend
    _legend_kwargs["handles"] = [
        mpatches.Patch(color=colors[i], label=uql[i]) for i in range(len(uql))
    ]

    axes = pd.plotting.scatter_matrix(df.iloc[:, :-1], color=colors[uql_inv], **_scatter_matrix_kwarg)

    fig = axes[0, 0].get_figure()
    fig.legend(**_legend_kwargs)

    # fig.savefig("iris_splom.pdf")

    return fig


def plot_parallel_coordinates(df: pd.DataFrame, colors: np.ndarray):
    """
    @df: Dataframe with values as the first n-1 columns and labels as the last column.
    @legend_kwargs: kwargs given either to ax.legend or plt.legend call.
    """
    plt.close()
    uql, uql_inv = np.unique(df.values[:, -1], return_inverse=True)
    ax = pd.plotting.parallel_coordinates(df, class_column=df.columns[-1], color=colors[:len(uql)])

    return ax.get_figure()


def plot_label_distribution_matrix_list(sample_points: list, distr_matrix_list: list, 
                                        *, stack_plot: bool = True, horizontal_layout: bool = False,
                                        stack_plot_kwargs: dict = None,
                                        stackplot_divider: bool = False,
                                        dim_labels: list = None, line_labels: list = None,
                                        discrete_dimension_ticks_and_labels: dict = None,
                                        discrete_dimension_ticks_and_labels_kwargs: dict = None,
                                        legend_kwargs: dict = None,
                                        fig_kwargs: dict = None,
                                        share_y_axis: bool = False,
                                        plot_legend: bool = True,
                                        title: str = "Label distribution per dimension"):
    """
    @discrete_dimension_ticks_and_labels: A dictionary that maps dimension index to a tuple of (ticks, ticklabels) which will be set for that dimension like
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)

        e.g., discrete_dimension_ticks_and_labels = { 0: ([0.0, 0.5, 1.0], ["Appel", "Orange", "Tea"]) }
    @legend_kwargs: kwargs given either to ax.legend or plt.legend call.
    @fig_kwargs: Arguments for creating the figure such as figsize and dpi etc.
    @share_y_axis: If True, uses shared y axis.
    @title: title for whole figure. not set if empty or None
    """
    plt.close()

    if discrete_dimension_ticks_and_labels is None:
        discrete_dimension_ticks_and_labels = dict()
    if discrete_dimension_ticks_and_labels_kwargs is None:
        discrete_dimension_ticks_and_labels_kwargs = dict()

    if stack_plot_kwargs is None:
        stack_plot_kwargs = dict()
    else:
        stack_plot_kwargs = stack_plot_kwargs.copy()

    N = len(sample_points)
    L = distr_matrix_list[0].shape[1]

    if "colors" not in stack_plot_kwargs:
        colors = plt.color_sequences.get("tab10")[:L]
    else:
        colors = stack_plot_kwargs["colors"][:L]
    stack_plot_kwargs["colors"] = colors

    _legend_kwargs = {
        "loc": "lower left",
        "ncol": N
    }
    if legend_kwargs is not None:
        _legend_kwargs.update(legend_kwargs)
    
    figsize = (6, N * 3)
    if horizontal_layout:
        figsize = (figsize[1], figsize[0])
    
    _fig_kwargs = dict(figsize=figsize, dpi=300)
    if fig_kwargs is not None:
        _fig_kwargs.update(fig_kwargs)

    fig = plt.figure(**_fig_kwargs)

    # create legend
    legend_patches = []
    for label_idx in range(L):
        l = f"{label_idx}"
        if line_labels is not None:
            l = line_labels[label_idx]
        legend_patches.append(mpatches.Patch(color=colors[label_idx], label=l))
    _legend_kwargs["handles"] = legend_patches

    axes = []

    for dim in range(N):
        add_subplot_kwargs = dict()
        if share_y_axis and dim > 0:
            add_subplot_kwargs["sharey"] = axes[0]

        if horizontal_layout:
            ax = fig.add_subplot(1, N, dim+1, **add_subplot_kwargs)
        else:
            ax = fig.add_subplot(N, 1, dim+1, **add_subplot_kwargs)

        if share_y_axis and dim > 0:
            ax.tick_params(labelleft=False)

        axes.append(ax)

        label_dist_matrix = distr_matrix_list[dim]

        if not stack_plot:
            for label_idx in range(label_dist_matrix.shape[1]):
                label_dist_values = label_dist_matrix[:, label_idx]
                ax.plot(sample_points[dim], label_dist_values, c=colors[label_idx])
                
        else:
            p = ax.stackplot(sample_points[dim], label_dist_matrix.T, **stack_plot_kwargs)

            if stackplot_divider:
                for polygon in p:
                    verts = np.vstack([p.vertices for p in polygon.get_paths()])
                    ax.plot(verts[:, 0], verts[:, 1], c="black", linewidth=0.5)

        if dim in discrete_dimension_ticks_and_labels:
            x_ticks, x_ticklabels = discrete_dimension_ticks_and_labels[dim]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels, **discrete_dimension_ticks_and_labels_kwargs)
        
        ax.set_title(dim_labels[dim] if dim_labels is not None else f"x{dim+1}")
        ax.set_ylabel("volumes per bin", color="black")

        # if not stack_plot:
        #     ax.set_ylim(0.0, sens_matrix.max() * 1.1)
        # else:
        #     ax.set_ylim(0.0, sens_matrix.sum(axis=(0, 1)).max() * 1.1)

        # ax.set_ylim(0.0, label_dist_matrix.max() * 1.1)

    if len(legend_patches) > 0 and plot_legend:
        fig.legend(**_legend_kwargs)

    if title is not None or title != "":
        fig.suptitle(title)

    fig.tight_layout()
    plt.show()
    return fig


def plot_aggregated_input_output_transitions(sample_points: list, in_out_agg_sens_matrix_list: list, *, 
                          stack_plot: bool = True, 
                          stack_plot_kwargs: dict = None,
                          stackplot_divider: bool = False,
                          n_ridges_matrix_list: list = None, dim_labels: list = None, 
                          line_labels: list = None,
                          discrete_dimension_ticks_and_labels: dict = None,
                          discrete_dimension_ticks_and_labels_kwargs: dict = None,
                          legend_kwargs: dict = None,
                          fig_kwargs: dict = None,
                          share_y_axis: bool = False,
                          title: str = "Aggregated input and output transitions per dimension"):
    """
    @discrete_dimension_ticks_and_labels: A dictionary that maps dimension index to a tuple of (ticks, ticklabels) which will be set for that dimension like
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)

        e.g., discrete_dimension_ticks_and_labels = { 0: ([0.0, 0.5, 1.0], ["Appel", "Orange", "Tea"]) }
    """
    plt.close()

    if discrete_dimension_ticks_and_labels is None:
        discrete_dimension_ticks_and_labels = dict()
    if discrete_dimension_ticks_and_labels_kwargs is None:
        discrete_dimension_ticks_and_labels_kwargs = dict()

    if stack_plot_kwargs is None:
        stack_plot_kwargs = dict()
    else:
        stack_plot_kwargs = stack_plot_kwargs.copy()

    N = len(sample_points)
    L = in_out_agg_sens_matrix_list[0].shape[0]

    if "colors" not in stack_plot_kwargs:
        colors = plt.color_sequences.get("tab10")[:L]
    else:
        colors = stack_plot_kwargs["colors"][:L]
    stack_plot_kwargs["colors"] = colors
    
    _legend_kwargs = {
        "loc": "lower left",
        "ncol": N
    }
    if legend_kwargs is not None:
        _legend_kwargs.update(legend_kwargs)

    # create legend
    legend_patches = []
    for label_idx in range(L):
        l = f"{label_idx}"
        if line_labels is not None:
            l = line_labels[label_idx]
        legend_patches.append(mpatches.Patch(color=colors[label_idx], label=l))
    _legend_kwargs["handles"] = legend_patches

    _fig_kwargs = dict(figsize=(N * 3, 6), dpi=300, sharex="col")
    if share_y_axis:
        _fig_kwargs["sharey"] = "all"

    if fig_kwargs is not None:
        _fig_kwargs.update(fig_kwargs)

    fig, axes = plt.subplots(2, N, **_fig_kwargs)

    for dim in range(N):
        ax_in, ax_out = axes[:, dim]

        in_out_agg_sens_matrix = in_out_agg_sens_matrix_list[dim]

        x_values = sample_points[dim]
        y_values = []
        y_colors = []

        for label_idx in range(L):
            sens_values = in_out_agg_sens_matrix[label_idx]
            y_values.append(sens_values)
            y_colors.append(colors[label_idx])
                
        if not stack_plot:
            for y_color, (sens_in, sens_out) in zip(y_colors, y_values):
                ax_in.plot(x_values, sens_in, c=y_color)
                ax_out.plot(x_values, sens_out, c=y_color)
        else:
            p_in = ax_in.stackplot(x_values, *[y_v_[0] for y_v_ in y_values], **stack_plot_kwargs)
            p_out = ax_out.stackplot(x_values, *[y_v_[1] for y_v_ in y_values], **stack_plot_kwargs)

            if stackplot_divider:
                for ax, p in zip([ax_in, ax_out], [p_in, p_out]):
                    for polygon in p:
                        verts = np.vstack([p.vertices for p in polygon.get_paths()])
                        ax.plot(verts[:, 0], verts[:, 1], c="black", linewidth=0.5)
        
        y_axis_labels = ["in-transitions per bin", "out-transitions per bin"]

        for y_axis_label, ax in zip(y_axis_labels, [ax_in, ax_out]):
            ax.set_title(dim_labels[dim] if dim_labels is not None else f"x{dim+1}")
            ax.set_ylabel(y_axis_label, color="black")

            if dim in discrete_dimension_ticks_and_labels:
                x_ticks, x_ticklabels = discrete_dimension_ticks_and_labels[dim]
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_ticklabels, **discrete_dimension_ticks_and_labels_kwargs)

        # if not stack_plot:
        #     ax.set_ylim(0.0, sens_matrix.max() * 1.1)
        # else:
        #     ax.set_ylim(0.0, sens_matrix.sum(axis=(0, 1)).max() * 1.1)
    
    if len(legend_patches) > 0:
        fig.legend(**_legend_kwargs)

    if title is not None or title != "":
        fig.suptitle(title)

    fig.tight_layout()

    plt.show()

    return fig


def plot_global_param_transitions(G: np.ndarray, dim_labels: list = None):
    plt.close()

    N = G.shape[0]

    fig, ax = plt.subplots(1, 1)
    bottom = np.zeros(N)

    if dim_labels is None:
        dim_labels = [f"x{i+1}" for i in range(N)]

    p = ax.bar(dim_labels, G, bottom=bottom)
    ax.bar_label(p, label_type='center')

    ax.set_title("Global Transition Count Per Dimension")
    ax.set_ylabel("transitions", color="black")

    ax.legend()

    plt.show()
    return fig


def plot_global_param_transitions_by_pair(G: np.ndarray, dim_labels: list = None, line_labels: list = None, *, 
                                          plot_labels=False, 
                                          plot_divider: bool = False,
                                          colors: list = None,
                                          hatch: str = "//",
                                          hatch_line_width: int = 1,
                                          x_axis_ticks_rotation: str = "vertical",
                                          x_axis_ha: str = "right",
                                          legend_kwargs: dict = None,
                                          plot_legend: bool = True,
                                          fig_kwargs: dict = None,
                                          title: str = "Aggregated input and output transitions per dimension"
                                          ):
    """
    @plot_divider: if True, plots black outlines around bars
    @plot_labels: if True, plots the actual value per stack in the center of the stack.
        Occludes if there are too many small stacks.
    
    """
    plt.close()

    old_hatch_width = plt.rcParams["hatch.linewidth"]
    plt.rcParams["hatch.linewidth"] = hatch_line_width

    N = G.shape[0]
    L = G.shape[1]

    if colors is None:
        colors = plt.color_sequences.get("tab10")[:L]

    _fig_kwargs = dict(dpi=300)

    if fig_kwargs is not None:
        _fig_kwargs.update(fig_kwargs)

    fig, ax = plt.subplots(1, 1, **_fig_kwargs)
    bottom = np.zeros(N)

    if dim_labels is None:
        dim_labels = [f"x{i+1}" for i in range(N)]

    if line_labels is None:
        line_labels = [f"{i}" for i in range(L)]

    _legend_kwargs = {
        "loc": "lower left",
        "ncol": G.shape[0] // 2
    }
    if legend_kwargs is not None:
        _legend_kwargs.update(legend_kwargs)

    # create legend
    legend_patches = []
    for label_idx in range(L):
        l = f"{label_idx}"
        if line_labels is not None:
            l = line_labels[label_idx]
        legend_patches.append(mpatches.Patch(color=colors[label_idx], label=l))
    _legend_kwargs["handles"] = legend_patches

    # zero bar to get patch values
    p = ax.bar(dim_labels, np.zeros(N), bottom=bottom)

    for label_idx_a in range(L):
        for label_idx_b in range(L):
            values = G[:, label_idx_a, label_idx_b]

            if values.sum() == 0:
                continue

            p_ = ax.bar(dim_labels, values, bottom=bottom, color=colors[label_idx_b], hatch=hatch, edgecolor=colors[label_idx_a], linewidth=0.0)
            bottom += values

            if plot_labels:
                ax.bar_label(p_, label_type="center")

            if plot_divider:
                for dim in range(len(values)):
                    v_ = values[dim]
                    if v_ == 0.0:
                        continue
                    ax.plot([p.patches[dim].xy[0], p.patches[dim].xy[0] + p.patches[dim].get_width()], [bottom[dim], bottom[dim]], c="black", linewidth=0.5)

    total_sum = G.sum(axis=(1, 2))

    for dim in range(N):
        ax.annotate(f"{float(total_sum[dim].item()):.3f}", (p.patches[dim].xy[0] + p.patches[dim].get_width() * 0.5, bottom[dim] + 0.01 * total_sum.max()), ha="center")

    ax.set_xticks(ticks=ax.get_xticks(), labels=dim_labels, rotation=x_axis_ticks_rotation, ha=x_axis_ha)
    if title is not None and title != "":
        ax.set_title("Global Transition Count per Dimension by Pairs")
    ax.set_ylabel("transitions", color="black")

    if plot_labels:
        ax.legend()

    if len(legend_patches) > 0 and plot_legend:
        fig.legend(**_legend_kwargs)

    fig.tight_layout()

    plt.show()
    plt.rcParams["hatch.linewidth"] = old_hatch_width
    return fig


def plot_clustered_normals_as_bar_chart(df: pd.DataFrame, dim_labels: list, plot_labels: bool = True, colors: list = None,
                                        title: str = "Clustered Transition-Vectors by Dim-Contribution"):
    plt.close()

    N = len(dim_labels)
    M = df.shape[0]  # n_clusters

    data = df[dim_labels].abs().values

    volume = df["volume"].values.reshape(-1, 1)

    values = (data / data.sum(1).reshape(-1, 1)) * volume

    cluster_size = df["cluster_size"].values
    cluster_distance = df["cluster_distance"].values

    fig, ax = plt.subplots(1, 1, figsize=(1.5 * M, 4))
    bottom = np.zeros(M)

    c_labels = make_object_array([f"$c_{i}$\n#{cluster_size[i]}" for i in range(M)])

    p = ax.bar(list(c_labels), np.zeros(M), bottom=bottom)

    if colors is None:
        colors = np.array(plt.color_sequences.get("Accent"))[:N]
    
    for dim in range(N):
        label = f"{dim_labels[dim]}"

        pos_indices = np.argwhere(df[dim_labels].values[:, dim] >= 0).flatten()
        neg_indices = np.argwhere(df[dim_labels].values[:, dim] < 0).flatten()

        # print(pos_indices, c_labels[pos_indices])

        if len(pos_indices) > 0:
            p1 = ax.bar(list(c_labels[pos_indices]), values[pos_indices, dim], label=label, bottom=bottom[pos_indices], color=colors[dim])
        if len(neg_indices) > 0:
            p2 = ax.bar(list(c_labels[neg_indices]), values[neg_indices, dim], bottom=bottom[neg_indices], hatch="/", color=p1.patches[0].get_facecolor() if len(pos_indices) > 0 else None)
        
        bottom += values[:, dim]

        if plot_labels:
            if len(pos_indices) > 0:
                ax.bar_label(p1, label_type="center")
            if len(neg_indices) > 0:
                ax.bar_label(p2, label_type="center")

    for j in range(M):
        ax.annotate(f"{float(volume[j].item()):.2f}\n~{cluster_distance[j]:.2f}", (p.patches[j].xy[0] + p.patches[j].get_width() * 0.5, bottom[j]), ha="center")

    if title != "" and title is not None:
        ax.set_title(title)
    ax.set_ylabel("transitions", color="black")

    ax.legend()

    plt.show()
    return fig
