from src.para_voro_plots import *
from src.para_voro import *


def compute_and_plot_para_voro(df: pd.DataFrame, normalize_samples: bool = True, n_samples_per_dim: int = 30, 
                               clip_convex_hull: bool = True, 
                               label_colors: list = None, 
                               dim_colors: list = None,
                               discrete_values_mapping: dict = None,
                               normalize_by_bandwidth: bool = True,
                               *, 
                               skip_label_distribution: bool = False,
                               plot_sens_matrix_list_kwargs: dict = None, plot_label_distr_list_kwargs: dict = None,
                               plot_aggregated_input_output_transitions_kwargs: dict = None,
                               plot_global_param_transitions_by_pair_kwargs: dict = None,
                               plot_sensitivities_kwargs: dict = None,
                               compute_clustered_normals_kwargs: dict = None,
                               plot_clustered_normals_as_bar_chart_kwargs: dict = None,
                               verbose_geom: bool = False, verbose_trans: bool = False, verbose_light: bool = True, n_proc: int = None,
                               percentile_ridge_volumes_to_keep: float = 1.0,
                               save_figs: bool = False, save_fig_prefix: str = "fig_", save_fig_suffix: str = ".pdf",
                               save_fig_kwargs: dict = None, 
                               precomputed_: dict = None):
    """
    @df: Dataframe with n columns. first n-1 are the samples, the n-th column are the labels (by name).
        The dimension names are taken from the column names.
    @normalize_samples: whether to normalize the samples
    @discrete_values_mapping: a mapping from discrete values to numerical values to achieve an ordering, e.g. {"col1": {"Apple": 0, "Banana": 1, ... }, "col2": {"Name": 0, "Date": 1, ...}, ...}
        It will be applied to columns that have dtype=object (strings).
        If none given, the mapping will be created in the order the values appear in the data. 
    @skip_label_distribution: skips computation of the label distribution. use if it takes too long.
    @save_figs: If True, saves the figures under {save_fig_prefix}_{current_plot}{save_fig_suffix}  # TODO: implement
    @save_fig_prefix: Prefix of the figure
    @save_fig_suffix: Suffix of the figure
    @normalize_by_bandwidth: If True, divides the transition and distribution matrices by the bandwidth per dimension. 
    @percentile_ridge_volumes_to_keep: After creating all ridge vertices and computing their volumes, only keep the percentile of volumes of this value.
        1.0 -> keep all ridges
        0.99 -> keep all ridges that make up the upper 99% of volumes
    @precomputed_: can provide the output of this function to avoid recomputations when adjusting the plot parameters
    """
    def pf(*args):
        if verbose_light:
            print(*args)
        else:
            pass

    if precomputed_ is None:
        precomputed_ = dict()

    ret = dict()

    save_fig_fmt = f"{save_fig_prefix}{{}}{save_fig_suffix}"
    _save_fig_kwargs = dict(
        bbox_inches="tight"
    )
    if save_fig_kwargs is not None:
        _save_fig_kwargs.update(save_fig_kwargs)

    if discrete_values_mapping is None:
        discrete_values_mapping = create_discrete_values_mapping(df)

    discrete_dims = {
        dim: col_name
            for dim, col_name in enumerate(df.columns[:-1]) if df[col_name].dtype == object
        }
    
    labels_names = df.values[:, -1]

    df_ = apply_discrete_values_mapping(df[df.columns[:-1]], discrete_values_mapping)

    pf(f"compute samples and labels...")
    samples = df_.values
    dim_labels = df_.columns.to_list()
    
    label_names, uq_idx, l_inv = np.unique(labels_names, return_index=True, return_inverse=True)
    labels = l_inv  # the label we use for computations are integers, while the label names may be strings / ints, floats / objects

    if label_colors is None:
        label_colors = np.array(plt.color_sequences.get("tab10"))[:len(label_names)]

    if dim_colors is None:
        dim_colors = np.array(plt.color_sequences.get("Accent"))[:len(dim_labels)]

    ret["label_names"] = labels_names.copy()
    ret["dim_labels"] = dim_labels.copy()
    ret["dim_colors"] = dim_colors.copy()

    # set default values
    _plot_sens_matrix_list_kwargs = dict(
        do_pairwise_labels=False, stackplot_fill=0, stackplot_color_hatch=True, stack_plot_kwargs=dict(baseline="sym", hatch="//", colors=label_colors), 
        stack_plot_gradient=False, horizontal_plot=True, stack_plot_outline=False, discrete_dimension_ticks_and_labels_kwargs=dict(rotation="vertical"),
        share_y_axis=True, stackplot_divider=True, fig_kwargs=dict(dpi=300), stackplot_hatch_linewidth=4,
        dim_labels=dim_labels, line_labels=label_names, title="Pairwise transitions divided by distribution"
    )
    
    _plot_label_distr_list_kwargs = dict(
        stack_plot=True, horizontal_layout=True, share_y_axis=True, fig_kwargs=dict(dpi=300), stack_plot_kwargs=dict(baseline="zero", colors=label_colors),
        discrete_dimension_ticks_and_labels_kwargs=dict(rotation="vertical"), stackplot_divider=True,
        dim_labels=dim_labels, line_labels=label_names,
    )

    _plot_aggregated_input_output_transitions_kwargs = dict(
        stack_plot=True, stack_plot_kwargs=dict(baseline="sym", colors=label_colors), stackplot_divider=True, discrete_dimension_ticks_and_labels_kwargs=dict(rotation="vertical"),
        fig_kwargs=dict(dpi=300), share_y_axis=True, 
        dim_labels=dim_labels, line_labels=label_names,
    )

    _plot_global_param_transitions_by_pair_kwargs = dict(
        dim_labels=dim_labels, line_labels=label_names, plot_labels=False, fig_kwargs=dict(figsize=(4,4)), 
        hatch="//", hatch_line_width=4, plot_divider=True, colors=label_colors,
    )

    _plot_sensitivities_kwargs = dict(
        dim_labels=dim_labels, discrete_dimension_ticks_and_labels_kwargs=dict(rotation="vertical"), bandwidths=None, 
        horizontal_plot=True, share_y_axis=True
    )

    _compute_clustered_normals_kwargs = dict(
        n_clusters=5, dim_labels=dim_labels, 
    )

    _plot_clustered_normals_as_bar_chart_kwargs = dict(
        dim_labels=dim_labels, plot_labels=False, colors=dim_colors
    )

    if normalize_samples:
        pf(f"normalize samples...")
        # normalize samples  # TODO: should rather be [-0.5, 0.5] than [0.0., 1.0]
        mins = np.min(samples, 0)
        maxs = np.max(samples, 0)

        samples = (samples - mins) / (maxs - mins)

    # create discrete_dimension_ticks_and_labels mapping
    discrete_dimension_ticks_and_labels = {}
    for dim in discrete_dims:
        col_name = discrete_dims[dim]

        m = discrete_values_mapping[col_name]
        reverse_mapping = {v: k for k, v in m.items()}

        tick_values = np.linspace(0, 1.0 if normalize_samples else len(m) - 1, len(m))
        tick_labels = [reverse_mapping[idx] for idx in range(len(m))]

        discrete_dimension_ticks_and_labels[dim] = (tick_values, tick_labels)
    
    pf(f"compute voronoi...")
    if "vor" in precomputed_:
        vor = precomputed_["vor"]
    else:
        vor = Voronoi(samples)
        ret["vor"] = vor

    pf(f"compute aabbox...")
    aabbox = np.vstack([np.min(samples, 0), np.max(samples, 0)]).T
    bandwidths = (aabbox[:, 1] - aabbox[:, 0]) / n_samples_per_dim
    step_size=None

    if samples.shape[1] <= 3:
        pf(f"plot data...")
        f = plot_data(samples, labels, aabbox, colors=label_colors, dim_names=dim_labels)
        if save_figs:
            f.savefig(save_fig_fmt.format("plot_data"), **_save_fig_kwargs)

    pf(f"compute ridge vertices...")
    if "generate_geometric_segment_boundaries_via_voronoi_result" in precomputed_:
        generate_geometric_segment_boundaries_via_voronoi_result = precomputed_["generate_geometric_segment_boundaries_via_voronoi_result"]
    else:
        generate_geometric_segment_boundaries_via_voronoi_result = generate_geometric_segment_boundaries_via_voronoi(samples, labels, aabbox, clip_convex_hull=clip_convex_hull, vor=vor, return_original_ridge_point_indices=True, verbose=verbose_geom)
        ret["generate_geometric_segment_boundaries_via_voronoi_result"] = generate_geometric_segment_boundaries_via_voronoi_result
    

    ridge_vertices, ridge_point_indices = generate_geometric_segment_boundaries_via_voronoi_result

    if samples.shape[1] <= 3:
        pf(f"plot ridges...")
        f = plot_ridges(ridge_vertices, samples, labels, aabbox, clip_convex_hull, dim_names=dim_labels)
        if save_figs:
            f.savefig(save_fig_fmt.format("plot_ridges"), **_save_fig_kwargs)

    pf(f"build directed ridge vertices selector matrix...")
    if "M_T" in precomputed_:
        M_T = precomputed_["M_T"]
    else:
        M_T = build_directed_ridge_vertices_selector_matrix(samples, labels, ridge_point_indices)
        ret["M_T"] = M_T

    pf(f"compute sample points...")
    if "sample_points_min_max" in precomputed_:
        sample_points_min_max = precomputed_["sample_points_min_max"]
    else:
        sample_points_min_max = compute_sample_points(clip_box=aabbox, bandwidths=bandwidths, step_size=step_size)
        ret["sample_points_min_max"] = sample_points_min_max

    pf(f"compute preparations...")
    if "ridge_based_para_sense_preparations_result" in precomputed_:
        ridge_based_para_sense_preparations_result = precomputed_["ridge_based_para_sense_preparations_result"]
    else:
        ridge_based_para_sense_preparations_result = ridge_based_para_sense_preparations(ridge_vertices=ridge_vertices,
                                                                                     bandwidths=bandwidths,
                                                                                     clip_box=aabbox,
                                                                                     step_size=step_size,
                                                                                     sample_points_min_max=sample_points_min_max,
                                                                                     verbose=verbose_geom)
        ret["ridge_based_para_sense_preparations_result"] = ridge_based_para_sense_preparations_result
    
    pf(f"compute directed transition cubes..")
    if "build_directed_transition_cubes_result" in precomputed_:
        build_directed_transition_cubes_result = precomputed_["build_directed_transition_cubes_result"]
    else:
        build_directed_transition_cubes_result = build_directed_transition_cubes(M_T, ridge_vertices, sample_points_min_max=sample_points_min_max, 
                                                 ridge_based_para_sense_preparations_result=ridge_based_para_sense_preparations_result,
                                                 bandwidths=bandwidths, clip_box=aabbox, step_size=step_size, n_proc=n_proc, verbose=verbose_trans)
        ret["build_directed_transition_cubes_result"] = build_directed_transition_cubes_result

    _, dtc_ret = build_directed_transition_cubes_result

    sens_matrix_list = [d[0] for d in dtc_ret]
    n_ridge_matrix_list = [d[1] for d in dtc_ret]

    if normalize_by_bandwidth:
        sens_matrix_list = normalize_transition_matrix_by_bandwidth(sense_matrix_list=sens_matrix_list, bandwidths=bandwidths)

    _compute_clustered_normals_kwargs["ridge_based_para_sense_preparations_result"] = ridge_based_para_sense_preparations_result
    if compute_clustered_normals_kwargs is not None:
        _compute_clustered_normals_kwargs.update(compute_clustered_normals_kwargs)

    pf(f"compute clustering...")
    if "compute_clustered_normals_result" in precomputed_:
        compute_clustered_normals_result = precomputed_["compute_clustered_normals_result"]
    else:
        compute_clustered_normals_result = compute_clustered_normals(ridge_vertices=ridge_vertices, **_compute_clustered_normals_kwargs)
        ret["compute_clustered_normals_result"] = compute_clustered_normals_result
    df_knn, kcs, normals_sorted, rvv_sorted, normals_sorted_tf = compute_clustered_normals_result

    # reconstruct sample points for non-discrete values
    if not normalize_samples:
        rec_sample_points = sample_points_min_max[0]
    else:
        rec_sample_points = [sp * maxs[dim] + mins[dim] if dim not in discrete_dims else sp for dim, sp in enumerate(sample_points_min_max[0])]

    _plot_sens_matrix_list_kwargs["discrete_dimension_ticks_and_labels"] = discrete_dimension_ticks_and_labels
    if plot_sens_matrix_list_kwargs is not None:
        _plot_sens_matrix_list_kwargs.update(plot_sens_matrix_list_kwargs)

    _plot_label_distr_list_kwargs["discrete_dimension_ticks_and_labels"] = discrete_dimension_ticks_and_labels
    if plot_label_distr_list_kwargs is not None:
        _plot_label_distr_list_kwargs.update(plot_label_distr_list_kwargs)

    _plot_aggregated_input_output_transitions_kwargs["discrete_dimension_ticks_and_labels"] = discrete_dimension_ticks_and_labels
    if plot_aggregated_input_output_transitions_kwargs is not None:
        _plot_aggregated_input_output_transitions_kwargs.update(plot_aggregated_input_output_transitions_kwargs)
    
    if plot_global_param_transitions_by_pair_kwargs is not None:
        _plot_global_param_transitions_by_pair_kwargs.update(plot_global_param_transitions_by_pair_kwargs)

    _plot_sensitivities_kwargs["discrete_dimension_ticks_and_labels"] = discrete_dimension_ticks_and_labels
    if plot_sensitivities_kwargs is not None:
        _plot_sensitivities_kwargs.update(plot_sensitivities_kwargs)

    if plot_clustered_normals_as_bar_chart_kwargs is not None:
        _plot_clustered_normals_as_bar_chart_kwargs.update(plot_clustered_normals_as_bar_chart_kwargs)

    # compute distribution
    if not skip_label_distribution:
        pf(f"compute label distribution...")
        if "compute_label_distribution_result" in precomputed_:
            compute_label_distribution_result = precomputed_["compute_label_distribution_result"]
        else:
            compute_label_distribution_result = compute_label_distribution(samples, labels, bandwidths, aabbox, clip_convex_hull=clip_convex_hull, vor=vor, sample_points_min_max=sample_points_min_max)
            ret["compute_label_distribution_result"] = compute_label_distribution_result
        D_i_s, _, _, _ = compute_label_distribution_result
        if normalize_by_bandwidth:
            D_i_s = normalize_label_distribution_by_bandwidth(distr_matrix_list=D_i_s, bandwidths=bandwidths)
        pf(f"plot label distribution...")
        f = plot_label_distribution_matrix_list(sample_points=rec_sample_points, distr_matrix_list=D_i_s, **_plot_label_distr_list_kwargs)
        if save_figs:
            f.savefig(save_fig_fmt.format("plot_label_distribution_matrix_list"), **_save_fig_kwargs)
    else:
        pf(f"skip label distribution")

    # plot sorted volumes over ridge indices
    pf(f"plot ridge volumes")
    f = plot_simple_plot(x=np.arange(len(rvv_sorted)), y=rvv_sorted, title="Sorted volumes of ridges")
    if save_figs:
        f.savefig(save_fig_fmt.format("plot_simple_plot"), **_save_fig_kwargs)

    pf(f"plot sorted clusterings")
    f = plot_clustered_normals_as_bar_chart(df_knn, **_plot_clustered_normals_as_bar_chart_kwargs)
    if save_figs:
        f.savefig(save_fig_fmt.format("plot_clustered_normals_as_bar_chart"), **_save_fig_kwargs)

    # plot pairwise transition matrix
    pf(f"plot pairwise transition matrix...")
    f = plot_sens_matrix_list(sample_points=rec_sample_points, sens_matrix_list=sens_matrix_list, **_plot_sens_matrix_list_kwargs)
    if save_figs:
        # hotfix: this is needed during save
        old_hatch_width = plt.rcParams["hatch.linewidth"]
        plt.rcParams["hatch.linewidth"] = _plot_sens_matrix_list_kwargs["stackplot_hatch_linewidth"]
        f.savefig(save_fig_fmt.format("plot_sens_matrix_list"), **_save_fig_kwargs)
        plt.rcParams["hatch.linewidth"] = old_hatch_width

    # plot in-out aggregation
    in_out_agg_sens_matrix_list = compute_in_out_agg_sens_matrix_list(sens_matrix_list)
    pf(f"plot in-out transition matrix...")
    f = plot_aggregated_input_output_transitions(sample_points=rec_sample_points, in_out_agg_sens_matrix_list=in_out_agg_sens_matrix_list, **_plot_aggregated_input_output_transitions_kwargs)
    if save_figs:
        f.savefig(save_fig_fmt.format("plot_aggregated_input_output_transitions"), **_save_fig_kwargs)

    # plot global aggregation
    global_param_trans_matrix_by_pair = compute_global_param_transitions_by_pairs(sens_matrix_list)
    pf(f"plot global transitions...")
    f = plot_global_param_transitions_by_pair(global_param_trans_matrix_by_pair, **_plot_global_param_transitions_by_pair_kwargs)
    if save_figs:
        # hotfix: this is needed during save
        old_hatch_width = plt.rcParams["hatch.linewidth"]
        plt.rcParams["hatch.linewidth"] = _plot_sens_matrix_list_kwargs["stackplot_hatch_linewidth"]
        f.savefig(save_fig_fmt.format("plot_global_param_transitions_by_pair"), **_save_fig_kwargs)
        plt.rcParams["hatch.linewidth"] = old_hatch_width

    # plot transitions with number of ridges
    S_i_s, R_i_s = aggregate_directed_transition_matrix(sens_matrix_list, n_ridge_matrix_list)
    pf(f"plot transitions and ridges matrix...")
    f = plot_sensitivities(sample_points=rec_sample_points, sensitivities=S_i_s, n_ridges=R_i_s, **_plot_sensitivities_kwargs)
    if save_figs:
        f.savefig(save_fig_fmt.format("plot_sensitivities"), **_save_fig_kwargs)
    
    if skip_label_distribution:
        pf(f"due to skip label distribution... done")
        return ret

    # normalized distribution
    D_i_s_normed = normalize_label_distribution(D_i_s)
    pf(f"plot normalized label distribution...")
    plot_label_distribution_matrix_list(sample_points=rec_sample_points, distr_matrix_list=D_i_s_normed, **_plot_label_distr_list_kwargs)

    # all plots but using the normed variant
    # plot pairwise transition matrix divided by total distribution per sample point
    trans_matrix_normed_by_dist = normalize_transition_matrix_by_distribution(sens_matrix_list, D_i_s)
    pf(f"plot normed pairwise transition matrix...")
    plot_sens_matrix_list(rec_sample_points, trans_matrix_normed_by_dist, **_plot_sens_matrix_list_kwargs)

    # plot in-out aggregation
    in_out_agg_sens_matrix_list_normed = compute_in_out_agg_sens_matrix_list(trans_matrix_normed_by_dist)
    pf(f"plot normed in-out transition matrix...")
    plot_aggregated_input_output_transitions(rec_sample_points, in_out_agg_sens_matrix_list_normed, **_plot_aggregated_input_output_transitions_kwargs)

    # plot global aggregation
    global_param_trans_matrix_by_pair_normed = compute_global_param_transitions_by_pairs(trans_matrix_normed_by_dist)
    pf(f"plot normed global transitions...")
    plot_global_param_transitions_by_pair(global_param_trans_matrix_by_pair_normed, **_plot_global_param_transitions_by_pair_kwargs)

    # plot transitions with number of ridges
    S_i_s_normed, R_i_s_normed = aggregate_directed_transition_matrix(trans_matrix_normed_by_dist, n_ridge_matrix_list)
    pf(f"plot normed transitions and ridges matrix...")
    plot_sensitivities(sample_points=rec_sample_points, sensitivities=S_i_s_normed, n_ridges=R_i_s_normed, **_plot_sensitivities_kwargs)
    return ret
