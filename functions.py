
def histograms(data, columns):
    """
    A method to make multiple histograms from the columns of a dataset.
    Will simply display the plot.
    """
    import matplotlib.pyplot as plt
    nr_plots = len(columns)
    plt.subplots(nr_plots, 1, figsize=(10, 10))
    for i in range(len(columns)):
        column = columns[i]
        plt.subplot(nr_plots, 1, i + 1)
        plt.hist(data[column], 50)
        plt.ylabel(column)
    plt.show()


def plot_density(data, title, base_plot, bandwidth=0.005, kernel='gaussian', metric="manhattan", save=False,
                 show=True, folder='', has_weight=False):
    X, Y, Z = make_density(data, bandwidth=bandwidth, kernel=kernel, metric=metric, has_weight=has_weight)
    plot_likelihood(X, Y, Z, title, base_plot, save=save, show=show, folder=folder)


def plot_likelihood(X, Y, Z, title, base_plot, save=False, show=True, folder=''):
    """
    A method that will overlay a visualization of a kernel density estimate of a cloud of points, on top of a base plot.

    :param X: X coordinates where the density was evaluated
    :param Y: Y coordinates where the density was evaluated
    :param Z: Z log-likelihood values corresponding to x in X, y in Y
    :param title: Plot title
    :param base_plot: A baseline plot as produced by osmnx...
    :param save: Boolean indicating whether the plot should be saved
    :param show: Boolean indicating whether the plot should be shown
    :param folder: Folder where the plot should be saved
    :return:
    """
    import matplotlib.pyplot as plt
    import osmnx as ox
    import numpy as np

    min_x = min(X.flatten())
    max_x = max(X.flatten())

    min_y = min(Y.flatten())
    max_y = max(Y.flatten())

    fig, ax = ox.plot_graph(base_plot, bbox=(max_y, min_y, max_x, min_x), margin=0, node_size=0, fig_height=20,
                            axis_off=False, use_geom=False,
                            show=False, close=False)

    CS = plt.contourf(X, Y, np.exp(Z), levels=20)
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.title(title, fontsize=20)

    if save:
        plt.savefig(f'{folder}/{title}.png', dpi=100)
    if show:
        plt.show()

    fig.clf()
    plt.close()


def make_density(data, bandwidth=0.005, kernel='gaussian', metric="manhattan", has_weight=False):

    """
    A method that will return a kernel density estimate of the first two columns in the dataset.

    :param has_weight: if True, the dataset should contain sample weights in the third column.
    :param data: A Pandas dataframe where the first two columns are of interest (longitude, latitude).
    :param bandwidth: Kernel bandwidth.
    :param kernel: Kernel type
    :param metric: Metric to be used for distance between points.
    :return: X, Y, Z triplet where X and Y are coordinates of points where the density was evaluated and Z holds the
    corresponding log-likelihood in those points.
    """

    from sklearn.neighbors import KernelDensity
    import numpy as np

    X_train = data.iloc[:, [0, 1]].to_numpy()
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel, algorithm='ball_tree', metric=metric)
    if has_weight:
        weight = data.iloc[:, 2].to_numpy()
        kde.fit(X_train, sample_weight=weight)
    else:
        kde.fit(X_train)

    x = np.linspace(min(X_train[:, 0]), max(X_train[:, 0]))
    y = np.linspace(min(X_train[:, 1]), max(X_train[:, 1]))
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    Z = kde.score_samples(XX)
    Z = Z.reshape(X.shape)
    return X, Y, Z


def make_even_meshgrid(gdf, spacing_lat=0.01, spacing_long=0.01, margins=None):
    import numpy as np
    from shapely.geometry import Point

    if margins is None:
        margins = [0.0, 0.0, 0.0, 0.0]
    n = gdf.bbox_north[0] - margins[0]
    s = gdf.bbox_south[0] + margins[1]
    e = gdf.bbox_east[0] - margins[2]
    w = gdf.bbox_west[0] + margins[3]
    ns_steps = abs(int((n - s) / spacing_lat))
    ew_steps = abs(int((e - w) / spacing_long))
    x = np.linspace(s, n, num=ns_steps)
    y = np.linspace(e, w, num=ew_steps)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    geom = gdf.loc[0, 'geometry']
    output = []
    for pt in XX:
        if geom.intersects(Point(pt[1], pt[0])):
            output.append(pt)
    return output


def get_city_graph(city, network_type='walk'):
    import osmnx as ox

    try:
        G = ox.graph_from_place(city, network_type=network_type)
    except TypeError as e:
        print(f'type error for {city}, trying with 2nd result')
        G = ox.graph_from_place(city, network_type=network_type, which_result=2)
    return G


def make_distances(graph, orig, dest):
    import osmnx as ox

    orig_node = orig[2]
    dest_node = dest[2]
    street_dist = nx.shortest_path_length(graph, orig_node, dest_node, weight='length')
    great_circle_dist = ox.great_circle_vec(orig[0], orig[1],
                                            dest[0], dest[1],
                                            earth_radius=6371009)
    return street_dist, great_circle_dist


def get_nearest_nodes(G, points):
    import osmnx as ox

    node_ids = set()
    out = []
    for pt in points:
        node = ox.get_nearest_node(G, (pt[0], pt[1]))
        if node not in node_ids:
            node_ids.add(node)
            out.append([G.nodes[node]['y'], G.nodes[node]['x'], node])
    return out


def distance_sample(G, nodes, count=True):
    import time

    distances = []
    sample_size = len(nodes)
    if count:
        print(f'sample size: {sample_size}')
    for i in range(0, sample_size - 1):
        if count:
            print(f'i: {i}')
        orig = nodes[i]
        start_time = time.time()
        for j in range(i + 1, sample_size):
            dest = nodes[j]
            distances.append(tuple(orig) + tuple(dest) + make_distances(G, orig, dest))
        duration = time.time() - start_time
        if count:
            print(f'{sample_size - i - 1} steps in: {duration} for avg speed: {duration/(sample_size - i - 1)}')
    return distances


def get_path(graph, row):
    import osmnx as ox

    origin = (row['orig_lat'].values[0], row['orig_long'].values[0])
    dest = (row['dest_lat'].values[0], row['dest_long'].values[0])
    orig_node = ox.get_nearest_node(graph, origin)
    dest_node = ox.get_nearest_node(graph, dest)
    return nx.shortest_path(graph, orig_node, dest_node, weight='length')


def map_to_nodes(G, data):
    data['orig_lat'] = data.apply(lambda x: G.nodes[x['nn_orig']]['y'], axis=1)
    data['orig_long'] = data.apply(lambda x: G.nodes[x['nn_orig']]['x'], axis=1)
    data['dest_lat'] = data.apply(lambda x: G.nodes[x['nn_dest']]['y'], axis=1)
    data['dest_long'] = data.apply(lambda x: G.nodes[x['nn_dest']]['x'], axis=1)
    return data


def stack_coordinates(data):
    import pandas as pd

    orig_data = data[['orig_lat', 'orig_long', 'street_dist', 'geo_dist', 'dist_ratio']]
    dest_data = data[['dest_lat', 'dest_long', 'street_dist', 'geo_dist', 'dist_ratio']]
    orig_data.columns = ['lat', 'long', 'street_dist', 'geo_dist', 'dist_ratio']
    dest_data.columns = ['lat', 'long', 'street_dist', 'geo_dist', 'dist_ratio']
    return pd.concat([orig_data, dest_data]).reset_index()


def plot_pts_on_graph(graph, pts):
    import osmnx as ox
    import numpy as np
    from matplotlib import pyplot as plt

    pts = np.array(pts)
    fig, ax = ox.plot_graph(graph, margin=0, node_size=0, fig_height=20,
                            axis_off=False, use_geom=False,
                            show=False, close=False)
    plt.scatter(pts[:, 1], pts[:, 0])
    plt.show()
