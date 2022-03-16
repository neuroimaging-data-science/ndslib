import tempfile
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D, proj3d # noqa
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from .data import load_data
from skimage.transform import AffineTransform, warp
import warnings
import PIL
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image, display
from viznet import NodeBrush, EdgeBrush, connecta2a, node_sequence
from sklearn.model_selection import KFold, learning_curve


def crop_inplace(fname, box=(400, 1200, 2240, 1700)):
    """
    Crops down an image stored in a file

    Parameters
    ----------
    fname : str
        File storing the image.

    Notes
    -----
    This overwrites the existing file with the new cropped image.
    """

    with PIL.Image.open(fname) as im:
        (left, upper, right, lower) = box
        im_crop = im.crop((left, upper, right, lower))
    im_crop.save(fname)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from nilearn.plotting import plot_roi, plot_stat_map
    from nilearn.image import new_img_like


def _make_arrays(kind, color='red'):
    x, y, z = np.indices((12, 12, 12))

    if kind == "one_d":
        voxels = (y == 3) & (z == 3)
    elif kind == "two_d":
        voxels = (y == 3)
    elif kind == "three_d":
        voxels = (x < 4) & (y < 4) & (z < 4)
    elif kind == "four_d":
        cube1 = (x < 3) & (y < 3) & (z < 3)
        cube2 = (x >= 2) & (y >= 3) & (z >= 2) & (x < 5) & (y < 6) & (z < 5)
        cube3 = (x >= 4) & (y >= 6) & (z >= 4) & (x < 7) & (y < 9) & (z < 7)
        cube4 = (x >= 6) & (y >= 9) & (z >= 6) & (x < 9) & (y < 12) & (z < 9)

        # combine the objects into a single boolean array
        voxels = cube1 | cube2 | cube3 | cube4

        # set the colors of each object
        colors = np.empty(voxels.shape, dtype=object)
        colors[cube1] = color
        colors[cube2] = color
        colors[cube3] = color
        colors[cube4] = color
    colors = np.empty(voxels.shape, dtype=object)
    colors[voxels] = color
    return voxels, colors


def draw_arrays():
    """
    Draws examples of ndarrays in nice 3D colored format

    Returns
    -------
    fig: MPL Figure object
    """
    fig = plt.figure()
    voxels, colors = _make_arrays("one_d", 'C0')
    ax = fig.add_subplot(141, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    voxels, colors = _make_arrays("two_d", 'C1')
    ax = fig.add_subplot(142, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    voxels, colors = _make_arrays("three_d", 'C2')
    ax = fig.add_subplot(143, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    voxels, colors = _make_arrays("four_d", 'C3')
    ax = fig.add_subplot(144, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    for ax in fig.get_axes():
        ax.set_axis_off()

    fig.set_size_inches([14, 8])
    return fig


class Arrow3D(FancyArrowPatch):
    """
    Class for drawing arrows in 3D
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]),(xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def draw_array_with_bytes(bytes=None):
    """
    Draw an array with the byte counts and values.

    Returns
    -------
    fig: MPL Figure object
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.indices((12, 12, 12))
    voxels1 = (y == 3) & (z == 3) & (x > 6)
    voxels2 = (y == 3) & (z == 3) & (x <= 6) & (x > 1)
    ax.voxels(voxels1, facecolors='C2', edgecolor='k', shade=True, alpha=0.1)
    ax.voxels(voxels2, facecolors='C0', edgecolor='k', shade=True, alpha=0.1)
    ax.text(12, 0, 2.15, "1", color="red", size=10)
    ax.text(11, 0, 2.15, "1", color="red", size=10)
    ax.text(10, 0, 2.15, "2", color="red", size=10)
    ax.text(9, 0, 2.15, "3", color="red", size=10)
    ax.text(8, 0, 2.15, "5", color="red", size=10)
    ax.text(7, 0, 2.15, "8", color="red", size=10)
    ax.text(6, 0, 2.15, "13", color="red", size=10)
    ax.text(5, 0, 2.15, "21", color="red", size=10)
    ax.text(4, 0, 2.15, "34", color="red", size=10)
    ax.text(3, 0, 2.15, "55", color="red", size=10)
    ax.view_init(elev=15., azim=95.)

    if bytes is None:
        bytes = np.arange(2.8, 12, 1.02)

    for xx in [(12.5 - i, 11.4 - i) for i in np.arange(0, 10, 1.03)]:
        st = np.max(xx) < np.max(bytes) + 1.5
        lt = np.min(xx) > np.min(bytes)-1.5
        if st and lt:
            a = Arrow3D(xx, (0, 0), (3.5, 3.5), mutation_scale=10,
                        lw=3, arrowstyle="<->", color="b")
            ax.add_artist(a)

    for xx in bytes:
        ax.text(xx, 0, 4.15, "8", color="blue", size=10)
    fig.set_size_inches([24, 12])
    ax.set_axis_off()
    return fig


def imshow_with_annot(im, vmax=40):
    """
    Like imshow, but with added annotation of the array values

    Parameters
    ----------
    im : numpy array object
    """
    fig, ax = plt.subplots(1)
    sns.heatmap(im, annot=True, ax=ax, cbar=False, cmap='gray',
                vmax=vmax, vmin=-vmax)
    plt.axis("off")
    ax.set_aspect("equal")


def bias_variance_dartboard():
    """
    Draws a bias-variance dartboard image.

    Returns
    -------
    fig: MPL Figure object
    """
    N_DARTS = 20
    LABEL_SIZE = 14
    fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
    fig.set_size_inches((8, 8))
    theta = np.linspace(0, 2*math.pi, 1000)

    data = np.array([[
        # low-bias, low-variance
        [np.random.uniform(0, 2*math.pi, N_DARTS),
        np.random.uniform(0, 2, N_DARTS)],
        # high-bias, low-variance
        [np.random.normal(3, 0.2, N_DARTS),
        np.random.normal(5, 0.5, N_DARTS)]],
        # low-bias, high-variance
        [[np.linspace(0, 2*math.pi, N_DARTS),
        np.random.uniform(0, 6, N_DARTS)],
        # high-bias, high-variance
        [np.random.normal(3, 0.5, N_DARTS),
        np.random.uniform(2, 9.8, N_DARTS)]
        ]])

    for i in range(2):
        for j in range(2):
            ax = axes[i,j]
            for k in range(4):
                ax.plot(theta, np.ones_like(theta)*k*3+0.4, lw=5, c='red',
                        alpha=0.5)
            ax.set_ylim(0, 11)
            ax.set_xticks([])
            ax.set_yticks([])
            x = data[i, j, 0, :]
            y = data[i, j, 1, :]
            ax.scatter(x, y, marker='x', c='navy', s=70, lw=1.6, zorder=100)
        axes[0,0].set_title('Low bias', fontsize=LABEL_SIZE)
        axes[0,1].set_title('High bias', fontsize=LABEL_SIZE)
        axes[0,0].set_ylabel('Low variance', fontsize=LABEL_SIZE, labelpad=20)
        axes[1,0].set_ylabel('High variance', fontsize=LABEL_SIZE, labelpad=20)
    return fig


def plot_hcp_mmp1(values=None, **kwargs):
    """
    Plots values of the HCP multi-modal parcellation as a nilearn stat map.
    """
    img = load_data("hcp-mmp1")
    if values is None:
        return plot_roi(img, **kwargs)

    else:
        img_data = np.round(img.get_fdata())
        for i in range(360):
            img_data[img_data == (i + 1)] = values[i]
        img = new_img_like(img, img_data, img.affine)
        return plot_stat_map(img, **kwargs)


# The following function are taken and adapted from the DIPY source code.
# The DIPY License follows:
#
# Unless otherwise specified by LICENSE.txt files in individual
# directories, or within individual files or functions, all code is:

# Copyright (c) 2008-2021, dipy developers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the dipy developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def draw_lattice_2d(nrows, ncols, delta):
    r"""Create a regular lattice of nrows x ncols squares.
    Creates an image (2D array) of a regular lattice of nrows x ncols squares.
    The size of each square is delta x delta pixels (not counting the
    separation lines). The lines are one pixel width.
    Parameters
    ----------
    nrows : int
        the number of squares to be drawn vertically
    ncols : int
        the number of squares to be drawn horizontally
    delta : int
        the size of each square of the grid. Each square is delta x delta
        pixels
    Returns
    -------
    lattice : array, shape (R, C)
        the image (2D array) of the segular lattice. The shape (R, C) of the
        array is given by
        R = 1 + (delta + 1) * nrows
        C = 1 + (delta + 1) * ncols
    """
    lattice = np.ndarray((1 + (delta + 1) * nrows,
                          1 + (delta + 1) * ncols),
                         dtype=np.float64)

    # Fill the lattice with "white"
    lattice[...] = 127

    # Draw the horizontal lines in "black"
    for i in range(nrows + 1):
        lattice[i * (delta + 1), :] = 0

    # Draw the vertical lines in "black"
    for j in range(ncols + 1):
        lattice[:, j * (delta + 1)] = 0

    return lattice


def plot_diffeomorphic_map(mapping, ax, delta=15,
                           direct_grid_shape=None, direct_grid2world=-1,
                           inverse_grid_shape=None, inverse_grid2world=-1,
                           show_figure=True, **fig_kwargs):
    r"""Draw the effect of warping a regular lattice by a diffeomorphic map.
    Draws a diffeomorphic map by showing the effect of the deformation on a
    regular grid. The resulting figure contains two images: the direct
    transformation is plotted to the left, and the inverse transformation is
    plotted to the right.
    Parameters
    ----------
    mapping : DiffeomorphicMap object
        the diffeomorphic map to be drawn
    delta : int, optional
        the size (in pixels) of the squares of the regular lattice to be used
        to plot the warping effects. Each square will be delta x delta pixels.
        By default, the size will be 10 pixels.
    fname : string, optional
        the name of the file the figure will be written to. If None (default),
        the figure will not be saved to disk.
    direct_grid_shape : tuple, shape (2,), optional
        the shape of the grid image after being deformed by the direct
        transformation. By default, the shape of the deformed grid is the
        same as the grid of the displacement field, which is by default
        equal to the shape of the fixed image. In other words, the resulting
        deformed grid (deformed by the direct transformation) will normally
        have the same shape as the fixed image.
    direct_grid2world : array, shape (3, 3), optional
        the affine transformation mapping the direct grid's coordinates to
        physical space. By default, this transformation will correspond to
        the image-to-world transformation corresponding to the default
        direct_grid_shape (in general, if users specify a direct_grid_shape,
        they should also specify direct_grid2world).
    inverse_grid_shape : tuple, shape (2,), optional
        the shape of the grid image after being deformed by the inverse
        transformation. By default, the shape of the deformed grid under the
        inverse transform is the same as the image used as "moving" when
        the diffeomorphic map was generated by a registration algorithm
        (so it corresponds to the effect of warping the static image towards
        the moving).
    inverse_grid2world : array, shape (3, 3), optional
        the affine transformation mapping inverse grid's coordinates to
        physical space. By default, this transformation will correspond to
        the image-to-world transformation corresponding to the default
        inverse_grid_shape (in general, if users specify an inverse_grid_shape,
        they should also specify inverse_grid2world).
    show_figure : bool, optional
        if True (default), the deformed grids will be plotted using matplotlib,
        else the grids are just returned
    fig_kwargs: extra parameters for saving figure, e.g. `dpi=300`.
    Returns
    -------
    warped_forward : array
        Image with the grid showing the effect of transforming the moving image to
        the static image.  The shape will be `direct_grid_shape` if specified,
        otherwise the shape of the static image.
    warped_backward : array
        Image with the grid showing the effect of transforming the static image to
        the moving image.  Shape will be `inverse_grid_shape` if specified,
        otherwise the shape of the moving image.
    Notes
    ------
    The default value for the affine transformation is "-1" to handle the case
    in which the user provides "None" as input meaning "identity". If we used
    None as default, we wouldn't know if the user specifically wants to use
    the identity (specifically passing None) or if it was left unspecified,
    meaning to use the appropriate default matrix.
    """
    if mapping.is_inverse:
        # By default, direct_grid_shape is the codomain grid
        if direct_grid_shape is None:
            direct_grid_shape = mapping.codomain_shape
        if direct_grid2world == -1:
            direct_grid2world = mapping.codomain_grid2world

        # By default, the inverse grid is the domain grid
        if inverse_grid_shape is None:
            inverse_grid_shape = mapping.domain_shape
        if inverse_grid2world == -1:
            inverse_grid2world = mapping.domain_grid2world
    else:
        # Now by default, direct_grid_shape is the mapping's input grid
        if direct_grid_shape is None:
            direct_grid_shape = mapping.domain_shape
        if direct_grid2world == -1:
            direct_grid2world = mapping.domain_grid2world

        # By default, the output grid is the mapping's domain grid
        if inverse_grid_shape is None:
            inverse_grid_shape = mapping.codomain_shape
        if inverse_grid2world == -1:
            inverse_grid2world = mapping.codomain_grid2world

    # The world-to-image (image = drawn lattice on the output grid)
    # transformation is the inverse of the output affine
    world_to_image = None
    if inverse_grid2world is not None:
        world_to_image = np.linalg.inv(inverse_grid2world)

    # Draw the squares on the output grid
    lattice_out = draw_lattice_2d(
        (inverse_grid_shape[0] + delta) // (delta + 1),
        (inverse_grid_shape[1] + delta) // (delta + 1),
        delta)
    lattice_out = lattice_out[0:inverse_grid_shape[0], 0:inverse_grid_shape[1]]

    # Warp in the forward direction (sampling it on the input grid)
    warped_forward = mapping.transform(lattice_out, 'linear', world_to_image,
                                       direct_grid_shape, direct_grid2world)

    # Now, the world-to-image (image = drawn lattice on the input grid)
    # transformation is the inverse of the input affine
    world_to_image = None
    if direct_grid2world is not None:
        world_to_image = np.linalg.inv(direct_grid2world)

    # Draw the squares on the input grid
    lattice_in = draw_lattice_2d((direct_grid_shape[0] + delta) // (delta + 1),
                                 (direct_grid_shape[1] + delta) // (delta + 1),
                                 delta)
    lattice_in = lattice_in[0:direct_grid_shape[0], 0:direct_grid_shape[1]]

    # Warp in the backward direction (sampling it on the output grid)
    warped_backward = mapping.transform_inverse(
        lattice_in, 'linear', world_to_image, inverse_grid_shape,
        inverse_grid2world)

    ax.set_axis_off()
    ax.imshow(warped_forward)
    ax.get_figure().set_size_inches([10, 6])

# This is the end of code taken from DIPY


def xform_naomi(naomi1, naomi2):
    """
    Apply a particular transformation to an image of Naomi

    Returns
    -------
    naomi1_xform, naomi2_xform : numpy arrays.

    """
    at = AffineTransform(rotation=-np.pi/65, scale=0.85, translation=(10, 20))
    naomi2_xform = (warp(naomi2, at) * 255).astype(np.uint8)
    naomi1_xform = naomi1
    naomi2_xform = naomi2_xform
    return naomi1_xform, naomi2_xform


def shear_naomi(naomi2):
    """
    Apply a set of shears to an image of Naomi

    Returns
    -------
    fig: MPL Figure object
    """
    fig, ax = plt.subplots(1, 5)
    shears = [0, np.pi/30, 2*np.pi/30, 3*np.pi/30, 4*np.pi/30]
    for ii, shear in enumerate(shears):
        at = AffineTransform(shear=shear, scale=1.5)
        naomi2_shear = (warp(naomi2, at) * 255).astype(np.uint8)
        ax[ii].matshow(naomi2_shear[:534], cmap="gray")
        ax[ii].axis("off")
    fig.set_size_inches(20, 4)
    fig.tight_layout()
    return fig


def plot_graphviz_tree(tree, feature_names):
    """
    Helper function that takes a tree as input, calls sklearn's export_graphviz
    function to generate an image of the tree using graphviz, and then
    plots the result in-line.

    Parameters
    ----------
    tree: sklearn tree object
    feature_names : sequence of strings
    """
    dot_file = tempfile.NamedTemporaryFile(suffix=".dot").name
    png_file = tempfile.NamedTemporaryFile(suffix=".png").name

    export_graphviz(tree, out_file=dot_file, max_depth=3, filled=True,
                    feature_names=feature_names, impurity=False, rounded=True,
                    proportion=False, precision=2)

    call(['dot', '-Tpng', dot_file, '-o', png_file, '-Gdpi=600'])
    display(Image(filename=png_file))


def draw_network(num_node_list, ax, radius, with_f=False):
    """
    Draw a neural network based on the number of nodes

    """
    num_hidden_layer = len(num_node_list) - 2
    token_list = ['\sigma^z'] + \
        ['y^{(%s)}' % (i + 1) for i in range(num_hidden_layer)] + ['\psi']
    kind_list = ['nn.input'] + ['nn.hidden'] * num_hidden_layer + ['nn.output']


    radius_list = [radius] * (num_hidden_layer + 2)
    if with_f:
        radius_list = [r+0.2 for r in radius_list]
    y_list = 2 * np.arange(len(num_node_list))

    seq_list = []
    for n, kind, radius, y in zip(num_node_list, kind_list, radius_list, y_list):
        kind = "basic"
        b = NodeBrush(kind, ax, size=radius)
        seq_list.append(node_sequence(b, n, center=(0, y), space=(3, 0)))

    eb = EdgeBrush('-->', ax)
    for st, et in zip(seq_list[:-1], seq_list[1:]):
        connecta2a(st, et, eb)

    for ii, layer in enumerate(seq_list):
        for jj, node in enumerate(layer):
            my_text = '$X_{%s%s}$'%(ii+1, jj+1)
            if with_f and ii > 0:
                my_text = "f(" + my_text + ")"
            node.text(my_text, 'center', fontsize=14)
    return ax


def plot_cv():
    """
    Draw a picture of K-Fold cross-validation with train/test.

    Return
    ------
    fig: MPL Figure object

    """
    X = np.random.randn(100, 10)

    percentiles_classes = [0.1, 0.3, 0.6]
    y = np.hstack([
        [ii] * int(100 * perc) for ii, perc in enumerate(percentiles_classes)])

    # Evenly spaced groups repeated once
    groups = np.hstack([[ii] * 10 for ii in range(10)])
    cmap = plt.cm.gray
    n_splits = 4

    fig, ax = plt.subplots()
    cv = KFold(5)
    ax.scatter(
        [7] * len(X),
        range(len(X)),
        c=[0] * len(X),
        marker="|",
        lw=20,
        cmap=cmap,
        vmin=-0.2,
        vmax=1.2)

    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
        indices = np.zeros(len(X))
        indices[tt] = 1
        indices[tr] = 0.6

        # Visualize the results
        ax.scatter(
            [ii + 0.5] * len(indices),
            range(len(indices)),
            c=indices,
            marker="|",
            lw=20,
            cmap=cmap,
            vmin=-0.2,
            vmax=1.2)

    xticklabels = (np.arange(n_splits+1, 0, -1))
    ax.set(
        xticks=np.arange(n_splits+1) + 0.5,
        xticklabels=xticklabels,
        xlim=[8, -0.2],
        ylim=[0, 100],
    )

    ax.annotate("", xy=(5, 50), xytext=(6.5, 50),
                arrowprops=dict(arrowstyle="->", linewidth=2))

    ax.text(7.75, 50, "DATA", rotation=90, ha="center", va="center", fontsize=14)

    ax.text(4.48, 40, "Train", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(4.48, 88, "Test", rotation=90, ha="center", va="center", fontsize=14)

    ax.text(3.48, 30, "Train", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(3.48, 68, "Test", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(3.48, 90, "Train", rotation=90, ha="center", va="center", fontsize=14)

    ax.text(2.48, 20, "Train", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(2.48, 48, "Test", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(2.48, 80, "Train", rotation=90, ha="center", va="center", fontsize=14)

    ax.text(1.48, 10, "Train", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(1.48, 28, "Test", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(1.48, 70, "Train", rotation=90, ha="center", va="center", fontsize=14)

    ax.text(.48, 10, "Test", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(.48, 60, "Train", rotation=90, ha="center", va="center", fontsize=14)


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_tick_params(width=0)
    fig.set_size_inches([10, 10])
    return fig


def plot_cv_tri():
    """
    Draw a picture of K-Fold cross-validation with train/validation/test

    Return
    ------
    fig: MPL Figure object
    """
    fig, ax = plt.subplots()
    cv = KFold(5)

    np.random.seed(1338)
    cmap = plt.cm.gray

    X = np.random.randn(100, 10)

    percentiles_classes = [0.1, 0.3, 0.6]
    y = np.hstack([[ii] * int(100 * perc) for ii, perc in enumerate(percentiles_classes)])

    # Evenly spaced groups repeated once
    groups = np.hstack([[ii] * 10 for ii in range(10)])

    n_splits=4

    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
        indices = np.zeros(len(X))
        indices[tt] = 1
        indices[tr] = 0.6

    ax.scatter(
        [9] * len(indices),
        range(len(indices)),
        c=np.zeros(indices.shape),
        marker="|",
        lw=20,
        cmap=cmap,
        vmin=-0.2,
        vmax=1.2)

    ax.scatter(
        [7] * len(indices),
        range(len(indices)),
        c=indices,
        marker="|",
        lw=20,
        cmap=cmap,
        vmin=-0.2,
        vmax=1.2)

    X = X[:80]
    y = y[:80]

    # Evenly spaced groups repeated once
    groups = np.hstack([[ii] * 8 for ii in range(10)])

    n_splits=4

    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
        indices = np.zeros(len(X))
        indices[tt] = 1
        indices[tr] = 0.9

        ax.scatter(
            [ii + 0.5] * len(indices),
            range(len(indices)),
            c=indices,
            marker="|",
            lw=20,
            cmap=cmap,
            vmin=-0.2,
            vmax=1.2)



    xticklabels = (np.arange(n_splits+1, 0, -1)) #+ ["class", "group"]
    ax.set(
        xticks=np.arange(n_splits+1) + 0.5,
        xticklabels=xticklabels,
        xlim=[10, -0.2],
        ylim=[0, 100],
    )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_tick_params(width=0)

    ax.text(10, 50, "DATA", rotation=90, ha="center", va="center", fontsize=14)

    ax.text(6.97, 30, "Train", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(6.97, 90, "Test", rotation=90, ha="center", va="center", fontsize=14)

    ax.text(4.48, 40, "Train", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(4.48, 71, "Valid", rotation=90, ha="center", va="center", fontsize=14)

    ax.text(3.48, 30, "Train", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(3.48, 55, "Valid", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(3.48, 71, "Train", rotation=90, ha="center", va="center", fontsize=14)

    ax.text(2.48, 20, "Train", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(2.48, 39, "Valid", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(2.48, 60, "Train", rotation=90, ha="center", va="center", fontsize=14)

    ax.text(1.48, 7, "Train", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(1.48, 23, "Valid", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(1.48, 48, "Train", rotation=90, ha="center", va="center", fontsize=14)

    ax.text(.48, 7, "Valid", rotation=90, ha="center", va="center", fontsize=14)
    ax.text(.48, 40, "Train", rotation=90, ha="center", va="center", fontsize=14)

    ax.annotate("", xy=(7.4, 50), xytext=(8.5, 50),
                arrowprops=dict(arrowstyle="->", linewidth=2))


    ax.annotate("", xy=(5, 40), xytext=(6.5, 40),
                arrowprops=dict(arrowstyle="->", linewidth=2))


    fig.set_size_inches([10, 10])
    return fig


def plot_learning_curves(estimators, X_sets, y, train_sizes, labels=None,
                         errors=True, **kwargs):
    ''' Generate multi-panel plot displaying learning curves for multiple
    predictor sets and/or estimators.

    Args:
        estimators (Estimator, list): A scikit-learn Estimator or list of
            estimators. If a list is provided, it must have the same number of
            elements as X_sets.
        X_sets (NDArray-like, list): An NDArray or similar object, or list. If
            a list is passed, it must have the same number of elements as
            estimators.
        y (NDArray): a 1-D numpy array (or pandas Series) representing the
            outcome variable to predict.
        train_sizes (list): List of ints providing the sample sizes at which to
            evaluate the estimator.
        labels (list): Optional list of labels for the panels. Must have the
            same number of elements as X_sets.
        errors (bool): If True, plots error bars representing 1 StDev.
        kwargs (dict): Optional keyword arguments passed on to sklearn's
            `learning_curve` utility.
    '''
    # Set up figure
    n_col = len(X_sets)
    fig, axes = plt.subplots(1, n_col, figsize=(4.5 * n_col, 4), sharex=True,
                             sharey=True)

    # If there's only one subplot, matplotlib will hand us back a single Axes,
    # so wrap it in a list to facilitate indexing inside the loop
    if n_col == 1:
        axes = [axes]

    # If estimators is a single object, repeat it n_cols times in a list
    if not isinstance(estimators, (list, tuple)):
        estimators = [estimators] * n_col

    cv = kwargs.pop('cv', 10)

    # Plot learning curve for each predictor set
    for i in range(n_col):
        ax = axes[i]
        results = learning_curve(estimators[i], X_sets[i], y,
                                 train_sizes=train_sizes, shuffle=True,
                                 cv=cv, **kwargs)
        train_sizes_abs, train_scores, test_scores = results
        train_mean = train_scores.mean(1)
        test_mean = test_scores.mean(1)
        ax.plot(train_sizes_abs, train_mean, 'o-', label='Train',
                lw=3)
        ax.plot(train_sizes_abs, test_mean, 'o-', label='Test',
                lw=3)
        # axes[i].set_xscale('log')
        axes[i].xaxis.set_major_formatter(ScalarFormatter())
        axes[i].grid(False, axis='x')
        axes[i].grid(True, axis='y')
        if labels is not None:
            ax.set_title(labels[i], fontsize=16)
        ax.set_xlabel('Num. obs.', fontsize=14)

        if errors:
            train_sd = train_scores.std(1)
            test_sd = test_scores.std(1)
            ax.fill_between(train_sizes, train_mean - train_sd,
                            train_mean + train_sd, alpha=0.2)
            ax.fill_between(train_sizes, test_mean - test_sd,
                            test_mean + test_sd, alpha=0.2)

    # Additional display options
    plt.legend(fontsize=14)
    plt.ylim(0, 1)
    axes[0].set_ylabel('$R^2$', fontsize=14)
    axes[-1].set_ylabel('$R^2$', fontsize=14)
    axes[-1].yaxis.set_label_position("right")


def plot_coef_path(estimator, X, y, alpha, **kwargs):
    """
    Plot the coefficient path for a sklearn estimator

    Parameters
    ----------
    estimator : sklearn estimator
        For example `Lasso()`
    X : ndarray (n, m)
        Feature matrix
    y : ndarray (n,)
        Target matrix

    Returns
    -------
    ax : MPL Axes object
    """
    fig, ax = plt.subplots()
    coefs = np.zeros((X.shape[1], len(alpha)))
    for (i, a) in enumerate(alpha):
        coefs[:,i] = estimator(alpha=a, **kwargs).fit(X, y).coef_
    ax.plot(alpha, coefs.T)
    ax.set_xlabel("Penalty (alpha)")
    ax.set_ylabel("Coefficient value")
    ax.set_xscale('log')
    return ax


def plot_train_test(x_range, train_scores, test_scores, label, hlines=None):
    """
    Plot train/test $R^2$

    Parameters
    ----------
    x_range : sequence
        The range of x values used (e.g., number of features, number of samples)
    train_scores : sequence
        The train r2_score corresponding to different x values
    test_scores : sequence
        The test r2_score corresponding to different x values
    label : str
        Used in the legend labels.
    hlines : dict
        A dictionary where keys are labels and values are y values for hlines.

    Returns
    -------
    ax : MPL Axes object.
    """
    fig, ax = plt.subplots()
    ax.plot(x_range, train_scores.mean(1), label=f'{label} (train)', linewidth=2)
    ax.plot(x_range, test_scores.mean(1), label=f'{label} (test)', linewidth=2)
    ax.grid(axis='y', linestyle='--')
    ax.set_xscale('log')
    ax.set_ylabel('$R^2$', fontsize=14)
    ax.set_xlabel('Penalty (alpha)', fontsize=14)
    ax.set_ylim(0, 1)

    if hlines:
        for lab, line in hlines.items():
            ax.hlines(line, x_range.min(), x_range.max(), linestyles='--', linewidth=2, label=lab)
    ll = ax.legend()
    return ax
