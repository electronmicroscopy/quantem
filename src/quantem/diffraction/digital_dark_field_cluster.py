import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
import matplotlib.gridspec as GridSpec

from tqdm import tqdm

from sklearn.cluster import DBSCAN

from quantem.core.datastructures.vector import Vector

    # ------------------------------------------------------------------ #
    # Digital Dark Field Basics
    # ------------------------------------------------------------------ #

def make_FullPointsVector_centres(vecs,centers):
    '''
    This may be a bit wasteful but it builds a new vector object that contains everything you need for 
    DDF imaging.  Maybe you could do this instead by augmenting the existing object
    from disk detection, but I couldn't figure out how
    azimuthal angle is measured anticlockwise from horizontal right

    Parameters
    ----------
    vecs: Vector
        Currently must contain fields for kx, ky and intensity
    centers: np.ndarray
        A (2,Rx,Ry) array of kx and ky centres

    Returns
    -------
    pointsvector: Vector
        Containing fields ["rx", "ry", "kx", "ky", "kr", "kphi", "intensity"]

    '''
    Rshape = centers.shape[1:]
    if 'q_row' in vecs.fields:
        fields = ["q_row","q_col"]
    elif 'kx' in vecs.fields:
        fields = ["kx","ky"]
    pointsvector = Vector.from_shape(
        shape=Rshape,
        fields=("rx", "ry", "kx", "ky", "kr", "kphi", "intensity"),
        units=("pixels", "pixels", "pixels", "pixels", "pixels", "degrees", "counts"),
        name="diffraction_vectors",
    )
 
    for rx in tqdm(range(Rshape[0])):
        for ry in range(Rshape[1]):
            kx = vecs[rx,ry].select_fields(fields[0]).flatten()-centers[0,rx,ry]
            ky = vecs[rx,ry].select_fields(fields[1]).flatten()-centers[1,rx,ry]
            kr = (kx**2+ky**2)**.5
            kphi = np.degrees(np.arctan2(-kx, ky))
            I = vecs[rx,ry].select_fields("intensity").flatten()
            
            pointsvector[rx, ry] = np.column_stack((
                rx * np.ones_like(kx), 
                ry * np.ones_like(kx), 
                kx, 
                ky, 
                kr,
                kphi,
                I
            ))
    return pointsvector

def generate_DDF_pointselect_array(
    Qshape, 
    g1=None, 
    g2=None, 
    g1min=-1,
    g1max=1,
    g2min=-1,
    g2max=1,
    arrayorigin=np.array([0,0]), 
    rmin=0, 
    rmax=100
):
    '''
    Drop in replacement for earlier functions for creating a list selection points for forming
    Digital Dark Field images.  The function is more compact in construction, however.  This is only
    for spots in regular arrangements: single spots, lines (2-beam conditions) or arrays (zone axes).

    If you specify neither basis vector, g1 or g2, it just produces one point at the array origin,
    i.e. classic bright or dark field with one aperture.

    If you specify a g1, then it will make a line of spots along this.  Default is that this will be
    -g, 0 and g.

    If you specify both g1 and g2, you get a grid, currently 3x3 by default.  You adjust this by changing
    g1min, g1max, g2min, and g2max, which are the maximum multipliers for g1 and g2 in negative and positive
    senses.

    An array need not be centered on 0,0, if you move array origin (e.g. to g1 / 2 for a half RL cell shift)

    It is convenient to get g1 and g2 from the strain module.

    If you want a grid but to skip the central beam, then just set rmin as something larger than 0.  1 pixel will
    usually work with aligned data (if working in uncalibrated pixels).

    You can set a maximum radius cutoff too, if required.  rmin and rmax measure from (0,0), regardless of what you
    set for an arrayorigin.

    Parameters
    ----------
    Qshape: tuple
        Shape of the diffraction pattern
    g1: np.ndarray
        A [kx,ky] vector
    g2: np.ndarray
        A [kx,ky] vector
    g1min, g1max, g2min, g2max: int
        maximum multiples of each g-vector in either direction
    arrayorigin: np.ndarray
        A [kx,ky] vector, which sets where either a single aperture or the centre of some line or grid
        will go
    rmin, rmax: int, float
        min and max radii from [0,0] within which points will be selected

    Returns
    -------
    selected_points: np.ndarray
        A Nx2 vector which lists a number of kx,ky points chosen as selection positions for DDF imaging
        
    '''
    if isinstance(g1, np.ndarray):
        if isinstance(g2, np.ndarray):
            # Compute an array of points
            grids = np.mgrid[
                g1min:g1max+1,
                g2min:g2max+1
                ]
            selected_points = np.outer(grids[0].flatten(),g1)+np.outer(grids[1].flatten(),g2)+arrayorigin
        else:
           # Compute a line of points
            grids = np.mgrid[
                g1min:g1max+1,
                ]
            selected_points = np.outer(grids,g1)+arrayorigin
    else:
        selected_points = np.array([[arrayorigin[0],arrayorigin[1]]])
    radii = (selected_points**2).sum(axis=1)**.5
    selected_points = selected_points[
        np.logical_and(
            radii>=rmin,
            radii<=rmax
        )
    ]
    return selected_points

def DDFpointsmask(pointsvector,selectionpoints,tolerance):
    '''
    This makes a Boolean mask for selection of diffraction peaks for DDF imaging from a set of selected
    positions in the reciprocal space plane.  This will work with regular arrangements from 
    generate_DDF_pointselect_array, as well as lists of points from other sources, such as the diffraction points
    extracted from some particular pixel in the dataset.
    
    If there are multiple points, then this will generate 
    multiple masks and the object will be MxN in size, where N is the length of the flattened pointsvector and
    M is the number of masks.  Each mask needs to be separate since multiple diffraction spots may contribute to
    total intensity in a pixel, so all need counting separately and adding and there are multiple contributions
    to the bright pixels

    Parameters
    ----------
    pointsvector: Vector
        Currently must contain fields for rx, ry, kx, ky and intensity
    selectionpoints: np.ndarray
        This will have shape (M,2) and will contain M pairs of kx,ky coordinates
    tolerance: int, float
        This is the tolerance for selection of a peak near any of the selectionpoints
        in whatever units are used for the selectionpoints (will work in pixels or calibrated units)

    Returns
    -------
    maskstack: np.ndarray
        A set of Boolean masks for selecting points.  Each will have the same length as the flattened fields
        in the pointsvector it is to be used on.
    '''
    if 'q_row' in pointsvector.fields:
        fields = ["q_row","q_col"]
    elif 'kx' in pointsvector.fields:
        fields = ["kx","ky"]
    maskstack = np.transpose(
        np.linalg.norm(
            pointsvector.select_fields(*fields).flatten()[:,None,:]-selectionpoints,axis=2
        )<tolerance
    )
    return maskstack

def DDFimage_from_maskstack(pointsvector,maskstack):
    '''
    This calculates a DDF image from a Boolean maskstack object, which is a set of Boolean mask layers.  
    Each mask layer may contribute to some of the same pixels, so calculating each separately is necessary.
    Since we are working with flattened arrays that are selected with a mask, each layer may have a different 
    length and any full stack array would be ragged, thus a short for loop was the simplest approach.

    Ultimately, this mask could be from anywhere. It could be from a single point pointsmask, a multipoint one 
    (whether a regular array or one templated on an experimental datapoint), or one or more Boolean masks from an 
    Unsupervised Clustering output.

    Parameters
    ----------
    pointsvector: Vector
        Must contain fields for rx, ry, kx, ky and intensity
    maskstack: np.ndarray
        This will be M layers, each with a length along axis 1 equal to the length of the flattened data.  dtype must be bool
    Returns
    -------
    im: np.ndarray
        The Digital Dark Field Image
    '''
    im = np.zeros(shape=pointsvector.shape)
    if len(maskstack.shape) == 1:
        maskstack = maskstack[None,:]
    for mask in maskstack:
        rx = pointsvector.select_fields("rx").flatten()[mask].astype(int)
        ry = pointsvector.select_fields("ry").flatten()[mask].astype(int)
        I = pointsvector.select_fields("intensity").flatten()[mask]
        im[rx,ry]+=I
    return im

    import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm

    # ------------------------------------------------------------------ #
    # Clustering Functions
    # ------------------------------------------------------------------ #

def DBSCAN_pointsvector(pointsvector, fields=['kx','ky'], scaling = [1,1], eps=0.5, min_samples=20, plot=True):
    '''
    Runs DBSCAN on selected fields in a pointsvector
    See scikit-learn documentation for general comments
    Experience suggests about eps should be about 0.3-0.5 for detecting diffraction spots in kx,ky 2D
    clustering, and about 1 will connect arcs/rings of spots for nanocrystalline / amorphous materials.
    Too small and you see no clusters.
    For 4D rx,ry,kx,ky clustering, eps needs to be larger, perhaps 5-10, depending on your scaling parameters.
    Alter the relative weighting of real and reciprocal space depending on your dataset and the size of your
    crystals in real space compared to the spacing of diffraction peaks in reciprocal space.

    Parameters
    ----------
    pointsvector: Vector
        Should contain any fields you are selecting to cluster on
    fields: list of str
        Strings in ["rx", "ry", 'kx', "ky", "kr", "kphi"] to cluster on
    scaling: list of int, float
        Relative scaling factors for different dimensions
    eps: float
        As defined in scikit-learn
    min_samples: int
        As defined in scikit-learn
    plot: bool
        Turns plotting on or off
    Returns
    -------
    
    '''
    for item in fields:
        assert item in ["rx", "ry", 'kx', "ky", "kr", "kphi"], "field not found in [rx, ry, kx, ky, kr, kphi]"
    assert len(scaling)==len(fields), "the scalings and fields must have the same number of entries"
    pointsarray = pointsvector.select_fields(*fields).flatten()*np.array(scaling)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(pointsarray)
    if plot:
        plot_L1_clusters_kspace(
            db.labels_, 
            pointsvector, 
            fields, 
            max_kr=int(pointsvector.select_fields('kx').flatten().max()*1.1)
        )
    return db.labels_

'''
A custom colormap for the k-space plots
'''

california = LinearSegmentedColormap.from_list(
    'cali',
    [
        (1,0.5,0),
        (.9,.9,0),
        (0.5,.9,0),
        (0,.9,.9),
        (0,.5,1)
    ],
    
    # bad='gray'
)
california.set_under('lightgrey')
california.set_bad('red')

def plot_L1_clusters_kspace(L1labels, pointsvector, fields, max_kr, cmap=california, figax=None):
    """
    Takes a L1 cluster result of running some cluster algorithm in Scikit-Learn (e.g. DBSCAN)
    on 4D data in a points array and plots the results in reciprocal and real space.  Everything
    is plotted in uncalibrated pixels, since this is just about seeing the results.
    It expects you have applied some radial filtering (although this is not necessary)
    and sets a maximum radius in reciprocal space, purely for visualisation (not calculation)
    purposes.  It also plots the unclustered points in pale grey.
    You can use it for simple inline visualisation in a notebook, or can return a figure for
    saving.

    Parameters
    ----------
    L1labels: np.ndarray
        The labels list from a clustering algorithm
    pointsvector: Vector
        A Vector object from this repo, preferably constructed with ["rx", "ry", 'kx', "ky", "kr", "kphi", "I]
        as the fields
    max_kr: int, float
        maximum radius for the reciprocal space plot
    cmap: colormap
        Either use the default one or provide your own.  Note it needs to be a colormap (not
        just a name of a colormap, such as matplotlib.colormaps["viridis"])
    figax: tuple or None
        If None, a fig and ax are provided.  But you can plot in your own defined axes.
    Returns
    -------
    """
    
    if figax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig, ax = figax
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    ax.set_title("DBSCAN "+", ".join(fields))
    ax.set_xlabel("kx", fontsize=24)
    ax.set_ylabel("ky", fontsize=24)
    ax.set_ylim(max_kr, -max_kr)
    ax.set_xlim(-max_kr, max_kr)

    uniquelabels = np.unique(L1labels)

    kx = pointsvector.select_fields("kx").flatten()
    ky = pointsvector.select_fields("ky").flatten()
    I = pointsvector.select_fields("intensity").flatten()
    kr = pointsvector.select_fields("kr").flatten()
    kphi = pointsvector.select_fields("kphi").flatten()

    ax.scatter(
        ky,kx, 
        s=0.1, alpha=0.2, 
        cmap = cmap, 
        c=L1labels, 
        norm=Normalize(vmin=0, vmax=L1labels.max(), clip=False),
        rasterized=True
    )

    for label in np.unique(L1labels):
        maxint = np.argmax(pointsvector.select_fields("intensity").flatten()[L1labels==label])
        r = kr[L1labels==label][maxint][0] + 6
        ang = np.radians(kphi[L1labels==label][maxint])[0]
        
        labx = np.sin(ang) * r
        laby = np.cos(ang) * r
        ax.annotate(
            label,
            (ky[L1labels==label][maxint][0], kx[L1labels==label][maxint][0]),
            (laby, -labx),
            horizontalalignment="center",
            verticalalignment="center",
            size=7,
        )

def show_L1_clusters_in_real_space(
    L1labels, pointsvector, col=5, gamma=0.25, cmapname='inferno'
):
    """
    Function to show real space plots of L1 clustering outputs

    Parameters
    ----------
    L1labels: np.ndarray
        The labels list from a clustering algorithm
    pointsvector: Vector
        A points array, as defined in py4DSTEM.process.diffraction.digital_dark_field
    cols: int
        number of columns to be used
    gamma: float
        Image gamma.  <1 boosts lower intensities in display.
    cmapname: str
        Must be a valid name for a colormap in matplotlib
    Returns
    -------
    """
    shape = pointsvector.shape
    cluster_list = np.unique(L1labels)[1:]
    l = cluster_list.shape[0]
    ar = shape[1] / shape[0]
    w = 10
    row = int(np.ceil(l / col))
    fig = plt.figure(figsize=(w, w * row / col / ar))
    gs = GridSpec.GridSpec(row, col)
    for n, cluster_label in enumerate(cluster_list):
        i, j = int(n / col), n % col
        ax = plt.subplot(gs[i, j])
        ax.set_axis_off()

        mask = L1labels == cluster_label
        DDFimage_from_maskstack
        selpoints = pointsarray[L1labels == cluster_label]
        im = DDFimage_from_maskstack(pointsvector,mask[None,:])
        ax.imshow(im, norm=colors.PowerNorm(gamma=gamma), cmap=cmapname)
        ax.text(
            5,
            5,
            cluster_label,
            color="w",
            size=14,
            fontweight="bold",
            verticalalignment="top",
        )