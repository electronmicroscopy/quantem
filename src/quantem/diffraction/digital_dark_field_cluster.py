import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
import matplotlib.gridspec as GridSpec

from tqdm import tqdm

from sklearn.cluster import DBSCAN

from quantem.core.datastructures.vector import Vector

    # ------------------------------------------------------------------ #
    # Create suitable Vector object
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

def make_FullPointsVector_from_pointsarray(pointsarray):
    '''
    For back compatibility, this reads in pointsarray objects made with py4DSTEM
    digital dark field
    x
    Parameters
    ----------
    pointsarray: np.ndarray
        Nx7 array

    Returns
    -------
    pointsvector: Vector
        Containing fields ["rx", "ry", "kx", "ky", "kr", "kphi", "intensity"]

    '''
    Rshape = (int(pointsarray.T[3].max()+1),int(pointsarray.T[4].max()+1))
    print(Rshape)
    pointsvector = Vector.from_shape(
        shape=Rshape,
        fields=("rx", "ry", "kx", "ky", "kr", "kphi", "intensity"),
        units=("pixels", "pixels", "pixels", "pixels", "pixels", "degrees", "counts"),
        name="diffraction_vectors",
    )
 
    for rx in tqdm(range(Rshape[0])):
        for ry in range(Rshape[1]):
            mask = np.logical_and(
                pointsarray.T[3]==rx,
                pointsarray.T[4]==ry,
            )
            kx = pointsarray.T[0][mask]
            ky = pointsarray.T[1][mask]
            kr = pointsarray.T[5][mask]
            kphi = pointsarray.T[6][mask]
            I = pointsarray.T[2][mask]
            
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

    # ------------------------------------------------------------------ #
    # Digital Dark Field Basics
    # ------------------------------------------------------------------ #

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

def DDFrphimask(pointsvector,r,rtol,phi=None,phitol=None):
    '''
    This selects points that fit within a certain radial range, and optionally,
    within a certain azimuthal angle range.  

    In general, the azimuthal angle is defined in the range -180 - 180, so
    selections are recommended in this range.

    Parameters
    ----------
    pointsvector: Vector
        Currently must contain fields for rx, ry, kr, kphi and intensity
    r: int, float
        The reciprocal space radius chosen
    rtol: int, float
        The tolerance on the reciprocal space radius chosen
    phi: None, int, float
        The azimuthal angle chosen (in degrees)
    phitol: int, float
        The tolerance on the azimuthal angle radius chosen (in degrees)

    Returns
    -------
    maskstack: np.ndarray
        A set of Boolean masks for selecting points.  Each will have the same length as the flattened fields
        in the pointsvector it is to be used on.
    '''
    radial_selection = np.abs(pointsvector.select_fields('kr').flatten()-r)<rtol
    if phi is not None:
        assert isinstance(phi, (float, int)), 'phi must be a float or integer'
        assert isinstance(phitol, (float, int)), 'phitol must be a float or integer'
        phi_selection = np.abs(pointsvector.select_fields('kphi').flatten()-phi)<phitol
        additional_phi_selection = np.zeros_like(phi_selection).astype(bool)
        if phi+phitol > 180:
            additional_phi_selection = np.abs(pointsvector.select_fields('kphi').flatten()-phi+360)<phitol
        elif phi-phitol < -180:
            additional_phi_selection = np.abs(pointsvector.select_fields('kphi').flatten()-phi-360)<phitol
        phi_selection = np.logical_or(phi_selection,additional_phi_selection)
        maskstack = np.logical_and(radial_selection,phi_selection)
    else:
        maskstack = radial_selection
    return maskstack.T

def DDFimage_from_mask(pointsvector,mask):
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
    cut = pointsvector.filter_rows(mask)
    im = cut.imgreduce()
    return im


    # ------------------------------------------------------------------ #
    # Clustering Functions
    # ------------------------------------------------------------------ #


def DBSCAN_pointsvector(
    pointsvector, 
    fields=['kx','ky'], 
    scaling = [1,1], 
    eps=0.5, 
    min_samples=20, 
    kr_min = 0,
    kr_max = 1000,
    plot=True
):
    '''
    Runs DBSCAN on selected fields in a pointsvector
    See scikit-learn documentation for general comments on their implementation of the DBSCAN function
    Experience suggests about eps should be about 0.3-0.5 for detecting diffraction spots in kx,ky 2D
    clustering, and about 1 will connect arcs/rings of spots for nanocrystalline / amorphous materials.
    Too small and you see no clusters at all.
    For 4D rx,ry,kx,ky clustering, eps needs to be larger, perhaps 3-10, depending on your scaling parameters.
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
    kr_min, kr_max: int, float
        minimum and maximum peak radii to use for clustering.  Setting min_kr>0 blocks the primary beam, which
        may be sensible.  Values will need adjusting for your data and detector, and whether you are working in
        calibrated units or raw pixels
    plot: bool
        Turns plotting on or off
    Returns
    -------
    pointsvector2: Vector
        A copy of the original Vector, with an additional field for L1labels.  It may be shorter than 
        pointsvector if radial filtering has been applied
    
    '''
    for item in fields:
        assert item in ["rx", "ry", 'kx', "ky", "kr", "kphi"], "field not found in [rx, ry, kx, ky, kr, kphi]"
    assert len(scaling)==len(fields), "the scalings and fields must have the same number of entries"

    # We need to return a new Vector as it is changing length once we select only part of the data
    pointsvector2 = pointsvector.copy()

    # making the mask is obvious
    kr = pointsvector.select_fields("kr").flatten()
    pointsvector2 = pointsvector.filter_rows((kr > kr_min) & (kr < kr_max))
    pointsarray = (np.array(scaling)*pointsvector2.select_fields(*fields).flatten())

    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(pointsarray)
    pointsvector2.add_fields('L1labels',db.labels_)
    if plot:
        plot_L1_clusters_kspace( 
            pointsvector2, 
            fields, 
            kr_max_plot=int(pointsvector2.select_fields('kx').flatten().max()*1.05)
        )
    return pointsvector2

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

def plot_L1_clusters_kspace(pointsvector, fields, kr_max_plot, cmap=california, figax=None):
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
    kr_max_plot: int, float
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
    ax.set_ylim(kr_max_plot, -kr_max_plot)
    ax.set_xlim(-kr_max_plot, kr_max_plot)

    kx = pointsvector.select_fields("kx").flatten()
    ky = pointsvector.select_fields("ky").flatten()
    I = pointsvector.select_fields("intensity").flatten()
    kr = pointsvector.select_fields("kr").flatten()
    kphi = pointsvector.select_fields("kphi").flatten()
    L1labels = pointsvector.select_fields("L1labels").flatten().astype(int)
    uniquelabels = np.unique(L1labels)

    ax.scatter(
        ky,kx, 
        s=0.1, alpha=0.2, 
        cmap = cmap, 
        c=L1labels, 
        norm=Normalize(vmin=0, vmax=L1labels.max(), clip=False),
        rasterized=True
    )

    for label in uniquelabels[1:]:
        clustermask = L1labels==label
        maxint = np.argmax(I[clustermask])
        r = kr[clustermask][maxint] + 3
        ang = np.radians(kphi[clustermask][maxint])
        
        labx = np.sin(ang) * r
        laby = np.cos(ang) * r
        ax.annotate(
            label,
            (ky[clustermask][maxint], kx[clustermask][maxint]),
            (laby, -labx),
            horizontalalignment="center",
            verticalalignment="center",
            size=8,
        )

def show_L1_clusters_in_real_space(
    pointsvector, ncols=5, gamma=0.25, cmapname='inferno', ordering='sequential', save_ims=False
):
    """
    Function to show real space plots of all L1 clustering outputs.  This is designed purely
    for in-line sanity checking, and not for publication quality output so there is no 
    savefig option.  It is likely that in many cases, the output will be verbose and need 
    scrolling through.
    There is an option to return the images themselves as a dict, which is especially useful
    for image similarity based computation of L2 clusters.

    Parameters
    ----------
    L1labels: np.ndarray
        The labels list from a clustering algorithm
    pointsvector: Vector
        A points array, as defined in py4DSTEM.process.diffraction.digital_dark_field
    ncols: int
        number of columns to be used
    gamma: float
        Image gamma.  <1 boosts lower intensities in display.
    cmapname: str
        Must be a valid name for a colormap in matplotlib
    ordering: str
        Either "sequential" for the ordering from the cluster output or "size" for ordering
        by cluster size
    save_ims: bool
        Can turn on return of an image
    Returns
    -------
    imdict: dict
        dictionary with cluster indices as keys and images as np.ndarray
    """
    assert ordering in ["sequential", "size"], "ordering must be either sequential or size"
    L1labels = pointsvector.select_fields("L1labels").flatten().astype(int)
    L1_unique_labels, L1_all_cluster_sizes = np.unique(L1labels, return_counts=True)
    shape = pointsvector.shape
    if ordering == "sequential":
        cluster_list = L1_unique_labels[1:]
    elif ordering == "size":
        cluster_list = L1_unique_labels[1:][np.argsort(L1_all_cluster_sizes[1:])[::-1]]
    
    # Set up aspect ration for plotting
    l = cluster_list.shape[0]
    ar = shape[1] / shape[0]
    w = 10
    row = int(np.ceil(l / ncols))
    
    # Set up plot
    fig = plt.figure(figsize=(w, w * row / ncols / ar))
    gs = GridSpec.GridSpec(row, ncols)
    
    # Do the plotting (and maybe save the images)
    if save_ims:
        ims = []
    for n, cluster_label in enumerate(cluster_list):
        i, j = int(n / ncols), n % ncols
        ax = plt.subplot(gs[i, j])
        ax.set_axis_off()

        mask = L1labels == cluster_label
        im = DDFimage_from_maskstack(pointsvector,mask[None,:])
        ax.imshow(im, norm=PowerNorm(gamma=gamma), cmap=cmapname)
        ax.text(
            5,
            5,
            cluster_label,
            color="w",
            size=14,
            fontweight="bold",
            verticalalignment="top",
        )
        if save_ims:
            ims+=[im]
    if save_ims:
        return np.array(ims)

def cluster_mask(cluster_labels, selected_cluster_labels):
    """
    Makes a mask that selects only the points in a particular cluster.  If applied on an output
    from clustering directly on a Vector object, then it can be used for Digital Dark Field imaging
    with that Vector using "DDFimage_from_maskstack".

    Parameters
    ----------
    cluster_labels: np.ndarray
        The labels list from a clustering algorithm
    selected: int, list of int
        An integer specifying one of the cluster labels in cluster_labels or a list of ints selecting
        more than one cluster
    Returns
    -------
    maskstack: np.ndarray

    """
    for cluster_label in selected_cluster_labels:
        assert cluster_label in cluster_labels, f"{cluster_label} not in the cluster labels"
    maskstack = (cluster_labels in selected_cluster_labels)
    return maskstack

def apply_maskstack_to_Vector(pointsvector,maskstack):
    """
    Applies a mask or stack of masks to a Vector to select one or more cluster components for further
    analysis (e.g. plotting or statistical analysis).  You could apply this to a Vector sampled from the
    original with just some of the fields selected if you do not need the whole thing.

    Parameters
    ----------
    pointsvector: Vector
        The raw Vector that was run through clustering 
    maskstack: np.ndarray
        A single mask or stack of masks selecting one or more clusters
    Returns
    -------
    maskstack: no.ndarray

    """
    assert isinstance(maskstack, np.ndarray), "the maskstack must be a numpy array"
    assert maskstack.shape[-1] == pointsvector.flatten.shape[1], "the mask size does not match the Vector size"
    if len(maskstack.shape) == 1:
        return pointsvector.flatten()[maskstack]
    else:
        mask = maskstack.sum(axis=0).astype(bool)
        return pointsvector.flatten()[mask]

def Cluster_COMs_R(pointsvector, weighted=True):
    """
    Calculates either real space centre of mass (weighted by intensity) or a simplified version with
    no intensity from a specific cluster after running cluster analysis
    with scikit.learn on a pointsarray

    Parameters
    ----------
    pointsvector: Vector
        The raw Vector that was run through L1 clustering.  Must have a column giving the L1labels.

    Returns
    -------
    COMs: np.ndarray
        [COMx,COMy]xNclusters, shape=(N,2)
    """
    assert "L1labels" in pointsvector.fields, "This Vector does not appear to have been clustered"

    rxy = pointsvector.select_fields("rx","ry").flatten()
    I = pointsvector.select_fields("intensity").flatten()
    L1labels = pointsvector.select_fields("L1labels").flatten().astype(int)

    L1_unique_labels = np.unique(L1labels)[1:]

    COMs = np.zeros_like(np.vstack((L1_unique_labels,L1_unique_labels)).T)
    for n, label in enumerate(L1_unique_labels):
        mask = np.squeeze(L1labels==label)
        if weighted:
            COMs[n] = (I * rxy)[mask].sum(axis=0) / I[mask].sum()
        else:
            COMs[n] = (rxy)[mask].sum(axis=0) / (rxy)[mask].shape[0]
    return COMs

def DBSCAN_L2(
    pointsvector,
    eps=5, 
    min_samples=2, 
    plot=True,
    method='COMs'
):
    assert method in ['COMs','Jaccard'], 'method currently restricted to COMs or Jaccard'
    
    db2 = DBSCAN(eps=eps, min_samples=min_samples)

    if method == "COMs":
        COMs = Cluster_COMs_R(pointsvector, weighted=True)
        db2.fit(COMs)

    elif method == "Jaccard":
        L1labels = pointsvector.select_fields("L1labels").flatten().astype(int)
        L1_unique_labels = np.unique(L1labels)
        ims = []
        for L1_label in L1_unique_labels[1:]:
            mask = L1labels == L1_label

        corrs = jaccard_image_similarity(imsarray, plot=False)

    L2_unique_labels, L2_all_cluster_sizes = np.unique(db2.labels_, return_counts=True)
    L2_unique_labels_proper = L2_unique_labels[1:]

    L1labels = np.squeeze(pointsvector.select_fields("L1labels").flatten().astype(int))
    L1_unique_labels_proper = np.unique(L1labels)[1:]

    Rshape = pointsvector.shape

    fig,ax = plt.subplots(1,1, figsize=(12,12*Rshape[0]/Rshape[1]))
    ax.set_ylim(Rshape[0],0)
    ax.set_xlim(0,Rshape[1])

    L1toL2mapping = {-1:-2}
    L2toL1mapping = {}
    for L2cluster in L2_unique_labels:
        L1labels_in_L2cluster = L1_unique_labels_proper[db2.labels_==L2cluster]
        [L1toL2mapping.update({L1label: L2cluster}) for L1label in L1labels_in_L2cluster]
        L2toL1mapping.update({L2cluster:L1labels_in_L2cluster})

    L2labels = [L1toL2mapping[L1label] for L1label in L1labels]
    if 'L2labels' in pointsvector.fields:
        pointsvector.remove_fields('L2labels')
    pointsvector.add_fields('L2labels',L2labels)
        
    if plotCOMs:
        for L2cluster in L2_unique_labels:
            L1labels_in_L2cluster = L2toL1mapping[L2cluster]
            chosenCOMs = COMs[L1labels_in_L2cluster]
            ax.scatter(
                chosenCOMs.T[1],
                chosenCOMs.T[0],
                cmap = california, 
                c=[L2cluster]*chosenCOMs.T[0].shape[0], 
                norm=Normalize(vmin=0, vmax=L2_unique_labels_proper.max(), clip=False),
                rasterized=True
            )
            if L2cluster!=-1:
                ax.text(
                    chosenCOMs.T[1].mean(),
                    chosenCOMs.T[0].mean(),
                    str(L2cluster),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                    path_effects = [
                        path_effects.Stroke(linewidth=3, foreground='w'),
                        path_effects.Normal()
                    ]
                )

def jaccard_image_similarity(imsarray, plot=False):

    imsmask[imsarray>1] = 1
    
    corrs = np.zeros(shape=(imsarray.shape[0],imsarray.shape[0]))
    masks = imsarray > 1
    
    for i, mask in enumerate(masks):
        either = (np.logical_or(masks,mask[np.newaxis,:,:])).sum(axis=(1,2))
        both = (masks*mask[np.newaxis,:,:]).sum(axis=(1,2))
        corrs[i] = both/either
    
    if plot:
        plt.imshow(corrs)
    return corrs