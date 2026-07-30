import numpy as np

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
        This will be M layers, each with a length along axis 1 of the flattened data.  dtype must be bool.
    Rshape: tuple
        This is the shape for the final image
    Returns
    -------
    im: np.ndarray


    
    '''
    im = np.zeros(shape=pointsvector.shape)
    for mask in maskstack:
        rx = pointsvector.select_fields("rx").flatten()[mask].astype(int)
        ry = pointsvector.select_fields("ry").flatten()[mask].astype(int)
        I = pointsvector.select_fields("intensity").flatten()[mask]
        im[rx,ry]+=I
    return im