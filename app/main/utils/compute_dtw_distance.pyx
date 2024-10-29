#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:02:18 2019

@author: andrewmorris
"""

cimport cython
import numpy as np
cimport numpy as np
cimport cython.inline

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# np.npy_float32

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

#---------------------------------------------------------------------------------------------------
#
## Fills out DTW cost matrix.
# \param nx - feature vectors in sequence 1
# \param Y - feature vectors in sequence 2
# \param cost_mat - persistant workspace for DTW total-cost matrix
# \param tc - horizontal and vertical transition cost
# \param bwd - beam width in number of frames
# \param tnt - apply 'top and tail' to increase time-offset tolerance
# \param vvd - choice of vec-vec distance measure
# \param find_path - is DTW path required?
# \return path - DTW path coords
# \return distance - DTW distance between sequences X and Y
#
@cython.boundscheck(True) # turn off bounds-checking for entire function
@cython.wraparound(True)  # turn off negative index wrapping for entire function
def compute_dtw_distance(np.ndarray[DTYPE_t, ndim=2] cost_mat,
                         np.ndarray[DTYPE_t, ndim=2] vec_dist,
                         DTYPE_t tc, int bwd, bint tnt, bint find_path):
    
    assert cost_mat.dtype == DTYPE
    assert vec_dist.dtype == DTYPE
    
    cdef DTYPE_t from_ww_cost
    cdef DTYPE_t from_sw_cost
    cdef DTYPE_t from_ss_cost
    cdef DTYPE_t distance
    cdef DTYPE_t aa
    cdef DTYPE_t bb
    cdef DTYPE_t cc
    cdef DTYPE_t vmin
    cdef DTYPE_t tcx
    cdef DTYPE_t tcy
    cdef DTYPE_t temp
    
    cdef ITYPE_t ymin
    cdef ITYPE_t ymax
    cdef ITYPE_t ix
    cdef ITYPE_t iy
    cdef ITYPE_t imin
    cdef ITYPE_t path_len
    cdef ITYPE_t nx
    cdef ITYPE_t ny
    cdef ITYPE_t val0
    

#    # Open diagnostics log file if debugging.
#    fpath = '/Users/andrewmorris/Liopa/Workspace/cython_log.txt'
#    try:
#        f = open(fpath, "w")
#    except Exception as Error:
#        print(Error)
#        raise(Error)

    # Initialise all of cost_mat to np.inf.
    cost_mat.fill(1e10)
    cost_mat[0,0] = 0
    
    nx = vec_dist.shape[0]
    ny = vec_dist.shape[1]
        
    # Fill out DTW matrix.
    distance = 0
    ix = 0
    iy = 0
    cdef float fval = float(ny+1)/nx
    
    for ix in range(1,nx+1):
        if bwd > 0:
            val0 = int(fval*ix)
            ymin = max(   1, val0 - bwd)
            ymax = min(ny+1, val0 + bwd)
        else:
            ymin = 1
            ymax = ny+1
            
        for iy in range(ymin, ymax):
            from_ww_cost = cost_mat[ix-1,  iy]
            if ix > 1 and ix < nx:
                from_ww_cost += tc
            from_ss_cost = cost_mat[ix,  iy-1]
            if iy > 1 and iy < ny:
                from_ss_cost += tc
            from_sw_cost = cost_mat[ix-1,iy-1]
            
            temp = min(from_ss_cost, from_sw_cost)
            cost_mat[ix,iy] = min(from_ww_cost, temp)
            cost_mat[ix,iy] += vec_dist[ix-1,iy-1]

## Please keep next 2 lines, because this 'list comprehension' took me some time to work out, and
## can be usefull if one needs to pass a cython array back to python as a python list.
##    # Return cost_mat as a flat python list.
##    cost_mat_as_flat_list = [v for row in cost_mat[:nx+1] for v in row[:ny+1]]
   
    # Trace back match path.
    # TODO. The top-and-tailing logic in the trace-back, below, could do with careful checking.
    ix = nx
    iy = ny
    distance = cost_mat[ix,iy]
    if find_path:
        path = []
        path.append((ix-1,iy-1))
    else:
        path = None
    path_len = 0
    while ix > 1 or iy > 1:
        if tnt and (ix == 1 or iy == 1):
            break
        if ix > 1 and ix < nx:
            tcx = tc
        else:
            tcx = 0
        if iy > 1 and iy < ny:
            tcy = tc
        else:
            tcy = 0
            
        if ix == 1:
            imin = 1
        elif iy == 1:
            imin = 0
        else:
            aa = cost_mat[ix-1,iy] + tcx
            bb = cost_mat[ix,iy-1] + tcy
            cc = cost_mat[ix-1,iy-1]
            temp = min(bb, cc)
            vmin = min(aa, temp)
            if aa == vmin:
                imin = 0
            elif bb == vmin:
                imin = 1
            else:
                imin = 2
# For whetever reason, both the calls below take FAR longer than setting imin as above.
#            imin = np.argmin([cost_mat[ix-1,iy] + tcx, cost_mat[ix,iy-1] + tcy, cost_mat[ix-1,iy-1]])
#            imin = np.argmin(aa, bb, cc)
            
        if imin == 0:
            ix -= 1
        elif imin == 1:
            iy -= 1
        else:
            ix -= 1
            iy -= 1
        
        if tnt:
            # Reset distance as long as we remain on the x or y limit border.
            if ix == nx or iy == ny:
                distance = cost_mat[ix,iy]
                assert not np.isnan(distance), '2: distance is nan'    # temp
                path_len = 0
                if find_path:
                    path = []
        path_len += 1
#        assert ix >= 1 and iy >= 1, 'ix or iy is < 1 in compute_dtw_distance'
#        if ix < 1 or iy < 1:
#            break
        if find_path:
            path.append((ix-1,iy-1))
    
    if find_path:
        path.reverse()
        
    if tnt:
        # Subtract any distance from path_len that was accumulated along the x or y start border.
        if iy > 1:
            distance -= cost_mat[ix,iy-1]
        elif ix > 1:
            distance -= cost_mat[ix-1,iy]

    distance /= path_len
    
#    f.close() # close log file

    return path, distance, cost_mat[:nx+1, :ny+1], path_len
    
#---------------------------------------------------------------------------------------------------
