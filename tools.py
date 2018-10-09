from __future__ import division
import numpy as np
import os

import giapy.plot_tools.interp_path
from amrfile import io as amrio
from giapy.giaflat import thickness_above_floating

def get_fnames(outpath, basename='plot'):
    # List of all output files, following basename
    fnames = [i for i in os.listdir(outpath) if basename in i]
    # Extract the timestep associated, for sorting
    steps = np.array([int(i.split('.')[-3]) for i in fnames])
    # Sort the fnames by timestep
    fnames = [fnames[i] for i in np.argsort(steps)]
    
    return fnames

def extract_ts_and_vols(outpath,skip=1,taf=False):
    fnames = get_fnames(outpath)[::skip]

    vols = []
    ts = []

    for f in fnames:
        amrID = amrio.load(outpath+f)
        lo,hi = amrio.queryDomainCorners(amrID, 0)
        xh,yh,bas1 = amrio.readBox2D(amrID, 0, lo, hi, "Z_base", 0)
        xh,yh,thk1 = amrio.readBox2D(amrID, 0, lo, hi, "thickness", 0)
        ts.append(amrio.queryTime(amrID))
        amrio.freeAll()
        
        if taf:
            thk1 = thickness_above_floating(thk1, bas1)
        vol = np.trapz(np.trapz(thk1, axis=1, x=xh), x=yh)

        vols.append(vol)

    vols = np.array(vols)
    
    return ts, vols

def collect_field(fnames, field_name, outpath='./'):
    flist = []
    for f in fnames:
        amrID = amrio.load(outpath+f)
        lo,hi = amrio.queryDomainCorners(amrID, 0)
        xh,yh,z = amrio.readBox2D(amrID, 0, lo, hi, field_name, 0)

        flist.append(z)
        amrio.free(amrID)
    return np.array(flist)


def intersect_grounding_and_center(fnames, centerline):
    """
    Finds the intersection between the grounding lines in fnames and a
    centerline, using the intersection tool from Sukhbinder Singh.
    """

    xints, yints, ts = [], [], []

    for f in fnames:
        amrID = amrio.load(outpath+f)
        xh,yh,bas = amrio.readBox2D(amrID, 0, lo, hi, "Z_base", 0)
        xh,yh,bot = amrio.readBox2D(amrID, 0, lo, hi, "Z_bottom", 0)

        p = plt.contour(xh, yh, np.abs(bas-bot), levels=[0.1], colors='r');
        glx, gly = p.allsegs[0][0].T
        xint, yint = intersection(glx, gly, centerline.xs, centerline.ys)
        xints.append(xint)
        yints.append(yint)
        ts.append(amrio.queryTime(amrID))

        amrio.free(amrID)

    xints = np.array(xints)
    yints = np.array(yints)
    ts = np.array(ts)
    plt.close()
    
    return ts, xints, yints

def find_centerline():
    xh,yh,xvel = amrio.readBox2D(amrID, 0, lo, hi, "xVel", 0)
    xh,yh,yvel = amrio.readBox2D(amrID, 0, lo, hi, "yVel", 0)
    points = np.array([[0,0]])

    for l in p.lines.get_paths():
        i = 0
        for s in l.iter_segments():
            if i == 0: 
                if s[0] not in points:
                    points = np.vstack([points, s[0]])
            i += 1
 
    center_x, center_y = points.T[0,sl], points.T[1,sl]
    center_d = np.r_[0,np.cumsum(np.sqrt((center_x[1:] - center_x[:-1])**2 + (center_y[1:] - center_y[:-1])**2))]/1000.

    centerline = giapy.plot_tools.interp_path.TransectPath(center_x, center_y, center_d)

    return centerline
