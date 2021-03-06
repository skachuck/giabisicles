from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import giapy.plot_tools.interp_path
from amrfile import io as amrio
from giapy.giaflat import thickness_above_floating
from intersection import intersection
from scipy.interpolate import interp2d

def gen_basemap():
    """Generate the standard projection."""
    m = Basemap(resolution='i',projection='spstere',\
                lat_ts=-71,lon_0=180, boundinglat=-64, ellps='WGS84')
    return m

def pig2proj(x, y, xmin=-5905823.470038407, xmax=0,
                    ymin=-5905823.470038407,ymax=0, 
                    xoff=1707000, yoff=384000):
    """Transform coordinates for Pine Island Glacier example to the standard
    projection coordinates."""
    return -x-0.5*(xmax-xmin)+xoff, -y-0.5*(ymax-ymin)+yoff

def genpigplot(m, xs, ys):
    xmin=-xs.max()+0.5*m.xmin+1707000
    ymin=ys.max()
    xmax=xs.min()
    ymax=ys.min()

    fig, ax = plt.subplots(1,1)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)

    return plt.gca()


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

def collect_field(fnames, field_name, outpath='./', return_ts=False):
    flist = []
    ts = []
    for f in fnames:
        amrID = amrio.load(outpath+f)
        lo,hi = amrio.queryDomainCorners(amrID, 0)
        xh,yh,z = amrio.readBox2D(amrID, 0, lo, hi, field_name, 0)
        ts.append(amrio.queryTime(amrID))

        flist.append(z)
        amrio.free(amrID)
    returnset= np.array(flist),
    if return_ts: returnset+= np.array(ts),
    return returnset


def intersect_grounding_and_center(fnames, centerline, outpath='./',
                                    return_depths=False, return_thks=False):
    """
    Finds the intersection between the grounding lines in fnames and a
    centerline, using the intersection tool from Sukhbinder Singh.


    """

    xints, yints, depths, thks, ts, slps = [], [], [], [], [], []

    thetas = np.arctan(np.gradient(centerline.xs)/np.gradient(centerline.ys))

    for f in fnames:
        amrID = amrio.load(outpath+f)
        lo,hi = amrio.queryDomainCorners(amrID, 0)
        xh,yh,bas = amrio.readBox2D(amrID, 0, lo, hi, "Z_base", 0)
        xh,yh,bot = amrio.readBox2D(amrID, 0, lo, hi, "Z_bottom", 0)
        xh,yh,thk = amrio.readBox2D(amrID, 0, lo, hi, "thickness", 0)

        p = plt.contour(xh, yh, np.abs(bas-bot), levels=[0.1], colors='r');
        glx, gly = p.allsegs[0][0].T
        xint, yint = intersection.intersection(glx, gly, centerline.xs, centerline.ys)
        xint = np.mean(xint)
        yint=np.mean(yint)
        xints.append(xint)
        yints.append(yint)
        ts.append(amrio.queryTime(amrID))
        depths.append(interp2d(xh, yh, bas)(xint, yint))
        thks.append(interp2d(xh, yh, thk)(xint, yint))

        xi = np.argmin(np.abs(xh-xint))
        yi = np.argmin(np.abs(yh-yint))
        slpx = bas[yi, xi+1]-bas[yi, xi-1]/(2*(xh[1]-xh[0]))
        slpy = bas[yi+1, xi]-bas[yi-1, xi]/(2*(yh[1]-yh[0]))

    

        thi = np.argmin((centerline.xs-xint)**2 + (centerline.ys-yint)**2)
        theta = thetas[thi]
      
        slp = slpx*np.cos(theta) + slpy*np.sin(theta)

        slp = np.sqrt(slpx**2 + slpy**2)
        slps.append(slp)

        amrio.free(amrID)

    xints = np.array(xints)
    yints = np.array(yints)
    depths = np.array(depths)
    ts = np.array(ts)
    plt.close()
    
    return_set = ts, xints, yints
    if return_depths: return_set+=depths,
    if return_thks: return_set+=thks,
    return_set += slps,
    return_set = [np.asarray(rs) for rs in return_set]
    return return_set

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
