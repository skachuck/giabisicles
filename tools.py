def get_fnames(outpath):
    basename = 'plot'

    # List of all output files, following basename
    fnames = [i for i in os.listdir(outpath) if basename in i]
    # Extract the timestep associated, for sorting
    steps = np.array([int(i.split('.')[-3]) for i in fnames])
    # Sort the fnames by timestep
    fnames = [fnames[i] for i in np.argsort(steps)]
    
    return fnames

def extract_ts_and_vols(outpath,skip=1):
    fnames = get_fnames(outpath)[::skip]

    vols = []
    ts = []

    for f in fnames:
        amrID = amrio.load(outpath+f)
        lo,hi = amrio.queryDomainCorners(amrID, 0)
        #xh,yh,topg0 = amrio.readBox2D(amrID_raw, 0, lo, hi, "topography", 0)
        xh,yh,thk1 = amrio.readBox2D(amrID, 0, lo, hi, "thickness", 0)
        ts.append(amrio.queryTime(amrID))
        amrio.freeAll()
        
        vol = np.trapz(np.trapz(thk1, axis=1, x=xh), x=yh)

        vols.append(vol)

    vols = np.array(vols)
    
    return ts, vols

def collect_field(fnames, field_name):
    flist = []
    for f in fnames:
        amrID = amrio.load(outpath+f)
        lo,hi = amrio.queryDomainCorners(amrID, 0)
        xh,yh,z = amrio.readBox2D(amrID, 0, lo, hi, field_name, 0)

        flist.append(z)
        amrio.free(amrID)
    return np.array(flist)

def intersection(x1,y1,x2,y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=np.diff(np.c_[x1,y1],axis=0)
    dxy2=np.diff(np.c_[x2,y2],axis=0)

    T=np.zeros((4,n))
    AA=np.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=np.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=np.NaN


    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]


# In[42]:


def intersect_grounding_and_center(fnames, centerline):

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

import giapy.plot_tools.interp_path

    xh,yh,xvel = amrio.readBox2D(amrID, 0, lo, hi, "xVel", 0)
    xh,yh,yvel = amrio.readBox2D(amrID, 0, lo, hi, "yVel", 0)
# In[110]:


#sl = np.s_[390:-15]

center_x, center_y = points.T[0,sl], points.T[1,sl]
center_d = np.r_[0,np.cumsum(np.sqrt((center_x[1:] - center_x[:-1])**2 + (center_y[1:] - center_y[:-1])**2))]/1000.

centerline = giapy.plot_tools.interp_path.TransectPath(center_x, center_y, center_d)

points = np.array([[0,0]])

for l in p.lines.get_paths():
    i = 0
    for s in l.iter_segments():
        if i == 0: 
            if s[0] not in points:
                points = np.vstack([points, s[0]])
        i += 1
 
