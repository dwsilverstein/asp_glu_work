#! /usr/bin/env python

from __future__ import division, print_function
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import RectBivariateSpline, SmoothBivariateSpline, InterpolatedUnivariateSpline
from scipy.interpolate import Rbf

def main():
    """\
    Script for plotting ci^2 values from EVB simulations along an US
    reaction coordinate 
    """

    # Number of bins for 2D histogram, first US window, last US window
    # *** The spline function seems to do weird things when I have
    # asymmetric binning of r and ci^2.  I'd keep these the same for
    # now.
    spline = False 
    # Perform boxcar averaging (or not)
    boxcar = True
    # Decide how to normalize the plot.  Either "Max" (normalize to the
    # maximum of the probability density) or "PDF" (construct the 
    # probability density).
    norm = "Max"
    if spline:
        ncbins = 450
        nrbins = 450
    else:
        ncbins = 600
        nrbins = 900
    fbin = 0
    lbin = 36
    # Minimum and maximum distances along the reaction coordinate
    rcmin = 1.0
    rcmax = 10.0 # May need to adjust this!!!

    # Style of plotting errors on the PMF
    error_style = 'dashed'

    # Thickness of the frame
    mpl.rcParams['axes.linewidth'] = 1.25

    # Plot parameters (font, font size, use LaTeX)
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family':'serif', 'serif': ['Times'], 'size': 45})

    # Make a figure object and two subplots with the same (shared) x-axis
    #fig = plt.figure(figsize=(7, 7))
    fig = plt.figure(figsize=(6, 6))
    sub = fig.add_subplot(111)

    # Adjust graph scale (my widescreen monitor makes graphs that are 
    # too large, so I reduce the dimensions)
    #fig.subplots_adjust(left=0.12, right=0.705, bottom=0.1, top=0.8)
    fig.subplots_adjust(left=0.12, right=0.588, bottom=0.1, top=0.8)

    # Axis labels
    sub.set_xlabel(r'RC (\AA)')
    sub.set_ylabel(r'c$_{\mathrm{max}}^2$')

    # Minor tick marks
    xminor = MultipleLocator(0.25)      # Reaction coordinate
    yminor = MultipleLocator(0.05)      # ci^2
    sub.xaxis.set_minor_locator(xminor)
    sub.yaxis.set_minor_locator(yminor)

    # Tick mark thickness
    sub.tick_params('both', length=5, width=1.25, which='major')
    sub.tick_params('both', length=2.5, width=1.25, which='minor')

    # x-axis limits
    sub.set_xlim([0.9,10])
    sub.set_xticks([i+1 for i in range(10)])

    # Bin along the RC and ci^2
    bin_r = np.linspace(rcmin, rcmax, nrbins) 
    bin_ci2 = np.linspace(0.40, 1.0, ncbins)   # Values are always bounded by 0 and 1 

    # Store the 2D histogram
    hist2d = None

    # Collect data from the files storing the RC and ci values
    for i in range(fbin,lbin+1):
        # Filenames as processed by the automate_ci2.sh script
        f = 'RCEC_CI2_BIN_' + str(i)     
        # Collect the data from columns (column 0 is the timestep, 
        # column 1 is the value of the RC, and column 2 is the
        # max ci^2 at that timestep).  We use NumPy's loadtxt which
        # is a fast way to collect data from columns.
        r, c = np.loadtxt(f, usecols=(1,2), unpack=True)

        # Since we "throw away" the first 100 ps when creating the PMF
        # we also throw away the first 100 ps when generating the ci^2
        # distribution.
        if len(r) > 190001:
            # The final 100 ps of my simulations do weird things, so
            # I throw them out as well.
            r = r[10001:190001]
            c = c[10001:190001]
        else:
            r = r[10001:]
            c = c[10001:]

        # Make a 2D histogram (stored in H)
        H, bin_ci2, bin_r = np.histogram2d(c, r, bins=(bin_ci2, bin_r))

        # Since H is a NumPy array, the addition operation acts like
        # adding two matrices.  This allows us to avoid writing loops
        # ourselves.
        if hist2d == None:
            hist2d = H
        else:
            hist2d += H

    # Determine normalization factors
    normfac = [0] * len(hist2d[0])
    for i in range(len(hist2d)):
        for j in range(len(hist2d[i])):
            if norm == "Max":
                if hist2d[i][j] > normfac[j]:
                    normfac[j] = hist2d[i][j]
            elif norm == "PDF":
                normfac[j] += hist2d[i][j]

    # Remove the island pixels
    for i in range(len(hist2d)):
        # Zero the "island" pixels (those that are surrounded by white)
        for j in range(len(hist2d[i])):
            if i != 0 and i != len(hist2d)-1:
                if j != 0 and j != len(hist2d[i])-1:
                    test = [] 
                    if hist2d[i-1][j] == 0: test.append( True )
                    if hist2d[i+1][j] == 0: test.append( True )
                    if hist2d[i][j-1] == 0: test.append( True )
                    if hist2d[i][j+1] == 0: test.append( True )
                    if len(test) == 4: hist2d[i][j] = 0
                    if len(test) == 3: hist2d[i][j] = 0
            elif i == 0:
                hist2d[i][j] = 0
            elif i == len(hist2d)-1:
                hist2d[i][j] = 0
        # Do another pass to verify that the "island" pixels are zeroed,
        # since the above algorithm does not catch all pixels
        for j in range(len(hist2d[i])):
            # Zero the "island" pixels (those that are surrounded by white)
            if i != 0 and i != len(hist2d)-1:
                if j != 0 and j != len(hist2d[i])-1:
                    test = [] 
                    if hist2d[i-1][j] == 0: test.append( True )
                    if hist2d[i+1][j] == 0: test.append( True )
                    if hist2d[i][j-1] == 0: test.append( True )
                    if hist2d[i][j+1] == 0: test.append( True )
                    if len(test) == 4: hist2d[i][j] = 0

    # Boxcar average the data
    if boxcar:
        hist2d_boxcar = smooth_data(hist2d) 

    # Normalize the histogram
    for i in range(len(hist2d)):
        # Zero the "island" pixels (those that are surrounded by white)
        for j in range(len(hist2d[i])):
            if int(normfac[j]) != 0:
                hist2d[i][j] = hist2d[i][j] / normfac[j]

    # Mask where hist2d is zero
    if boxcar:
        histmasked = np.ma.masked_where(hist2d_boxcar==0, hist2d)
    else:
        histmasked = np.ma.masked_where(hist2d==0, hist2d)
    # 3D plot
    # The 'cmap' option controls the color of the 3D plot.  Other
    # options are given at the website (http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps)
    # or we could make our own.

    cdict = {'red': ((0.0, 0.0, 0.0),
                     (0.02, 0.0, 0.0),
                     (0.05, 0.0, 0.0),
                     (0.2, 0.75, 0.75),
                     (1.0, 1.0, 1.0)),
             'green': ((0.0, 0.0, 0.0),
                       (0.02, 0.75, 0.75),
                       (0.05, 1.0, 1.0),
                       (0.2, 0.75, 0.75),
                       (1.0, 0.0, 0.0)),
             'blue': ((0.0, 1.0, 1.0),
                      (0.02, 0.75, 0.75),
                      (0.05, 0.0, 0.0),
                      (0.2, 0.0, 0.0),
                      (1.0, 0.0, 0.0))}
    my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    
    # Decide whether or not to spline the data
    if not spline:
        plt.pcolormesh(bin_r, bin_ci2, histmasked, cmap=my_cmap)
    else:
        # In order for the spline function to be happy, the dimensions of 
        # the histogram (nrbins x ncbins) must match the dimensions of 
        # the x- and y-variables.  The bin variables, bin_r and bin_ci2
        # have an extra value corresponding to one of the endpoints.  
        # To get the right lengths of all arrays and not shift the 
        # location of the features of ci^2, the midpoint of each bin is 
        # taken.
        avg_r = [ ( bin_r[i] + bin_r[i-1] ) / 2 for i in range(1,nrbins) ] 
        avg_ci2 = [ ( bin_ci2[i] + bin_ci2[i-1] ) / 2 for i in range(1,ncbins) ]
        # Use SciPy's function to spline the data
        #spl = RectBivariateSpline(avg_r, avg_ci2, histmasked, kx=3, ky=3)
        spl = RectBivariateSpline(avg_r, avg_ci2, hist2d, kx=3, ky=3)
        # For the splined data, we use more points than we did for
        # making the 2D histogram.
        nr = np.linspace(rcmin, rcmax, 600)
        nci2 = np.linspace(0.4, 1.0, 600)
        nh = spl(nr, nci2)
        # This part doesn't really work as well as we'd like.
        nhmasked = np.ma.masked_where(nh <= 8.0E-3, nh)
        nhmasked /= np.amax(nhmasked)
        plt.pcolormesh(nr, nci2, nhmasked, cmap=my_cmap)

    # This allows us to put a colorbar scale for the z-dimension
    # of the 2D plot. 
    cbar = plt.colorbar(orientation='horizontal')

    # Make a second subplot 
    sub2 = sub.twinx()
    sub2.set_xlim([0.9,10])
    sub2.set_ylabel(r'Free Energy (kcal/mol)', rotation = 270)
    y2minor = MultipleLocator(0.5)      # PMF (Free energy) 
    sub2.yaxis.set_minor_locator(y2minor)
    sub2.set_ylim([-11,1.0])
    sub2.set_yticks([-10,-8,-6,-4,-2,0])

    # Tick mark thickness
    sub2.tick_params('both', length=5, width=1.25, which='major')
    sub2.tick_params('both', length=2.5, width=1.25, which='minor')

    # Collect and plot the PMF.  We need to do this second or the
    # line will be behind the 3D plot.
    fh = 'all.pmf.crt'
    r, pmf = np.loadtxt(fh, usecols=(0,2), unpack=True)

    # Error bars
    fh = 'lst.pmf.crt'
    rl,pmf_l = np.loadtxt(fh,usecols=(0,2),unpack=True)
    
    fh = 'frst.pmf.crt'
    rf,pmf_f = np.loadtxt(fh,usecols=(0,2),unpack=True)
    
    # Spline data
    s = InterpolatedUnivariateSpline(r, pmf)
    rs = np.linspace(r[0], r[-1], 1000)
    pmfs = s(rs)

    sl = InterpolatedUnivariateSpline(rl, pmf_l)
    rls = np.linspace(rl[0], rl[-1], 1000)
    pmfls = sl(rls)
    
    sf = InterpolatedUnivariateSpline(rf, pmf_f)
    rfs = np.linspace(rf[0], rf[-1], 1000)
    pmffs = sf(rfs)

    if error_style == 'error_bar':
        # Use fewer points than used in the splines to do error bars
        reb = np.linspace(r[0], r[-1], 125) 
        pmfeb = s(reb)
        pmfleb = sl(reb)
        pmffeb = sf(reb)
        
        # Error bars
        ebar = []
        ebar.append(pmfeb - pmfleb)
        ebar.append(pmffeb - pmfeb)

        sub2.errorbar( reb, pmfeb, yerr=ebar, ecolor='k',
                       fmt=None, elinewidth=3, markeredgewidth=3)
    if error_style == 'shade':
        sub2.fill_between( rs, pmffs, pmfls, lw=0, facecolor='k', alpha=0.5,)
    sub2.plot(rs, pmfs, 'k', linewidth=3)
    if error_style == 'dashed':
        sub2.plot(rls[2:], pmfls[2:], 'k', linewidth=2, linestyle='--', dashes=(4,3))
        sub2.plot(rfs[2:], pmffs[2:], 'k', linewidth=2, linestyle='--', dashes=(4,3))

    #fig.tight_layout()
    plt.show()

    # Uncomment this line if you want to save a figure
    fig.savefig('pmf_2d_ci2histogram.png', dpi=300, transparent=True, format='png', bbox_inches='tight')

def smooth_data(old_data):
    '''Function for boxcar averaging the ci^2 histogram.'''

    dim1 = len(old_data)
    dim2 = len(old_data[0])
    new_data = np.zeros((dim1,dim2))

    for i in range(dim1):
        if i == 0:
            continue
        elif i == 1:
            continue
        elif i == dim1 - 1:
            continue
        else:
            for j in range(dim2):
                if j == 0:
                    continue
                elif j == dim2-1:
                    continue
                else:
                    new_data[i][j] = ( old_data[i-1][j-1] 
                                     + old_data[i-1][j]
                                     + old_data[i-1][j+1]
                                     + old_data[i][j-1]
                                     + old_data[i][j]
                                     + old_data[i][j+1]
                                     + old_data[i+1][j-1]
                                     + old_data[i+1][j]
                                     + old_data[i+1][j+1] ) / 9

    return new_data

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
