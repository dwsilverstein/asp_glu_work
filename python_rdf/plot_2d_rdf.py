#! /usr/bin/env python

from __future__ import print_function
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib as mpl
from scipy.interpolate import griddata
from textwrap import dedent

# Initialize variables (MODIFY AS NEEDED)
fwindow = 0           # First US window
lwindow = 24          # Last US window
fprefix1 = 'BIN_'     # File prefix
fprefix2 = '_2D_RDF_' # Secondary file prefix
spline = True         # Spline the data? 
gspace = 300j         # Grid spacing for spline 
# List of all pairs (these are all of the pairs calculated by
# make_2d_rdf.py).
pairs = [ "CC-HW", "CC-OW", "CC-OAM",                 # CEC-... 
          "CC-CP", "CC-OC", "CC-OP",
          "OAM-HW", "OAM-OW", "OAM-HO", "OAM-OH",     # Amide O-...
          "OAM-HG", "OAM-OG",
          "PCOM-CC", "PCOM-HW", "PCOM-OW",            # Protonated COM-...
          "CP-OW", "CP-HW",                           # Carbon in carboxylic acid
          "OC-OW", "OC-HW",                           # Carbonyl oxygen in carboxylic acid
          "OP-OW", "OP-HW",                           # Hydroxyl oxygen in carboxylic acid
          "DCOM-CC", "DCOM-HW", "DCOM-OW",            # Deprotonated COM-...
          "DCOM-HO", "DCOM-OH", "DCOM-HG", "DCOM-OG",
          "CD-OW", "CD-OH", "CD-OG",                  # Carbon in carboxylate
          "CD-HW", "CD-HO", "CD-HG",                  
          "OD-OW", "OD-OH", "OD-OG",                  # Oxygen in carboxylate
          "OD-HW", "OD-HO", "OD-HG",                  
          "HO-HW", "OH-HW", "HO-OW", "OH-OW",         # Hydronium-...
        ]
# Pairs that exist for all RC bins
p_all = [ "CC-HW", "CC-OW", "CC-OAM",
          "OAM-HW", "OAM-OW",
        ]
# Pairs that exist only for the protonated amino acid
p_prot = [ "PCOM-CC", "PCOM-HW", "PCOM-OW", 
           "CC-CP", "CC-OC", "CC-OP",
           "CP-OW", "CP-HW",
           "OC-OW", "OC-HW",
           "OP-OW", "OP-HW", ]
# Pairs that exist only for the deprotonated amino acid
p_deprot = [ "OAM-HO", "OAM-OH",
             "OAM-HG", "OAM-OG",
             "DCOM-CC", "DCOM-HW", "DCOM-OW",
             "DCOM-HO", "DCOM-OH", "DCOM-HG", "DCOM-OG",
             "CD-OW", "CD-OH", "CD-OG",
             "CD-HW", "CD-HO", "CD-HG",
             "OD-OW", "OD-OH", "OD-OG",
             "OD-HW", "OD-HO", "OD-HG",
             "HO-HW", "OH-HW", "HO-OW", "OH-OW", 
           ]

# Loop through all atom pairs
for pair in pairs:
    print("Processing data for the " + pair + " pair.")
    # Collect the data
    first_loop = True
    total_ts = 0      # Total number of timesteps in each RC bin
    total_gr = 0      # Total RDF in each RC bin
    for i in range(fwindow,lwindow+1):
        # Name of the file
        fname = fprefix1 + str(i) + fprefix2 + pair + '.dat'
        # Get the total number of RC bins.  We use try here because 
        # the pairs don't exist in all of the bins.
        try:
            f = open(fname)
            tmp = f.readline()
            tmp = f.readline()
            nrcbins = int(tmp.split()[6])
            f.close()
        except IOError:
            continue
        # Data in the file is stored in columns: r, RC, g(r), number of
        # timesteps corresponding to the RC bin.
        r,rc,gr,nts = np.loadtxt(fname,usecols=(0,1,2,3),unpack=True) 
        # Since Numpy arrays can be directly multiplied, we use this
        # to our advantage.  Accumulate the number of timesteps and total RDF
        if first_loop:
            total_ts = nts
            total_gr = gr * nts
            first_loop = False
        else:
            total_ts += nts
            total_gr += gr * nts

    # Renormalize the RDF to the total number of timesteps contributing
    # to a particular RC bin.  Since we multiplied g(r) by the number of
    # timesteps contributing to each RC bin above, dividing by total_ts
    # gives a weighted average.
    try:
        for i in range(len(total_gr)):
            if total_ts[i] != 0.0:
                total_gr[i] /= total_ts[i]
    # This protects against situations where total_gr doesn't have anything
    # appended to it. 
    except TypeError:
        continue

    # Output the processed data
    fmt = '{0:10.6f}    {1:10.6f}    {2:10.6f}'
    groutput = 'TOTAL' + fprefix2 + pair + '.dat'
    f = open(groutput, 'w')
    # Spline or don't, it's up to you
    if spline:
        # Reshape copies of the arrays
        r2 = np.reshape(r, (nrcbins, -1))
        rc2 = np.reshape(rc, (nrcbins, -1))
        total_gr2 = np.reshape(total_gr, (nrcbins, -1))

        # Put the r and RC data into one location
        rrc = np.array([ [r2[i][j], rc2[i][j]] for i in range(len(r2)) for j in range(len(r2[i])) ])
        # Reorganize total_gr2
        new_gr = [ total_gr2[i][j] for i in range(len(total_gr2)) for j in range(len(total_gr2[i])) ]
        # Spline the data.  In NumPy's mgrid, specifying the grid spacing
        # as a complex number (with a "j" suffix) causes the spacing to
        # include the final point.  See the definition of gspace above!
        grid_r, grid_rc = np.mgrid[0:9.975:gspace, 1:10:gspace]
        grid_gr = griddata(rrc, new_gr, (grid_r, grid_rc), method='cubic', fill_value=0.0)

        sp_r = [ grid_r[j][i] for i in range(len(grid_r)) for j in range(len(grid_r[0])) ]
        sp_rc = [ grid_rc[j][i] for i in range(len(grid_rc)) for j in range(len(grid_rc[0])) ]
        sp_gr = [ 0.0 if grid_gr[j][i] < 0.0 else grid_gr[j][i] for i in range(len(grid_gr)) for j in range(len(grid_gr[0])) ]
        # The following removes the last few RC bins from the file, since 
        # they are never fully sampled.  For the deprotonated states, we
        # also remove some of the first few RC bins. 
        nblocks = int(len(sp_gr) / len(grid_gr))
        #beginremove = 6 * nblocks
        #if pair in p_deprot:
        #    beginremove = 22 * nblocks
        #endremove = 4 * nblocks
        if pair in p_deprot:
            minrc = 2.0
            maxrc = 7.0
        elif pair in p_prot:
            minrc = 1.1
            maxrc = 7.0
        else:
            minrc = 1.1
            maxrc = 7.0
        for i in range(len(sp_gr)):
            if ( minrc <= sp_rc[i] <= maxrc ):
                print( fmt.format( sp_r[i], sp_rc[i], sp_gr[i] ), file = f )
                if ( i >= nblocks-1 ) and ( i % nblocks == nblocks-1 ):
                    print( '', file = f )
        # Determine the maximum of total_gr
        maxgr = max(sp_gr)
        maxgr = int(maxgr) + 1
    # Non-splined data
    else:
        nblocks = int(len(total_gr) / nrcbins) 
        # The following removes the last few RC bins from the file, since 
        # they are never fully sampled.  For the deprotonated states, we
        # also remove some of the first few RC bins. 
        beginremove = 0
        if pair in p_deprot:
            beginremove = 5 * nblocks
        endremove = 4 * nblocks 
        for i in range(beginremove, len(total_gr) - endremove):
            print( fmt.format( r[i], rc[i], total_gr[i] ), file = f )
            if ( i >= nblocks-1 ) and ( i % nblocks == nblocks-1 ): 
                print( '', file = f )
        # Determine the maximum of total_gr
        maxgr = max(total_gr[beginremove:len(total_gr)-endremove])
        maxgr = int(maxgr) + 1
    f.close()

    # Make a Gnuplot file
    gnufile = "vis_2D_RDF_" + pair + ".gnp"
    picfile = "2D_RDF_" + pair + ".png" 
    string = dedent('''\
                    set encoding iso_8859_1
                    
                    # Font
                    set font "Times-Roman, 30"
                    
                    # Palette (colors are defined using HTML hex notation,
                    # see http://www.w3schools.com/html/html_colors.asp)
                    set palette defined ( 0 '#000090',\\
                                          1 '#000fff',\\
                                          2 '#0090ff',\\
                                          3 '#0fffee',\\
                                          4 '#90ff70',\\
                                          5 '#ffee00',\\
                                          6 '#ff7000',\\
                                          7 '#ee0000',\\
                                          8 '#7f0000')
    
                    # Figure size
                    set size 1,1
    
                    # Tick marks
                    set xtics out
                    set ytics out
                    
                    # Axis labels
                    set xlabel "r (\\305)"
                    set ylabel "CV (\\305)"
                    set zlabel "g(r)" offset 2.5,1,0

                    # Axis range
                    set zrange [-2:{1}]
                    set ztics 0,1

                    # Colorbar range (uncomment as needed)
                    #set cbrange [0:4]
    
                    # Plot the data
                    splot '{0}' noti with pm3d at b, '{0}' using 1:2:3 noti with pm3d

                    pause -1 "Hit Enter for hard copy: ctr-c to quit"
                    
                    set terminal pngcairo size 1200,900 font "Times-Roman,20"
                    set output '{2}'
                    
                    # PLOT 1
                    #
                    replot
                    # end of plot file
                    '''.format(groutput,str(maxgr),picfile))
    f = open(gnufile, 'w')
    print(string, file=f)
    f.close()

## Reshape the r, RC, and total_gr arrays.
#r = np.reshape(r, (nrcbins, -1))
#rc = np.reshape(rc, (nrcbins, -1))
#total_gr = np.reshape(total_gr, (nrcbins, -1))
#
## Remove the last few RC bins (these are always undersampled)
## (MODIFY AS NEEDED)
#nremove = 2
#for i in range(nremove):
#    r = np.delete(r, -1, 0)
#    rc = np.delete(rc, -1, 0)
#    total_gr = np.delete(total_gr, -1, 0)
#
## Gard's colormap
#cdict = {'red': ((0.0, 0.0, 0.0),
#                 (0.02, 0.0, 0.0),
#                 (0.05, 0.0, 0.0),
#                 (0.2, 0.75, 0.75),
#                 (1.0, 1.0, 1.0)),
#         'green': ((0.0, 0.0, 0.0),
#                   (0.02, 0.75, 0.75),
#                   (0.05, 1.0, 1.0),
#                   (0.2, 0.75, 0.75),
#                   (1.0, 0.0, 0.0)),
#         'blue': ((0.0, 1.0, 1.0),
#                  (0.02, 0.75, 0.75),
#                  (0.05, 0.0, 0.0),
#                  (0.2, 0.0, 0.0),
#                  (1.0, 0.0, 0.0))}
#my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
#
#plt.rc('text', usetex=True)
#plt.rc('font', **{'family':'serif', 'serif': ['Times'], 'size': 30})
#
### Put the r and RC data into one location
##rrc = np.array([ [r[i][j], rc[i][j]] for i in range(len(r)) for j in range(len(r[i])) ])
### Reorganize total_gr
##new_gr = [ total_gr[i][j] for i in range(len(total_gr)) for j in range(len(total_gr[i])) ]
### Spline the data
##grid_r, grid_rc = np.mgrid[0:10:150j, 1:10:150j]
##grid_gr = griddata(rrc, new_gr, (grid_r, grid_rc), method='cubic', fill_value=0.0)
##gr_masked = np.ma.masked_where(grid_gr<0, grid_gr)
#
#fig = plt.figure()
#sub = fig.add_subplot(111, projection='3d')
##sub.plot_surface(grid_r, grid_rc, gr_masked, rstride=1, cstride=1, cmap=my_cmap,
##                 linewidth=0, antialiased=False)
#sub.plot_surface(r, rc, total_gr, rstride=1, cstride=1, cmap=my_cmap,
#                 linewidth=0, antialiased=True)
#
## Contours
##cset = sub.contourf(r, rc, total_gr, zdir='z', offset=-1, cmap=cm.coolwarm)
##cset = sub.contourf(r, rc, total_gr, zdir='x', offset=-1, cmap=cm.coolwarm)
##cset = sub.contourf(r, rc, total_gr, zdir='y', offset=12, cmap=cm.coolwarm)
#
##fig.subplots_adjust(left=0.12, right=0.90, bottom=0.1, top=0.9)
#fig.subplots_adjust(left=0.12, right=0.705, bottom=0.1, top=0.7)
#
## Title, labels
#sub.set_xlabel(r'r (\AA)')
#sub.set_ylabel(r'RC (\AA)')
#sub.set_zlabel(r'g(r)')
#
## Axis limits
#sub.set_xticks([1,2,3,4,5,6,7,8,9])
#sub.set_xticklabels([1,2,3,4,5,6,7,8,9])
#sub.set_yticks([1,2,3,4,5,6,7,8,9,10]) 
#sub.set_yticklabels([1,2,3,4,5,6,7,8,9,10]) 
#
### Minor tick marks
##xminor = MultipleLocator(0.1)
##yminor = MultipleLocator(0.5)
##sub.xaxis.set_minor_locator(xminor)
##sub.yaxis.set_minor_locator(yminor)
#
#plt.show()    
#
##fig.savefig('test_2D_RDF.png', transparent=False, format='png', bbox_inches='tight')
