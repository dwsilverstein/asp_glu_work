#! /usr/bin/env python

from __future__ import print_function, division
import sys
from numpy import array, histogram2d, histogram, linspace, pi, sqrt, power, sum, zeros_like
from datetime import datetime

def main():
    """\
    Script for producing RDFs.  The steps are:

    (1) Collect the data from the LAMMPS trajectory (dump) file.

    (2) Collect the data from the CEC coordinate file.

    (3) Calculate distances between the atom centers and bin the data

    (4) Normalize the data based on an ideal system.

    (5) Output RDFs. 
    """

    # This version of the script now requires the file prefix
    if len(sys.argv) != 6:
        error1 = 'Expected 5 arguments but was given ' + str(len(sys.argv)-1)
        nl = '\n'
        tb = '='*len(error1)
        sys.exit(nl + tb + nl + error1 + nl + tb + nl)

    # Check that the first file is a trajectory file.
    test = open(sys.argv[1]).readline().rstrip()
    if test != 'ITEM: TIMESTEP':
        error2 = 'The first file must be a LAMMPS trajectory (dump) file'
        nl = '\n'
        tb = '='*len(error2)
        sys.exit(nl + tb + nl + error2 + nl + tb + nl)

    # Get the command line arguments
    trajfile = sys.argv[1] # LAMMPS dump file
    cecfile = sys.argv[2]  # CEC XYZ file
    fprefix = sys.argv[3]  # Prefix for naming file
    fts = int(sys.argv[4]) # First timestep to collect
    lts = int(sys.argv[5]) # Last timestep to collect

    # Turn on/off the water-water RDFs (this should probably never be
    # turned on unless you like really slow calculations)
    do_water = False

    # This part is specific to Glu and Asp at the moment
    # ----------------------------------------
    # GLU-P or ASP-P with H2O
    # 3 = methylene C next to alpha C
    # 4 = H on alpha carbon (methine group)
    # 5 = methylene C next to carboxylic acid
    # 6 = methyl or methylene H
    # 8 = C carbonyl of amide group
    # 9 = O in carbonyl of amide group
    # 10 = N of amide group
    # 11 = H on polar group (N-H or O-H)
    # 14 = O from -OH group of carboxylic acid
    # 15 = C in methyl group
    # 36 = C in carboxylic acid
    # 37 = carbonyl O of carboxylic acid
    # 38 = Ow
    # 39 = Hw
    # ----------------------------------------
    # GLU-D or ASP-D with H3O+
    # 3 = methylene C next to alpha C
    # 4 = H on alpha carbon (methine group)
    # 5 = methylene C next to carboxylic acid
    # 6 = methyl or methylene H
    # 8 = C carbonyl in amide
    # 9 = O in carbonyl of amide group
    # 10 = N of amide group
    # 11 = H on polar group (N-H or O-H)
    # 15 = C in methyl group
    # 20 = C in COO-
    # 28 = O in COO-
    # 38 = Ow
    # 39 = Hw
    # 40 = O (H3O)
    # 41 = H (H3O)
    # ----------------------------------------

    # List of important atoms
    # ****** Protonated Amino Acid ******
    # Atoms from COOH 
    prot = [ '36', '14', '37' ]
    prot_com = [ '36', '14', '37' ] # Center of mass atoms
    # ****** Deprotonated Amino Acid ******
    # Atoms from COO-
    deprot = [ '20', '28' ]
    deprot_com = [ '20', '28' ] # Center of mass atoms
    # ****** Water ******
    water = [ '38', '39' ]
    # ****** Hydronium ******
    hydronium = [ '40', '41' ]
    # ****** CEC ******
    # This index is simply chosen to not correpsond to any atom indices
    # present in the LAMMPS trajectory file (the value is arbitrary).
    # I choose a negative index since this cannot be chosen in LAMMPS 
    cec = [ '-1' ]
    # ****** Backbone ******
    # WARNING: I haven't come up with a way to distinguish backbone
    # atoms in the protonated and deprotonated amino acids.  Not sure
    # if this is an issue. 
    # All backbone atoms
    #backbone = [ '3', '4', '5', '6', '8', '9', '10',
    #         '11', '15' ]
    # Amide oxygen
    backbone = [ '9' ]

    # Masses for determining the center of mass based on the carboxylate
    # group.
    mass = array([ 12.011, 15.999, 15.999 ])
    totalmass = sum(mass)
    mass = mass / totalmass

    # Atoms to search for
    atomsearch = []
    for i in prot:
        atomsearch.append(i)
    for i in deprot:
        atomsearch.append(i)
    for i in water:
        atomsearch.append(i)
    for i in hydronium:
        atomsearch.append(i)
    for i in backbone:
        atomsearch.append(i)

    # Atom number to atom name correspondence (values are arbitrary, but
    # keys are specific to the system).  Use this for naming files.
    table = { '9'  : 'OAM',  # Amide oxygen in backbone
              '14' : 'OP',   # Protonated oxygen in carboxylic acid
              '36' : 'CP',   # Carbon in carboxylic acid
              '37' : 'OC',   # Carbonyl oxygen in carboxylic acid
              '20' : 'CD',   # Carbon in carboxylate
              '28' : 'OD',   # Oxygen in carboxylate
              '38' : 'OW',   # Oxygen in water
              '39' : 'HW',   # Hydrogen in water
              '40' : 'OH',   # Oxygen in hydronium
              '41' : 'HO',   # Hydrogen in hydronium
              '-1' : 'CC',   # CEC
              '-2' : 'PCOM', # Protonated amino acid center of mass
              '-3' : 'DCOM', # Deprotonated amino acid center of mass
            }

    # Start the timer (for total time)
    startTime = datetime.now() 

    #####################################################
    # Collect data from the LAMMPS trajectory (dump) file
    #####################################################

    # Timer for LAMMPS collection
    lammpsTime = datetime.now()

    # Read the trajectory file into memory.  Store as a tuple to 
    # prevent the user from accidentally changing the values.
    with open(trajfile) as fl:
        f = tuple([line.rstrip() for line in fl])

    # Get the box volume
    x = f[5].split()               # 6th line has x-dimensions
    lx = float(x[1]) - float(x[0])
    y = f[6].split()               # 7th line has y-dimensions
    ly = float(y[1]) - float(y[0])
    z = f[7].split()               # 8th line has z-dimensions
    lz = float(z[1]) - float(z[0])
    # Total volume of the box
    vtotal = lx * ly * lz

    # Side length for minimum image convention
    side = ( lx + ly + lz ) / 3.0
    box_half = side / 2.0

    # Determine what the user put in their dump file
    headings = ( "ITEM: ATOMS id type q x y z",
                 "ITEM: ATOMS id mol type q x y z",
               )
    # Determine how the data should be collected
    collect = -1
    for i in range(len(headings)):
        if f[8] == headings[i]:
            collect = i

    # Get the number of atoms (this is always in the same place in the
    # file)
    natoms = int(f[3])
    # Rather than reading the file, use the fact that the data is 
    # distributed in regular intervals throughout the file.
    noncoordlines = 9 # Ignores the lines defining timesteps and the box
    # Variables for checking that the number of timesteps given is not
    # longer than the number of timesteps that exist.
    filelength = len(f)   
    nts_total = int( filelength / ( natoms + noncoordlines) )
    # Check that the number of timesteps given by the user isn't more
    # than what exists in the file.  If this is true, warn the user
    # and set the last timestep to the total number.
    if lts > nts_total:
        print("Warning!  The last timestep in the range given is larger")
        print("than the total number of timesteps in the file.  I'm")
        print("resetting the last timestep from " + str(lts) + " to " + str(nts_total))
        lts = nts_total
    # If the user gave "-1" to lts, we reset this variable to the
    # total number of timesteps
    elif lts == -1:
        print("Resetting the last timestep from " + str(lts) + " to " + str(nts_total))
        lts = nts_total
    coordblocks = [ [ i * (natoms + noncoordlines) + noncoordlines, 
                      (i + 1) * (natoms + noncoordlines) ] 
                    for i in range(fts,lts) ] 

    # Number of timesteps
    ntimesteps = lts - fts

    # Store the coordinates in a dictionary of lists
    coords = {}
    # Store the number of each atom type in a dictionary
    nattype = {}
    for i in atomsearch:
        coords[i] = [ [] for x in range(ntimesteps) ]
        nattype[table[i]] = 0
    # Gather coordinates from the blocks.  Although we don't explicitly
    # store the timestep, each list corresponds to a timestep
    timestep = 0
    for l in coordblocks:
        s = l[0]
        e = l[1]
        for i in range(s,e):
            # The atom index depends on how the dump file was output
            if collect == 0:
                at = f[i].split()[1]
            elif collect == 1:
                at = f[i].split()[2]
            # Skip the atom if it isn't in one of the lists above 
            # (prot, deprot, water, or hydronium).  This avoids
            # wasting time and memory storing data we don't care about.
            if at in coords.keys():
                # x-, y-, and z- coordinates
                if collect == 0:
                    tmp = [ float(f[i].split()[3]),
                            float(f[i].split()[4]),
                            float(f[i].split()[5]) ]
                elif collect == 1:
                    tmp = [ float(f[i].split()[4]),
                            float(f[i].split()[5]),
                            float(f[i].split()[6]) ]
                coords[at][timestep].append(array(tmp))
                # Get the number of atoms for each type.
                nattype[table[at]] += 1 
        timestep += 1

    # Find timesteps where the (de)protonated amino acid is not
    # present.  This is needed in US windows where the bonding 
    # topology changes at different timesteps.  The code gets 
    # confused otherwise. 
    empty_prot   = [i for i,x in enumerate(coords['36']) if x == []]
    empty_deprot = [i for i,x in enumerate(coords['20']) if x == []]

    # Correct the number of each atomtype
    for key in coords.keys():
        if nattype[table[key]] != 0:
            # Protonated amino acid
            if key in prot:
                nattype[table[key]] /= ntimesteps - len(empty_prot)
            # Deprotonated amino acid
            elif key in deprot:
                nattype[table[key]] /= ntimesteps - len(empty_deprot)
            # Hydronium
            elif key in hydronium:
                nattype[table[key]] /= ntimesteps - len(empty_deprot)
            # Water
            elif key in water:
                nattype[table[key]] /= ntimesteps
            # Backbone atoms
            elif key in backbone:
                nattype[table[key]] /= ntimesteps 
            # Convert to integers
            nattype[table[key]] = int(nattype[table[key]])

    # Check that the number of empty amino acid coordinates sums to
    # the number of timesteps.  Die here if not.
    if ( len(empty_prot) + len(empty_deprot) != ntimesteps ):
        print('There was a problem collecting coordinates.')
        print('Perhaps this is an US window with both protonated and deprotonated species?')
        sys.exit('Coordinate collection error!')

    # Free up memory from storing the trajectory file (we don't need 
    # the file anymore).
    f = 0
    fl.close()

    # Print out the LAMMPS data collection timing
    timing( lammpsTime, datetime.now(), 'LAMMPS collection time:' )

    ###########################################
    # Collect data from the CEC coordinate file 
    ###########################################

    # Timer for CEC collection
    cecTime = datetime.now()

    # Add the CEC label
    atomsearch.append(cec[0])
    nattype['CC'] = 1

    # Read the CEC coordinate file into memory.  Store as a tuple to
    # prevent the user from accidentally changing any values.
    with open(cecfile) as fl:
        f = tuple([line.rstrip() for line in fl])

    # Store the CEC coordinates in the coords dictionary
    ### OLD METHOD
    ###coords[cec[0]] = [ [ array( [ float(f[l].split()[0]),
    ###                              float(f[l].split()[1]),
    ###                              float(f[l].split()[2]) ] ) ] 
    ###                   for l in range(ntimesteps) ]
    ### OLD METHOD
    coords[cec[0]] = [ [ array( [ float(f[l].split()[0]),
                                  float(f[l].split()[1]),
                                  float(f[l].split()[2]) ] ) ] 
                       for l in range(fts,lts) ]

    # Free up memory from storing the CEC coordinate file 
    f = 0
    fl.close()

    # Print out the CEC coordinate collection timing
    timing( cecTime, datetime.now(), 'CEC collection time:' )

    ##############################
    # Determine the center of mass
    ##############################

    # Timer for COM calculation
    comTime = datetime.now()

    # Remove any empty dictionary keys from coords and nattype.  Also 
    # remove atoms from the atomsearch.
    for i in coords.keys():
        if nattype[table[i]] == 0:
            coords.pop(i, None)
            atomsearch.remove(i)
            nattype.pop(table[i], None)

    # Determine the center of mass coordinate
    # ****WARNING: The COM calculation is hardcoded!
    # Error check here to avoid wasting time
    com_error = True
    if ( '36' in atomsearch and
         '14' in atomsearch and
         '37' in atomsearch ):
        com_error = False
        # Add COM to atomsearch
        atomsearch.append('-2')
        # Initialize the number of COM atoms (always 1 in every timestep)
        nattype['PCOM'] = 1
        # List of lists for the COM
        #coords['-2'] = [ [] for x in range(len(coords['36'])) ]
        coords['-2'] = [ [] for x in range(ntimesteps) ]
        # Get the COM from COOH atoms (36, 14, 37).  Recall that
        # we divided by the total mass above.  For the protonated
        # amino acids, there is only one of each type
        #for i in range(len(coords['36'])):
        for i in range(ntimesteps):
            # Check that the carboxylic acid exists in the current
            # timestep.
            if i not in empty_prot:
                # Minimum image distance for C
                cx = coords['36'][i][0][0] - coords['-1'][i][0][0] 
                cy = coords['36'][i][0][1] - coords['-1'][i][0][1] 
                cz = coords['36'][i][0][2] - coords['-1'][i][0][2] 
                if cx >= box_half: coords['36'][i][0][0] -= side
                if cx <= -box_half: coords['36'][i][0][0] += side
                if cy >= box_half: coords['36'][i][0][1] -= side
                if cy <= -box_half: coords['36'][i][0][1] += side
                if cz >= box_half: coords['36'][i][0][2] -= side
                if cz <= -box_half: coords['36'][i][0][2] += side
                # Minimum image distance for O1 
                cx = coords['14'][i][0][0] - coords['-1'][i][0][0]
                cy = coords['14'][i][0][1] - coords['-1'][i][0][1]
                cz = coords['14'][i][0][2] - coords['-1'][i][0][2]
                if cx >= box_half: coords['14'][i][0][0] -= side
                if cx <= -box_half: coords['14'][i][0][0] += side
                if cy >= box_half: coords['14'][i][0][1] -= side
                if cy <= -box_half: coords['14'][i][0][1] += side
                if cz >= box_half: coords['14'][i][0][2] -= side
                if cz <= -box_half: coords['14'][i][0][2] += side
                # Minimum image distance for O2 
                cx = coords['37'][i][0][0] - coords['-1'][i][0][0]
                cy = coords['37'][i][0][1] - coords['-1'][i][0][1]
                cz = coords['37'][i][0][2] - coords['-1'][i][0][2]
                if cx >= box_half: coords['37'][i][0][0] -= side
                if cx <= -box_half: coords['37'][i][0][0] += side
                if cy >= box_half: coords['37'][i][0][1] -= side
                if cy <= -box_half: coords['37'][i][0][1] += side
                if cz >= box_half: coords['37'][i][0][2] -= side
                if cz <= -box_half: coords['37'][i][0][2] += side
                # Construct the COM array
                coords['-2'][i].append( mass[0] * coords['36'][i][0]
                                      + mass[1] * coords['14'][i][0]
                                      + mass[2] * coords['37'][i][0] )
        # Remove the COOH atoms from coords, nattype, and atomsearch
        #for i in [ '36', '14', '37' ]:
        #    coords.pop(i, None)
        #    atomsearch.remove(i)
        #    nattype.pop(table[i], None)
    if ( '20' in atomsearch and
         '28' in atomsearch ):
        com_error = False
        # Add COM to atomsearch
        atomsearch.append('-3')
        # Initialize the number of COM atoms (always 1)
        nattype['DCOM'] = 1
        # Initialize the PCOM coordinates
        coords['-3'] = [ [] for x in range(ntimesteps) ]
        # Get the COM from COO- atoms (20, 28).  Recall that
        # we divided by the total mass above.  For the deprotonated
        # amino acids, there are two carboxylate oxygens. 
        #for i in range(len(coords['20'])):
        for i in range(ntimesteps):
            # Check that the carboxylate exists in the current timestep
            if i not in empty_deprot:
                # Minimum image distance for C
                cx = coords['20'][i][0][0] - coords['-1'][i][0][0]
                cy = coords['20'][i][0][1] - coords['-1'][i][0][1]
                cz = coords['20'][i][0][2] - coords['-1'][i][0][2]
                if cx >= box_half: coords['20'][i][0][0] -= side
                if cx <= -box_half: coords['20'][i][0][0] += side
                if cy >= box_half: coords['20'][i][0][1] -= side
                if cy <= -box_half: coords['20'][i][0][1] += side
                if cz >= box_half: coords['20'][i][0][2] -= side
                if cz <= -box_half: coords['20'][i][0][2] += side
                # Minimum image distance for O1 
                cx = coords['28'][i][0][0] - coords['-1'][i][0][0]
                cy = coords['28'][i][0][1] - coords['-1'][i][0][1]
                cz = coords['28'][i][0][2] - coords['-1'][i][0][2]
                if cx >= box_half: coords['28'][i][0][0] -= side
                if cx <= -box_half: coords['28'][i][0][0] += side
                if cy >= box_half: coords['28'][i][0][1] -= side
                if cy <= -box_half: coords['28'][i][0][1] += side
                if cz >= box_half: coords['28'][i][0][2] -= side
                if cz <= -box_half: coords['28'][i][0][2] += side
                # Minimum image distance for O2 
                cx = coords['28'][i][1][0] - coords['-1'][i][0][0]
                cy = coords['28'][i][1][1] - coords['-1'][i][0][1]
                cz = coords['28'][i][1][2] - coords['-1'][i][0][2]
                if cx >= box_half: coords['28'][i][1][0] -= side
                if cx <= -box_half: coords['28'][i][1][0] += side
                if cy >= box_half: coords['28'][i][1][1] -= side
                if cy <= -box_half: coords['28'][i][1][1] += side
                if cz >= box_half: coords['28'][i][1][2] -= side
                if cz <= -box_half: coords['28'][i][1][2] += side
                # Construct the COM array
                coords['-3'][i].append( mass[0] * coords['20'][i][0]
                                      + mass[1] * coords['28'][i][0]
                                      + mass[2] * coords['28'][i][1] )
        # Remove the COO- atoms from coords, nattype, and atomsearch
        #for i in [ '20', '28' ]:
        #    coords.pop(i, None)
        #    atomsearch.remove(i)
        #    nattype.pop(table[i], None)
    # Exit if the center of mass atoms weren't located.
    if (com_error):
        sys.exit('Could not locate COM atoms')

    # Print out the CEC coordinate collection timing
    timing( comTime, datetime.now(), 'COM calculation time:' )

    ################################################################
    # Calculate distances between atom pairs and generate histograms
    ################################################################

    # Timer for distance calculation 
    distTime = datetime.now()

    # Create pair lists for RDFs
    plist = []
    for i in atomsearch:
        # Make RDFs with water atoms
        if i not in water:
            for j in water:
                plist.append([i,j])
        # For deprotonated amino acid, make RDFs with hydronium also
        if i in deprot:
            for j in hydronium:
                plist.append([i,j])
        # For backbone atoms, make RDFs with hydronium
        if i in backbone and '-3' in atomsearch:
            for j in hydronium:
                plist.append([i,j])
        # For making RDFs between the CEC and atoms in the amino acid
        if i in cec:
            # Behavior of this part is buggy, so I removed it.
            #for j in prot:
            #    if j in atomsearch:
            #        plist.append([i,j])
            #for j in deprot:
            #    if j in atomsearch:
            #        plist.append([i,j])
            for j in backbone:
                if j in atomsearch:
                    plist.append([i,j])
        # COM with CEC and/or hydronium
        if i == '-2':
            plist.append([i,'-1']) # PCOM-CEC
        if i == '-3':
            plist.append([i,'-1']) # DCOM-CEC
            plist.append([i,'40']) # DCOM-O*
            plist.append([i,'41']) # DCOM-H*
    # Add pair lists for water atoms explicitly (this prevents double
    # counting of the OW-HW RDF without requiring a loop).
    if do_water:
        plist.append([water[0],water[0]])
        plist.append([water[0],water[1]])
        plist.append([water[1],water[1]])

    # Make the the histogram dictionary.  Save the histogram of the 
    # RC also, since this tells us the number of timesteps used in each 
    # RC bin.
    hist = {}
    shistrc = {}
    for pair in plist:
        # Here we use the table specified above to rename atoms with
        # letters (e.g. 'OW') instead of atom indices (e.g. '38')
        p = table[pair[0]] + '-' + table[pair[1]]
        hist[p] = 0
        shistrc[p] = 0

    # Minimum and maximum distance for RDF, number of bins.  The
    # maximum distance should be larger than the size of the box
    mindist = 0.0
    maxdist = 20.0
    nbins = 401

    # Bins for histogramming the data (for RDF)
    bin_r = linspace(mindist, maxdist, nbins) 

    # Minimum and maximum distance for RC.  Need to bin up to 11 A
    # or the last window will have issues when the RC gets gets large 
    minrc = 1.0
    maxrc = 11.0
    nrcbins = 51

    # Cutoff value for the histogramming of the RC (made up at the
    # moment).
    rcbin_cutoff = 0

    # Bins for histogramming along the RC
    bin_rc = linspace(minrc, maxrc, nrcbins)

    # Determine the distance between each point and bin the data
    #### DEBUG
    #count = 0
    #### DEBUG
    for pair in plist:
        at1 = pair[0]
        at2 = pair[1]
        p = table[at1] + '-' + table[at2]
        tmp = datetime.now()
        h = 0
        histrc = 0
        # Check if this is the first time the histogram is being added to
        first_add = True
        first_addrc = True
        #### DEBUG
        #count += 1
        #### DEBUG
        for ts in range(ntimesteps):
            # Check if we're dealing with one of the COM coordinates.
            # If so, make sure the coordinate exists during this timestep.
            if ( (ts in empty_prot) and (at1 == '-2') ): continue        # PCOM
            if ( (ts in empty_prot) and (at1 == '14') ): continue        # OP
            if ( (ts in empty_prot) and (at1 == '36') ): continue        # CP
            if ( (ts in empty_prot) and (at1 == '37') ): continue        # OC
            if ( (ts in empty_deprot) and (at1 == '-3') ): continue      # DCOM
            if ( (ts in empty_deprot) and (at1 == '20') ): continue      # CD
            if ( (ts in empty_deprot) and (at1 == '28') ): continue      # OD
            if ( (ts in empty_deprot) and (at1 in hydronium) ): continue
            if ( (ts in empty_deprot) and (at2 in hydronium) ): continue
            ###########################################################
            # Differences in the x-, y-, and z- directions
            mind = []
            for i in range(len(coords[at1][ts])):
                for j in range(len(coords[at2][ts])):
                    # Get differences in x-, y-, and z-directions
                    x = coords[at1][ts][i][0] - coords[at2][ts][j][0]
                    y = coords[at1][ts][i][1] - coords[at2][ts][j][1]
                    z = coords[at1][ts][i][2] - coords[at2][ts][j][2]
                    # Minimum image distances
                    if x > box_half: x -= side
                    if x < -box_half: x += side
                    if y > box_half: y -= side
                    if y < -box_half: y += side
                    if z > box_half: z -= side
                    if z < -box_half: z += side
                    mind.append( x*x + y*y + z*z )
            ###########################################################
            # Get the distance by taking the sqrt of mind.  Do all 
            # square roots together with NumPy, which is faster than
            # doing them as part of the double loop
            mind = sqrt(mind)
            ###########################################################
            # Distance along the reaction coordinate. 
            rcx = rcy = rcz = 0
            # Protonated amino acid
            if '-2' in coords.keys() and coords['-2'][ts] != []:
                rcx = coords['-2'][ts][0][0] - coords['-1'][ts][0][0]
                rcy = coords['-2'][ts][0][1] - coords['-1'][ts][0][1]
                rcz = coords['-2'][ts][0][2] - coords['-1'][ts][0][2]
            # Deprotonated amino acid
            elif '-3' in coords.keys() and coords['-3'][ts] != []:
                rcx = coords['-3'][ts][0][0] - coords['-1'][ts][0][0]
                rcy = coords['-3'][ts][0][1] - coords['-1'][ts][0][1]
                rcz = coords['-3'][ts][0][2] - coords['-1'][ts][0][2]
            # Minimum image distances for the reaction coordinate
            if rcx > box_half: rcx -= side
            if rcx < -box_half: rcx += side
            if rcy > box_half: rcy -= side
            if rcy < -box_half: rcy += side
            if rcz > box_half: rcz -= side
            if rcz < -box_half: rcz += side
            # Reaction coordinate distance (this is constructed in a silly
            # looking way because the length of mind and mindrc must
            # be equal for histogram2d to work).
            mindrc = array([sqrt( rcx*rcx + rcy*rcy + rcz*rcz )]*len(mind))
            #### DEBUG
            #if count == 1:
            #    print(ts)
            #    print(mindrc[0])
            #    print(coords['-3'][ts],coords['-1'][ts])
            #else:
            #    sys.exit()
            #### DEBUG
            ###########################################################
            # Use NumPy's histogram to bin the data
            h, bin_rc, bin_r = histogram2d( mindrc, mind, bins=(bin_rc, bin_r) )
            # Make a running total of the binned data.  Here, we divide
            # by the total number of pairs (sum(h)) to normalize the
            # data for each timestep.  This is used in the number
            # density later
            if first_add:
                hist2d = h / sum(h)
                first_add = False
            else:
                hist2d += h / sum(h)
            ###########################################################
            # Histogram the RC.  This actually shows the number of 
            # timesteps used in each bin of the 2D histogram.
            hrc, bin_rc = histogram( mindrc[0], bins=bin_rc )
            if first_addrc:
                histrc = hrc
                first_addrc = False
            else:
                histrc += hrc 
            ###########################################################
        # OLD CODE 
        ## Divide the histogram by the number of timesteps to get the
        ## average.
        #for i in range(len(hist2d)):
        #    # Only follow through with the division if more than 
        #    # rcbin_cutoff timesteps were used.  Zero the array
        #    # otherwise.
        #    if histrc[i] > rcbin_cutoff:
        #        hist2d[i] = hist2d[i] / histrc[i]
        # OLD CODE 
        hist[p] = hist2d  
        # Store the RC histogram to check the number of timesteps used
        # for the RC bins.  This should be identical for all pairs unless
        # we're in the TS region of the PMF. 
        shistrc[p] = histrc
        # DEBUG 
        #print(histrc)
        #print(sum(histrc), ntimesteps)
        #print(len(hist[table[at1] + '-' + table[at2]]))
        #for i in range(len(hist[table[at1] + '-' + table[at2]])):
        #    print( hist[table[at1] + '-' + table[at2]][i] )
        # DEBUG
        timing( tmp, datetime.now(), p + ':' )
    # Get rid of coords dictionary since it isn't needed anymore
    coords = 0

    # For symmetric atom pairs (e.g. OW-OW or HW-HW), we need to zero
    # the first element of the RDF.  This is a consequence of 
    # calculating distances between all atom pairs, including the atom
    # with itself.  Not doing this will result in a peak at 0.
    if do_water:
        hist['OW-OW'][0] = 0.0
        hist['HW-HW'][0] = 0.0

    # Group the hydronium atoms and water atoms.  Do this for the
    # deprotonated amino acid only.
    if '-3' in atomsearch:
        for i in atomsearch:
            if ( i not in water and 
                 i not in hydronium and
                 i not in cec and 
                 i not in prot and
                 i != '-2' ):
                # Pairs with O (OG is the grouped O)
                p_ow = table[i] + '-' + 'OW'
                p_oh = table[i] + '-' + 'OH'
                p_og = table[i] + '-' + 'OG'
                # Add the histograms together
                hist[p_og] = hist[p_ow] + hist[p_oh]
                # Add the numbers of timesteps together
                shistrc[p_og] = shistrc[p_ow]
                # Pairs with H (HG is the grouped H)
                p_hw = table[i] + '-' + 'HW'
                p_ho = table[i] + '-' + 'HO'
                p_hg = table[i] + '-' + 'HG'
                # Add the histograms together
                hist[p_hg] = hist[p_hw] + hist[p_ho]
                # Add the numbers of timesteps together
                shistrc[p_hg] = shistrc[p_hw]

    # Normalize based on the number of timesteps
    for pair in hist.keys():
        for i in range(len(hist[pair])):
            if shistrc[pair][i] > rcbin_cutoff:
                hist[pair][i] = hist[pair][i] / shistrc[pair][i]

    # Print out the distance calculation timing 
    timing( distTime, datetime.now(), 'Distance calculation time:' )

    ####################
    # Normalize the data
    ####################

    # Timer for normalizing data 
    normTime = datetime.now()

    # RDF is built using the equation:
    #
    # g_{ab}(r) = dn(r) / normfac
    # normfac = rho_{ab} * V_{shell} * N_{ts} 
    # rho_{ab} = N_{ab} / V_{total}
    # V_{shell} = (4/3) * pi * ( r_{i}^3 - r_{i-1}^3 )
    # 
    # Here:
    # dn(r) is the number of particles in a particular bin
    # rho_{ab} is the number density of a-b atom pairs 
    # N_{ab} is the total number of a-b atom pairs in the system
    # V_{total} is the total volume of the simulation box
    # V_{shell} is the volume of a spherical shell
    # N_{ts} is the number of timesteps in the simulation (this was done above)

    # Broadcast the numpy arrays to quickly find shell volumes
    shvol = (4.0/3.0) * pi * ( power(bin_r[1:],3) - power(bin_r[:-1],3) )

    # Number of RC bins
    nrcbins = len(bin_rc) - 1

    for key in hist.keys():
        # Number density of the a-b atom pairs in the RDF.  We already
        # divided by the number density of a-b atom pairs above, so
        # this is just 1 / V_{total}. 
        rho = 1.0 / vtotal
        # Normalization factor
        normfac = rho
        #### DEBUG
        #print(key)
        #### DEBUG
        for j in range(nrcbins):
            for i in range(len(shvol)):
                # hist is now g(r)
                hist[key][j][i] = hist[key][j][i] / ( normfac * shvol[i] )
            #### DEBUG
            #print(hist[key][j])
            #### DEBUG

    # Print out the normalization timing 
    timing( normTime, datetime.now(), 'Normalization time:' )

    #################
    # Output the data
    #################

    # Width of the bins (dr)
    dr = ( maxdist - mindist ) / ( nbins - 1 )

    # Calculate average bin position (midpoints of bins)
    avg_r = [ bin_r[i-1] + dr/2 for i in range(1, len(bin_r)) ] 

    # RC bin width
    drc = ( maxrc - minrc ) / ( nrcbins - 1 )
    
    # Average RC bin position
    avg_rc = [ bin_rc[i-1] + drc/2 for i in range(1, len(bin_rc)) ]

    # Truncate plotting along the interparticle distance (r in g(r))
    trunc = 10.0

    # Output a file for each type of RDF calculated.
    for key in hist.keys():
        # Filename is based on the type of RDF
        fname = fprefix + 'RDF_' + key + '.dat'
        f = open(fname, 'w')
        # Format for output
        fmt = '{0:10.6f}    {1:10.6f}    {2:10.6f}    {3:6}'
        # Print data and header.
        print('# RDF file for ' + key, file=f)
        print('# Number of RC bins = ' + str(nrcbins), file=f)
        print( '# r (Angstrom)  RC (Angstrom)  g(r)     Num. RC TS', 
               file=f ) 
        for j in range(nrcbins):
            for i in range(len(avg_r)): 
                # Only plot r up to a certain distance
                if avg_r[i] < trunc:
                    print( fmt.format( avg_r[i], 
                                       avg_rc[j], 
                                       hist[key][j][i],
                                       shistrc[key][j] ), 
                           file=f )
        f.close()

    # Print out the final timing
    timing( startTime, datetime.now(), 'Total time:' )

def timing( start, current, tstring ):
    '''Output the current timing information.'''
    Time = current - start
    hour_min_sec = str(Time).split(':')
    hour_min_sec[0] = float(hour_min_sec[0])
    hour_min_sec[1] = float(hour_min_sec[1])
    hour_min_sec[2] = float(hour_min_sec[2])
    fmt = '{0:>26} {1:2.0f} {2} {3:2.0f} {4} {5:5.2f} {6}'
    print( fmt.format( tstring,
                       hour_min_sec[0], "hours,",
                       hour_min_sec[1], "minutes, and",
                       hour_min_sec[2], "seconds." ) )

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
