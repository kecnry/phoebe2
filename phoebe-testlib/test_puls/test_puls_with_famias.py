import glob
import os
import phoebe
from phoebe.parameters import tools
import numpy as np
import matplotlib.pyplot as plt
from phoebe.utils import utils

basedir = os.path.dirname(os.path.abspath(__file__))

def test_famias():
    """
    Pulsations: comparison with output from FAMIAS
    """
    # read in synthetic line profiles generated by FAMIAS
    files = np.array([os.path.join(basedir,'data/{}.dat'.format(i)) for i in range(301)])
    times = np.loadtxt(os.path.join(basedir,'data/times.dat'), dtype=str, unpack=True)
    times = np.array(times[1], float)
    velocity = np.arange(-150, 151, 3.0)
    wavelength = phoebe.convert('km/s', 'nm', velocity, wave=(4508.288, 'AA'))

    # the line profiles generated by FAMIAS are a normal time series,
    # in the following I phase them up and only select 1 in 10 profiles
    freq = 7.37645
    phases, phased, indices = utils.phasediagram(times, [np.arange(len(times))],
                                                1./freq, return_sortarray=True)
    files = files[indices]
    files = files[::10]
    phases = phases[::10]

    # Put the FAMIAS line profiles in one big matrix, so that they are
    # in a format readable by Phoebe. Add a noise/continuum array.
    flux = []
    for i, ff in enumerate(files):
        x, y = np.loadtxt(ff, unpack=True)
        flux.append(y)
    flux = np.array(flux)
    sigma = 0.001*np.ones_like(flux)
    cont = np.ones_like(flux)
        
    # Create a star with the same properties as was used to generated
    # the FAMIAS line profiles. Add logg and vsini as a parameter
    star = phoebe.PS('star', mass=2.4, teff=6800., incl=55., atm='kurucz',
                    ld_coeffs='kurucz', ld_func='claret', shape='sphere')
    tools.add_surfgrav(star, 3.32, derive='radius')
    tools.add_vsini(star, 106.3, derive='rotperiod')
    mesh = phoebe.PS('mesh:marching')

    # Create and spdep and spobs
    spdep = phoebe.PS('spdep', atm='kurucz', ld_coeffs='kurucz', ld_func='claret',
                    passband='GENEVA.B2',
                    vmicro=8.3, depth=0.72, ref='sp')                  
    spobs = phoebe.SPDataSet(columns=['time', 'wavelength', 'flux', 'sigma', 'continuum'],
                            time=phases/freq, wavelength=wavelength,
                            flux=flux, sigma=sigma, continuum=cont, ref='sp')

    # Define pulsational parameters
    puls = phoebe.PS('puls', freq=freq+1./star.request_value('rotperiod','d'),
                    l=3, m=-1, ampl=0.0045, phase=0.45, k=0.0015,
                    amplteff=0.0, phaseteff=0., amplgrav=0, scheme='coriolis')

    # Create the Star body
    mystar = phoebe.Star(star, mesh=mesh, puls=[puls], pbdep=[spdep], obs=[spobs])

    # Fit or compute
    mystar.compute()

    # Evaluate
    obs, ref = mystar.get_parset(category='sp', type='obs', ref=0)
    syn, ref = mystar.get_parset(category='sp', type='syn', ref=0)
    syn = syn.asarray()
    x = obs['wavelength']
    y1 = obs['flux'] / obs['continuum']
    e_y1 = obs['sigma'] / obs['continuum']
    y2 = syn['flux'] / syn['continuum']
    
    if __name__!="__main__":
        assert((np.abs(y1-y2)/e_y1).max()<=0.9)
    else:
        for i in range(len(y1)):
            plt.plot(x, y1[i]+i/20., 'k-')
            plt.plot(x, y2[i]+i/20., 'r-')
        print (np.abs(y1-y2)/e_y1).max()<=0.9
        print (np.abs(y1-y2)/e_y1).max()


if __name__=="__main__":
    test_famias()
    plt.show()