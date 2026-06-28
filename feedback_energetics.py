# feedback_energetics module
# Ross Turner, 25 Jun 2026

# import packages
import numpy as np
import os, warnings
from astropy import constants as const

# import RAiSE package (conditional based on install method)
try:
    from RAiSE import RAiSEHD as RAiSE
except:
    import RAiSEHD as RAiSE


## Define global variables that can be adjusted to customise model output
# basic constants
year = 365.2422*24*3600 # average year in seconds
maverage = (0.6*const.m_p.value) # kg average particle mass
k_B = const.k_B.value # Boltzmann constant
gammaCValue = 5/3. # adiabatic index of lobe/shocked shell/ambient gas
 
# properties of mock clusters consdiered in Turner et al. (2026)
clusters = [['A', 2.41, 144, 0.38, 3.46], \
                ['B', 2.41, 72, 0.38, 1.73], \
                ['C', 4.82, 72, 0.38, 6.92], \
                ['D', 14.46, 28.8, 0.57, 0.692], \
                ['E', 7.23, 72, 0.57, 1.73], \
                ['F', 14.46, 72, 0.57, 3.46], \
                ['G', 4.82, 144, 0.57, 3.46], \
                ['H', 72.3, 28.8, 0.76, 0.692], \
                ['J', 14.46, 72, 0.76, 0.692], \
                ['K', 24.1, 72, 0.76, 1.73]]


## Define functions to calculate spatial distribution of gas heating by AGN feedback
# function to calculate feedback energetics for a given jet power, active age and ambient medium described by a modified beta profile
def feedback_energetics(jet_power, active_age, rho0Value=2.41e-24, r_c=144., beta_prime=0.38, temperature=3.46e7, axis_ratio=2.83, r=None, alpha=0):

    # define (unimportant) variables for RAiSE
    redshift = 0.1
    frequency = 1.5e8
    rand_profile = False
    jet_lorentz = 5
    spectral_index = 0.7
    equipartition = -1.5

    # define ambient medium
    regions = 10**np.arange(0, 4, 0.01)
    betas = (3 * beta_prime * regions**2) / (regions**2 + r_c**2)

    # run RAiSE for the specified jet power, active age and halo mass
    df, _, additional_outputs = RAiSE.RAiSE_run(np.log10(frequency), redshift, axis_ratio, jet_power, active_age, active_age=active_age, rho0Value=rho0Value, regions=regions, betas=betas, temperature=temperature, equipartition=equipartition, angle=0., brightness=False, resolution=None, gravity=True, particle_data=None, additional_outputs=True)

    # unpack additional outputs
    nregions, betas, regions, kValues, temperature, lobe_angles, lobe_lengths, lobe_pressures, shock_lengths, shock_pressures, shock_masses, gravitational_energy = additional_outputs

    # define vector of r and theta
    if r is None or len(r) == 0:
        r = 10**np.arange(0, 3 + 1e-9, 0.02) * const.kpc.value
    dr = 10**((np.log10(r[1]) - np.log10(r[0]))/2)
    angles = np.arange(0, lobe_angles, 1).astype(np.int_)
    dtheta = (np.pi/2)/(lobe_angles - 1)
    theta = dtheta*angles

    # define differential volume element (each hemisphere)
    dchi = 2*np.pi/3.*(np.cos(np.maximum(theta - dtheta/2., 0.)) - np.cos(np.minimum(theta + dtheta/2., np.pi/2)))

    # find correct region for shocked shell
    regionPointers = np.zeros(len(theta)).astype(np.int_)
    for anglePointer in range(0, len(theta)):
        while (regionPointers[anglePointer] + 1 < nregions and shock_lengths[anglePointer,0] > regions[regionPointers[anglePointer] + 1]):
            regionPointers[anglePointer] = regionPointers[anglePointer] + 1
    
    ## ACTIVE PHASE
    # define arrays for AGN energetics
    internal_energy = np.zeros((len(r), lobe_angles))
    cavity_energy = np.zeros((len(r), lobe_angles))
    shell_energy = np.zeros((len(r), lobe_angles))
    
    regionPointer = 0
    for i in range(0, len(r)):
        # calculate the appropriate density profile for current radius
        while (regionPointer + 1 < nregions and r[i] > regions[regionPointer + 1]):
            regionPointer = regionPointer + 1

        # calculate internal energy of the lobe-shocked shell system
        internal_energy[i,:] = (shock_pressures[:,0] - kValues[regionPointer]*r[i]**(-betas[regionPointer])* (k_B*temperature/maverage))/(gammaCValue - 1)
        idx = r[i] > shock_lengths[:,0]
        internal_energy[i,idx] = 0

        # calculate internal energy of the lobe/cavity
        cavity_energy[i,:] = shock_pressures[:,0]/(gammaCValue - 1)
        idx = r[i] > lobe_lengths[:,0]
        cavity_energy[i,idx] = 0
        
        # calculate internal energy of the shocked shell
        shell_energy[i,:] = internal_energy[i,:].copy()
        idx = r[i] <= lobe_lengths[:,0]
        shell_energy[i,idx] = 0

    # convert the energy volume densities to energies (per radial and polar bin)
    internal_energy = internal_energy[:,:]*dchi[None,:] * ((r*dr)**3 - (r/dr)**3)[:,None]
    cavity_energy = cavity_energy[:,:]*dchi[None,:] * ((r*dr)**3 - (r/dr)**3)[:,None]
    shell_energy = shell_energy[:,:]*dchi[None,:] * ((r*dr)**3 - (r/dr)**3)[:,None]

    ## REMNANT PHASE
    # define arrays for AGN energetics
    shocked_shell_energy_1 = np.zeros((len(r), lobe_angles))
    shocked_shell_energy_2 = np.zeros((len(r), lobe_angles))
    
    regionPointer = 0
    for i in range(0, len(r)):
        # calculate the appropriate density profile for current radius
        while (regionPointer + 1 < nregions and r[i] > regions[regionPointer + 1]):
            regionPointer = regionPointer + 1

        # calculate internal energy of the collapsed shocked shell (shocked shell only and shocked shell + cavity)
        shocked_shell_energy_1[i,:] = kValues[regionPointer]*r[i]**(-betas[regionPointer])/(shock_masses[:,0]*3*dchi[:]) * (np.sum(shell_energy[:,:], axis=0) - gravitational_energy[:,0])
        shocked_shell_energy_2[i,:] = kValues[regionPointer]*r[i]**(-betas[regionPointer])/(shock_masses[:,0]*3*dchi[:]) * (np.sum(internal_energy[:,:], axis=0) - gravitational_energy[:,0])
        idx = r[i] > shock_lengths[:,0]
        shocked_shell_energy_1[i,idx] = 0
        shocked_shell_energy_2[i,idx] = 0
        
    ## BUBBLE PHASE
    # define arrays for AGN energetics
    bubble_energy = np.zeros((len(r), lobe_angles))

    # find constants needed for bubble volume and area evolution
    pressure = kValues[regionPointers[:]]*shock_lengths[:,0]**(-betas[regionPointers[:]])*(k_B*temperature/maverage)
    volume = np.sum((gammaCValue - 1)*cavity_energy)/pressure
    C_values = pressure**(1/gammaCValue)*volume

    regionPointer = 0
    for i in range(0, len(r)):
        # calculate the appropriate density profile for current radius
        while (regionPointer + 1 < nregions and r[i] > regions[regionPointer + 1]):
            regionPointer = regionPointer + 1

        # calculate internal energy of the buoyant bubble
        bubble_energy[i,:] = np.sum((gammaCValue + 1)*cavity_energy)*(gammaCValue - 1)*betas[regionPointer]/(gammaCValue*r[i])* (kValues[regionPointer]*r[i]**(-betas[regionPointer])/ (kValues[regionPointers[:]]*shock_lengths[:,0]**(-betas[regionPointers[:]])))**((gammaCValue - 1)/gammaCValue)
        idx = r[i] < shock_lengths[:,0]
        bubble_energy[i,idx] = 0

        # calculate pressure of the buoyant bubble
        pressure = kValues[regionPointer]*r[i]**(-betas[regionPointer])*(k_B*temperature/maverage)
        
        # calculate volume and area (for same axis ratio as shocked shell) of bubble
        volume = (C_values**(1/3.)*pressure**(-1/(3*gammaCValue)) - r[i]*gammaCValue*alpha/(3*gammaCValue - betas[regionPointer]))**3
        cross_section = np.pi*(3*volume/(2*np.pi)/axis_ratio)**(2./3)

        # find polar angles within lobe cross-section
        theta_crit = np.arccos(np.maximum(0, 1 - cross_section/(2*np.pi**2*r[i]**2)))
        bubble_energy[i,:] = bubble_energy[i,:] / (2*np.pi*(1 - np.cos(theta_crit))*r[i]**2)
        idx = theta > theta_crit
        bubble_energy[i,idx] = 0
    
    ## ABLATION (optional output)
    if alpha != 0:
        # define arrays for AGN energetics
        ablation_energy = np.zeros((len(r), lobe_angles))
        
        # find the maximum radius reached by the bubble
        pressure = kValues[regionPointers[:]]*shock_lengths[:,0]**(-betas[regionPointers[:]])*(k_B*temperature/maverage)#/(gammaCValue - 1)
        R_max = shock_lengths[:,0]*(C_values**(1/3.)*(3*gammaCValue - betas[regionPointers[:]])/(shock_lengths[:,0]*gammaCValue*alpha)* pressure**(-1/(3*gammaCValue)))**(3*gammaCValue/(3*gammaCValue - betas[regionPointers[:]]))
        
        regionPointer = 0
        for i in range(0, len(r)):
            # calculate the appropriate density profile for current radius
            while (regionPointer + 1 < nregions and r[i] > regions[regionPointer + 1]):
                regionPointer = regionPointer + 1
    
            # calculate pressure of the buoyant bubble
            pressure = kValues[regionPointer]*r[i]**(-betas[regionPointer])*(k_B*temperature/maverage)
            
            ablation_energy[i,:] = pressure*(C_values**(1/3.)*pressure**(-1/(3*gammaCValue)) - r[i]*gammaCValue*alpha/(3*gammaCValue - betas[regionPointer]))**2* (C_values**(1/3.)*(gammaCValue + 1)*betas[regionPointer]/(r[i]*gammaCValue)*pressure**(-1/(3*gammaCValue)) - alpha*(gammaCValue + 1)*betas[regionPointer]/(3*gammaCValue - betas[regionPointer]))
            idx = np.logical_or(r[i] < shock_lengths[:,0], r[i] > R_max)
            ablation_energy[i,idx] = 0

            # calculate volume and area (for same axis ratio as shocked shell) of bubble
            volume = (C_values**(1/3.)*pressure**(-1/(3*gammaCValue)) - r[i]*gammaCValue*alpha/(3*gammaCValue - betas[regionPointer]))**3
            cross_section = np.pi*(3*volume/(2*np.pi)/axis_ratio)**(2./3)
    
            # find polar angles within lobe cross-section
            theta_crit = np.arccos(np.maximum(0, 1 - cross_section/(2*np.pi**2*r[i]**2)))
            ablation_energy[i,:] = ablation_energy[i,:] / (2*np.pi*(1 - np.cos(theta_crit))*r[i]**2)
            idx = theta > theta_crit
            ablation_energy[i,idx] = 0
    
        return r, dr, theta, dchi, shocked_shell_energy_1 + ablation_energy, shocked_shell_energy_2
    else:
        return r, dr, theta, dchi, shocked_shell_energy_1 + bubble_energy, shocked_shell_energy_2

# function to run feedback energetic and write data to file; list of jet powers, active ages and cluster properties expected
def feedback_tabulator(jet_power, active_age, clusters, axis_ratio=2.83, r=None, alpha=0):
    
    # convert inputs to array/lists if floats
    if not isinstance(jet_power, (list, tuple, np.ndarray)):
        jet_power = [jet_power]
    if not isinstance(active_age, (list, tuple, np.ndarray)):
        active_age = [active_age]
    if len(clusters) > 0 and not isinstance(clusters[0], (list, tuple, np.ndarray)):
        clusters = [clusters]
    
    for n in range(0, len(clusters)):
        # run grid of jet powers and active ages for each cluster environment
        bubble_grid = np.empty((len(jet_power), len(active_age)), dtype=object)
        for i in range(0, len(jet_power)):
            for j in range(0, len(active_age)):
                r_ret, _, theta_ret, _, bubble_energy, _ = feedback_energetics(jet_power[i], active_age[j], rho0Value=clusters[n][1]*1e-24, r_c=clusters[n][2], beta_prime=clusters[n][3], temperature=clusters[n][4]*1e7, axis_ratio=axis_ratio, r=r, alpha=alpha)
                bubble_grid[i,j] = bubble_energy
        
        # convert to efficient data type
        bubble_grid = np.array([[np.array(x)/1e-12 for x in row] for row in bubble_grid], dtype=np.float32)
        
        # average data over polar angle
        #theta_ret = theta[::3]
        #dim0, dim1, dim2, dim3 = bubble_grid.shape
        #averaged_grid = ((bubble_grid[:, :, :, 2:dim3-2].copy()).reshape(dim0, dim1, dim2, -1, 3)).mean(axis=-1)
        #bubble_grid = np.concatenate([bubble_grid[:, :, :, 0:1]/3 + 2*bubble_grid[:, :, :, 1:2]/3, averaged_grid, 2*bubble_grid[:, :, :, (dim3-2):(dim3-1)]/3 + bubble_grid[:, :, :, (dim3-1):dim3]/3], axis=-1)
        
        # save feedback energetics to file
        np.savez_compressed(
        'feedback_energetics_cluster={:}.npz'.format(clusters[n][0]),
            jet_power=np.asarray(jet_power).astype(np.float16),
            active_age=np.asarray(active_age).astype(np.float16),
            r=(r_ret/const.kpc.value).astype(np.float16),
            theta=theta_ret.astype(np.float16),
            bubble_energy=bubble_grid.astype(np.float16)
        )

# function to read-in feedback energetics from existing data files
def feedback_reader(cluster_name):
    
    # read-in data file
    data = np.load('feedback_energetics_cluster={:}.npz'.format(cluster_name), allow_pickle=True)
    
    # return feedback energetics as a dictionary
    return {
        "jet_power": data["jet_power"],
        "active_age": data["active_age"],
        "r": data["r"],
        "theta": data["theta"],
        "bubble_energy": data["bubble_energy"].astype(np.float32)*1e-12
    }
