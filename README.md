# Energetics of AGN feedback

The tabulated feedback energetics provided in this repository are run using the version of RAiSE included and the feedback_energetics.py file using the following commands:

```
import feedback_energetics
```

```
active_age = np.arange(5.5, 8.5 + 1e-9, 0.1)
jet_power = np.arange(36, 39 + 1e-9, 0.1)
clusters = feedback_energetics.clusters

feedback_energetics.feedback_tabulator(jet_power, active_age, clusters, axis_ratio=2.83, r=None, alpha=0)
```

The data files can be read and output as a dictionary using the following function (for the mock cluster 'A'):

```
dict = feedback_energetics.feedback_reader('A')
```

The dictionary comprises the fields below with units shown in the comments. The bubble_energy array is ordered (jet_power, active_age, r, theta).

```
jet_power = dict['jet_power'] # log10 W
active_age = dict['active_age'] # log10 years
r = dict['r'] # kpc
theta = dict['theta'] # radians
bubble_energy = dict['bubble_energy'] # Joules/m^-3
```
Contact Ross Turner (ross.turner@utas.edu.au) for any corrections or clarifications.
