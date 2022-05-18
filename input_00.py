########################
## PARAMETER PREAMBLE ##
########################

## Domain definition: Omega_0 and Omega_i
# Omega_0 = building platform
# Omega_1 = powder layers

wid     = 5          # Width of domain [mm]
hei     = 0.25       # Height of powder layer [mm]
hei0    = 2          # Height of building platform [mm] [paper = 2]

nlay    = 2          # Number of powder layers

# <---------- wid ----------> #
#
# ------------------------- # ^
#           Layer 3         # hei
# ------------------------- # ˅
#           Layer 2         #
# ------------------------- #
#           Layer 1         #
# ------------------------- # ^
#                           # |
#       Building platform   # hei0
#                           # |
# ------------------------- # ˅

## Scanning parameters

lenlas  = 5         # Lenght of the laser path [mm]
vel     = 1         # Feed velocity of laser	[mm/s]

## Laser parameters

beta    = 0.5          # switch on/off for laser (in (0,1))
					   # (treatment and idle time) / {max power of laser at beta*tau/2}
ton	    = -1-(-wid/2)  # Time to switch on
toff    = 1-(-wid/2)   # Time to switch off
P0      = 1250         # Laser power
xx3     = 5     	   # Parameter to describe the decay of the exponential in the z-direction


## TIME and MESH Discretization
grf	    = 1         # Global Resolution Factor (space and time)

tau = lenlas/vel	# Time for a deposition [s] {3.33s for lenlas=125 vel=33.33}
num_steps = 25*grf	# Number of time steps between depositions/powder layers
dt	= tau/num_steps	# Time step size

T = tau*nlay      	# Final time


## Process parameters
theta_a = 25 	# Ambient temperature (C)
theta_s = 25 	# Building platform inital temperature (C)
theta_m = 25 	# Powder temperature (C)

## Material properties (constant)
# Heat conductivity
kappa_m = 0.021		# Heat conductivity of material [W.mm-1.K]
kappa_s = 0.021		# Heat conductivity of building platform (substrate) [[W.mm-1.K] [TODO]
# Volumetric heat capacity (constant)
rho_m 	= 0.008		# Density of material [g.mm-3]
rho_s 	= 0.008		# Density of building platform (substrate) [g.mm-3] [TODO]
cp_m 	= 0.5		# Specific heat capacity of material [J.g-1.K-1]
cp_s 	= 0.5		# Specific heat capacity of building platform (substrate) [J.g-1.K-1] [TODO]
volhc_m = rho_m*cp_m# Volumetric Specific heat capacity of material [J.mm-3.K-1]
volhc_s = rho_s*cp_s# Volumetric Specific heat capacity of building platform (substrate) [J.mm-3.K-1]

## Liquid phase
tau_l = 1e-1;
meltingtemp = 1500;

## Solid phase
Ms=400;
Me=20;
tau_s = 1e-1;
