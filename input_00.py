########################
## PARAMETER PREAMBLE ##
########################

## Domain definition: Omega_0 and Omega_i
#l	= 0.75				# Width of deposition [mm]
#h	= 0.2				# Height of deposition [mm]
#h0	= 0.4			# Height of building platform [mm] [paper = 2]
l	= 5				# Width of deposition [mm]
h	= 0.25				# Height of deposition [mm]
h0	= 2			# Height of building platform [mm] [paper = 2]

n 	= 3			# Number of floors of the structure ##############
					# (number of passes of the laser)

## Scanning

lenlas = 5			# Lenght of the laser path [mm]
vel = 1				# Feed velocity of laser	[mm/s]

## Laser parameter

beta= 0.5			# switch on/off for laser (in (0,1))
					# (treatment and idle time) / {max power of laser at beta*tau/2}
#ton	= -1-(-l/2)
#toff= 1-(-l/2)
ton	= -1-(-5/2)
toff= 1-(-5/2)
P0	= 1250
xx3	= 5	# Parameter to describe the decay of the exponential in the z-direction


# TIME and MESH Discretization
grf	= 1		#global_resolution_factor

tau = lenlas/vel	# Time for a deposition [s] {3.33s for lenlas=125 vel=33.33}
num_steps = 25*grf	# number of time steps for a deposition/powder layer
dt	= tau/num_steps	# time step size

T = tau*n      		# final time


## Process parameters
theta_a = 25 	# Ambient temperature (C)
theta_s = 25 	# Building platform inital temperature (C)
theta_m = 25 	# Molten powder temperature (C) = 2200 K
#theta_m = 1927 	# Molten powder temperature (C) = 2200 K
# theta_a = 0.0 	# Ambient temperature (C)
# theta_s = -2000.0 	# Building platform inital temperature (C)
# theta_m = 2000.0 	# Molten powder temperature (C) = 2200 K


# Material properties (constant)
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
