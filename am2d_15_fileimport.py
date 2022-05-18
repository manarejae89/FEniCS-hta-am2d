"""
"""

from fenics import *
import time
from mshr import *
import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np
import pandas as pd

import importlib
import sys, getopt

#import input_15 as data

set_log_level(30) #https://fenicsproject.org/qa/810/how-to-disable-message-solving-linear-variational-problem/

#parameters["num_threads"] = 2;

########################
## PARAMETER PREAMBLE ##
########################

parameters["reorder_dofs_serial"] = False
parameters["allow_extrapolation"] = True  # V_coarse to V projection


# Omega_0 = Rectangle(Point(-data.wid/2, -data.hei0), Point(data.wid/2, 0))

########################
### STATE DEFINITION ###
########################

def state(i):

    tol	= DOLFIN_EPS
    # # Sub-domain definitions:
    # omega0 = CompiledSubDomain('x[1] <= 0.0 + tol', tol=tol)
    # omega1 = CompiledSubDomain('x[1] >= 0.0 - tol', tol=tol)

    # Mesh and Function Space
    global mesh
    resolution_factor=data.grf+1
    xelem = 25#number of nodes in the x-direction
    yelem = 5#number of nodes in the y-direction per powder layer
    mesh = RectangleMesh(Point(-data.wid/2, -data.hei0), Point(data.wid/2, i*data.hei), resolution_factor*xelem, resolution_factor*(yelem*(int(data.hei0/data.hei)+i)), "crossed")

    global V
    V = FunctionSpace(mesh, 'P', 1)

    # Sub-boundary definitions:
    # Left / top 	= Sigma1
    Sigma1 = CompiledSubDomain('on_boundary && near(x[0], side, tol) && x[1] >= 0', tol=tol, side=-data.wid/2)
    # Right / top 	= Sigma2
    Sigma2 = CompiledSubDomain('on_boundary && near(x[0], side, tol) && x[1] >= 0', tol=tol, side=data.wid/2)
    # Left / bottom	= Sigma01
    Sigma01 = CompiledSubDomain('on_boundary && near(x[0], side, tol) && x[1] <= 0', tol=tol, side=-data.wid/2)
    # Right / bottom= Sigma02
    Sigma02 = CompiledSubDomain('on_boundary && near(x[0], side, tol) && x[1] <= 0', tol=tol, side=data.wid/2)
    # Bottom 		= Gamma_-1
    Gamma_minus1 = CompiledSubDomain('on_boundary && near(x[1], height, tol)', tol=tol, height=-data.hei0)
    # Top 			= Gamma_i
    Gamma1 = CompiledSubDomain('on_boundary && near(x[1], height, tol)', tol=tol, height=i*data.hei)
    # Interphase	= Gamma_0
    Gamma0 = CompiledSubDomain('near(x[1], height, tol)', tol=tol, height=0)

    # MARKING
    # Boundaries
    subboundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) #Facets
    subboundaries.set_all(0)
    Sigma1.mark(subboundaries, 1)		# Left / top 	= Sigma1
    Sigma2.mark(subboundaries, 2)		# Right / top 	= Sigma2
    Sigma01.mark(subboundaries, 3)		# Left / bottom	= Sigma01
    Sigma02.mark(subboundaries, 4)		# Right / bottom= Sigma02
    Gamma_minus1.mark(subboundaries, 5)	# Bottom 		= Gamma_-1
    Gamma1.mark(subboundaries, 6)		# Top 			= Gamma_i
    Gamma0.mark(subboundaries, 7)		# Interphase	= Gamma_0
    ## Save sub-boundaries to VTK files
    #file = File(inputfile + '_results/subboundaries.pvd')
    #file << subboundaries
    ## Save sub-domains to VTK files
    # subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains()) #Cells
    #file = File(inputfile + '_results/subdomains.pvd')
    #file << subdomains
    ## Save material-domains to VTK files
    # materials = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains()) #Cells
    # materials.set_all(1)
    # omega0.mark(materials, 0)
    #file = File(inputfile + '_results/materials.pvd')
    #file << materials

    # Define initial values
    global theta_n, liq_n, sol_n

    if i==1:
        theta_0 = Expression('x[1] <= 0.0 + tol ? theta_s : theta_m', degree=0, tol=tol, theta_s=data.theta_s, theta_m=data.theta_m)
        theta_n = interpolate(theta_0, V)
        liq_0 = Expression('0.0', degree=0)
        liq_n = interpolate(liq_0, V)
        sol_0 = Expression('0.0', degree=0)
        sol_n = interpolate(liq_0, V)
    else:
        theta_n = Function(V)
        liq_n = Function(V)
        sol_n = Function(V)


    # Material properties
    global kappa, volhc
    # Constant heat conductivity
    KAPPA = Expression('x[1] <= 0.0 + tol ? kappa_s : kappa_m', degree=0, tol=tol, kappa_s=data.kappa_s, kappa_m=data.kappa_m)
    kappa = interpolate(KAPPA, V)
    # Volumetric Specific heat capacity
    VOLHC = Expression('x[1] <= 0.0 + tol ? volhc_s : volhc_m', degree=0, tol=tol, volhc_s=data.volhc_s, volhc_m=data.volhc_m)
    volhc = interpolate(VOLHC, V)

    # Define variational problem
    global theta, v, liq, sol
    theta = TrialFunction(V)
    liq = TrialFunction(V)
    sol = TrialFunction(V)
    v = TestFunction(V)

    # Redefine boundary integration measure
    dss = ds(subdomain_data=subboundaries)
    #ds = Measure('ds', domain=mesh, subdomain_data=subboundaries)

    # Collect Robin integrals
    alpha1=5.0	# Left / top 	= Sigma1
    alpha2=5.0	# Right / top 	= Sigma2
    alpha3=1.0	# Left / bottom	= Sigma01
    alpha4=1.0	# Right / bottom= Sigma02
    alpha5=0.0	# Bottom 		= Gamma_-1
    alpha6=1.0	# Top 			= Gamma_i
    # alpha7=0.0# Interphase	= Gamma_0
    alpha=([alpha1,alpha2,alpha3,alpha4,alpha5,alpha6])#,alpha7]
    global integrals_R
    integrals_R = []
    Robin_index = []
    #Robin_index=([1,2])
    Robin_index=([3,4])
    for i in Robin_index:
        integrals_R.append(alpha[i-1]*(theta - data.theta_a)*v*dss(i))


def solver_temp(i):
    #TODO Define phi0(z), phi1(t) , then phi=phi0(z-ih)*phi1(t-(i-1)tau)
    global phi
    #phi = Expression('( t >= (i-1)*tau && t<= (i-1+beta)*tau ) ?  x_1*exp(x_2*(x[1]-i*data.hei)) * exp(-x_3*pow((t-(i-1)*tau)-beta/2*tau, 2)) : 0.0', degree=2, x_1=3185, x_2=10, x_3=100, i=i, h=h, t=t, beta=beta, tau=tau)
    #texp= Expression('( t >= (i-1)*tau && t<= (i-1+beta)*tau ) ? 1 : 0.0', degree=2, i=i,h=h,t=t,beta=beta,tau=tau)
    #phi = Expression('texp * exp(-x_3*pow((t-(i-1)*tau)-beta/2*tau, 2))', texp=texp(t), degree=2, x_1=3185, x_2=10, x_3=100, i=i, h=h, t=t, beta=beta, tau=tau)


    powerfunc= Expression('( t >= (ton+(i-1)*tau) && t<= (toff+(i-1)*tau) ) ? P0 : 0.0', degree=2,i=i,t=t,tau=data.tau,ton=data.ton,toff=data.toff,P0=data.P0)

    phicenter= Expression('( t >= (i-1)*tau && t<= (i)*tau ) ? (-wid/2 + vel*(t-(i-1)*tau)) : 0.0', degree=2,i=i,t=t,tau=data.tau,vel=data.vel,wid=data.wid)

    phiheight= Expression('exp(xx3*(x[1]-i*height))', degree=2,i=i,height=data.hei,xx3=data.xx3)

    phi = Expression('phiheight * powerfunc * exp(-xx1*pow((x[0]-phicenter), 2))', degree=2, phiheight=phiheight, powerfunc=powerfunc, phicenter=phicenter, xx1=100 )

    global phi_source
    phi_source = interpolate(phi, V)
    dt=data.dt

    F = volhc*theta*v*dx + dt*kappa*dot(grad(theta), grad(v))*dx - (volhc*theta_n + dt*phi)*v*dx + sum(integrals_R)
    a, L = lhs(F), rhs(F)

    solve(a == L, thetavar)

    return thetavar

def solver_sol(i):

    Ms=data.Ms;
    Me=data.Me;
    tau_s = data.tau_s;

    s_KM = Expression(' 1-exp(-(Ms-theta)/Me) ', degree=2, Ms=Ms, theta=thetavar, Me=Me )

    sol_equilfunc = Expression(' std::min( s_KM, liq+sol ) ', degree=2, s_KM=s_KM, liq=liq_n, sol=sol_n )

    pospartfunc = Expression('(sol_equilfunc-solid) > 0 ? (sol_equilfunc-solid) : 0.0', degree=2, sol_equilfunc=sol_equilfunc, solid=sol_n)

    global solphaseDOT
    solphaseDOT = Expression('(1/tau)*pospartfunc', degree=2, tau=tau_s, pospartfunc=pospartfunc)

    if i==1:
        solphaseRHS = Expression('solidk == 0 ? solphaseDOT*dt+solidk : solphaseDOT*dt+solidk-dt*liqphaseDOT', degree=2, solphaseDOT=solphaseDOT, dt=data.dt, solidk=sol_n, liqphaseDOT=0)
    else :
        solphaseRHS = Expression('solidk == 0 ? solphaseDOT*dt+solidk : solphaseDOT*dt+solidk-dt*liqphaseDOT', degree=2, solphaseDOT=solphaseDOT, dt=data.dt, solidk=sol_n, liqphaseDOT=liqphaseDOT)
        #solphaseRHS = Expression('solphaseDOT*dt+solidk', degree=2, solphaseDOT=solphaseDOT, dt=data.dt, solidk=sol_n )
        #solphaseRHS = Expression('solidk == 0 ? solphaseDOT*dt+solidk : solphaseDOT*dt+solidk-dt*liqphaseDOT', degree=2, solphaseDOT=solphaseDOT, dt=data.dt, solidk=sol_n, liqphaseDOT=liqphaseDOT)
    solphaseRHS2= Expression('solphaseRHS <0 ? 0.0 : solphaseRHS', degree=2, solphaseRHS=solphaseRHS)

    global solvar
    solvar = interpolate(solphaseRHS2, V)

    return solvar

def solver_liq(i):

    tau_l = data.tau_l;
    meltingtemp = data.meltingtemp;

    #phaseequilfunc = Expression('( theta >= meltingtemp && x[1] >= 0 ) ? 1.0 : 0.0', degree=2, theta=theta, meltingtemp=meltingtemp)
    heaviside = Expression(' theta >= meltingtemp ? 1.0 : 0.0', degree=2, theta=thetavar, meltingtemp=meltingtemp)

    delta=50;
    regularizedheavisidepol = Expression('(10/pow(delta,6))*pow(TT-MT-delta,6) - (24/pow(delta,5)) * pow(TT-MT-delta,5) + (15/pow(delta,4)) * pow(TT-MT-delta,4)',degree=2,TT=thetavar,MT=meltingtemp,delta=delta)
    regularizedheaviside = Expression('((TT-MT-delta)>=delta)+regularizedheavisidepol*((delta>(TT-MT-delta))&((TT-MT-delta)>=0))', degree=2 , TT=thetavar,MT=meltingtemp,delta=delta, regularizedheavisidepol=regularizedheavisidepol)

    liq_equilfunc = Expression(' x[1] >= 0 ? heaviside : 0.0 ', degree=2, heaviside=heaviside)

    pospartfunc = Expression('(phaseequilfunc-phase) > 0 ? (phaseequilfunc-phase) : 0.0', degree=2, phaseequilfunc=liq_equilfunc, phase=liq_n)

    global liqphaseDOT
    liqphaseDOT = Expression('(1/tau)*pospartfunc', degree=2, tau=tau_l, pospartfunc=pospartfunc)

    liqphaseRHS = Expression('liqphaseDOT*dt+liquidk-dt*solphaseDOT', degree=2, dt=data.dt, liqphaseDOT=liqphaseDOT, liquidk=liq_n, solphaseDOT=solphaseDOT)

    global liqvar
    liqvar = interpolate(liqphaseRHS, V)

    return liqvar

def setValueAboveHeight(func, value, height): #set value above height
    func_arr = func.vector().get_local()
    coor = mesh.coordinates()
    for ii in range(mesh.num_vertices()):
        y = coor[ii][1]
        if(y > height):
            func_arr[ii] = value
    func.vector()[:] = func_arr
    return func

def inlineoptions(argv):
    global inputfile, pngplotFlag, vtkFlag
    inputfile = "input_00"
    pngplotFlag = 1
    vtkFlag = 1
    try:
        opts, args = getopt.getopt(argv,"hi:p:v:",["inlinelist="])
    except getopt.GetoptError:
        print('am2d_simulation.py -i <datafile> -p <1|0> -v <1|0>')
        print('am2d_simulation.py -input <datafile> -pngplotOutputFlag <TRUE|FALSE> -vtkOutputFlag <TRUE|FALSE>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('am2d_simulation.py -i <datafile> -p <1|0> -v <1|0>')
            print('am2d_simulation.py -input <datafile> -pngplotOutputFlag <TRUE|FALSE> -vtkOutputFlag <TRUE|FALSE>')
            sys.exit()
        elif opt in ("-i", "--inlinelist"):
            inputfile = arg
        elif opt in ("-p", "--inlinelist"):
            pngplotFlag = int(arg)
        elif opt in ("-v", "--inlinelist"):
            vtkFlag = int(arg)
    print('Data input file is '+inputfile+'.py')
    print('pngFlag is', pngplotFlag)
    print('vtkFlag is', vtkFlag)


if __name__ == '__main__':

    inlineoptions(sys.argv[1:])
    data=importlib.import_module(inputfile)

    # Create vectors for png plots
    if pngplotFlag==1:
        # Time vector
        timevec=ndarray((data.nlay*data.num_steps+1),float)
        # Max temperature vector
        maxtemp=ndarray((data.nlay*data.num_steps+1),float)
        # Integral vectors
        Intpow=ndarray((data.nlay*data.num_steps+1),float)
        Intliq=ndarray((data.nlay*data.num_steps+1),float)
        Intsol=ndarray((data.nlay*data.num_steps+1),float)
        # Sensor definition and sensor vectors
        point1=[0.0,0.0] # Top of building platform
        sensor1temp=ndarray((data.nlay*data.num_steps+1),float)
        point2=[0.0,0.8*data.hei] # First layer
        sensor2temp=ndarray((data.nlay*data.num_steps+1),float)
        sensor2liqu=ndarray((data.nlay*data.num_steps+1),float)
        sensor2soli=ndarray((data.nlay*data.num_steps+1),float)

    # Create VTK files for saving raw solution
    if vtkFlag==1:
        vtkfiletemp = File(inputfile + '_results/temperature.pvd')
        vtkfileliqphase = File(inputfile + '_results/liquid_phase.pvd')
        vtkfilesolphase = File(inputfile + '_results/solid_phase.pvd')
        vtkfilesource = File(inputfile + '_results/source.pvd')

    ###################
    # Loop for layers #
    ###################
    t = 0	# Time variable initialization
    for i in range(1,data.nlay+1):
        print('Layer =', i)

        if i==1:
            state(i)
            ## EXPORT
            # Export VTKs with initial condition
            if vtkFlag==1:
                theta_n.rename("Temperature", "label") # Export vtk with initial temp
                vtkfiletemp << (theta_n, t)
                liq_n.rename("Liquid phase", "label") # Export vtk with initial liq
                vtkfileliqphase << (liq_n, t)
                sol_n.rename("Solid phase", "label") # Export vtk with initial sol
                vtkfilesolphase << (sol_n, t)
                sol_n.rename("Source", "label") # Export vtk with initial source
                vtkfilesource << (sol_n, t)
            # Store initial values for png plots
            if pngplotFlag==1:
                # Vector with time steps for plotting
                timevec[0]= 0.0
                # Store maximum temperature
                maxtemp[0]=	np.array(theta_n.vector()).max()
                # Calculate and store integrals
                Intpow[0] = 0.0		# Vector storing [ Int_Omega phi dx ] for each time step
                Intliq[0] = 0.0		# Vector storing [ Int_Omega liq dx ] for each time step
                Intsol[0] = 0.0		# Vector storing [ Int_Omega sol dx ] for each time step
                # Sensor data
                sensor1temp[0]=theta_n(point1)
                sensor2temp[0]=theta_n(point2)
                sensor2liqu[0]=liq_n(point2)
                sensor2soli[0]=sol_n(point2)

        else:
            state(i)
            #Temperature
            theta_n.interpolate(theta2proj)     # Import last temperature field from previous layer
            setValueAboveHeight(theta_n, data.theta_m, (i-1)*data.hei)  # Initial temperature in new layer is theta_m
            #Liquid phase
            liq_n.interpolate(liq2proj)
            setValueAboveHeight(liq_n, 0.0, (i-1)*data.hei)
            #Solid phase
            sol_n.interpolate(sol2proj)
            setValueAboveHeight(sol_n, 0.0, (i-1)*data.hei)
            ## EXPORT VTKs
            if vtkFlag==1:
                theta_n.rename("Temperature", "label")
                vtkfiletemp << (theta_n, t)
                liq_n.rename("Liquid phase", "label")
                vtkfileliqphase << (liq_n, t)
                sol_n.rename("Solid phase", "label")
                vtkfilesolphase << (sol_n, t)
                phi_source.rename("Source", "label")
                vtkfilesource << (phi_source, t)

        #################
        # Time-stepping #
        #################
        thetavar = Function(V)  # solution variable initialization
        liqvar = Function(V)    # solution variable initialization
        solvar = Function(V)    # solution variable initialization
        for k in range(data.num_steps):
            ##Percentage of computation print
            print('Progress: {:.2%}'.format((k+(i-1)*data.num_steps+1)/(data.nlay*data.num_steps)))
            # Update current time
            t += data.dt

            # Compute solution
            solver_temp(i)
            solver_sol(i)
            solver_liq(i)

            ## EXPORT
            # VTKs
            if (k!=data.num_steps-1 and vtkFlag==1):
                thetavar.rename("Temperature", "label")
                vtkfiletemp << (thetavar, t)
                liqvar.rename("Liquid phase", "label")
                vtkfileliqphase << (liqvar, t)
                solvar.rename("Solid phase", "label")
                vtkfilesolphase << (solvar, t)
                phi_source.rename("Source", "label")
                vtkfilesource << (phi_source, t)
            # png plots
            if pngplotFlag==1:
                # Time vector
                timevec[k+(i-1)*data.num_steps+1]=t
                # Store maximum temperature
                maxtemp[k+(i-1)*data.num_steps+1]=np.array(thetavar.vector()).max()
                # Calculate and store integrals
                Intpow[k+(i-1)*data.num_steps+1] = assemble(phi*dx(domain=mesh))		# Vector storing [ Int_Omega phi dx ] for each time step
                Intliq[k+(i-1)*data.num_steps+1] = assemble(liqvar*dx(domain=mesh))/(data.nlay*data.wid*data.hei) * 100		# Vector storing [ Int_Omega phi dx ] for each time step
                Intsol[k+(i-1)*data.num_steps+1] = assemble(solvar*dx(domain=mesh))/(data.nlay*data.wid*data.hei) * 100		# Vector storing [ Int_Omega phi dx ] for each time step
                # Sensors								 	# Vector with time steps for plotting
                sensor1temp[k+(i-1)*data.num_steps+1]=thetavar(point1)
                sensor2temp[k+(i-1)*data.num_steps+1]=thetavar(point2)
                sensor2liqu[k+(i-1)*data.num_steps+1]=liqvar(point2)
                sensor2soli[k+(i-1)*data.num_steps+1]=solvar(point2)

            # Update previous solution
            theta_n.assign(thetavar)
            liq_n.assign(liqvar)
            sol_n.assign(solvar)
            # Store solution to prepare for new layer
            if k==data.num_steps-1:
                theta2proj=Function(V)
                theta2proj.assign(thetavar)
                liq2proj=Function(V)
                liq2proj.assign(liqvar)
                sol2proj=Function(V)
                sol2proj.assign(solvar)

    ## Create PNG Plots and CSVs
    if pngplotFlag==1:
        # Plot maximum temperature vs time
        plt.plot(timevec, maxtemp, 'b', linewidth=2)
        plt.title('Maximum temperature during printing process')
        plt.xlabel('Time $(s)$')
        plt.ylabel('Temperature $(\degree C)$')
        plt.legend(['$max(\\theta)$'], loc='upper right')
        plt.grid(True)
        plt.savefig(inputfile + '_results/TempMax.png', dpi=320)
        plt.close()
        df = pd.DataFrame({'time' : timevec, 'maxtemperature' : maxtemp })
        df.to_csv(inputfile + '_results/TempMax.csv', index=False)

        # Plot integral of laser power vs time
        plt.plot(timevec, Intpow, 'b', linewidth=2)
        plt.title('Integral of laser power during printing process')
        plt.xlabel('Time $(s)$')
        plt.legend(['$\int_\Omega \phi $d$x$'], loc='upper right')
        plt.grid(True)
        plt.savefig(inputfile + '_results/LaserPowerInt.png', dpi=320)
        plt.close()
        df = pd.DataFrame({'time' : timevec, 'integral_power' : Intpow })
        df.to_csv(inputfile + '_results/LaserPowerInt.csv', index=False)

        # Plot integral of laser power vs time
        plt.plot(timevec, Intliq, 'b', linewidth=2)
        plt.plot(timevec, Intsol, 'r', linewidth=2)
        for i in range(data.nlay): # vertical lines on deposition times
            plt.axvline(x=i*data.num_steps*data.dt, color='k', ls=':', lw=1)
        plt.title('Phases concentration in powder area during printing process')
        plt.xlabel('Time $(s)$')
        plt.ylabel('Phase concentration $(\\%)$')
        plt.legend(['liquid','solid'], loc='upper left')
        #plt.legend(['$\int_\Omega l $d$x$','$\int_\Omega s $d$x$'], loc='upper left')
        plt.grid(True)
        plt.savefig(inputfile + '_results/IntPhases.png', dpi=320)
        plt.close()
        df = pd.DataFrame({'time' : timevec, 'integral_liquid' : Intliq ,
                            'integral_solid' : Intsol })
        df.to_csv(inputfile + '_results/IntPhases.csv', index=False)

        # Plot temperature vs time (Sensor 1)
        plt.plot(timevec, sensor1temp, 'b', linewidth=2)
        plt.title('Temperature at layer 1 during printing process')
        plt.xlabel('Time $(s)$')
        plt.ylabel('Temperature $(\degree C)$')
        plt.legend(['$\\theta$ at sensor in layer $1$'], loc='upper right')
        plt.grid(True)
        plt.savefig(inputfile + '_results/Sensor1_Temp.png', dpi=320)
        plt.close()
        df = pd.DataFrame({'time' : timevec, 'temperature' : sensor1temp })
        df.to_csv(inputfile + '_results/Sensor1_Temp.csv', index=False)

        # Plot max temperature and laser power (2 Subplots)
        fig, axs = plt.subplots(2)
        fig.suptitle('Maximum temperature and laser power during printing process')
        axs[0].plot(timevec, maxtemp, 'b', linewidth=2)
        axs[0].set(ylabel='Temperature $(\degree C)$')
        axs[0].legend(['$max(\\theta)$'], loc='upper right')
        axs[0].grid(True)
        axs[1].plot(timevec, Intpow, 'b', linewidth=2)
        axs[1].set(xlabel='Time $(s)$')
        axs[1].legend(['$\int_\Omega \phi $d$x$'], loc='upper right')
        axs[1].grid(True)
        plt.savefig(inputfile + '_results/TempMax_LaserInt.png', dpi=320)
        plt.close()
        df = pd.DataFrame({'time' : timevec, 'maxtemperature' : maxtemp,
                            'integral_pow' : Intpow })
        df.to_csv(inputfile + '_results/TempMax_LaserInt.csv', index=False)

        # Plot liquid and Solid phase vs temp (2 y-axis)
        fig, axs = plt.subplots()
        fig.suptitle('Phases and temperature at sensor in layer 1 during printing process')
        axs.plot(timevec, sensor2soli, 'g', linewidth=2, label = 'Solid - $s(\\theta)$')
        axs.plot(timevec, sensor2liqu, 'b', linewidth=2, label = 'Liquid - $l(\\theta)$')
        axs.set(xlabel='Time $(s)$')
        axs.set(ylabel='Phase concentration $(\\%)$')
        axs2=axs.twinx()
        axs2.plot(timevec, sensor2temp, 'r', linewidth=2, label = 'Temperature - $\\theta$')
        axs2.set_ylabel('Temperature $(\degree C)$', color="red")
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
        fig.savefig(inputfile + '_results/Sensor2_LiqSolTemp.png', dpi=320)
        plt.close()
        df = pd.DataFrame({'time' : timevec, 'temperature' : sensor2temp,
                            'liquid' : sensor2liqu, 'solid' : sensor2soli })
        df.to_csv(inputfile + '_results/Sensor2_LiqSolTemp.csv', index=False)

        ## Plot liquid phase vs temp (Sensor 2)
        # fig, axs = plt.subplots(2)
        # fig.suptitle('Liquid phase and temperature at sensor in layer 1 during printing process')
        # axs[0].plot(timevec, sensor2liqu, 'b', linewidth=2)
        # axs[0].set(ylabel='Phase concentration $(\\%)$')
        # axs[0].legend(['$l(\\theta)$'], loc='upper right')
        # axs[0].grid(True)
        # axs[1].plot(timevec, sensor2temp, 'b', linewidth=2)
        # axs[1].set(xlabel='Time $(s)$')
        # axs[1].set(ylabel='Temperature $(\degree C)$')
        # axs[1].legend(['$\\theta$'], loc='upper right')
        # axs[1].grid(True)
        # plt.savefig(inputfile + '_results/Sensor2_LiqTemp.png', dpi=320)
        # plt.close()
