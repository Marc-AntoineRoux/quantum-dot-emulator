# -*- coding: utf-8 -*-
"""
Created on Tue Oct  20 21:59:45 2020

@author: Marc-Antoine Roux

See 10.1103/RevModPhys.75.1
See http://savoirs.usherbrooke.ca/handle/11143/5054
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

# Single dot functions
def Ng(Vg, Cg=1, e=1):
    """

    Parameters
    ----------
    Vg : Float
        Gate voltage.
    Cg : Float, optional
        Gate capacitance. The default is 1.
    e : Float, optional
        Elementary charge. The default is 1.

    Returns
    -------
    Float
        Dimensionless gate charge.

    """
    return e*Cg*Vg

def U(N, Vg, Cs=0.1, Cg=1, Cd=0.1, e=1):
    """

    Parameters
    ----------
    N : Int
        Number of electrons on the dot.
    Vg : Float
        Gate voltage.
    Cs : Float, optional
        Capacitance between dot and source. The default is 0.1.
    Cg : Float, optional
        Gate capacitance. The default is 1.
    Cd : Float, optional
        Capacitance between dot and drain. The default is 0.1.
    e : Float, optional
        Elementary charge. The default is 1.

    Returns
    -------
    Float
        Electrostatic energy of the single dot.

    """
    # Charging energy
    Ec = e**2/(Cs+Cg+Cd)
    return 0.5*Ec*(N-Ng(Vg, Cg, e))**2

def U_moy(Vg, Cs=0.1, Cg=1, Cd=0.1, N_max=10, kBT=0.01, e=1):
    """

    Parameters
    ----------
    Vg : Float
        Gate voltage.
    Cs : Float, optional
        Capacitance between dot and source. The default is 0.1.
    Cg : Float, optional
        Gate capacitance. The default is 1.
    Cd : Float, optional
        Capacitance between dot and drain. The default is 0.1.
    N_max : Int, optional
        Maximum number of electrons in a dot. The default is 10.
    kBT : Float, optional
        Thermal energy. The default is 0.01.
    e : Float, optional
        Elementary charge. The default is 1.

    Returns
    -------
    Float
        Average electrostatic energy of the single dot.

    """
    Z = 0 # partition function
    moy = 0
    for N in range(N_max+1): # loop from [0, N_max]
        E = U(N, Vg, Cs, Cg, Cd, e)
        Z = Z + np.exp(-E/kBT)

        moy = moy + E*np.exp(-E/kBT)

    return moy/Z

def N_moy(Vg, Cs=0.1, Cg=1, Cd=0.1, N_max=10, kBT=0.01, e=1):
    """

    Parameters
    ----------
    Vg : Float
        Gate voltage.
    Cs : Float, optional
        Capacitance between dot and source. The default is 0.1.
    Cg : Float, optional
        Gate capacitance. The default is 1.
    Cd : Float, optional
        Capacitance between dot and drain. The default is 0.1.
    N_max : Int, optional
        Maximum number of electrons in a dot. The default is 10.
    kBT : Float, optional
        Thermal energy. The default is 0.01.
    e : Float, optional
        Elementary charge. The default is 1.

    Returns
    -------
    Float
        Average electron number in the single dot.

    """
    Z = 0
    moy = 0
    for N in range(N_max+1): # loop from [0, N_max]
        E = U(N, Vg, Cs, Cg, Cd, e)
        Z = Z + np.exp(-E/kBT)

        moy = moy + N*np.exp(-E/kBT)

    return moy/Z

###############################################################################

# Double dot functions
def f(N1, N2, Vg1, Vg2, Ec1, Ec2, Cg1, Cg2, Ecm, e):
    """

    Parameters
    ----------
    N1 : Int
        Number of electrons on dot 1.
    N2 : Int
        Number of electrons on dot 1.
    Vg1 : Float
        Voltage on gate 1.
    Vg2 : Float
        Voltage on gate 2.
    Ec1 : Float
        Charging energy of dot 1.
    Ec2 : Float
        Charging energy of dot 2.
    Cg1 : Float
        Capacitance of gate 1.
    Cg2 : Float
        Capacitance of gate 2.
    Ecm : Float
        Electrostatic coupling energy between the two dots.
    e : Float
        Elementary charge.

    Returns
    -------
    Float
        Electrostatic energy of charges on the gates.

    """
    return -1/e*(Cg1*Vg1*(N1*Ec1+N2*Ecm)+Cg2*Vg2*(N1*Ecm+N2*Ec2))\
        +1/e**2*(0.5*Cg1**2*Vg1**2*Ec1+0.5*Cg2**2*Vg2**2*Ec2+Cg1*Vg1*Cg2*Vg2*Ecm)


def U_DQD(N1, N2, Vg1, Vg2, Cg1, Cg2, Cm, CL, CR, e=1):
    """

    Parameters
    ----------
    N1 : Int
        Number of electrons on dot 1.
    N2 : Int
        Number of electrons on dot 1.
    Vg1 : Float
        Voltage on gate 1.
    Vg2 : Float
        Voltage on gate 2.
    Cg1 : Float
        Capacitance of gate 1.
    Cg2 : Float
        Capacitance of gate 2.
    Cm : Float
        Capacitance between the two dots.
    CL : Float
        Capacitance of the source.
    CR : Float
        Capacitance of the drain.
    e : Float, optional
        Elementary charge. The default is 1.

    Returns
    -------
    Float
        Electrostatic energy of the double dot.

    """
    C1 = CL+Cg1+Cm # sum of the capacitance attached to dot 1
    C2 = CR+Cg2+Cm
    Ec1 = e**2*(C2/(C1*C2-Cm**2)) #Ec1 = e**2/C1*(1/(1-(Cm**2/(C1*C2)))) # version from paper but diverges at Cm=0
    Ec2 = e**2*(C1/(C1*C2-Cm**2)) #Ec2 = e**2/C2*(1/(1-(Cm**2/(C1*C2))))
    Ecm = e**2*(Cm/(C1*C2-Cm**2)) #Ecm = e**2/Cm*(1/((C1*C2/Cm**2)-1))

    return 0.5*N1**2*Ec1+0.5*N2**2*Ec2+N1*N2*Ecm+f(N1, N2, Vg1, Vg2, Ec1, Ec2, Cg1, Cg2, Ecm, e=1)

def N_moy_DQD(Vg1, Vg2, Cg1, Cg2, Cm, CL, CR, N_max = 10, kBT=0.01, e=1):
    """

    Parameters
    ----------
    Vg1 : Float
        Voltage on gate 1.
    Vg2 : Float
        Voltage on gate 2.
    Cg1 : Float
        Capacitance of gate 1.
    Cg2 : Float
        Capacitance of gate 2.
    Cm : Float
        Capacitance between the two dots.
    CL : Float
        Capacitance of the source.
    CR : Float
        Capacitance of the drain.
    N_max : Int, optional
        Maximum number of electrons in a dot. The default is 10.
    kBT : Float, optional
        Thermal energy. The default is 0.01.
    e : Float, optional
        Elementary charge. The default is 1.

    Returns
    -------
    Float
        Average number of electrons in the double dot.

    """
    Z = 0 # partition function
    moy = 0
    for N2 in range(N_max+1): # loop from [0, N_max]
        for N1 in range(N_max+1):
            E = U_DQD(N1, N2, Vg1, Vg2, Cg1, Cg2, Cm, CL, CR, e=e)
            Z = Z + np.exp(-E/kBT)

            moy = moy + (N1+N2)*np.exp(-E/kBT)

    return moy/Z

def plot_diagram(vi_1D, vf_1D, nx, vi_2D, vf_2D, ny, params_list=[1.4, 1.2, 0.2, 0.4, 0.4, 5, 0.01, 1], save_to_hdf5=False, save_to_txt=False, filename="sim_DQD"):
    """
    This function plot a simulated stability diagram. It can also be saved in hdf5 format.

    Parameters
    ----------
    vi_1D : float
        Initial voltage of the x-axis.
    vf_1D : float
        Final voltage of the x-axis.
    nx : int
        Number of points of the x-axis.
    vi_2D : float
        Initial voltage of the y-axis.
    vf_2D : float
        Final voltage of the y-axis.
    ny : int
        Number of points of the y-axis.
    params_list : list of floats, optional
        List of the 8 parameters for the simulation (Cg1, Cg2, Cm, CL, CR, N_max, kBT, e). 
        The default is [1.4, 1.2, 0.2, 0.4, 0.4, 5, 0.01, 1].
    save_to_hdf5 : bool, optional
        Option to save the diagram in hdf5 format. The default is False.
    filename : string, optional
        Name of the hdf5 file. The filename is used only if save_to_hdf5 is True. The default is "sim_DQD".

    Returns
    -------
    None.

    """
    # Create the data from the inputs
    x = np.linspace(vi_1D, vf_1D, nx)
    y = np.linspace(vi_2D, vf_2D, ny)
    xv, yv = np.meshgrid(x, y)
    z = N_moy_DQD(xv, yv, *params_list)

    # Average both vertical and horizontal derivatives to have identical transitions
    derivatives = np.gradient(z, axis=None)
    deriv_moy = np.sum(derivatives, axis=0)/len(derivatives)

    # Plot the data
    plt.figure()
    plt.pcolormesh(xv,yv,deriv_moy,cmap="viridis",shading="auto")
    plt.xlabel("Vg1 (V)")
    plt.ylabel("Vg2 (V)")
    cbar = plt.colorbar()
    cbar.set_label("dN/dVg", rotation=90)
    plt.title("Stability diagram\nCg1:{}, Cg2:{}, Cm:{}, CL:{}, CR:{}, N_max:{}, kBT:{}, e:{}".format(*params_list))

    if save_to_hdf5 == True:
        path = os.path.dirname(filename)
        if path != "":
            print("Creating path {}".format(path))
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        filename = filename.rstrip(".hdf5")
        with h5py.File("{}.hdf5".format(filename), "w") as file:
            dset = file.create_dataset(name="data", data=deriv_moy)
            dset.attrs["xgate"] = "Vg1"
            dset.attrs["xaxis"] = [vi_1D, vf_1D, nx]
            dset.attrs["ygate"] = "Vg2"
            dset.attrs["yaxis"] = [vi_2D, vf_2D, ny]
            dset.attrs["static_gates"] = []
            dset.attrs["static_gates_values"] = []
            dset.attrs["params_values"] = params_list

    if save_to_txt == True:
        path = os.path.dirname(filename)
        if path != "":
            print("Creating path {}".format(path))
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        filename = filename.rstrip(".txt")
        header_info = "xgate=Vg1, xaxis=[{},{},{}], ygate=Vg2, yaxis=[{},{},{}], params_values={}".format(vi_1D,vf_1D,nx,vi_2D,vf_2D,ny,params_list)
        np.savetxt(filename+".txt", deriv_moy, header=header_info)


if __name__ == "__main__":

    # Code examples
    plt.close("all")

    ###############################################################################
    # Single quantum dot
    ###############################################################################
    # Electrostatic energy without temperature
    Vg = np.linspace(-1,1,101)
    plt.figure()
    U_array = []
    for n in range(-1,3):
        y = U(n,Vg+n)
        U_array.append(y)
        plt.plot(Vg+n,y)

    plt.title("Electrostatic energy without temperature")
    plt.xlabel("Gate voltage")
    plt.ylabel("Electrostatic energy")

    # Electrostatic energy with temperature
    Vg = np.linspace(-0.5,3.1,1001)
    y = U_moy(Vg)
    plt.figure()
    plt.plot(Vg,y)

    plt.title("Electrostatic energy with temperature")
    plt.xlabel("Gate voltage")
    plt.ylabel("Electrostatic energy")

    # Charge transition at 2 temperatures and conductance peaks
    fig, ax1 = plt.subplots()
    Vg = np.linspace(0, 4, 1001)
    ax1.plot(Vg, N_moy(Vg), label="N (kBT=0)")
    kBT = 0.1
    ax1.plot(Vg, N_moy(Vg, kBT=kBT), label="N (kBT={:.02f})".format(kBT))
    ax1.set_xlabel("Gate voltage")
    ax1.set_ylabel("Nb of electrons")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(Vg, np.gradient(N_moy(Vg)), label="Coulomb peaks (kBT=0)", color="C2")
    ax2.plot(Vg, np.gradient(N_moy(Vg, kBT=kBT)), label="Coulomb peaks (kBT={:.02f})".format(kBT), color="C3")
    ax2.set_ylabel("dN/dVg")
    ax2.set_ylim(0, 0.4)
    ax2.legend(loc="upper right")

    plt.title("Electronic occupation")
    plt.legend()

    ###############################################################################
    # Double quantum dot
    ###############################################################################
    nx, ny = (201, 201)
    vi_1D, vf_1D = 0,1
    vi_2D, vf_2D = 0,1
    x = np.linspace(vi_1D, vf_1D, nx)
    y = np.linspace(vi_2D, vf_2D, ny)
    xv, yv = np.meshgrid(x, y)
    # Different parameters to test
    # z = N_moy_DQD(xv, yv, Cg1=0.7*2, Cg2=0.6*2, Cm=0.1*2, CL=0.2*2, CR=0.2*2, N_max = 5, kBT = 0.01, e=1)
    # z = N_moy_DQD(xv, yv, Cg1=0.4*2, Cg2=0.4*2, Cm=0.1, CL=0.4*2, CR=0.4*2, N_max = 5, kBT = 0.008, e=1)
    z = N_moy_DQD(xv, yv, Cg1=2, Cg2=2, Cm=0.2, CL=1, CR=1, N_max = 5, kBT = 0.005, e=1)

    # Stability diagram
    plt.figure()
    plt.pcolormesh(xv,yv,z,cmap="viridis",shading="auto")
    plt.xlabel("Vg1 (V)")
    plt.ylabel("Vg2 (V)")
    cbar = plt.colorbar()
    cbar.set_label("Nb of electrons", rotation=90)
    plt.title("Stability diagram")

    # Derivative of the stability diagram
    plt.figure()
    derivatives = np.gradient(z, axis=None)
    deriv_moy = np.sum(derivatives, axis=0)/len(derivatives)  # Average both vertical and horizontal derivatives to have identical transitions

    plt.pcolormesh(xv,yv,deriv_moy,cmap="viridis",shading="auto")
    plt.xlabel("Vg1 (V)")
    plt.ylabel("Vg2 (V)")
    cbar = plt.colorbar()
    cbar.set_label("dN/dVg", rotation=90)
    plt.title("Stability diagram")

    # Trace in the stability diagram
    plt.figure()
    # index = round(len(z)/2)-30
    index = np.where(y>=0)[0][0]
    z2 = z[:,index]
    plt.plot(x,z2, label="Vg2 = {:.3f} V".format(y[index]))
    plt.xlabel("Vg1 (V)")
    plt.ylabel("Nb of electrons")
    plt.legend()
    plt.title("Trace of the stability diagram")

    # Save data if needed, params : (Cg1, Cg2, Cm, CL, CR, N_max, kBT, e)
    plot_diagram(-1,1,201,-1,1,201, params_list=[5, 2.8, 50, 0.2, 0.2, 10, 0.01, 1], filename="DQD_save_example", save_to_hdf5=True, save_to_txt=True)

    # Read HDF5 file
    with h5py.File("DQD_save_example.hdf5", "r") as file:

        print("File keys:",file.keys())
        print("File attributes:",file["data"].attrs.keys())
        data = file["data"][:]

        plt.figure()
        x = np.linspace(*file["data"].attrs["xaxis"])
        y = np.linspace(*file["data"].attrs["yaxis"])
        plt.pcolor(x, y, data, shading="auto")
        plt.xlabel(file["data"].attrs["xgate"])
        plt.ylabel(file["data"].attrs["ygate"])

        cbar = plt.colorbar()
        cbar.set_label("dN/dVg", rotation=90)
        plt.title("HDF5 stability diagram")

    plt.show(block=True)
