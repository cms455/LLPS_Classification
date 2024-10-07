import numpy as np
import pickle
import matplotlib.pyplot as plt
import mpltern
#import matplotlib as mpl
import matplotlib as mpl

from mpltern.datasets import get_shanon_entropies


def add_discrete_colorbar(ax, colors, vmin=0, vmax=None, label=None, fontsize=None, option='normal',
                 tight_layout=True, ticklabelsize=None, ticklabel=None,
                 aspect = None, useMiddle4Ticks=False,**kwargs):
    fig = ax.get_figure()
    if vmax is None:
        vmax = len(colors)
    tick_spacing = (vmax - vmin) / float(len(colors))
    if not useMiddle4Ticks:
        vmin, vmax = vmin -  tick_spacing / 2., vmax -  tick_spacing / 2.
    ticks = np.linspace(vmin, vmax, len(colors) + 1) + tick_spacing / 2.  # tick positions

    # if there are too many ticks, just use 3 ticks
    if len(ticks) > 10:
        n = len(ticks)
        ticks = [ticks[0], ticks[n//2]-1, ticks[-2]]
        if ticklabel is not None:
            ticklabel = [ticklabel[0], ticklabel[n/2], ticklabel[-1]]


    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # dummy mappable

    if option == 'scientific':
        cb = fig.colorbar(sm, ticks=ticks, format=sfmt, ax=ax, **kwargs)
    else:
        cb = fig.colorbar(sm, ticks=ticks, ax=ax, **kwargs)

    if ticklabel is not None:
        cb.ax.set_yticklabels(ticklabel)

    if not label is None:
        if fontsize is None:
            cb.set_label(label)
        else:
            cb.set_label(label, fontsize=fontsize)
    if ticklabelsize is not None:
        cb.ax.tick_params(labelsize=ticklabelsize)

    # Adding a color bar may distort the aspect ratio. Fix it.
    if aspect=='equal':
        ax.set_aspect('equal')

    # Adding a color bar may disport the overall balance of the figure. Fix it.
    if tight_layout:
        fig.tight_layout()

    return cb


def get_discrete_cmap_norm(colors, vmin=0, vmax=None, label=None, fontsize=None, option='normal',
                              tight_layout=True, ticklabelsize=None, ticklabel=None,
                              aspect=None, useMiddle4Ticks=False, **kwargs):
    """
    Return a colormap and a norm for a discrete colorbar

    Parameters
    ----------
    colors
    vmin
    vmax
    label
    fontsize
    option
    tight_layout
    ticklabelsize
    ticklabel
    aspect
    useMiddle4Ticks
    kwargs

    Returns
    -------
    cmap, norm
    """
    if vmax is None:
        vmax = len(colors)

    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


def get_color_from_cmap(cmap='viridis', n=10, lut=None, reverse=False):

    '''
    A simple function which returns a list of RGBA values from a cmap (evenly spaced)
    ... If one desires to assign a color based on values, use get_colors_and_cmap_using_values()
    ... If one prefers to get colors between two colors of choice, use get_color_list_gradient()
    Parameters
    ----------
    cmapname: str, standard cmap name
    n: int, number of colors
    lut, int, HELLO
        #... If lut is not None it must be an integer giving the number of entries desired in the lookup table,
    #    and name must be a standard mpl colormap name.

    Returns
    -------
    colors

  '''

    cmap = mpl.cm.get_cmap(cmap, lut)
    if reverse:
        cmap = cmap.reversed()
    colors = cmap(np.linspace(0, 1, n, endpoint=True))
    return colors



def plot_ternary(data):
    npts = data.shape[0]  # number of points
    phi1 = np.asarray([data[:, 0][i][0] for i in range(npts)])
    phi2 = np.asarray([data[:, 0][i][1] for i in range(npts)])
    phi3 = np.asarray([data[:, 0][i][2] for i in range(npts)])
    nPhases = np.asarray(data[:, 1]).astype(int)

    # PLOTTING
    fig = plt.figure(figsize=(10.8, 4.8))
    fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)

    nComp = 3
    colors = get_color_from_cmap(cmap='tab10', n=10)[:nComp]
    cmap_discrete, norm = get_discrete_cmap_norm(colors, vmin=1, vmax=nComp)

    ax = fig.add_subplot(111, projection='ternary')
    cs = ax.tripcolor(phi1, phi2, phi3, nPhases, cmap=cmap_discrete)
    ax.set_title("Evolutionary Algorithm")

    add_discrete_colorbar(ax, colors, vmin=1, vmax=nComp + 1, fraction=0.02, label='No. of Phases')
    # ax.set_axis_off()

    ax.set_tlabel("Component 1")
    ax.set_llabel("Component 2")
    ax.set_rlabel("Component 3")

    plt.show()
    

def plot_time(data):

    plt.figure(figsize=(8, 6))  # Creates a figure with a specified size
    plt.plot(data[:, 0], data[:, 1], marker='o', linestyle='-', color='b')  # Plot data
    plt.title('Seconds Vs #Components')  # Title of the plot
    plt.xlabel('# of Components')  # X-axis label
    plt.ylabel('Seconds')  # Y-axis label
    plt.grid(True)  # Enable grid
    plt.show()  # Display the plot

def plot_comp_volume_fraction(data,readout_func):
    '''
    Plots each of the components of the volume fraction as a scatter plot 
    Data is a numpy array
    '''
    num_comps = data.shape[1]
    num_of_points = data.shape[0]
    norm_points = norm_points = np.linspace(0,1, num_of_points)

    
    # Customize the plot
    plt.title('Plot of Component Volume Fractions')
    plt.xlabel('Initial Phi_In')
    plt.ylabel('Volume Fraction')
    plt.ylim(0, 1)
    
    # Create scatter plot for the first set of y-values
    for i in range(1,(num_comps)-1):
        plt.scatter(norm_points, data[:,i], color='green', marker='o')
    plt.scatter(norm_points,data[:,0], color='blue', marker='o')
    plt.scatter(norm_points, data[:,-1], color='red', marker='o')

    output_comp_points = []
    for point in data:
        x = readout_func(point[0],point[-1])
        output_comp_points.append(x)
    output_comp_points = np.array(output_comp_points)
    plt.scatter(norm_points,output_comp_points, color = 'purple', marker='x')
    
    
    # Show plot
    plt.show()

def plot_comp_volume_fraction_v2(data,readout_func):
    '''
    Plots each of the components of the volume fraction as a scatter plot 
    Data is a numpy array
    '''
    num_comps = data.shape[1]
    num_of_points = data.shape[0]
    norm_points = norm_points = np.linspace(0,1, num_of_points)

    
    # Customize the plot
    plt.title('Plot of Component Volume Fractions')
    plt.xlabel('Initial Phi_In')
    plt.ylabel('Volume Fraction')
    plt.ylim(0, 1)
    
    # Create scatter plot for the first set of y-values
    for i in range(1,(num_comps)-2):
        plt.scatter(norm_points, data[:,i], color='green', marker='o', label = f'Phi_H {i}')
    plt.scatter(norm_points,data[:,0], color='blue', marker='o',label = 'Phi_In')
    plt.scatter(norm_points, data[:,-2], color='red', marker='o', label = 'Phi_Out')
    plt.scatter(norm_points, data[:,-1], color='yellow', marker='o', label = 'Phi_Sol')

    output_comp_points = []
    for point in data:
        x = readout_func(point[0],point[-2])
        output_comp_points.append(x)
    output_comp_points = np.array(output_comp_points)
    plt.scatter(norm_points,output_comp_points, color = 'purple', marker='x')
    plt.plot(norm_points,output_comp_points, color = 'purple', linewidth = 2.0, label = 'Readout Value')
    plt.legend()
    # Show plot
    plt.show()



def plot_comp_volume_fraction_v3(data,readout_func):
    '''
    Plots each of the components of the volume fraction as a scatter plot 
    Data is a numpy array
    '''
    num_comps = data.shape[1]
    num_of_points = data.shape[0]
    norm_points = norm_points = np.linspace(0,1, num_of_points)

    
    # Customize the plot
    plt.title('Plot of Component Volume Fractions')
    plt.xlabel('Initial Phi_In')
    plt.ylabel('Volume Fraction')
    plt.ylim(0, 1)
    
    # Create scatter plot for the first set of y-values
    for i in range(1,(num_comps)-2):
        plt.scatter(norm_points, data[:,i], color='green', marker='o', label = f'Phi_H {i}')
    plt.scatter(norm_points,data[:,0], color='blue', marker='o',label = 'Phi_In')
    plt.scatter(norm_points, data[:,-2], color='red', marker='o', label = 'Phi_Out')
    plt.scatter(norm_points, data[:,-1], color='yellow', marker='o', label = 'Phi_Sol')

    output_comp_points = []
    for point in data:
        x = readout_func(point[-2],point[-1])
        output_comp_points.append(x)
    output_comp_points = np.array(output_comp_points)
    plt.scatter(norm_points,output_comp_points, color = 'purple', marker='x')
    plt.plot(norm_points,output_comp_points, color = 'purple', linewidth = 2.0, label = 'Readout Value')
    plt.legend()
    # Show plot
    plt.show()
