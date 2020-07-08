import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import svd_tools as svdt

'''
Plot a data matrix as a surface
'''
def surface_plot(ax, data, xlabel, ylabel, zlabel, title):
    xs = np.arange(0, data.shape[1])
    ys = np.arange(0, data.shape[0])
    xgrid, ygrid = np.meshgrid(xs, ys)

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax.plot_surface(xgrid, ygrid, data, rstride=1, cstride=1, cmap='jet')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)


'''
Plot the projection of a 2D data matrix onto one component of its SVD
'''
def svd_component_plot(ax, data, component, xlabel, ylabel, zlabel, topic='', title=None, centering='s'):
    letter = centering[0].upper()
    if topic != '':
        topic = topic + ': '
    if title is None:
        title = "%s%sSVD, SV%d" % (topic, letter, component+1)
    surface_plot( ax     = ax,
                  data   = svdt.svd_project(data, [component], centering=centering),
                  title  = title,
                  xlabel = 'Days since 1/22/20',
                  ylabel = 'County',
                  zlabel = 'Total Deaths'
                )

'''
Plot the projection of a 2D data matrix onto its rank <rank> approximation
'''
def svd_approx_plot(ax, data, rank, xlabel, ylabel, zlabel, topic='', title=None, centering='s'):
    letter = centering[0].upper()
    if topic != '':
        topic = topic + ': '
    if title is None:
        title = '%sRank %d Approx (%sSVD)' % (topic, rank, letter)
    surface_plot( ax     = ax, 
                  data   = svdt.rank_k_approx(data, rank=rank),
                  title  = title,
                  xlabel = 'Days since 1/22/20',
                  ylabel = 'County',
                  zlabel = 'Total Deaths'
                )

'''
Plot the residuals of the rank <rank> approximation of a data matrix
'''
def svd_residual_plot(ax, data, rank, xlabel, ylabel, zlabel, topic='', title=None, centering='s'):
    letter = centering[0].upper()
    residuals = data - svdt.rank_k_approx(data, rank=rank, centering=centering)

    if topic != '':
        topic = topic + ': '
    if title is None:
        title = '%sRank %d Approx Residuals (%sSVD)' % (topic, rank, letter)

    surface_plot( ax     = ax, 
                  data   = residuals,
                  title  = title,
                  xlabel = 'Days since 1/22/20',
                  ylabel = 'County',
                  zlabel = 'Total Deaths'
                )

def plot_svd_surfaces(data, title, xlabel, ylabel, zlabel, centering='s'):
    fig = plt.figure()

    p_original = fig.add_subplot('231', projection='3d')
    p_sv = list()
    p_sv.append(fig.add_subplot('232', projection='3d'))
    p_sv.append(fig.add_subplot('233', projection='3d'))
    p_sv.append(fig.add_subplot('234', projection='3d'))
    p_approx = fig.add_subplot('235', projection='3d')
    p_residuals = fig.add_subplot('236', projection='3d')

    surface_plot( ax     = p_original, 
                  data   = data, 
                  title  = title,
                  xlabel = xlabel,
                  ylabel = ylabel,
                  zlabel = zlabel
            )

    for i in range(3):
        svd_component_plot( ax        = p_sv[i], 
                            data      = data, 
                            component = i, 
                            xlabel    = xlabel, 
                            ylabel    = ylabel, 
                            zlabel    = zlabel,
                            centering = centering
                            )
    svd_approx_plot( ax = p_approx,
                     data      = data,
                     rank      = 3,
                     xlabel    = xlabel,
                     ylabel    = ylabel,
                     zlabel    = zlabel,
                     centering = centering
                   )
    svd_residual_plot( ax = p_residuals,
                       data      = data,
                       rank      = 3,
                       xlabel    = xlabel,
                       ylabel    = ylabel,
                       zlabel    = zlabel,
                       centering = centering
                     )

    plt.show()
