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
Plot a data matrix as an aggregate of line plots
'''
def lines_plot(ax, data, axis, xlabel, ylabel, title, labels=None):
    if axis[0].lower() == 'r':
        # Each row is a line
        data = data.T
        num_pts = data.shape[0]
    elif axis[0].lower() == 'c':
        # Each column is a line
        num_pts = data.shape[0]

    xs = np.arange(num_pts)

    for i in range(data.shape[1]):
        ax.plot(xs, data[:, i])
        if labels is not None:
            ax.text(xs[-1], data[-1, i], labels[i], horizontalalignment='right')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)



'''
Plot the projection of a 2D data matrix onto one component of its SVD
'''
def svd_component_plot(ax, data, component, xlabel, ylabel, zlabel='', topic='', title=None, centering='s', style='surface', axis = 'rows', labels = None):
    letter = centering[0].upper()
    if topic != '':
        topic = topic + ': '
    if title is None:
        title = "%s%sSVD, SV%d" % (topic, letter, component+1)

    if style == 'surface':
        surface_plot( ax     = ax,
                      data   = svdt.svd_project(data, [component], centering=centering),
                      title  = title,
                      xlabel = xlabel,
                      ylabel = ylabel,
                      zlabel = zlabel
                    )
    if style == 'lines':
        lines_plot( ax     = ax,
                    data   = svdt.svd_project(data, [component], centering=centering),
                    title  = title,
                    labels = labels,
                    axis   = 'rows',
                    xlabel = xlabel,
                    ylabel = ylabel,
                  )


'''
Plot the projection of a 2D data matrix onto its rank <rank> approximation
'''
def svd_approx_plot(ax, data, rank, xlabel, ylabel, zlabel='', topic='', title=None, centering='s',
        style='surface', axis='rows', labels=None):
    letter = centering[0].upper()
    if topic != '':
        topic = topic + ': '
    if title is None:
        title = '%sRank %d Approx (%sSVD)' % (topic, rank, letter)

    if style == 'surface':
        surface_plot( ax     = ax, 
                      data   = svdt.rank_k_approx(data, rank=rank, centering=centering),
                      title  = title,
                      xlabel = xlabel,
                      ylabel = ylabel,
                      zlabel = zlabel,
                    )
    elif style == 'lines':
        lines_plot( ax     = ax, 
                    data   = svdt.rank_k_approx(data, rank=rank, centering=centering),
                    title  = title,
                    xlabel = xlabel,
                    ylabel = ylabel,
                    axis   = axis,
                    labels = labels
                  )

'''
Plot the residuals of the rank <rank> approximation of a data matrix
'''
def svd_residual_plot(ax, data, rank, xlabel, ylabel, zlabel='', topic='', title=None, centering='s',
        style='surface', axis='rows', labels=None):
    letter = centering[0].upper()
    residuals = data - svdt.rank_k_approx(data, rank=rank, centering=centering)

    if topic != '':
        topic = topic + ': '
    if title is None:
        title = '%sRank %d Approx Residuals (%sSVD)' % (topic, rank, letter)

    if style == 'surface':
        surface_plot( ax     = ax, 
                      data   = residuals,
                      title  = title,
                      xlabel = xlabel,
                      ylabel = ylabel,
                      zlabel = zlabel
                    )
    elif style == 'lines':
        lines_plot( ax     = ax, 
                    data   = residuals,
                    title  = title,
                    xlabel = xlabel,
                    ylabel = ylabel,
                    axis   = axis,
                    labels = labels
                  )


def svd_plots(fig, data, title, xlabel, ylabel, zlabel='', centering='s', style='surface', labels=None, axis='rows'):
    if style == 'surface':
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
    elif style == 'lines':
        p_original = fig.add_subplot('231')
        p_sv = list()
        p_sv.append(fig.add_subplot('232'))
        p_sv.append(fig.add_subplot('233'))
        p_sv.append(fig.add_subplot('234'))
        p_approx = fig.add_subplot('235')
        p_residuals = fig.add_subplot('236')

        lines_plot( ax     = p_original, 
                    data   = data, 
                    title  = title,
                    xlabel = xlabel,
                    ylabel = ylabel,
                    labels = labels,
                    axis   = axis
                  )

    for i in range(3):
        svd_component_plot( ax        = p_sv[i], 
                            data      = data, 
                            component = i, 
                            xlabel    = xlabel, 
                            ylabel    = ylabel, 
                            zlabel    = zlabel,
                            centering = centering,
                            style     = style,
                            labels    = labels,
                            axis      = axis
                            )
    svd_approx_plot( ax = p_approx,
                     data      = data,
                     rank      = 3,
                     xlabel    = xlabel,
                     ylabel    = ylabel,
                     zlabel    = zlabel,
                     centering = centering,
                     style     = style,
                     labels    = labels,
                     axis      = axis
                   )
    svd_residual_plot( ax = p_residuals,
                       data      = data,
                       rank      = 3,
                       xlabel    = xlabel,
                       ylabel    = ylabel,
                       zlabel    = zlabel,
                       centering = centering,
                       style     = style,
                       labels    = labels,
                       axis      = axis
                     )

def svd_plots_four(fig, data, title, xlabel, ylabel, zlabel='', centering='s', style='surface', labels=None, axis='rows'):

    if style == 'surface':
        p_original = fig.add_subplot('141', projection='3d')
        p_sv = list()
        p_sv.append(fig.add_subplot('142', projection='3d'))
        p_sv.append(fig.add_subplot('143', projection='3d'))
        p_sv.append(fig.add_subplot('144', projection='3d'))

        surface_plot( ax     = p_original, 
                      data   = data, 
                      title  = title,
                      xlabel = xlabel,
                      ylabel = ylabel,
                      zlabel = zlabel
                    )
    elif style == 'lines':
        p_original = fig.add_subplot('141')
        p_sv = list()
        p_sv.append(fig.add_subplot('142'))
        p_sv.append(fig.add_subplot('143'))
        p_sv.append(fig.add_subplot('144'))

        lines_plot( ax     = p_original, 
                    data   = data, 
                    title  = title,
                    xlabel = xlabel,
                    ylabel = ylabel,
                    labels = labels,
                    axis   = axis
                  )

    for i in range(3):
        svd_component_plot( ax        = p_sv[i], 
                            data      = data, 
                            component = i, 
                            xlabel    = xlabel, 
                            ylabel    = ylabel, 
                            zlabel    = zlabel,
                            centering = centering,
                            style     = style,
                            labels    = labels,
                            axis      = axis
                            )

def svd_plots_four_resid(fig, data, title, xlabel, ylabel, zlabel='', centering='s', style='surface', labels=None, axis='rows'):

    if style == 'surface':
        p_sv = list()
        p_sv.append(fig.add_subplot('141', projection='3d'))
        p_sv.append(fig.add_subplot('142', projection='3d'))
        p_sv.append(fig.add_subplot('143', projection='3d'))
        p_residuals = fig.add_subplot('144', projection='3d')

    elif style == 'lines':
        p_sv = list()
        p_sv.append(fig.add_subplot('141'))
        p_sv.append(fig.add_subplot('142'))
        p_sv.append(fig.add_subplot('143'))
        p_residuals = fig.add_subplot('144')

    for i in range(3):
        svd_component_plot( ax        = p_sv[i], 
                            data      = data, 
                            component = i, 
                            xlabel    = xlabel, 
                            ylabel    = ylabel, 
                            zlabel    = zlabel,
                            centering = centering,
                            style     = style,
                            labels    = labels,
                            axis      = axis
                            )

    svd_residual_plot( ax = p_residuals,
                       data      = data,
                       rank      = 3,
                       xlabel    = xlabel,
                       ylabel    = ylabel,
                       zlabel    = zlabel,
                       centering = centering,
                       style     = style,
                       labels    = labels,
                       axis      = axis
                     )



def svd_plots_three(fig, data, title, xlabel, ylabel, zlabel='', centering='s', style='surface', labels=None, axis='rows'):
    if style == 'surface':
        p_sv = list()
        p_sv.append(fig.add_subplot('131', projection='3d'))
        p_sv.append(fig.add_subplot('132', projection='3d'))
        p_sv.append(fig.add_subplot('133', projection='3d'))

        surface_plot( ax     = p_original, 
                      data   = data, 
                      title  = title,
                      xlabel = xlabel,
                      ylabel = ylabel,
                      zlabel = zlabel
                    )
    elif style == 'lines':
        p_sv = list()
        p_sv.append(fig.add_subplot('131'))
        p_sv.append(fig.add_subplot('132'))
        p_sv.append(fig.add_subplot('133'))
       
    for i in range(3):
        svd_component_plot( ax        = p_sv[i], 
                            data      = data, 
                            component = i, 
                            xlabel    = xlabel, 
                            ylabel    = ylabel, 
                            zlabel    = zlabel,
                            centering = centering,
                            style     = style,
                            labels    = labels,
                            axis      = axis
                            )

def svd_plots_five(fig, data, title, xlabel, ylabel, zlabel='', centering='s', style='surface', labels=None, axis='rows'):
    if style == 'surface':
        p_sv = list()
        p_sv.append(fig.add_subplot('151', projection='3d'))
        p_sv.append(fig.add_subplot('152', projection='3d'))
        p_sv.append(fig.add_subplot('153', projection='3d'))
        p_approx = fig.add_subplot('154', projection='3d')
        p_residuals = fig.add_subplot('155', projection='3d')

    elif style == 'lines':
        p_sv = list()
        p_sv.append(fig.add_subplot('151'))
        p_sv.append(fig.add_subplot('152'))
        p_sv.append(fig.add_subplot('153'))
        p_approx = fig.add_subplot('154')
        p_residuals = fig.add_subplot('155')
        
    for i in range(3):
        svd_component_plot( ax        = p_sv[i], 
                            data      = data, 
                            component = i, 
                            xlabel    = xlabel, 
                            ylabel    = ylabel, 
                            zlabel    = zlabel,
                            centering = centering,
                            style     = style,
                            labels    = labels,
                            axis      = axis
                            )
    svd_approx_plot( ax = p_approx,
                     data      = data,
                     rank      = 3,
                     xlabel    = xlabel,
                     ylabel    = ylabel,
                     zlabel    = zlabel,
                     centering = centering,
                     style     = style,
                     labels    = labels,
                     axis      = axis
                   )
    svd_residual_plot( ax = p_residuals,
                       data      = data,
                       rank      = 3,
                       xlabel    = xlabel,
                       ylabel    = ylabel,
                       zlabel    = zlabel,
                       centering = centering,
                       style     = style,
                       labels    = labels,
                       axis      = axis
                     )



def filter_labels(labels, chosen):
    if chosen is None:
        return labels
    else:
        return list(map(lambda label: label if label in chosen else '', labels))
