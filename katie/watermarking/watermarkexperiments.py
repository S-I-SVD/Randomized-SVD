#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:25:29 2020

@author: katie
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from timeit import timeit
from mpl_toolkits import mplot3d
import scipy
from fractions import Fraction
import svd as svd
from matplotlib.image import imread
from skimage import color
from skimage import io
from watermarking import *
from watermarkjain import * 
from watermarkjainmod import *

coffee = imageio.imread(r'/Users/katie/Downloads/coffee.png')
cat = imageio.imread(r'/Users/katie/opt/anaconda3/pkgs/scikit-image-0.15.0-py37h0a44026_0/lib/python3.7/site-packages/skimage/data/eye.png')
cat=cat[:,:,:-1]
randommatrix = np.random.rand(600,600)

def watermarkedplot(M,W,plottype,random):
    alphas = np.arange(0.05,2.05,0.05)
    differences = []
    #liu tan
    if plottype == 1:
        for alpha in alphas:
            difference = perceptibility_liutan(M,alpha,W,0,random)
            differences.append(difference)
    #jain
    if plottype == 2:
        for alpha in alphas:
            difference = perceptibility_jain(M,alpha,W,0,random)
            differences.append(difference)
            
    #jain mod
    if plottype == 3:
        for alpha in alphas:
            difference = perceptibility_jainmod(M,alpha,W,0,random)
            differences.append(difference)
    
    drawgraph_difference(alphas,differences,random)
        
        
def extractedwatermarkplot(M,W,plottype):
    alphas = np.arange(0.05,2.05,0.05)
    differences = []
    #liu tan
    if plottype == 1:
        for alpha in alphas:
            difference = extractedwatermarkdifference_liutan(M,alpha,W)
            differences.append(difference)
    #jain
    if plottype == 2:
        for alpha in alphas:
            difference = extractedwatermarkdifference_jain(M,alpha,W)
            differences.append(difference)
            
    #jain mod
    if plottype == 3:
        for alpha in alphas:
            difference = extractedwatermarkdifference_jainmod(M,alpha,W)
            differences.append(difference)
    
    drawgraph_watermarkdifference(alphas,differences)

def cropplotcolumn(M,a,W,plottype,random):
    columns = np.arange(0,201,1)
    differences = []
    #liutan
    if plottype == 1:
        differences = []
        M1 = watermarkonly_liutan(M,a,W)
        for column in columns:
            M1 = M1[:,:-1,:]
            difference = extractedwatermarkdifferencecropped_liutan(M,a,W,M1,0,random)
            differences.append(difference)
    #jain
    elif plottype == 2:
        M1 = watermarkonly_jain(M,a,W)
        for column in columns:
            M1 = M1[:,:-1,:]
            difference = extractedwatermarkdifferencecropped_jain(M,a,W,M1,0,random)
            differences.append(difference)
    #jain mod
    elif plottype == 3:
        M1 = watermarkonly_jainmod(M,a,W)
        for column in columns:
            M1 = M1[:,:-1,:]
            difference = extractedwatermarkdifferencecropped_jainmod(M,a,W,M1,0,random)
            differences.append(difference)  
            
    drawgraph_cropdifferencecolumn(columns,differences,a,random)
    
def cropplotrow(M,a,W,plottype,random):
    rows = np.arange(0,201,1)
    differences = []
    #liutan
    if plottype == 1:
        M1 = watermarkonly(M,a,W)
        for row in rows:
            M1 = M1[:-1,:,:]
            difference = extractedwatermarkdifferencecropped_liutan(M,a,W,M1,0,random)
            differences.append(difference)
    #jain
    elif plottype == 2:
        M1 = watermarkonly_jain(M,a,W)
        for row in rows:
            M1 = M1[:-1,:,:]
            difference = extractedwatermarkdifferencecropped_jain(M,a,W,M1,0,random)
            differences.append(difference)
    #jain mod
    elif plottype == 3:
        M1 = watermarkonly_jainmod(M,a,W)
        for row in rows:
            M1 = M1[:-1,:,:]
            difference = extractedwatermarkdifferencecropped_jainmod(M,a,W,M1,0,random)
            differences.append(difference)  
            
    drawgraph_cropdifferencerow(rows,differences)
    

def drawgraph_difference(x,y,random):
    plt.plot(x,y,marker='o')
    plt.xlabel('Scaling factors')
    plt.ylabel('Error (Frobenius Norm)')
    if random == 1:
        plt.title('Differences Between Original and SVD Watermarked Image By Adjusting Scaling Factor \n Random Matrix')
    else:
        plt.title('Differences Between Original and SVD Watermarked Image By Adjusting Scaling Factor')
    plt.show()
    
def drawgraph_watermarkdifference(x,y,random):
    plt.plot(x,y,marker='o')
    plt.xlabel('Scaling factors')
    plt.ylabel('Error (Frobenius Norm)')
    if random == 1:
        plt.title('Differences Between Original and Extracted Watermark By Adjusting Scaling Factor \n Random Matrix')
    else:
        plt.title('Differences Between Original and Extracted Watermark By Adjusting Scaling Factor')
    plt.show()
    
def drawgraph_cropdifferencecolumn(x,y,a,random):
    plt.plot(x,y,marker='o')
    plt.xlabel('Columns Removed')
    plt.ylabel('Relative Error')
    if random == 1:
        plt.title('Differences Between Original and Extracted Watermark By Cropping \n Random Matrix, a = {0}'.format(a),pad=10)
    else:
        plt.title('Differences Between Original and Extracted Watermark By Cropping \n a = {0}'.format(a),pad=10)
    plt.show()
    
def drawgraph_cropdifferencerow(x,y,a,random):
    plt.plot(x,y,marker='o')
    plt.xlabel('Rows Removed')
    plt.ylabel('Relative Error')
    if random == 1:
        plt.title('Differences Between Original and Extracted Watermark By Cropping \n Random Matrix, a = {0}'.format(a),pad=10)
    else:
        plt.title('Differences Between Original and Extracted Watermark By Cropping \n a = {0}'.format(a),pad=10)
    plt.show()
    
def drawgraph_extractiondifference(x,y,scalingfactor,random):
    #random=1 if it's a random matrix, random=0 otherwise
    plt.plot(x,y,marker='o')
    plt.xlabel('Rank of Watermarked Image Approximation')
    plt.ylabel('Error (Frobenius Norm)')
    if random == 1:
        plt.title('Differences Between Original and Extracted Watermark \n Random Matrix, a = {0}'.format(scalingfactor))
    else:
        plt.title('Differences Between Original and Extracted Watermark \n a = {0}'.format(scalingfactor))
    plt.show()
    
    
