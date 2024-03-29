# Randomized-SVD

This GitHub repository documents work completed by the Randomized SVD team at [Summer@ICERM 2020](https://icerm.brown.edu/summerug/2020/). The authors of the code provided here are David Melendez, Jennifer Zheng, and Katie Keegan. As the tasks that we each completed were often distinct and requiring different functions, our code is organized based on the author.

The below information should help clarify how to reproduce specific results, which have been documented in two written reports: our [report for Summer@ICERM]() as well as a [submitted manuscript for SIURO]().

## Contents
* [Data](#data)
* [David](#david)
* [Jennifer](#jennifer)
* [Katie](#katie)

## Data
Our project utilizes the following sources of data.

## David
- `watermark.py`: Contains watermark embedding and watermark extraction functions using the Liu & Tan, Jain, proposed Modified Jain watermarking schemes, which can be applied to matrices.
- `image_tools.py`: Contains the same watermark-related functions as in `watermark.py`, but usable on color images. Each function "stacks" the *M x N x C* images into 2-dimensional matrices, applies the watermarking function, and then "unstacks" the result back into an *M x N x C* image.
	
## Jennifer

Jennifer's code contains all functions needed to reproduce our watermarking and media processing work on audio files. 

## Katie

/Watermarking_new/image_tools_copy.py, /Watermarking_new/svd_tools_copy.py, and /Watermarking_new/watermark_copy.py are nearly exact copies of three files in David's code with minor changes, and were moved to this folder in order to make edits easier.

The following notebook illustrates how to use the code in /Watermarking_new/watermarking_experiments.py: https://colab.research.google.com/drive/1a4zlPgRrjzMdJ4XjOKJdVf0R0k19V_8d?usp=sharing 
