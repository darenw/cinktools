"""
	DARTOOFI: Daren's Tools for Image work
	
	Simple (or almost simple) routines in Python for viewing numpy arrays
	as images, read/write them as image files, and do some basic
	image processing.
	
	See also dswimgproc.py, dswimgtest.py
	
	Official source is on GitHub  https://github.com/darenw/dartoofi.git
	
	License: MIT
"""


import numpy as np
import math
from PIL import Image


def BiasedGamma(img, gam, bias, maxval=None):
	if not maxval:
		maxval = SoftMax(img)
	t = ( ((img+bias)/(maxval+bias)).clip(0,None) )**(1/gam)
	t0 = (bias/(bias+maxval))**(1/gam)
	tmax = 1.0
	a = maxval / (tmax-t0)
	return a*(t-t0)



def SoftMax(A):
	"""
		Calculate a mushy rough maximum, ignoring single or few pixels,
		but weighted by the more numerous of the brighter pixels
		Useful for setting levels based on generic sky, white objects, etc
		but not stars, cosmic rays, tiny specular hotspots, sun glints
	"""
	s = A[::8,::8]
	w = (s-s.mean()).clip(0,None)**3
	return (s*w).mean() / w.mean() 


def calcstats(img, subsample=7):
	xmin = img.min()
	xmax = img.max()
	nmin = np.count_nonzero(img==xmin)
	nmax = np.count_nonzero(img==xmax)
	s = subsample
	if s==None or s==0:
		s=1
	subx = img[s//2:-1:s, s//2:-1:s].flatten()
	subx.sort()
	xmean = subx.mean()
	n = len(subx)
	x01 = subx[n//100]
	x10 = subx[n//10]
	x90 = subx[n-n//10-1]
	x99 = subx[n-n//100-1]
	xmed = subx[n//2]
	return {"min":xmin,"max":xmax,"nmin":nmin,"nmax":nmax,
	        "mean":xmean,
	        "median":xmed,
	        "percentiles": {0:xmin,1:x01, 10:x10, 50:xmed, 90:x90, 99:x99, 100:xmax }
	        }

stats_format = "%7.3f  (%5d) %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f (%5d)"

def printstatsline(D):
	"""
		Print result of statscalc2D on one line
	"""
	per = D["percentiles"]
	print(stats_format  \
	    %  ( D["mean"], D["nmin"], D["min"], per[1], per[10], per[50], per[90], per[99], 
	         D["max"],  D["nmax"] ))


def stats(img, subsample=7,  header=False):
	"""
		Print basic stats of an image.
		Image may be 2D, 3D, or tuple of three color planes
		ARGS
				img: a 2D array, tuple of 2D, or 3D [y,x,rgb] or [rgb,y,x]
				subsample: stride for subsampling
	"""
	if header:
		print("  mean     Nmin    min      1%     10%    median   90%     99%     max     Nmax" )
	
	# In case given tuple of one array, unwrap it.
	A=img
	if isinstance(img, (list,tuple)):
			for L in img:
				stats(L)
			return
	
	if   len(img.shape)==2:
		printstatsline(calcstats(img, subsample=subsample))
	
	elif len(A.shape)==3:
		if   A.shape[0]==3 and A.shape[2]>3:
			printstatsline(calcstats(img[0,:,:], subsample=subsample))
			printstatsline(calcstats(img[1,:,:], subsample=subsample))
			printstatsline(calcstats(img[2,:,:], subsample=subsample))
		elif A.shape[3]==3:
			printstatsline(calcstats(img[:,:,0], subsample=subsample))
			printstatsline(calcstats(img[:,:,1], subsample=subsample))
			printstatsline(calcstats(img[:,:,2], subsample=subsample))
		else:
			printf("Array has non-image shape ", A.shape)
			return None


def sho(A, maxval=None, magnify=1, inv=False):
	"""
		Display numpy array data as an image, given a single object.
		This may be a 2D array for display as a monochrome image,
		or a 3D array with first or last dimension of size 3,
		or a list or tuple of three 2D arrays, to display as a color image
		
		Uses ImageMagick's displayer, by PIL's design.
		       
		ARGS
			A:   the numpy array
			maxval:  Value to be mapped to white.
			        Default maxval=None uses maximum value in array.
			magnify:  integer.  Magnify size of image as displayed.
			        Default 1.  Integers only.  
			        Don't use more than maybe 10 or 20, only for small images.
			inv:  show zero as white, maxval as black.
	"""
	
	# Deal with lists/tuples for color images
	if isinstance(A, (list,tuple)):
		if len(A)==3:
			shorgb(A[0],A[1],A[2], maxval=maxval, magnify=magnify, inv=inv)
			return
		else:
			# probably is a useless 1-element list/tuple wrapping the data
			A = A[0]
	
	# Note: we won't know what to do if given a 3xNx3 array.
	# Solution: Insist on 4x4 or larger image.
	# This is done in shorgb().
	if len(A.shape)==2:
		# Someday: allow for colorizing, non-black zero, non-white 1.0
		shorgb(A,A,A, maxval=maxval, magnify=magnify, inv=inv)
	
	elif len(A.shape)==3:
		if A.shape[0]==3 and A.shape[2]>3:
			shorgb(A[0,:,:],A[1,:,:],A[2,:,:], maxval=maxval, magnify=magnify, inv=inv)
		elif A.shape[2]==3:
			shorgb(A[:,:,0],A[:,:,1],A[:,:,2], maxval=maxval, magnify=magnify, inv=inv)
		else:
			printf("Array has three dimensions but no RGB dimension.  Shape=", A.shape)
			return
	
	elif len(A.shape) > 3:
		print("Array has %d (too many) dimensions"  %  (len(A.shape), ) )
		return
	else:
		print("Array must have two or three dimensions")
		return



def shorgb(r,g,b, maxval=None, magnify=1, inv=False):
	"""
		Display three 2D numpy arrays as a color image
		ARGS
		    same as for sho()
	"""
	if not (r.shape==g.shape and g.shape==b.shape):
		print("Dimensions of R,G,B don't match. ", r.shape, g.shape, b.shape)
		return
		
	if not len(r.shape)==2:
		print("A must be two dimensions")
		return
	
	if r.shape[0]<4 or r.shape[1]<4:
		print("images must be at least 4x4")
		return
	
	A = np.stack( [r,g,b], axis=2).astype('float32')
	
	minval=0.0
	if maxval is None:
		maxval = np.max(A)
	
	A = ((A-minval)*255.0/maxval).clip(0.0, 255.0).astype(np.uint8) 
	if inv:
		A = 255 - A
	
	if magnify>1:
		A = np.repeat( A, magnify, axis=0)
		A = np.repeat( A, magnify, axis=1)
	
	Image.fromarray(A).show()



def EightStrips(images, labels, xspan):
	"""
		Display several images as vertical strips in one image.
		Great for comparing different effects of filters, or time series of images.

		ARGS
			images: list or tuple of several numpy arrays to use as images.
			        These may be 2D (monochrom) or 3D (rgb color).
			labels: list/tuple of strings to print along bottoms of each strip.
			xspan:  column index range, location of vertical strip to take from images.
			mag:    magnify images.  Integer only. Default magnify=1.
		RETURN
			image, as a PIL Image object.
			         Height = height of input images,
			         Width = width of xspan range * number of images.
			         
		To view result, use  Image.show() method on returned object.
		Save as file with Image.write('somename.png')
	"""
	# to be written...
	pass  



def MakeXY(arraysize, center=False):
	"""
	Create two 2D arrays, filled with x coordinate values and y coordinate values
	Arrays are sized (arraysize, arraysize)
	ARGS
		arraysize: (W,H) integers - size of image
		centered: put (0,0) at center of image
	RETURN
		(xcoords, ycoords)    tuple with coord value arrays
	"""
	if center:
	   icenter1 = arraysize[1] // 2
	   icenter0 = arraysize[0] // 2
	else:
	   icenter0 = icenter1 = 0
	xx = np.outer( np.ones(arraysize[0]),  np.arange(arraysize[1])-icenter1 )
	yy = np.outer( np.arange(arraysize[0])-icenter0, np.ones(arraysize[1])    )
	return xx,yy



def ReadImageFile(fname):
	"""
		Read an image file, any format supported automatically
		by PIL, making it available as three numpy arrays.
		INPUT
			fname: name of image file
		RETURN
			tuple (r,g,b) of 2D numpy arrays
			if image is grayscale, is just a single array, not a tuple with one array
	"""
	
	im=Image.open(fname)
	rrr = np.asarray(im).astype('f')
	if len(rrr.shape)==3:
		return ( rrr[:,:,0], rrr[:,:,1],  rrr[:,:,2] )
	else:
		return rrr



def WriteImageFile(fname, A,  maxval=None):
	"""
		Write data in 2D (grayscale) or 3D (color) numpy array as image file.
		ARGS
			fname: string.  Name of file to write.
			A:  the array, shaped as [H,W] or as [H,W,3] for color
			maxval:  value in A to map to maximum white.
	"""
	if not maxval:
		maxval = np.max(A)
		print ("max val = %.2f" %  (maxval,) )
	
	Z = (A.astype('float32')*255.0/maxval).clip(0.0, 255.0).astype(np.uint8) 
	Image.fromarray(Z).save(fname)



def WriteImageFileRGB(fname, r,g,b, maxval=None):
	A = np.stack( [r,g,b], axis=2)
	WriteImageFile(fname, A, maxval=maxval)


def FourStamp(im):
	H,W = im.shape
	z = np.empty( (H*2, W*2), dtype=im.dtype)   # doesn't work, too lazy to debug
	z[0:H, 0:W]=im
	z[H:,  0:W]=im
	z[0:H, W: ]=im
	z[H:,  W: ]=im
	return z

def HalfSize(im):
	"""
		Reduce an image to half the size.
		Each pixel is average of four from the original.
		Lops off final row/column if odd width or height.
		
		INPUT
		im:  image in 2D numpy array
		RETURN
			Halfsized image in numpy array.
	"""
	
	h,w=im.shape
	h=(h//2)*2
	w=(w//2)*2
	s= im[0:h:2,0:w:2] + im[1:h:2,0:w:2] + im[1:h:2,1:w:2] + im[0:h:2,1:w:2]
	return s/4.0



def StandardRGB_to_XYZ(sR,sG,sB):
	"""
		Convert sRGB to CIE XYZ
		Assumes R,G,B are byte values, 0 to 255
	"""
	def linearize(r):
		rhot = r > 0.04045
		r1 = r/12.92
		r2 = ( (r+0.055)/1.055 )**2.4
		return  rhot*r2 + (1-rhot)*r1
		
	r = linearize( (sR/255).clip(0,255) )
	g = linearize( (sG/255).clip(0,255) )
	b = linearize( (sB/255).clip(0,255) )
	r*=100.0
	g*=100.0
	b*=100.0
	X = 0.4124*r + 0.3576*g + 0.1805*b
	Y = 0.2126*r + 0.7152*g + 0.0722*b
	Z = 0.0193*r + 0.1192*g + 0.9505*b
	return X,Y,Z



def XYZ_to_LAB(X,Y,Z, ref=[100,94,108]):
	"""
		Convert CIE XYZ to LAB
	"""
	def curve(x):
		xhot = x > 0.008856
		x1 = 7.787*x + 16.0/116.0
		x2 = x ** (1./3.)
		return xhot*x2 + (1-xhot)*x1
		
	xn = curve(X/ref[0])
	yn = curve(Y/ref[1])
	zn = curve(Z/ref[2])
	return  116.0*yn - 16.0,  500*(xn-yn),  200*(yn-zn)


def LAB_to_XYZ(L,A,B, ref=[100,94,108]):
	def fixc(t):
		thot = t > (0.008856**(1./3.))
		t1 = (t-16.0/116.0) / 7.787
		t2 = t**3
		return thot*t2 + (1-thot)*t1
	
	ty = (L + 16)/116.0
	tx = A / 500.0 + ty
	tz = ty - B / 200.0
	return  ref[0]*fixc(tx), ref[1]*fixc(ty), ref[2]*fixc(tz)

def XYZ_to_StandardRGB(X,Y,Z):
	def nonlin(v):
		v = (v/100.0).clip(0,1.0)
		vhot = v > 0.0031308
		a1 = 12.92*v
		a2 = 1.055*(v**(1/2.4)) - 0.055
		return vhot*a2 + (1-vhot)*a1
	
	vr =   3.2406*X - 1.5372*Y - 0.4986*Z
	vg =  -0.9689*X + 1.8758*Y + 0.0415*Z
	vb =   0.0557*X - 0.2040*Y + 1.0570*Z
	return 255*nonlin(vr), 255*nonlin(vg), 255*nonlin(vb)
	
