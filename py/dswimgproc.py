"""
	DARTOOFI: Daren's Tools for Image work -
	Additional image processing & analysis operations
	
	Simple (or almost simple) routines in Python for viewing numpy arrays
	as images, read/write them as image files, and do some basic
	image processing.
	
	Official source is on GitHub  https://github.com/darenw/dartoofi.git
	
	License: MIT
"""


from numpy import *
import scipy.ndimage as spim
import math


def Oiliate(im, r):
	return spim.filters.median_filter(im,size=r)
	

def ppnoise(im):
	"""
		Measure pixel-to-pixel noise.
		Tries to ignore edges, hard transitions, busy areas, bad pixels, and other
		phenomena that might throw the stats off.
		
		INPUT
			im - 2D numpy array image  (single color channnel only)
		
		RETURN 
			tuple with measurements
			[0] is general RMS noise level
			[1] is a tuple (rmsx,rmsy) for noise along x, along y
			[2]  (future use)
	"""
	
	a  =im[3:-5:7, 3:-5:7]
	ax =im[3:-5:7, 4:-4:7]
	axx=im[3:-5:7, 5:-3:7]
	ay =im[4:-4:7, 3:-5:7]
	ayy=im[5:-3:7, 3:-5:7]
	ddx=ax-0.5*(axx+a)
	ddy=ay-0.5*(ayy+a)
	
	# first cut of bad data, typically hard edges, cosmic rays. 
	lim=25*min( std(ddx), std(ddy))
	good = ((abs(ddx)+abs(ddy)) <lim).astype('f')
	if sum(good) < 10:
		return math.nan
	
	# subtract bias due to large-scale curvature
	ddx = ddx - sum(ddx*good)/sum(good)
	ddx = ddx - sum(ddx*good)/sum(good)
	
	# compute std dev, weighted by goodness, assuming zero mean
	stddevx = math.sqrt( sum(good*(ddx**2))/sum(good))
	stddevy = math.sqrt( sum(good*(ddy**2))/sum(good))
	
	# Repeat, imposing tighter limit on defining "good" data
	lim=6*(stddevx+stddevy)/2.
	good = ((abs(ddx)+abs(ddy)) <lim).astype('f')
	ddx = ddx - sum(ddx*good)/sum(good)
	ddx = ddx - sum(ddx*good)/sum(good)
	stddevx = math.sqrt( sum(good*(ddx**2))/sum(good))
	stddevy = math.sqrt( sum(good*(ddy**2))/sum(good))
	rms = math.sqrt(stddevx**2+stddevy**2)
	return (rms,  (stddevx, stddevy),  0)



def smooth3x3(im):
	"""
	3x3 box average. Edges are handled.
	"""
	
	zz=empty_like(im)
	zz[1:-1, :]= (im[0:-2, :] + im[1:-1,:] + im[2:,:])/3
	zz[0,:]= (1.5*im[0,:]+im[1,:]+.5*im[2,:])/3
	zz[-1,:]= (1.5*im[-1,:]+im[-2,:]+.5*im[-3,:])/3
	zz[:,1:-1]=(zz[:,0:-2]+zz[:,1:-1]+zz[:,2:])/3
	zz[:, 0]= (1.5*zz[:, 0]+zz[:, 1]+.5*zz[:,2])/3
	zz[:,-1]= (1.5*zz[:, -1]+zz[:, -2]+.5*zz[:,-3])/3
	return zz





def Edginate(img):
	"""
	   Crude edge detector
	   No parameters
	"""
	gx = gradsmoothx(img)
	gy = gradsmoothy(img)
	return np.abs(gx)+np.abs(gy)



def MultiEdginate(bub, nrepeats):
	Q = bub
	for i in range(nrepeats):
		Q = edgify(Q)
		Q = smooth3x3(Q)
		Q = np.tanh( Q / (0.5*Q.max()) )
	texture = np.sqrt(bub.clip(0)) * Q
	return texture



def SimpleEdgeDetector321(im, fuzziness=1.0):
	"""
		Detect edges using short-smoothed gradients
		suppressing noise.  
		Handles single channel only.
		ARGS
			im:   image as 2D array 
			fuzziness  lowers threshold to make edges more sensitive to noise
			 		default value 1.0
	"""
	
	ee = abs(gradsmoothx(im)) + abs(gradsmoothy(im)) 
	maxish = smooth3x3(ee).max()
	pp = ppnoise(ee)
	nn = np.sqrt(1.5*(pp[0]+pp[1]))
	thresh = np.sqrt(nn*maxish)
	t = np.tanh((ee-thresh)/(nn/fuzziness))
	return t.clip(0,.75)/0.75




def gradsmoothx(im):
	"""
	Return the y-derivative of an image, using
	short-range smoothing to reduce sensitivity to noise.
	Values are central difference.  Care is taken
	to return meaningful values along edges, at corners

	INPUT
		im: grayscale image as a 2D numpy array, or 3D for color.
		     If color, must be shaped as [H,W,3]
	RETURN
		y-gradient of image in same-size array
	"""
	zz=empty_like(im)
	if len(im.shape)==3:
		zz[:,:,0]=gradsmoothx(im[:,:,0])
		zz[:,:,1]=gradsmoothx(im[:,:,1])
		zz[:,:,2]=gradsmoothx(im[:,:,2])
		return zz
	
	# main inside area of image
	# pattern is weighted average   (1,2,1),1
	zz[1:-1, 2:-2]=(    im[1:-1, 4:  ] 
				  + 2*im[1:-1, 3:-1]
				  +   im[0:-2, 3:-1]
				  +   im[2:  , 3:-1]
				  - 2*im[1:-1, 1:-3]
				  -   im[1:-1, 0:-4]
				  -   im[0:-2, 1:-3]
				  -   im[2:  , 1:-3] )/12
	zz[1:-1, 1]=(   + 2*im[1:-1, 2]
				  +   im[0:-2, 2]
				  +   im[2:  , 2]
				  - 2*im[1:-1, 0]
				  -   im[0:-2, 0]
				  -   im[2:  , 0] )/8
	zz[1:-1, 0] = (2*im[1:-1,1]
				  +im[0:-2,1]
				  +im[2:  ,1]
				-2*im[1:-1,0]
				  -im[0:-2,0]
				  -im[2:  ,0]) /4
	zz[1:-1, -2]=(   + 2*im[1:-1, 2]
				  +   im[0:-2, 2]
				  +   im[2:  , 2]
				  - 2*im[1:-1, 0]
				  -   im[0:-2, 0]
				  -   im[2:  , 0] )/8
	zz[1:-1,-1] = (2*im[1:-1, -1]
				  +im[0:-2, -1]
				  +im[2:  , -1]
				-2*im[1:-1, -2]
				  -im[0:-2, -2]
				  -im[2:  , -2]) /4
	zz[0, 2:-2] = (     im[0, 4:  ]
				  + 2*im[0, 3: -1]
				  - 2*im[0, 1:-3]
				  -   im[0, 0:-4]   )/8
	zz[-1, 2:-2] = (    im[-1, 4:  ]
				  + 2*im[-1, 3: -1]
				  - 2*im[-1, 1:-3]
				  -   im[-1, 0:-4]   )/8
	zz[0, 1] = (2*im[0,2]+im[1,2] - 2*im[0,0]-im[1,0])/6.
	zz[0, -2] = (2*im[0,-1]+im[1,-1] - 2*im[0,-3]-im[1,-3])/6.
	zz[-1, 1] = (2*im[-1,2]+im[-2,2] - 2*im[-1,0]-im[-2,0])/6.
	zz[-1, -2] = (2*im[-1,-1]+im[-2,-1] - 2*im[-1,-3]-im[-2,-3])/6.
	zz[0,0]  = (zz[1,0]+zz[0,1])/2
	zz[-1,0] = (zz[-2,0]+zz[-1,1])/2
	zz[0,-1] = (zz[1,-1]+zz[0,-2])/2
	zz[-1,-1] = (zz[-1,-2]+zz[-2,-1])/2 
	return zz




def gradsmoothy(im):
	"""Return the y-derivative of an image, using
	a short-range smoothing to reduce sensitivity to noise.
	Values are central difference.  Care is taken
	to return meaningful values along edges, at corners

	INPUT
	   im: image as a 2D numpy array
	RETURN
	   same-size array with d/dy values
	"""

	zz=empty_like(im)
	if len(im.shape)==3:
	  zz[:,:,0]=gradsmoothy(im[:,:,0])
	  zz[:,:,1]=gradsmoothy(im[:,:,1])
	  zz[:,:,2]=gradsmoothy(im[:,:,2])
	  return zz

	# main inside area of image
	# pattern is weighted average   (1,2,1),1
	zz[2:-2, 1:-1]=(    im[ 4:  , 1:-1] 
				  + 2*im[ 3:-1, 1:-1]
				  +   im[ 3:-1, 0:-2]
				  +   im[ 3:-1, 2:]
				  - 2*im[ 1:-3, 1:-1]
				  -   im[ 0:-4, 1:-1]
				  -   im[ 1:-3, 0:-2]
				  -   im[ 1:-3, 2:  ] )/12
	zz[1, 1:-1]=(   + 2*im[2, 1:-1]
				  +   im[2, 0:-2]
				  +   im[2, 2:  ]
				  - 2*im[0, 1:-1]
				  -   im[0, 0:-2]
				  -   im[0, 2:  ] )/8
	zz[0, 1:-1] = (2*im[1, 1:-1]
				  +im[1, 0:-2]
				  +im[1, 2:  ]
				-2*im[0, 1:-1]
				  -im[0, 0:-2]
				  -im[0, 2:  ]) /4
	zz[-2, 1:-1]=(  2*im[2, 1:-1]
				  +   im[2, 0:-2]
				  +   im[2, 2:  ]
				  - 2*im[0, 1:-1]
				  -   im[0, 0:-2]
				  -   im[0, 2:  ] )/8
	zz[-1, 1:-1] = (2*im[-1, 1:-1]
				  +im[-1, 0:-2]
				  +im[-1, 2:  ]
				-2*im[-2, 1:-1]
				  -im[-2, 0:-2]
				  -im[-2, 2:  ]) /4
	zz[2:-2, 0] = (    im[ 4:  , 0]
				   + 2*im[ 3: -1, 0]
				   - 2*im[ 1:-3, 0]
				   -   im[ 0:-4, 0]   )/8
	zz[2:-2, -1] = (   im[ 4: ,  -1 ]
				   + 2*im[ 3: -1,-1]
				   - 2*im[ 1:-3, -1]
				   -   im[ 0:-4, -1]   )/8
	zz[1, 0] = (2*im[2, 0]+im[2, 1] - 2*im[0,0]-im[0, 1])/6.
	zz[-2, 0] = (2*im[-1, 0]+im[-1, 1] - 2*im[-3,0]-im[-3,1])/6.
	zz[1, -1] = (2*im[2, -1]+im[2, -2] - 2*im[0, -1]-im[0, -2])/6.
	zz[-2, -1] = (2*im[-1,-1]+im[-1,-2] - 2*im[-3,-1]-im[-3,-2])/6.

	zz[0,0]  = (zz[1,0]+zz[0,1])/2
	zz[-1,0] = (zz[-2,0]+zz[-1,1])/2
	zz[0,-1] = (zz[1,-1]+zz[0,-2])/2
	zz[-1,-1] = (zz[-1,-2]+zz[-2,-1])/2 
	return zz





def Scatter(img, Nrepeat=1, seed=9009):
	"""
		Short-range scramble pixels
		Can handle plain single-channel image, RGB color, or list/tuple of R,G,B
		ARGS
			img: 2D array, or 3D array [y,x,rgb] or [rgb,y,x], or tuple (R,G,B)
		See also:
			Scatter2D() if for sure you have a 2D array
			ScatterRGB(r,g,b) when you explicitly give three color planes
	"""
	
	# Deal with lists/tuples for color images
	if isinstance(img, (list,tuple)):
		if len(img)==3:
			return ScatterRGB(img[0], img[1], img[2], Nrepeat=Nrepeat, seed=seed)
		else:
			# Probably is a useless 1-element list/tuple wrapping the data
			# Unwrap it and go on
			img = img[0]
	
	ndims = len(img.shape)
	if ndims==2:	
		return Scatter2D(A, Nrepeat=Nrepeat, seed=seed)
	
	elif len(A.shape)==3:
		if A.shape[0]==3 and A.shape[2]>3:
			return ScatterRGB(img[0,:,:], img[1,:,:], img[2,:,:], Nrepeat=Nrepeat, seed=seed)
		elif A.shape[3]==3:
			return ScatterRGB(img[:,:,0], img[:,:,1], img[:,:,2], Nrepeat=Nrepeat, seed=seed)
		else:
			printf("Array has three dimensions but no RGB dimension.  Shape=", A.shape)
			return None
	
	elif len(A.shape) > 3:
		print("Array has %d (too many) dimensions"  %  (len(A.shape), ) )
		return None
	else:
		print("Array must have two or three dimensions")
		return None



def ScatterRGB(R,G,B, Nrepeat=1, seed=9009):
	"""
		Scatters pixels when given three color planes
		All planes get same pixel swaps.
		Takes three explicitly given 2D arrays.
		For 3D array [H,W,3] use Scatter()
		See ScatterPixels() for 
	"""
	Rs,Gs,Bs = R.copy(), G.copy(), B.copy()
	H,W = B.shape
	np.random.seed(seed)
	for n in range(Nrepeat):
		for i in range(W*H//2):
			# Do an adjacent horizontal swap
			x = np.random.randint(1,W)
			y = np.random.randint(0,H)
			Rs[y,x-1], Rs[y,x] = Rs[y,x], Rs[y,x-1] 
			Gs[y,x-1], Gs[y,x] = Gs[y,x], Gs[y,x-1] 
			Bs[y,x-1], Bs[y,x] = Bs[y,x], Bs[y,x-1] 
			# Do an adjacent vertical swap
			x = np.random.randint(0,W)
			y = np.random.randint(1,H)
			Rs[y,x], Rs[y-1,x] = Rs[y-1,x], Rs[y,x] 
			Gs[y,x], Gs[y-1,x] = Gs[y-1,x], Gs[y,x] 
			Bs[y,x], Bs[y-1,x] = Bs[y-1,x], Bs[y,x] 
	return Rs,Gs,Bs


def Scatter2D(	img, Nrepeat=1, seed=9009):
	"""
		Short-range scrambling of pixels
		Original pixel values kept, just moved around like one of those sliding tile
		puzzles.  
		This version, ScatterPixels1(), takes only simple 2D array.
		For list or tuple, or 3D array with a color channel dimension,
		use ScatterPixel()  (not having a '1' at end of name)
		Sets RNG seed according to image size, for reproducibility
		ARGS
			img:  2D array holding image
			Nrepeat: repeat the operation for more scattering. 
			       Default 1.  Reasonable everyday max around 5 to 10.
			       The bigger this is, the slower it runs.
		
	"""
	B = A.copy()
	H,W = B.shape
	np.random.seed(seed)
	for n in range(Nrepeat):
		for i in range(W*H//2):
			# Do an adjacent horizontal swap
			x = np.random.randint(1,W)
			y = np.random.randint(0,H)
			B[y,x-1], B[y,x] = B[y,x], B[y,x-1] 
			# Do an adjacent vertical swap
			x = np.random.randint(0,W)
			y = np.random.randint(1,H)
			B[y,x], B[y-1,x] = B[y-1,x], B[y,x] 
	return B


