#%%

# '#%%' allows me to run matplotlib on the Jupyter server

import numpy as np
from numpy.random.mtrand import randn
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr
from random import normalvariate

#--- BASIC ARRAY CREATION ---
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1) # creating arrays

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]] #looks like a matrix when printed
arr2 = np.array(data2)

arr2.ndim #gives the rows in an array or how many lists within the array
arr1.shape #gives the shape columns by rows

arr2.dtype #gives the type of array

np.zeros(10) #gives a zero array of 10 columns
np.ones(10) #gives a ones array of 10 columns
np.zeros((3,6)) #zeros array with 3     s and 6 columns
#(rows,columns)
np.empty((2,4,3)) #gives 2 arrays of 4 rows and 3 columns
#(arrays, rows, columns)
#np.empty does not always return an array of all zeros, it returns almost zero values

np.arange(4) #creates an ordered array of len 4

np.eye(4) #creates an identity matrix of 4 rows and columns
#---------------------------------------------------------------------


#--- ARRAY TYPES ---
arr1.astype(str)#astype allows arrays to be converted to different types
arr1.astype(arr2.dtype) #u can convert an array to the same type as anothers
#----------------------------------------------------------------------


#--- OPERATIONS BETWEEN ARRAYSAND SCALARS ---
#batch operations on data can be expressed without writing any for loops
#arithmetic operations with scalars are as you would expect it interacts with each element.
#--------------------------------------------------------------------------


#--- BASIC INDEXING & SLICING ---
# 1 dim arrays work like python lists
#however unlike lists slices of arrays are views of the original array,
#so anny changes to the slice will be reflected in the source array.
arr1[5:7]=12 #means index 5 and 6 now equals the scalar value (12)
#to COPY a slice of the array use:
arr1[5:7].copy()

#There are many more options available with higher dimensional arrays

#for 2 dimensional arrays...
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2] #gives the second index --> [7, 8, 9]
#thinking of 2-d arrays as matrices elements and indexes can be selected using this format:
# arr2d[row][column] or arr2d[row,column]
arr2d[0][2] #gives --> 3
arr2d[0,2] #gives --> 3

#for 3 dimensional arrays
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
#3-d arrays contain 2 arrays and elements and indexes can be selected like this:
# arr3d[array, row, column]
arr3d[1,0,2] #gives --> 9

#by mixing integer indexes and slices, you get a lower dimensional view of the array
arr3d[1:,:,1:2] #gives [[8]
#                       [11]]
#-------------------------------------------------------------------------


#--- BOOLEAN INDEXING ---
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = randn(7,4) #return an array of shape (column,rows) of random values from the standard normal distribution

names=='Bob' #states where within the names array this boolean statement is true
data[names=='Bob'] #returns data[0] and data[3] as names[0] and [3] =='Bob'
#The boolean array must be of the same length as the axis it is indexing so...
#   - boolean column length must match the row length of the other array
data[names=='Bob', 2:] #returns values from data[:,2:]
#as names=='Bob' already filter the row - everything after the ',' filters the columns
data[names!='Bob'] == data[~(names=='Bob')] #these are the same things
# We can also use other boolean arithmetic operators like:
#   & - and
#   ¦ - or
# python keywords 'and' and 'or' don't work with boolean arrays
data[data < 0] = 0 #sets all negative values to zero
#---------------------------------------------------------------------------


#--- FANCY INDEXING ---
#This refers to indexing using integer arrays
#unlike slicing it always copies data into a new array
arr = np.empty((8,4))
for i in range(8):
    arr[i] = i #row i has elemental values of i
arr[[4,3,0,6]] #returns a matrix of arr's 5th, 4th, 1st and 7th row
arr[[-4,-5,-8,-2]] #returns the same thing as arr[[4,3,0,6]]

arr = np.arange(32).reshape((8,4)) #shapes a 1-D array into 8x4 matrix

arr[[1, 5, 7, 2], [0, 3, 1, 2]]
#the above selects the elements: (1,0),(5,3),(7,1),(2,2) from arr
#----------------------------------------------------------------------------


#--- TRANSPOSING AND SWAPPING AXES ---
#transposing is a special form of reshaping and returns a view
arr= np.arange(15).reshape((3,5))
arr.T #returns a transposed view of arr

#to do transpose of arr has been multiplied by arr
np.dot(arr.T,arr) #np.dot() is used for matrix dot product computations

arr = np.arange(16).reshape((2,2,4)) #a 3-D array - reshape(dim,rows,cols)
arr.transpose((1,0,2)) #axis 0 and 1 are swapped, axis 2 remains un changed

arr.swapaxes(1,2) 
#axis 0 - most outer axes
#largest axis are the most inner
#-----------------------------------------------------------------------------------


#--- UNIVERSAL FUNCTIONS: FAST ELEMENT-WISE ARRAY FUNCTIONS ---
#ufunc's are elementwise operations on data in ndarrays
arr= np.arange(10)
np.sqrt(arr) #elementwise square root of arr
np.exp(arr) #elementwise exponent of arr

x = randn(8)
y = randn(8)
np.maximum(x,y) #creates an array that contains the max-elemental values of x and y
np.minimum(y,x) #does the opposite of maximum

arr = randn(7) * 5
#print(arr)
np.modf(arr)
#???????????????
#??????????????????
#????????????????????
#---------------------------------------------------------------------


#--- DATA PROCESSING USING ARRAYS ---
#Using numpy arrays lets u express data processing tasks as array expressions
#without the need to write loops.
#This is known as vectorisation and code runs a lot faster this way

#suppose we wantedto evaluate the function sqrt(x^2 + y^2) across a grid of values
points = np.arange(-5,5,0.01) #creates 1000 equally spaced points
xs, ys = np.meshgrid(points, points) #xs and ys both have the values 'points'
z = np.sqrt(xs ** 2 + ys ** 2)

plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
#imshow used to create image plot from 2-D array of function values
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
# '$\sqrt{x^2 + y^2}$' encases the text in the sqr root symbol
#----------------------------------------------------------------------------------


#--- EXPRESSING CONDITIONAL LOGIC AS ARRAY OPERATIONS ---
#the numpy.where function is a vectorized version of the expression:
# x if condition else y

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
#suppose we want to take a value from xarr whenever the correspoding value in cond is True
#otherwise we take the yarr value
#we can do
result = [(x if c else y)
            for x, y, c in zip(xarr, yarr, cond)]
#this has multiple problems:
#1  - it wont be fast for large arrays as all the work is being done in pure python
#2  - it won't work for multi-dimensional arrays
#with np.where it can be written much more concisely

result = np.where(cond, xarr, yarr) #a lot quicker and smoother
#2nd and 3rd argments of cond don't acc have to be arrays and can be scalars

#suppose you had a random matrix and wanted to replace all +'s with 2 and -'s with -2
arr = randn(3,5)
newArr = np.where(arr<0, -2, 2)

#where can be used to express more complicated logic
"""
result = []
for i in range(n):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:
        result.append(3)
"""# can be expressed more cleanly like this
"""
result = np.where(cond1 & cond2, 0,
                np.where(cond1, 1,
                np.where(cond2,2,3)))"""
#where completes removes the need for if, else statements for numerical logic operations
#------------------------------------------------------------------------------


#--- MATHEMATICAL AND STATISTICAL METHODS ---
arr = np.random.randn(5,4)
arr.mean() #finds the mean of the array
np.mean(arr) #also finds the mean of the array 'arr'
arr.sum() #sums the array

#functions like mean and sum take optional axis arguments
arr.mean(axis=1) #the mean along axis 1
arr.sum(0) #the sum along axis 0

#other methods like cumsum and cumprod do not aggregate
#check out the print statements below
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr.cumsum(0) #cumulative sum of elements on axis=0
#print(arr.cumsum(0))
arr.cumprod(1) #cumulative multiplication of elements on axis=1
#print(arr.cumprod(1))
#------------------------------------------------------------------------------


#--- METHODS FOR BOOLEAN ARRAYS ---
#boolean values are coerced to 1 (true) and 0 (false) in the above methods.
#so sum is often used to count true values in a boolean array.
arr = randn(100)
(arr>0).sum() #sum of positive values

#any and all are vital for boolean arrays
#any tests wether one or more values in an array is true
#all checks if every value is true
bools = np.array([False, False, True, False])
bools.any() #outputs True
bools.all() #outputs False
#these methods also work with non-bool arrays where nonzero evaluaes to True
#----------------------------------------------------------------------------------


#--- SORTING ---
arr = randn(8)
arr.sort() #sorts in numerical order (least to most)

#multi-dim arrays can have each dim section sorted
arr = randn(5,3)
arr.sort(1) #sorts the array along axis=1
#this returns a sorted copy of the array rather than modifying the original
#--------------------------------------------------------------------


#--- UNIQUE AND OTHER SET LOGIC ---
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names) #outputs unique values in the array

#np.in1d, tests membership of vaues from one array in another
values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])
#outputs:
#        array([ True, False, False, True, True, False, True], dtype=bool)
#-----------------------------------------------------------------------


#--- SAVING ARRAYS ON DISK IN BINARY FORMAT ---
"""
arr = np.arange(10)
np.save('some array', arr)  #saves the array as a .npy file
np.load('some array.npy')   #loads the saved file

#multiple array can be saved in a zipfile using np.savez
np.savez('array_archive.npz', a=arr, b=arr) #two arrays are saved here
arch = np.load('array_archive.npz')
arch['b'] #outputs array b
#this works like a dict object
"""
#-------------------------------------------------------------------------


#--- SAVING AND LOADING TEXT FILES ---
"""
Take a simple case of a comma-separated file (CSV) like this:

In [191]: !cat array_ex.txt
0.580052,0.186730,1.040717,1.134411
0.194163,-0.636917,-0.938659,0.124094
-0.126410,0.268607,-0.695724,0.047428
-1.484413,0.004176,-0.744203,0.005487
2.302869,0.200131,1.670238,-1.881090
-0.193230,1.047233,0.482803,0.960334

This can be loaded into a 2D array like so:

In [192]: arr = np.loadtxt('array_ex.txt', delimiter=',')
In [193]: arr
Out[193]:
array([[ 0.5801, 0.1867, 1.0407, 1.1344],
 [ 0.1942, -0.6369, -0.9387, 0.1241],
 [-0.1264, 0.2686, -0.6957, 0.0474],
 [-1.4844, 0.0042, -0.7442, 0.0055],
 [ 2.3029, 0.2001, 1.6702, -1.8811],
 [-0.1932, 1.0472, 0.4828, 0.9603]])
"""
#np.savetxt perform the inverse operation
#----------------------------------------------------------------------


#--- LINEAR ALGEBRA ---
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x.dot(y) # equivalently np.dot(x, y)

X = randn(5,5)
mat = X.T.dot(X) #transposes X and dot multiplies it by X
inv(mat)
mat.dot(inv(mat)) #gives a 5x5 ones matrix
q, r = qr(mat)
print(q)
print(r) #what do either of q and r output?????
#???????????????????
#??????????????????????????
#????????????????????????????
#---------------------------------------------------------------------


#--- RANDOM NUMBER GENERATION ---
#you can get samples from the standard normal distribution using normal
#pythons random is a LOT slwower than np.random!
#---------------------------------------------------------------------

# %%