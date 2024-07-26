                                            # NumPy Getting Started

# import numpy

# arr = numpy.array([1, 2, 3, 4, 5])

# print(arr)


                                            # NumPy Creating Arrays


import numpy as np

# arr = np.array([1, 2, 3, 4, 5])

# print(arr)


# print(np.__version__)

# arr = np.array([1, 2, 3, 4, 5])

# print(arr)

# print(type(arr))

# arr = np.array((1, 2, 3, 4, 5))

# print(arr)




                                            # Dimensions in Arrays
 
 
                                            #  1-D Arrays
 
# arr = np.array(42)

# print(arr)


# arr = np.array([1, 2, 3, 4, 5])

# print(arr)

                                            # 2-D Arrays
   
   
                                            
# arr = np.array([[1, 2, 3], [4, 5, 6]])

# print(arr)


                                            # 3-D arrays
                                            
                                            
# arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

# print(arr)


                                            # Check Number of Dimensions?
                                            


# a = np.array(42)
# b = np.array([1, 2, 3, 4, 5])
# c = np.array([[1, 2, 3], [4, 5, 6]])
# d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

# print(a.ndim)
# print(b.ndim)
# print(c.ndim)
# print(d.ndim)


                                            # Higher Dimensional Arrays
                                            
                                            

# import numpy as np

# arr = np.array([1, 2, 3, 4], ndmin=5)

# print(arr)
# print('number of dimensions :', arr.ndim)



                                            # NumPy Array Indexing
                                            
                                            
# arr = np.array([1, 2, 3, 4])

# print(arr[0])


# arr = np.array([1, 2, 3, 4])

# print(arr[2] + arr[3])


                             
                                            # Access 2-D Arrays  
                                            
                                               
                                                                              
# arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

# print('2nd element on 1st row: ', arr[0, 1])




                                            # Access 3-D Arrays
                                            
 
 
 
 
#  {
      
# arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# print(arr[0, 1, 2]) 
    

            # The first number represents the first dimension

            # Since we selected 0, we are left with the first array:

            # The third number represents the third dimension, which contains three values: 

#  }



                                            # Negative Indexing
                                            
                                            


# arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

# print('Last element from 2nd dim: ', arr[1, -1])




                                            # NumPy Array Slicing
                                            
                                            
 
# arr = np.array([1, 2, 3, 4, 5, 6, 7])

# print(arr[1:5])

# arr = np.array([1, 2, 3, 4, 5, 6, 7])

# print(arr[4:])

# arr = np.array([1, 2, 3, 4, 5, 6, 7])

# print(arr[:4])



                                            # Negative Slicing
                                            
                                            
                                

# arr = np.array([1, 2, 3, 4, 5, 6, 7])

# print(arr[-3:-1])

# arr = np.array([1, 2, 3, 4, 5, 6, 7])

# print(arr[1:5:2])

# arr = np.array([1, 2, 3, 4, 5, 6, 7])

# print(arr[::2])



                                            # Slicing 2-D Arrays
                                            
                                            


# arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

# print(arr[1, 1:4])



            # Note: Remember that second element has index 1.
            
            


# arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

# print(arr[0:2, 2])

# arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

# print(arr[0:2, 1:4]) 
    


# NumPy Data Types

# arr = np.array([1, 2, 3, 4])

# print(arr.dtype)                                      



# arr = np.array(['apple', 'banana', 'cherry'])

# print(arr.dtype)




                                            # Creating Arrays With a Defined Data Type
                                            
                                            

# arr = np.array([1, 2, 3, 4], dtype='S')

# print(arr)
# print(arr.dtype)    



# arr = np.array([1, 2, 3, 4], dtype='i4')

# print(arr)
# print(arr.dtype)        

  

                                            # What if a Value Can Not Be Converted?      
 
 
 
            # A non integer string like 'a' can not be converted to integer (will raise an error):

# arr = np.array(['a', '2', '3'], dtype='i')   


# arr = np.array([1.1, 2.1, 3.1])

# newarr = arr.astype('i')

# print(newarr)
# print(newarr.dtype)        



# arr = np.array([1.1, 2.1, 3.1])

# newarr = arr.astype(int)

# print(newarr)
# print(newarr.dtype)                                                        
                          
                          
# arr = np.array([1, 0, 3])

# newarr = arr.astype(bool)

# print(newarr)
# print(newarr.dtype)      



                                            # NumPy Array Copy vs View
                                            
 
# arr = np.array([1, 2, 3, 4, 5])
# x = arr.copy()
# arr[0] = 42

# print(arr)
# print(x)


# arr = np.array([1, 2, 3, 4, 5])
# x = arr.view()
# arr[0] = 42

# print(arr)
# print(x)

# arr = np.array([1, 2, 3, 4, 5])
# x = arr.view()
# x[0] = 31

# print(arr)
# print(x)



                                            # Check if Array Owns its Data


# arr = np.array([1, 2, 3, 4, 5])

# x = arr.copy()
# y = arr.view()

# print(x.base)
# print(y.base)


            # The copy returns None.
            # The view returns the original array. 
            
                
 
 
                                            # NumPy Array Shape                                                 


# arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# print(arr.shape)


# arr = np.array([1, 2, 3, 4], ndmin=5)

# print(arr)
# print('shape of array :', arr.shape)




                                            # NumPy Array Reshaping
                                            
                            
                                            
                                            
# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# newarr = arr.reshape(4, 3)

# print(newarr)




# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# newarr = arr.reshape(2, 3, 2)

# print(newarr)



                                            # Can We Reshape Into any Shape?
                                            

            # Try converting 1D array with 8 elements to a 2D array with 3 elements in each dimension (will raise an error):                                         


# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# newarr = arr.reshape(3, 3)

# print(newarr)



                                            # Returns Copy or View?
                                            
                                            

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# print(arr.reshape(2, 4).base)


                                            # Unknown Dimension


# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# newarr = arr.reshape(2, 2, -1)

# print(newarr)




            # Note: We can not pass -1 to more than one dimension.
            
            

                                            # Flattening the arrays
                                            
                                            


# arr = np.array([[1, 2, 3], [4, 5, 6]])

# newarr = arr.reshape(-1)

# print(newarr)



            # Note: There are a lot of functions for changing the shapes of arrays in numpy flatten, ravel and also for rearranging the elements rot90, flip, fliplr, flipud etc. These fall under Intermediate to Advanced section of numpy.



                                            # NumPy Array Iterating


# arr = np.array([1, 2, 3])

# for x in arr:
#   print(x)
  
  
                                            # Iterating 2-D Arrays
  
  
  
  
# arr = np.array([[1, 2, 3], [4, 5, 6]])

# for x in arr:
#   print(x)
  
  
            # If we iterate on a n-D array it will go through n-1th dimension one by one.
  
  
  
# arr = np.array([[1, 2, 3], [4, 5, 6]])

# for x in arr:
#   for y in x:
#     print(y)
    
    
    
                                            # Iterating 3-D Arrays
                                            
    
    
# arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# for x in arr:
#   print(x)
  
  
  
#arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# for x in arr:
#   for y in x:
#     for z in y:
#       print(z)
      
      
                                            #Iterating Arrays Using nditer()
      
      
      
      
# arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# for x in np.nditer(arr):
#   print(x)
  
  
  
                                            #Iterating Array With Different Data Types
  
  
  
#arr = np.array([1, 2, 3])

# for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
#   print(x)
  
  
                                            #Iterating With Different Step Size
  
  
  
#arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# for x in np.nditer(arr[:, ::2]):
#   print(x)
        
        
        
                                            #Enumerated Iteration Using ndenumerate()
        
        
        
        
# arr = np.array([1, 2, 3])

# for idx, x in np.ndenumerate(arr):
#   print(idx, x)
  
  
  
#arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# for idx, x in np.ndenumerate(arr):
#   print(idx, x)



                                            # NumPy Joining Array
                                            
 
# arr1 = np.array([1, 2, 3])

# arr2 = np.array([4, 5, 6])

# arr = np.concatenate((arr1, arr2))

# print(arr)      



# arr1 = np.array([[1, 2], [3, 4]])

# arr2 = np.array([[5, 6], [7, 8]])

# arr = np.concatenate((arr1, arr2), axis=1)

# print(arr)



                                            #Joining Arrays Using Stack Functions



# arr1 = np.array([1, 2, 3])

# arr2 = np.array([4, 5, 6])

# arr = np.stack((arr1, arr2), axis=1)

# print(arr)



                                            # Stacking Along Rows



# arr1 = np.array([1, 2, 3])

# arr2 = np.array([4, 5, 6])

# arr = np.hstack((arr1, arr2))

# print(arr)



                                            # Stacking Along Columns



# arr1 = np.array([1, 2, 3])

# arr2 = np.array([4, 5, 6])

# arr = np.vstack((arr1, arr2))

# print(arr)



                                            # Stacking Along Height (depth)



# arr1 = np.array([1, 2, 3])

# arr2 = np.array([4, 5, 6])

# arr = np.dstack((arr1, arr2))

# print(arr)




                                            # NumPy Splitting Array
                                            
                                            


# arr = np.array([1, 2, 3, 4, 5, 6])

# newarr = np.array_split(arr, 3)

# print(newarr)



            # Note: The return value is a list containing three arrays.
            


# arr = np.array([1, 2, 3, 4, 5, 6])

# newarr = np.array_split(arr, 4)

# print(newarr)



            # Note: We also have the method split() available but it will not adjust the elements when elements are less in source array for splitting like in example above, array_split() worked properly but split() would fail.



                                            # Split Into Arrays



# arr = np.array([1, 2, 3, 4, 5, 6])

# newarr = np.array_split(arr, 3)

# print(newarr[0])
# print(newarr[1])
# print(newarr[2])




                                            # Splitting 2-D Arrays



# arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

# newarr = np.array_split(arr, 3)

# print(newarr)




# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

# newarr = np.array_split(arr, 3)

# print(newarr)




# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

# newarr = np.array_split(arr, 3, axis=1)

# print(newarr)




            # An alternate solution is using hsplit() opposite of hstack()
            



# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

# newarr = np.hsplit(arr, 3)

# print(newarr)



            # Note: Similar alternates to vstack() and dstack() are available as vsplit() and dsplit().
            
            
            
            
                                            #  NumPy Searching Arrays           
                                     
 
 
# arr = np.array([1, 2, 3, 4, 5, 4, 4])

# x = np.where(arr == 4)

# print(x)  


            # The example above will return a tuple: (array([3, 5, 6],)

            # Which means that the value 4 is present at index 3, 5, and 6.  


# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# x = np.where(arr%2 == 0)

# print(x)




# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# x = np.where(arr%2 == 1)

# print(x)     




                                            # Search Sorted
                                            
                                            
                                            


            # The searchsorted() method is assumed to be used on sorted arrays.



# arr = np.array([6, 7, 8, 9])

# x = np.searchsorted(arr, 7)

# print(x)




# arr = np.array([6, 7, 8, 9])

# x = np.searchsorted(arr, 7, side='right')

# print(x)



# arr = np.array([1, 3, 5, 7])

# x = np.searchsorted(arr, [2, 4, 6])

# print(x)




            # The return value is an array: [1 2 3] containing the three indexes where 2, 4, 6 would be inserted in the original array to maintain the order.  
            
            
            
            
            
                                            # NumPy Sorting Arrays
          
            
            
# arr = np.array([3, 2, 0, 1])

# print(np.sort(arr))  


            # Note: This method returns a copy of the array, leaving the original array unchanged.




# arr = np.array(['banana', 'cherry', 'apple'])

# print(np.sort(arr))




# arr = np.array([True, False, True])

# print(np.sort(arr))



                                            # Sorting a 2-D Array



# arr = np.array([[3, 2, 4], [5, 0, 1]])

# print(np.sort(arr))                       





                                            # NumPy Filter Array
                                            
                                            

            # A boolean index list is a list of booleans corresponding to indexes in the array.



# arr = np.array([41, 42, 43, 44])

# x = [True, False, True, False]

# newarr = arr[x]

# print(newarr)



                                        # Creating the Filter Array



# arr = np.array([41, 42, 43, 44])

            # Create an empty list
            
            
# filter_arr = []

            # go through each element in arr
            
            
# for element in arr:
    
            # if the element is higher than 42, set the value to True, otherwise False:
            
            
#   if element > 42:
#     filter_arr.append(True)
#   else:
#     filter_arr.append(False)

# newarr = arr[filter_arr]

# print(filter_arr)
# print(newarr)



# arr = np.array([1, 2, 3, 4, 5, 6, 7])

            # Create an empty list

# filter_arr = []

            # go through each element in arr

# for element in arr:
    
             # if the element is completely divisble by 2, set the value to True, otherwise False
  
#   if element % 2 == 0:
#     filter_arr.append(True)
#   else:
#     filter_arr.append(False)

# newarr = arr[filter_arr]

# print(filter_arr)
# print(newarr)


                                            # Creating Filter Directly From Array



# arr = np.array([41, 42, 43, 44])

# filter_arr = arr > 42

# newarr = arr[filter_arr]

# print(filter_arr)
# print(newarr)



arr = np.array([1, 2, 3, 4, 5, 6, 7])

filter_arr = arr % 2 == 0

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)                                          
                                            
                                            