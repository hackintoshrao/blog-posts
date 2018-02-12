import numpy as np 
import time 

# generate milliion numbers between 0 to 1.
a = np.random.randn(1000000)
b = np.random.randn(1000000)

# calculating the computation time.
start = time.time()
# multiplying the vectorized way.
vector_output = np.dot(a, b)

end = time.time()

print("Vectorized output: ",vector_output)
print("Time taken in milliseconds: ", str((end - start) * 1000))

for_loop_output = 0
# Now calculate the time taken for the loop based calculation of dor product.
start = time.time()
# Calculate the dot product using the loop.
for i in range(1000000):
	for_loop_output += a[i] * b[i]

end = time.time()


print("")
print("For loop output: ", for_loop_output)
print("Time taken in milliseconds: ", str((end - start) * 1000))
