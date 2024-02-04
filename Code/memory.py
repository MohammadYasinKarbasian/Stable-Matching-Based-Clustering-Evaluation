import math

int_size = 2 # 2 bytes
cluster_size =  1000
permutation_size = math.factorial(cluster_size)*(cluster_size*int_size)

print(f'{permutation_size} Bytes')
print(f'{permutation_size/1024} KB')
print(f'{permutation_size/1024/1024} MB')
print(f'{permutation_size/1024/1024/1024} GB')
print(f'{permutation_size/1024/1024/1024/1024} TB') 