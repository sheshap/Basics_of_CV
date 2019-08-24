#This script is to read text data separated by semicolon, normalize it, center it and write to text file
import numpy as np
#read text data
input = np.loadtxt('1.txt', delimiter=';')
#find minimum of each column
mins = np.min(input,axis=0)
#find maximum of each column
maxs = np.max(input, axis=0)
#difference of maximum and minimum
diff_max_min = maxs-mins
#normalize
norm = (input-mins)/diff_max_min
np.savetxt('norm.txt',norm)
#center normalized data with mean to get centered data
center = norm - np.mean(norm, axis=0)
np.savetxt('normcenter.txt',center)
#randomly select 512 rows
np.savetxt('random.txt',center[np.random.choice(center.shape[0], 512 , replace=False), :])
