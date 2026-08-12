import numpy as np
       # x,y,r,g,b
data = [[3,2], 
        [6,3], 
        [2,4], 
        [7,4], 
        [8,5], 
        [3,7], 
        [6,8], 
        [5,9], 
        [2,9]
        ]

data = np.array(data, dtype=np.float64)
numD = data.shape[0]
print("Data Count:", numD)

k = 2
C = [[2,6],
     [6,5]
     ]
C = np.array(C, dtype=np.float64)
numC = C.shape[0]

distance = np.zeros((numD, k))
iter = 2

for it in range(iter):
    for cntD in range(numD):
        for cntC in range(numC):
            distance[cntD, cntC] = np.linalg.norm(data[cntD] - C[cntC])
    labels = np.argmin(distance, axis=1)
    wcss = 0
    for cntC in range(numC):
        indices_of_C = np.nonzero(labels == cntC)
        if len(indices_of_C[0]) > 0:
            wcss += np.sum(np.linalg.norm(data[indices_of_C] - C[cntC], axis=1)**2)
            C[cntC] = np.mean(data[indices_of_C], axis=0)

    print("Iteration ",it)
    print("Labels:", labels)
    print("WCSS:", wcss)