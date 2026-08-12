# run_kmeans.py

# wcss = within-cluster sum of sqaure : ยิ่งทำควรยิ่งต่ำจนคงที่ไป
# เหมือนตัววัดประสิทธิภาพ

# iteration: 5
# label: [0 0 0 0 0 0 0 0 0 1]
# wcss: 708495.1111111111

import numpy as np
       # x,y,r,g,b
data = [[193,131,226,243,242], 
        [483,85,0,155,255], 
        [449,263,232,252,200], 
        [479,497,91,82,121], 
        [551,553,108,101,135], 
        [679,483,255,240,172], 
        [685,485,255,191,51], 
        [751,513,207,147,37], 
        [611,641,0,141,255],
        [799,753,233,255,207]
        ]

data = np.array(data, dtype=np.float64)
numD = data.shape[0]
print("Data Count:", numD)

k = 3
C = [[627,743,106,179,59],
     [609,889,253,243,237],
     [613,991,252,213,136]
     ]
C = np.array(C, dtype=np.float64)
numC = C.shape[0]

distance = np.zeros((numD, k))
iter = 20

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
