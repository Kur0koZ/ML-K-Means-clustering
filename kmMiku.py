# run_kmeans.py

# wcss = within-cluster sum of sqaure : ยิ่งทำควรยิ่งต่ำจนคงที่ไป
# เหมือนตัววัดประสิทธิภาพ



# meh kinda suck, maybe i will make a picture with 10 colours
# iteration: 4
# label: [2 0 0 0 1 0 0 1 2 2]
# wcss: 18137.297333333332

import numpy as np 
       # x,y,r,g,b
data = [[31, 26, 47.1, 82.0, 79.6],
        [25, 65, 24.3, 55.7, 55.3],
        [24, 91, 19.2, 26.3, 31],
        [21, 76, 8.2, 9.0, 18.8],
        [57, 71, 80.8, 84.3, 87.1],
        [86, 16, 79.2, 24.7, 25.9],
        [80, 4, 19.6, 27.5, 31.8],
        [51, 51, 100.0, 92.9, 87.5],
        [13, 17, 100.0, 100.0, 100.0],
        [48, 17, 65.9, 96.1, 92.9]
]

data = np.array(data, dtype = np.float64)
numD = data.shape[0]
print(f"\ndata count: {numD}")

# print(data)

k = 3 # i guess like column maybe ? or how many group

c = [[90, 45, 47.1, 82.0, 79.6], 
     [73, 80, 100.0, 92.9, 87.5],
     [70, 63, 100.0, 100.0, 100.0]
] # how many class ?

c = np.array(c, dtype = np.float64)
numC = c.shape[0]

distance = np.zeros((numD, k))
iteration = 20

for iterate in range(iteration):

    for countD in range(numD):

        for countC in range(numC):

            distance[countD, countC] = np.linalg.norm(data[countD] - c[countC])

    label = np.argmin(distance, axis = 1)

    wcss = 0


    # find a way to break loop if wcss doesnt change for a few time maybe

    for countC in range(numC):

        indiceOfC = np.nonzero(label == countC)

        if len(indiceOfC[0]) > 0:

            wcss += np.sum(np.linalg.norm(data[indiceOfC] - c[countC], axis = 1) ** 2)
            c[countC] = np.mean(data[indiceOfC], axis = 0)

    print(f"\n")
    print(f"iteration: {iterate}")
    print(f"label: {label}")
    print(f"wcss: {wcss}")