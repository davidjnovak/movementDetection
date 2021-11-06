import cv2
import numpy as np

#define constants
image1 = cv2.imread("sphere1.jpg",0)
image2 = cv2.imread("sphere2.jpg",0)
numRows = image1.shape[0]
numCols = image1.shape[1]
mag = np.zeros((numRows,numCols,1),np.float32)
uscalar = np.zeros((numRows,numCols,1),np.float32)
vscalar = np.zeros((numRows,numCols,1),np.float32)
ixt = np.zeros((numRows,numCols,1),np.float32)
iyt = np.zeros((numRows,numCols,1),np.float32)
it = np.zeros((numRows,numCols,1),np.float32)
ix = np.zeros((numRows,numCols,1),np.float32)
iy = np.zeros((numRows,numCols,1),np.float32)
ixx = np.zeros((numRows,numCols,1),np.float32)
ixy = np.zeros((numRows,numCols,1),np.float32)
iyy = np.zeros((numRows,numCols,1),np.float32)

#finds determinant of 2x2 only
def determinant(M):
    return (M[0][0]*M[1][1])-(M[0][1]*M[1][0])

#finds the trace of a 2x2 only
def trace(M):
    return M[0][0]+M[1][1]

#finds the inverse of a 2x2 only
def inverse(M):
    newMat = [[M[1][1],-M[0][1]],[-M[1][0],M[0][0]]]
    for i in (0,1):
        for j in (0,1):
            newMat[i][j] = newMat[i][j]/determinant(M)
    return newMat


#calculate temporal gradient
for i in range(numRows):
    for j in range(numCols):
        it[i][j] = float(image2[i][j])-float(image1[i][j])

#calculate spacial gradients
for i in range(1, numRows-1):
    for j in range(1, numCols-1):
        iy[i][j] = float(image2[i][j-1]) - float(image1[i][j+1])
        ix[i][j] = float(image2[i-1][j]) - float(image1[i+1][j])
        ixx[i][j] = ix[i][j]**2
        iyy[i][j] = iy[i][j]**2
        ixy[i][j] = ix[i][j]*iy[i][j]
        ixt[i][j] = ix[i][j]*it[i][j]
        iyt[i][j] = (iy[i][j])*it[i][j]

#calculate scalar values
nb = 10

for i in range(10,numRows-10):
    for j in range(10,numCols-10):
        ixxsum = 0
        iyysum = 0
        ixysum = 0
        iytsum = 0
        ixtsum = 0
        #iterate through however many pixels nb is set to
        for y in range(nb):
            for x in range(nb):
                y -= int(nb/2)
                x -= int(nb/2)
                iytsum += iyt[i+y][j+x]
                ixtsum += ixt[i+y][j+x]
                ixxsum+=ixx[i+y][j+x]
                iyysum+=iyy[i+y][j+x]
                ixysum+=ixy[i+y][j+x]
        #update scalars
        G = [[ixxsum,ixysum],[ixysum,iyysum]]
        uscalar[i][j] = (-iyysum*ixtsum + ixysum*iytsum)/determinant(G)
        vscalar[i][j] = (ixysum*ixtsum - ixxsum*ixtsum)/determinant(G)


#calculate scalar values

cv2.imshow("v scalar",vscalar)
cv2.imshow("u scalar,",uscalar)

cv2.waitKey(0)
cv2.destroyAllWindows()