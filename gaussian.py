import numpy as np
xrange = range

def print_dot(a, b):
    print("(", end="")
    print(a[0], " x " ,b[0], end="")
    for i in range(1,len(a)):
        print(" + ", a[i], " x " ,b[i], end="")
    print(")", end="")

def GENP(A, b):
    '''
    Gaussian elimination with no pivoting.
    % input: A is an n x n nonsingular matrix
    %        b is an n x 1 vector
    % output: x is the solution of Ax=b.
    % post-condition: A and b have been modified. 
    '''
    n =  len(A)
    if b.size != n:
        raise ValueError("Invalid argument: incompatible sizes between A & b.", b.size, n)
    for pivot_row in xrange(n-1):
        print("Considering Column ", pivot_row+1)
        print("----------------------")
        print()

        for row in xrange(pivot_row+1, n):
            print("Row", row+1)
            
            print("\tmultiplier = ", end="")
            print(A[row][pivot_row], " / ", A[pivot_row][pivot_row])

            multiplier = A[row][pivot_row]/A[pivot_row][pivot_row]
            multiplier = round(multiplier,2)
            print("\t\t = ", multiplier)
            print()
            #the only one in this column since the rest are zero
            print("\tapplying multiplier to the row")
            print("\t\tr"+(str)(row+1)," - ", multiplier,"r"+(str)(pivot_row+1))

            for col in xrange(pivot_row, n):
                print("\t\t", A[row][col]," - (", multiplier, " x ", A[pivot_row][col], ")")
                A[row][col] = A[row][col] - multiplier * A[pivot_row][col]
                print("\t\t\t = ", A[row][col])
            #Equation solution column
            print("\t\t", b[row]," - (", multiplier, " x ", b[pivot_row], ")")
            b[row] = b[row] - multiplier * b[pivot_row]
            print("\t\t\t = ", b[row])
            print("\t", A)
            print()

    print("Performing back substitution")
    x = np.zeros(n)
    k = n-1
    print("\tx"+(str)(k+1),"=", b[k], " / ", A[k,k])
    x[k] = round(b[k]/A[k,k],2)
    print("\t\t= ", x[k])
    k-=1
    while k >= 0:
        print("\tx"+(str)(k+1),"=", b[k], " - ", end="")
        print_dot(A[k,k+1:],x[k+1:])
        print(" / ", A[k,k])
        x[k] = round((b[k] - np.dot(A[k,k+1:],x[k+1:]))/A[k,k],2)
        print("\t\t= ", x[k])
        k = k-1
    return x

def GEPP(A, b):
    '''
    Gaussian elimination with partial pivoting.
    % input: A is an n x n nonsingular matrix
    %        b is an n x 1 vector
    % output: x is the solution of Ax=b.
    % post-condition: A and b have been modified. 
    '''
    n =  len(A)
    if b.size != n:
        raise ValueError("Invalid argument: incompatible sizes between A & b.", b.size, n)
    # k represents the current pivot row. Since GE traverses the matrix in the upper 
    # right triangle, we also use k for indicating the k-th diagonal column index.
    for k in xrange(n-1):
        #Choose largest pivot element below (and including) k
        maxindex = abs(A[k:,k]).argmax() + k
        if A[maxindex, k] == 0:
            raise ValueError("Matrix is singular.")
        #Swap rows
        if maxindex != k:
            A[[k,maxindex]] = A[[maxindex, k]]
            b[[k,maxindex]] = b[[maxindex, k]]
        for row in xrange(k+1, n):
            multiplier = A[row][k]/A[k][k]
            #the only one in this column since the rest are zero
            A[row][k] = multiplier
            for col in xrange(k + 1, n):
                A[row][col] = A[row][col] - multiplier*A[k][col]
            #Equation solution column
            b[row] = b[row] - multiplier*b[k]
    print(A)
    print(b)
    x = np.zeros(n)
    k = n-1
    x[k] = b[k]/A[k,k]
    while k >= 0:
        x[k] = (b[k] - np.dot(A[k,k+1:],x[k+1:]))/A[k,k]
        k = k-1
    return x

if __name__ == "__main__":
    A = np.array([
        [ 1, 2, 3],
        [ 2,-3,-5],
        [-6,-8, 1]
    ])
    b =  np.array([
        [-7],
        [ 9],
        [-22]
    ])
    b = np.array([-7, 9, -22])
    GENP(np.copy(A), np.copy(b))