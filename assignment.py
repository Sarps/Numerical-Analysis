import numpy as np

from pylatex import Document, Section, Subsection, Table, Math, TikZ, Axis, Command, Plot, NoEscape
from pylatex.math import Matrix
from pylatex.utils import italic


def print_dot(a, b):
    section = []
    section.append("(")
    section.extend([a[0], " x " ,b[0]])
    for i in range(1,len(a)):
        section.extend([" + ", a[i], " x " ,b[i]])
    section.append(")")
    return section

def GENP(A, b):
    
    section = Section('The first section')

    n =  len(A)
    if b.size != n:
        raise ValueError("Invalid argument: incompatible sizes between A & b.", b.size, n)
    for pivot_row in range(n-1):
        section.append("Considering Column " + (str)(pivot_row+1))

        for row in range(pivot_row+1, n):
            section.append(italic("Row" +(str)(row+1)))
            section.append(Subsection('\t', data=[Math(data=["\tmultiplier", '=', A[row][pivot_row], " / ", A[pivot_row][pivot_row]])]))

            multiplier = A[row][pivot_row]/A[pivot_row][pivot_row]
            multiplier = round(multiplier,2)
            section.append(Math(data=["\t\tmultiplier", '=', multiplier]))
            section.append("\n")
            #the only one in this column since the rest are zero
            section.append("\tapplying multiplier to the row")
            section.append(Math(data=[
                NoEscape("\t\tr_"+(str)(row+1))," - ", multiplier,NoEscape("r_"+(str)(pivot_row+1))
            ]))

            for col in range(pivot_row, n):
                section.append(Math(data=["\t\t", A[row][col]," - (", multiplier, " x ", A[pivot_row][col], ")"]))
                A[row][col] = A[row][col] - multiplier * A[pivot_row][col]
                section.append(Math(data=["\t\t\t = ", A[row][col]]))
            #Equation solution column
            section.append(Math(data=["\t\t", b[row]," - (", multiplier, " x ", b[pivot_row], ")"]))
            b[row] = b[row] - multiplier * b[pivot_row]
            section.append(Subsection('\t\t\t', data=[Math(data=['=', b[row]])]))

            section.append(Math(data=['A', '=', Matrix(A)]))
            section.append(Math(data=['b', '=', b]))
            section.append("\n")

    section.append("Performing back substitution")
    x = np.zeros(n)
    k = n-1

    section.append(Math(data=[NoEscape("\tr_"+(str)(k+1)),"=", b[k], " / ", A[k,k]]))
    x[k] = round(b[k]/A[k,k],2)
    section.append(Math(data=["\t\t= ", x[k]]))

    k-=1

    while k >= 0:
        eqn = [NoEscape("\tr_"+(str)(k+1)),"=", b[k], " - "]
        eqn.extend(print_dot(A[k,k+1:],x[k+1:]))
        eqn.extend([" / ", A[k,k]])
        section.append(Math(data=eqn))

        x[k] = round((b[k] - np.dot(A[k,k+1:],x[k+1:]))/A[k,k],2)
        section.append(Math(data=["\t\t= ", x[k]]))
        k = k-1

    return section


def GEPP(A, b):

    section = Section('The first section')
    
    n =  len(A)
    if b.size != n:
        raise ValueError("Invalid argument: incompatible sizes between A & b.", b.size, n)
    for k in range(n-1):
        section.append("Considering Column " + (str)(k+1))

        #Choose largest pivot element below (and including) k
        maxindex = abs(A[k:,k]).argmax() + k
        if A[maxindex, k] == 0:
            raise ValueError("Matrix is singular.")
        #Swap rows
        if maxindex != k:
            A[[k,maxindex]] = A[[maxindex, k]]
            b[[k,maxindex]] = b[[maxindex, k]]
        for row in range(k+1, n):
            
            section.append(italic("Row" +(str)(row+1)))
            section.append(Subsection('\t', data=[Math(data=["\tmultiplier", '=', A[row][k], " / ", A[k][k]])]))

            multiplier = A[row][k]/A[k][k]
            multiplier = round(multiplier,2)
            section.append(Math(data=["\t\tmultiplier", '=', multiplier]))
            section.append("\n")
            #the only one in this column since the rest are zero
            section.append("\tapplying multiplier to the row")
            section.append(Math(data=[
                NoEscape("\t\tr_"+(str)(row+1))," - ", multiplier,NoEscape("r_"+(str)(k+1))
            ]))

            for col in range(k, n):
                section.append(Math(data=["\t\t", A[row][col]," - (", multiplier, " x ", A[k][col], ")"]))
                A[row][col] = A[row][col] - multiplier * A[k][col]
                section.append(Math(data=["\t\t\t = ", A[row][col]]))
            #Equation solution column
            b[row] = b[row] - multiplier*b[k]
            section.append(Math(data=["\t\t", b[row]," - (", multiplier, " x ", b[k], ")"]))
            b[row] = b[row] - multiplier * b[k]
            section.append(Math(data=['=', b[row]]))

            section.append(Math(data=['A', '=', Matrix(A)]))
            section.append(Math(data=['b', '=', b]))
            section.append("\n")
    
    section.append("Performing back substitution")
    x = np.zeros(n)
    k = n-1
    x[k] = b[k]/A[k,k]

    section.append(Math(data=[NoEscape("\tr_"+(str)(k+1)),"=", b[k], " / ", A[k,k]]))
    x[k] = round(b[k]/A[k,k],2)
    section.append(Math(data=["\t\t= ", x[k]]))

    k-=1

    while k >= 0:
        eqn = [NoEscape("\tr_"+(str)(k+1)),"=", b[k], " - "]
        eqn.extend(print_dot(A[k,k+1:],x[k+1:]))
        eqn.extend([" / ", A[k,k]])
        section.append(Math(data=eqn))

        x[k] = (b[k] - np.dot(A[k,k+1:],x[k+1:]))/A[k,k]
        section.append(Math(data=["\t\t= ", x[k]]))
        k = k-1
    return x


if __name__ == "__main__":
    A = np.array([
        [ 1, 2, 3],
        [ 2,-3,-5],
        [-6,-8, 1]
    ])
    b = np.array([-7, 9, -22])

    doc = Document()
    doc.preamble.append(Command('title', 'Assignment Title'))
    doc.preamble.append(Command('author', 'Sarps'))
    doc.preamble.append(Command('date', NoEscape('\\today')))
    doc.append(NoEscape("\\maketitle"))

    doc.append(GENP(np.copy(A), np.copy(b)))
    doc.generate_pdf()