import numpy as np
from scipy.optimize import minimize
import csv
epsilon = 10**(-4)
lammbda = 1.0
x1 = []
x2 = []
y = []
with open('c:/Users/Pedro Henrique Lenzi/Desktop/ECA - 20.2/IA/Trabalho2- redeNeural/redeNeural/examples.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
       x1.append(float(row[0]))
       x2.append(float(row[1]))
       y.append(float(row[2]))
   

x0 = [1 for i in range(len(x1))]
X = [x0,x1,x2]
Y = y
#a1=x1

def contronstroeThetaTeste():
    out = [[[1,2,3],[4,5,6]]]
    out.append([[7,8,9],[10,11,12]])
    out.append([[13,14,15]])

    print("contronstroeThetaTeste = ",out)
    return out

theta1 = [[(np.random.rand()*2*epsilon-epsilon) for i in range(3)]for i in range(2)]
theta2 = [[(np.random.rand()*2*epsilon-epsilon)  for i in range(3)]for i in range(2)]
theta3 = [(np.random.rand()*2*epsilon-epsilon)  for i in range(3)]
Theta = [theta1, theta2, theta3]
Theta1 = contronstroeThetaTeste()
print("Theta = ",Theta)
A = [[0 for i in range(3)]for i in range(3)]#ativacao dos neuronios
A[0][0]=1
A[1][0]=1
A[2][0]=1
A.append([0])



def G(input):
    return 1/(1+(np.exp(-input)))

def H(A_lin,theta_num):
    return G(A_lin[0]*theta_num[2][0]+A_lin[1]*theta_num[2][1]+A_lin[2]*theta_num[2][3])

def forwardProgagation(Theta,X, index):
    A[0][1]= X[1][index]
    A[0][2]= X[2][index]
    i=0
    j=0

    for i in range(2):
        for j in range(2):
            # print("i = {} e j = {}".format(i,j))
            parcela1 =  Theta[i][j][0]*A[i][0]
            parcela2 = Theta[i][j][1]*A[i][1]
            parcela3 = Theta[i][j][2]*A[i][2]
            somaParcelas = parcela1+parcela2+parcela3
            A[i+1][j+1] = G(somaParcelas)
            #print("A[{}][{}] = {}".format(i+1,j+1,A[i+1][j+1]))
    parcela1 = Theta[2][0]*A[2][0]
    parcela2 = Theta[2][1]*A[2][1]
    parcela3 = Theta[2][2]*A[2][2]
    somaParcelas = parcela1+parcela2+parcela3
    A[3] = G(somaParcelas)
    #print("saida forwardprop, Ativacao A = ",A)


def J(theta):
    somatorio1 = 0#somatorio entropia cruzada
    for i in range(len(Y)):
        #print("i = ",i)
        forwardProgagation(Theta,X, i)
        custo = Y[i]*np.log(A[3])+(1-Y[i])*(np.log(1-(A[3])))
        somatorio1 =+ custo
    somatorio2 = 0#somatório quadrado dos theta
    for i in range(3):#coluna dos thetas
        for j in range(1,3):# para cada linha(núcleo) de theta, pulando theta bias
            if i != 2:#se não éa ultima coluna
                for k in range(2):
                    somatorio2 =+ Theta[i][k][j]**2
            else:
                somatorio2 =+ Theta[i][j]**2
    return 1/(len(Y))*(somatorio1)+( lammbda/(2*len(Y)) )*somatorio2
               

custo = J(Theta)
print("Custo = ",custo)



# def gradiente(theta,X,Y):