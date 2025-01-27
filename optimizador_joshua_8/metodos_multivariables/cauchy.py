from ..baseopt import gradient_methods, optimizador_univariable
from ..metodos_univariables import goldensearch, fibonacci,interval, biseccion, secante
import numpy as np
from ..funciones.base import funcion
class cauchy(gradient_methods):
    def __init__(self, variables, f:funcion, epsilon, epsilon2, iter=100, opt: optimizador_univariable = goldensearch):
        super().__init__(variables, f, epsilon, iter)
        self.epsilon2 = epsilon2
        self.opt = opt
        self.gradiente = []
        self.data=f
    def testalpha(self, alfa):
        return self.funcion(self.variables - (alfa * np.array(self.gradiente)))
    
    def gradiente_calculation(self,x,delta=0.0001):
        if delta == None: 
            delta=0.00001
        vector_f1_prim=[]
        x_work=np.array(x)
        x_work_f=x_work.astype(np.float64)
        if isinstance(delta,int) or isinstance(delta,float):
            for i in range(len(x_work_f)):
                point=np.array(x_work_f,copy=True)
                vector_f1_prim.append(self.primeraderivadaop(point,i,delta))
            return vector_f1_prim
        else:
            for i in range(len(x_work_f)):
                point=np.array(x_work_f,copy=True)
                vector_f1_prim.append(self.primeraderivadaop(point,i,delta[i]))
            return vector_f1_prim

    def primeraderivadaop(self,x,i,delta):
        mof=x[i]
        p=np.array(x,copy=True)
        p2=np.array(x,copy=True)
        nump1=mof + delta
        nump2 =mof - delta
        p[i]= nump1
        p2[i]=nump2
        numerador=self.funcion(p) - self.funcion(p2)
        return numerador / (2 * delta) 
    def segundaderivadaop(self,x,i,delta):
        mof=x[i]
        p=np.array(x,copy=True)
        p2=np.array(x,copy=True)
        nump1=mof + delta
        nump2 =mof - delta
        p[i]= nump1
        p2[i]=nump2
        numerador=self.funcion(p) - (2 * self.funcion(x)) +  self.funcion(p2)
        return numerador / (delta**2) 

    def derivadadodadoop(self,x,index_principal,index_secundario,delta):
        mof=x[index_principal]
        mof2=x[index_secundario]
        p=np.array(x,copy=True)
        p2=np.array(x,copy=True)
        p3=np.array(x,copy=True)
        p4=np.array(x,copy=True)
        if isinstance(delta,int) or isinstance(delta,float):#Cuando delta es un solo valor y no un arreglo 
            mod1=mof + delta
            mod2=mof - delta
            mod3=mof2 + delta
            mod4=mof2 - delta
            p[index_principal]=mod1
            p[index_secundario]=mod3
            p2[index_principal]=mod1
            p2[index_secundario]=mod4
            p3[index_principal]=mod2
            p3[index_secundario]=mod3
            p4[index_principal]=mod2
            p4[index_secundario]=mod4
            numerador=((self.funcion(p)) - self.funcion(p2) - self.funcion(p3) + self.funcion(p4))
            return numerador / (4*delta*delta)
        else:#delta si es un arreglo 
            mod1=mof + delta[index_principal]
            mod2=mof - delta[index_principal]
            mod3=mof2 + delta[index_secundario]
            mod4=mof2 - delta[index_secundario]
            p[index_principal]=mod1
            p[index_secundario]=mod3
            p2[index_principal]=mod1
            p2[index_secundario]=mod4
            p3[index_principal]=mod2
            p3[index_secundario]=mod3
            p4[index_principal]=mod2
            p4[index_secundario]=mod4
            numerador=((self.funcion(p)) - self.funcion(p2) - self.funcion(p3) + self.funcion(p4))
            return numerador / (4*delta*delta)

        
    def hessian_matrix(self,x,delt=0.0001):# x es el vector de variables
        matrix_f2_prim=[([0]*len(x)) for i in range(len(x))]
        x_work=np.array(x)
        x_work_f=x_work.astype(np.float64)
        for i in range(len(x)):
            point=np.array(x_work_f,copy=True)
            for j in range(len(x)):
                if i == j:
                    matrix_f2_prim[i][j]=self.segundaderivadaop(point,i,delt)
                else:
                    matrix_f2_prim[i][j]=self.derivadadodadoop(point,i,j,delt)
        return matrix_f2_prim
    
    def optimizaralpha(self,test):
        a=self.data.get_limiteinf()
        b=self.data.get_limitesup()
        opt=self.opt(a,b ,test,self.epsilon)
        alfa=opt.optimize()
        return alfa
    
    def optimize(self):
        terminar = False
        xk = self.variables
        k = 0
        while not terminar:
            grad = np.array(self.gradiente_calculation(xk))
            if np.linalg.norm(grad) < self.epsilon or k >= self.iteraciones:
                terminar = True
            else:
                def alpha_funcion(alpha):
                    return self.funcion(xk - alpha * grad)
                alpha = self.optimizaralpha(alpha_funcion)
                xk_1 = xk - alpha * grad
                if np.linalg.norm(xk_1 - xk) / (np.linalg.norm(xk) + 0.0001) < self.epsilon2:
                    terminar = True
                xk = xk_1
                k += 1
        print(k)
        return xk
