""" Created on Fri Nov 5 18:49:14 2021

@author: samriddhi """

f = open('/Users/samriddhi/Desktop/cep.txt', 'r') content=f.read()

f2 = open('/Users/samriddhi/Downloads/hubble.txt', 'r') content2=f2.read()

split_content=content.split("\n") split_content2=content2.split("\n")

print(len(split_content)) print(len(split_content2))

cep_ap_m=[] for i in range(0,len(split_content)-1): if i==6: cep_ap_m.append(10.464) else: x=list() x=split_content[i].split("\t")[2] cep_ap_m.append(x)

cep_ap_m_er=list() for i in range(0,len(split_content)-1): if i==6: cep_ap_m_er.append(0.009) else: x=list() x=split_content[i].split("\t")[3] cep_ap_m_er.append(x)

dist_est_cep=list() for i in range(0,len(split_content)-1): if i==6: dist_est_cep.append(29.135) else: x=list() x=split_content[i].split("\t")[4] dist_est_cep.append(x)

dist_est_cep_err=list() for i in range(0,len(split_content)-1): if i==6: dist_est_cep_err.append(0.045) else: x=list() x=split_content[i].split("\t")[5] dist_est_cep_err.append(x)

redshift_z=[] for i in range(0,len(split_content2)-1): x=list() x=split_content2[i].split("\t")[2] redshift_z.append(x)

hubble_ap_m=[] for i in range(0,len(split_content2)-1): x=list() x=split_content2[i].split("\t")[3] hubble_ap_m.append(x)

hubble_ap_m_er=list() for i in range(0,len(split_content2)-1): x=list() x=split_content2[i].split("\t")[4] hubble_ap_m_er.append(x)

#############everything good so far, data is good#################

import matplotlib.pyplot as plt import numpy as np import scipy.stats from scipy.stats import norm

M=list() #cepheid M hat for i in range (0,len(split_content)-1): x=(float(cep_ap_m[i])-float(dist_est_cep[i])) #object types M.append(x)

#The M is going to be used for cepheid likelihood

#F hat for the Hubble sample
c=3.0*(10**5 )
F=[] for i in range (0,len(split_content2)-1): x=(float(hubble_ap_m[i])-25.0-5.0*((np.log((float(c)*float(redshift_z[i]))/100))/(np.log(10.0)))) F.append(x)

intrinsic_error=0.096 #probably need to change this, check the literature

#error for the cepheid sigmak=[] for i in range(0,len(split_content)-1): n=((intrinsic_error**2)+(float(cep_ap_m_er[i])**2)+(float(dist_est_cep_err[i])**2))**0.5 sigmak.append(n)

#hubble fow sample errors

sigmas=[] for i in range(0,len(split_content2)-1): m=((intrinsic_error**2)+(float(hubble_ap_m_er[i])**2))**0.5 sigmas.append(m)

def fun(y): Mo=y[0] theta=y[1] term1=0.0 term2=0.0 for a in range(0,len(split_content)-1): term1= term1 - 0.5*((((M[a]-Mo)**2)/(sigmak[a]**2))+(np.log(2np.pi(sigmak[a]*2)))) for b in range(0,len(split_content2)-1): term2= term2 - 0.5((((F[b]-Mo+theta)**2)/(sigmas[b]**2))+(np.log(2np.pi(sigmas[b]*2)))) return -1(term1+term2) y0=[0,1] import scipy.optimize as so sol=so.minimize(fun,y0,method='SLSQP') print(sol)

#sol.x returns the array you want

theta=sol.x[1] M0=sol.x[0]

h=10**(theta/5.0)

H0=h*100 #you need to make sure of what units you are in, but this gives us an estimte of the hubble constqant print(H0)

#the value we want to get near is around 67.4 km/s/Mpc +-10

#This is the most important part so far, we now have a MLE estimate of H0

'''transition_model = lambda x: [x[0],np.random.normal(x[1],0.5,(1,))[0]]

def prior(x): if(x[1]<=0): return 0 return 1

def likelihood(x,data): #not used scipy algebraic #x[0]=mu, x[1]=sigma (new or current) #data = the observation return np.sum(-np.log(x[1] * np.sqrt(2* np.pi) )-((data-x[0])**2) / (2*x[1]**2)) ##you can also do norm

def acceptance(x, x_new): if x_new>x: return True else: accept=np.random.uniform(0,1) boolean=(accept<(np.exp(x_new-x))) return boolean

def metrop(likelihood, prior, transition_model, param_init, N, acceptance_rule): #Step 1 MH algo x=param_init

#################
accepted = []#this is just for plotting later
rejected=[]
################

#Step i=1,...,N MH algo
for i in range(N):
    #propose a new value
    x_new=transition_model(x) #samples/moves to a very close value from the previous
    x_lik=likelihood(x,data)
    x_new_lik=likelihood(x_new,data)  #r=x_new_lik/x_lik
#acceptance_rule means acceptance=True
if(acceptance_rule(x_lik+np.log(prior(x)),x_new_lik+np.log(prior(x_new)))): #this is magic why the priors x=x_new accepted.append(x_new) else: rejected.append(x_new) return np.array(accepted),np.array(rejected)'''

#transition_model = lambda x: [x+np.random.normal(0,0.5)]

#turn the transition model into a funciton which takes in the old value x #and returns a value which is

def prior(v): if(1==0): #this will never be false (me getting rid of the prior) return 0 return 1

def acceptance(v, v_new): if v_new>v: return True else: accept=np.random.uniform(0,1) boolean=(accept<(np.exp(v_new-v))) return boolean

#we want to first take M0 as the MLE

def metrop(fun, prior, param_init, N, acceptance_rule): #Step 1 MH algo v=param_init #just for theta

#################
accepted =[]#this is just for plotting later
rejected=[]
################

#Step i=1,...,N MH algo
for i in range(N):
    #propose a new value
    v_new=v + np.random.normal(0,5.0) #pertubes the value to a new one we want it to reject
   
    y=list()
    y.append(M0)
    y.append(v)
    v_lik=fun(y)
       
    y_new=list()
    y_new.append(M0)
    y_new.append(v_new)
    v_new_lik=fun(y)
#acceptance_rule means acceptance=True
if(acceptance_rule(v_lik+np.log(prior(v)),v_new_lik+np.log(prior(v_new)))): #this is magic why the priors v=v_new accepted.append(v_new) else: rejected.append(v_new) return np.array(accepted),np.array(rejected)

accept,reject=metrop(fun, prior, theta, 200000, acceptance) #burn in accept_burnin=[] for i in range (0,len(accept)): if i>((len(accept))/2): accept_burnin.append(accept[i]) theta_bar=(sum(accept_burnin)/len(accept_burnin)) h=10**(theta_bar/5.0)

H0=h*100
print(H0)
