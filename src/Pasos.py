from Logica import pred_W18
from Logica import solve_sn
import matplotlib.pyplot as plt
import numpy as np
import numpy_financial as npf
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
def make_simulated_transit(TPD=402.39,vc=0.5,cd=1.0,size=5000,n=360,seedint=63442967)->tuple[np.array,np.array]:
    """
    Funtion to generate a bunch of simulated transit\n
    n:int = meses de diseño\n
    seedint = semilla de rng, 0 para no usar semilla.\n
    return tuple of array like of size*n length of traffic and acumulative traffic at time index (monthly)
    """
    if seedint != 0:
        np.random.default_rng(seed=seedint)  
    #Uniform distribution between [0,1)
    res = np.random.random(size*n)
    acum = np.zeros(size*n)
    for i in range(0,size*n,n):
        res[i]=TPD*365/12*vc*cd
        acum[i]=res[i]
        for j in range(1,n):
            #Incrementing the distribution to [0.87-1.17) sometimes traffic decreases for the month. It is an acumulative traffic 
            res[i+j]=(res[i+j]+2.9)*0.3*res[i+j-1]
            acum[i+j]=acum[i+j-1]+res[i+j]
    return res,acum
#print(make_simulated_transit(100,size=2,n=3))
def calculate_break(arr:np.array,SN_dis,Reliavility,Standard_Deviation,Delta_PSI,Mr)->int:
    """
    arr: array like with acumulative transit\n
    Makes a binary search for the postion when the design fails first\n
    return len(arr)+1 if doesnt fail
    """
    low = 0
    high = len(arr) - 1
    mid = 0
    while low <= high:
        mid = (high + low) // 2
        sn = solve_sn(Reliavility, Standard_Deviation, Delta_PSI, Mr,arr[mid])
        if sn < SN_dis:
            low = mid + 1
        elif sn > SN_dis:
            high = mid - 1
        # means sn IS EQUAL to sn_dis at mid
        else:
            return mid
    # We should reach here meaning mid is the lowest possible without being over SN_dis, so we get the next when it fails 
    #note, it returns len(arr)+1 if dont fail on all the array
    return mid+1

def W18_prediction(TPD=402.39,step=3,vc=0.5,cd=1.0,i=0.05,n=30)->list:
    """Generate the W18 predictions used for the traditional approach in flexible pavement \n
    Additionally creates the additional predictions of W18 to generate an aproach of flexibility in the design with intermediate steps. \n
    TPD:int = Trafico promedio diario\n
    vc:float =Distribución por sentido (usalmente 0.5)\n
    cd:float =Carril de diseño (usualmente 1.0 si es de un solo carril por sentido)\n
    i:float =indice de crecimiento \n
    n:int =años de diseño\n
    return list of pred_w18 for TPD, last is complete
    """
    res = []
    for i in range(1,step+1):
        res.append(pred_W18(TPD,vc,cd,i,n//step*i))
    return res
def W18_linear_regression(arr:np.array)->np.poly1d:
    """\n
    getting a new TPD recalculate with pred_W18
    """
    x= np.arange(len(arr))
    y = arr
    #m, b = np.polyfit(x, y, deg=1)
    #plt.axline(xy1=(0, b), slope=m, label=f'$y = {m:.1f}x {b:+.1f}$')

    coef = np.polyfit(x,y,1)
    poly1d_fn = np.poly1d(coef) 
    #plt.plot(x,y, 'yo', x, poly1d_fn(x), '--k')
    #plt.show()
    return poly1d_fn
def npv(r,arr):
    sum_pv = arr[0]
    for i in range(1,len(arr)+1):
        sum_pv += arr[i-1] / ((1 + r) ** i)
    return sum_pv
def evaluate_flexibility(size,n,rate,sn_design)->np.array:
    """
    Funtion to evaluate the design flexibility\n
    size:int = tamaño de muestra aleatoria\n
    n:int = meses de diseño\n
    return tuple of array like of size*n length of traffic and acumulative traffic at time index (monthly)
    """
    res = np.zeros(size)
    random_transit,acumulated = make_simulated_transit()

    for i in range(size):
        random_cost = np.zeros(n)
        n_break = calculate_break()
        #redesign when break, and add to the cost on period n_break. 
        while n_break<=n:
            #TODO
            #Then take the random transit and redesign
            fun=W18_linear_regression()
            #random_cost[n_break]= #cost of redesign
            n_break = calculate_break()
        #calculate NPV from redesign until n
        res[i]= np.npv(rate,random_cost)
    return res 
print(npv(0.05,make_simulated_transit(100,size=2,n=3)[0]))