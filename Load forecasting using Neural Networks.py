import numpy as np

weatherInput = [-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1];
dayInput = [-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1];
holidayInput = [-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1];
cityInput = [-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1];
loadOutput= [28.75,57.5,35,70,33.75,67.5,40,80,38.75,77.5,45,90,43.75,87.5,50,100];#normalize the load output by dividing with 100

for i in range(0,16):
    loadOutput[i] = loadOutput[i]/100;

hidden1Weight = [0.5,0.3,0.3,0.5];#used
hidden2Weight = [0.6,0.4,0.4,0.6];#used

outerWeightOld = [0.5,0.5];#used
outerWeightNew = outerWeightOld;

alpha = 0.5;
lmse =35768;#used
replace = -1;

le1Weights = [];#used
le2Weights = [];#used
leOuter = [];#used
errorPerEpoch = [];#per every epoch
errorList = [];#per every input cycle and used to sum the all errors in 16cycles.

for x in range(0,16):
    errorList.append(-1);


class Neuron:
    
    def __init__(self):
        
        self.weightSum = -35768;#used
        self.bias = 0.5;#used
        self.out = -35768;#used
        
    def sigmoid(self,val):
        return 1/(1-np.exp(val));
        
    def weightedSum(self,index):
        global replace;
        self.weightSum = weatherInput[index]*hidden1Weight[0] + dayInput[index]*hidden1Weight[1] + holidayInput[index]*hidden1Weight[2] + cityInput[index]*hidden1Weight[3] + self.bias;
        replace = self.weightSum;
        
        self.out = self.sigmoid(replace);
        return self.out;
        
        
        
        
class OuterNeuron(Neuron):
    
    def __init__(self):
        self.weightSum = -35768;#used
        self.bias = 0.5;#used
        self.out = -35768;#used
        self.error = -35768.00;#used
        self.delErr = -35768.00;#used
        
    def weightedOuterSum(self,n1,n2):
        global replace;
        self.weightSum = n1*outerWeightNew[0] + n2*outerWeightNew[1] + self.bias;
        replace = self.weightSum;
        
        self.out = self.sigmoid(replace);
        return self.out;
        
    def weightUpdate(self,H1,H2):
        outerWeightNew[0] = outerWeightNew[0] - alpha*self.delErr*self.out*(1-self.out)*H1;
        outerWeightNew[1] = outerWeightNew[1] - alpha*self.delErr*self.out*(1-self.out)*H2;
    
    def setError(self,err,ind):
        self.error = err;
          
        self.delErr = self.out - loadOutput[index];

        
h1 = Neuron();
h2 = Neuron();

o = OuterNeuron();

for epoch in range(0,1000):
    for  index in range(0,16):
        var1 = h1.weightedSum(index);
        var2 = h2.weightedSum(index);
        var3 = o.weightedOuterSum(var1,var2);
        err = 0.5*pow((var3-loadOutput[index]),2);
        o.setError(err,index);
        errorList[index] = o.error;
        #weightUpdation needs to done from here.
        #first update outer neuron connected weights
        #and then the hidden layer connected weights
        o.weightUpdate(var1,var2);
        delError = o.delErr;
        
        hidden1Weight[0] = hidden1Weight[0] - alpha*delError*outerWeightOld[0]*var1*(1-var1)*weatherInput[index];
        hidden1Weight[1] = hidden1Weight[1] - alpha*delError*outerWeightOld[0]*var1*(1-var1)*dayInput[index];
        hidden1Weight[2] = hidden1Weight[2] - alpha*delError*outerWeightOld[0]*var1*(1-var1)*holidayInput[index];
        hidden1Weight[3] = hidden1Weight[3] - alpha*delError*outerWeightOld[0]*var1*(1-var1)*weatherInput[index];
        
        hidden2Weight[0] = hidden2Weight[0] - alpha*delError*outerWeightOld[1]*var2*(1-var2)*weatherInput[index];
        hidden2Weight[1] = hidden2Weight[1] - alpha*delError*outerWeightOld[1]*var2*(1-var2)*dayInput[index];
        hidden2Weight[2] = hidden2Weight[2] - alpha*delError*outerWeightOld[1]*var2*(1-var2)*holidayInput[index];
        hidden2Weight[3] = hidden2Weight[3] - alpha*delError*outerWeightOld[1]*var2*(1-var2)*weatherInput[index];
        
        outerWeightOld = outerWeightNew;
    sumErr = 0;
    for j in range(0,16):
        sumErr+= errorList[j];
    errorPerEpoch.append(sumErr);    
    if sumErr < lmse:
        lmse = sumErr;
        le1Weights = hidden1Weight;
        le2Weights = hidden2Weight;
        leOuter = outerWeightOld;
        
print(le1Weights);
print(le2Weights);
print(leOuter);