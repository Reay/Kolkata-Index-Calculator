'''

This project caculates the Gini indexes and Kolkata Indexes of the yearly 
Impact factors of science and social sciencies* Journals. Kolkata index
was created by Econophyscists based in India, their motivation was to
have a more sensitive index than the conventional gini-index. 


*Social Science Journals were included in Web Of Science Database on November
2009 and by 2014 it had a comprehensive multi-disiplinary database of Journals.

Results - Gini and Kolkata indexes of Journals from web of science (yearly) generally
reduces over time. This may be loosely interpretaed that academia is publishing
more widely instead of being hihgly specialised (which may lead to more inequality).


'''



# 1 Import the necessary Libraries

import numpy as np    # Maths
import pandas as pd   # Loading, reading e.t.c
from matplotlib import pyplot as plt
import glob           # Helps to deal with multiple data files 
import os             # File directory purposes  
from shapely.geometry import LineString # This library helps to find an 
                                        # intercept to calculate the Kolkata
                                        # index.

import scipy.interpolate as interpolate
import scipy.optimize as optimize
import matplotlib.pyplot as plt


def GRLC(values):
    
    '''
    Calculate Gini index, Gini coefficient, Robin Hood index, and points of 
    Lorenz curve based on the instructions given in 
    www.peterrosenmai.com/lorenz-curve-graphing-tool-and-gini-coefficient-calculator
    Lorenz curve values as given as lists of x & y points [[x1, x2], [y1, y2]]
    @param values: List of values
    @return: [Gini index, Gini coefficient, Robin Hood index, [Lorenz curve]] 
    '''
    
    n = len(values)
    assert(n > 0), 'Empty list of values'
    sortedValues = sorted(values) #Sort smallest to largest

    #Find cumulative totals
    cumm = [0]
    for i in range(n):
        cumm.append(sum(sortedValues[0:(i + 1)]))

    #Calculate Lorenz points
    LorenzPoints = [[], []]
    sumYs = 0           #Some of all y values
    robinHoodIdx = -1   #Robin Hood index max(x_i, y_i)
    for i in range(1, n + 2):
        x = 100.0 * (i - 1)/n
        y = 100.0 * (cumm[i - 1]/float(cumm[n]))
        LorenzPoints[0].append(x)
        LorenzPoints[1].append(y)
        sumYs += y
        maxX_Y = x - y
        if maxX_Y > robinHoodIdx: robinHoodIdx = maxX_Y   
    
    giniIdx = 100 + (100 - 2 * sumYs)/n #Gini index 
    
   

    return [giniIdx, giniIdx/100, robinHoodIdx, LorenzPoints]


#Example
#sample1 = [1, 2, 1.2, 2, 3, 1.9, 2.2, 4.5]
#sample2 = [812, 841, 400, 487, 262, 442, 972, 457, 491, 461, 430, 465, 991, \
#           479, 427, 456]
#result = GRLC(a99if)

def plot_series(gg, year):
    print year
    result = GRLC(gg)
    #print 'Gini Index', result[0]  
    #print 'Gini Coefficient', result[1]
    #print 'Robin Hood Index', result[2]
    #print 'Lorenz curve points', result[3]

    #Plot

    plt.plot(result[3][0], result[3][1], [0,100], [0,100], '--',label='first')
    g = np.arange(0,100)
    y = 100 - g
    x = np.arange(0,100)

    l1, l2 = [], []
    for i in range(len(result[3][1])):
        l1.append((result[3][0][i], result[3][1][i]))

    for i in range(len(y)):
        l2.append((i,y[i]))

    line1 = LineString(l1)
    line2 = LineString(l2)

    intersection = (line1.intersection(line2))
    print intersection

    plt.plot(y, label='second')
    plt.xlabel('% of pupulation')
    plt.ylabel('% of values')
    plt.legend()
    plt.show()

os.chdir('C:\Users\wagis\Desktop\DataSets\WebOfScience\Journals')

path = 'C:\Users\wagis\Desktop\DataSets\WebOfScience\Journals'


# 2 Combine Journal Data CSV files into one DataFrame 

everyfile =  glob.glob(os.path.join(path,"*.csv"))

array = []

for file_ in everyfile:
    dataframe = pd.read_csv(file_)
    array.append(dataframe.as_matrix())
    
combine_array = np.vstack(array)

Combined_dataframe = pd.DataFrame(combine_array)    

# 3 Pre-Process the data for analysis ~
#Col_Filtered = Combined_dataframe[[2,3,4]]

#for i in range (3,170182):
#    if Col_Filtered.iloc(i,0) == 'Total Cites':
#        Year_1997 = Col_Filtered.iloc(i-1,0)                        
#    break                    

Jour1997 = Combined_dataframe[0     : 6637 ]
Jour1998 = Combined_dataframe[6637  : 13786]
Jour1999 = Combined_dataframe[13786 : 21038]
Jour2000 = Combined_dataframe[21038 : 28424]
Jour2001 = Combined_dataframe[28424 : 35861]
Jour2002 = Combined_dataframe[35861 : 43449]
Jour2003 = Combined_dataframe[43449 : 51073]
Jour2004 = Combined_dataframe[51073 : 58757] 
Jour2005 = Combined_dataframe[58757 : 66595]
Jour2006 = Combined_dataframe[66595 : 74532]
Jour2007 = Combined_dataframe[74532 : 82827]
Jour2008 = Combined_dataframe[82827 : 91435]
Jour2009 = Combined_dataframe[91435 : 101082]
Jour2010 = Combined_dataframe[101082: 111889]
Jour2011 = Combined_dataframe[111889: 123194]
Jour2012 = Combined_dataframe[123194: 134715]
Jour2013 = Combined_dataframe[134715: 146337]
Jour2014 = Combined_dataframe[146337: 158153]
Jour2015 = Combined_dataframe[158153: 170182]

Jour1997cites = Jour1997.iloc[1:6635, 2]
Jour1998cites = Jour1998.iloc[1:7147, 2]
Jour1999cites = Jour1999.iloc[1:7250, 2]
Jour2000cites = Jour2000.iloc[1:7384, 2]
Jour2001cites = Jour2001.iloc[1:7435, 2]
Jour2002cites = Jour2002.iloc[1:7584, 2]
Jour2003cites = Jour2003.iloc[1:7622, 2]
Jour2004cites = Jour2004.iloc[1:7682, 2]
Jour2005cites = Jour2005.iloc[1:7836, 2]
Jour2006cites = Jour2006.iloc[1:7935, 2]
Jour2007cites = Jour2007.iloc[1:8293, 2]
Jour2008cites = Jour2008.iloc[1:8606, 2]
Jour2009cites = Jour2009.iloc[1:9645, 2]
Jour2010cites = Jour2010.iloc[1:10805, 2]
Jour2011cites = Jour2011.iloc[1:11303, 2]
Jour2012cites = Jour2012.iloc[1:11519, 2]
Jour2013cites = Jour2013.iloc[1:11620, 2]
Jour2014cites = Jour2014.iloc[1:11814, 2]
Jour2015cites = Jour2015.iloc[1:12027, 2]

# cool? Yes
#gg = pd.to_numeric(Jour1998if, errors='ignore')  <---- even this has problems
#print gg.dtype
# is it good? Yes, any idea why this happens? system function (simple .tolist() functions) was not able to convert one of the value into float.
# so it was stuck at that point. but now we say that, if it does not able to cnvert the value, it can ignore that one.
# similar to exception handling with try except statement. 
# any tips on how to loop this baby and get all the gini indexes? for the datasets.
# for this try making a function that takes the series, and gives you theplot.
# and run a for loop to call tht function.



Jour1997if = Jour1997.iloc[1:6635, 3].fillna('0')
gg = []
for i in Jour1997if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '1997')

#s = Jour1997if.tolist()

Jour1998if = Jour1998.iloc[1:6635, 3].fillna('0')
gg = []
for i in Jour1998if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg,'1998')

Jour1999if = Jour1999.iloc[1:7250, 3].fillna('0')

gg = []
for i in Jour1999if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '1999')

Jour2000if = Jour2000.iloc[1:7384, 3].fillna('0')

gg = []
for i in Jour2000if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2000')

Jour2001if = Jour2001.iloc[1:7435, 3].fillna('0')

gg = []
for i in Jour2001if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2001')

Jour2002if = Jour2002.iloc[1:7584, 3].fillna('0')

gg = []
for i in Jour2002if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2002')

Jour2003if = Jour2003.iloc[1:7622, 3].fillna('0')

gg = []
for i in Jour2003if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2003')

Jour2004if = Jour2004.iloc[1:7682, 3].fillna('0')

gg = []
for i in Jour2004if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2004')

Jour2005if = Jour2005.iloc[1:7836, 3].fillna('0')

gg = []
for i in Jour2005if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2005')

Jour2006if = Jour2006.iloc[1:7935, 3].fillna('0')

gg = []
for i in Jour2006if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2006')

Jour2007if = Jour2007.iloc[1:8293, 3].fillna('0')

gg = []
for i in Jour2007if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2007')

Jour2008if = Jour2008.iloc[1:8606, 3].fillna('0')

gg = []
for i in Jour2008if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2008')

Jour2009if = Jour2009.iloc[1:9645, 3].fillna('0')

gg = []
for i in Jour2009if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2009')

Jour2010if = Jour2010.iloc[1:10805, 3].fillna('0')
gg = []
for i in Jour2010if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2010')

Jour2011if = Jour2011.iloc[1:11303, 3].fillna('0')

gg = []
for i in Jour2011if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2011')

Jour2012if = Jour2012.iloc[1:11519, 3].fillna('0')

gg = []
for i in Jour2012if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2012')

Jour2013if = Jour2013.iloc[1:11620, 3].fillna('0')

gg = []
for i in Jour2013if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2013')

Jour2014if = Jour2014.iloc[1:11814, 3].fillna('0')

gg = []
for i in Jour2014if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2014')

Jour2015if = Jour2015.iloc[1:12027, 3].fillna('0')

gg = []
for i in Jour2015if.tolist():
    try:
        gg.append(float(i))
    except:
        continue
plot_series(gg, '2015')

#a = Jour1997if[:].tolist()
#a = map(float, a)

# Gini-index Calculator

# cool?  nyoews you have all the plots.




import matplotlib.pyplot as plt

x = [1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
y = [70.66973240109687,70.74177611927036,70.54134001581703,70.41193897244236,70.24897155604072,70.09760065266484,69.96922913816228,69.74947263562309,69.31480254510558,69.71648831918246,68.78697876465138,68.23370103792684,68.67603743159262,69.02080368995155,68.98510648916688,68.69108089492519,69.76190202014053 ,68.20219277996949,63.74854075963432 ]
plt.plot(x,y,'-')
plt.xlabel('Year')
plt.ylabel('K-Index')

plt.plot([1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015],
         [56.46, 56.80, 56.06, 55.65, 55.60, 55, 54.68,54.38, 53.15, 52.56, 51.87, 50.28, 51.56, 52.61, 52.47, 51.97, 51.38, 50.52, 49.56])
plt.xlabel('Year')
plt.ylabel('Gini Index')


x = [1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
y = [70.66973240109687,70.74177611927036,70.54134001581703,70.41193897244236,70.24897155604072,70.09760065266484,69.96922913816228,69.74947263562309,69.31480254510558,69.71648831918246,68.78697876465138,68.23370103792684,68.67603743159262,69.02080368995155,68.98510648916688,68.69108089492519,69.76190202014053 ,68.20219277996949,63.74854075963432 ]
zini =  [56.46, 56.80, 56.06, 55.65, 55.60, 55, 54.68,54.38, 53.15, 52.56, 51.87, 50.28, 51.56, 52.61, 52.47, 51.97, 51.38, 50.52, 49.56]

plt.plot(x,y,x,zini,'-')




#plt.title ('')





'''
