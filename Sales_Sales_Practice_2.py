#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:





# In[2]:


sales = pd.read_csv(r"C:\Users\shrir\Desktop\Adavanced_Analytics_Using_Statistics\Day10\reg\Sales_2021.csv")


# In[3]:


sales


# In[4]:


sales.head()


# In[5]:


sales.corr()


# In[6]:


sales.describe()


# In[7]:


sales.duplicated().sum()


# In[8]:


sales.shape


# In[9]:


sales.info()


# # Model1_1(Direct)

# In[10]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('Sales~Advt' , data = sales).fit()
print(model.summary())


# In[11]:


model1 = sm.stats.anova_lm(model)


# In[12]:


print(model1)


# In[13]:


pre1 = model.predict()
pre1


# In[14]:


res1 = sales['Sales'].values - pre1
res1


# In[15]:


pre_1 = pd.DataFrame(pre1 , columns = ['pre1'])
pre_1


# In[16]:


res_1 = pd.DataFrame(res1 , columns = ['res1'])
res_1


# In[17]:


from scipy import stats
zscore1 = stats.zscore(res1)
zscore1


# In[18]:


zscore_1 = pd.DataFrame(zscore1 , columns = ['zscore1'])


# In[19]:


zscore_1


# In[20]:


sales1 = pd.concat([sales,pre_1,res_1,zscore_1],axis=1)
sales1


# In[21]:


sales_1 = pd.DataFrame(sales1)


# In[22]:


sales1[sales1['zscore1'] > 1.96]


# In[23]:


sales1[sales1['zscore1'] < -1.96]


# # model1_2 Applying (dummy)

#     We are applying dummy , where the value is above 1.96 as 1 and below -1.96 as 0 because those are the outliers to improve the model dummy variable used

# In[24]:


a = sales_1.copy()


# In[28]:


a.head(19)


# In[30]:


for i in range(0,len(a)):
    if (np.any(a['zscore1'].values[i] > 1.96)):
        a['zscore1'].values[i] = 0
    else:
        a['zscore1'].values[i] = 1
    test = a['zscore1']
    
sales_1['dummy'] =  test
sales_1.head(19)


# In[31]:


x = sales_1[['Advt','dummy']]
y = sales_1['Sales']
x.head()


# In[33]:


plt.scatter(res1,y)
plt.xlabel("res1")
plt.ylabel("Sales")


# In[36]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.20 , random_state = 0)


# In[38]:


x_train.head()


# In[39]:


x_test.head()


# In[40]:


y_train.head()


# In[41]:


y_test.head()


# In[45]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
x_train1 = sm.add_constant(x_train)
model = sm.OLS(y_train , x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[48]:


regr = linear_model.LinearRegression()
regr.fit(x_train , y_train)

print("Intercept : ", regr.intercept_)
print("Coefficient : ", regr.coef_)


# In[50]:


y_pred = regr.predict(x_test)
y_pred


# In[51]:


from sklearn import metrics
print("Mean absolute error :", metrics.mean_absolute_error(y_test , y_pred))
print("Mean squared error :", metrics.mean_squared_error(y_test , y_pred))
print("Root mean squared error :",np.sqrt(metrics.mean_squared_error(y_test , y_pred )))


# In[7]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


sales = pd.read_csv(r'C:\Users\shrir\Desktop\Adavanced_Analytics_Using_Statistics\Day10\reg\Sales_2021.csv')


# In[12]:


sales


# In[13]:


sales.isnull().sum()


# In[14]:


sales.shape


# In[15]:


sales.info()


# In[16]:


sales.describe


# In[17]:


sales.corr()


# In[18]:


sales.cov()


# In[38]:


import statistics as s 
s.harmonic_mean(sales['Sales'])


# In[19]:


sales.boxplot()


# In[20]:


sales.plot(kind = 'box' , subplots = True , layout = (4,2) , sharex = False , sharey = False , figsize=(20,40))
plt.show


# In[23]:


sales.plot(kind = 'hist', figsize = (20,40))
plt.show()


# In[25]:


import statsmodels.api as sm
from statsmodels.formula.api import ols 
model = ols("Sales~Advt" , data = sales).fit()
model.summary()


# In[26]:


sales['pre1'] = model.predict()


# In[27]:


sales.columns 


# In[28]:


sales['res1'] = sales.Sales - sales.pre1


# In[29]:


sales.columns 


# In[30]:


from scipy.stats import zscore
sales['z_score1'] = zscore(sales['res1'])


# In[31]:


sales.columns 


# In[32]:


sales


# In[ ]:


sales[sales['z_score1'] > 1.96]


# In[35]:


sales[sales['z_score1'] < -1.96]


# # 1.2 Model_1 Applying (dummy)

# We are applying dummy where the value is above 1.96 as 1 and leass than 1.96 as 0 because those of the outliers to improve the model dummy variable is used

# In[39]:


sales['dummy'] = sales['res1']


# In[42]:


a = sales['dummy']
for i in range(0,len(a)):
    if (np.any(sales['z_score1'].values[i] > 1.96)):
        sales['dummy'].values[i] = 0
    else :
        sales['dummy'].values[i] = 1
sales.head(19)


# In[43]:


x = sales[['Advt' , 'dummy']]
y = sales['Sales']
x


# In[44]:


y


# In[45]:


plt.scatter(sales['res1'] , y)
plt.show()


# In[46]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.20 , random_state= 0)


# In[48]:


x_train.head()


# In[49]:


x_test.head()


# In[50]:


y_train.head()


# In[51]:


y_test.head()


# In[54]:


import statsmodels.api as sm
from statsmodels.formula.api import ols 
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
x_train1 = sm.add_constant(x_train)
model = sm.OLS(y_train , x_train1).fit()
model.summary()


# In[57]:


regr = linear_model.LinearRegression()
regr.fit(x_train , y_train)

print("Intercept :" , regr.intercept_)
print("Coefficint :" , regr.coef_)


# In[60]:


y_pred = regr.predict(x_test)
y_pred


# In[61]:


from  sklearn import metrics
print("mean Absolute error :",metrics.mean_absolute_error(y_test , y_pred))
print("Mean Squred error :", metrics.mean_squared_error(y_test , y_pred))
print("Root mean squared error :",np.sqrt(metrics.mean_squared_error(y_test , y_pred)))


# # Stepwise Regression

# # Forward Regression

# In[65]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
lg = LinearRegression()
sfs2 = sfs(lg , k_features= 2 , forward = True , verbose= 2 , scoring= 'neg_mean_squared_error')
sfs1 = sfs2.fit(x , y)


# In[66]:


features_name = list(sfs1.k_feature_names_)
print(features_name)


# In[69]:


lg = LinearRegression()
sfs3 = sfs(lg , k_features=2 , forward= False , verbose=2 , scoring='neg_mean_squared_error')
sfs3 = sfs1.fit(x , y)


# In[70]:


features_name =  list(sfs3.k_feature_names_)
print(features_name)


# # Homoscedasticity && Hetroscedascity

# In[71]:


from statsmodels.stats.diagnostic import het_breuschpagan


# In[73]:


model = ols('y~x' , data = sales).fit()
_ , pvalue , _ , _ = het_breuschpagan(model.resid , model.model.exog)

print(pvalue)

if pvalue > 0.05:
    print("This is Hetroscardecity")
else :
    print('This is homoscardicity')


# # Chi2 Test
# 

# In[77]:


from scipy.stats import chi2_contingency
x1 = sales['Advt']
y1 = sales['Sales']

stat , p , dof , expected = chi2_contingency(x1 , y1)

alpha = 0.05
print("P value is ",p)

if p <= alpha:
    print("Dependent (reject H0")
else :
    print("Independent (H0 holds True)")


# In[ ]:




