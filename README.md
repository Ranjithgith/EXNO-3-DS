### NAME :RANJITHKUMAR R
### REG.NO :212224240131

## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```

<img width="631" height="372" alt="image" src="https://github.com/user-attachments/assets/2c8d1ad3-e1f3-47b0-aa48-27aa848d7566" />


```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder 
pm=['Hot','Warm','Cold'] 
e1=OrdinalEncoder(categories=[pm]) 
e1.fit_transform(df[["ord_2"]])
```

<img width="562" height="245" alt="image" src="https://github.com/user-attachments/assets/3b3d856c-7f2b-45f0-85a8-169e72ccf6f7" />


```
df['bo2']=e1.fit_transform(df[["ord_2"]]) 
df
```

<img width="497" height="383" alt="image" src="https://github.com/user-attachments/assets/2e120bb7-0683-4e4e-834f-f40c01b1fc9c" />



```
le=LabelEncoder() 
dfc=df.copy() 
dfc['ord_2']=le.fit_transform(dfc['ord_2']) 
dfc
```

<img width="547" height="376" alt="image" src="https://github.com/user-attachments/assets/db81cb2e-552c-4175-8640-d83fab229388" />


```
from sklearn.preprocessing import OneHotEncoder 
ohe=OneHotEncoder(sparse_output=False) 
df2=df.copy() 
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) 
# Orders in Alphabetical Order Blue , Green, Red 
df2=pd.concat([df2,enc],axis=1) 
df2
```

<img width="571" height="432" alt="image" src="https://github.com/user-attachments/assets/c5ebde62-0cad-42cf-ae6b-8d86cef83e0d" />



```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="768" height="372" alt="image" src="https://github.com/user-attachments/assets/103b0498-0fb0-4efa-854c-50834d63a1a9" />


```
!pip install category_encoders
import pandas as pd
from category_encoders import BinaryEncoder
df=pd.read_csv('/content/data.csv') 
df
```

<img width="1291" height="590" alt="image" src="https://github.com/user-attachments/assets/83150b74-a9de-4cc8-ba1f-1442f299b8d7" />



```
be=BinaryEncoder() 
nd=be.fit_transform(df['Ord_2']) 
dfb=pd.concat([df,nd],axis=1) 
dfb
```

<img width="876" height="375" alt="image" src="https://github.com/user-attachments/assets/56fb5c82-3aef-4345-baf3-f949591ec8b3" />



```
from category_encoders import TargetEncoder 
te=TargetEncoder() 
CC=df.copy() 
new=te.fit_transform(X=CC["City"],y=CC["Target"]) 
CC=pd.concat([CC,new],axis=1) 
CC
```

<img width="790" height="381" alt="image" src="https://github.com/user-attachments/assets/9683677c-28d4-44f6-885f-cd6148b36790" />


```
import pandas as pd 
from scipy import stats 
import numpy as np 
df=pd.read_csv("/content/Data_to_Transform.csv") 
df
```

<img width="977" height="437" alt="image" src="https://github.com/user-attachments/assets/7a6870f8-1125-4d13-a934-09660f287269" />



```
df.skew()
```

<img width="523" height="222" alt="image" src="https://github.com/user-attachments/assets/845c7801-79fc-4939-9852-392242cd6853" />


```
np.log(df["Highly Positive Skew"])
```

<img width="551" height="467" alt="image" src="https://github.com/user-attachments/assets/d4c0aa1a-d2da-41ef-aec0-f084c5300e38" />



```
# 2. RECIPROCAL TRANSFORMATION 
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="486" height="475" alt="image" src="https://github.com/user-attachments/assets/638224d6-39fc-4c3e-9ca3-cfc1b356f345" />



```
# 4. SQUARE ROOT TRANSFORMATION 
np.sqrt(df["Highly Positive Skew"])
```


<img width="435" height="461" alt="image" src="https://github.com/user-attachments/assets/89411f79-0a02-41cc-b650-6e7d3c23ac66" />



```
# 5. SQUARE TRANSFORMATION 
np.square(df["Highly Positive Skew"])
```



<img width="472" height="465" alt="image" src="https://github.com/user-attachments/assets/452a2fd7-bf6a-49c7-b278-eb3ea3665442" />



```
# POWER TRANSFORMATIONS 
#        BOX COX 
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
df
```

<img width="1242" height="430" alt="image" src="https://github.com/user-attachments/assets/84f7d363-84b8-42ac-8808-be8b94f914e5" />



```
df.skew()
```

<img width="463" height="235" alt="image" src="https://github.com/user-attachments/assets/a2c14ae6-cf15-45ad-b6e1-695cf957ad81" />



```
# YEO_JOHNSON 
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
df.skew()
```

<img width="662" height="286" alt="image" src="https://github.com/user-attachments/assets/bd738267-d22f-44af-8d74-faf8fb7b7399" />


```
# QUANTILE TRANSFORMATION 
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal') 
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
df
```

<img width="1672" height="427" alt="image" src="https://github.com/user-attachments/assets/a1a6e398-4cc7-4fe0-814a-349f8ad60c7c" />



```
import seaborn as sns 
import statsmodels.api as sm 
# STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION 
import matplotlib.pyplot as plt 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
# QQ - QUANTILE QUANTILE PLOT 
plt.show()
```


<img width="907" height="470" alt="image" src="https://github.com/user-attachments/assets/0f979e28-f94a-4641-8fdf-26ce900f3e19" />



```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL 
plt.show()
```

<img width="942" height="467" alt="image" src="https://github.com/user-attachments/assets/e5ebc274-b161-41bf-b933-eea3b92165af" />



```
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891) 
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()
```


<img width="803" height="460" alt="image" src="https://github.com/user-attachments/assets/83a2d942-0f14-4026-980d-1d28074d822e" />


      
# RESULT:
      

          Thus the Implementation of Feature Encoding and Feature Transformation executed successfully.
