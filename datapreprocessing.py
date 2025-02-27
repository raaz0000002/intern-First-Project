
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt


# load data
df=pd.read_csv(r"C:\Users\Administrator\Desktop\Intern\mobile_phone_price.csv")

#check null values
numbers_of_null=df.isnull().sum()

# K nearest neighbour 
#sns.scatterplot(x=df['battery_power'],y=df['px_height'])

# separating dependept and independent variables
Xdf=df.drop(columns=['price_range'])

Ydf=df['price_range']

# separating numerical and categorical values of independent variable
Xdf_numerical=Xdf.select_dtypes(include=['int64','float64','int','float'])

Xdf_categorical=Xdf.select_dtypes(include=['object'])

#-------------------------------------preprocessing for x----------------------------

# --------------------------------preprocessing for numerical values---------------------------------------------------------- 
# missing value imputation(KNN Imputer)
knn_imputer=KNNImputer()
knn_imputer.fit(Xdf_numerical)

values_output=knn_imputer.transform(Xdf_numerical)

null_value_filled=pd.DataFrame(values_output,columns=Xdf_numerical.columns)


# ploting value to show distribution and mean suing kde plot
sns.kdeplot(Xdf['battery_power'])

mean_val = Xdf['battery_power'].mean() #mean value of battery_power

plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}') #ploting mean line

plt.legend() #show legend
plt.show() #show figure

Xdf_numerical_final=null_value_filled
#-------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------preprocessing for categorical values ------------------------------------
from sklearn.preprocessing import OrdinalEncoder

categorical_encoder=OrdinalEncoder()

encoded_values=categorical_encoder.fit_transform(Xdf_categorical) #returns encoded values in form of array so, let's make data frame


Xdf_categorical_final=pd.DataFrame(encoded_values,columns=Xdf_categorical.columns) #final categorical imputing

#---------------------------------final data of x--------------------------------------------
final_X=pd.concat([Xdf_numerical_final,Xdf_categorical_final],axis=1)

#------------------------------------------preprocessing for y ----------------------------
# ['medium', 'high', 'very_high', 'low']
unique_y_values=Ydf.unique()
dict_for_mapping={'high':0,'low':1,'medium':2,'very_high':3}
final_Y=Ydf.map(dict_for_mapping)


















