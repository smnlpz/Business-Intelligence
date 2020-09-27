#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:54:36 2019

@author: simon
"""

import time
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


SEED=26504148


def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []

    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("F1 score (tst): {:.4f}, tiempo: {:6.2f} segundos"
			  .format(f1_score(y[test],y_pred,average='micro') , tiempo))
        y_test_all = np.concatenate([y_test_all,y[test]])

    print("")

    return modelo, y_test_all


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)



# ----------------------------------------------------
# ----------------- LECTURA DE DATOS -----------------
# ----------------------------------------------------

data_x = pd.read_csv('../dataset/nepal_earthquake_tra.csv')
data_y = pd.read_csv('../dataset/nepal_earthquake_labels.csv')
data_x_tst = pd.read_csv('../dataset/nepal_earthquake_tst.csv')

#se quitan las columnas que no se usan
data_x.drop(labels=['building_id'], axis=1,inplace = True)
data_x_tst.drop(labels=['building_id'], axis=1,inplace = True)
data_y.drop(labels=['building_id'], axis=1,inplace = True)


# Número de instancias y contar valores perdidos
'''
print(data_y.describe())
print(data_x_tst.describe())
print("Valores perdido en train:")
print(data_x.isnull().sum())
print("\n\nValores perdidos en test:")
print(data_x_tst.isnull().sum())
'''


# Número de instancias de cada clase
'''
#print(data_y['damage_grade'].value_counts())	
#	2    148259
#	3     87218
#	1     25124

data_y['damage_grade'].value_counts(sort=False).plot(
			kind='bar', color=['sienna', 'skyblue', 'C1'], figsize=(15,10)
		)
plt.savefig("./plots/value_counts.png")
#plt.show()
plt.clf()
'''


# Correlacion entre variables
'''
fig,ax=plt.subplots(figsize=(20,20))
sns.heatmap(data_x.corr(), ax=ax,cmap="YlGnBu",cbar=False)
plt.savefig("./plots/correlation.png")
#plt.show()
plt.clf()

print("Correlacion has_secondary_use con has_secondary_use_agriculture:")
print(data_x.corr()['has_secondary_use']['has_secondary_use_agriculture'])

print("Correlacion height_percentage con count_floors_pre_eq:")
print(data_x.corr()['height_percentage']['count_floors_pre_eq'])

print("Valores has_secondary:")
print(data_x['has_secondary_use_agriculture'].sum())
print(data_x['has_secondary_use_hotel'].sum())
print(data_x['has_secondary_use_rental'].sum())
print(data_x['has_secondary_use_institution'].sum())
#[...]
print(data_x['has_secondary_use_other'].sum())
'''



# ----------------------------------------------------
# ------------- CATEGÓRICAS A NUMÉRICAS --------------
# ----------------------------------------------------
# Profesor
'''
from sklearn.preprocessing import LabelEncoder
mask = data_x.isnull()
data_x_tmp = data_x.fillna(9999)
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform)
data_x_nan = data_x_tmp.where(~mask, data_x)

#mÃ¡scara para luego recuperar los NaN
mask = data_x_tst.isnull()
#LabelEncoder no funciona con NaN, se asigna un valor no usado
data_x_tmp = data_x_tst.fillna(9999)

#se convierten categóricas en numéricas
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform)

#se recuperan los NaN
data_x_tst_nan = data_x_tmp.where(~mask, data_x_tst)

X = data_x_nan.values
X_tst = data_x_tst_nan.values
y = np.ravel(data_y.values)
'''


'''
# Cantidad de valores para cada atributo
for col in data_x.columns:
	print(data_x[col].value_counts())
'''



# ----------------------------------------------------
# --------------- PRUEBA DE ALGORITMOS ---------------
# ----------------------------------------------------
		
		
'''
from sklearn.linear_model import LogisticRegression
print('LOGISTIC REGRESSION\n')
lr = LogisticRegression()
X_norm = preprocessing.normalize(X)
lr, result = validacion_cruzada(lr,X_norm,y,skf)


from sklearn.naive_bayes import GaussianNB
print('NAIVE BAYES\n')
nb = GaussianNB()
nb, result = validacion_cruzada(nb,X,y,skf)


from sklearn.ensemble import RandomForestClassifier
print('RANDOM FOREST\n')
#rfm = RandomForestClassifier(n_estimators=20, n_jobs=1,
#				 random_state=SEED, max_features=None, min_samples_leaf=30)
rfm = RandomForestClassifier(random_state=SEED)
rfm, result = validacion_cruzada(rfm,X,y,skf)


import lightgbm as lgb
print("LIGHT GBM")
#lgbm = lgb.LGBMClassifier(objective='regression_l1',n_estimators=200,n_jobs=2)
lgbm = lgb.LGBMClassifier(random_state=SEED)
lgbm, result = validacion_cruzada(lgbm,X,y,skf)


from sklearn.linear_model import SGDClassifier
print('STOCHASTIC GRADIENT DESCENT\n')
sgd = SGDClassifier(loss='modified_huber', shuffle=True, random_state=SEED)
X_norm = preprocessing.normalize(X)
sgd, result = validacion_cruzada(sgd,X_norm,y,skf)


from sklearn.ensemble import AdaBoostClassifier
print('ADA BOOST')
ada = AdaBoostClassifier()
ada, result = validacion_cruzada(ada,X,y,skf)


from sklearn.neural_network import MLPClassifier
print('MLP CLASSIFIER')
mlp = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=SEED)
X_norm = preprocessing.normalize(X)
mlp, result = validacion_cruzada(mlp,X_norm,y,skf)


from catboost import CatBoostClassifier
print('CATBOOST')

categorical=[]
i=0
for col in data_x.columns:
	if(type(data_x[col][0]) is str):
		#print(i)
		categorical.append(i)
	i+=1
	
#print(categorical)

X = data_x.values
X_tst = data_x_tst.values
y = np.ravel(data_y.values)

cb = CatBoostClassifier(n_estimators=100, loss_function='MultiClass', logging_level='Silent')
cb, result = validacion_cruzada(cb,X,y,skf,categorical)
'''






# ----------------------------------------------------
# ----------------- POSIBLES SUBIDAS -----------------
# ----------------------------------------------------


# ***************
# *** PRIMERA ***
# ***************

# Enviado 1
'''
from sklearn.ensemble import RandomForestClassifier
print('RANDOM FOREST\n')
rfm = RandomForestClassifier(n_estimators=20, n_jobs=1,
				 random_state=SEED, max_features=None, min_samples_leaf=30)
rfm, result = validacion_cruzada(rfm,X,y,skf)

'''


# ***************
# *** SEGUNDA ***
# ***************

'''
import lightgbm as lgb
print("LIGHT GBM")
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_estimators=200,n_jobs=2)
lgbm, result = validacion_cruzada(lgbm,X,y,skf)
'''

# ***************
# *** TERCERA ***
# ***************

'''
import lightgbm as lgb
print("LIGHT GBM")
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_estimators=400,n_jobs=4)
lgbm, result = validacion_cruzada(lgbm,X,y,skf)
'''

# ***************
# *** CUARTA ****
# ***************

# Enviado 2
'''
from sklearn.ensemble import RandomForestClassifier
print('RANDOM FOREST\n')
rfm = RandomForestClassifier(n_estimators=125, n_jobs=1,
				 random_state=SEED, max_features=None, max_depth = 20)
rfm, result = validacion_cruzada(rfm,X,y,skf)
'''

# ***************
# *** QUINTA ****
# ***************

# Enviado 3
'''
import lightgbm as lgb
print("LIGHT GBM")
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_estimators=1000,n_jobs=4)
lgbm, result = validacion_cruzada(lgbm,X,y,skf)
'''

# ***************
# **** SEXTA ****
# ***************
'''
dict1 = {
	1: data_y.count()/(25124+data_y.count()),
	2: data_y.count()/(148259+data_y.count()),
	3: data_y.count()/(87218+data_y.count())
} # Ta bien


dict2 = {
	1: 1-(25124/data_y.count()),
	2: 1-(148259/data_y.count()),
	3: 1-(87218/data_y.count())
} # Well...

dict3 = {
	1: (148259/data_y.count()),
	2: (25124/data_y.count()),
	3: (87218/data_y.count())
} # No vale


import lightgbm as lgb
print("LIGHT GBM")
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_estimators=1000, 
						  n_jobs=4, class_weight=dict1)
lgbm, result = validacion_cruzada(lgbm,X,y,skf)
'''


# *************************
# ****** GRID SEARCH ******
# *************************
'''
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb
lgbm = lgb.LGBMClassifier(objective='regression_l1')


#import sklearn
#print(sorted(sklearn.metrics.SCORERS.keys()))


params_lgbm = {
		'feature_fraction':[i/10.0 for i in range(3,6)], 
		'learning_rate':[0.05,0.1], 
		'num_leaves':[30,50], 
		'n_estimators':[200]
}
grid = GridSearchCV(lgbm, params_lgbm, cv=3, n_jobs=4, verbose=1,
					scoring='f1_micro')
grid.fit(X,y)
print("Mejores parÃ¡metros:")
print(grid.best_params_)
#MEJORES PARAMETROS:
#{'feature_fraction': 0.5, 'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 50}

#print("\n------ LightGBM con los mejores parÃ¡metros de GridSearch...")
#gs, y_test_gs, y_prob_gs = validacion_cruzada(grid.best_estimator_,X,y,skf)
'''

# ***************
# **** SIETE ****
# ***************

# Enviado 4
'''
import lightgbm as lgb
print("LIGHT GBM")
lgbm = lgb.LGBMClassifier(objective='regression_l1',feature_fraction=0.5,
						  n_estimators=1000,num_leaves=50,n_jobs=4)
lgbm, result = validacion_cruzada(lgbm,X,y,skf)
'''

# ***************
# **** OCHO *****
# ***************

'''
import lightgbm as lgb
print("LIGHT GBM")
lgbm = lgb.LGBMClassifier(objective='regression_l1',feature_fraction=0.5,
						  n_estimators=2000,num_leaves=50,n_jobs=-1)
lgbm, result = validacion_cruzada(lgbm,X,y,skf)
'''


# ***************
# **** NUEVE ****
# ***************

# Enviado 5
'''
import lightgbm as lgb
import smote_variants as sv

#oversampler = sv.kmeans_SMOTE(proportion=1.0, n_neighbors=3,n_jobs=-1,random_state=SEED)

oversampler = sv.polynom_fit_SMOTE(proportion=1.0,topology='star',random_state=SEED)

X,y=oversampler.sample(X,y)

print("LIGHT GBM")
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_estimators=2000,
						  num_leaves=50,n_jobs=-1)
lgbm, result = validacion_cruzada(lgbm,X,y,skf)
'''


# ***************
# **** DIEZ *****
# ***************
# Con Grid Search
'''
import lightgbm as lgb
import smote_variants as sv

#oversampler = sv.kmeans_SMOTE(proportion=1.0, n_neighbors=3,n_jobs=-1,random_state=SEED)

oversampler = sv.polynom_fit_SMOTE(proportion=1.0,topology='star',random_state=SEED)

X,y=oversampler.sample(X,y)


from sklearn.model_selection import GridSearchCV

lgbm = lgb.LGBMClassifier(objective='regression_l1',n_jobs=-1)

#import sklearn
#print(sorted(sklearn.metrics.SCORERS.keys()))


params_lgbm = {
		'num_leaves':[i*5 for i in range(8,13)], 
		'n_estimators':[i*250 for i in range(7,13)]
}

grid = GridSearchCV(lgbm, params_lgbm, cv=2, n_jobs=-1, verbose=1,
					scoring='f1_micro')
grid.fit(X,y)
print("Mejores parámetros:")
print(grid.best_params_)
#{'n_estimators': 1750, 'num_leaves': 40}

print("\n------ LightGBM con los mejores parÃ¡metros de GridSearch...")
lgbm, result = validacion_cruzada(grid.best_estimator_,X,y,skf)
'''


# ***************
# **** ONCE *****
# ***************

# Enviado 6
'''
import smote_variants as sv

#oversampler = sv.kmeans_SMOTE(proportion=1.0, n_neighbors=3,n_jobs=-1,random_state=SEED)

oversampler = sv.polynom_fit_SMOTE(proportion=1.0,topology='star',random_state=SEED)

X,y=oversampler.sample(X,y)


from sklearn.ensemble import RandomForestClassifier
print('RANDOM FOREST\n')
rfm = RandomForestClassifier(n_estimators=125, n_jobs=-1,
				 random_state=SEED, max_features=None, max_depth = 20)
rfm, result = validacion_cruzada(rfm,X,y,skf)
'''


# ***************
# **** DOCE *****
# ***************

# Enviado 7
'''
import smote_variants as sv

oversampler = sv.MulticlassOversampling(
		sv.polynom_fit_SMOTE(topology='mesh',random_state=SEED)
		)

X,y=oversampler.sample(X,y)

import collections

print(collections.Counter(y))

from sklearn.ensemble import RandomForestClassifier
print('RANDOM FOREST\n')
rfm = RandomForestClassifier(n_estimators=125, n_jobs=-1,
				 random_state=SEED, max_features=None, max_depth = 20)
rfm, result = validacion_cruzada(rfm,X,y,skf)
'''


# ***************
# *** TRECE *****
# ***************


'''
from sklearn.ensemble import BaggingClassifier
import lightgbm as lgb
print('BAGGING')
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_estimators=200,n_jobs=-1)
bag = BaggingClassifier(base_estimator=lgbm, n_jobs=-1, random_state=SEED)
bag, result = validacion_cruzada(bag,X,y,skf)
'''


# ***************
# *** CATORCE ***
# ***************


# Enviado 8
'''
import lightgbm as lgb
import smote_variants as sv

oversampler = sv.MulticlassOversampling(
		sv.polynom_fit_SMOTE(topology='mesh',random_state=SEED)
		)

X,y=oversampler.sample(X,y)

import collections

print(collections.Counter(y))


from sklearn.model_selection import GridSearchCV

lgbm = lgb.LGBMClassifier(objective='regression_l1',n_jobs=-1)

params_lgbm = {
		'learning_rate':[i*0.01 for i in range(8,13)], #8-13
		'num_leaves':[i*5 for i in range(3,6)], #3-6
		'n_estimators':[i*250 for i in range(4,8)] #4-8
}

grid = GridSearchCV(lgbm, params_lgbm, cv=2, n_jobs=-1, verbose=1,
					scoring='f1_micro')
grid.fit(X,y)
print("Mejores parámetros:")
print(grid.best_params_)
{'learning_rate': 0.12, 'n_estimators': 1750, 'num_leaves': 25}

print("\n------ LightGBM con los mejores parámetros de GridSearch...")
lgbm, result = validacion_cruzada(grid.best_estimator_,X,y,skf)
'''


# ***************
# *** QUINCE ****
# ***************

# Enviado 9
'''
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

lgbm = lgb.LGBMClassifier(objective='regression_l1',n_jobs=-1)

params_lgbm = {
		'learning_rate':[i*0.01 for i in range(8,13)], #0.08-0.12
		'num_leaves':[i*5 for i in range(3,6)], #15-25
		'n_estimators':[i*250 for i in range(8,12)] #1000-2750
}

grid = GridSearchCV(lgbm, params_lgbm, cv=2, n_jobs=-1, verbose=1,
					scoring='f1_micro')
grid.fit(X,y)
print("Mejores parámetros:")
print(grid.best_params_)
#{'learning_rate': 0.12, 'n_estimators': 2750, 'num_leaves': 20}

print("\n------ LightGBM con los mejores parámetros de GridSearch...")
lgbm, result = validacion_cruzada(grid.best_estimator_,X,y,skf)
'''










'''''''''''''''''''''''''''''''''''''''''''''''''''''
******* A PARTIR DE AQUI PREPROCESADO PERFES ********
'''''''''''''''''''''''''''''''''''''''''''''''''''''

# ----------------------------------------------------
# ------------- CATEGÓRICAS A NUMÉRICAS --------------
# ----------------------------------------------------


cat_dict = {}

for col in data_x.columns:
	if(type(data_x[col][0]) is str):
		categories = data_x[col].value_counts().index
		add_dict = {}
		for i in range(0,categories.size):
			key = categories[i]
			value = ord(categories[i])
			add_dict[key] = value
		
		cat_dict[col] = add_dict

data_x.replace(cat_dict, inplace=True)
data_x_tst.replace(cat_dict, inplace=True)

X = data_x.values
X_tst = data_x_tst.values
y = np.ravel(data_y.values)



# ***************
# ** DIECISEIS **
# ***************
# Los resultados del gridsearch anterior
# pero con buen preprocesado

# Enviado 10
'''
import lightgbm as lgb
print("LIGHT GBM")
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_jobs=-1,
						  learning_rate=0.12,n_estimators=2750,num_leaves=20)
lgbm, result = validacion_cruzada(lgbm,X,y,skf)
'''



# ***************
# * DIECISIETE **
# ***************

# Enviado 11
'''
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

lgbm = lgb.LGBMClassifier(objective='regression_l1',n_jobs=-1)

params_lgbm = {
		'learning_rate':[i*0.005 for i in range(20,26)], #0.1-0.125
		'num_leaves':[i*5 for i in range(5,8)], #25-35
		'n_estimators':[i*250 for i in range(10,16)] #2500-3750
}

grid = GridSearchCV(lgbm, params_lgbm, cv=2, n_jobs=-1, verbose=1,
					scoring='f1_micro')
grid.fit(X,y)
print("Mejores parámetros:")
print(grid.best_params_)
#{'learning_rate': 0.1, 'n_estimators': 2750, 'num_leaves': 25}


print("\n------ LightGBM con los mejores parámetros de GridSearch...")
lgbm, result = validacion_cruzada(grid.best_estimator_,X,y,skf)
'''




# ***************
# ** DIECIOCHO **
# ***************
# Realmente es una prueba para ver si estoy haciendo bien el gridsearch
'''
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

lgbm = lgb.LGBMClassifier(objective='regression_l1',n_jobs=-1)

params_lgbm = {
		'learning_rate':[i*0.01 for i in range(10,12)],
		'num_leaves':[i*5 for i in range(4,6)],
		'n_estimators':[i*500 for i in range(4,6)]
}

grid = GridSearchCV(lgbm, params_lgbm, cv=2, n_jobs=-1, verbose=1,
					scoring='f1_micro')
grid.fit(X,y)
print("Mejores parámetros:")
print(grid.best_params_)
#{'learning_rate': 0.1, 'n_estimators': 2500, 'num_leaves': 25}

print("\n------ LightGBM con los mejores parámetros de GridSearch...")
lgbm, result = validacion_cruzada(grid.best_estimator_,X,y,skf)

'''






#	****************************************
#	***************** NOTA *****************
#	****************************************
'''
Probar con objective = 'multiclassova' (one-vs-all) 
Con multiclassova utilizar el is_unbalance
Probar SMOTE
'''

'''
INFO QUE TENGO HASTA AHORA:
learning_rate <= 0.1
num_leaves aprox 25
2250 <= n_estimators <= 2750	
Hacer
params_lgbm = {
		'learning_rate':[0.08,0.09,0.1],
		'num_leaves':[23,24,25,26,27],
		'n_estimators':[2300,2400,2500,2600,2700]
}

'''








# ***************
# * DIECINUEVE **
# ***************


# Enviado 12
'''
import smote_variants as sv

oversampler = sv.MulticlassOversampling(
		sv.polynom_fit_SMOTE(topology='mesh',random_state=SEED)
		)

X,y=oversampler.sample(X,y)

import collections
print(collections.Counter(y))

import lightgbm as lgb
print("LIGHT GBM")
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_jobs=-1,
						  learning_rate=0.1,n_estimators=2750,num_leaves=25)
#lgbm, result = validacion_cruzada(lgbm,X,y,skf)
'''


# ****************
# ****** 20 ******
# ****************

# Enviado 13
'''
from sklearn.ensemble import BaggingClassifier
import lightgbm as lgb
print('BAGGING LGBM')
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_jobs=-1,
						  learning_rate=0.1,n_estimators=2750,num_leaves=25)
bag = BaggingClassifier(base_estimator=lgbm, n_jobs=-1, random_state=SEED)
bag, result = validacion_cruzada(bag,X,y,skf)
'''



# ****************
# ****** 21 ******
# ****************

# Enviado 14
'''
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

lgbm = lgb.LGBMClassifier(objective='regression_l1',n_jobs=-1)

params_lgbm = {
		'learning_rate':[0.08,0.09,0.1],
		'num_leaves':[23,24,25,26,27,28],
		'n_estimators':[2300,2400,2500,2600,2700]
}

grid = GridSearchCV(lgbm, params_lgbm, cv=2, n_jobs=-1, verbose=1,
					scoring='f1_micro')
grid.fit(X,y)
print("Mejores parámetros:")
print(grid.best_params_)
#{'learning_rate': 0.09, 'n_estimators': 2700, 'num_leaves': 28}


print("\n------ LightGBM con los mejores parámetros de GridSearch...")
lgbm, result = validacion_cruzada(grid.best_estimator_,X,y,skf)
'''


# ****************
# ****** 22 ******
# ****************

# Enviado 15
'''
from sklearn.ensemble import BaggingClassifier
import lightgbm as lgb
print('BAGGING LGBM')
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_jobs=-1,
						  learning_rate=0.1,n_estimators=2750,num_leaves=25)
bag = BaggingClassifier(base_estimator=lgbm,n_jobs=-1,
						n_estimators=20,random_state=SEED)
#bag, result = validacion_cruzada(bag,X,y,skf)
'''




# ****************
# ****** 23 ******
# ****************
# Probamos el algoritmo con mejores
# resultados en el anterior preprocesado

# Enviado 16
'''
import smote_variants as sv

oversampler = sv.MulticlassOversampling(
		sv.polynom_fit_SMOTE(topology='star',random_state=SEED)
		)

X,y=oversampler.sample(X,y)

#import collections
#print(collections.Counter(y))


from sklearn.ensemble import RandomForestClassifier
print('RANDOM FOREST\n')
rfm = RandomForestClassifier(n_estimators=125, n_jobs=-1,
				 random_state=SEED, max_features=None, max_depth = 20)
#rfm, result = validacion_cruzada(rfm,X,y,skf)
'''



# ****************
# ****** 24 ******
# ****************

# Enviado 17
'''
import smote_variants as sv

oversampler = sv.MulticlassOversampling(
		sv.polynom_fit_SMOTE(topology='star',random_state=SEED)
		)

X,y=oversampler.sample(X,y)

#import collections
#print(collections.Counter(y))


from sklearn.ensemble import RandomForestClassifier
print('RANDOM FOREST\n')
rfm = RandomForestClassifier(n_estimators=500, n_jobs=-1,
				 random_state=SEED, max_features=None, max_depth = 20)
#rfm, result = validacion_cruzada(rfm,X,y,skf)
'''


# ****************
# ****** 25 ******
# ****************

# Enviado 18
'''
from sklearn.ensemble import RandomForestClassifier
print('RANDOM FOREST\n')
rfm = RandomForestClassifier(n_estimators=500, n_jobs=-1,
				 random_state=SEED, max_features=None, max_depth = 20)
#rfm, result = validacion_cruzada(rfm,X,y,skf)
'''





# ****************
# ****** 26 ******
# ****************

# Enviado 20
'''
from sklearn.ensemble import RandomForestClassifier
print('RANDOM FOREST\n')
rfm = RandomForestClassifier(n_estimators=1200, n_jobs=-1,
				 random_state=SEED, max_features=None, max_depth = 20, verbose=1)
#rfm, result = validacion_cruzada(rfm,X,y,skf)
'''



# ****************
# ****** 27 ******
# ****************
# Igualamos el numero de elementos de cada clase al número de elementos
# de la clase intermedia (87218 instancias)


# Enviado 19
'''
X_all = np.append(X,X_tst,axis=0)

from sklearn import preprocessing

X_all_normalized = preprocessing.normalize(X_all, norm='l2')

X_norm = X_all_normalized[:len(X)]
X_tst_norm = X_all_normalized[len(X):]

#from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

print("\nTotal:")
print(sorted(Counter(y).items()))


#under = ClusterCentroids(sampling_strategy='majority',random_state=SEED,ratio={2:87218})
under = RandomUnderSampler(random_state=SEED, ratio={2:87218})
X_resampled, y_resampled = under.fit_resample(X_norm, y)

print("\nUndersampling")
print(sorted(Counter(y_resampled).items()))

import smote_variants as sv
oversampler = sv.kmeans_SMOTE(proportion=1.0, n_neighbors=3,n_jobs=-1,random_state=SEED)
X,y=oversampler.sample(X_resampled,y_resampled)
X_tst=X_tst_norm

print("\nOversampling:")
print(sorted(Counter(y).items()))


import lightgbm as lgb
print("LIGHT GBM")
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_jobs=-1,
						  learning_rate=0.1,n_estimators=2750,num_leaves=25)
#lgbm, result = validacion_cruzada(lgbm,X,y,skf)

'''



# ****************
# ****** 28 ******
# ****************

# Enviado 21
'''
import smote_variants as sv

oversampler = sv.MulticlassOversampling(
		sv.polynom_fit_SMOTE(topology='mesh',random_state=SEED)
		)

X,y=oversampler.sample(X,y)

#import collections
#print(collections.Counter(y))

import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

lgbm = lgb.LGBMClassifier(objective='regression_l1',n_jobs=-1,
						  n_estimators=2750)

params_lgbm = {
		'learning_rate':[0.06,0.08,0.1],
		'max_bin':[200,250,300,350,400,450],
		'num_leaves':[12,16,20,24,28]
}

grid = GridSearchCV(lgbm, params_lgbm, cv=2, n_jobs=-1, verbose=1,
					scoring='f1_micro')
grid.fit(X,y)
print("Mejores parámetros:")
print(grid.best_params_)
#0.1, 400, 28

print("\n------ LightGBM con los mejores parámetros de GridSearch...")
lgbm, result = validacion_cruzada(grid.best_estimator_,X,y,skf)
'''


# ****************
# ****** 29 ******
# ****************

# Enviado 22
'''
import smote_variants as sv

oversampler = sv.MulticlassOversampling(
		sv.polynom_fit_SMOTE(topology='mesh',random_state=SEED)
		)

X,y=oversampler.sample(X,y)

#import collections
#print(collections.Counter(y))

import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

lgbm = lgb.LGBMClassifier(objective='regression_l1',n_jobs=-1,
						  n_estimators=2750, learning_rate=0.1, max_bin=400)

params_lgbm = {
		'num_leaves':[28,29,30,31,32,33,34]
}

grid = GridSearchCV(lgbm, params_lgbm, cv=2, n_jobs=-1, verbose=1,
					scoring='f1_micro')
grid.fit(X,y)
print("Mejores parámetros:")
print(grid.best_params_)
#{'num_leaves': 34}

print("\n------ LightGBM con los mejores parámetros de GridSearch...")
lgbm, result = validacion_cruzada(grid.best_estimator_,X,y,skf)
'''

# ****************
# ****** 30 ******
# ****************
# NO LO HE ENVIADO PERO PUEDE SER BUENO
'''
#import smote_variants as sv
#oversampler = sv.MulticlassOversampling(
#		sv.polynom_fit_SMOTE(topology='mesh',random_state=SEED)
#		)
#X,y=oversampler.sample(X,y)
#import collections
#print(collections.Counter(y))

import lightgbm as lgb
print("LIGHT GBM")
lgbm = lgb.LGBMClassifier(objective='multiclassova',n_jobs=-1,
						  learning_rate=0.09,n_estimators=3000,num_leaves=34,
						  max_bin=400)
lgbm, result = validacion_cruzada(lgbm,X,y,skf)
'''


# ****************
# ****** 31 ******
# ****************
# El anterior parece bueno asi que voy a hacerle bagging

# Enviado 23
'''
from sklearn.ensemble import BaggingClassifier
import lightgbm as lgb
print('BAGGING LGBM')
lgbm = lgb.LGBMClassifier(objective='multiclassova',n_jobs=-1,
						  learning_rate=0.09,n_estimators=3000,num_leaves=34,
						  max_bin=400)
bag = BaggingClassifier(base_estimator=lgbm,n_jobs=-1,
						n_estimators=15,random_state=SEED)
#bag, result = validacion_cruzada(bag,X,y,skf)
'''


# ****************
# ****** 32 ******
# ****************
# Grid de bagging para la ultima noche de competición :)
# NO VA :()
# TerminatedWorkerError: A worker process managed by the executor
# was unexpectedly terminated. This could be caused by a segmentation
# fault while calling the function or by an excessive memory usage
# causing the Operating System to kill the worker. The exit codes of
# the workers are {SIGKILL(-9)}
'''
import lightgbm as lgb
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV

lgbm = lgb.LGBMClassifier(objective='multiclassova',n_jobs=-1,
						  learning_rate=0.09,n_estimators=3000,num_leaves=34,
						  max_bin=400)

bag = BaggingClassifier(base_estimator=lgbm,n_jobs=-1,random_state=SEED)

params_bag = {
		'n_estimators':[15,17,19],
		'bootstrap':[True, False]
}

grid = GridSearchCV(bag, params_bag, cv=2, n_jobs=-1, verbose=2,
					scoring='f1_micro')
grid.fit(X,y)
print("Mejores parámetros:")
print(grid.best_params_)

#print("\n------ Bagging con los mejores parámetros de GridSearch...")
#lgbm, result = validacion_cruzada(grid.best_estimator_,X,y,skf)
bag = grid.best_estimator_
print(bag)
'''



# ****************
# ****** 33 ******
# ****************
# SMOTE MESH

# Enviado 24
'''
import smote_variants as sv
oversampler = sv.MulticlassOversampling(
		sv.polynom_fit_SMOTE(topology='mesh',random_state=SEED)
		)
X,y=oversampler.sample(X,y)

import collections
height=[collections.Counter(y)[1],collections.Counter(y)[2],collections.Counter(y)[3]]
print(height)
print(collections.Counter(y))
plt.bar(['low damage','medium damage','almost complete destruction'],
		height,color=['sienna', 'skyblue', 'C1'])
plt.savefig("./plots/value_counts_SMOTE_mesh.png")
plt.clf()


from sklearn.ensemble import BaggingClassifier
import lightgbm as lgb
print('BAGGING LGBM')
lgbm = lgb.LGBMClassifier(objective='multiclassova',n_jobs=-1,
						  learning_rate=0.09,n_estimators=3000,num_leaves=34,
						  max_bin=400)
bag = BaggingClassifier(base_estimator=lgbm,n_jobs=-1,
						n_estimators=15,random_state=SEED)
#bag, result = validacion_cruzada(bag,X,y,skf)
'''


# ****************
# ****** 34 ******
# ****************
# SMOTE STAR

# Enviado 25
'''
import smote_variants as sv
oversampler = sv.MulticlassOversampling(
		sv.polynom_fit_SMOTE(topology='star',random_state=SEED)
		)
X,y=oversampler.sample(X,y)


import collections
height=[collections.Counter(y)[1],collections.Counter(y)[2],collections.Counter(y)[3]]
print(height)
print(collections.Counter(y))
plt.bar(['low damage','medium damage','almost complete destruction'],
		height,color=['sienna', 'skyblue', 'C1'])
plt.savefig("./plots/value_counts_SMOTE_star.png")
#plt.show()
plt.clf()



from sklearn.ensemble import BaggingClassifier
import lightgbm as lgb
print('BAGGING LGBM')
lgbm = lgb.LGBMClassifier(objective='multiclassova',n_jobs=-1,
						  learning_rate=0.09,n_estimators=3000,num_leaves=34,
						  max_bin=400)
bag = BaggingClassifier(base_estimator=lgbm,n_jobs=-1,
						n_estimators=15,random_state=SEED)
#bag, result = validacion_cruzada(bag,X,y,skf)

'''



# ****************
# ****** 35 ******
# ****************
# ÚLTIMO INTENTO !!!!!!!!!!!!!

# Enviado 26
from sklearn.ensemble import BaggingClassifier
import lightgbm as lgb
print('BAGGING LGBM')
lgbm = lgb.LGBMClassifier(objective='multiclassova',n_jobs=-1,
						  learning_rate=0.09,n_estimators=3250,num_leaves=34,
						  max_bin=400)
bag = BaggingClassifier(base_estimator=lgbm,n_jobs=-1,
						n_estimators=15,random_state=SEED, bootstrap=False)
bag, result = validacion_cruzada(bag,X,y,skf)







clf = bag

t = time.time()
clf = clf.fit(X,y)
tiempo = time.time()-t
y_pred_tra = clf.predict(X)
print("F1 score (tra): {:.4f}, tiempo: {:6.2f} segundos"
	  .format(f1_score(y,y_pred_tra,average='micro'),tiempo))

t = time.time()
y_pred_tst = clf.predict(X_tst)
tiempo = time.time()-t

print("Tiempo predict: {:6.2f} segundos".format(tiempo))


df_submission = pd.read_csv('../dataset/nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission35_bagging_lgbm.csv", index=False)


