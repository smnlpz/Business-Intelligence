#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:35:18 2019

@author: simon
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
from sklearn import mixture
#from sklearn.cluster import KMeans
from sklearn import metrics
from math import floor

SEED=26504148

def norm_to_zero_one(df):
	return (df - df.min()) * 1.0 / (df.max() - df.min())


def scatter(X,nombre):
	print("\nGenerando scatter matrix...")
	#se aÃ±ade la asignaciÃ³n de clusters como columna a X
	#X_kmeans = pd.concat([X, clusters], axis=1)
	sns.set()
	variables = list(X)
	variables.remove('cluster')
	sns_plot = sns.pairplot(X, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
	sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
	sns_plot.savefig("./resultados/scatter/"+nombre+"_scatter.png")
	plt.clf()
	print('Hecho!')



def kplot(X, name, k, usadas):
	print("\nGenerando kplot...")
	n_var = len(usadas)
	fig, axes = plt.subplots(k, n_var, sharex='col', figsize=(15,10))
	fig.subplots_adjust(wspace=0.2)
	colors = sns.color_palette(palette=None, n_colors=k, desat=None)
	
	for i in range(k):
		dat_filt = X.loc[X['cluster']==i]
		for j in range(n_var):
			sns.kdeplot(dat_filt[usadas[j]], shade=True, color=colors[i], ax=axes[i,j])
	
	plt.savefig("./resultados/kplot/"+name+"_kplot.png")
	plt.clf()
	print('Hecho!')


def box_plot(X, name, k, usadas):
	print("\nGenerando boxplot...")
	n_var = len(usadas)
	fig, axes = plt.subplots(k, n_var, sharey=True, figsize=(16, 16))
	fig.subplots_adjust(wspace=0.4, hspace=0.4)
	colors = sns.color_palette(palette=None, n_colors=k, desat=None)
	rango = []

	for i in range(n_var):
		rango.append([X[usadas[i]].min(), X[usadas[i]].max()])

	for i in range(k):
		dat_filt = X.loc[X['cluster']==i]
		for j in range(n_var):
			ax = sns.boxplot(dat_filt[usadas[j]], color=colors[i], ax=axes[i, j])
			ax.set_xlim(rango[j][0], rango[j][1])
	
	plt.savefig("./resultados/boxplot/" + name + "_boxplot.png")
	plt.clf()
	print('Hecho!')

	
def heat_map(Data, algorithm, nombre,cuantos):
	print('\nGenerando heatmap...')
	
	
	Data_norm = Data.apply(norm_to_zero_one)
	Data_norm["cluster"]=Data_norm["cluster"]*(cuantos-1)
	
	centers = Data_norm.groupby("cluster").mean()
	centers_desnormal = Data.groupby("cluster").mean()
	
	plt.figure(figsize = (12,8.4))
	hm = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
	hm.figure.savefig("./resultados/heatmap/"+nombre+"_heatmap.png")
	hm.figure.clf()
	print('Hecho!')
	



def cluster_it(Data, algoritmo, nombre, usadas, jerarquico=False, method_jerar='ward'):
	print('EJECUTANDO: ' + nombre + '\n\n')
	
	Data_norm = Data.apply(norm_to_zero_one)
	name, algorithm = (nombre, algoritmo)
	cluster_predict = {}
	k = {}
	
	t = time.time()
	cluster_predict[name] = algorithm.fit_predict(Data_norm)
	tiempo = time.time() - t
	k[name] = len(set(cluster_predict[name]))
	
	
	print("Número de clusters (k): {:3.0f} ".format(k[name]),end='\n')	
	print("Tiempo: {:.2f} segundos ".format(tiempo), end='\n')
	
	metric_CH = metrics.calinski_harabasz_score(Data_norm, cluster_predict[name])
	#metric_CH = metrics.calinski_harabasz_score(Data_norm, clusters)
	print("Calinski-Harabaz Index: {:.3f}".format(metric_CH), end='\n')
	
	#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos,
	#digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
	muestra_silhoutte = 0.2 if (len(Data) > 10000) else 1.0
	metric_SC = metrics.silhouette_score(Data_norm, cluster_predict[name],
						  metric='euclidean',
						  sample_size=floor(muestra_silhoutte*len(Data)),
						  random_state=SEED)
	
	print("Silhouette Coefficient: {:.5f}".format(metric_SC))
	
	#se convierte la asignaciÃ³n de clusters a DataFrame
	clusters = pd.DataFrame(cluster_predict[name],
						 index=Data.index,columns=['cluster'])
	
	#y se aÃ±ade como columna a X
	Data_cluster = pd.concat([Data, clusters], axis=1)
	
	if(k[name]>10):
		#Filtro quitando los elementos (outliers) que caen en clusters muy pequeÃ±os en el jerÃ¡rquico
		min_size = len(Data_cluster.index)*0.05
		Data_filtrado = Data_cluster[Data_cluster.groupby('cluster').cluster.transform(len) > min_size]
		k_filtrado = len(set(Data_filtrado['cluster']))
		print("De los {:.0f} clusters hay {:.0f} con mÃ¡s de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k[name],k_filtrado,min_size,len(Data_cluster),len(Data_filtrado)))
		print(Data_filtrado.describe())
		
		Data_cluster=Data_filtrado
		k[name]=k_filtrado
	
	print("\nTamaño del subset:")
	print(len(Data.index))
	
	print("Tamaño de cada cluster:")
	size=clusters['cluster'].value_counts()
	
	for num,i in size.iteritems():
		print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
	
	
	#scatter(Data_cluster, name)
	#heat_map(Data_cluster, algorithm, name,k[name])	
	#box_plot(Data_cluster, name, k[name], usadas)
	#kplot(Data_cluster, name, k[name], usadas)
	'''
	if(jerarquico):
		print("\nGenerando dendogram...")
		Data_norm_DF = pd.DataFrame(Data_norm,index=Data.index,columns=usadas)
		hm = sns.clustermap(Data_norm_DF, method=method_jerar, col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)
		hm.fig.subplots_adjust(wspace=.03, hspace=.03)
		hm.fig.savefig("./resultados/dendogram/"+name+"_dendo.png")
		plt.clf()
		print('Hecho!')
	'''
	print('\n\n')
	
	





''' LECTURA DE DATOS Y EJECUCIÓN DE LOS ALGORITMOS '''

datos = pd.read_csv('../dataset/mujeres_fecundidad_INE_2018.csv')

for col in datos:
	datos[col].fillna(datos[col].mean(), inplace=True)


'''
SUBSETS
Madres con un embarazo no deseado que tiene un hijo
Andaluces con tres hijos y más de 5000 euros al mes (mu poquitos)
Dos o más hijos y menos de 2000 euros al mes
Militares con hijos
Tuvieron el primero hijo a los 18 años
Fachas (Ideas conservadoras)
Putos rojos (Ideas liberales)
'''
'''
Interesantes:
INGREHOG_INTER -> Ingresos mensuales netos del hogar en intervalos
EDADHIJO1 -> Edad al primer hijo biológico
EMBDESEADO -> Embarazo deseado (1 si; 6 no)
NHIJOS -> Número de hijos suyos o de su pareja
NHIJOBIO -> Número de hijos biológicos
OCUPA -> Ocupación que desempeña en este trabajo
	06	Profesionales de la salud
	62	Ocupaciones militares

T_OPINA = 1 ACUERDO, 2 NSNC, 3 DESACUERDO
V_CRECERFELIZ -> Un niño necesita un hogar con su padre y su madre para crecer felizmente
V_HOMOSEXUAL -> Está bien que las parejas homosexuales tengan los mismos derechos que las parejas heterosexuales
V_PRIORIDADM -> Para una mujer, la prioridad debe ser su familia más que la carrera profesional
V_TARDOM -> Los hombres deben participar en las tareas domésticas en la misma medida que las mujeres

TRABAJAACT -> Tiene en estos momentos un trabajo remunerado
SITUACIONACT -> Si no trabaja, cuál es su sitiación actual
ESTUDIOSA -> Nivel de estudios alcanzados

INGRESOSPAR -> Ingresos mensuales netos de la pareja
DIFICULTAD -> Tiene dificultades para llegar a fin de mes


'''

subset_dcha = datos.loc[(datos['V_CRECERFELIZ']==1) & (datos['V_HOMOSEXUAL']==3)]
subset_izda = datos.loc[(datos['V_HOMOSEXUAL']==1) & (datos['V_PRIORIDADM']==3) & (datos['V_TARDOM']==1)]
subset_militares = datos.loc[(datos['NHIJOS']>0) & (datos['OCUPA']==62)]
subset_2000 = datos.loc[(datos['NHIJOS']>0) & (datos['SITSENTI']==1)]
subset_18 = datos.loc[(datos['EDADHIJO1']<19)]
subset_paro = datos.loc[(datos['TRABAJAACT']==6) & (datos['SITSENTI']>1)]


usadas_paro = ['EDAD', 'NHIJOS', 'INGRESOSPAR', 'METROSVI','PAGOVI']
usadas_ideas = ['EDAD', 'NHIJOS', 'INGREHOG_INTER', 'ESTUDIOSA']
usadas_2000 = ['EDAD', 'NHIJOS', 'INGREHOG_INTER', 'NDESEOHIJO']


X_dcha = subset_dcha[usadas_ideas]
X_izda = subset_izda[usadas_ideas]
X_2000 = subset_2000[usadas_2000]
X_paro = subset_paro[usadas_paro]

k_means = cluster.KMeans(init='k-means++', n_clusters=5, n_init=5, random_state=SEED)
mini_batch = cluster.MiniBatchKMeans(init='k-means++', n_clusters=5, n_init=5, random_state=SEED)
agg_ward = cluster.AgglomerativeClustering(n_clusters=5, linkage='ward')
aff_prop = cluster.AffinityPropagation()
birch_ = cluster.Birch(n_clusters=5, threshold=0.1)
db_scan = cluster.DBSCAN()
gaussian = mixture.GaussianMixture(init_params='kmeans', random_state=SEED)
mean_shift = cluster.MeanShift(bandwidth=0.33)


subconjuntos=[X_paro,X_2000,X_izda,X_dcha]
usadas=[usadas_paro, usadas_2000, usadas_ideas, usadas_ideas]
names_sub=["paro","2000","izda","dcha"]

algoritmos=[k_means,mini_batch,agg_ward,birch_,mean_shift]
names_alg=["_kmeans","_minibatch","_aggward","_birch","_meanshift"]

i=0
for sub in subconjuntos:
	j=0
	for alg in algoritmos:
		jerar=False
		if names_alg[j]=="_aggward":
			jerar=True
		cluster_it(sub,alg,names_sub[i]+names_alg[j],usadas[i],jerar)
		j=j+1
	i=i+1


'''

MODIFICANDO PARÁMETROS DE LOS ALGORITMOS

'''

k_means_3=cluster.KMeans(init='k-means++', n_clusters=2, n_init=5, random_state=SEED)
k_means_8=cluster.KMeans(init='k-means++', n_clusters=8, n_init=5, random_state=SEED)
k_means_random=cluster.KMeans(init='random', n_clusters=5, n_init=5, random_state=SEED)

agg_complete = cluster.AgglomerativeClustering(n_clusters=5, linkage='complete')
agg_average = cluster.AgglomerativeClustering(n_clusters=5, linkage='average')


algoritmos_params=[k_means_3,k_means_8,k_means_random,agg_complete,agg_average]
names_params=["_kmeans_3","_kmeans_8","_kmeans_random","_aggcomplete","_aggaverage"]


i=0
for sub in subconjuntos:
	j=0
	for alg in algoritmos_params:
		jerar=False
		method=''
		if names_params[j]=="_aggcomplete":
			jerar=True
			method='complete'
		if names_params[j]=="_aggaverage":
			jerar=True
			method='average'
		
		cluster_it(sub,alg,names_sub[i]+names_params[j],usadas[i],jerar,method)
		j=j+1
	i=i+1





