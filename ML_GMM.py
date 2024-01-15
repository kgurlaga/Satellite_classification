import os
import rasterio as rio
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
os.chdir("C:/Users/kgurlaga14/Desktop/ML/zaliczenie")


#%%
with rio.open('zeland.tif') as input_raster:
    input_data = input_raster.read()
    profile = input_raster.profile
    
    with rio.open('maska.tif') as mask_raster:
        mask_data = mask_raster.read(1)
        
        masked_data = np.zeros_like(input_data)
        for band_idx in range(input_raster.count):
            masked_data[band_idx] = np.where(mask_data == 1, 0, input_data[band_idx])
        with rio.open('terrain.tif', 'w', **profile) as dest:
            dest.write(masked_data)
            

#%%
with rio.open('terrain.tif') as src:
    raster_array = src.read()
    reshaped_array = raster_array.reshape(raster_array.shape[0], -1).T
    terrain = pd.DataFrame(reshaped_array, columns=[f'Band_{i+1}' for i in range(raster_array.shape[0])])

terrain = terrain.rename(columns={'Band_11':'Klasa'})

#%%
unique = terrain['Klasa'].unique()

terrain['Klasa'] = terrain['Klasa'].replace(0, np.nan)


#%%class to array
with rio.open('terrain.tif') as src:
    band_11_data = src.read(11)

band_11_data = band_11_data.astype(float)
band_11_data[band_11_data == 0.0] = np.nan


#%%GMM

features = terrain.iloc[:, :-1]
class_column = terrain.iloc[:, -1]

nan_rows = class_column.isna()

data_for_gmm = features.loc[~nan_rows]


gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(data_for_gmm)

nan_rows_data = features.loc[nan_rows]
predicted_labels = gmm.predict(nan_rows_data)

class_column[nan_rows] = predicted_labels

terrain.iloc[:, -1] = class_column



#%% Export

raster_data = terrain.values
raster_data = raster_data.reshape(11, 5490, 5490)
with rio.open('terrain2.tif', 'w', **profile) as dest:
        dest.write(raster_data)













