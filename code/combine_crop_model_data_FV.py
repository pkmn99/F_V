import pandas as pd
import numpy as np

# To create data for F&V model

# Get end year for file of different crop types 
def get_end_year(crop_type):
    s = {'tomatoes':2017,'potatoes':2016,'sweetcorn':2013}
    return s[crop_type]

"""
Load soil variable 
"""
def get_soil_data(var):
    
    soil1 = pd.read_csv('/mnt/d/Project/data/GEE/US_soil_corn_county/zonal_corn_%s_mean_0_5.csv'%var,
                header=0, names=['FIPS', 'mean'],usecols=[1,2])
    soil2 = pd.read_csv('/mnt/Project/data/GEE/US_soil_corn_county/zonal_corn_%s_mean_5_15.csv'%var,
                header=0, names=['FIPS', 'mean'],usecols=[1,2])
    soil3 = pd.read_csv('/mnt/Project/data/GEE/US_soil_corn_county/zonal_corn_%s_mean_15_30.csv'%var,
                header=0, names=['FIPS', 'mean'],usecols=[1,2])
    
    soil1[var] = soil1['mean']/6 +   soil2['mean']/3 + soil3['mean']/2
    
    soil1.dropna(subset=['FIPS'], inplace=True)

    soil1['FIPS'] = soil1['FIPS'].map(lambda x:"%05d"%(x))
    
    return soil1[['FIPS',var]]

def load_prism_data(end_year):
    climate1 = pd.read_csv('../data/prism_climate_monthly_1981_2017.csv',dtype={'FIPS':str})
    return climate1
    
def combine_and_save_data(crop_type='corn'):
    end_year = get_end_year(crop_type)
    nass = pd.read_csv('../data/nass_yield_area_1981_%d_%s.csv'%(end_year,crop_type),dtype={'FIPS':str})
    climate = load_prism_data(end_year)
     
    # Include soil for corn 
    if crop_type == 'corn':
        om = get_soil_data('om')
        awc = get_soil_data('awc')
    
        df_final=nass.merge(climate,on=['year','FIPS'],how='outer')\
                      .merge(evi,on=['year','FIPS'],how='left')\
                      .merge(lst,on=['year','FIPS'],how='left')\
                      .merge(om,on=['FIPS'],how='left')\
                      .merge(awc,on=['FIPS'],how='left')\
                      .merge(county[['FIPS','land_area']],on=['FIPS'],how='left')
    else:
        df_final=nass.merge(climate,on=['year','FIPS'],how='left')
        
    df_final.to_csv('../data/%s_model_data_%d.csv'%(crop_type,end_year),index=False)
    print('Crop model data csv for %s until %d file saved'%(crop_type,end_year))

# Extract monthly climate for each state, depending on their growing season
def extract_climate_state(df_all, df_st_month, st_name, data='prism'):
    c = df_all['State'] == st_name
    M_range = range(int(df_st_month.loc[st_name, 'plant_month']), 
             int(df_st_month.loc[st_name, 'harvest_month']+2)) # +2 to have 6 months
    v1 = ['tmax' + str(m) for m in M_range] 
    v2 = ['tmin' + str(m) for m in M_range] 
    v3 = ['vpdmax' + str(m) for m in M_range] 
    v4 = ['precip' + str(m) for m in M_range] 
    if data=='prism':
        v0 = ['year','FIPS','County','State','yield']
    else:
        v0 = ['year','location','State']
    return df_all.loc[c,v0+v1+v2+v3+v4].values

def save_update_climate_yield_data(crop='potatoes'):
    fname={'potatoes':'potatoes_model_data_2016',
           'tomatoes':'tomatoes_model_data_2017'}
    df = pd.read_csv('../data/%s.csv'%fname[crop], dtype={'FIPS':str}) 
    df_st_month = pd.read_csv('../data/%s_plant_harvest_dates.csv'%crop, index_col=0) #.set_index('State')

    v1 = ['tmax_p' + str(m) for m in range(1,7)] 
    v2 = ['tmin_p' + str(m) for m in range(1,7)] 
    v3 = ['vpdmax_p' + str(m) for m in range(1,7)] 
    v4 = ['precip_p' + str(m) for m in range(1,7)] 
    v0 = ['year','FIPS','County','State','yield']
    d_model = pd.DataFrame(np.zeros([df.shape[0],5+24]), columns=v0+v1+v2+v3+v4)

    frame = [extract_climate_state(df, df_st_month, s) for s in df_st_month.index.values]
    a3=np.concatenate(frame, axis=0)
    d_model.iloc[:,:] = a3

    # Correct the Oregon high value error for potatoes
    if crop == 'potatoes':
        c = d_model['yield']>2000
        d_model.loc[c,'yield'] = d_model.loc[c,'yield'] /10

    d_model.to_csv('../data/%s_model_data_update.csv'%crop,index=False)
    print('updated yield data %s saved'%crop)

if __name__ == "__main__":
#    combine_and_save_data(crop_type='sweetcorn')
#    combine_and_save_data(crop_type='potatoes')
#    combine_and_save_data(crop_type='tomatoes')
    save_update_climate_yield_data(crop='tomatoes')
