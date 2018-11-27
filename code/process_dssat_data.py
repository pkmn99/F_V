import pandas as pd
import numpy as np
# This file contains functions to process the dssat climate data for F_V model purpose

# Get the basic infomration of the 32 counties, name, location, plant/harves dates
def get_county_information(crop='potatoes'):
    # Load geographical information and location
    d_link = pd.read_csv('../data/latlon_county_link.csv')

    # Load plant/harvest date
    d_gs = pd.read_csv('../data/%s_protocol.csv'%crop)

    # function to convery doy to month
    month_converter = lambda x: pd.datetime.strptime(str(x), '%j').month
    day_converter = lambda x: pd.datetime.strptime(str(x), '%j').day

    d_gs['plant_month'] = d_gs['Baseline'].apply(month_converter)
    d_gs['plant_day'] = d_gs['Baseline'].apply(day_converter)

    d_gs['harvest_month'] = (d_gs['Baseline']+d_gs['Season Length']).apply(month_converter)
    d_gs['harvest_day'] = (d_gs['Baseline']+d_gs['Season Length']).apply(day_converter)

    d_gs['gs_month'] = d_gs['harvest_month'] - d_gs['plant_month'] + 1

    return d_link.merge(d_gs,on='County').sort_values(by='No.').set_index('County')

# Calculate monthly climate and add VPD for each location,
# Input f is the location file name 
def get_monthly_climate(f,period='historical'):
    # Parse date format
    dateparse = lambda d: pd.datetime.strptime(d, '%y%j')

    if period == 'historical':
        file_string = period
    else:
        file_string = 'future/RCP8.5/%s'%period 

    # Use fixed width length function
    d = pd.read_fwf('../data/Weather/%s/%s'%(file_string,f), widths=[5,6,6,6,6,6,6],skiprows=3,skipfooter=1,
                    index_col=[0],parse_dates=['@DATE'], date_parser=dateparse)

    # Calculate VPD
    d['vpdmax'] = 6.1078 * (np.exp(d['TMAX'] *17.269/(237.3+d['TMAX'])) - np.exp(d['TDEW'] *17.269/(237.3+d['TDEW'])))
    d['vpdmin'] = 6.1078 * (np.exp(d['TMIN'] *17.269/(237.3+d['TMIN'])) - np.exp(d['TDEW'] *17.269/(237.3+d['TDEW'])))

    # Get monthly mean values
    d_mon = d.resample('1M').mean().copy()

    # Use sum for rain
    d_mon.loc[:,'RAIN'] = d.resample('1M').sum()['RAIN']

    # Assigin location 
    d_mon['location'] = f
    return d_mon

# save the lat and lon of the 32 counties
def save_lat_lon_counties():
    import glob
    files = glob.glob("../data/Weather/historical/*.WTH")
    # Extract the file name
    get_loc = lambda fs: [f.split("/")[-1] for f in fs]
    points = get_loc(files)

    # Extract latitude and longitide
    lat = [float(l[0:5]) for l in points]
    lon = [-float(l[6:12]) for l in points]
    d = {'name': points,
        'latitude' : lat,
         'longitude' : lon}
    df_lat_lon = pd.DataFrame(d)
    df_lat_lon.to_csv('../data/Weather/lat_lon_counties.csv',index=False)
    print('lat lon file saved')

# Read data from all locations and save to a single file
def save_monthly_climate(period='historical'):
    import glob
    if period == 'historical':
        file_string = period
    else:
        file_string = 'future/RCP8.5/%s'%period 
    files = glob.glob("../data/Weather/%s/*.WTH"%file_string)

    # Extract the file name 
    get_loc = lambda fs: [f.split("/")[-1] for f in fs]
    points = get_loc(files)
    
    frame = [get_monthly_climate(p, period=period) for p in points]
    d_final = pd.concat(frame)

    d_final['year'] = d_final.index.year
    d_final['month'] = d_final.index.month
    d_final.to_csv('../data/Weather/monthly_climate_%s_raw.csv'%period)
    print('Monthly data of %s file saved'%period)

"""
Convert the data to crop model format
"""
def convert_to_gs_monthly(df_mon,name_input, name_output):
    test = df_mon.set_index(['year','location']).pivot(columns='month')
    return test.loc[:,name_input].rename(columns= lambda x:name_output + str(x))

# Extract monthly climate variable for each county based on its plant/harvest month
def extract_climate_county(df_all, df_county, county_name,adaptation):
    c = df_all['location'] == df_county.loc[county_name,'location']
    if adaptation:
        M_range = range(int(df_county.loc[county_name, 'plant_month']-1),
             int(df_county.loc[county_name, 'harvest_month']))
    else:
        M_range = range(int(df_county.loc[county_name, 'plant_month']),
                int(df_county.loc[county_name, 'harvest_month']+1))

    v1 = ['tmax' + str(m) for m in M_range]
    v2 = ['tmin' + str(m) for m in M_range]
    v3 = ['vpdmax' + str(m) for m in M_range]
    v4 = ['precip' + str(m) for m in M_range]
    v0 = ['year','location','State','County']
    return df_all.loc[c,v0+v1+v2+v3+v4].values

# Save the climate data to model format with 12 months
def save_monthly_climate_model_12mon(period='historical'):  
    # Read saved monthly data
    d_final = pd.read_csv('../data/Weather/monthly_climate_%s_raw.csv'%period)
    t1 = convert_to_gs_monthly(d_final,'RAIN','precip')
    t2 = convert_to_gs_monthly(d_final,'TMAX','tmax')
    t3 = convert_to_gs_monthly(d_final,'TMIN','tmin')
    t4 = convert_to_gs_monthly(d_final,'vpdmax','vpdmax')
    t5 = convert_to_gs_monthly(d_final,'vpdmin','vpdmin')
    t = t1.join(t2).join(t3).join(t4).join(t5)
    t.to_csv('../data/Weather/monthly_climate_%s_model_12mon.csv'%period)
    print('%s climate data at model format for 12 mon saved'%period)
    return t

# Save monthly climate variable during the growing season for counties with specific gs month length(4,5,6)
def save_monthly_climate_model_gs(m,crop='potatoes',period='historical',adaptation=False):
    # State specific plant and harvest month
    d_county = get_county_information(crop=crop)

    # Read 12mon model data and merge with county
    d = pd.read_csv('../data/Weather/monthly_climate_%s_model_12mon.csv'%period)

    # Make adaptation December as var0
    d.loc[:,'precip0'] = np.nan
    d.loc[:,'tmax0'] = np.nan
    d.loc[:,'tmin0'] = np.nan
    d.loc[:,'vpdmax0'] = np.nan
    d.loc[:,'vpdmin0'] = np.nan

    # correct the year issue for future, 2069-2099
    if period != 'historical':
        c = (d['year']>=1969)&(d['year']<=1999)
        d.loc[c,'year'] = d.loc[c,'year'] + 100

    # Use previous year December as month zero
    c12 = (d['year']>=d['year'].min())&(d['year']<=d['year'].max()-1)
    c0 = (d['year']>=d['year'].min()+1)&(d['year'].max()<=d['year'].max())

    d.loc[c0,'precip0'] = d.loc[c12,'precip12'].values
    d.loc[c0,'tmax0'] = d.loc[c12,'tmax12'].values
    d.loc[c0,'tmin0'] = d.loc[c12,'tmin12'].values
    d.loc[c0,'vpdmax0'] = d.loc[c12,'vpdmax12'].values
    d.loc[c0,'vpdmin0'] = d.loc[c12,'vpdmin12'].values


    # County specific plant and harvest month
    v1 = ['tmax_p' + str(m) for m in range(1,m+1)]
    v2 = ['tmin_p' + str(m) for m in range(1,m+1)]
    v3 = ['vpdmax_p' + str(m) for m in range(1,m+1)]
    v4 = ['precip_p' + str(m) for m in range(1,m+1)]
    v0 = ['year','location','State','County']

    d = d.merge(d_county.reset_index(),on='location')

    # only extract data for counties with gs month length, m
    d_m = d.loc[d['gs_month']==m,:]

    # Data frame to save the results
    d_model = pd.DataFrame(np.zeros([d_m.shape[0],4+4*m]), columns=v0+v1+v2+v3+v4)

    frame = [extract_climate_county(d_m, d_county, s,adaptation) for s in d_m['County'].unique()]
    a3=np.concatenate(frame, axis=0)
    d_model.iloc[:,:] = a3
    if adaptation:
        d_model.to_csv('../data/Weather/monthly_climate_%s_adaptation_model_gs_%s_m%d.csv'%(period,crop,m),index=False)
    else:
        d_model.to_csv('../data/Weather/monthly_climate_%s_model_gs_%s_m%d.csv'%(period,crop,m),index=False)

    print('climate data for month length %d saved'%m)

# Save growing season monthly climate variable of histroical and future for model prediction
# Final data
def save_climate_gs_model(p):
    if p == 'historical':
        f = ['historical']
    else:
        f = ['GFDL-ESM2M', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM-CHEM', 'NorESM1-M']
# only need to run once 
#    for p in f:
#        save_monthly_climate(period=p)
#        save_monthly_climate_model_12mon(period=p)
    
    # process different growing season months and with/without adaptation
    for p in f:
        for m in range(4,7): # models with 4,5,6 months
            save_monthly_climate_model_gs(m,period=p)
            save_monthly_climate_model_gs(m,period=p,adaptation=True)
            print('%s for month %d saved'%(p,m))

if __name__ == "__main__":
    save_climate_gs_model('historical')
    save_climate_gs_model('future')
