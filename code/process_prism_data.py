"""
Load daily PRSIM county data for a single year, 1981-2015
df = load_prism_county_data('tdeman', 2000, freq='1d')
freq: '1M', '1A'
"""
import numpy as np
import pandas as pd
from functools import reduce


def load_prism_county_year(variable, year, freq='1d'):
    fn_path = '../../data/PRISM/data/county_level/'
    fn = variable + '_daily_' + str(year) +'_county.csv'
    df = pd.read_csv(fn_path + fn, index_col=[0], parse_dates=[0])
#     df.set_index('Unnamed: 0', inplace=True)
    if freq!='1d':
        if variable != 'ppt': # return sum for ppt
            return df.resample(freq).mean()
        else:
            return df.resample(freq).sum()
    else:
        return df

# Load data for a year range    
def load_prism_county_year_range(variable, start_year, end_year, freq='1d'):
    df = [load_prism_county_year(variable, i, freq) for i in range(start_year,end_year+1)]
    df_all = pd.concat(df)
    return df_all.dropna(axis='columns', how='all')

"""
Load PRISM data monthly and then convert the data to crop model format
Note that this function auto drops all NaN county after the conversion
"""
def convert_to_gs_monthly(df_mon,var_name):
    # Select growing season
    df_gs = df_mon[(df_mon.index.month>0)&(df_mon.index.month<=12)]

    # make some rearrangement of the data layout 
    df_gs_1 = df_gs.stack().to_frame('value').reset_index()
    df_gs_1.rename(columns={'level_1':'FIPS'}, inplace=True)

    # Add year and month as column
    df_gs_1['year'] = df_gs_1['level_0'].apply(lambda x: x.year)
    df_gs_1['mon'] = df_gs_1['level_0'].apply(lambda x: x.month)

    # Seperate monthly lst as column by mutle-index and pivot
    df_gs_2 = df_gs_1.iloc[:,1::].set_index(['year','FIPS']).pivot(columns='mon')

    # drop multi-index of columns
    df_gs_2.columns = df_gs_2.columns.droplevel(0)

    # rename lst column
    df_gs_2 = df_gs_2.reset_index().rename(columns={1:'%s1'%var_name,
                                                    2:'%s2'%var_name,
                                                    3:'%s3'%var_name,
                                                    4:'%s4'%var_name,
                                                    5:'%s5'%var_name,
                                                    6:'%s6'%var_name,
                                                    7:'%s7'%var_name,
                                                    8:'%s8'%var_name,
                                                    9:'%s9'%var_name,
                                                    10:'%s10'%var_name,
                                                    11:'%s11'%var_name,
                                                    12:'%s12'%var_name})
    return df_gs_2

"""
To generate climate data (temperature, vpd, and vpd)
"""
def get_climate_for_crop_model(year_start=1981,year_end=2016):
    # Load monthly data
    tmax_monthly = load_prism_county_year_range('tmax', year_start, year_end, freq='1M')
    tmin_monthly = load_prism_county_year_range('tmin', year_start, year_end, freq='1M')
    vpdmin_monthly = load_prism_county_year_range('vpdmin', year_start, year_end, freq='1M')
    vpdmax_monthly = load_prism_county_year_range('vpdmax', year_start, year_end, freq='1M')
    prec_monthly = load_prism_county_year_range('ppt', year_start, year_end, freq='1M')
    
    # Growing season monthly
    precip_gs = convert_to_gs_monthly(prec_monthly,'precip')
    tmax_gs = convert_to_gs_monthly(tmax_monthly,'tmax')
    tmin_gs = convert_to_gs_monthly(tmin_monthly,'tmin')
    vpdmax_gs = convert_to_gs_monthly(vpdmax_monthly,'vpdmax')
    vpdmin_gs = convert_to_gs_monthly(vpdmin_monthly,'vpdmin')
    
#    # Get averaged temperature and vpd
#    tave_gs = tmax_gs.copy()
#    tave_gs.iloc[:,2::] = (tmax_gs.iloc[:,2::].values + tmin_gs.iloc[:,2::].values)/2
#    tave_gs.rename(columns={'tmax5':'tave5','tmax6':'tave6','tmax7':'tave7','tmax8':'tave8','tmax9':'tave9'},
#                   inplace=True)
#    
#    vpdave_gs = vpdmax_gs.copy()
#    vpdave_gs.iloc[:,2::] = (vpdmax_gs.iloc[:,2::].values + vpdmin_gs.iloc[:,2::].values)/2
#    vpdave_gs.rename(columns={'vpdmax5':'vpdave5','vpdmax6':'vpdave6',
#                              'vpdmax7':'vpdave7','vpdmax8':'vpdave8','vpdmax9':'vpdave9'},
#                     inplace=True)
#    
#    dfs = [tmax_gs, tmin_gs, tave_gs, vpdmax_gs, vpdmin_gs, vpdave_gs, precip_gs]
    dfs = [tmax_gs, tmin_gs, vpdmax_gs, vpdmin_gs, precip_gs]
    df_final = reduce(lambda left,right: pd.merge(left,right,on=['year','FIPS']), dfs)
    
    df_final.to_csv('/mnt/d/Project/F_V/data/prism_climate_monthly_%d_%d.csv'%(year_start,year_end),index=False)
    print('Climate data saved to ~/Project/F_V/data/prism_climate_monthly_%d_%d.csv'%(year_start,year_end))
    return df_final


if __name__ == '__main__':
    # save climate data for crop model
#    df = get_climate_for_crop_model(year_start=2017,year_end=2018)
    df = get_climate_for_crop_model(year_start=1981,year_end=2017)
   
