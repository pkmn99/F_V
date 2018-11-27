import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

# simplified predictor constructor
def predictor_constructor(varname,month1,month2,form):
    if form == 'linear':
        v = ['%s%s'%(varname,i) for i in range(month1,month2+1)]
    if form == 'poly':
        v = ['%s%s + np.power(%s%s,2)'%(varname,i,varname,i) for i in range(month1,month2+1)]
    return ' + ' + ' + '.join(v)

"""
Estimate the global yield trend
"""
def yield_trend(df, order=1):
    if order==1:
        trend_model_txt = "Q('yield') ~ year"
    if order == 2:
        trend_model_txt = "Q('yield') ~ year + np.power(year,2)"
    trend_results = smf.ols(trend_model_txt, data=df).fit()
    return trend_results

# Train the model, specify gs months 
def train_model(m,form='poly',crop='potatoes'):    
    model_txt = "Q('yield_ana') ~ " + predictor_constructor('tmax_p',1,m,form) + predictor_constructor('tmin_p',1,m,form)
#    model_txt = "Q('yield_ana') ~ " + predictor_constructor('tmax_p',1,m,form)
    
    # Load data
    d_model = pd.read_csv('../data/%s_model_data_2016_update.csv'%crop,dtype={'FIPS':str})
    
    m = yield_trend(d_model, order=2)
    d_model['yield_ana'] = d_model['yield'] - m.predict(d_model)
    model =  smf.ols(model_txt, data=d_model).fit()
    return model

# Use the trained model to predict yield (function)
def get_yield_prediction(crop='potatoes',name='historical'):
    yield_unit_func = lambda x: x*100*0.45359237/0.404686 * 0.2 /1000 # cwt/acre to dry matter t/ha, 80% moisture
    d4 = pd.read_csv('../data/Weather/monthly_climate_%s_model_gs_%s_m%d.csv'%(name,crop,4))
    d5 = pd.read_csv('../data/Weather/monthly_climate_%s_model_gs_%s_m%d.csv'%(name,crop,5))
    d6 = pd.read_csv('../data/Weather/monthly_climate_%s_model_gs_%s_m%d.csv'%(name,crop,6))

    model4 = train_model(4,crop=crop)
    model5 = train_model(5,crop=crop)
    model6 = train_model(6,crop=crop)
    
    d4.loc[:,'yield_ana_predicted'] = yield_unit_func(model4.predict(d4)) # predict and convert unit
    d5.loc[:,'yield_ana_predicted'] = yield_unit_func(model5.predict(d5)) # predict and convert unit
    d6.loc[:,'yield_ana_predicted'] = yield_unit_func(model6.predict(d6)) # predict and convert unit

    simple_col = ['year','location','yield_ana_predicted']
        
    return pd.concat([d4,d5,d6])[simple_col].sort_values(by='year').dropna().reset_index(drop=True)    
    
#   # correct the year issue for future, 2069-2099
#    if name!='historical':
#        c = (d['year']>=1969)&(d['year']<=1999)
#        d.loc[c,'year'] = d.loc[c,'year'] + 100
        
    return d[simple_col].sort_values(by='year').dropna().reset_index(drop=True)

# save the csv file for the linkage of latlon and county names
def save_latlon_county_link():
    latlon=pd.read_csv('../data/latlon_county_list_full.csv')
    llc = latlon[['name','name_2']].rename(columns={'name':'location','name_2':'County'}).copy()
    llc.to_csv('../data/latlon_county_link.csv', index=False)
    print('The link from latlon to county file is saved')
    return 

# Load the potatoes baseline yield data
def get_yield_obs():
    y_hist_obs = pd.read_csv('../data/county_namelist32.csv',header=1,index_col=[0])
    llc = pd.read_csv('../data/latlon_county_link.csv')
    return y_hist_obs[['State','County','Corrected Tuber Dry Weight (t/ha) (for calibration)']] \
          .merge(llc,on='County').rename(columns={'Corrected Tuber Dry Weight (t/ha) (for calibration)':'yield_base'})

# Calculate the co2 effect on potatoes yield
def get_co2_effect():
    co2 = pd.read_csv('../data/co2_1981-2070.csv')
    co2_func = lambda x: 83.5 + 0.275 * x # equation from literature
    yield_unit_func = lambda x: x*100*0.45359237/0.404686 * 0.2 /1000 # cwt/acre to dry matter t/ha, 80% moisture
    co2['co2_yield_effect'] = yield_unit_func(co2_func(co2['co2']) - co2_func(340))
    return co2

# Use stat model to predict historical and future yield
def get_prediction_result(crop='potatoes',name='historical'):
    y_hist_obs = get_yield_obs()
    d = get_yield_prediction(crop=crop, name=name)
    co2 = get_co2_effect()
   # if name == 'historical':
   #     y = y_hist_obs.merge(d, on='location')
   # else:    
   #     y = y_hist_obs.merge(d, on='location').merge(co2,on='year')
    y = y_hist_obs.merge(d, on='location').merge(co2,on='year')
    y.loc[:,'yield_predicted_withco2'] = y.loc[:,'yield_base'] + y.loc[:,'yield_ana_predicted'] + y.loc[:,'co2_yield_effect']
    y.loc[:,'yield_predicted'] = y.loc[:,'yield_base'] + y.loc[:,'yield_ana_predicted']
    return y       

# Batch save the prediction results 
def save_prediction_result(crop='potatoes'):
    names = ['historical','GFDL-ESM2M', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM-CHEM', 'NorESM1-M']
    for f in names:
       # Without adaptation
        y = get_prediction_result(crop=crop,name=f)
        y.sort_values(by=['State','County','year'])\
            .to_csv('../data/result/%s_yield_prediction_%s.csv'%(crop,f),index=False)

       # With adaptation
        y = get_prediction_result(crop=crop,name=f+'_adaptation')
        y.sort_values(by=['State','County','year'])\
            .to_csv('../data/result/%s_yield_prediction_%s.csv'%(crop,f+'_adaptation'),index=False)
        print('%s_yield_prediction_%s.csv saved'%(crop,f))

# Load the final prediction results
def load_prediction_result(name,crop='potatoes'):
    t = pd.read_csv('../data/result/%s_yield_prediction_%s.csv'%(crop,name))
    return t

# Save the predicition results to the template excel format (need to mannual copy)
def save_excel_format():
    # Without Adaptation 
    d1 = load_prediction_result('GFDL-ESM2M').set_index(['State','County','year'])
    d2 = load_prediction_result('HadGEM2-ES').set_index(['State','County','year'])
    d3 = load_prediction_result('IPSL-CM5A-LR').set_index(['State','County','year'])
    d4 = load_prediction_result('MIROC-ESM-CHEM').set_index(['State','County','year'])
    d5 = load_prediction_result('NorESM1-M').set_index(['State','County','year'])

    d = pd.concat([d1, d2,d3,d4,d5], axis=1, 
              keys=['GFDL-ESM2M', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM-CHEM', 'NorESM1-M'])

    d.loc[(slice(None),slice(None),slice(2021,2050)),(slice(None),'yield_predicted')].to_csv('../data/result/block_2030s_noada_noco2.csv')
    d.loc[(slice(None),slice(None),slice(2041,2070)),(slice(None),'yield_predicted')].to_csv('../data/result/block_2050s_noada_noco2.csv')

    d.loc[(slice(None),slice(None),slice(2021,2050)),(slice(None),'yield_predicted_withco2')].to_csv('../data/result/block_2030s_noada_co2.csv')
    d.loc[(slice(None),slice(None),slice(2041,2070)),(slice(None),'yield_predicted_withco2')].to_csv('../data/result/block_2050s_noada_co2.csv')
    
    # With Adaptation 
    d1a = load_prediction_result('GFDL-ESM2M_adaptation').set_index(['State','County','year'])
    d2a = load_prediction_result('HadGEM2-ES_adaptation').set_index(['State','County','year'])
    d3a = load_prediction_result('IPSL-CM5A-LR_adaptation').set_index(['State','County','year'])
    d4a = load_prediction_result('MIROC-ESM-CHEM_adaptation').set_index(['State','County','year'])
    d5a = load_prediction_result('NorESM1-M_adaptation').set_index(['State','County','year'])

    da = pd.concat([d1a, d2a,d3a,d4a,d5a], axis=1, 
              keys=['GFDL-ESM2M', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM-CHEM', 'NorESM1-M'])

    da.loc[(slice(None),slice(None),slice(2021,2050)),(slice(None),'yield_predicted_withco2')].to_csv('../data/result/block_2030s_ada_co2.csv')
    da.loc[(slice(None),slice(None),slice(2041,2070)),(slice(None),'yield_predicted_withco2')].to_csv('../data/result/block_2050s_ada_co2.csv')
    print('All excel blocks saved')        

if __name__ == "__main__":
    save_prediction_result()
#    save_excel_format()
