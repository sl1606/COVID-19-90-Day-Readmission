import pymongo
import pandas as pd
import csv
import os
import bson

from bson import datetime as bson_datetime
from pymongo import MongoClient
from bson.son import SON  
from collections import Counter
from collections import defaultdict
from datetime import datetime, timedelta

# Connecting database configuration
mydb = client['optum_covid19_elii_20220120']
diag_records = mydb['cov_diag_records']
vis_records = mydb['cov_vis_records']
pt_records = mydb['cov_pt_records']
lab_records = mydb['cov_lab_records']
lab_tii = mydb['cov_lab_tii']
vac = mydb['covid_vaccine_history']
first = mydb['first_covid_confirmed_date']
obs = mydb['cov_obs_records']

patients = pd.read_csv(nrd_pts)
pts = patients.to_dict('records')
group_50 = list(range(50))
group_100 = list(range(50, 100))
pt_w50 = [patient for patient in pts if patient['pt_group'] in group_50]
pt_w100 = [patient for patient in pts if patient['pt_group'] in group_100]

# example of querying the BMI for the patients
# we separate the patients into 100 groups for stable querying
# each time querying 50 groups
for patient_group in range(50):
    print(f"Processing Patient Group: {patient_group}")
    patients_in_group = [patient for patient in pt_w50 if patient['pt_group'] == patient_group]
    
    for patient in patients_in_group:
        ptid = patient['PTID']
        pt_group = patient['pt_group']
        ad_start_date = datetime.strptime(patient['AD_START_DATE'], '%m/%d/%y')
        ad_end_date = datetime.strptime(patient['AD_END_DATE'], '%m/%d/%y')
        obs_sort = list(obs.find({
            'PTID': ptid,
            'pt_group': pt_group,
            'OBS_TYPE': "BMI",
            'date': {'$gte': ad_start_date, '$lte': ad_end_date}
            }, {'OBS_RESULT': 1}).sort('date', 1).limit(10))
        if obs_sort:
            earliest_record = obs_sort[0]
            patient['BMI'] = earliest_record['OBS_RESULT']

for patient_group in range(50, 100):
    print(f"Processing Patient Group: {patient_group}")
    patients_in_group = [patient for patient in pt_w100 if patient['pt_group'] == patient_group]
    
    for patient in patients_in_group:
        ptid = patient['PTID']
        pt_group = patient['pt_group']
        ad_start_date = datetime.strptime(patient['AD_START_DATE'], '%m/%d/%y')
        ad_end_date = datetime.strptime(patient['AD_END_DATE'], '%m/%d/%y')
        obs_sort = list(obs.find({
            'PTID': ptid,
            'pt_group': pt_group,
            'OBS_TYPE': "BMI",
            'date': {'$gte': ad_start_date, '$lte': ad_end_date}
            }, {'OBS_RESULT': 1}).sort('date', 1).limit(10))
        if obs_sort:
            earliest_record = obs_sort[0]
            patient['BMI'] = earliest_record['OBS_RESULT']
