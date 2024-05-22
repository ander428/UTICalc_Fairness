import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as sl
import joblib


def UTICalc_v1(reweighted=False):
    columns = [
        'agemolt12',
        'nonblack',
        'Uncircumcised (Female or uncircumcised Male)',
        'nosourceyn',
        'maxtempanywherege39'
    ]

    UTICalc_v1 = ''

    if not reweighted:
        UTICalc_v1_clinical_coeffs = {
            'intercept': -5.907 + np.log(.061/(1-.061)) - np.log(.321/(1-.321)),
            'agemolt12': 1.156,
            'nonblack': 0.965,
            'Uncircumcised (Female or uncircumcised Male)': 2.409,
            'nosourceyn': 1.394,
            'maxtempanywherege39': 0.873
        }

        intercepts = [UTICalc_v1_clinical_coeffs['intercept']]
        coefficients = np.array([list(UTICalc_v1_clinical_coeffs.values())[1:]])

        UTICalc_v1 = LogisticRegression()
        UTICalc_v1.intercept_ = intercepts
        UTICalc_v1.coef_ = coefficients
        UTICalc_v1.classes_ = np.array([0, 1])
        UTICalc_v1.feature_names_in_ = columns

    else:
        UTICalc_v1 = joblib.load('models/UTICalc_V1_reweighted.sav')
    return UTICalc_v1, columns

def UTICalc_all(reweighed=False):
    columns = ['agemolt12', 'maxtempanywherege39', 'History of UTI',
        'Uncircumcised (Female or uncircumcised Male)', 'nosourceyn',
        'fever_duration_hrsge48', 'nonblack', 'le_trace_5cat', 'le_1_5cat',
        'le_2_5cat', 'le_3_5cat', 'le_1_3cat', 'le_3or2_3cat',
        'x_nitrite_result_0or1', 'WBC (cumm) result', 'Bacteria (gram stain)']
    if not reweighed:
        return joblib.load('models/UTICalc_all.sav'), columns
    else:
        return joblib.load('models/UTICalc_AF_reweighted.sav'), columns

def UTICalc_v3():
    columns = [
        'agemolt12',
        'History of UTI',
        'Uncircumcised (Female or uncircumcised Male)',
        'nosourceyn',
        'maxtempanywherege39',
        'fever_duration_hrsge48'
    ]

    UTICalc_v3_clinical_coeffs = {
        'intercept': -5.41511 + np.log(.061/(1-.061)) - np.log(.321/(1-.321)),
        'agemolt12': 1.29113,
        'History of UTI': 0.99667,
        'Uncircumcised (Female or uncircumcised Male)': 2.33482,
        'nosourceyn': 1.44400,
        'maxtempanywherege39': 0.73447,
        'fever_duration_hrsge48': 0.68319
    }

    intercepts = [UTICalc_v3_clinical_coeffs['intercept']]
    coefficients = np.array([list(UTICalc_v3_clinical_coeffs.values())[1:]])

    UTICalc_v3 = LogisticRegression()
    UTICalc_v3.intercept_ = intercepts
    UTICalc_v3.coef_ = coefficients
    UTICalc_v3.classes_ = np.array([0, 1])
    UTICalc_v3.feature_names_in_ = columns

    return UTICalc_v3, columns

# convert yes/no button results to binary
def YN_int(yn):
    return 1 if yn == 'Yes' else 0

# get onehot encodeing for Leukocyte esterase result
def onehot_LE(LE):
    # model was trained with onehot variables in order of:
    #   Trace, 1+, 2+, 3+, 1+ (in 3cat setting), 2 or 3 (in 3cat setting)

    LE_result = [0,0,0,0,0,0]
    if LE == 'None':
        return LE_result
    elif LE == 'Trace':
        LE_result[0] = 1
    elif LE == '1+':
        LE_result[1] = 1
        LE_result[4] = 1
    elif LE == '2+':
        LE_result[2] = 1
        LE_result[5] = 1
    elif LE == '3+':
        LE_result[3] = 1
        LE_result[5] = 1

    return LE_result

sl.title("UTICalc: UTI Risk Calculator")

model = sl.selectbox('UTICalc Version', options=['V1', 'V1 (Reweighted)', 'All Features', 'All Features Reweighted', 'V3'], index=4)
age = YN_int(sl.radio("Age < 12 months", options=['Yes', 'No']))

if 'V3' not in model:
    nonblack = YN_int(sl.radio("Patient race is black", options=['Yes', 'No']))^1 # flip bit for nonblack logic
    
if 'V1' not in model:
    history = YN_int(sl.radio("History of UTI*", options=['Yes', 'No']))

sex = YN_int(sl.radio("Uncircumcised (Female or uncircumcised Male)", options=['Yes', 'No']))
nosourceyn = YN_int(sl.radio("Other fever source**", options=['Yes', 'No']))^1 # flip bit for no fever source logic
maxtempanywherege39 = YN_int(sl.radio("Maximum temperature ≥ 39°C (i.e., 102.2°F)", options=['Yes', 'No']))
fever_duration = YN_int(sl.radio("Duration of fever ≥ 48 hrs", options=['Yes', 'No']))

if 'All' in model:
    nitrate = YN_int(sl.radio("Nitrate", options=['Yes', 'No']))
    LE = onehot_LE(sl.selectbox("Leukocyte esterase", options=['None', 'Trace', '1+', '2+', '3+']))
    WBC = sl.number_input("WBC/mm3")
    gr_stain = YN_int(sl.radio("Bacteria on Gram stain", options=['Yes', 'No']))

if sl.button("Calculate UTI Risk"):
    UTICalc = None
    columns = ['agemolt12', 'maxtempanywherege39', 'History of UTI',
        'Uncircumcised (Female or uncircumcised Male)', 'nosourceyn',
        'fever_duration_hrsge48', 'nonblack', 'le_trace_5cat', 'le_1_5cat',
        'le_2_5cat', 'le_3_5cat', 'le_1_3cat', 'le_3or2_3cat',
        'x_nitrite_result_0or1', 'WBC (cumm) result', 'Bacteria (gram stain)']
    data = pd.DataFrame([np.repeat(0, len(columns))], columns=columns)
    reweighted = 'Reweighted' in model

    if 'V1' in model:
        UTICalc, columns = UTICalc_v1(reweighted=reweighted)

        # data[columns] could be used, but columns written out for readability
        data[['agemolt12',
        'nonblack',
        'Uncircumcised (Female or uncircumcised Male)',
        'nosourceyn',
        'maxtempanywherege39']] = [age, nonblack, sex, nosourceyn, maxtempanywherege39]

    elif 'All' in model:
        # default columns is set to all
        print(LE)
        data[columns] = [age, maxtempanywherege39, history, 
                         sex, nosourceyn, 
                         fever_duration, nonblack, *LE,
                         nitrate, WBC, gr_stain]
        
        UTICalc, columns = UTICalc_all(reweighed=reweighted)

    elif 'V3' in model:        
        UTICalc, columns = UTICalc_v3()
        data[['agemolt12',
        'History of UTI',
        'Uncircumcised (Female or uncircumcised Male)',
        'nosourceyn',
        'maxtempanywherege39',
        'fever_duration_hrsge48']] = [age, history, sex, nosourceyn, maxtempanywherege39, fever_duration]

    risk_score = UTICalc.predict_proba(data[columns])[0,1]
    sl.write(f"Probability of UTI: {risk_score:.2%}")

    note = ''
    if risk_score < 0.02:
        note = "Please note: The pretest probability of UTI for your patient is relatively LOW (i.e., less than 2%). Many clinicians would not obtain a urine sample in such a patient."
    elif risk_score >= 0.02 and risk_score < 0.05:
        note = "The predicted probability of UTI for your patient is between 2% and 5%. Please refer to your patient's predicted probability and consider testing based on individual factors and shared decision making with the family. Please note that, at a population level, a cutoff of ≥2% detects ~95% of UTIs, and cutoffs of ≥3% and ≥4% detect ~90% of UTIs. "
    elif risk_score >= 0.05:
        note = "The predicted probability of UTI for your patient is ≥ 5%. Strongly consider obtaining urine sample to test for UTI."
    sl.write(note)

if 'V1' not in model:
    sl.write("*Parent reported or documented history of UTI")
sl.write("**Other fever source can include (but is not limited to):" +
         "acute otitis media, upper respiratory tract infection (i.e., any cough or congestion)," +\
         "gastroenteritis, pneumonia, meningitis, bronchiolitis, and viral syndrome.")