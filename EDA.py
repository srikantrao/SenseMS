import matplotlib.pyplot as plt
import numpy as np
import utils.readinutils as rut
import utils.plotutils as plu12
import utils.eyetrace as eyetrace
import utils.statutils as stu
import utils.data_formatutils as dfu
import streamlit as st


tracedir = '/data/envision_working_traces/'
patientfile = './data/patient_stats.csv'

trials = rut.readin_traces(tracedir, patientfile)
st.write('Number of trials before filtering on filename', str(len(trials)))
trials = [trial for trial in trials if trial.sub_ms]
st.write('Number of trials after filtering on filename', str(len(trials)))

#sort into patient and normal trial sets
control_trials = [trial for trial in usable_trials if trial.sub_ms == '0']
patient_trials = [trial for trial in usable_trials if trial.sub_ms == '1']

st.write('Number of patient trials:', str(len(patient_trials)))
st.write('Number of control trials:', str(len(control_trials)))

subjectids = []
for trial in trials:
    sid = trial.subjid
    subjectids.append(sid)
subjectids = np.array(subjectids)

patids = []
for trial in patient_trials:
    sid = trial.subjid
    patids.append(sid)
patids = np.array(patids)

conids = []
for trial in control_trials:
    sid = trial.subjid
    conids.append(sid)
conids = np.array(conids)

st.write('Number of unique subjects:', str(len(set(subjectids))))
st.write('Number of unique patients:', str(len(set(patids)))
st.write('Number of unique controls:', str(len(set(conids)))

st.write("CONTROL TRIALS:")
plu12.plot16traces(control_trials, plot_interp=False, eqax=False)
st.pyplot()
st.write("PATIENT TRIALS:")
plu12.plot16traces(patient_trials, plot_interp=False, eqax=False)
