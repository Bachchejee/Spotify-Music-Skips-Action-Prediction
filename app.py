import streamlit as st
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.sidebar('input data')      
def input_data():
    pause_before_play = st.sidebar('Pause before play', 0, 1, 0)
    hist_user_behavior_n_seekfwd = st.sidebar('Seekforward', 0, 60, 1)
    hist_user_behavior_n_seekback = st.sidebar('Seekback', 0, 151, 0)
    hist_user_behavior_is_shuffle = st.sidebar('Shuffle', 0, 1, 0)
    Start_reason_appload = st.sidebar('Start reason - appload', 0, 1, 0)
    Start_reason_backbtn = st.sidebar('Start reason - back button', 0, 1, 0)
    Start_reason_clickrow = st.sidebar('Start reason - clickrow', 0, 1, 0)
    Start_reason_fwdbtn = st.sidebar('Start reason - forward button', 0, 1, 0)
    Start_reason_remote = st.sidebar('Start reason - remote', 0, 1, 0)
    Start_reason_trackdone = st.sidebar('Start reason - trackdone', 0, 1, 0)
    End_reason_backbtn = st.sidebar('End reason - back button', 0, 1, 0)
    End_reason_endplay = st.sidebar('End reason - end play', 0, 1, 0)
    End_reason_fwdbtn = st.sidebar('End reason - forward button', 0, 1, 0)
    End_reason_logout = st.sidebar('End reason - logout', 0, 1, 0)
    End_reason_remote = st.sidebar('End reason - remote', 0, 1, 0)
    End_reason_trackdone = st.sidebar('End reason - trackdone', 0, 1, 0)
    acousticness = st.sidebar('Acousticness', 0.00, 1.00, 0.05)
    bounciness = st.sidebar('Bounciness', 0.00, 1.00, 0.05)
    energy = st.sidebar('Energy', 0.00, 1.00, 0.05)
    instrumentalness = st.sidebar('Instrumentalness', 0.00, 1.00, 0.00)
    liveness = st.sidebar('Liveness', 0.00, 1.00, 0.05)
    loudness = st.sidebar('Loudness', -60.00, 10.00, -7.00)
    mechanism = st.sidebar('Mechanism', 0.00, 1.00, 0.05)
    organism = st.sidebar('Organism', 0.00, 1.00, 0.05)
    speechiness = st.sidebar('Speechiness', 0.00, 1.00, 0.05)
    time_signature = st.sidebar('Time signature', 0, 5, 4)
    valence = st.sidebar('Valence', 0.00, 1.00, 0.05)
    acoustic_vector_0 = st.sidebar('Acoustic vector 0', -1.20, 1.20, -0.50)
    acoustic_vector_1 = st.sidebar('Acoustic vector 1', -1.20, 1.20, 0.20)
    acoustic_vector_2 = st.sidebar('Acoustic vector 2', -1.20, 1.20, 0.20)
    acoustic_vector_3 = st.sidebar('Acoustic vector 3', -1.20, 1.20, 0.00)
    acoustic_vector_4 = st.sidebar('Acoustic vector 4', -1.20, 1.20, -0.10)
    acoustic_vector_5 = st.sidebar('Acoustic vector 5', -1.20, 1.20, -0.05)
    acoustic_vector_6 = st.sidebar('Acoustic vector 6', -1.20, 1.20, -0.20)


    data = {'Duration': duration,
            'Release year': release_year,
            'US popularity estimate': us_popularity_estimate,
            'Session complete': session_comp,
            'Context switch': context_switch,
            'No pause before play': no_pause_before_play,
            'Pause before play': pause_before_play,
            'Seekforward': hist_user_behavior_n_seekfwd,
            'Seekback': hist_user_behavior_n_seekback,
            'Shuffle': hist_user_behavior_is_shuffle,
            'Start reason - appload': Start_reason_appload,
            'Start reason - back button': Start_reason_backbtn,
            'Start reason - clickrow': Start_reason_clickrow,
            'Start reason - forward button': Start_reason_fwdbtn,
            'Start reason - remote': Start_reason_remote,
            'Start reason - trackdone': Start_reason_trackdone,
            'End reason - back button': End_reason_backbtn,
            'End reason - end play': End_reason_endplay,
            'End reason - forward button': End_reason_fwdbtn,
            'End reason - logout': End_reason_logout,
            'End reason - remote': End_reason_remote,
            'End reason - trackdone': End_reason_trackdone,
            'Acousticness': acousticness,
            'Bounciness': bounciness,
            'Energy ': energy,
            'Instrumentalness': instrumentalness,
            'Liveness': liveness,
            'Loudness': loudness,
            'Mechanism': mechanism,
            'Organism': organism,
            'Speechiness': speechiness,
            'Time signature': time_signature,
            'Valence': valence,
            'Acoustic vector 0': acoustic_vector_0,
            'Acoustic vector 1': acoustic_vector_1,
            'Acoustic vector 2': acoustic_vector_2,
            'Acoustic vector 3': acoustic_vector_3,
            'Acoustic vector 4': acoustic_vector_4,
            'Acoustic vector 5': acoustic_vector_5,}

    features = pd.DataFrame(data, index=[0])
    return features
    def loadData():
        my_new_instance = pickle.loads('pkfile.pkl', 'rb')
def classify(dt):
    if dt == 1:
        return 'Skipped'
    else:
        return 'Not-Skipped'
if __name__ == '__main__':
      storeData()
      loadData() 