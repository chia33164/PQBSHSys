import numpy as np
from pitch_detection import get_notes
import noisereduce as nr
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter
import sys
import os
import pretty_midi
import matplotlib.pyplot as plt
import pandas as pd

def matching(query_notes, query_onsets, corpus_notes, corpus_onsets, q_bpm):

    scores = []

    def preprocess(note, onset, bpm):
        b_length = (60/bpm)/8
        note = np.array(note)
        onset = np.array(onset)
        interval = note[1:] - note[:-1]
        #onset = np.around(onset / b_length)
        #onset = onset * b_length
        IOI = onset[1:] - onset[:-1]
        IOI = np.around(IOI/b_length)
        index = []
        index.append(interval < -6)
        index.append(np.logical_or(interval == -5, interval == -6))
        index.append(np.logical_or(interval == -3, interval == -4))
        index.append(np.logical_or(interval == -1, interval == -2))
        index.append(interval == 0)
        index.append(np.logical_or(interval == 1, interval == 2))
        index.append(np.logical_or(interval == 3, interval == 4))
        index.append(np.logical_or(interval == 5, interval == 6))
        index.append(interval > 6)
        codes = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        for i, c in zip(index, codes):
            interval[i] = c
        np.insert(interval, 0, 0)
        np.insert(IOI, 0, 0)
        index = IOI != 0
        interval = interval[index]
        IOI = IOI[index]
        return interval, IOI


    def scoring(query_n, query_o, song_n, song_o, song_bpm, q_bpm):
        q_i, q_IOI = preprocess(query_n, query_o, q_bpm)
        s_i, s_IOI = preprocess(song_n, song_o, song_bpm)
        D = np.zeros(shape=(len(q_i), len(s_i)))
        D_from = [[(0, 0) for _ in range(len(s_i))] for _ in range(len(q_i))]
        D[0, 0] = D[1, 0] = 0
        D[1, 1] = D[0, 0]
        D[1, 0] = 1
        K = 0.05
        C = 1
        for i in range(2, len(q_i)):
            D[i, 0] = D[i - 1, 0] + 1 + K * np.abs(q_IOI[i] / q_IOI[i-1])
            D[i, 1] = D[i - 1, 0]
            #D_from[i][0] = (i - 1, 0)
            #D_from[i][1] = (i - 1, 0)
        for j in range(1, len(s_i)):
            D[0, j] = D[1, j] = 0
            #D_from[0][j] = (1, j)

        for i in range(1, len(q_i)):
            for j in range(2, len(s_i)):
                from_candidate = [(i-1, j), (i-1, j-1), (i, j-1), (i-2, j-1), (i-1, j-2)]
                candidate = [D[i-1, j] + 1 + K  * np.abs(q_IOI[i]/q_IOI[i-1]),
                            D[i-1, j-1] + C/9*np.abs(q_i[i] - s_i[j]) + K * np.abs(q_IOI[i]/q_IOI[i-1] - s_IOI[j]/s_IOI[j-1]),
                            D[i, j-1] + 1 + K * np.abs(s_IOI[j]/ s_IOI[j-1])]
                if q_i[i-1] + q_i[i] == s_i[j] and i > 2:
                    candidate.append(D[i-2, j-1] + 1 + K * np.abs((q_IOI[i-1] + q_IOI[i])/q_IOI[i-2] - s_IOI[j]/s_IOI[j-1]))
                if q_i[i] == s_i[j-1] + s_i[j]:
                    candidate.append(D[i-1, j-2] + 1 + K * np.abs(q_IOI[i]/q_IOI[i-1] - (s_IOI[j-1] + s_IOI[j])/s_IOI[j-2]))
                selected = np.argmin(candidate)
                D[i, j] = candidate[selected]
                D_from[i][j] = from_candidate[selected]

                """
                if i == 17 and j == 47:
                    print(D_from[i][j])
                    print(from_candidate, candidate, D_from[i][j], selected, from_candidate[selected])
                """

        # print(D_from[17][47])
        #exit()
        end = np.argmin(D[-1, :])


        pointer = (len(q_i)-1, end)
        path = [end]
        query_path = [len(q_i)-1]
        start = 0
        if end == 0:
            return np.min(D[-1, :]), start, end+1, path
        last_match = None
        while pointer[0] > 0:
            new_pointer = D_from[pointer[0]][pointer[1]]
            #print(pointer)
            if new_pointer[0] == pointer[0] - 1 and new_pointer[1] == pointer[1] - 1:
                path.append(new_pointer[1])
                query_path.append(new_pointer[0])
                last_match = new_pointer
            pointer = new_pointer
        start = last_match[1]
        #exit()
        path = path[::-1]
        query_path= query_path[::-1]

        return np.min(D[-1, :]), start, end+1, path, query_path

    bss = []
    bes = []
    paths = []
    query_paths = []
    for song_n, song_o, song_bpm in zip(corpus_notes, corpus_onsets, corpus_bpms):
        score, bs, be, path, query_path = scoring(query_notes, query_onsets, song_n, song_o, song_bpm, q_bpm)
        scores.append(score)
        bss.append(bs)
        bes.append(be)
        paths.append(path)
        query_paths.append(query_path)


    scores = np.array(scores)
    rank = np.argsort(scores)

    # scores = scores[rank]

    return rank, scores, bss, bes, paths, query_paths

def diff_smoothing(note):
    note = np.array(note)
    interval = note[1:] - note[:-1]

    index = []
    index.append(interval < -6)
    index.append(np.logical_or(interval == -5, interval == -6))
    index.append(np.logical_or(interval == -3, interval == -4))
    index.append(np.logical_or(interval == -1, interval == -2))
    index.append(interval == 0)
    index.append(np.logical_or(interval == 1, interval == 2))
    index.append(np.logical_or(interval == 3, interval == 4))
    index.append(np.logical_or(interval == 5, interval == 6))
    index.append(interval > 6)
    codes = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    for i, c in zip(index, codes):
        interval[i] = c
    np.insert(interval, 0, 0)
    return interval

def diff(note):
    note = np.array(note)
    interval = note[1:] - note[:-1]
    np.insert(interval, 0, 0)
    return interval

def diff_all(corpus):
    corpus_scale = []
    for i in range(len(corpus)):
        corpus_scale.append([])
        for j in range(1, len(corpus[i])):
            corpus_scale[i].append(corpus[i][j] - corpus[i][j-1])
    return corpus_scale

#scale spectral
def scalespectral(scale, max_n, min_n):
    ss = np.zeros(((max_n - min_n)+1))
    for i in range(len(scale)):
        ss[scale[i]-min_n] += 1
    return ss

# t and t+1 sequence spectral
def time_scalespectral(scale, max_n, min_n):
    t_ss = np.zeros((((max_n - min_n)+1),((max_n - min_n)+1)))
    for i in range(1,len(scale)):
        if abs(scale[i-1]-min_n) >= ((max_n - min_n)+1) or abs(scale[i]-min_n) >= ((max_n - min_n)+1):
            #print('over scale')
            continue
        else:
            t_ss[scale[i-1]-min_n][scale[i]-min_n] += 1
    return t_ss

def time_ss_scoring(a, b, idf, value):
    score = 0
    #sequence probaiblity Note frequency * midi scale hypothesis weight * scale-IDF
    for x in range(value):
        for y in range(value):
            if a[x][y] > 0:
                score += a[x][y] * b[x][y] * idf[x][y]
    return score

#scale-time series distrubution matrix
def scale_ts_dm(corpus_scale):
    #get max, min
    max_n = min_n = 0
    for i in range(len(corpus_scale)):
        if max_n < np.max(corpus_scale[i]):
            max_n = np.max(corpus_scale[i])
        if min_n > np.min(corpus_scale[i]):
            min_n = np.min(corpus_scale[i])
    #print('max: ', max_n,'min: ', min_n)
    #get scale_ts_dm_all, scale_ts_dm[i]
    ss_dm=[]
    ss_all = np.zeros((((max_n - min_n)+1),((max_n - min_n)+1)))
    for i in range(len(corpus_scale)):
        ss_dm.append(time_scalespectral(corpus_scale[i], max_n, min_n))
        ss_all += ss_dm[i]
    return max_n, min_n, ss_dm, ss_all

#scale_ts_dm_IDF
def scale_ts_dm_idf(scale_all, scale_each, value):
    scale_all_idf = np.copy(scale_all)
    #search all in each
    for x in range(value):
        for y in range(value):
            #scale exit
            if scale_all[x][y] > 0:
                count = 0
                for num in range(len(scale_each)):
                    if scale_each[num][x][y] > 0:
                        count += 1
                scale_all_idf[x][y] = np.log(48/count)
    return scale_all_idf

#key detection
def key_detection_corpus(corpus):
    corpus_key = []
    for i in range(len(corpus)):
        corpus_key.append(key_detection_note(corpus[i]))
    return corpus_key

def key_detection_note(notes):
    #accumulate note
    avg_key = (np.sum(notes)/np.count_nonzero(notes))
    #search the note near by avg_key
    index = np.argmin(np.abs(notes - avg_key))
    most_appear_note = np.argmax(np.bincount(notes))
    #store in array
    if abs(notes[index] - most_appear_note) > 3:
        note_key = notes[index]
    else:
        note_key = most_appear_note
    return note_key

#tempo from duration from onset
def tempo_detection_corpus(corpus):
    corpus_tempo = []
    for i in range(len(corpus)):
        corpus_tempo.append(tempo_detection_onset(corpus[i]))
    return corpus_tempo

def tempo_detection_onset(onsets):
    dur = []
    #get duration from diff onsets
    for i in range(1, len(onsets)):
        dur.append(onsets[i]-onsets[i-1])
    #find smallest duration as base tempo
    dur = np.array(dur)
    base_tempo = np.min(dur)
    #get tempo
    diff_tempo = dur/base_tempo
    tempo = []
    #recover to tempo
    for i in range(1, len(onsets)):
        tempo.append(diff_tempo[i-1])
    tempo.append(np.argmax(np.bincount(tempo)))
    return np.array(tempo, dtype=int)

#expand notes by tempo
def expand_notes_by_tempo_corpus(corpus_notes, corpus_tempo):
    new_corpus_notes = []
    new_corpus_tempo = []
    new_corpus_onsets = []
    for i in range(len(corpus_notes)):
        temp_notes, temp_tempo, temp_onsets = expand_notes_by_tempo(corpus_notes[i], corpus_tempo[i])
        new_corpus_notes.append(temp_notes)
        new_corpus_tempo.append(temp_tempo)
        new_corpus_onsets.append(temp_onsets)
    return new_corpus_notes, new_corpus_tempo, new_corpus_onsets

def expand_notes_by_tempo(notes, tempo):
    new_notes = []
    new_tempo = []
    new_onsets = []
    #check notes, tempo equal length
    if len(notes) == len(tempo):
        #get note for array
        for i in range(len(notes)):
            #get tempo for the note
            for j in range(tempo[i]):
                new_notes.append(notes[i])
                new_tempo.append(1)
                new_onsets.append(np.sum(new_tempo)-1)
        return np.array(new_notes), np.array(new_tempo), np.array(new_onsets)
    else:
        print('error')
        return notes, tempo, tempo

#remove zero note
def remove_zero_note(notes, onsets):
    new_notes = []
    new_onsets = []
    if len(notes) == len(onsets):
        for i in range(len(notes)):
            if notes[i] > 0:
                new_notes.append(notes[i])
                new_onsets.append(onsets[i])
        return np.array(new_notes), np.array(new_onsets)
    else:
        print('error')
        return notes, onsets

#feedback tuning note
def feedback_tuning(user_model, o_query_notes, feedback_notes):
    new_query_notes = np.copy(o_query_notes)
    #get diff
    o_diff = diff(o_query_notes)
    f_diff = diff(feedback_notes)

    #find same diff position
    point = 0
    for i in range(len(o_diff)):
        if o_diff[i] == f_diff[i]:
            point = i
            #print(point)
            break

    #search feedback and original
    if point < len(o_diff):
        for i in range(point + 1, len(o_diff)):
            #if diff
            if o_diff[i] != f_diff[i]:
                #update new_query i+1
                new_query_notes[i+1] -= (o_diff[i] - f_diff[i])
                #update o_diff
                if i + 1 < len(o_diff):
                    o_diff[i+1] += (o_diff[i] - f_diff[i])
    if point > 0:
        for i in range(point - 1, -1, -1):
            if o_diff[i] != f_diff[i]:
                #update new_query i+1
                new_query_notes[i] += (o_diff[i] - f_diff[i])
                #update o_diff
                if i - 1 >= 0:
                    o_diff[i-1] += (o_diff[i] - f_diff[i])
    #print(o_query_notes, new_query_notes)

    #update user model
    for i in range(len(o_query_notes)):
        user_model[o_query_notes[i]][new_query_notes[i]] += 1

    return user_model, new_query_notes

#initial user mode
def initial_user_model():
    user_model = np.zeros((128,128)) #user model array [original note number][offset occure times]
    for i in range(128): #total note number = 128
        user_model[i][i] += 1
    return user_model

#pick up notes by path
def pickup_notes_by_path(notes, path):
    new_notes = np.zeros((len(path)), dtype=int)
    for i in range(len(path)):
        #new_notes[i] = notes[path[i]+1]
        new_notes[i] = notes[path[i]]
    return new_notes

def getWorst():
    list_path = './results/worst30.txt'
    # list_path = 'test.txt'
    fp = open(list_path, 'r')
    user = [line.split('\'')[1] for line in fp.readlines()]
    return user

##--------------main code--------------##
datapath = "./DataSet/MIR-QBSH-corpus/midiFile"
userpath = "./DataSet/MIR-QBSH-corpus/waveFile/"
worst_list = getWorst()

corpus_notes = np.load('./DataSet/MIR-QBSH-corpus/mp_integer.npy', allow_pickle=True)
corpus_onsets = np.load('./DataSet/MIR-QBSH-corpus/mp_onset.npy', allow_pickle=True)
corpus_bpms= np.load('./DataSet/MIR-QBSH-corpus/mp_bpm.npy', allow_pickle=True)
#get tempo from onset
corpus_tempo = tempo_detection_corpus(corpus_onsets)
#expand corpus notes by tempo
#corpus_notes, corpus_tempo, corpus_onsets = expand_notes_by_tempo_corpus(corpus_notes, corpus_tempo)
exp_corpus_notes, exp_corpus_tempo, exp_corpus_onsets = expand_notes_by_tempo_corpus(corpus_notes, corpus_tempo)
#get corpus key
corpus_key = key_detection_corpus(corpus_notes)

#get corpus scale
corpus_scale = diff_all(corpus_notes)
corpus_scale = np.array(corpus_scale, dtype=object)
corpus_tempo_scale = diff_all(corpus_tempo)
corpus_tempo_scale = np.array(corpus_tempo_scale, dtype=object)
#get midi scale-time series
max_n, min_n, ss_dm , ss_all = scale_ts_dm(corpus_scale)
#get scale-time idf
ss_all_idf = scale_ts_dm_idf(ss_all, ss_dm, max_n-min_n+1)

test = 0 #for pre-test function using



##-----setting for MIR-QBSH System-----##
#scoring system
hitf = 0
hito = 0
hitt = 0
all_f = 0
step = 0
MRR_numerator = 0
MRR_denominater = 0
scoring_method = 1 #0 is matching only, 1 is matching mix TD-IDF
use_worst_list = 0
use_user_model = [0, 1, 2]
save_user_model = 0
save_result = 0

# save results in csv
users = []
mrrs_fb0 = []
mrrs_fb1 = []
mrrs_fb2 = []


#feedback system
fb = 1
fb_times_peruser = 0
fb_iteration_loop = 0
user_profile = 0
user_visable_result = 0 #That is a scene, the user can easy reach the results for feedback from a shorter list @3

#matching
#1 is simulator (samples for midi captured)
#2 is form dataset folder
#0 is disable matching
matching_way = 2
matching_folder = "./DataSet/MIR-QBSH-corpus/waveFile/year2004b/person00023"
matching_user = "year2004b/person00023"
simulater_error_injection = 1 #only available when matching_way = 1


##------------------------------------##
print("Start matching")
print('feedback =', fb)
print('sorting =', end=' ')
if scoring_method == 1:
    print('Matching TFIDF')
else:
    print('Matching Only')
if matching_way == 1:
    for um in use_user_model:
        hitf = 0
        hito = 0
        hitt = 0
        all_f = 0
        step = 0
        MRR_numerator = 0
        MRR_denominater = 0
        user_visable_result = 0

        print('simulator, error_injection =', simulater_error_injection)
        if um == 0:
            user_model = initial_user_model()
        elif um == 1:
            user_model = np.load('./results/user_models/simulator_feedback_times_1.npy')
        elif um == 2:
            user_model = np.load('./results/user_models/simulator_feedback_times_2.npy')

        fb_c = fb_times_peruser
        for index in range(len(corpus_notes)):
            query_notes = corpus_notes[index][0:8]
            query_onsets = corpus_onsets[index][0:8]
            q_bpm = corpus_bpms[index]
            tempo_a = tempo_detection_onset(query_onsets)
            exp_query_notes , exp_tempo_a, exp_query_onsets = expand_notes_by_tempo(query_notes, tempo_a)
            or_a = diff(query_notes)
            ss_a = time_scalespectral(or_a, max_n, min_n)
            key_a = key_detection_note(query_notes)

            if simulater_error_injection == 1:
                #error injection
                for i in range(len(query_notes)):
                    if query_notes[i] == 67: #a kind unstable pitch for G4
                        query_notes[i] = 69
                    # elif query_notes[i] == 60: #a kind worst pitch for c4
                    #     query_notes[i] = 57
                    elif query_notes[i] == 72: #a kind fake pitch for c5
                        query_notes[i] = 68
                    elif query_notes[i] == 76: #a kind WTF pitch for e5
                        query_notes[i] = 84

            #modified replace each query notes by highest probility notes
            if fb == 1:
                ori_query_notes = np.copy(query_notes)
                for i in range(len(query_notes)):
                    query_notes[i] = np.argmax(user_model[query_notes[i]])

            soc=[]
            for i in range(len(corpus_notes)):
                soc.append(time_ss_scoring(ss_a, (ss_dm[i]/np.sum(ss_dm[i])), ss_all_idf, max_n-min_n+1))
            soc = np.array(soc)
            soc = soc/np.max(soc)

            r, s, bss, bes, paths, query_paths = matching(query_notes, query_onsets, corpus_notes, corpus_onsets, q_bpm)
            s = np.array(s)
            s += 0.000001
            s = 1/s
            s = s/np.max(s)

            scores = []
            for i in range(len(soc)):
                if scoring_method == 1:
                    scores.append((0.3*soc[i])+(0.7*s[i]))
                else:
                    scores.append(s[i])
            rank = np.argsort(scores)

            gt = index
            rank = rank

            if save_result == 2:
                if '{}.wav'.format(index + 1) not in users:
                    users.append('{}.wav'.format(index + 1))
                if um == 0:
                    mrrs_fb0.append(1 / ((47-rank.tolist().index(gt)) + 1))
                elif um == 1:
                    mrrs_fb1.append(1 / ((47-rank.tolist().index(gt)) + 1))
                elif um == 2:
                    mrrs_fb2.append(1 / ((47-rank.tolist().index(gt)) + 1))

            if gt == rank[47]:
                hitf += 1
                hito += 1
                hitt += 1
            elif gt in rank[45:47]:
                hitf += 1
                hitt += 1
            elif gt in rank[42:47]:
                hitf += 1
            all_f += 1
            MRR_denominater += 1
            MRR_numerator += (1 / ((47-rank.tolist().index(gt)) + 1))
            step += 1
            # print(step, end='\r')

            #feedback model
            if fb == 1 and fb_c > 0 and user_visable_result != hitt:
                user_visable_result = hitt
                #print('gt_midi=',gt)
                #gt_notes= corpus_notes[gt][paths[gt][0]:paths[gt][-1]]
                #m_query_notes = query_notes[query_paths[gt][0]:query_paths[gt][-1]]
                gt_notes = pickup_notes_by_path(corpus_notes[gt], paths[gt])
                m_query_notes = pickup_notes_by_path(query_notes, query_paths[gt])
                user_model, new_query_notes = feedback_tuning(user_model, m_query_notes, gt_notes)
                fb_c -= 1
                print('get user feedback')

        # print(hito, hitt, hitf, all_f)
        # print("ACC@1: {}, ACC@3: {}, ACC@5: {}, MRR: {}".format(hito/all_f, hitt/all_f, hitf/all_f, MRR_numerator/MRR_denominater))
        print('Feedback {} MRR: {}'.format(um, MRR_numerator/MRR_denominater))
        if save_user_model == 1:
            np.save('./results/user_models/simulator_feedback_times_1', user_model)
            print('save feedback model 1')

        #Rightnow get user feedback results from click rank @ TOP in first round, and cancel feedback function when loop > 1 avoid overfit
        for loop in range(fb_iteration_loop):
            if fb == 1:
                hitf = 0
                hito = 0
                hitt = 0
                all_f = 0
                step = 0
                MRR_numerator = 0
                MRR_denominater = 0
                user_visable_result = 0
                if loop == 0:
                    fb_c = 48
                else:
                    fb_c = 0
                print('do again!! loop =', loop)
                for index in range(len(corpus_notes)):
                    query_notes = corpus_notes[index][0:8]
                    query_onsets = corpus_onsets[index][0:8]
                    q_bpm = corpus_bpms[index]
                    tempo_a = tempo_detection_onset(query_onsets)
                    exp_query_notes , exp_tempo_a, exp_query_onsets = expand_notes_by_tempo(query_notes, tempo_a)
                    or_a = diff(query_notes)
                    ss_a = time_scalespectral(or_a, max_n, min_n)
                    key_a = key_detection_note(query_notes)

                    if simulater_error_injection == 1:
                        #error injection
                        for i in range(len(query_notes)):
                            if query_notes[i] == 67: #a kind unstable pitch for G4
                                query_notes[i] = 69
                            # elif query_notes[i] == 60: #a kind worst pitch for c4
                            #     query_notes[i] = 57
                            elif query_notes[i] == 72: #a kind fake pitch for c5
                                query_notes[i] = 68
                            elif query_notes[i] == 76: #a kind WTF pitch for e5
                                query_notes[i] = 84

                    #modified replace each query notes by highest probility notes
                    if fb == 1:
                        ori_query_notes = np.copy(query_notes)
                        for i in range(len(query_notes)):
                            query_notes[i] = np.argmax(user_model[query_notes[i]])

                    soc=[]
                    for i in range(len(corpus_notes)):
                        soc.append(time_ss_scoring(ss_a, (ss_dm[i]/np.sum(ss_dm[i])), ss_all_idf, max_n-min_n+1))
                    soc = np.array(soc)
                    soc = soc/np.max(soc)

                    r, s, bss, bes, paths, query_paths = matching(query_notes, query_onsets, corpus_notes, corpus_onsets, q_bpm)
                    s = np.array(s)
                    s += 0.000001
                    s = 1/s
                    s = s/np.max(s)

                    scores = []
                    for i in range(len(soc)):
                        if scoring_method == 1:
                            scores.append((0.3*soc[i])+(0.7*s[i]))
                        else:
                            scores.append(s[i])
                    rank = np.argsort(scores)

                    gt = index
                    rank = rank

                    if gt == rank[47]:
                        hitf += 1
                        hito += 1
                        hitt += 1
                    elif gt in rank[45:47]:
                        hitf += 1
                        hitt += 1
                    elif gt in rank[42:47]:
                        hitf += 1
                    all_f += 1
                    MRR_denominater += 1
                    MRR_numerator += (1 / ((47-rank.tolist().index(gt)) + 1))
                    step += 1
                    print(step, end='\r')

                    #feedback model
                    if fb == 1 and user_visable_result != hitt: # and fb_c > 0:
                        user_visable_result = hitt
                        #print('gt_midi=',gt)
                        #gt_notes= corpus_notes[gt][paths[gt][0]:paths[gt][-1]]
                        #m_query_notes = query_notes[query_paths[gt][0]:query_paths[gt][-1]]
                        gt_notes = pickup_notes_by_path(corpus_notes[gt], paths[gt])
                        m_query_notes = pickup_notes_by_path(query_notes, query_paths[gt])
                        user_model, new_query_notes = feedback_tuning(user_model, m_query_notes, gt_notes)
                        fb_c -= 1
                        print('get user feedback')

                print(hito, hitt, hitf, all_f)
                print("ACC@1: {}, ACC@3: {}, ACC@5: {}, MRR: {}".format(hito/all_f, hitt/all_f, hitf/all_f, MRR_numerator/MRR_denominater))
                if save_user_model == 1:
                    np.save('./results/user_models/simulator_feedback_times_2', user_model)
                    print('save feedback model 2')
    if save_result == 2:
        result = {
            "file": users,
            "mrrs_fb0": mrrs_fb0,
            "mrrs_fb1": mrrs_fb1,
            "mrrs_fb2": mrrs_fb2
        }
        selected_df = pd.DataFrame(result)
        print(selected_df)
        selected_df.to_csv('./results/result_per_user/simulator.csv')


elif matching_way == 2:
    if use_worst_list == 0:
        worst_list = [matching_user]
    for user in worst_list:
        if save_result == 1:
            users.append(user)

        matching_user = userpath + user
        print('dataset user =', matching_user)
        print('query = 00032.wav')
        for um in use_user_model:
            hitf = 0
            hito = 0
            hitt = 0
            all_f = 0
            step = 0
            MRR_numerator = 0
            MRR_denominater = 0
            user_visable_result = 0

            for root, dirs, files in os.walk(matching_user):
                for file in files:
                    try:
                        if ".wav" not in file:
                            continue
                        if file != '00032.wav':
                            continue

                        ffile = os.path.join(root, file)
                        y, sr = librosa.load(ffile)
                        q_bpm = librosa.beat.tempo(y, sr)[0]
                        query_notes, query_onsets = get_notes(ffile)

                        # if len(query_notes) < 6:
                            # print('query_notes is too short')
                        #    continue

                        #check diff user
                        if user_profile != root:
                            user_profile = root
                            user_model = initial_user_model()
                            fb_c = fb_times_peruser
                            print('new user')

                        if um != 0:
                            user_model_path = './results/user_models/{}_feedback_times_{}.npy'.format('_'.join(user.split('/')), um)
                            user_model = np.load(user_model_path)

                        #modified replace each query notes by highest probility notes
                        if fb == 1:
                            ori_query_notes = np.copy(query_notes)
                            for i in range(len(query_notes)):
                                query_notes[i] = np.argmax(user_model[query_notes[i]])

                        tempo_a = tempo_detection_onset(query_onsets)
                        #query_notes , tempo_a, query_onsets = expand_notes_by_tempo(query_notes, tempo_a)
                        exp_query_notes , exp_tempo_a, exp_query_onsets = expand_notes_by_tempo(query_notes, tempo_a)
                        or_a = diff(query_notes)
                        ss_a = time_scalespectral(or_a, max_n, min_n)
                        key_a = key_detection_note(query_notes)

                        #print(key_a)
                        soc=[]
                        for i in range(len(corpus_notes)):
                            soc.append(time_ss_scoring(ss_a, (ss_dm[i]/np.sum(ss_dm[i])), ss_all_idf, max_n-min_n+1))
                        soc = np.array(soc)
                        soc = soc/np.max(soc)

                        r, s, bss, bes, paths, query_paths = matching(query_notes, query_onsets, corpus_notes, corpus_onsets, q_bpm)
                        s = np.array(s)
                        s += 0.000001
                        s = 1/s
                        s = s/np.max(s)

                        scores = []
                        for i in range(len(soc)):
                            if scoring_method == 1:
                                scores.append((0.3*soc[i])+(0.7*s[i]))
                            else:
                                scores.append(s[i])
                        rank = np.argsort(scores)

                        fff = os.path.split(file)[-1][:-4]
                        gt = int(fff)

                        rank = rank + 1
                        if gt == rank[47]:
                            hitf += 1
                            hito += 1
                            hitt += 1
                            # print('Hit:',ffile)
                        elif gt in rank[45:47]:
                            hitf += 1
                            hitt += 1
                        elif gt in rank[42:47]:
                            hitf += 1
                        all_f += 1
                        MRR_denominater += 1
                        MRR_numerator += (1 / ((47-rank.tolist().index(gt)) + 1))

                        if save_result == 2:
                            if file.split('/')[-1] not in users:
                                users.append(file.split('/')[-1])
                            if um == 0:
                                mrrs_fb0.append((1 / ((47-rank.tolist().index(gt)) + 1)))
                            elif um == 1:
                                mrrs_fb1.append((1 / ((47-rank.tolist().index(gt)) + 1)))
                            elif um == 2:
                                mrrs_fb2.append((1 / ((47-rank.tolist().index(gt)) + 1)))

                        #feedback model
                        if fb == 1 and fb_c > 0 and user_visable_result != hitt:
                            user_visable_result = hitt
                            #check path length
                            if len(paths[gt-1]) != len(query_paths[gt-1]):
                                print('error path length',len(paths[gt-1]),len(query_paths[gt-1]))
                                print(ffile)
                            #gt_notes= corpus_notes[gt-1][np.min(paths[gt-1]):np.max(paths[gt-1])]
                            #m_query_notes = ori_query_notes[np.min(query_paths[gt-1]):np.max(query_paths[gt-1])]
                            #gt_notes= corpus_notes[gt-1][paths[gt-1][0]:paths[gt-1][-1]]
                            #m_query_notes = ori_query_notes[query_paths[gt-1][0]:query_paths[gt-1][-1]]
                            gt_notes = pickup_notes_by_path(corpus_notes[gt-1], paths[gt-1])
                            m_query_notes = pickup_notes_by_path(ori_query_notes, query_paths[gt-1])
                            if len(gt_notes) != len(m_query_notes):
                                print('error note length',len(gt_notes),len(m_query_notes))
                                print(ffile)
                            else:
                                user_model, new_query_notes = feedback_tuning(user_model, m_query_notes, gt_notes)
                                fb_c -= 1
                                # print('get user feedback')

                    except Exception as e:
                        print(e)
                        continue
                    step += 1
                    print('  {}'.format(step), end='\r')
            # print(hito, hitt, hitf, all_f)
            # print("ACC@1: {}, ACC@3: {}, ACC@5: {}, MRR: {}".format(hito/all_f, hitt/all_f, hitf/all_f, MRR_numerator/MRR_denominater))
            print('Feedback {} MRR: {}'.format(um, MRR_numerator/MRR_denominater))

            if save_result == 1:
                if um == 0:
                    mrrs_fb0.append(MRR_numerator/MRR_denominater)
                elif um == 1:
                    mrrs_fb1.append(MRR_numerator/MRR_denominater)
                elif um == 2:
                    mrrs_fb2.append(MRR_numerator/MRR_denominater)


            if save_user_model == 1:
                path = '_'.join(root.split('/')[4:])
                np.save('./results/user_models/{}_feedback_times_1'.format(path), user_model)
                print('save feedback model 1')

            #Rightnow get user feedback results from click rank @ TOP in first round, and cancel feedback function when loop > 1 avoid overfit
            for loop in range(fb_iteration_loop):
                if fb == 1:
                    hitf = 0
                    hito = 0
                    hitt = 0
                    all_f = 0
                    step = 0
                    MRR_numerator = 0
                    MRR_denominater = 0
                    user_visable_result = 0
                    if loop == 0:
                        fb_c = 1
                    else:
                        fb_c = 0
                    print('do again!! loop =', loop)
                    for root, dirs, files in os.walk(matching_user):
                        for file in files:
                            try:
                                if ".wav" not in file:
                                    continue

                                ffile = os.path.join(root, file)
                                y, sr = librosa.load(ffile)
                                q_bpm = librosa.beat.tempo(y, sr)[0]
                                query_notes, query_onsets = get_notes(ffile)

                                #if len(query_notes) < 6:
                                #    print('query_notes is too short')
                                #    continue

                                #modified replace each query notes by highest probility notes
                                if fb == 1:
                                    ori_query_notes = np.copy(query_notes)
                                    for i in range(len(query_notes)):
                                        query_notes[i] = np.argmax(user_model[query_notes[i]])

                                tempo_a = tempo_detection_onset(query_onsets)
                                #query_notes , tempo_a, query_onsets = expand_notes_by_tempo(query_notes, tempo_a)
                                exp_query_notes , exp_tempo_a, exp_query_onsets = expand_notes_by_tempo(query_notes, tempo_a)
                                or_a = diff(query_notes)
                                ss_a = time_scalespectral(or_a, max_n, min_n)
                                key_a = key_detection_note(query_notes)

                                #print(key_a)
                                soc=[]
                                for i in range(len(corpus_notes)):
                                    soc.append(time_ss_scoring(ss_a, (ss_dm[i]/np.sum(ss_dm[i])), ss_all_idf, max_n-min_n+1))
                                soc = np.array(soc)
                                soc = soc/np.max(soc)

                                r, s, bss, bes, paths, query_paths = matching(query_notes, query_onsets, corpus_notes, corpus_onsets, q_bpm)
                                s = np.array(s)
                                s += 0.000001
                                s = 1/s
                                s = s/np.max(s)

                                scores = []
                                for i in range(len(soc)):
                                    if scoring_method == 1:
                                        scores.append((0.3*soc[i])+(0.7*s[i]))
                                    else:
                                        scores.append(s[i])
                                rank = np.argsort(scores)

                                fff = os.path.split(file)[-1][:-4]
                                gt = int(fff)

                                rank = rank + 1
                                if gt == rank[47]:
                                    hitf += 1
                                    hito += 1
                                    hitt += 1
                                    print('Hit:',ffile)
                                elif gt in rank[45:47]:
                                    hitf += 1
                                    hitt += 1
                                elif gt in rank[42:47]:
                                    hitf += 1
                                all_f += 1
                                MRR_denominater += 1
                                MRR_numerator += (1 / ((47-rank.tolist().index(gt)) + 1))

                                #feedback model
                                if fb == 1 and fb_c > 0 and user_visable_result != hito:
                                    user_visable_result = hito
                                    #check path length
                                    if len(paths[gt-1]) != len(query_paths[gt-1]):
                                        print('error path length',len(paths[gt-1]),len(query_paths[gt-1]))
                                        print(ffile)
                                    gt_notes = pickup_notes_by_path(corpus_notes[gt-1], paths[gt-1])
                                    m_query_notes = pickup_notes_by_path(ori_query_notes, query_paths[gt-1])
                                    if len(gt_notes) != len(m_query_notes):
                                        print('error note length',len(gt_notes),len(m_query_notes))
                                        print(ffile)
                                    else:
                                        user_model, new_query_notes = feedback_tuning(user_model, m_query_notes, gt_notes)
                                        fb_c -= 1
                                        print('get user feedback')

                            except Exception as e:
                                print(e)
                                continue
                            step += 1
                            print(step, end='\r')
                    print(hito, hitt, hitf, all_f)
                    print("ACC@1: {}, ACC@3: {}, ACC@5: {}, MRR: {}".format(hito/all_f, hitt/all_f, hitf/all_f, MRR_numerator/MRR_denominater))
                    if save_user_model == 1:
                        np.save('./user_models/{}_feedback_times_{}'.format(path, loop + 2), user_model)
                        print('save feedback model {}'.format(loop + loop + 22))
    if save_result == 1:
        result = {
            "user": users,
            "mrrs_fb0": mrrs_fb0,
            "mrrs_fb1": mrrs_fb1,
            "mrrs_fb2": mrrs_fb2
        }
        selected_df = pd.DataFrame(result)
        print(selected_df)
        selected_df.to_csv('result.csv')
    elif save_result == 2:
        print(len(users))
        print(len(mrrs_fb0))
        print(len(mrrs_fb1))
        print(len(mrrs_fb2))
        result = {
            "file": users,
            "mrrs_fb0": mrrs_fb0,
            "mrrs_fb1": mrrs_fb1,
            "mrrs_fb2": mrrs_fb2
        }
        selected_df = pd.DataFrame(result)
        print(selected_df)
        selected_df.to_csv('./results/result_per_user/{}.csv'.format('_'.join(user.split('/'))))












##Pre-testing using##
if test == 1:
    #filename = "../dataset/MIR-QBSH-corpus/waveFile/year2006b/person00008/00016.wav" #bad
    #filename = "../dataset/MIR-QBSH-corpus/waveFile/year2006b/person00008/00034.wav" #good
    #filename = "../dataset/MIR-QBSH-corpus/waveFile/year2006b/person00008/00022.wav" #good
    filename = "./Dataset/MIR-QBSH-corpus/waveFile/year2004b/person00023/00048.wav" #testing
    query_notes, query_onsets = get_notes(filename)
    #query_notes, query_onsets = remove_zero_note(query_notes, query_onsets)
    y, sr = librosa.load(filename)
    q_bpm = librosa.beat.tempo(y, sr)[0]
    tempo_a = tempo_detection_onset(query_onsets)
    #query_notes , tempo_a, query_onsets = expand_notes_by_tempo(query_notes, tempo_a)
    exp_query_notes , exp_tempo_a, exp_query_onsets = expand_notes_by_tempo(query_notes, tempo_a)
    or_a = diff(query_notes)
    ss_a = time_scalespectral(or_a, max_n, min_n)
    key_a = key_detection_note(query_notes)

    soc=[]
    for i in range(len(corpus_notes)):
        soc.append(time_ss_scoring(ss_a, (ss_dm[i]/np.sum(ss_dm[i])), ss_all_idf, max_n-min_n+1))

    r, s, bss, bes, paths, query_paths= matching(query_notes, query_onsets, corpus_notes, corpus_onsets, q_bpm)
    soc = np.array(soc)

    scores = []
    for i in range(len(soc)):
        #scores.append((0.3*soc[i])*(0.7*(1/s[i])))
        #scores.append(soc[i])
        scores.append(1/(s[i]+0.000001))
    rank = np.argsort(scores)
    print(rank)

    #feedback model
    user_model = initial_user_model()
    fff = filename[60:-4]
    gt = int(fff) - 1
    print('gt_midi=',gt)
    #gt_notes= corpus_notes[gt][paths[gt][0]:paths[gt][-1]]
    #m_query_notes = query_notes[query_paths[gt][0]:query_paths[gt][-1]]
    gt_notes = pickup_notes_by_path(corpus_notes[gt], paths[gt])
    m_query_notes = pickup_notes_by_path(query_notes, query_paths[gt])
    if len(gt_notes) != len(m_query_notes):
        print('error note length',len(gt_notes),len(m_query_notes))
    else:
        user_model, new_query_notes = feedback_tuning(user_model, m_query_notes, gt_notes)

