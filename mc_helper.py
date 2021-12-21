import numpy as np
import pandas as pd
import itertools
import calendar

def get_states(daily_timeseries, ex_thresh, dr_thresh, dry_const=False):
    xtreme_month = daily_timeseries.groupby(daily_timeseries.index.month).quantile(q=ex_thresh).reindex(daily_timeseries.index.month)
    if dry_const:
        dry_month = daily_timeseries.groupby(daily_timeseries.index.month).quantile(q=dr_thresh).reindex(daily_timeseries.index.month)
    else:
        dry_month = dr_thresh
    states = np.where(daily_timeseries.values <= dry_month, 0, np.where(daily_timeseries.values >= xtreme_month, 2, 1))
    return pd.Series(states, index=daily_timeseries.index)

def get_frequencies(daily_timeseries, ex_thresh, dr_thresh):
    unique_states = [0, 1, 2]
    state_index = [i for i in itertools.product(unique_states, unique_states)]
    states_timeseries = get_states(daily_timeseries=daily_timeseries, ex_thresh=ex_thresh, dr_thresh=dr_thresh)
    transition = [i for i in zip(states_timeseries.values[:-1], states_timeseries.values[1:])]
    transition_df = pd.DataFrame({"tr": transition, "count": np.ones(len(transition))}, index=states_timeseries.index[:-1])
    count = transition_df.groupby([transition_df.index.month, transition_df.tr]).count().unstack(level=0)
    count = count.reindex(pd.MultiIndex.from_tuples(state_index, names=("current", "next")))
    return count, states_timeseries

def get_frequencies_multiple(year, reps, daily_timeseries, ex_thresh, dr_thresh):
    available_freqs = []
    for y_ix in range(year.shape[0]):
        yy = year[y_ix]; rep = reps[y_ix]
        daily_freq, daily_states = get_frequencies(daily_timeseries.loc[str(yy)], ex_thresh=ex_thresh, dr_thresh=dr_thresh)
        daily_freq.replace(np.nan, 0, inplace=True)
        daily_freq *= rep
        available_freqs.append(daily_freq)
    return pd.DataFrame(np.array([i.values for i in available_freqs]).sum(0), index=daily_freq.index, columns=daily_freq.columns)

def build_transition_matrix(month, transition_counts):
    i = month - 1
    raw = transition_counts.iloc[:, i].unstack().values
    M = np.where(np.isnan(raw), 0, raw)
    P = M / M.sum(axis=1, keepdims=True)
    if np.any(np.all(np.isnan(P), axis=1)):
        ix = np.where(np.all(np.isnan(P), axis=1))[0]
        P[ix, :] = 1
    P = P / np.sum(P, axis=1, keepdims=True)
    return P

def simulate_year(year, frequencies):
    # np.random.seed(year)
    current_state = np.random.randint(0, 2)
    state_list = [current_state]
    cal = calendar.Calendar()
    day = 0
    for month in range(1, 13):
        transition_mat = build_transition_matrix(transition_counts=frequencies, month=month)
        for date in cal.itermonthdays(year, month):
            if date == 0:
                pass
            else:
                new_state = np.random.choice(a=[0, 1, 2], size=1, p=transition_mat[current_state, :])
                state_list.append(new_state[0])
                current_state = new_state[0]
                day += 1
    return state_list[1:]
