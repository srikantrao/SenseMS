import utils.eyetrace as eyetrace
import math


class DatasetAtFrequency:
    """Takes in frequency, int, and creates dataset at freq.
       Expects full_dataset to be in form [patient_trials, control_trials]."""
    def __init__(self, freq, full_dataset):
        self.full_speed = 480
        self.freq = freq
        self.num_to_skip = self.full_speed/self.freq
        self.dataset = full_dataset
        self.patient_trials, self.control_trials = self.dataset[0], self.dataset[1]

        self.new_pat_data = self.skip_by_freq(self.patient_trials)
        self.new_cntr_data = self.skip_by_freq(self.control_trials)

    def split_data(self, trials):
        """Takes either patient or control data and splits
           it into x, y, and t components."""

        xvals, yvals, tvals = [], [], []
        for trial in trials:
            # For each trial, split the x, y, and t values
            # and put them in corresponding lists
            xvals.append(trial.x[~trial.interp_idx])
            yvals.append(trial.y[~trial.interp_idx])
            tvals.append(trial.time[~trial.interp_idx])

        return xvals, yvals, tvals

    def skip_by_freq(self, trials):
        '''Takes eyetrace data and returns eyetrace with
           less temporal data.'''
        # split up the data
        xs, ys, ts = self.split_data(trials)
        new_xs, new_ys, new_ts = [], [], []

        # skip the appropriate number of pts given Hz for each trace
        self.num_to_skip = 480/self.freq

        # only print this message once for the sake of output cleanliness
        if(trials[0].sub_ms == '1'):
            print("Building datasets...")
            print("Freq: {}, Num. of data points to skip for downsampling: {}".format(self.freq, self.num_to_skip))

        for trace in range(len(xs)):
            trace_xs, trace_ys, trace_ts = [], [], []
            loc = 0
            while loc <  len(xs[trace]):
                idx_below, idx_above = int(loc//1), int(loc//1+1)
                if(idx_above < len(xs[trace])):
                    dx = xs[trace][idx_above] - xs[trace][idx_below]
                    dy = ys[trace][idx_above] - ys[trace][idx_below]
                    dt = ts[trace][idx_above] - ts[trace][idx_below]
                    trace_xs.append(xs[trace][idx_below] + dx*(loc - idx_below))
                    trace_ys.append(ys[trace][idx_below] + dy*(loc - idx_below))
                    trace_ts.append(ts[trace][idx_below] + dt*(loc - idx_below))
                else:
                # if it is the last item in the list, append it
                    trace_xs.append(xs[trace][idx_below])
                    trace_ys.append(ys[trace][idx_below])
                    trace_ts.append(ts[trace][idx_below])
                loc += self.num_to_skip
            # append trace values to full_array outside of loop
            traces = [trace_xs, trace_ys, trace_ts]
            full_array = [new_xs, new_ys, new_ts]
            for vals in list(zip(traces, full_array)):
                vals[1].append(vals[0])
        eyetraces = self.create_eyetraces(new_xs, new_ys, new_ts, trials)
        return eyetraces

    def create_eyetraces(self, xs, ys, ts, trials):
        """Takes modified location data. Returns
           eyetraces including metadata."""
        new_eyetraces = []
        for i in range(len(xs)):
            if xs[i] != [0.0]:
                trace = eyetrace.EyeTrace(xs[i], ys[i], ts[i],
                                         None,
                                         trials[i].interp_times,
                                         trials[i].blink_times,
                                         trials[i].blink_num,
                                         trials[i].ppd,
                                         trials[i].fname,                                      trials[i].subinfo)
                new_eyetraces.append(trace)

        return new_eyetraces

    def get_new_data(self):
        return [self.new_pat_data, self.new_cntr_data]
