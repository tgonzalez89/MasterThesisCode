class Histogram:
    def __init__(self, data, num_bins, lo_cap=None, hi_cap=None, include_outside_of_range=False):
        self.data = data
        self.num_bins = num_bins
        self.min = lo_cap if lo_cap is not None else min(self.data)
        self.max = hi_cap if hi_cap is not None else max(self.data)
        self.ranges = [{'min': 0, 'max': 0} for i in range(self.num_bins)]
        self.bins = [[] for i in range(self.num_bins)]
        self.leftovers = {'lo': [], 'hi': []}
        self.compute(include_outside_of_range)

    def compute(self, include_outside_of_range=False):
        self.compute_ranges()
        self.compute_bins(include_outside_of_range)

    def compute_ranges(self):
        step = (self.max - self.min) / self.num_bins
        for i in range(len(self.ranges)):
            self.ranges[i]['min'] = i     * step + self.min
            self.ranges[i]['max'] = (i+1) * step + self.min
        self.ranges[-1]['max'] = float(self.max)

    def compute_bins(self, include_outside_of_range=False):
        for d in self.data:
            for i in range(len(self.ranges)):
                if d >= self.ranges[i]['min'] and d < self.ranges[i]['max']:
                    self.bins[i].append(d)
                    break
                elif i == 0 and d < self.ranges[i]['max']:
                    if include_outside_of_range:
                        self.bins[i].append(d)
                    else:
                        self.leftovers['lo'].append(d)
                    break
                elif i == len(self.ranges) - 1 and d >= self.ranges[i]['min']:
                    if include_outside_of_range or d == self.ranges[i]['max']:
                        self.bins[i].append(d)
                    else:
                        self.leftovers['hi'].append(d)
                    break

    def show(self, precision=10, max_bar_len=10):
        max_len = max([len(b) for b in self.bins])
        max_count_len = max([len(str(len(b))) for b in self.bins])
        max_min_len = max([len(str(round(r['min'], precision))) for r in self.ranges])
        for i in range(len(self.bins)):
            bar_len = int(max_bar_len*len(self.bins[i])/max_len)
            lo = round(self.ranges[i]['min'], precision)
            hi = round(self.ranges[i]['max'], precision)
            print(f"[{'#'*bar_len}{' '*(max_bar_len - bar_len)}]"
                  f" {len(self.bins[i])}{' '*(max_count_len - len(str(len(self.bins[i]))))}"
                  f" ({lo}{' '*(max_min_len - len(str(lo)))} - {hi})")
