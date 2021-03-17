# # external libraries to be installed
import pandas as pd
import numpy as np
# import cufflinks as cf
# from IPython.display import display # needed to pretty-print DataFrames in IPython
# import plotly.io as IO # Install orca: conda install -c plotly plotly-orca psutil requests
# from plotly.offline import iplot
#
# # local libraries
#from MS3.ms3 import *
#
# #built-in libraries
import os, re, logging # , mmap, hashlib
# from fractions import Fraction
# from collections import defaultdict
# from operator import itemgetter

################################################################################
# CONSTANTS
################################################################################

regex = r"^(\.)?((?P<key>[a-gA-G](b*|\#*)|(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\.)?((?P<pedal>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\[)?(?P<chord>(?P<numeral>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr))(?P<form>[%o+M])?(?P<figbass>(9|7|65|43|42|2|64|6))?(\((?P<changes>(\+?(b*|\#*)\d)+)\))?(/\.?(?P<relativeroot>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)))?)(?P<pedalend>\])?(?P<phraseend>\\\\)?$"
features = ['chord','key','pedal','numeral','form','figbass','changes','relativeroot','pedalend','phraseend']

MAJ_PCS = [0,2,4,5,7,9,11]
MIN_PCS = [0,2,3,5,7,8,10]
C_MAJ_TPCS = [0,2,4,-1,1,3,5]
C_MIN_TPCS = [0,2,-3,-1,1,-4,-2]
A_MIN_TPCS = [3,5,0,2,4,-1,1]
NAME_TPCS = {'C': 0,
             'D': 2,
             'E': 4,
             'F': -1,
             'G': 1,
             'A': 3,
             'B': 5}
NAME_RN={'C': 'I',
         'D': 'II',
         'E': 'III',
         'F': 'IV',
         'G': 'V',
         'A': 'VI',
         'B': 'VII'}
RN_TPCS_MAJ = {'I': 0,
               'II': 2,
               'III': 4,
               'IV': -1,
               'V': 1,
               'VI': 3,
               'VII': 5}
RN_TPCS_MIN = {'I': 0,
               'II': 2,
               'III': -3,
               'IV': -1,
               'V': 1,
               'VI': -4,
               'VII': -2}
TPC_MAJ_RN =  {0: 'IV',
               1: 'I',
               2: 'V',
               3: 'II',
               4: 'VI',
               5: 'III',
               6: 'VII'}
TPC_MIN_RN =  {0: 'VI',
               1: 'III',
               2: 'VII',
               3: 'IV',
               4: 'I',
               5: 'V',
               6: 'II'}
MAJ_RN = ['I','II','III','IV','V','VI','VII']
MIN_RN = ['i','ii','iii','iv','v','vi','vii']
TPC_INT_NUM = [4, 1, 5, 2, 6, 3, 7]
TPC_INT_QUA = {0: ['P', 'P', 'P', 'M', 'M', 'M', 'M'],
               -1:['D', 'D', 'D', 'm', 'm', 'm', 'm']}
P_IV_QUAL = {
    'A': 1,
    'P': 0,
    'D': -1
}
I_IV_QUAL = {
    'A': 1,
    'M': 0,
    'm': -1,
    'D': -2
}
SHIFTS =    [[0,0,0,0,0,0,0],
            [0,0,1,0,0,0,1],
            [0,1,1,0,0,1,1],
            [0,0,0,-1,0,0,0],
            [0,0,0,0,0,0,1],
            [0,0,1,0,0,1,1],
            [0,1,1,0,1,1,1]]
TRIADS = {'M': (4,7),
          'm': (3,7),
          'o': (3,6),
          '+': (4,8)
         }
SEVENTHS = {'M7': 11,
            'm7': 10,
            '%7': 10,
            'o7': 9
           }

idx = pd.IndexSlice
#####################################################################
# This serves as a shortcut for Multiindex slicing, so that a lengthy
# df.loc[(slice(None),slice(0,1)),:]
# becomes
# df.loc[idx[:,0:1],:]
#####################################################################





################################################################################
# GENERAL HELPERS
################################################################################




class SliceMaker(object):
    """ This class serves for passing slice notation such as :3 as function arguments.

    Example
    -------

        SL = SliceMaker()
        some_function( slice_this, SL[3:8] )

    """
    def __getitem__(self, item):
        return item

SL, SM = SliceMaker(), SliceMaker()



def isnan(num):
    return pd.isnull(num)
    # return num != num # does not work with pd.NA



################################################################################
# MS3 NOTE LIST HELPERS
################################################################################

def apply_to_pieces(f, df, *args, **kwargs):
    """ Applies a function `f` which works on single pieces only to a dataframe of
    concatenated pieces distinguished by the first MultiIndex level.
    """
    return df.groupby(level=0).apply(f, *args, **kwargs)



def bag_of_notes(df, tpc='tpc', row=None):
    """ Returns a bag of note with counts and aggregated durations,
    respectively absolute and normalized.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Note list including the columns ['tpc', 'duration']
    tpc : {'tpc', 'name', 'degree', 'pc'} or any concatenation via '+' such as 'tpc+name'
        How you want the pitches in the index to be called.
    row : str, optional
        Return result as a row with index `row`. Useful to build a DataFrame with multiple BONs.

    Returns
    -------
    :obj:`pandas.DataFrame`
        Bag of notes
    """
    tpcs = df.tpc
    occurring = np.sort(tpcs.unique())
    bag = pd.DataFrame(index=occurring, columns=['count_a', 'count_n', 'duration_a', 'duration_n'])
    GB = df.groupby('tpc')
    bag.count_a = GB.size()
    bag.count_n = bag.count_a / bag.count_a.sum()
    bag.duration_a = GB['duration'].sum().astype(float)
    bag.duration_n = (bag.duration_a / bag.duration_a.sum()).astype(float)
    if tpc != 'tpc':
        names = tpc.split('+')
        note_names = []
        for n in names:
            if n == 'tpc':
                note_names.append(occurring)
            elif n == 'name':
                note_names.append(tpc2name(occurring))
            elif n == 'degree':
                note_names.append(tpc2degree(occurring))
            elif n == 'pc':
                note_names.append(tpc2pc(occurring))
            else:
                logging.warning("Parameter tpc can only be {'tpc', 'name', 'degree', 'pc'} or a combination such as 'tpc+pc' or 'name+degree+tpc'.")
        L = len(note_names)
        if L == 0:
            note_names.append(bag.index)
            L = 1
        if L == 1:
            bag.index = note_names[0]
        else:
            bag.index = [f"{t[0]} ({', '.join(str(e) for e in t[1:])})" for t in zip(*note_names)]
    if row is not None:
        bag = bag.T
        bag.insert(0, 'id', row)
        bag.set_index('id', append=True, inplace=True)
        bag = bag.unstack(level=0)
    return bag



def cumulative_fraction(S, start_from_zero=False):
    """ Accumulate the value counts of a Series so they can be plotted.
    """
    x = pd.DataFrame(S.value_counts()).rename(columns={S.name: 'x'})
    total = x.x.sum()
    x['y'] = [x.x.iloc[:i].sum() / total for i in range(1, len(x) + 1)]
    if start_from_zero:
        return pd.DataFrame({'x': 0, 'y': 0}, index=[0]).append(x)
    return x



def get_bass(note_list, onsets=None, by='mn', resolution=None):
    """ Iterate through measures and retrieve the lowest note on every beat if any is sounding.
    For finer resolution, pass, e.g., `resolution=1/8` to detect bass notes on eights notes
    between beats. Pass lowest staff only for "bassier" result.

    Returns
    -------
    generator
        A MultiIndex tuple and a note list for every mc or mn
    """
    if onsets is not None:
        onsets = [frac(os) for os in onsets]
    if resolution is not None:
        resolution = frac(resolution)
    if by == 'mn':
        for mn, notes in iter_measures(note_list, volta=-1):
            if onsets is None:
                timesig = notes.timesig.unique()[0]
                beatsize = notes.beatsize.unique()[0]
                for beat in range(1, int(1 + frac(timesig) / beatsize)):
                    if resolution is None:
                        yield (mn, beat), get_lowest(get_slice(notes, mn=mn, beat=beat))
                    else:
                        n_steps = int(beatsize / resolution)
                        for sub in range(n_steps):
                            b = f"{beat}.{frac(sub / n_steps)}" if sub > 0 else str(beat)
                            yield (mn, b), get_lowest(get_slice(notes, mn=mn, beat=b))
            else:
                for os in onsets:
                    yield (mn, os), get_lowest(get_slice(note_list, mn=mn, onset=os))
    elif by == 'mc':
        mcs = note_list.mc.unique()
        for mc in mcs:
            notes = note_list[note_list.mc == mc]
            if onsets is None:
                timesig = notes.timesig.unique()[0]
                beatsize = notes.beatsize.unique()[0]
                for beat in range(1, int(1+frac(timesig)/beatsize)):
                    yield (mc, beat), get_lowest(get_slice(note_list, mc=mc, beat=beat))
            else:
                for os in onsets:
                    yield (mc, os), get_lowest(get_slice(note_list, mc=mc, onset=os))
    else:
        raise ValueError("by needs to be 'mc' or 'mn'")

def bass_per_beat(df, resolution=None):
    """ Uses get_bass() to return a proto bass voice while filtering out duplicates. """
    ix, rows = zip(*get_bass(df, resolution=resolution))
    bass = pd.concat(rows, keys=ix, names=['mn', 'beat'] + df.index.names)
    bass = bass.reset_index(['section', 'ix']).drop_duplicates(['section', 'ix']).droplevel(['mn', 'beat'])
    bass['segment_id'] = range(len(bass))
    return bass.set_index(['segment_id','section', 'ix'], append=True).droplevel('id').sort_values(['mc', 'onset', 'midi'])




def get_lowest(note_list):
    lowest = note_list.midi.min()
    return note_list[note_list.midi == lowest]



def get_slice(note_list, mc=None, mn=None, onset=None, beat=None, staff=None,):
    """ Returns all sounding notes at a given onset or beat.
    Function relies on the columns ['beats', 'beatsize', 'beat', 'subbeat'] which you can add using beat_info(note_list, measure_list)
    Function does not handle voltas. Regulate via `note_list`

    Parameters
    ----------
    note_list : :obj:`pd.DataFrame`
        From where to retrieve the slice
    mc : :obj:`int`
        Measure count
    mn : :obj:`int`
        Measure number
    onset : numerical
    beat : numerical or str
    staff : :obj:`int`
        Return slice from this staff only.

    Examples
    --------
        get_slice(test, mn=3, beat=3.5)         # slice at measure NUMBER 3 at half of beat 3
        get_slice(test, mc=3, beat='2.1/8')     # slice at measure COUNT 3 at 1/8th of beat 2
        get_slice(test, mn=7, onset=1/8)        # slice at measure NUMBER 7 at the second eight
    """
    if mc is None:
        assert mn is not None, "Pass either mn or mc"
        res = note_list[note_list.mn == mn].copy()
    else:
        assert mn is None, "Pass either mn or mc"
        res = note_list[note_list.mc == mc].copy()

    if staff is not None:
        res = res[res.staff == staff]

    if beat is None:
        assert onset is not None, "Pass either onset or beat"
        coocurring = res.onset == onset
        still_sounding = (res.onset < onset) & (onset < (res.onset + res.duration))
    else:
        assert onset is None, "Pass either onset or beat"
        beats = val2beat(beat)
        beat, subbeat = split_beat(beats)
        b = beat2float(beats)
        dec_beats = res.beats.apply(beat2float)
        endings = dec_beats + (res.duration / res.beatsize)
        coocurring = (res.beat == beat) & (res.subbeat == subbeat)
        still_sounding = (dec_beats < b) & (b < endings)

    return res[coocurring | still_sounding]



def get_block(note_list, start, end, cut_durations=False, staff=None, merge_ties=True):
    """ Whereas get_slice() gets sounding notes at a point, get_block() retrieves
    sounding notes within a range.
    The function adds the column `overlapping` whose values follow the same logic as `tied`:
        NaN for events that lie entirely within the block.
        -1  for events crossing `start` and ending within the block.
        1   for events crossing `end`.
        0   for events that start before and end after the block.

    Parameters
    ----------
    note_list : :obj:`pandas.DataFrame`
        Note list from which to retrieve the block.
    start, end : :obj:`tuple` of (:obj:int, numerical)
        Pass (mc, onset) tuples. `end` is exclusive
    cut_durations : :obj:`bool`, optional
        Set to True if the original note durations should be cut at the block boundaries.
    staff : :obj:`int`, optional
        Return block from this staff only.
    merge_ties : :obj:`bool`, optional
        By default, tied notes are merged so that they do not appear as two different onsets.
    """
    a_mc, a_onset = start
    b_mc, b_onset = end
    a_mc, b_mc = int(a_mc), int(b_mc)
    assert a_mc <= b_mc, f"Start MC ({a_mc}) needs to be at most end MC ({b_mc})."
    a_onset, b_onset = frac(a_onset), frac(b_onset)
    if a_mc == b_mc:
        assert a_onset <= b_onset, "Start onset needs to be at most end onset."

    res = note_list[(note_list.mc >= a_mc) & (note_list.mc <= b_mc)]

    if staff is not None:
        res = res[res.staff == staff]

    in_a = (res.mc == a_mc)
    in_b = (res.mc == b_mc)
    endpoint = res.onset + res.duration
    crossing_left = in_a & (res.onset < a_onset) & (a_onset < endpoint)
    on_onset = in_a & (res.onset == a_onset)
    crossing_right = in_b & (endpoint > b_onset)

    if a_mc == b_mc:
        in_between = in_a & (res.onset >= a_onset) & (res.onset < b_onset)
    else:
        onset_in_a = in_a & (res.onset >= a_onset)
        onset_in_b = in_b & (res.onset < b_onset)
        in_between = onset_in_a | onset_in_b
    if a_mc + 1 < b_mc:
        in_between = in_between | ((res.mc > a_mc) & (res.mc < b_mc))


    res = res[crossing_left | in_between].copy()
    res['overlapping'] = pd.Series([pd.NA]*len(res.index), index=res.index, dtype='Int64')

    # start_tie = lambda S: S.replace({np.nan:  1, -1:  0, 0: 0, 1: 1})
    # end_tie   = lambda S: S.replace({np.nan: -1, -1: -1, 0: 0, 1: 0})
    start_tie = lambda S: S.fillna(1).replace({-1:  0, 0: 0})
    #start_tie = lambda S: print(S)
    # def start_tie(S):
    #     try:
    #         res =  S.fillna(1).replace({-1:  0, 0: 0})
    #     except:
    #         print(S)
    #     return res
    end_tie   = lambda S: S.fillna(-1).replace({0: 0, 1: 0})

    if crossing_left.any():
        res.loc[crossing_left, 'overlapping'] = end_tie(res.loc[crossing_left, 'overlapping'])

    if crossing_right.any():
        res.loc[crossing_right, 'overlapping'] = start_tie(res.loc[crossing_right, 'overlapping'])

    if res.tied.notna().any():
        tied_from_left = on_onset & res.tied.isin([-1, 0])
        if tied_from_left.any():
            res.loc[tied_from_left, 'overlapping'] = end_tie(res.loc[tied_from_left, 'overlapping'])
        tied_to_right = in_b & (endpoint == b_onset) & res.tied.isin([0, 1])
        if tied_to_right.any():
            res.loc[tied_to_right, 'overlapping'] = start_tie(res.loc[tied_to_right, 'overlapping'])


    if cut_durations:
        if crossing_left.any():
            res.loc[crossing_left, 'duration'] = res.loc[crossing_left, 'duration'] - a_onset + res.loc[crossing_left, 'onset']
            res.loc[crossing_left, ['mc', 'onset']] = [a_mc, a_onset]
        if crossing_right.any():
            res.loc[crossing_right, 'duration'] = b_onset - res.loc[crossing_right, 'onset']

    if merge_ties & res.tied.any():
        merged, changes = merge_tied_notes(res, return_changed=True)
        if len(changes) > 0:
            new = [ix for ix, index_list in changes.items() if len(index_list) > 0 and na_to_dummy(res.at[index_list[-1], 'overlapping'], None) in [0,1] ]
            tie_over = merged.index.isin(new)
            merged.loc[tie_over, 'overlapping'] = start_tie(merged.loc[tie_over, 'overlapping'])
            res = merged

    return res



def get_lists(dir, index_col=[0,1], index_names=None):
    """Aggregate all TSV files within `dir` and return a multiindexed dataframe.
    Function uses read_dump() to load TSVs.

    Parameters
    ----------
    dir : str
        Folder containing TSV files. Not recursive.
    index_col : int or list, optional
        Which columns of the TSVs to use as index.
    index_names : list, optional
        If None, the multiindex will have the names of the index columns and 'id' for the new first level.
    """
    files = sorted(os.listdir(dir))
    lists = [read_dump(os.path.join(dir, file), index_col=index_col) for file in files if file.endswith('.tsv')]
    ids = [int(file[:-4]) for file in files if file.endswith('.tsv')]
    if index_names is None:
        names = ['id'] + lists[0].index.names
    else:
        names = index_names
    return pd.concat(lists, keys=ids, names = names)



def get_onset_distance(a, b, lengths=None):
    """ Get distance in fractions of whole notes.
    Parameters
    ----------
    a, b : :obj:`tuple`
        (mc, onset) `(int, Fraction)`
    timesig : :obj:str or number, optional
        Only needed if onsets are in different measures
    lengths : :obj:`dict` or :obj:`pandas.Series`
        For every measure count, the actual length (`act_dur`) in quarter beats.
        Single level index!
    """
    mc_a, os_a = a
    mc_b, os_b = b
    mc_a = int(mc_a)
    mc_b = int(mc_b)
    os_a = frac(os_a)
    os_b = frac(os_b)
    if mc_a == mc_b:
        return abs(os_b - os_a)
    else:
        assert lengths is not None, "If the onsets are in different measures, you need to pass the measure lengths."
        if mc_b < mc_a:
            mc_a, mc_b = mc_b, mc_a
            swapped = True
        else:
            swapped = False
        l1 = lengths[mc_a] - os_a
        inter_mc = sum(lengths[mc] for mc in range(mc_a + 1, mc_b))
        val = l1 + inter_mc + os_b
        res = None
        try:
            res = -frac(val) if swapped else frac(val)
            return res
        except:
            logging.error(f"""Not possible to turn {val} into fraction when trying to calculate the distance between
MC {mc_a} OS {os_a} and MC {mc_b} OS {os_b}. The result is the sum {l1} + {inter_mc} + {os_b}.""")



def iter_measures(note_list, volta=None, staff=None):
    """ Iterate through measures' note lists by measure number.

    Parameters
    ----------
    note_list : :obj:`pandas.DataFrame()`
        Note list which you want to iterate through.
    volta : :obj:`int`, optional
        Pass -1 to only see the last volta, 1 to only see the first, etc.
        Defaults to `None`: In that case, if measure number 8 has two voltas, the result
        holds index 8 with all voltas, index '8a' for volta 1 only and '8b' for
        volta 2 including other parts of this measure number (e.g. following anacrusis).
    staff : :obj:`int`, optional
        If you want to iterate through one staff only.

    Examples
    --------
        numbers, measures = zip(*iter_measures(note_list))
        pd.concat(measures, keys=numbers, names=['mn'] + note_list.index.names)

    """
    mns = [int(mn) for mn in note_list.mn.unique()]
    if note_list.volta.notna().any() and volta is None:
        voltas = note_list[note_list.volta.notna()]
        to_repeat = []
        for mn in voltas.mn.unique():
            to_repeat.extend([mn] * len(voltas.volta[voltas.mn == mn].unique()))
        mns.extend(disambiguate_repeats(to_repeat))
        mns = sorted(mns, key=lambda k: k if type(k)==int else int(k[:-1]))
    if volta is None:
        volt = None
    if staff is not None:
        note_list = note_list[note_list.staff == staff]

    for number in mns:
        if number.__class__ == str:
            mn = int(number[:-1])
            volt = ord(number[-1])-96 # 'a' -> 1, 'b' -> 2
        else:
            mn = number
        sl = note_list[note_list.mn == mn]
        if sl.volta.notna().any():
            last_volta = sl.volta.max()
            if volta is not None:
                volt = volta_parameter(volta, last_volta)
                sl = sl[(sl.volta == volt) | sl.volta.isna()]
            elif volt is not None:
                if volt == last_volta:
                    sl = sl[(sl.volta == volt) | sl.volta.isna()]
                else:
                    sl = sl[(sl.volta == volt)]
                volt = None
        yield number, sl



def na_to_dummy(val, dummy):
    """ If `val` is pd.NA (or other null value), return `dummy` to make it comparable.
    Otherwise, return `val` as is.
    """
    return dummy if pd.isnull(val) else val



def name2rn(nn, key=0, minor=False):
    if nn.__class__ == float and isnan(nn):
        return nn
    if nn.__class__ != str:
        return apply_function(name2rn, nn, key=0, minor=False)
    tpc = name2tpc(nn)
    return tpc2rn(tpc, key, minor)



def name2tpc(nn):
    if nn.__class__ == float and isnan(nn):
        return nn
    if nn.__class__ != str:
        return apply_function(name2tpc, nn)
    nn_step, nn_acc = split_note_name(nn)
    nn_step = nn_step.upper()
    step_tpc = NAME_TPCS[nn_step]
    return step_tpc + 7 * nn_acc.count('#') - 7 * nn_acc.count('b')


parse_lists = lambda l: [int(mc) for mc in l.strip('[]').split(', ') if mc != '']
parse_tuples = lambda t: tuple(i.strip("\',") for i in t.strip("() ").split(", ") if i != '')
parse_lists_of_str_tuples = lambda l: [tuple(t.split(',')) for t in re.findall(r'\((.+?)\)', l)]
parse_lists_of_int_tuples = lambda l: [tuple(int(i) for i in t.split(',')) for t in re.findall(r'\((.+?)\)', l)]


DTYPES = {'bass_step': 'string',
         'barline': 'string',
         'beat': 'Int64',
         'beats': 'string',
         'changes': 'string',
         'chord': 'string',
         'chords': 'string',
         'dont_count': 'Int64',
         'figbass': 'string',
         'form': 'string',
         'globalkey': 'string',
         'gracenote': 'string',
         'key': 'string',
         'keysig': 'Int64',
         'marker': 'string',
         'mc': 'Int64',
         'mc_next': 'Int64',
         'midi': 'Int64',
         'mn': 'Int64',
         'next_chord_id': 'Int64',
         'note_names': 'string',
         'numbering_offset': 'Int64',
         'numeral': 'string',
         'octaves': 'Int64',
         'overlapping': 'Int64',
         'pedal': 'string',
         'phraseend': 'string',
         'relativeroot': 'string',
         'repeats': 'string',
         'section': 'Int64',
         'staff': 'Int64',
         'tied': 'Int64',
         'timesig': 'string',
         'tpc': 'Int64',
         'voice': 'Int64',
         'voices': 'Int64',
         'volta': 'Int64'}

def frac_or_empty(val):
    return '' if val == '' else frac(val)

CONVERTERS = {'act_dur': frac_or_empty,
            'beatsize': frac_or_empty,
            'cn': parse_lists_of_int_tuples,
            'ncn': parse_lists_of_int_tuples,
            'chord_length':frac_or_empty,
            'chord_tones': parse_lists,
            'duration':frac_or_empty,
            'next': parse_lists,
            'nominal_duration': frac_or_empty,
            'onset':frac_or_empty,
            'onset_next':frac_or_empty,
            'scalar':frac_or_empty,
            'subbeat': frac_or_empty,}



def read_dump(file, index_col=[0,1], converters={}, dtypes={}, **kwargs):
    conv = CONVERTERS
    types = DTYPES
    types.update(dtypes)
    conv.update(converters)
    return pd.read_csv(file, sep='\t', index_col=index_col,
                                dtype=types,
                                converters=conv, **kwargs)


def read_note_list(file, index_col=[0,1], converters={}, dtypes={}):
    conv = CONVERTERS
    types = DTYPES
    types.update(dtypes)
    conv.update(converters)
    return pd.read_csv(file, sep='\t', index_col=index_col,
                                dtype=types,
                                converters=conv)


def read_measure_list(file, index_col=[0]):
    return pd.read_csv(file, sep='\t', index_col=index_col,
                                   dtype=DTYPES,
                                   converters=CONVERTERS)


def iv_quality2acc(qual, perfect=False):
    """Turns an interval quality {'P', 'M', 'm', 'A', 'D'} into a shift in semitones from the
    generic interval (perfect or major)."""
    if perfect:
        if qual in P_IV_QUAL:
            return P_IV_QUAL[qual]
        else:
            acc = qual.count('A') - qual.count('D')
            assert acc != 0, f"Problem calculating quality {qual} for perfect interval"
            return acc
    elif qual in I_IV_QUAL:
        return I_IV_QUAL[qual]
    else:
        acc = qual.count('A')
        if 'D' in qual:
            acc -= 1 + qual.count('D')
        assert acc != 0, f"Problem calculating quality {qual} for imperfect interval"
        return acc

def iv2tpc(iv):
    """Turn interval string such as 'M3' into a note in C major."""
    qual, iv = split_interval_name(iv)
    base_iv = (iv-1)%7
    acc = iv_quality2acc(qual, base_iv in [0, 3, 4])
    return C_MAJ_TPCS[base_iv] + 7 * acc


def tpc2int(tpc):
    logging.warning('Function has been renamed to tpc2iv')
    return tpc2iv(tpc)


def tpc2iv(tpc):
    """Return interval name of a tonal pitch class where
       0 = 'P1', -1 = 'P4', -2 = 'm7', 4 = 'M3' etc.
    """
    if tpc.__class__ == float and isnan(tpc):
        return tpc
    try:
        tpc = int(tpc)
    except:
        return apply_function(tpc2int, tpc)
    tpc += 1
    pos = tpc % 7
    int_num = TPC_INT_NUM[pos]
    qual_region = tpc // 7
    if qual_region in TPC_INT_QUA:
        int_qual = TPC_INT_QUA[qual_region][pos]
    elif qual_region < 0:
        int_qual = (abs(qual_region) - 1) * 'D'
    else:
        int_qual = qual_region * 'A'
    return f"{int_qual}{int_num}"


####################################################
# Update this function also in ms3.py
####################################################
def tpc2key(tpc):
    """Return the name of a key signature, e.g.
    0 = C/a, -1 = F/d, 6 = F#/d# etc."""
    try:
        tpc = int(tpc)
    except:
        return apply_function(tpc2key, tpc)

    return f"{tpc2name(tpc)}/{tpc2name(tpc+3).lower()}"



####################################################
# Update this function also in ms3.py
####################################################
def tpc2name(tpc):
    """Return name of a tonal pitch class where
       0 = C, -1 = F, -2 = Bb, 1 = G etc.
    """
    if tpc.__class__ == str:
        try:
            tpc = int(tpc) + 1 # to make the lowest name F = 0 instead of -1
        except:
            logging.warning(f"'{tpc}' is not a TPC.")
            return tpc
    try:
        tpc += 1 # to make the lowest name F = 0 instead of -1
    except:
        return apply_function(tpc2name, tpc)

    if tpc < 0:
        acc = abs(tpc // 7) * 'b'
    else:
        acc = tpc // 7 * '#'
    return PITCH_NAMES[tpc % 7] + acc



####################################################
# Update this function also in ms3.py
####################################################
def tpc2pc(tpc):
    """Turn a tonal pitch class into a MIDI pitch class"""
    try:
        tpc = int(tpc)
    except:
        return apply_function(tpc2pc, tpc)

    return 7 * tpc % 12








def tpc2rn(tpc, key=0, minor=False):
    """Return scale degree of a tonal pitch class where
       0 = I, -1 = IV, -2 = bVII, 1 = V etc.
    """
    try:
        tpc = int(tpc)
    except:
        return apply_function(tpc2rn, tpc, key=0, minor=False)

    tpc -= key - 1

    if tpc < 0:
        acc = abs(tpc // 7) * 'b'
    else:
        acc = tpc // 7 * '#'

    if minor:
        return acc + TPC_MIN_RN[tpc % 7]
    else:
        return acc + TPC_MAJ_RN[tpc % 7]




def transpose_name(nn, fifths=0):
    if nn.__class__ == float and isnan(nn):
        return nn
    try:
        nn.upper()
    except:
        return apply_function(transpose_name, nn, fifths)
    return tpc2name(name2tpc(nn) + fifths)


################################################################################
# CORPUS HELPERS
################################################################################



# get_files() moved to corpus_helpers.py


# get_new_files() moved to corpus_helpers.py


# check_files() moved to corpus_helpers.py



# get_annotated() moved to corpus_helpers.py



################################################################################
# LABEL HELPERS
################################################################################
def add_chord_boundaries(chord_list, measure_list, next_ids='chord_id', multiple_pieces=False):
    """Single piece!"""
    if 'id' in measure_list.index.names:
        ids = measure_list.index.get_level_values('id').unique()
        if len(ids) > 1:
            try:
                id = chord_list.index.get_level_values('id')[0]
            except:
                logging.error("If measure_list contains multiple pieces, the index of chord_list must specify the piece's ID.")
                return
            measure_list = measure_list.loc[id]
        else:
            logging.debug(f"Removing index level ID containing ID {ids[0]} from measure_list.")
            measure_list = measure_list.droplevel('id')
    if 'id' in chord_list.index.names:
        ids = chord_list.index.get_level_values('id').unique()
        if len(ids) > 1:
            logging.error("Pass chord_list only for a single piece.")
            return
        logging.debug(f"Removing index level ID containing ID {ids[0]} from chord_list.")
        chord_list = chord_list.droplevel('id')
    chord_list = chord_list.copy()
    ix = chord_list.index
    shifted = chord_list.reset_index()[['mc', 'onset', next_ids]].shift(-1).astype({'mc': 'Int64', next_ids:'Int64'})
    shifted.index = ix
    chord_list[['mc_next', 'onset_next', 'next_'+next_ids]] = shifted
    pos_cols = [chord_list.columns.get_loc('mc_next'), chord_list.columns.get_loc('onset_next')]
    chord_list.iloc[-1, pos_cols] = [measure_list.index.get_level_values('mc').max() + 1, 0]
    chord_list['chord_length'] = chord_list.apply(lambda r: get_onset_distance((r.mc, r.onset), (r.mc_next, r.onset_next), measure_list['act_dur']), axis=1)
    return chord_list



def appl(n,k,global_minor=False):
    """Returns the absolute key of an applied chord, given the local key.

    Parameters
    ----------
    n : str
        The key that the chord is applied to.
    k : str
        The local key, i.e. the tonal context where the applied chord appears.
    global_minor : bool, optional
        Has to be set to True if k is to be interpreted in a minor context.
        NB: If you use the function to transpose labels, pass True if the local
            key is in the minor.

    Example
    -------
    If the label viio6/V appears in the context of the key of bVI,
    viio6 pertains to the absolute key bIII.

    >>> appl("V","bVI")
    'bIII'

    """
    if isnan(n):
        return np.nan
    if n in ['Ger','It','Fr']:
        return n
    shift = n.count('#') - n.count('b') + k.count('#') - k.count('b')
    for char in "#b":
        n = n.replace(char,'')
        k = k.replace(char,'')
    steps = MAJ_RN if n.isupper() else MIN_RN
    i = steps.index(n)
    j = MAJ_RN.index(k.upper())
    step = steps[(i+j)%7]
    if k.islower() and i in [2,5,6]:
        shift -= 1
    if global_minor:
        j = (j-2)%7
    shift += SHIFTS[i][j]
    acc = shift * '#' if shift > 0 else -shift * 'b'
    return acc+step



def appl_backwards(n,k,global_minor=False):
    """ Returns the absolute key `n` as an applied key inside the absolute key `k`.

    Parameters
    ----------
    n : str
        The absolute key that you want to be expressed inside another absolute key.
    k : str
        The absolute key in which you want to express `n` locally
    global_minor : bool, optional
        Has to be set to True if k is to be interpreted in a minor context.

    Example
    -------
    In a minor context, the key of II would appear within the key of vii as #III.

    >>> appl_backwards("II",'vii',True)
    '#III'

    """
    shift = n.count('#') - n.count('b') - k.count('#') + k.count('b')
    for char in "#b":
        n = n.replace(char,'')
        k = k.replace(char,'')
    steps = MAJ_RN if n.isupper() else MIN_RN
    j = MAJ_RN.index(k.upper())
    i = (steps.index(n) - j) % 7
    step = steps[i]
    if k.islower() and i in [2,5,6]:
        shift += 1
    if global_minor:
        j = (j-2)%7
    shift -= SHIFTS[j][i]
    acc = shift * '#' if shift > 0 else -shift * 'b'
    return acc+step



def beat2float(beat):
    """Converts a quarter beat in the form ``2.1/3`` to a float
    """
    if isinstance(beat,float):
        return beat
    beat = str(beat)
    split = beat.split('.')
    val = float(split[0])
    if len(split) > 1:
        val += float(frac(split[1]))
    return val



def chord2tpc(chord=None, numeral=None, form=None, figbass=None, relativeroot=None, changes='', key='C', minor=None):
    """Express a chord label as tonal pitch class in a certain key.
    Convenience function for using stufen(..., alle=True)

    Parameters
    ----------
    chord : str, optional
        If you pass an entire chord label, the parameters are taken from that.
    numeral, form, figbass, relativeroot, changes : str
        Chord features.
    key : str
        Specify the root of the key. If `minor` is None, a lowercase root indicates a minor key.
    minor : bool, optional
        If specified, `key` being upper or lowercase will be ignored and it can be a TPC integer as well.
    """
    logging.warning("This is an older function that calculates scale degrees first and converts them to TPCs. It works but only for regular chord tones, i.e. 1, 3, 5 and 7. Use the newer chord2tpcs() for full functionality.")
    if chord is not None:
        try:
            chord_features = re.match(regex, chord).groupdict()
            numeral, form, figbass, relativeroot, changes = tuple(chord_features[f] for f in ('numeral', 'form', 'figbass', 'relativeroot', 'changes'))
        except:
            raise ValueError(f"{chord} is not a valid chord label.")
    if minor is None:
        try:
            minor = key.islower()
        except:
            raise ValueError(f"If parameter 'minor' is not specified, 'key' needs to be a string, not {key}")
    #print(numeral)
    try:
        steps = stufen(numeral, form, figbass, relativeroot, changes, minor, True)
    except:
        logging.error(f"{numeral}{form}{figbass}{'('+changes+')' if changes is not None else ''}{'/'+relativeroot if relativeroot is not None else ''}")
        return None
    #print(steps)
    return step2tpc(steps, root=key, minor=minor)





def chord2tpcs(chord=None, numeral=None, form=None, figbass=None, relativeroot=None, changes='', minor=False):
    """Calculates the bass step ("Bassstufe") given a roman numeral annotation.

    Parameters
    ----------
    chord : str, optional
        If you pass an entire chord label, the parameters down to `changes` are taken from that.
    numeral: str
        Roman numeral of the chord's root
    form: {None, 'M', 'o', '+' '%'}, optional
        Indicates the chord type if not a major or minor triad (for which `form`is None).
        '%' and 'M' can only occur as tetrads, not as triads.
    figbass: {None, '6', '64', '7', '65', '43', '2'}, optional
        Indicates chord's inversion. Pass None for triad root position.
    relativeroot: str, optional
        Pass a Roman scale degree if `numeral` is to be applied to a different scale
        degree of the local key, as in 'V65/V'
    changes: str, optional
        Added steps such as '+6' or suspensions such as '4' or any combination such as (9+64).
        Numbers need to be in descending order.
    minor: bool, optional
        Pass True if the the local key is not major but minor.
        This affects calculation of chords related to III, VI and VII.
    """
    if chord is None:
        numeral, form, figbass, relativeroot, changes = tuple(None if isnan(val) else val for val in (numeral, form, figbass, relativeroot, changes))
    else:
        try:
            chord_features = re.match(regex, chord).groupdict()
            numeral, form, figbass, relativeroot, changes = tuple(chord_features[f] for f in ('numeral', 'form', 'figbass', 'relativeroot', 'changes'))
        except:
            raise ValueError(f"{chord} is not a valid chord label.")

    if changes is None:
        changes = ''

    if minor is None:
        try:
            minor = key.islower()
        except:
            raise ValueError(f"If parameter 'minor' is not specified, 'key' needs to be a string, not {key}")

    if form in ['%', 'M']:
        assert figbass in ['7', '65', '43', '2'], f"{numeral}{form if form is not None else ''}{figbass if figbass is not None else ''}{'(' + changes + ')' if changes != '' else ''}: {form} requires figbass since it specifies a chord's seventh."


    if relativeroot is not None:
        rel_minor = True if relativeroot.islower() else False
        transp = rn2tpc(relativeroot, minor)
        res = chord2tpcs(numeral=numeral, form=form, figbass=figbass, relativeroot=None, changes=changes, minor=rel_minor)
        return [tpc + transp for tpc in res]

    num_acc, num_degree = split_sc_dg(numeral)

    if num_degree in ['Ger','Fr','It']:
        assert figbass == '6', f"{num_degree} needs to have figbass == '6'"
        assert form is None, f"{num_degree} cannot be combined with chord form {form}"
        assert changes == '', f" Chord changes ({changes}) are not defined for {num_degree}"
        if num_degree == 'Ger':
            return [-4, 0, -3, 6]
        elif num_degree == 'Fr':
            return [-4, 0, 2, 6]
        else:
            return [-4, 0, 6]

    # build 2-octave diatonic scale on C major/minor
    root = MAJ_RN.index(num_degree.upper())
    tpcs = 2 * C_MIN_TPCS if minor else 2 * C_MAJ_TPCS
    tpcs = tpcs[root:] + tpcs[:root]               # starting the scale from chord root
    root_alteration = num_acc.count('#') - num_acc.count('b')
    root = tpcs[0] + 7 * root_alteration
    tpcs[0] = root                                 # octave stays diatonic

    def set_iv(scale_degree, interval):
        nonlocal tpcs, root
        iv = root + interval
        i = scale_degree - 1
        tpcs[i] = iv
        tpcs[i+7] = iv



    if form == 'o':
        set_iv(3, -3)
        set_iv(5, -6)
        if figbass in ['7', '65', '43', '2']:
            set_iv(7, -9)
    elif form == '%':
        set_iv(3, -1)
        set_iv(5, 6)
        set_iv(7, -2)
    elif form == '+':
        set_iv(3, 4)
        set_iv(5, 8)
    else: # triad with or without major seven
        set_iv(5, 1)
        if num_degree.isupper():
            set_iv(3, 4)
        else:
            set_iv(3, -3)
        if form == 'M':
            set_iv(7, 5)




    # apply changes
    alts = [t for t in re.findall("((\+)?(#+|b+)?(1\d|\d))",changes)]
    added_notes = []

    for full, added, acc, sd in alts:
        added = True if added == '+' else False
        sd = int(sd) - 1
        if sd == 0 or sd > 13:
            logging.warning(f"Alteration of scale degree {sd+1} is meaningless and ignored.")
            continue
        next_octave = True if sd > 7 else False
        transp = 7 * (acc.count('#') - acc.count('b'))
        new_val = tpcs[sd] + transp
        if added:
            added_notes.append(new_val)
        elif '#' in acc:
            tpcs[sd + 1] = new_val
            if sd == 6:
                added_notes.append(new_val)
        elif sd in [1, 3, 5, 8, 10, 12]: # these are changes to scale degree 2, 4, 6 that replace the lower neighbour unless they have a #
            tpcs[sd - 1] = new_val
        else:
            if tpcs[sd] == new_val:
                logging.warning(f"The change {full} has no effect in {numeral}{form if form is not None else ''}{figbass if figbass is not None else ''}")
            tpcs[sd] = new_val
        if next_octave:
            added_notes.append(new_val)



    if figbass is None:
        res = [tpcs[i] for i in [0,2,4]]
    elif figbass == '6':
        res = [tpcs[i] for i in [2,4,0]]
    elif figbass == '64':
        res = [tpcs[i] for i in [4,0,2]]
    elif figbass == '7':
        res = [tpcs[i] for i in [0,2,4,6]]
    elif figbass == '65':
        res = [tpcs[i] for i in [2,4,6,0]]
    elif figbass == '43':
        res = [tpcs[i] for i in [4,6,0,2]]
    elif figbass == '2':
        res = [tpcs[i] for i in [6,0,2,4]]
    else:
        raise ValueError

    return sort_tpcs(res + added_notes, start=res[0])






def chord_notes(df, notes, **kwargs):
    """ Helper function for all_chord_notes()
    Parameters
    ----------
    df : :obj:`pandas.DataFrame`:
        A DataFrame with chord information, as created by a groupby('chord_id') on
        a chord list that has been treated with add_chord_boundaries(). Only the
        first row of `df` is being used.
    notes : :obj:`pandas.DataFrame`:
        Note list from which the harmony block is extracted
    """
    S = df.iloc[0]
    # try:
    notes = get_block(notes, (S.mc, S.onset), (S.mc_next, S.onset_next), cut_durations=True)
    # except:
    #     print(S)
    if 'rel_label' in S:
        notes.chords = S.rel_label
    return notes.sort_values(['mc', 'onset', 'midi'])

def all_chord_notes(chord_list, note_list, by='chord_id', multiple_pieces=False):
    """
    Parameters
    ----------
    """
    if multiple_pieces:
        id = chord_list.index.get_level_values('id')[0]
        chord_list = chord_list.droplevel('id')
        note_list = note_list.loc[id]
    return chord_list.groupby(by=by).apply(chord_notes, note_list)



def chords_by_id(cn, cl):
    """This function should be elaborated to a more generic one.
    At the moment, it slices `cn` by using the (single) index (of) `cl` on the 2nd MultiIndex level.
    In the future, add an int parameter to decide on the level and integrate the function into the convenience
    function select_by_index()."""
    if cl.__module__ in ['pandas.core.indexes.base', 'pandas.core.indexes.multi', 'pandas.core.indexes.numeric']:
        ix = cl
    else:
        ix = cl.index.get_level_values('chord_id')
    return cn.loc[idx[:,ix],]



def eval_changes(numeral,changes,minor=False):
    """Turn (chord changes) into deviations in semi-tones.

    Parameters
    ----------
    numeral : :obj:`str`
        Chord numeral.
    changes : :obj:`str`
        String contained in (brackets) following the DCML syntax.

    Returns
    -------
    :obj:`list`
        Alteration of chord third, fifth and seventh in semi-tones
    :obj:`list`
        Suspension of chord root, third, fifth and seventh in semi-tones
    """
    alts = [t for t in re.findall("((\+)?(#+|b+)?(1\d|\d))",changes) if t[0][0] != '+']
    alterations = [0,0,0]
    suspensions = [0,0,0,0]

    for full, _, acc, sd in alts:
        if len(sd) > 1:
            continue        #changes to intervals bigger than 9 cannot alter chord tones

        sd = int(sd)
        if sd in [2,4,6]:
            if '#' in acc:
                suspensions[int(sd/2)] = get_interval(numeral, sd+1, full, minor)
            else:
                suspensions[int(sd/2-1)] = get_interval(numeral,sd-1, full, minor)
        elif sd in [3,5,7]:
            alterations[int((sd-3)/2)] = get_interval(numeral, sd, full, minor)
    return alterations, suspensions



def expand_labels(df, column, grouped=True, features=['chord','key','pedal','numeral','form','figbass','changes','relativeroot','pedalend','phraseend'], steps=True):
    """ Split harmony labels complying with the DCML syntax into their features.
    `df` can have only one or two index levels.
    Pass chords for one piece at a time. For concatenated pieces, use apply_to_pieces().

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Dataframe where one column contains DCML chord labels.
    column : str
        Name of the column that holds the harmony labels.
    grouped : :obj:`bool`, optional
        Defaults to True, meaning that the first index level designates different pieces.
        Set to False if df contains only one piece and therefore a single index.
    features : :obj:`list`
        Features you want to retain

    """
    df = df.copy()
    if df[column].isna().any():
        df = df[df[column].notna()]
        logging.info("Removed NaN values.")
    if df[column].str.contains('-').any():
        df.loc[:, column] = df[column].str.rsplit('-', expand=True)[0]
    spl = df[column].str.extract(regex, expand=True).astype('string') # needs pandas 1.0?
    for f in features:
        df.loc[:, f] = spl[f]

    def fill_values(frame):
        if grouped:
            piece = frame.index.get_level_values('id')[0]
            frame = frame.droplevel('id')
        else:
            piece = None
        ####################################
        # expand global key to all harmonies
        ####################################
        global_key = frame.iloc[0].key
        if isnan(global_key):
            logging.warning(f'Global key not specified in score with ID {piece}')
            print(frame)
            raise IOError(f'Global key not specified in score with ID {piece}')
        frame.loc[:,'globalkey'] = global_key
        ####################################
        # expand local keys to all harmonies
        ####################################
        # n_k = len(frame.index)
        # localkeys = frame.key[frame.key.notna()]
        # n_l = len(localkeys)
        # try:
        global_minor = True if global_key[0].islower() else False
        # except:
        #     logging.error("Could not derive mode from global key.")
        #     print(frame)
        # localkeys[0] = 'i' if global_minor else 'I'
        # for i,(begins,localkey) in enumerate(localkeys.iteritems()):
        #     ends = localkeys.index[i+1] if i < n_l-1 else n_k
        #     frame.key.iloc[begins:ends] = localkey
        frame.iat[0, frame.columns.get_loc('key')] = 'i' if global_minor else 'I'
        frame.key.fillna(method='ffill', inplace=True)
        ##########################################
        #expand pedal notes to concerned harmonies
        ##########################################
        beginnings = frame.pedal[frame.pedal.notna()]
        n_b = len(beginnings)
        endings = frame.pedalend[frame.pedalend.notna()]
        n_e = len(endings)
        assert n_b == n_e, f"{'Error' if piece is None else piece}: {n_b} organ points started, {n_e} ended! Beginnings:\n{frame[frame.pedal.notna()]}\nEndings:\n{frame[frame.pedalend.notna()]}"
        for i,(begins,pedal) in enumerate(beginnings.iteritems()):
            section = frame.iloc[begins:endings.index[i]+1]
            section.pedal = pedal
            keys = list(section.key.unique())
            if len(keys) > 1:
                original_key = keys[0]
                for k in keys[1:]:
                    section.pedal[section.key == k] = appl_backwards(appl(pedal,original_key,global_minor),k,global_minor)
        frame = frame.drop(columns='pedalend')
        #####################
        # don't fill in chord types because 'M' stands for a major 7th
        #####################
        # frame.form[frame.numeral.isin(['I','II','III','IV','V','VI','VII'])] = 'M'
        # frame.form[frame.numeral.isin(['i','ii','iii','iv','v','vi','vii'])] = 'm'

        return frame
    if grouped:
        df = df.groupby('id',sort=False).apply(fill_values)
    else:
        df = fill_values(df)
    if steps:
        df = df.apply(add_steps,axis=1)
    return df



def float2beat(b):
    """ 2.5 -> '2.1/2' """
    if b.__class__ == int or (b.__class__ == float and b.is_integer()):
        return str(b)
    beat = str(b)
    split = beat.split('.')
    val = int(split[0])
    sub = frac("0."+split[1])
    return f"{val}.{sub}"



def fractionize(b,hide_zeros=True):
    """ Turns a float into a string

    This is a very similar function to piece.float2beat

    Parameters
    ----------

    b: float
        The beat to transform
    hide_zeros: bool, optional
        set to False if you want beats like 1.0 instead of 1

    Examples
    --------

        >>> fractionize(1.5)
        '1.1/2'
        >>> fractionize(1,False)
        '1.0'
    """
    try:
        b = float(b)
    except:
        return b
    F = Fraction(b%1)
    if hide_zeros:
        return f"{int(b//1)}.{F}" if F != 0 else f"{int(b//1)}"
    else:
        return f"{int(b//1)}.{F}"



def get_acc(n):
    return n * '#' if n > 0 else -n * 'b'



def get_steps(numeral,triad=None,seventh=None,minor_key=False,alterations=[0,0,0],suspensions=[0,0,0,0], return_pcs=False, root_tpc=None):
    """ Returns arabic scale degrees (Bassstufen) given a Roman numeral and type of triad or tetrad.

    Parameters
    ----------

    numeral: str
        Roman numeral (lowercase or uppercase does not change result). Can come with accidentals.
        Examples: '#vii' or 'bbbII'
    triad: {None, 'M','m','o','+'}, optional
        Major, minor, diminished, augmented.
        If `triad` is None, the function returns the tetrad on `numeral` natural to the given mode.
    seventh: {None, 'M7', 'm7', '%7', 'o7', 'n'}, optional
        No seventh, major, minor, minor, diminished or natural
    minor_key: bool, optional
        Changes the results of chords related to III, VI and VII.
    alterations: collection, optional
        A list of one to three elements, indicating alterations of [third, fifth, seventh] in semitones.
        E.g., for a diminished fifth: [0,-1] or for an augmented seventh: [0,0,1]
        NB: Alteration of root is given in `numeral`
    suspensions: collection, optional
        A list of one to four elements, indicating suspensions and retardations of [root, third, fifth, seventh] in semitones.
        E.g., for a fourth suspension in major: [0,1], in minor: [0, 2], retardation of major third: [0,-1]
        NB: The difference with `alterations` is that with `suspensions`, the neighbouring step is returned.
    return_pcs: bool, optional
        If True, the pitch classes of the natural steps of the given major or minor scale are returned,
        starting with the root of the returned chord.
    root_tpc : str or int
        Pass the root of a key as note name or tonal pitch class to return the steps expressed as TPC in relation to this root.
    """
    num_acc, num_degree = split_sc_dg(numeral)
    root = MAJ_RN.index(num_degree.upper())
    pcs = MIN_PCS if minor_key else MAJ_PCS
    pcs = pcs[root:] + [p+12 for p in pcs[:root]] #rotating the scales's pitch classes to calculate intervals
    root_alteration = num_acc.count('#') - num_acc.count('b')
    root_pcs = pcs[0] + root_alteration

    if triad is None:
        if alterations == [0, 0, 0] and suspensions == [0, 0, 0, 0]:
            res = [f"{num_acc}{root+1}" if i==0 else str((root+i)%7+1) for i in [0,2,4,6]]
            return (res, [p%12 for p in pcs]) if return_pcs else res
        else:
            triad = 'M' if numeral[0].isupper() else 'm'

    if len(suspensions) > 0 and suspensions[0] != 0:
        dist = suspensions[0]
        neighbour, neigh_pcs = ((root-1)%7+1, pcs[6]-12) if dist < 0 else ((root+1)%7+1, pcs[1])
        diff = neigh_pcs - root_pcs
        neigh_acc = dist - diff
        root_res = f"{get_acc(neigh_acc)}{neighbour}"
    else:
        root_res = f"{num_acc}{root+1}"

    third_iv, fifth_iv = TRIADS[triad]

    third = pcs[2] - root_pcs
    if len(suspensions) > 1 and suspensions[1] != 0:
        dist = suspensions[1]
        neighbour, neigh_pcs = ((root+1)%7+1, pcs[1] - root_pcs) if dist < 0 else ((root+3)%7+1, pcs[3] - root_pcs)
        diff = neigh_pcs - third_iv
        neigh_acc = dist - diff
        third_res = f"{get_acc(neigh_acc)}{neighbour}"
    else:
        third_iv = third_iv + alterations[0] if len(alterations) > 0 else third_iv
        third_acc = third_iv - third
        third_res = f"{get_acc(third_acc)}{(root+2)%7+1}"

    fifth = pcs[4] - root_pcs
    if len(suspensions) > 2 and suspensions[2] != 0:
        dist = suspensions[2]
        neighbour, neigh_pcs = ((root+3)%7+1, pcs[3] - root_pcs) if dist < 0 else ((root+5)%7+1, pcs[5] - root_pcs)
        diff = neigh_pcs - fifth_iv
        neigh_acc = dist - diff
        fifth_res = f"{get_acc(neigh_acc)}{neighbour}"
    else:
        fifth_iv = fifth_iv + alterations[1] if len(alterations) > 1 else fifth_iv
        fifth_acc = fifth_iv - fifth
        fifth_res = f"{get_acc(fifth_acc)}{(root+4)%7+1}"


    ret = root_res, third_res, fifth_res

    if seventh == 'n':
        ret += (str((root+6)%7+1),)
    elif seventh is not None:
        seventh_iv = SEVENTHS[seventh]

        if len(suspensions) > 3 and suspensions[3] != 0:
            dist = suspensions[3]
            neighbour, neigh_pcs = ((root+5)%7+1, pcs[5] - root_pcs) if dist < 0 else (root+1, 12 - root_alteration)
            diff = neigh_pcs - seventh_iv
            neigh_acc = dist - diff
            ret += (f"{get_acc(neigh_acc)}{neighbour}",)
        else:
            seventh_iv = seventh_iv + alterations[2] if len(alterations) > 2 else seventh_iv
            sev = pcs[6] - root_pcs
            sev_acc = seventh_iv - sev
            ret += (f"{get_acc(sev_acc)}{(root+6)%7+1}",)

    if root_tpc is not None:
        ret = step2tpc(ret, root_tpc, minor_key)
    return (ret, [p%12 for p in pcs]) if return_pcs else ret




def get_interval(numeral,chordtone,alt,minor_key=False):
    """ Returns an interval in semitones given a chord tone and an alteration.

    Parameters
    ----------

    numeral: str
        Roman numeral. Can come with accidentals.
        Examples: '#vii' or 'bbbII'
    chordtone: {1,3,5,7}
        From which of the chord tones the interval is calculated.
    alt: str
        The alteration making up for the calculated interval.
    minor_key: bool, optional
        Changes the results of chords related to III, VI and VII.
    """
    try:
        chordtone = int(chordtone)
        triad = 'M' if numeral[0].isupper() else 'm'
        steps, scale = get_steps(numeral, triad, seventh='n', minor_key=minor_key, return_pcs=True)
        chordtone_step = steps[int((chordtone - 1)/2)]
    except:
        print(f"{numeral} has no chordtone {chordtone}")
        return

    alt_acc, alt_num = split_sc_dg(alt)
    new_pc = scale[(int(alt_num)-1)%7] + alt_acc.count('#') - alt_acc.count('b')
    old_pc = step2pc(chordtone_step,minor_key)
    if abs(new_pc-old_pc) > 6:
        if old_pc >=6 and new_pc < 6:
            new_pc += 12
        if old_pc <=6 and new_pc > 6:
            old_pc += 12
    return new_pc-old_pc



def parse_all(dir, new_folder=None, extensions=['mscx'], overwrite=False, target_extension='tsv', keep_extension=False, recursive=True, MS='mscore', **kwargs):
    """ Convert all files in `dir` that have one of the `extensions` to .mscx format using the executable `MS`.

    Parameters
    ----------
    dir : str
        Directory
    extensions : list, optional
        If you want to parse only certain formats, give those, e.g. ['mscx', 'mscz']
    target_extension : :obj:`str`, optional
        Extensions of the TSV files.
    keep_extension : bool, optional
        If True, the original file format is included in the new file name.
    recursive : bool, optional
        Subdirectories as well.
    MS : str, optional
        Give the path to the MuseScore executable on your system. Need only if
        the command 'mscore' does not execute MuseScore on your system.
    **kwargs : arguments for the function Score.get_notes()
    """
    for subdir, dirs, files in os.walk(dir):
        dirs.sort()
        if not recursive:
            dirs[:] = []
        else:
            dirs.sort()
        old_subdir = os.path.relpath(subdir, dir)
        if new_folder is not None:
            new_subdir = os.path.abspath(os.path.join(new_folder, old_subdir) if old_subdir != '.' else new_folder)
            if not os.path.isdir(new_subdir):
                os.mkdir(new_subdir)
        else:
            new_subdir = old_subdir

        for file in files:
        #if file.endswith(".txt") or file.endswith(".log"):
            m = re.match(r'(.*)\.(.{1,4})$',file)
            if m and (m.group(2) in extensions or extensions == []):
                if keep_extension:
                    neu = '%s_%s.%s' % (m.group(1),m.group(2), target_extension)
                else:
                    neu = '%s.%s' % (m.group(1), target_extension)
                old = os.path.join(subdir, file)
                new = os.path.join(new_subdir, neu)
                if os.path.isfile(new):
                    if overwrite:
                        logging.info(f"{new} overwritten.")
                    else:
                        logging.info(f"{new} not overwritten.")
                else:
                    logging.info(f"Parsing {os.path.join(old_subdir, file)}")
                    S = Score(old)
                    S.get_notes(**kwargs).to_csv(new, sep='\t')
                    logging.info(f"{new} created.")



def rn2tpc(rn, global_minor=False):
    """Turn a Roman numeral into a TPC interval (e.g. for transposition purposes)."""
    try:
        rn_acc, rn_step = split_sc_dg(rn)
        rn_step = rn_step.upper()
    except:
        return apply_function(rn2tpc, rn, global_minor=global_minor)
    step_tpc = RN_TPCS_MIN[rn_step] if global_minor else RN_TPCS_MAJ[rn_step]
    return step_tpc + 7 * rn_acc.count('#') - 7 * rn_acc.count('b')



def split_interval_name(nn):
    nn = str(nn)
    m = re.match("^(P|M|m|A+|D+)(\d+)$", nn)
    assert m is not None, nn + " is not a valid interval name."
    quality = m.group(1)
    i = int(m.group(2))
    return quality, i



def split_note_name(nn):
    nn = str(nn)
    m = re.match("^([A-G]|[a-g])(#*|b*)$", nn)
    assert m is not None, nn + " is not a valid note name."
    return m.group(1), m.group(2)



def split_sc_dg(sd):
    """ Splits a scale degree such as 'bbVI' or a bass step such as 'b6' into accidentals and numeral/step.
    """
    assert sd.__class__ == str, f"Pass string, not {sd.__class__}"
    m = re.match("^(#*|b*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|\d)$",sd)
    if m is None:
        logging.error(sd + " is not a valid numeral.")
        return None, None
    return m.group(1), m.group(2)




def step2pc(sd,minor=False):
    try:
        sd_acc, sd_num = split_sc_dg(sd)
    except:
        return apply_function(step2pc, sd, minor=minor)
    try:
        sd_num = int(sd_num)
    except:
        raise ValueError(f"{sd} does not seem to be a valid scale degree.")
    pcs = MIN_PCS if minor else MAJ_PCS
    return (pcs[(sd_num-1)%7] + sd_acc.count('#') - sd_acc.count('b'))%12



def step2tpc(sd, root='C', minor=False):
    try:
        sd_acc, sd_num = split_sc_dg(sd)
    except:
        return apply_function(step2tpc, sd, root=root, minor=minor)
    try:
        sd_num = int(sd_num) - 1
    except:
        raise ValueError(f"{sd} does not seem to be a valid scale degree.")
    if root.__class__ == str:
        root = name2tpc(root)
    step_tpc = C_MIN_TPCS[sd_num] if minor else C_MAJ_TPCS[sd_num]
    return root + step_tpc + 7 * sd_acc.count('#') - 7 * sd_acc.count('b')



def stufen(numeral, form=None, figbass=None, relativeroot=None, changes='', minor=False, alle=False, local_minor=None):
    """Calculates the bass step ("Bassstufe") given a roman numeral annotation.

    Parameters
    ----------
    numeral: str
        Roman numeral of the chord's root
    form: {None, 'M', 'o', '+' '%'}, optional
        Indicates the chord type if not a major or minor triad (for which `form`is None).
        '%' and 'M' can only occur as tetrads, not as triads.
    figbass: {None, '6', '64', '7', '65', '43', '2'}, optional
        Indicates chord's inversion. Pass None for triad root position.
    relativeroot: str, optional
        Pass a Roman scale degree if `numeral` is to be applied to a different scale
        degree of the local key, as in 'V65/V'
    changes: str, optional
        Added steps such as '+6' or suspensions such as '4' or any combination such as (9+64).
        Numbers need to be in descending order.
    minor: bool, optional
        Pass True if the the local key is not major but minor.
        This affects calculation of chords related to III, VI and VII.
    alle : bool, optional
        Return all degrees of the chord, not only the bass.
    local_minor: bool, optional
        Old parameter name, changed to `minor`. Don't use anymore.

    """
    if local_minor is not None:
        logging.warning("stufen(): Use parameter `minor` instead of `local_minor`.")
    else:
        local_minor = minor
    numeral, form, figbass, relativeroot, changes = tuple(None if isnan(val) else val for val in (numeral, form, figbass, relativeroot, changes))
    if changes is None:
        changes=''
    if form in ['%', 'M']:
        assert figbass in ['7', '65', '43', '2'], f"{numeral}{form}{figbass}{'(' + changes + ')' if changes != '' else ''}: {form} requires figbass since it specifies a chord's seventh."

    def send_out(steps, ix):
        steps = list(steps)
        if alle:
            if ix < len(steps) -1:
                return [steps[ix]] + steps[ix+1:] + steps[:ix]
            else:
                return [steps[ix]] + steps[:ix]
        else:
            return steps[ix]

    num_acc, num_degree = split_sc_dg(numeral)

    if num_degree in ['Ger','Fr','It']:
        if figbass in ['6', None]:
            if alle:
                # aug_steps = {'minor': {'Ger': ['6', '1', '3', '#4'],
                #                        'Fr':  ['6', '1', '2', '#4'],
                #                        'It':  ['6', '1', '#4']},
                #              'major': {'Ger': ['b6', '1', 'b3', '#4'],
                #                        'Fr':  ['b6', '1', '2', '#4'],
                #                        'It':  ['b6', '1', '#4']}}
                # if local_minor:
                #     return send_out(aug_steps['minor'][num_degree], 0)
                # else:
                #     return send_out(aug_steps['major'][num_degree], 0)
                if num_degree in ['Ger', 'It']:
                    numeral = '#iv'
                    form = 'o'
                    changes = 'b3'
                    if num_degree == 'Ger':
                        seventh = 'o7'
                        figbass = '65'
                    elif num_degree == 'It':
                        figbass = '6'
                else:
                    numeral = 'II'
                    form = 'M'
                    seventh = 'm7'
                    changes = 'b5'
                    figbass = '43'
            elif local_minor:
                return '6'
            else:
                return 'b6'
        else:
            return send_out(['Unclear'], 0)

    if form is not None and form in 'o+%':
        if form == '%':
            triad = 'o'
        else:
            triad = form
    else:
        triad = 'M' if num_degree.isupper() else 'm'

    num = numeral if relativeroot is None else appl(numeral,relativeroot,local_minor)


    if relativeroot is not None:
        _, rel_num = split_sc_dg(relativeroot)
        rel_minor = True if rel_num.islower() else False
    else:
        rel_minor = local_minor

    alterations, suspensions = eval_changes(numeral,changes,rel_minor)

    if alle or figbass == '2':
        if figbass in ['7', '65', '43', '2']:
            sev = form + '7' if form in ['o', '%', 'M'] else 'm7'
        else:
            sev = None
        steps = get_steps(num, triad, sev, minor_key=local_minor,alterations=alterations,suspensions=suspensions)
    else:
        steps = get_steps(num, triad, minor_key=local_minor,alterations=alterations,suspensions=suspensions)

    if figbass in [None, '6', '64', '7', '65', '43']:
        if figbass in [None, '7']:
            return send_out(steps,0)
        if figbass in ['6', '65']:
            return send_out(steps,1)
        if figbass in ['64', '43']:
            return send_out(steps,2)
    elif figbass == '2':
        return send_out(steps,3)
    else:
        return send_out(['Error'], 0)



def relativize_labels(frame, to='I', drop=False):
    """Rewrites all labels in relation to `to`. Works like `transpose_labels()` but
    transposes only the `relativeroot` and expresses all labels standing in a different
    key as applied chords.
    Pass chords for one piece at a time. For concatenated pieces, use apply_to_pieces().
    """
    frame = frame.copy()
    def transpose_row(row):
        if row.key != to:
            row.key = appl_backwards(row.key, to, global_minor)
        if not isnan(row.relativeroot):
            new_rel = appl_backwards(appl(row.relativeroot,row.old_key,new_global_minor),to, new_global_minor)
            if new_rel in ['i','I']:
                row.relativeroot = np.nan
            else:
                row.relativeroot = new_rel
        else:
            if row.old_key != to:
                row.relativeroot = appl_backwards(row.old_key, to, global_minor)

        if not isnan(row.relativeroot):
            if row.relativeroot in ['I', 'i']:
                row.relativeroot = np.nan
                chord = row.numeral
            elif row.numeral in ['I', 'i'] and  ( (row.numeral.isupper() and row.relativeroot[0].isupper()) or (row.numeral.islower() and row.relativeroot[0].islower()) ):
                chord = row.relativeroot
                row.relativeroot = np.nan
            else:
                chord = row.numeral
        else:
            chord = row.numeral

        if isnan(chord):
            row.rel_label = '@none'
            return row

        if not isnan(row.form):
            chord += row.form
        if not isnan(row.figbass):
            chord += row.figbass
        if not isnan(row.changes):
            chord += f"({row.changes})"
        if isnan(row.relativeroot):
            if row.key == to:
                row.rel_label = chord
            else:
                row.rel_label = f"{chord}/{row.key}"
        else:
            row.rel_label = f"{chord}/{row.relativeroot}"

        return add_steps(row)

    features = ['measure', 'beat', 'label', 'position']
    global_minor = True if frame.globalkey.iloc[0][0].islower() else False
    new_global_minor = True if to.islower() else False
    minor_keys = []
    major_keys = []
    #local_minor = {k: True if re.search("(vii|vi|v|iv|iii|ii|i)",k) else False for k in frame.key.unique()}
    def local_minor(k):
        if k in minor_keys:
            return True
        elif k in major_keys:
            return False
        elif re.search("(vii|vi|v|iv|iii|ii|i)",k):
            minor_keys.append(k)
            return True
        else:
            major_keys.append(k)
            return False

    rel_key = {k: appl_backwards(k,to,global_minor) for k in frame.key.unique()}

    frame['old_relativeroot'] = frame.relativeroot
    frame['rel_label'] = np.nan
    frame['old_key'] = frame.key
    frame = frame.apply(transpose_row,axis=1).astype({'next_id':'Int64'})
    frame.key = to
    if drop:
        frame = frame[[col for col in frame.columns if not col in ['old_relativeroot', 'old_key']]]
    else:
        frame = frame
    return frame



def transpose_to_C(chord_notes, chord_list, minor_to_A=False):
    """Transpose all notes in the note list `chord_notes` to C/c or C/a depending on global and local key information within `chord_list`"""
    transp = - name2tpc(chord_list.globalkey) - chord_list.apply(lambda r: rn2tpc(r.key, r.globalminor), axis=1)
    if minor_to_A:
        transp += 3 * chord_list.key.str.islower()
    transp.rename('transp', inplace=True)
    cnt = chord_notes.join(transp)
    return cnt.groupby('transp').apply(lambda df: transpose_note_list(df, df.transp.iloc[0])).drop(columns='transp')



def sort_intervals(coll):
    order = ['P', 'M', 'm', 'D', 'A']
    def s(iv):
        m = re.match(r"(M|m|P|A+|D+)(\d)", iv)
        size, i = m.group(1), m.group(2)
        return int(i) + order.index(size)*0.2

    return sorted(coll, key=s)



def sort_tpcs(tpcs, ascending=True, start=None):
    """ Sort tonal pitch classes by order on the piano.

    Parameters
    ----------
    tpcs : collection of int
        Tonal pitch classes to sort.
    ascending : bool, optional
        Pass False to
    start : int, optional
        Start on or above this TPC.
    """
    res = sorted(tpcs, key=lambda x: (tpc2pc(x),-x))
    if start is not None:
        pcs = tpc2pc(res)
        start = tpc2pc(start)
        i = 0
        while i < len(pcs) - 1 and pcs[i] < start:
            i += 1
        res = res[i:] + res[:i]
    return res if ascending else list(reversed(res))



def transpose_labels(frame,to):
    """Transposes all chord labels from different keys to the key `to`, including their relative keys.
    Pass chords for one piece at a time. For concatenated pieces, use apply_to_pieces().
    """
    frame = frame.copy()
    def transpose_row(row):
        if isnan(row.relativeroot):
            row.numeral = appl(row.numeral,rel_key[row.key],local_minor[row.key])
            row.chord = row.numeral
            if not isnan(row.form):
                row.chord += row.form
            if not isnan(row.figbass):
                row.chord += row.figbass
        else:
            new_rel = appl_backwards(appl(row.relativeroot,row.key,local_minor[row.key]),to,global_minor)
            if new_rel not in ['i','I']:
                row.relativeroot = new_rel
                row.chord = row.old_chord.split('/')[0] + '/' + row.relativeroot
            else:
                row.relativeroot = np.nan
                row.chord = row.old_chord.split('/')[0]
        row.key = to
        return add_steps(row)

    global_minor = True if frame.globalkey.iloc[0][0].islower() else False
    local_minor = {k: True if re.search("(vii|vi|v|iv|iii|ii|i)",k) else False for k in frame.key.unique()}
    rel_key = {k: appl_backwards(k,to,global_minor) for k in frame.key.unique()}

    frame['old_key'] = frame.key
    frame['old_chord'] = frame.chord
    frame.loc[frame.key != to] = frame.loc[frame.key != to].apply(transpose_row,axis=1)
    return frame



def val2beat(b):
    """Turns any value in a beat string of format '2.1/2' (i.e. 'beat.subbeat')"""
    if b.__class__ == str:
        return b
    else:
        return float2beat(b)


def add_steps(row):

    #glob = re.search("[A-Ga-g]",row.globalkey).string
    #global_minor = True if glob.islower() else False

    loc_acc, loc_degree = split_sc_dg(row.key)
    loc_minor = True if loc_degree.islower() else False

    rel = row.relativeroot if not isnan(row.relativeroot) else None
    fig = row.figbass if not isnan(row.figbass) else None
    form = row.form if not isnan(row.form) else None

    alts = row.changes if not isnan(row.changes) else ''



    row['bass_step'] = stufen(row.numeral, form, fig, rel, loc_minor, alts) if not isnan(row.numeral) else np.nan
    return row



def get_scale_pcs(numeral,minor_key=False):
    num_acc, num_degree = split_sc_dg(numeral)
    root = MAJ_PCS.index(num_degree.upper())
    pcs = MIN_PCS if minor_key else MAJ_PCS
    pcs = pcs[root:] + [p+12 for p in pcs[:root]]
    rootshift = pcs[0]+num_acc.count('#')-num_acc.count('b')
    pcs = [pc - pcs[0] if i==0 else pc - (rootshift) for i,pc in enumerate(pcs)]
    pcs += [pc + 12 for pc in pcs] + [pc + 24 for pc in pcs]
    return pcs

def changes2pc(numeral,changes,minor_key=False):
    """ Receives `changes` as a string (such as '+139#74') and returns a list of pitch classes relative to the local tonic 0"""
    pcs = get_scale_pcs(numeral,minor_key)

    alts = [t[2:] for t in re.findall("((\+)?(#+|b+)?(1\d|\d))",changes)]
    res = []
    for a in alts:
        acc, step = a
        res.append(pcs[int(step)-1]+acc.count('#')-acc.count('b'))
    return res

def transpose_changes(changes,old_num,old_key,new_key,global_minor):
    old_minor = True if re.search("(vii|vi|v|iv|iii|ii|i)",old_key) else False
    new_minor = True if re.search("(vii|vi|v|iv|iii|ii|i)",new_key) else False
    ivls = changes2pc(old_num,changes,old_minor)

    new_num = appl_backwards(appl(old_num,old_key,global_minor),new_key,global_minor)
    num_acc, num_degree = split_sc_dg(old_num)
    pcs = get_scale_pcs(new_num,new_minor)

    alts = [t[1:] for t in re.findall("((\+)?(#+|b+)?(1\d|\d))",changes)]
    res = ''
    for i, a in enumerate(alts):
        diff = ivls[i]-pcs[int(a[2])-1]
        acc = abs(diff) * 'b' if diff <0 else diff * '#'
        res += a[0] + acc + a[2]
    return res



def transpose_note_list(note_list, fifths, midi_direction=0, inplace=False, note_names=True, octaves=True):
    """Transpose the notes and corresponding information.

    Parameters
    ----------
    note_list : pd.DataFrame
        Note list with columns ['tpc', 'midi']
    fifths : int
        Number of fifths that you want to substract from the tonal pitch classes TPC.
    midi_direction : {-1, 0, 1}, optional
        0 : automatic behaviour, move MIDI pitches by minimum amount
        -1: shift downwards
        1:  shift upwards
    """
    if not inplace:
        note_list = note_list.copy()
    if fifths != 0:
        note_list.tpc += fifths
        midi_transposition = tpc2pc(fifths) if midi_direction >= 0 else -tpc2pc(fifths)
        if midi_direction == 0:
            midi_transposition = midi_transposition if midi_transposition <= 6 else midi_transposition % -12
        note_list.midi += midi_transposition
        if note_names:
            note_list['note_names'] = tpc2name(note_list.tpc)
        if octaves:
            note_list['octaves'] = midi2octave(note_list.midi)
    return note_list.astype({'tpc': 'Int64', 'midi': 'Int64'})



def index_tuples(df,sl=SL[:], nested=False):
    """ Get tuples representing a slice of multiindex levels."""
    def separate(frame):
        S = frame.iloc[:,-1]
        beginnins = list(frame[S - S.shift() != 1].itertuples(index=False,name=None))
        endings = list(frame[S.shift(-1) - S != 1].itertuples(index=False,name=None))
        sections = list(zip(beginnins,endings))
        return sections

    def get_tuples(frame):
        return [t[sl] for t in frame.index.to_list()]

    lvls = len(df.index.names)

    if nested:
        if lvls == 2:
            res = sorted([t[sl] for t in df.index.to_list()])
            res_df = pd.DataFrame(res)
            return res_df.groupby(0).apply(separate).sum()
        elif lvls > 2:
            return df.groupby(level=list(range(lvls-2))).apply(get_tuples).to_dict()
        else:
            print("Not implemented for single index.")
            return []
    else:
        res = sorted([t[sl] for t in df.index.to_list()])
        return res




# def selec(idx,tuple_list,slic=SL[:]):
#     """(Helper function for `select_from_partial_index`) Map this function on an Multiindex to get a boolean
#     mask depending on whether the first n level values occur in `tuple_list`.
#
#     Parameters
#     ----------
#
#     idx: tuple
#         A single multiindex tuple to be sliced
#     tuple_list: list of tuples
#         List which the sliced multiindex tuple is checked against.
#     slic: slice or SliceMaker
#         Which part of the multiindex tuple to check. Defaults to checking the
#         first n levels where n is the tuples' length in `tuple_list`.
#     """
#     try:
#         n = len(tuple_list[0])
#     except:
#         print("This does not seem to be a list of tuples.")
#         return False
#
#     if slic == SL[:]:
#         slic = SL[:n]
#
#     assert all(len(t) == n for t in tuple_list), "All tuples need to have the same length as the first one"
#     return True if idx[slic] in tuple_list else False


def select_from_partial_index(df,tuple_list,slic=SL[:]):
    """ .loc works only with lists of complete multiindex key, i.e. with tuples encompassing all index levels.
    This function makes it possible to slice using partial multiindex keys starting from level 0. If you want
    to check starting from a different level, pass the corresponding slice for the multiindex tuples.

    Parameters
    ----------

    df: DataFrame or Series
        Multiindexed DataFrame or Series you want to slice
    tuple_list: list of tuples
        List which the sliced multiindex tuples are checked against.
    slic: slice or SliceMaker
        Which part of the multiindex tuple to check. Defaults to checking the
        first n levels where n is the tuples' length in `tuple_list`.

    Examples
    --------

        >>> df              # This is a DataFrame with three index levels
                    col
        A	x	0	0
         	y	1   1
        B	x	2	2
        	y	3   3

        >>> select_from_partial_index(df,tuple_list=[('B','x')])
                    col
        B	x	2	2

        >>> select_from_partial_index(df, [('y',)], SL[1:2])
                    col
        A	y	1	1
        B	y	3	3


    """
    try:
        n = len(tuple_list[0])
    except:
        print("This does not seem to be a list of tuples.")
        return False

    if slic == SL[:]:
        slic = SL[:n]
    return df[df.index.map(lambda x:selec(x,tuple_list,slic))]



def construct_range_values(l):
    """ Pass a list of 2-collections (lists or tuples) to get all included values.

    Example
    -------

        >>> construct_range_values([(1,3),(2,4)])
        [1, 2, 3, 2, 3, 4]
    """
    if nest_level(l,True) == 1:
        l = [l]
    return sum([list(range(c[0],c[1]+1)) for c in l],[])



def selec(idx,tuple_set,slic=SL[:]):
    """(Helper function for `select_from_partial_index`) Map this function on an Multiindex to get a boolean
    mask depending on whether the first n level values occur in `tuple_set`.

    Parameters
    ----------

    idx: tuple
        A single multiindex tuple to be sliced
    tuple_set: set of tuples
        Set which the sliced multiindex tuple is checked against.
    slic: slice or SliceMaker
        Which part of the multiindex tuple to check. Defaults to checking the
        first n levels where n is the tuples' length in `tuple_list`.
    """
    # try:
    # n = len(tuple_set[0])
    # except:
    #     print("This does not seem to be a list of tuples.")
    #     return False

    if tuple_set.__class__ != set:
        logging.warning(f"selec(): Pass a set for better performance, not {tuple_set.__class__}")
        ts = set(tuple_set)
    else:
        ts = tuple_set

    if slic == SL[:]:
        slic = SL[:n]

    #assert all(len(t) == n for t in tl), "All tuples need to have the same length as the first one"
    return True if idx[slic] in ts else False



def select_from_partial_index(df,tuple_list,slic=SL[:], return_index=False):
    """ .loc works only with lists of complete multiindex key, i.e. with tuples encompassing all index levels.
    This function makes it possible to slice using partial multiindex keys starting from level 0. If you want
    to check starting from a different level, pass the corresponding slice for the multiindex tuples.

    Parameters
    ----------

    df: DataFrame or Series
        Multiindexed DataFrame or Series you want to slice
    tuple_list: list of tuples
        List which the sliced multiindex tuples are checked against.
    slic: slice or SliceMaker
        Which part of the multiindex tuple to check. Defaults to checking the
        first n levels where n is the tuples' length in `tuple_list`.
    return_index : bool, optional
        If set to True, the completed index for slicing is returned instead of the sliced DataFrame.

    Examples
    --------

        >>> df              # This is a DataFrame with three index levels
                    col
        A	x	0	0
         	y	1   1
        B	x	2	2
        	y	3   3

        >>> select_from_partial_index(df,tuple_list=[('B','x')])
                    col
        B	x	2	2

        >>> select_from_partial_index(df, [('y',)], SL[1:2])
                    col
        A	y	1	1
        B	y	3	3


    """

    #try:
    n = len(tuple_list[0])
    # except:
    #     raise ValueError("This does not seem to be a list of tuples.")

    try:
        if n == len(df.index.levels):
            return df.loc[tuple_list].index if return_index else df.loc[tuple_list]
    except:
        raise ValueError("df needs to have a multiindex. Otherwise, use simple slicing.")

    if slic == SL[:]:
        slic = SL[:n]

    ix = df.index.map(lambda x:selec(x,set(tuple_list),slic))
    return df.index[ix] if return_index else df[ix]



def select_by_index(df, index_df, **kwargs):
    """Shortcut for using select_from_partial_index.

    Parameters
    ----------
    df: DataFrame or Series
        Multiindexed DataFrame or Series you want to slice
    index_df: DataFrame or Series
        Use the index values of this DataFrame or Series to extract values from `df`.
    slic: slice or SliceMaker
        Which part of the multiindex tuple to check. Defaults to checking the
        first n levels where n is the number of index levels of `index_df`.
    """
    if len(index_df) == 0:
        return pd.DataFrame()
    if index_df.__module__ in ['pandas.core.indexes.base', 'pandas.core.indexes.multi']:
        ix_tp = index_df.to_list()
    else:
        ix_tp = index_df.index.to_list()
    return select_from_partial_index(df, ix_tp, **kwargs)



def select_by_values(df,col,val,every=False):
    """ Retrieve a slice of `df` where one or all elements of `col` is/are contained in the list `val`.

    Parameters
    ----------

    df: DataFrame
        DataFrame to slice.
    col: str
        Column where to look for the values. For Columns containing lists or strings, containment is
        checked for all elements within `val`.
    val: value or list of values
        Function returns rows where `col` contains (one of) `val`.
    every: bool, optional
        If True, all elements within `val` need to be contained in val. If False, one suffices.
    """

    selec = df[col][df[col].notna()]

    if val.__class__ != list:
        val = [val]

    def choose(series_val):
        if series_val.__class__ in [list,str]:
            if every:
                return all([v in series_val for v in val])
            else:
                return any([v in series_val for v in val])

        else:
            return series_val in val

    return df.loc[selec[selec.apply(choose)].index]




def plot_bar(df,kind='bar', fname='test.png', width=1100, height=250, legend='v', **kwargs):
    fig = df.iplot(asFigure=True,kind=kind, **kwargs)
    fig.update(layout={'legend':
                           {'bgcolor': '#FFFFFF', 'orientation': legend, 'y': -0.21 if legend=='h' else 0},
                       'paper_bgcolor': '#FFFFFF',
                       'plot_bgcolor': '#FFFFFF',
                       'margin':
                           {'l':40, 'r':0, 'b':0 if legend=='h' else 50, 't':0, 'pad':0}
                      })
    IO.write_image(fig,fname,width=width,height=height)
    iplot(fig)


def plot_pie(df,kind='pie', fname='test.png', width=500, height=500, legend='v', **kwargs):
    fig = df.iplot(asFigure=True,kind=kind, **kwargs)
    fig.update(layout={'legend':
                           {'bgcolor': '#FFFFFF', 'orientation': legend, 'x':0.15},# 'y': 0, 'yanchor': 'top'},
                       'paper_bgcolor': '#FFFFFF',
                       'plot_bgcolor': '#FFFFFF',
                       'margin':
                           {'l':0, 'r':0, 'b':0 , 't':0, 'pad':0}
                      })
    IO.write_image(fig,fname,width=width,height=height)
    iplot(fig)



# class dataset(object):
#     """ This class is designed to hold, evaluate and manipulate the dataset in question.
#
#     Attributes
#     __________
#
#     df : DataFrame
#         Entire dataset in one multiindexed DataFrame as displayed by `__init__`.
#     dir : str
#         Directory holding the dataset.
#     parsed_score : piece.Piece
#         To avoid parsing the same score consecutive times.
#     recursive : bool
#         Whether or not subdirectories were scanned.
#     sequences : DataFrame
#         Can hold various subsets of the dataset.
#     stages : DataFrame
#         Has the same index as `self.sequences` and contains information about cadence stages.
#     scoredir : str
#         Directory holding the MSCX (uncompressed MuseScore 2) files.
#
#     Methods
#     -------
#
#     expand_labels(column)
#         Split harmony labels complying with the DCML syntax into their features.
#     extract_sequences(column,value,direction='up',per_piece=False)
#         Create sequences by joining every `value` in `column` together with adjecent NaNs.
#     show(piece,ind,context=0,scoredir=None)
#         Display the scores of a data entry from the corresponding MSCX file (uncompressed MuseScore 2 format).
#     show_corpus()
#         Show the names of the pieces contained in the dataset.
#     show_value_counts(columns,per_piece=False,showna=False)
#         Show the value counts for one or several columns, aggregated or for each piece.
#     """
#
#     def __init__(self, dir, scoredir=None, suffix=None, recursive=True, extensions=['tsv', 'txt', 'csv'], sep=['\t', '\t', ','], dtypes={'beat':str,}, beat_treatment=None):
#         """ Read all separated values files ending on `extensions` and store as a dataframe.
#
#         Parameters
#         ----------
#
#         dir : str
#             Directory to scan for files ending on `extensions`.
#         suffix : str, optional
#             Instead of calculating anew, files stored by `self.dump()` are being loaded.
#         scoredir : str, optional
#             Directory holding the MSCX (uncompressed MuseScore 2) files.
#         recursive : :obj:`bool`, optional
#             Scan subdirectories as well? Defaults to `True`.
#         extensions : :obj:`list` of :obj:`str`, optional
#             File extensions to consider. Defaults to `['tsv','csv']`.
#             If the list is shorter than `extensions`, the last element is used.
#         sep : :obj:`list` of :obj:`str`, optional
#             The separator symbols corresponding to the extensions. Defaults to `['\t',',']`.
#         dtypes: dictionary
#             Here, you can specify the data types for every column name.
#         beat_treatment: {None,fractionize}
#             If needed, pass a function to transform the representation of beats.
#             Use `fractionize` if the beats come as floats, to get a representation
#             such as 1.1/2 instead of 1.5
#
#         """
#
#         self.dir = dir
#         self.recursive = recursive
#         self.scoredir = scoredir
#         self.parsed_score = None
#         self.sequences = None
#         self.stages = None
#
#         if suffix is not None:
#
#             for k in ['dataset', 'sequences', 'stages']:
#                 fname = f"{k}{suffix}.tsv"
#                 path = os.path.join(dir,fname)
#                 if k == 'dataset':
#                     if not os.path.isfile(path):
#                         print(path + ' not found.')
#                         self.df = None
#                         return None
#                     else:
#                         label_order = ['measure','beat','position','duration','timesig','globalkey','key','cadence',
#                                         'cadence_subtype','pedal','chord','bass_step','numeral','ext_numeral','relativeroot',
#                                         'form','figbass','changes','phraseend','label','alt_label','num_stage',
#                                         'bs_stage','num_stage_of','bs_stage_of','next','previous']
#                         self.df = pd.read_csv(path,sep='\t',index_col=[0,1],dtype={'figbass':'Int64'})[label_order]
#                         # Convert columns to lists or tuples
#                         for col in ['num_stage','bs_stage','num_stage_of','bs_stage_of','next','previous']:
#                             if col in self.df:
#                                 self.df[col] = self.df[col][self.df[col].notna()].apply(eval)
#                 elif k == 'stages':
#                     self.stages = pd.read_csv(path,sep='\t',index_col=[0,1,2,3],dtype={'num_stage':'Int64','bs_stage':'Int64','figbass':'Int64'})
#                 elif k == 'sequences':
#                     self.sequences = pd.read_csv(path,sep='\t',index_col=[0,1,2,3],dtype={'figbass':'Int64'})
#                 print(fname + ' loaded.')
#
#         else:
#
#             df_dict = {}
#             for subdir, dirs, files in os.walk(dir):
#                 dirs.sort()
#                 if not recursive:
#                     dirs[:] = []
#
#                 exts = '|'.join(extensions)
#                 for file in files:
#                     m = re.match(f'(.+)\.({exts})$',file)
#                     if m:
#                         name = m[1]
#                         ext  = m[2]
#                         ind = extensions.index(ext)
#                         s = sep[-1] if ind >= len(sep) else sep[ind]
#                         path = os.path.join(subdir,file)
#                         if not name in df_dict:
#                             df_dict[name] = pd.read_csv(path, sep=s, dtype=dtypes, error_bad_lines=False, warn_bad_lines=True)
#                         else:
#                             print("!!! More than one file read for " + name)
#                             df_dict[name] = pd.concat([df_dict[name],pd.read_csv(path, sep=s, dtype=dtypes, error_bad_lines=False, warn_bad_lines=True)])
#
#             self.df = pd.concat(df_dict,names=['piece', 'ix'],sort=True)
#             if beat_treatment is not None:
#                 self.df.beat = self.df.beat.apply(beat_treatment)
#
#
#
#
#
#     def __repr__(self):
#         display(self.df)
#         series = self.df.index.get_level_values(0)
#         n_series = len(series.unique())
#         n_elements = len(self.df)
#         vc_series = series.value_counts()
#         ma = f"{vc_series.max()} ({vc_series.idxmax()})"
#         mi = f"{vc_series.min()} ({vc_series.idxmin()})"
#         return f"#series: {n_series}\n#elements: {n_elements}\navg. series length: {round(n_elements/n_series,1)}\nmax length: {ma}\nmin length: {mi}"
#
#
#
#     def __getitem__(self, ix):
#         try:
#             return self.sequences.loc[ix]
#         except:
#             return f"Available keys: {self.sequences.index.get_level_values(0).unique().tolist()}"
#
#
#     def expand_labels(self,column):
#         """ Split harmony labels complying with the DCML syntax into their features.
#
#         Parameters
#         ----------
#
#         column : str
#             Name of the column that holds the harmony labels.
#
#         """
#         spl = self.df.loc[:,column].str.extract(regex, expand=True)
#         for f in features:
#             self.df.loc[:,f] = spl[f]
#
#         def fill_values(frame):
#             piece = frame.index[0][0]
#             frame = frame.droplevel(0)
#             ####################################
#             # expand global key to all harmonies
#             ####################################
#             global_key = frame.key[0]
#             frame.loc[:,'globalkey'] = global_key
#             ####################################
#             # expand local keys to all harmonies
#             ####################################
#             n_k = len(frame.key.index)
#             localkeys = frame.key[frame.key.notna()]
#             n_l = len(localkeys)
#             global_minor = True if global_key[0].islower() else False
#             localkeys[0] = 'i' if global_minor else 'I'
#             for i,(begins,localkey) in enumerate(localkeys.iteritems()):
#                 ends = localkeys.index[i+1] if i < n_l-1 else n_k
#                 frame.key.iloc[begins:ends] = localkey
#             ##########################################
#             #expand pedal notes to concerned harmonies
#             ##########################################
#             beginnings = frame.pedal[frame.pedal.notna()]
#             n_b = len(beginnings)
#             endings = frame.pedalend[frame.pedalend.notna()]
#             n_e = len(endings)
#             assert n_b == n_e, f"{piece}: {n_b} organ points started, {n_e} ended! Beginnings:\n{frame[frame.pedal.notna()]}\nEndings:\n{frame[frame.pedalend.notna()]}"
#             for i,(begins,pedal) in enumerate(beginnings.iteritems()):
#                 section = frame.iloc[begins:endings.index[i]+1]
#                 section.pedal = pedal
#                 keys = list(section.key.unique())
#                 if len(keys) > 1:
#                     original_key = keys[0]
#                     for k in keys[1:]:
#                         section.pedal[section.key == k] = appl_backwards(appl(pedal,original_key,global_minor),k,global_minor)
#             frame = frame.drop(columns='pedalend')
#             #####################
#             # don't fill in chord types because 'M' stands for a major 7th
#             #####################
#             # frame.form[frame.numeral.isin(['I','II','III','IV','V','VI','VII'])] = 'M'
#             # frame.form[frame.numeral.isin(['i','ii','iii','iv','v','vi','vii'])] = 'm'
#
#             return frame
#
#         self.df = self.df.groupby('piece',sort=False).apply(fill_values)
#         self.df = self.df.apply(add_steps,axis=1)
#         self.df.head()
#
#
#
#
#     def extract_sequences(self,column,add=1,unify_keys=False):
#         """ Returns sections of the DataFrame that span existing values in `column`.
#
#         Parameters
#         ----------
#
#         column : str
#             Column on which to base the segmentation.
#         add : int, optional
#             Choose how many rows of the subsequent section should be added.
#
#         """
#
#         def sequences(frame):
#             """Extract sequences for one piece at a time. `Frame` needs to have a consecutive index at the second multiindex level."""
#
#             def previous_harmony(row):
#                 """Copies the values of the previous row concerning the columns defined in `features`"""
#                 piece, ix = row.name
#                 previous = frame.loc[row.name]
#                 while isnan(previous.chord):
#                     ix -= 1
#                     previous = frame.loc[(piece,ix)]
#                 features = ['label', 'alt_label', 'chord', 'pedal', 'numeral', 'form', 'figbass', 'changes', 'relativeroot', 'phraseend', 'bass_step']
#                 row.loc[features] = previous.loc[features] # Note that this creates a consecutive repetition of the previous label.
#                                                            # In order to filter the repetition out, it is left without a 'duration' attribute
#                 row['keyword'] = 'terminal_inferred'
#                 return row
#
#             # For cadences without harmony label: Copy the previous harmony
#             col = frame.loc[:,column]
#             without_label = frame.loc[col.notna() & frame.chord.isna()].apply(previous_harmony,axis=1)
#             frame.loc[col.notna() & frame.chord.isna()] = without_label
#
#             # Select the existing values as terminals and create section indices based on the consecutive index
#             value_ixs = [0] + frame[col.notna()].index.get_level_values(1).tolist()
#             section_ixs = list(zip(value_ixs[:-1],value_ixs[1:]))
#
#             # Store the slices in `sections` and create the overarching multiindex in `keys`, making use of the defaultdict `instances`
#             sections = []
#             keys = []
#             for fro, to in section_ixs:
#                 category = col.iloc[to]
#                 keys.append((category, instances[category]))
#                 instances[category] += 1
#                 if to+1+add < len(frame.index):
#
#                     seq = frame.iloc[fro:to+1+add].copy()
#                     seq['keyword'].iloc[-add:] = 'next_seq'
#                     if isnan(seq['keyword'].iloc[-(1+add)]):
#                         seq['keyword'].iloc[-(1+add)] = 'terminal'
#                     #if (category, instances[category]) == ('DEC',1): print(seq)
#                 else:
#
#                     seq = frame.iloc[fro:to+1].copy()
#                     seq['keyword'].iloc[-1] = 'terminal'
#
#                 if not isnan(seq['cadence'].iloc[0]):
#                     seq['keyword'].iloc[0] = 'previous_terminal'
#
#                 sections.append(seq)
#             #print(str(len(sections)),str(len(keys)))
#             return pd.concat(sections,keys=keys,names=['type','instance','piece','#'],sort=True)
#
#
#         def transpose_frame(frame,to):
#             """Transposes all chords in different keys to the key `to`."""
#             frame = frame.copy()
#             def transpose_row(row):
#                 old_chord = row.chord.split('/')[0] if not isnan(row.chord) else np.nan
#                 old_numeral = row.numeral
#                 row.old_key = row.key
#                 if isnan(row.relativeroot):
#                     row.numeral = appl(row.numeral,rel_key[row.key],local_minor[to])
#                     row.chord = row.numeral if isnan(row.figbass) else row.numeral + row.figbass
#                     if not isnan(row.changes):
#                         row.changes = transpose_changes(row.changes,old_numeral,row.old_key,to,global_minor)
#                 else:
#                     new_rel = appl_backwards(appl(row.relativeroot,row.key,local_minor[row.key]),to,global_minor)
#                     if new_rel not in ['i','I']:
#                         row.relativeroot = new_rel
#                         row.chord = old_chord + '/' + row.relativeroot
#                     else:
#                         row.relativeroot = np.nan
#                         row.chord = old_chord
#                 if not isnan(row.pedal):
#                     old = row.pedal
#                     row.pedal = appl(row.pedal,rel_key[row.key],local_minor[to])
#                 row.key = to
#                 return row
#
#             features = ['measure', 'beat', 'label', 'position']
#             global_minor = True if frame.globalkey.iloc[0][0].islower() else False
#             local_minor = {k: True if re.search("(vii|vi|v|iv|iii|ii|i)",str(k)) else False for k in np.append(frame.key.unique(), frame.relativeroot.unique())}
#             rel_key = {k: appl_backwards(k,to,global_minor) for k in frame.key.unique()}
#
#             frame['old_key'] = np.nan
#             frame[frame.key != to] = frame[frame.key != to].apply(transpose_row,axis=1)
#             return frame
#
#
#         def unify_key(frame,threshold=1):
#             """ Transposes the chords to the last occurring `key` unless it occurs less equal the `threshold` in which case
#             the second last key is used."""
#             shortframe = frame[frame.keyword != 'next_seq']
#
#             # This block transposes to the terminal's relativeroot in the given case
#             if shortframe.cadence.iloc[-1] in ['PAC','HC','IAC']:
#                 if isnan(shortframe.relativeroot.iloc[-1]):
#                     relativeroot = None
#                 elif (shortframe.relativeroot.iloc[-1] == shortframe.relativeroot.iloc[-2]) or (shortframe.relativeroot.iloc[-1] == shortframe.numeral.iloc[-2]):
#                     relativeroot = shortframe.relativeroot.iloc[-1]
#                 elif (shortframe.relativeroot.iloc[-2] == shortframe.numeral.iloc[-1]):
#                     relativeroot = shortframe.relativeroot.iloc[-2]
#                 else:
#                     relativeroot = None
#                 if relativeroot is not None:
#                     transposed = transpose_frame(frame,relativeroot)
#                     return transposed.apply(add_steps,axis=1)
#
#
#
#             seq = shortframe.key
#             order = seq[seq.shift() != seq] # getting keys in appearing order
#             counts = seq.value_counts()
#             if len(order) > 1:
#                 to = order[-2] if counts[order[-1]] <= threshold else order[-1]
#                 transposed = transpose_frame(frame,to)
#
#                 return transposed.apply(add_steps,axis=1)
#             else:
#                 return frame
#
#
#         instances = defaultdict(lambda: 0)
#         df = self.df.copy()
#         df['keyword'] = np.nan
#         seq_df = df.groupby('piece',group_keys=False,sort=False).apply(sequences)
#
#         if unify_keys:
#             seq_df = seq_df.groupby(level=[0,1],group_keys=False,sort=False).apply(unify_key)
#
#         label_order = ['measure','beat','position','duration','timesig','globalkey','key','cadence',
#                        'cadence_subtype','pedal','chord','bass_step','numeral','relativeroot',
#                        'form','figbass','changes','phraseend','keyword','label','alt_label','old_key']
#         seq_df = seq_df.sort_index()[label_order]
#         self.sequences = seq_df
#         return seq_df
#
#
#
#     def compute_stages(self):
#         self.stages = self.sequences.copy()
#         self.stages['ext_numeral'] = self.stages.numeral
#         self.stages.ext_numeral[self.stages.relativeroot.notna()] = self.stages.numeral + '/' + self.stages.relativeroot
#         self.stages = self.stages.groupby(level=[0,1]).apply(add_stages)
#
#
#
#     def add_stage_info(self):
#
#         def add_info(row):
#             piece, ix = row.name
#             st_rows = self.stages.loc(axis=0)[idx[:,:,piece,ix]]
#
#             num_stages = st_rows[st_rows.num_stage.notna()]
#             bs_stages = st_rows[st_rows.bs_stage.notna()]
#
#             num_ix = [t[:2] for t in num_stages.index]
#             bs_ix = [t[:2] for t in bs_stages.index]
#             num_ix = np.nan if num_ix == [] else num_ix
#             bs_ix = np.nan if bs_ix == [] else bs_ix
#
#             num_st = list(num_stages.num_stage.values)
#             bs_st = list(bs_stages.bs_stage.values)
#             num_st = np.nan if num_st == [] else num_st
#             bs_st = np.nan if bs_st == [] else bs_st
#
#             row['num_stage'] = num_st
#             row['bs_stage'] = bs_st
#             row['num_stage_of'] = num_ix
#             row['bs_stage_of'] = bs_ix
#             return row
#
#         self.df = self.df.apply(add_info,axis=1)
#         self.df['ext_numeral'] = self.df.numeral
#         self.df.ext_numeral[self.df.relativeroot.notna()] = self.df.numeral + '/' + self.df.relativeroot
#
#
#
#     def add_neighbours(self):
#
#
#         def neighbours(frame):
#             frame['ixs'] = frame.index
#             frame['next'] = frame.ixs.shift(-1)
#             frame['previous'] = frame.ixs.shift()
#             frame = frame.drop(columns='ixs')
#             return frame
#
#
#         self.df = self.df.groupby(level=0,group_keys=False).apply(neighbours)
#
#
#
#     def dump(self,suffix=None,dir=None):
#         """Pickle the object's data for later reuse"""
#
#         if suffix is None:
#             suffix = ''
#
#         dfs = {
#                 'dataset': self.df,
#                 'sequences': self.sequences,
#                 'stages': self.stages
#         }
#
#         keys = [k for k,v in dfs.items() if v is not None]
#
#         for k in keys:
#             fname = f"{k}{suffix}.tsv"
#             if dir is not None:
#                 fname = os.path.join(dir,fname)
#             dfs[k].to_csv(fname, sep='\t')
#             print("Stored " + fname)
#
#
#
#     def sel(self,col,val):
#         """ Retrieve a slice of `self.df` with values where `col` contains `val`, especially for columns with lists.
#         """
#         print('Replace the function dataset.sel() by select_by_values()')
#         selec = self.df[col][self.df[col].notna()]
#         return self.df.loc[selec[selec.apply(lambda x: val in x )].index]
#
#
#     def iterate_sequence_types(self):
#         for ix in self.sequences.index.get_level_values(0).unique():
#             yield ix, self.sequences.loc[ix]
#
#
#     def show(self,piece,ind=None,measures=None,context=0,scoredir=None,ms=None):
#         """ Display the scores of a data entry from the corresponding MSCX file (uncompressed MuseScore 2 format).
#
#         Needs MuseScore 2 installed and callable in the console via `mscore`.
#         Depends on the variable `self.scoredir` that holds the MSCX files.
#
#         Parameters
#         ----------
#
#         piece: tuple or str or int
#             You can indicate the datapoint either by passing its complete index
#             as a tuple, e.g. ('K279-1', 6), or, otherwise, by only indicating the
#             piece and passing the position of the datapoint as `ind`. In that case,
#             `piece` is simply the index of the piece in question, e.g. 'K297-1',
#             or its position, e.g. 0
#         ind: int, optional
#             The position of the data point you want to display. Must be passed
#             if `piece` is not a complete index tuple and `measures` is not passed.
#             Is ignored if you set `measures`.
#         measures: tuple, optional
#             If you pass an integer tuple (start,stop), not the indicated datapoint
#             is displayed, but the measures. Therefore, it is enough to pass
#             `piece` a string.
#         context: int, optional
#             How many measures before and after the point you want to see.
#             Is ignored if you set `measures`.
#         scoredir: str, optional
#             Directory with the MSCX files
#         ms: str, optional
#             Command that starts MuseScore2 on your system.
#             Has to be set on Apple or Windows systems only. If MS2 is installed under
#             the standard path, you can set `ms` to 'win' or 'mac'.
#
#         Example
#         -------
#
#             >>> chords = dataset(chorddir,scoredir)
#             >>> chords.show('K279-1',12,context=1)      # show the 12th datapoint
#             >>> chords.show(('K279-1',12),context=1)    # show the datapoint with index 12
#
#         """
#         if scoredir is not None:
#             self.scoredir = scoredir
#         assert self.scoredir is not None and os.path.isdir(self.scoredir), "Indicate the directory with the scores."
#         df = self.df
#         try:
#             piece = int(piece)
#             piece = df.index.get_level_values(0).unique()[piece]
#         except:
#             pass
#
#             if ind is None:
#                 c = df.loc[piece]
#                 if not c.__class__ == pd.core.series.Series:
#                     if measures is None:
#                         print(f"self.df.loc[{piece}] yields more than 1 rows. You can select one by passing its position as 'ind='.")
#                         return
#                 else:
#                     piece = piece[0]
#             else:
#                 try:
#                     c = df.loc[piece].iloc[ind]
#                 except:
#                     if measures is None:
#                         print(f"Could not show the {ind}th element in {piece}, which has {len(df.loc[piece])} entries.")
#                         return
#
#         if self.parsed_score is None or self.parsed_score.piece != piece+".mscx":
#             nam = os.path.join(self.scoredir,piece+".mscx")
#             self.parsed_score = Piece(nam,ms=ms)
#         else:
#             nam = self.parsed_score.source
#
#
#         if measures is not None and len(measures) == 2:
#             self.parsed_score.get(measures[0],measures[1])
#         else:
#             self.parsed_score.get(c.measure-context,c.measure+context)
#
#
#
#
#     def show_corpus(self):
#         """ Show the names of the pieces contained in the dataset.
#         """
#         return self.df.index.get_level_values(0).unique()
#
#
#
#
#     def show_value_counts(self,columns,per_piece=False,showna=False):
#         """ Show the value counts for one or several columns, aggregated or for each piece.
#
#         Parameters
#         ----------
#
#         columns : {list,tuple} or str
#             Columns of which you want to see the value counts.
#             Type is decisive for the output:
#             - Pass just one column name as a string to see the value counts of one column.
#             - For a slice from `col1` through `col2`, pass a **tuple** (col1,col2).
#             - A list [col1,col2] will only give the two columns.
#             - for len != 2, you can use lists or tuples alike to designate columns.
#         per_piece: bool, optional
#             Show the value_counts() for every piece separately
#         showna: bool, optional
#             Set to True if you want to see the count of NaN as well.
#
#         Examples
#         --------
#
#             >>> merged.show_value_counts('pedal')
#             I     670
#             V     594
#             i      98
#             ii      9
#             IV      6
#             Name: pedal, dtype: int64
#
#             >>> merged.show_value_counts(['pedal'])
#                             pedal
#             feature	values
#             pedal	I	    670
#                     V	    594
#                     i	    98
#                     ii      9
#                     IV      6
#
#             >>> merged.show_value_counts('pedal',True).head()
#                     V   I   i   ii  IV
#             piece
#             K279-1	27.	20.	NaN	NaN	NaN
#             K279-2	NaN	NaN	NaN	NaN	NaN
#             K279-3	13.	3.	NaN	NaN	NaN
#             K280-1	13.	18.	NaN	NaN	NaN
#             K280-2	NaN	NaN	NaN	NaN	NaN
#
#             >>> merged.show_value_counts(['pedal','form'],showna=True)
#                             pedal	form
#             feature	values
#             pedal	NaN	    13692.0	NaN
#                     I	    670.0	NaN
#                     V	    594.0	NaN
#                     i	    98.0	NaN
#                     ii      9.0	    NaN
#                     IV	    6.0	    NaN
#             form	M	    NaN	    10630.0
#                     m	    NaN	    3653.0
#                     o	    NaN	    500.0
#                     NaN	    NaN	    275.0
#                     %	    NaN	    11.0
#
#             >>> merged.show_value_counts(('pedal','form')).head()
#                             pedal   numeral form
#             feature	values
#             form	M       NaN     NaN	    10630.0
#                     m       NaN	    NaN	    3653.0
#                     o       NaN	    NaN	    500.0
#                     %       NaN	    NaN	    11.0
#             numeral	V	    NaN	    5615.0	NaN
#         """
#         if columns.__class__ == tuple and len(columns) == 2:
#             df = self.df.loc[idx[:,columns[0]:columns[1]]]
#         else:
#             df = self.df.loc[:,columns]
#         if columns.__class__ == str:
#             if per_piece:
#                 return df.groupby('piece').apply(lambda col: col.value_counts(dropna=not showna).to_frame().transpose()).droplevel(1)
#             else:
#                 return df.value_counts(dropna=not showna)
#         else:
#
#             if per_piece:
#                 return df.groupby('piece',group_keys=True).apply(lambda f: f.apply(lambda col: pd.concat([col.value_counts(dropna=not showna)], keys=[col.name], names=['feature','values'])))
#                 #return self.df.groupby('piece',group_keys=True).apply(lambda f: f[f.loc[:,columns].notna()].loc[:,columns].value_counts().to_frame().transpose()).droplevel(1)
#             else:
#                 return df.apply(lambda col: pd.concat([col.value_counts(dropna=not showna)], keys=[col.name], names=['feature','values'])).groupby('feature',group_keys=False).apply(lambda f: f.sort_values(f.index[0][0],ascending=False))
#
#
#
#     def get_bigrams(self, index_list, col, infer=False):
#         """ Returns the transition bigrams from self.df for the values of `col` at the given indices.
#         Just input indices of first elements, the following ones are automatically retrieved.
#
#         Parameters
#         ----------
#
#         index_list : list
#             List of index tuples with tuple length 2.
#         col : str
#             Column name from self.df
#         infer : bool, optional
#             If the value of `col` at a given index is NaN and `infer` is True,
#             the previous value which is not NaN is selected.
#
#         """
#
#         def bigrams(row):
#
#             fro = row[col]
#             if isnan(fro):
#                 fro = row.label
#             if infer:
#                 ix = row.name
#                 while isnan(fro):
#                     ix = self.df.loc[ix].previous
#                     if isnan(ix):
#                         return None
#                     fro = self.df.loc[ix][col]
#
#             if not isnan(row.next):
#
#                 nxt = self.df.loc[row.next][col]
#                 ix = row.next
#                 while isnan(nxt) or nxt == fro:
#                     ix = self.df.loc[ix].next
#                     if isnan(ix):
#                         nxt = '∅'
#                         break
#                     nxt = self.df.loc[ix][col]
#
#             else:
#                 nxt = '∅'
#
#             vals = {'from': fro, 'to': nxt}
#             vals['string'] = ' '.join([vals['from'], vals['to']])
#             return pd.Series(vals, name=row.name)
#
#         res = self.df.loc[index_list].apply(bigrams, axis=1)
#         return res[res.to.notnull()]
#
#
#     def instances_from_index(self,index_tuples):
#         """Pass a multiindex as a list of tuples to see all sequences within
#         self.stages corresponding to the values of the first two levels / tuple values."""
#         ix = [t[:2] for t in index_tuples]
#         return select_from_partial_index(self.stages,ix)
#
#
#
#     def instances_by_value(self,col,vals):
#         """ Returns all sequences from `self.stages` in which at least one of the
#         values of `col` is in `vals`.
#
#         Parameters
#         ----------
#
#         col: str
#             Column name of `self.stages`.
#         vals: list or value
#             Values for which you want to see the corresponding sequences.
#
#         Example
#         -------
#
#             >>> instances('bs_max',[0,1])
#
#         """
#         if vals.__class__ != list:
#             vals = [vals]
#         instances = self.stages[self.stages[col].isin(vals)].index.to_list()
#         return self.instances_from_index(instances)
#
#
#
# class merged_dataset(dataset):
#     """ Subclass that merges two datasets together.
#
#     The merge is performed as an outer join on the columns
#     ['piece','measure','beat','position','timesig']. Use `prepare_data.ipynb` to
#     achieve this format.
#
#     The properties of the first dataset are kept.
#
#     Attributes
#     ----------
#
#     ds1, ds2: dataset
#         The two `dataset` objects to be merged.
#     """
#     def __init__(self, data1, data2, suffixes=('_x','_y'), compute_all=False):
#         """
#         Parameters
#         ----------
#
#         data1, data2: dataset
#             The two `dataset` objects to be merged.
#         suffixes: tuple, optional
#             Give the two suffixes appended to columns with identical names
#         """
#         self.ds1, self.ds2 = data1, data2
#         self.df = pd.merge(data1.df,data2.df,on=['piece','measure','beat','position','timesig'],how='outer',suffixes=suffixes)
#         def order(f):
#             return f.sort_values('position').reset_index().drop(columns='piece')
#         self.df = self.df.groupby('piece',sort=False).apply(order)
#         self.df.index.names = ['piece', 'ix']
#         self.dir = data1.dir
#         self.recursive = data1.recursive
#         self.scoredir = data1.scoredir
#         self.sequences = data1.sequences
#         self.parsed_score = None
#         self.sequences = None
#         self.stages = None
#
#         if compute_all:
#             self.expand_labels('label')
#             print('Labels split into separate columns.')
#             self.extract_sequences('cadence',unify_keys=True)
#             print('Extracted all cadence sequences.')
#             self.compute_stages()
#             print('Computed cadence stages.')
#             self.add_stage_info()
#             print('Added stage information to self.df')
#             self.add_neighbours()
#             print('Added neighbour indices to self.df')





def add_stages(frame):
    """ Summarizes subsequent identical chord roots and bass steps.

    """

    terminal = frame[(frame.keyword == 'terminal') | (frame.keyword == 'terminal_inferred')]
    if len(terminal.index) > 1:
        frame['num_stage'] = f'Error: {len(terminal.index)} terminals'
        frame['bs_stage'] = f'Error: {len(terminal.index)} terminals'
        return frame
    elif len(terminal.index) == 0:
        frame['num_stage'] = np.nan
        frame['bs_stage'] = np.nan
        print(f"{frame.index} has to terminal.")
        return frame

    final_harmony_row = terminal.iloc[0]
    term_ix = final_harmony_row.name

    num_stage_col = pd.Series()
    bs_stage_col = pd.Series()
    last_num = final_harmony_row.ext_numeral
    last_bs = final_harmony_row.bass_step
    num_counter, sec_num_counter = 0, 0
    bs_counter, sec_bs_counter = 0, 0
    num_separate = False
    bs_separate = False

    for i,r in frame[:term_ix][::-1].iterrows():

        next_num = r.ext_numeral
        next_bs = r.bass_step

        if next_num == last_num:
            if sec_num_counter == 0 or (i[0] == 'HC' and sec_num_counter == 1 and num_counter == 2): # HC's with only two stages get an extra one
                num_stage = num_counter
            else:
                num_stage = np.nan
            num_stage_col = num_stage_col.append(pd.Series({i:num_stage}, dtype='Int64'))
        else:
            if num_separate:
                sec_num_counter +=1
                if i[0] == 'HC' and sec_num_counter == 1 and num_counter == 1:
                    num_counter += 1
                    num_stage = num_counter
                else:
                    num_stage = np.nan
            else:
                num_counter += 1
                num_stage = num_counter
                if r.numeral in ['I','i'] and not 'I64' in r.chord and num_counter > 0:
                    num_separate = True

            num_stage_col = num_stage_col.append(pd.Series({i:num_stage}, dtype='Int64'))
            last_num = next_num

        if next_bs == last_bs:
            if sec_bs_counter == 0 or (i[0] == 'HC' and sec_bs_counter == 1 and bs_counter == 2):
                bs_stage = bs_counter
            else:
                bs_stage = np.nan
            bs_stage_col = bs_stage_col.append(pd.Series({i:bs_stage}, dtype='Int64'))
        else:
            if bs_separate:
                sec_bs_counter +=1
                if i[0] == 'HC' and sec_bs_counter == 1 and bs_counter == 1:
                    bs_counter += 1
                    bs_stage = bs_counter
                else:
                    bs_stage = np.nan
            else:
                bs_counter += 1
                bs_stage = bs_counter
                if next_bs in ['1','3'] and bs_counter > 0:
                    bs_separate = True

            bs_stage_col = bs_stage_col.append(pd.Series({i:bs_stage}, dtype='Int64'))
            last_bs = next_bs

    frame['num_stage'] =  num_stage_col
    frame['bs_stage'] =  bs_stage_col
    label_order = ['measure','beat','position','duration','timesig','globalkey','key','cadence','cadence_subtype','num_stage','bs_stage','pedal','chord','bass_step','numeral','relativeroot',
               'ext_numeral','form','figbass','changes','phraseend','keyword','label','alt_label','old_key']
    return frame[label_order]































def inspect_cadences(ds,cad=None,instances=None,piece=None,context=None,musescore=None):
    """ Iterate through the sequences and scores of `instances` of type `cad` and/or those contained in `piece`.

    Parameters
    ----------
    ds: dataset
        A `dataset` object with stored subsequences in `ds.sequences`
    cad: {'PAC','HC','IAC','DEC','EVCAD'}, optional
        Pass if you want to see only one certain type.
    instances: collection or int, optional
        The instances (integers) you want to see. Pass a list with individual values
        or a 2-tuple (from, to) for a slice. If you pass an int, the cadences is
        displayed right away without having to iterate through the generator object.
    piece: str, optional
        Pass if you only want to see cadences from one piece.
    context: int, optional
        If you pass an int, only the sequence's last point with a
        context of `context` is displayed as a score. Otherwise,
        the entire sequence.

    Returns
    -------

    A generator to iterate through the cadences.
    """

    def inspect(df):
        for i, section in df.groupby(level=[0,1]):
            #print(section.loc[:,idx['measure':'label']])
            ix = section.index[-1][2:]
            yield ds.show(ix,context=context,ms=musescore)

    def show(df):
        for i, section in df.groupby(level=[0,1]):
            #print(section.loc[:,idx['measure':'label']])
            ix = section.index[-1][2:]
            ds.show(ix,context=context,ms=musescore)

    def inspect_sections(df):
        for i, section in df.groupby(level=[0,1]):
            #print(section.loc[:,idx['measure':'label']])
            measures = (section.measure.min(),section.measure.max())
            yield ds.show(section.index[0][2],ind=section.index[-1][3],measures=measures,ms=musescore)

    def show_section(df):
        for i, section in df.groupby(level=[0,1]):
            #print(section.loc[:,idx['measure':'label']])
            measures = (section.measure.min(),section.measure.max())
            ds.show(section.index[0][2],ind=section.index[-1][3],measures=measures,ms=musescore)


    piece = SM[:] if piece is None else SM[piece]

    if instances is None:
        inst = SM[:]
    elif instances.__class__ == tuple and len(instances) == 2:
        inst = SM[instances[0]:instances[1]]
    elif instances.__class__ == list and instances[0].__class__ == tuple and len(instances[0]) == 2:
        return inspect_sections(select_from_partial_index(ds.stages,instances)) if context is None else inspect(select_from_partial_index(ds.stages,instances))
    elif instances.__class__ == slice:
        inst = instances
    else:
        inst = SM[instances]

    cad = SM[:] if cad is None else SM[cad]

    if context is None:
        if instances.__class__ == int:
            show_section(ds.stages.loc[idx[cad,inst,piece],])
        G = inspect_sections(ds.stages.loc[idx[cad,inst,piece],])
    else:
        if instances.__class__ == int:
            show(ds.stages.loc[idx[cad,inst,piece],])
        G = inspect(ds.stages.loc[idx[cad,inst,piece],])

    return G





def nest_level(obj, include_tuples=False):
    """Recursively calculate the depth of a nested list.

    """
    if obj.__class__ != list:
        if include_tuples:
            if obj.__class__ != tuple:
                return 0
        else:
            return 0
    max_level = 0
    for item in obj:
        max_level = max(max_level, nest_level(item, include_tuples=include_tuples))
    return max_level + 1


def grams(l, n=2):
    """Returns a list of n-gram tuples for given list. List can be nested.

    Use nesting to exclude transitions between pieces or other units.

    """
    if nest_level(l) > 1:
        ngrams = []
        no_sublists = []
        for item in l:
            if isinstance(item,list):
                ngrams.extend(grams(item,n))
            else:
                no_sublists.append(item)
        if len(no_sublists) > 0:
            ngrams.extend(grams(no_sublists,n))
        return ngrams
    else:
        #if len(l) < n:
        #    print(f"{l} is too small for a {n}-gram.")
        ngrams = [l[i:(i+n)] for i in range(len(l)-n+1)]
        # convert to tuple of strings
        return [tuple(str(g) for g in gram) for gram in ngrams]



def ngram_string(df, col, ixs=None, remove_duplicates=True, sep=' ', slic=SL[:]):
    """ Creates a string from the values in `col` within the given DataFrame.

    Parameters
    ----------

    df : DataFrame
        DataFrame including the column name passed as `col`.
        If you pass index tuples to `ixs`, df needs to have the same number of
        index levels as the tuple length.
    col : str
        Column name from which you want to create the string.
    ixs : list or 2-tuple or dictionary, optional
        Pass a list or a list of lists/tuples with the indices of the values you want to concatenate. Every list will become one string.
        If you pass a tuple with two index tuples, it will be interpreted as a slice from the first index to the last (inclusive).
        The same applies if you pass a dictionary of lists/tuples. Then, the return value will be a dictionary of strings.
    remove_duplicates : bool, optional
        By default, this function filters out subsequent identical values. Pass False to prevent that.
    sep : str, optional
        The string dividing the values. Defaults to ' '
    slic: slice or SliceMaker, optional
        If `df`'s multtindex has more levels than the length of the index tuples in `ixs`, pass a slice or SliceMaker object to
        select the multtindex levels that you want to match.
    """

    def return_col(val):
        #print(val)
        if remove_duplicates:
            val = val[val != val.shift()]
        return val.str.cat(sep=sep)


    if ixs == None:
        return return_col(df[col])


    if ixs.__class__ == dict:
        return {k: ngram_string(df,col,ixs=v,remove_duplicates=remove_duplicates,sep=sep) for k,v in ixs.items()}

    L = []
    nesting = nest_level(ixs,include_tuples=True)
    if nesting <= 1:
        raise Exception('Only works with a list of index tuples or a (list of) list/tuple of index tuples.')
    elif nesting > 2:
        for subcollection in ixs:
            recurse = ngram_string(df, col, ixs=subcollection, remove_duplicates=remove_duplicates, sep=sep)
            if nest_level(recurse) == 0:
                L.append(recurse)
            else:
                L.extend(recurse)
        return L
    else:
        if ixs.__class__ == tuple and len(ixs) == 2:
            if slic != SL[:]:
                print("""Multiindex slicing has not been implemented for (from, to) tuples. You can either pass
                the complete list of index tuples you want to select or drop the superfluous levels of df.""")
                return
            else:
                return return_col(df[col].loc(axis=0)[ixs[0]:ixs[1]])
        else:
            if slic == SL[:]:
                return return_col(df[col].loc(axis=0)[ixs])
            else:
                return return_col(select_from_partial_index(df[col],ixs,slic))



def transition_matrix(l=None, gs=None, n=2, smooth=0, normalize=False, IC=False, filt=None, dist_only=False,sort=False, decimals=None):
    """Returns a transition table from a list of symbols.

    Column index is the last item of grams, row index the n-1 preceding items.

    Parameters
    ----------

    l: list, optional
        List of elements between which the transitions are calculated.
        List can be nested.
    gs: list, optional
        List of tuples being n-grams
    n: int, optional
        get n-grams
    smooth: number, optional
        initial count value of all transitions
    normalize: bool, optional
        set True to divide every row by the sum of the row.
    IC: bool, optional
        Set True to calculate information content.
    filt: list, optional
        elements you want to exclude from the table. All ngrams containing at least one
        of the elements will be filtered out.
    dist_only: bool, optional
        if True, n-grams consisting only of identical elements are filtered out
    """
    if gs is None:
        assert (n>0), f"Cannot print {n}-grams"
        gs = grams(l, n=n)
    elif l is not None:
        assert True, "Specify either l or gs, not both."

    if filt:
        gs = list(filter(lambda n: not any(g in filt for g in n),gs))
    if dist_only:
        gs = list(filter(lambda tup: any(e != tup[0] for e in tup),gs))
    ngrams = pd.Series(gs).value_counts()
    ngrams.index = [(' '.join(t[:-1]),t[-1]) for t in ngrams.index.tolist()]
    context = pd.Index(set([ix[0] for ix in ngrams.index]))
    consequent = pd.Index(set([ix[1] for ix in ngrams.index]))
    df = pd.DataFrame(smooth, index=context, columns=consequent)


    for i, (cont, cons) in enumerate(ngrams.index):
        try:
            df.loc[cont, cons] += ngrams[i]
        except:
            continue

    if sort:
        h_sort = list(df.max().sort_values(ascending= False).index.values)
        v_sort = list(df.max(axis=1).sort_values(ascending= False).index.values)
        df = df[h_sort].loc[v_sort]

    SU = df.sum(axis=1)

    if normalize or IC:
        df = df.div(SU,axis=0)



    if IC:
        ic = np.log2(1/df)
        ic['entropy'] = (ic * df).sum(axis=1)
        ############## Identical calucations:
        #ic['entropy2'] = scipy.stats.entropy(df.transpose(),base=2)
        #ic['entropy3'] = -(df * np.log2(df)).sum(axis=1)
        df = ic
        if normalize:
            df['entropy'] = df['entropy'] / np.log2(len(df.columns)-1)
    else:
        df['total'] = SU

    if decimals is not None:
        df = df.round(decimals)

    return df



def stage_lengths(stages,cad=None,kind='pie'):
    """ Learn about how many cadences of type `cad` have how many stages.

    Parameters
    ----------

    stages: DataFrame
        Multiindex with levels (cad_type, instance). Columns are stages.
    cad: {'PAC','HC','IAC','DEC','EVCAD'}, optional
        Cadence type. If None, show the ensemble.
    kind: any
        If kind is not 'pie', a bar chart is plotted.

    """
    if cad is None:
        df = stages.drop('beyond',axis=1,level=0)
    else:
        df = stages.loc[cad].drop('beyond',axis=1,level=0)
    counts = defaultdict(lambda: 0)
    ixs = defaultdict(list)
    for i,r in df.iterrows():
        n = r.dropna().groupby(level=0).ngroups
        counts[n] += 1
        ixs[n].append(i)

    counts = pd.DataFrame(counts.items(),columns=['stages','counts'])
    if kind == 'pie':
        counts.iplot(kind='pie',labels='stages',values='counts',textposition='outside',textinfo='value+percent')
    else:
        counts.set_index('stages').iplot(kind='bar')
    return dict(ixs)




def inspect_stage_divisions(ds, stages,cad=None,instances=None,piece=None,context=None,musescore=None):
    """ Iterate through the stages and scores of cadences of type `cad`.


    Parameters
    ----------
    ds: dataset
        Dataset from which to fetch the music extracts.
    stages: DataFrame
        Multiindex with levels (cad_type, instance). Columns are stages.
    cad: {'PAC','HC','IAC','DEC','EVCAD'}
        Pass for which type you want to inspect the stages.
    instances: collection, optional
        The instances (integers) you want to see. Pass a list with individual values
        or a tuple (from, to) for a slice.
    piece: str, optional
        Pass if you only want to see cadences from one piece.
    context: int, optional
        If you pass an int, only the sequence's last point with a
        context of `context` is displayed as a score. Otherwise,
        the entire sequence.
    """

    def inspect(df):
         for i,r in df.iterrows():
            section = ds.sequences.loc[(cad,i[1])]
            d = pd.DataFrame(r.dropna()).transpose().sort_index(axis=1,level=0)
            display(d)
            ix = section.index[-1]
            yield ds.show(ix,context=context,ms=musescore)


    def show(df):
         for i,r in df.iterrows():
            section = ds.sequences.loc[(cad,i[1])]
            d = pd.DataFrame(r.dropna()).transpose().sort_index(axis=1,level=0)
            display(d)
            ix = section.index[-1]
            ds.show(ix,context=1,ms=musescore)


    def inspect_sections(df):
        for i,r in df.iterrows():
            section = ds.sequences.loc[(cad,i[1])]
            d = pd.DataFrame(r.dropna()).transpose().sort_index(axis=1,level=0)
            display(d)
            measures = (section.measure.min(),section.measure.max())
            yield ds.show(section.index[0][0],section.index[-1][1],measures=measures,ms=musescore)


    def show_section(df):
        for i,r in df.iterrows():
            section = ds.sequences.loc[(cad,i[1])]
            d = pd.DataFrame(r.dropna()).transpose().sort_index(axis=1,level=0)
            display(d)
            measures = (section.measure.min(),section.measure.max())
            ds.show(section.index[0][0],ind=section.index[-1][1],measures=measures,ms=musescore)




    cad = SM[:] if cad is None else SM[cad]
    piece = SM[:] if piece is None else SM[piece]

    if instances is None:
        inst = SM[:]
    elif instances.__class__ == tuple and len(instances) == 2:
        inst = SM[instances[0]:instances[1]]
    elif instances.__class__ == list and instances[0].__class__ == tuple and len(instances[0]) == 2:
        cad, inst = list(zip(*instances))
    else:
        inst = SM[instances]


    if context is None:
        if instances.__class__ == int:
            show_section(stages.loc[idx[cad,inst,piece],])
        G = inspect_sections(stages.loc[idx[cad,inst,piece],])
    else:
        if instances.__class__ == int:
            show(stages.loc[idx[cad,inst,piece],])
        G = inspect(stages.loc[idx[cad,inst,piece],])

    return G
