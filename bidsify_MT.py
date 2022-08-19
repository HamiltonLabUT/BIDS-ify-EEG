import mne
import os
import numpy as np
import pandas as pd
import glob
import re
from mne_bids import write_raw_bids, BIDSPath, print_dir_tree
from mne_bids.stats import count_events

# subjects
data_dir = '/Users/liberty/Library/CloudStorage/Box-Box/MovieTrailersTask/Data/EEG/Participants'
output_path = '/Users/liberty/Library/CloudStorage/Box-Box/MovieTrailersTask/Data/EEG/bids_dataset'

eeg_files = np.sort(glob.glob(f'{data_dir}/*/downsampled_128/*DS128.vhdr'))
run = 1

event_id = {'TIMIT': 1, 'MT': 2, 'VisualOnlyMT': 3,
            'AuditoryOnlyMT': 4}

for eeg_file in eeg_files:
	subject = re.findall('MT[0-9]+',eeg_file)[0]
	block = re.findall('B[0-9]+', eeg_file)

	run = 1
	# if len(block) > 0:
	# 	run = int(re.findall('[0-9]+', block[0])[0])
	# else:
	# 	run = 1
	print(f'Getting data from {eeg_file}, subject: {subject}, run: {run}')

	raw = mne.io.read_raw_brainvision(eeg_file, preload=True)
	raw_preproc = mne.io.read_raw_fif(f'{data_dir}/{subject}/downsampled_128/{subject}_rejection_mas_raw.fif', preload=False)
	if len(raw_preproc.info['bads']) > 0:
		print('Bad channels:')
		print(raw_preproc.info['bads'])
		raw.info['bads'] = raw_preproc.info['bads']

	raw.info['line_freq'] = 60

	raw.plot_sensors(show_names=True)
	raw.set_channel_types({'hEOG': 'eog', 'vEOG': 'eog'})
	raw.set_eeg_reference(ref_channels=['TP9','TP10'])

	raw.plot_psd()
	raw.plot(scalings='auto', block=True)

	mt_event = f'{data_dir}/{subject}/audio/{subject}_MovieTrailers_events.txt'
	mt_ev = pd.read_csv(mt_event, delim_whitespace=True, header=None, names=['onset','offset','event_id','name'])
		
	timit_event = f'{data_dir}/{subject}/audio/{subject}_TIMIT_all_events.txt'
	timit_ev = pd.read_csv(timit_event, delim_whitespace=True, header=None, names=['onset','offset','event_id','name'])
	
	all_ev = mt_ev.append(timit_ev)
	annotations = mne.Annotations(all_ev.onset, all_ev.offset-all_ev.onset, all_ev.name)

	raw.set_annotations(annotations)
	
	bids_path = BIDSPath(subject=subject, session='01',
                         task='MovieTrailersTIMIT', run=f'{run:02d}', root=output_path,
                         datatype='eeg')

	pos = raw.get_montage().get_positions()
	biosem=mne.channels.make_standard_montage('biosemi64')

	new_montage = mne.channels.make_dig_montage(
    	ch_pos=pos['ch_pos'],
    	nasion=biosem.get_positions()['nasion'],
    	lpa=biosem.get_positions()['lpa'],
    	rpa=biosem.get_positions()['rpa'],
    	hsp=None,
    	hpi=None,
    	coord_frame='head',
	)	
	raw.set_montage(new_montage)

	write_raw_bids(raw, bids_path, allow_preload=True, format='BrainVision', overwrite=True)
