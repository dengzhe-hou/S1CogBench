"""Original Tomov key-event replay, separate from the damaged Psych transcript.

Targets are original recorded keyCodes, never inferred from future stations.
Source paths only verify a forward replay of the published experiment program.
"""
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import dataclass
from io import StringIO
import json
import math
from pathlib import Path
import re
import statistics
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCES=ROOT/'literature/subway_mapping_20261017'
ARCHIVE=SOURCES/'chunking_master.zip'
MATCHES=SOURCES/'resolved_original_sources.json'
KEYS=('38','37','40','39','32')
KEY_INDEX={39:0,38:1,37:2,40:3}
KEY_NAMES={'38':'up','37':'left','40':'down','39':'right','32':'space'}
VERSION_BY_DIRECTORY={
    'subway9':'exp_v1_3','subway9_control':'exp_v1_4','subway9_goodchunks':'exp_v1_5',
    'subway10_map':'exp_v3_2','subway9_map':'exp_v3_3',
    'subway9_map_control_1':'exp_v3_4','subway9_map_control_2':'exp_v3_5',
    'subway9_map_goodchunks':'exp_v3_6','mines10_map':'exp_v4',
}
EXPERIMENTS=('tomov2020discovery/exp2.csv','tomov2020discovery/exp4.csv',
             'tomov2020discovery/exp5.csv','tomov2020discovery/exp7.csv')
PROTOCOL_VERSION='tomov-original-key-events-symbolic-text-v1'


def raw_sessions():
    """Use already verified unique original records, preserving all aliases."""
    report=json.loads(MATCHES.read_text())
    with zipfile.ZipFile(ARCHIVE) as archive:
        for match in report['sessions']:
            if match['distinct_full_original_key_path_sequences']!=1:
                raise ValueError('Original record has not been uniquely resolved.')
            path=next((x for x in match['origin_files'] if '/ARCHIVE/' not in x),None)
            if path is None:
                raise ValueError('Need the explicit original condition directory, not an inferred ARCHIVE condition.')
            directory=path.split('/')[-2];version=VERSION_BY_DIRECTORY[directory]
            rows=list(csv.DictReader(StringIO(archive.read(path).decode('utf8')),skipinitialspace=True))
            trials=[]
            for index,row in enumerate(rows):
                trials.append({'raw_trial_index':index,'start':int(row['start']),'goal':int(row['goal']),
                    'path':list(map(int,row['path'].split())),'keys':list(map(int,row['keys'].split())),
                    'stage':row['stage'].strip(),'human_terminal_reward':float(row['reward']) if row.get('reward') else None,
                    'raw_record':row})
            html=archive.read('chunking-master/exp/'+version+'.html').decode('utf8')
            js=archive.read('chunking-master/exp/'+version+'.js').decode('utf8')
            template=re.search(r'<textarea[^>]*id="experiment"[^>]*>(.*?)</textarea>',html,re.S)
            if template is None:raise ValueError('Original task HTML has no experiment template.')
            lines=template.group(1).strip().splitlines();n=int(lines[0])
            numbers=[list(map(float,line.split())) for line in lines[1:n+1]]
            adjacency=[list(map(int,row[:4])) for row in numbers]
            coordinates=[row[4:6] for row in numbers] if len(numbers[0])>=6 else None
            yield {'experiment':match['experiment'],'participant':match['participant'],
                'source_row':match['source_row'],'source_path':path,'source_aliases':match['origin_files'],
                'condition':directory,'version':version,'trials':trials,'adjacency_template':adjacency,
                'coordinate_template':coordinates,'html':html,'js':js,
                'visible_reward_vectors_recorded':False if version=='exp_v4' else None}


def candidate_layouts(session):
    adjacency=session['adjacency_template'];coords=session['coordinate_template']
    # These are exactly the alternatives implemented by the corresponding JS.
    if session['version'].startswith('exp_v1_'):
        for rotation in range(4):
            yield {'rotation':rotation,'flip':False,'adjacency':[[row[(j+rotation)%4] for j in range(4)] for row in adjacency],
                   'coordinates':None}
    else:
        for flip in (False,True):
            adj=deepcopy(adjacency);xy=deepcopy(coords)
            if flip:
                for row in adj:row[0],row[2]=row[2],row[0]
                maximum=max(x[0] for x in xy)
                for point in xy:point[0]=maximum-point[0]
            yield {'rotation':None,'flip':flip,'adjacency':adj,'coordinates':xy}


def replay_trial(trial,adjacency,collect=True):
    """Exact key ordering and state update from checkKeyPressed; no path lookup."""
    current=trial['start'];path=[current];ended=False;events=[]
    for index,key in enumerate(trial['keys']):
        if ended:return {'valid':False,'reason':'key_recorded_after_termination','events':events}
        before=current;next_station=adjacency[current-1][KEY_INDEX[key]] if key in KEY_INDEX else -1
        feedback='no_change';kind='other_key'
        if next_station>=0:
            path.append(next_station);kind='move';feedback='moved'
            # The test branch logs the intended next station without updating
            # cur. The display stays at the start until the probe is interrupted.
            if trial['stage']=='train':current=next_station
        elif key in KEY_INDEX:
            kind='blocked_direction';feedback='no_neighbor'
        if trial['stage']=='train':
            if key==32:
                kind='finish_attempt'
                if trial['goal']==-1 or current==trial['goal']:
                    ended=True;feedback='reward_received' if trial['human_terminal_reward'] is not None else 'success'
                else:feedback='incorrect_goal'
        elif trial['stage']=='test':
            if next_station>=0:ended=True;feedback='probe_interrupted'
        else:raise ValueError('Unsupported original stage, not silently converted.')
        if collect:
            events.append({'key_index_in_raw_trial':index,'original_key_code':key,'current_station_before':before,
                'station_after':current,'event_kind':kind,'feedback_after':feedback,'trial_ended_after':ended,
                'logged_path_destination':next_station if next_station>=0 else None,
                'human_source_key_in_declared_actions':str(key) in KEYS,
                'human_terminal_reward_after':trial['human_terminal_reward'] if ended else None})
    return {'valid':ended and path==trial['path'],'path_match':path==trial['path'],'ended':ended,
            'replayed_path':path,'events':events}


def resolve_layout(session):
    matches=[];diagnostic=[]
    for layout in candidate_layouts(session):
        checks=[replay_trial(t,layout['adjacency'],False) for t in session['trials']]
        good=sum(x['valid'] for x in checks)
        diagnostic.append({'rotation':layout['rotation'],'flip':layout['flip'],
            'valid_original_trials':good,'original_trials':len(checks)})
        if good==len(checks):matches.append(layout)
    if len(matches)!=1:
        return {'status':'unresolved_original_layout','candidates':diagnostic,'matching_layout_count':len(matches)}
    return {'status':'unique_all_event_forward_replay','layout':matches[0],'candidates':diagnostic,
        'provenance':'Original JS enumerated static randomization, selected only if all recorded keys forward-replay to every raw path and boundary. No target is inferred from a next station.'}


def reconstruct_session(session):
    resolved=resolve_layout(session)
    result={'experiment':session['experiment'],'participant':session['participant'],'source_row':session['source_row'],
            'source_path':session['source_path'],'source_aliases':session['source_aliases'],'condition':session['condition'],
            'version':session['version'],'raw_trials':len(session['trials']),
            'raw_key_events':sum(len(t['keys']) for t in session['trials']),'layout_resolution':resolved,
            'symbolic_identifier_adaptation':'Original station-name permutation was not logged; use stable source node IDs and state this textual adaptation.',
            'events':[]}
    if resolved['status']!='unique_all_event_forward_replay':
        result['input_status']='blocked_layout';return result
    layout=resolved['layout']
    for trial in session['trials']:
        replay=replay_trial(trial,layout['adjacency'])
        if not replay['valid']:raise AssertionError('A previously validated original trial no longer replays.')
        for event in replay['events']:
            result['events'].append({**event,'raw_trial_index':trial['raw_trial_index'],'start':trial['start'],
                'goal':trial['goal'],'stage':trial['stage'],'free_choice':trial['goal']==-1,
                'visible_reward_vector':None if session['version']=='exp_v4' else 'not_applicable'})
    result['input_status']='blocked_missing_visible_reward_vectors' if session['version']=='exp_v4' else 'ready_symbolic_text_adaptation'
    return result


@dataclass(frozen=True)
class SubwayTrial:
    experiment: str
    participant: object
    source_row: int
    trial_index: int
    raw_trial_index: int
    raw_key_index: int
    state: str
    original_choice_label: str
    source_event: dict

    task='subway_planning'
    output_keys=KEYS

    @property
    def session_id(self):return f'{PROTOCOL_VERSION}:{self.experiment}:row={self.source_row}'
    @property
    def id(self):return f'{self.session_id}:trial={self.raw_trial_index}:key={self.raw_key_index}'
    @property
    def prefix_end(self):return len(self.state)
    @property
    def question(self):
        return {'type':'choice','instructions':'Which instructed key do you press next?',
                'criteria':{k:f'Press {KEY_NAMES[k]} (keyCode {k}).' for k in KEYS}}

    def identifiers(self):
        return {'id':self.id,'protocol_version':PROTOCOL_VERSION,'task':self.task,
            'experiment':self.experiment,'participant':self.participant,'source_row':self.source_row,
            'session_id':self.session_id,'trial_index':self.trial_index,
            'raw_trial_index':self.raw_trial_index,'raw_key_index':self.raw_key_index,
            'prefix_end_character':len(self.state),'output_keys':list(KEYS),
            'original_choice_label':self.original_choice_label,
            'source_event_for_scoring_only':self.source_event,
            'human_source_key_in_declared_actions':self.original_choice_label in KEYS}


def session_inputs(session,reconstructed=None):
    """Complete human-history input. Future keys/path/RTs are scoring-only."""
    r=reconstructed or reconstruct_session(session)
    if r['input_status']!='ready_symbolic_text_adaptation':
        raise ValueError(f"Cannot invent unrecorded visible reward vectors: {session['source_path']}")
    layout=r['layout_resolution']['layout'];adj=layout['adjacency']
    no_map=session['version'].startswith('exp_v1_')
    history=('You travel through subway stations. Reach each specified goal using arrow keys, '
        'then press space. Keys: 38=up, 37=left, 40=down, 39=right, 32=space. '
        'You can move only along a displayed connection. A direction without a connection does nothing. '
        'Pressing space away from the goal displays Incorrect. '
        'Station IDs are stable. The record below retains every preceding key press and observation.\n')
    if no_map:
        history+='Plan your trip as quickly as possible. Only the current station, its directly connected neighbors and the goal are displayed.\n'
    else:
        history+='The following complete map remains visible. Coordinates give the original map positions.\n'
        for node,(row,xy) in enumerate(zip(adj,layout['coordinates']),1):
            neighbors=', '.join(f'{KEY_NAMES[str(key)]}={row[index] if row[index]>=0 else "none"}' for key,index in KEY_INDEX.items())
            history+=f'Station {node}: position=({xy[0]:g},{xy[1]:g}); {neighbors}.\n'
    stated_count=re.search(r'There will be (\d+) trials',session['html'])
    if stated_count:history+=f'The instructions state there will be {stated_count.group(1)} trials.\n'
    events_by_trial={}
    for event in r['events']:events_by_trial.setdefault(event['raw_trial_index'],[]).append(event)
    output=[];response_index=0
    for trial in session['trials']:
        if no_map and trial['stage']=='test':
            history+='Unreliable trips: an accident has occurred; remaining trips may be interrupted and a new trial may begin automatically.\n'
        history+=f"Trip {trial['raw_trial_index']+1}: start={trial['start']}; goal={trial['goal']}.\n"
        for event in events_by_trial[trial['raw_trial_index']]:
            before=event['current_station_before']
            history+=f'Current={before}; goal={trial["goal"]}'
            if no_map:
                row=adj[before-1]
                for key,index in KEY_INDEX.items():history+=f'; {KEY_NAMES[str(key)]}={row[index] if row[index]>=0 else "none"}'
            history+='. You press '
            source_event={**event,'source_path':session['source_path'],'condition':session['condition'],
                          'version':session['version'],'original_raw_trial_record':trial['raw_record']}
            output.append(SubwayTrial(session['experiment'],session['participant'],session['source_row'],
                response_index,trial['raw_trial_index'],event['key_index_in_raw_trial'],history,
                str(event['original_key_code']),source_event))
            response_index+=1
            # Human actions, not model guesses, determine the next prediction input.
            history+=f"<<{event['original_key_code']}>>. "
            feedback=event['feedback_after']
            if feedback=='moved':history+=f"Current becomes {event['station_after']}.\n"
            elif feedback=='incorrect_goal':history+='Incorrect. Current station unchanged.\n'
            elif feedback=='success':history+='SUCCESS!! Trip ended.\n'
            elif feedback=='probe_interrupted':history+='Trip interrupted; experiment ended.\n'
            elif feedback in ('no_change','no_neighbor'):history+='No displayed change.\n'
            else:raise ValueError(f'Unexpected subway feedback: {feedback}')
    if len(output)!=r['raw_key_events']:raise AssertionError('An original key event was dropped.')
    return output


def load_trials(experiments=EXPERIMENTS):
    requested=set(experiments)
    if not requested or not requested.issubset(EXPERIMENTS):raise ValueError('Explicit original experiment selection required.')
    result=[];found=set()
    for session in raw_sessions():
        if session['experiment'] in requested:
            result.extend(session_inputs(session));found.add(session['experiment'])
    if found!=requested:raise ValueError('Requested original experiment is absent.')
    return result


def inventory(trials):
    sessions={}
    for t in trials:
        entry=sessions.setdefault(t.session_id,{'source_row':t.source_row,'experiment':t.experiment,
            'participant':t.participant,'responses':0,'source_path':t.source_event['source_path']})
        entry['responses']+=1
    return {'task':'subway_planning','protocol_version':PROTOCOL_VERSION,'sessions':len(sessions),
        'responses':len(trials),'session_counts':sessions,'experiments':sorted({t.experiment for t in trials}),
        'source_key_counts':dict(Counter(t.original_choice_label for t in trials)),
        'source_keys_outside_declared_actions':sum(t.original_choice_label not in KEYS for t in trials),
        'maximum_full_prefix_characters':max(map(lambda t:len(t.state),trials)),
        'split':'Original Psych-101-test participant identities mapped to original Tomov raw sessions; no resplitting.',
        'history':'Complete original key-event sequence and visible local neighbors/full map, symbolic station-ID text adaptation; no history truncation.',
        'adaptations':['Symbolic node IDs replace unlogged random city-name labels.',
            'Text replaces the original visual/spatial interface and does not emulate display timing.',
            'All raw key events retained, including keys outside the five instructed actions.',
            'Typed human-response prediction, not original full-vocabulary Psych-101 likelihood.'],
        'whole_subway_task_status':'incomplete_exp7_visible_reward_vectors_unrecorded',
        'original_damaged_psych_transcript':'Preserved, not overwritten or rescored as repaired.',
        'model_closed_loop_reward':None,'latency_interpretation':'Shared-device runtime, not human reaction time.'}


def score(trial,prediction):
    probs=[float(p) for p in prediction['probabilities']]
    if len(probs)!=len(KEYS) or any(not math.isfinite(p) or p<0 or p>1 for p in probs) or abs(sum(probs)-1)>1e-7:
        raise ValueError('Expected normalized finite probabilities over the five instructed keys.')
    label=trial.original_choice_label
    p=probs[KEYS.index(label)] if label in KEYS else 0.0
    native=prediction.get('native_choice');choice=native if native is not None else KEYS[max(range(len(KEYS)),key=probs.__getitem__)]
    row=trial.identifiers()
    row.update(probabilities=dict(zip(KEYS,probs)),choice=choice,native_choice=native,
        invalid_response=native is not None and native not in KEYS,
        matches_human_choice=choice==label,human_choice_probability=p,
        typed_choice_nll=-math.log(p) if p else None,typed_choice_nll_is_infinite=p==0,
        source_outside_action_target_semantics='Observed outside-instruction key gets zero mass from a five-action model; retain event and infinite all-event NLL.',
        model_closed_loop_reward=None,original_protocol_nll=None)
    for k in ('raw_probabilities','input_tokens','allowed_token_mass','label_mapping','batch_id','batch_size','batch_elapsed_seconds','inference_interface'):
        if k in prediction:row[k]=prediction[k]
    return row


def summarize(rows,expected):
    def metrics(group):
        zeros=sum(r['typed_choice_nll_is_infinite'] for r in group)
        return {'responses':len(group),'typed_choice_mean_nll':None if not group or zeros else statistics.mean(r['typed_choice_nll'] for r in group),
            'typed_choice_mean_nll_is_infinite':bool(zeros),'zero_human_probability_count':zeros,
            'human_response_agreement':statistics.mean(r['matches_human_choice'] for r in group) if group else None,
            'invalid_response_rate':statistics.mean(r['invalid_response'] for r in group) if group else None}
    groups={}
    for row in rows:groups.setdefault(row['experiment'],[]).append(row)
    legal=[r for r in rows if r['human_source_key_in_declared_actions']]
    times=sorted(r['batch_elapsed_seconds'] for r in rows)
    return {'status':'complete' if len(rows)==expected else 'partial',**metrics(rows),'expected_responses':expected,
        'completion_scope':'Selected original subexperiments, not complete four-experiment Subway task.',
        'whole_subway_task_status':'incomplete_exp7_visible_reward_vectors_unrecorded',
        'by_experiment':{k:metrics(v) for k,v in groups.items()},
        'conditional_on_instructed_source_keys_diagnostic':metrics(legal),
        'all_source_events_retained':True,'source_keys_outside_declared_actions':len(rows)-len(legal),
        'latency_p50_seconds':statistics.median(times) if times else None,
        'latency_p95_seconds':times[math.ceil(.95*len(times))-1] if times else None,
        'model_closed_loop_reward':None,'timing':'Shared-device scalar wall time, not human reaction time.',
        'original_protocol_nll':None,'original_protocol_nll_status':'Not the damaged original Psych transcript or full-vocabulary scoring.'}
