class Config(object):
    corpus_splitting = 1            # 1 for Lin and 2 for Ji
    max_sent_len = 80
    
    wordvec_path = '~/Projects/GoogleNews-vectors-negative300.bin.gz'
    wordvec_dim = 300


    i2sense = [
        'Temporal.Asynchronous', 'Temporal.Synchrony', 'Contingency.Cause',
        'Contingency.Pragmatic cause', 'Comparison.Contrast', 'Comparison.Concession',
        'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Restatement',
        'Expansion.Alternative','Expansion.List'
    ]
    sense2i = {
        'Temporal.Asynchronous':0, 'Temporal.Synchrony':1, 'Contingency.Cause':2,
        'Contingency.Pragmatic cause':3, 'Comparison.Contrast':4, 'Comparison.Concession':5,
        'Expansion.Conjunction':6, 'Expansion.Instantiation':7, 'Expansion.Restatement':8,
        'Expansion.Alternative':9,'Expansion.List':10
    }
    i2senseclass = ['Temporal', 'Contingency', 'Comparison', 'Expansion']
    senseclass2i = {'Temporal':0, 'Contingency':1, 'Comparison':2, 'Expansion':3}