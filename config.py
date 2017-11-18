import torch
from datetime import datetime

class Config(object):
    i2sensesecond = [
        'Temporal.Asynchronous', 'Temporal.Synchrony', 'Contingency.Cause',
        'Contingency.Pragmatic cause', 'Comparison.Contrast', 'Comparison.Concession',
        'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Restatement',
        'Expansion.Alternative','Expansion.List'
    ]
    sense2isecond = {
        'Temporal.Asynchronous':0, 'Temporal.Synchrony':1, 'Contingency.Cause':2,
        'Contingency.Pragmatic cause':3, 'Comparison.Contrast':4, 'Comparison.Concession':5,
        'Expansion.Conjunction':6, 'Expansion.Instantiation':7, 'Expansion.Restatement':8,
        'Expansion.Alternative':9,'Expansion.List':10
    }
    i2senseclass = ['Temporal', 'Contingency', 'Comparison', 'Expansion']
    senseclass2i = {'Temporal':0, 'Contingency':1, 'Comparison':2, 'Expansion':3}

    four_or_eleven = 11
    if four_or_eleven == 4:
        i2sense = i2senseclass
        sense2i = senseclass2i
        sense_idx = 1
    elif four_or_eleven == 11:
        i2sense = i2sensesecond
        sense2i = sense2isecond
        sense_idx = 0

    corpus_splitting = 2            # 1 for Lin and 2 for Ji
    max_sent_len = 80

    wordvec_path = '~/Projects/GoogleNews-vectors-negative300.bin.gz'
    wordvec_dim = 300

    nonlinear = torch.nn.Tanh()

    embed_dropout = 0.1
    sent_repr_dim = 256
    conv_filter_set_num = 3
    conv_kernel_size = [2, 4, 8]

    arg_rep_dim = sent_repr_dim * conv_filter_set_num * 2
    arg_encoder_fc_num = 0
    arg_encoder_fc_dim = 512
    arg_encoder_dropout = 0.4

    pair_rep_dim = arg_encoder_fc_dim if arg_encoder_fc_num > 0 else arg_rep_dim

    clf_fc_num = 0
    clf_fc_dim = 512
    clf_class_num = 11
    clf_dropout = 0.4
    
    discr_fc_dim = 1024
    discr_dropout = 0.1

    kd = 1
    thresh_high = 0.7
    thresh_low = 0.5
    alpha = 0.9

    seed = 666
    batch_size = 128
    shuffle = False
    lambda1 = 0.1
    lr = 0.002
    lr_t = 0.001
    lr_d = 0.0001
    l2_penalty = 0
    grad_clip = 1

    pre_i_epochs = 250
    pre_a_epochs = 300

    first_stage_epochs = 8
    together_epochs = 300

    logdir = './res/' + datetime.now().strftime('%B%d-%H:%M:%S')