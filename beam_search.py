
import torch as T
from train_util import get_cuda, config


class Beam(object):
    def __init__(self, start_id, end_id, unk_id, hidden_state, context):
        h,c = hidden_state                                              #(n_hid,)
        self.tokens = T.LongTensor(config.beam_size,1).fill_(start_id)  #(beam, t) after t time steps
        self.scores = T.FloatTensor(config.beam_size,1).fill_(-30)      #beam,1; Initial score of beams = -30
        self.tokens, self.scores = get_cuda(self.tokens), get_cuda(self.scores)
        self.scores[0][0] = 0                                           #At time step t=0, all beams should extend from a single beam. So, I am giving high initial score to 1st beam
        self.hid_h = h.unsqueeze(0).repeat(config.beam_size, 1)         #beam, n_hid
        self.hid_c = c.unsqueeze(0).repeat(config.beam_size, 1)         #beam, n_hid
        self.context = context.unsqueeze(0).repeat(config.beam_size, 1) #beam, 2*n_hid
        self.sum_temporal_srcs = None
        self.prev_s = None
        self.done = False
        self.end_id = end_id
        self.unk_id = unk_id

    def get_current_state(self):
        tokens = self.tokens[:, -1].clone()
        for i in range(len(tokens)):
            if tokens[i].item() >= config.vocab_size:
                tokens[i] = self.unk_id
        return tokens


    def advance(self, prob_dist, hidden_state, context, sum_temporal_srcs, prev_s):
        '''Perform beam search: Considering the probabilites of given n_beam x n_extended_vocab words, select first n_beam words that give high total scores
        :param prob_dist: (beam, n_extended_vocab)
        :param hidden_state: Tuple of (beam, n_hid) tensors
        :param context:   (beam, 2*n_hidden)
        :param sum_temporal_srcs:   (beam, n_seq)
        :param prev_s:  (beam, t, n_hid)
        '''
        n_extended_vocab = prob_dist.size(1)
        h, c = hidden_state
        log_probs = T.log(prob_dist+config.eps)                         #beam, n_extended_vocab

        scores = log_probs + self.scores                                #beam, n_extended_vocab
        scores = scores.view(-1,1)                                      #beam*n_extended_vocab, 1
        best_scores, best_scores_id = T.topk(input=scores, k=config.beam_size, dim=0)   #will be sorted in descending order of scores
        self.scores = best_scores                                       #(beam,1); sorted
        beams_order = best_scores_id.squeeze(1)/n_extended_vocab        #(beam,); sorted
        best_words = best_scores_id%n_extended_vocab                    #(beam,1); sorted
        self.hid_h = h[beams_order]                                     #(beam, n_hid); sorted
        self.hid_c = c[beams_order]                                     #(beam, n_hid); sorted
        self.context = context[beams_order]
        if sum_temporal_srcs is not None:
            self.sum_temporal_srcs = sum_temporal_srcs[beams_order]     #(beam, n_seq); sorted
        if prev_s is not None:
            self.prev_s = prev_s[beams_order]                           #(beam, t, n_hid); sorted
        self.tokens = self.tokens[beams_order]                          #(beam, t); sorted
        self.tokens = T.cat([self.tokens, best_words], dim=1)           #(beam, t+1); sorted

        #End condition is when top-of-beam is EOS.
        if best_words[0][0] == self.end_id:
            self.done = True

    def get_best(self):
        best_token = self.tokens[0].cpu().numpy().tolist()              #Since beams are always in sorted (descending) order, 1st beam is the best beam
        try:
            end_idx = best_token.index(self.end_id)
        except ValueError:
            end_idx = len(best_token)
        best_token = best_token[1:end_idx]
        return best_token

    def get_all(self):
        all_tokens = []
        for i in range(len(self.tokens)):
            all_tokens.append(self.tokens[i].cpu().numpy())
        return all_tokens


def beam_search(enc_hid, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, model, start_id, end_id, unk_id):

    batch_size = len(enc_hid[0])
    beam_idx = T.LongTensor(list(range(batch_size)))
    beams = [Beam(start_id, end_id, unk_id, (enc_hid[0][i], enc_hid[1][i]), ct_e[i]) for i in range(batch_size)]   #For each example in batch, create Beam object
    n_rem = batch_size                                                  #Index of beams that are active, i.e: didn't generate [STOP] yet
    sum_temporal_srcs = None                                            #Number of examples in batch that didn't generate [STOP] yet
    prev_s = None

    for t in range(config.max_dec_steps):
        x_t = T.stack(
            [beam.get_current_state() for beam in beams if beam.done == False]      #remaining(rem),beam
        ).contiguous().view(-1)                                                     #(rem*beam,)
        x_t = model.embeds(x_t)                                                 #rem*beam, n_emb

        dec_h = T.stack(
            [beam.hid_h for beam in beams if beam.done == False]                    #rem*beam,n_hid
        ).contiguous().view(-1,config.hidden_dim)
        dec_c = T.stack(
            [beam.hid_c for beam in beams if beam.done == False]                    #rem,beam,n_hid
        ).contiguous().view(-1,config.hidden_dim)                                   #rem*beam,n_hid

        ct_e = T.stack(
            [beam.context for beam in beams if beam.done == False]                  #rem,beam,n_hid
        ).contiguous().view(-1,2*config.hidden_dim)                                 #rem,beam,n_hid

        if sum_temporal_srcs is not None:
            sum_temporal_srcs = T.stack(
                [beam.sum_temporal_srcs for beam in beams if beam.done == False]
            ).contiguous().view(-1, enc_out.size(1))                                #rem*beam, n_seq

        if prev_s is not None:
            prev_s = T.stack(
                [beam.prev_s for beam in beams if beam.done == False]
            ).contiguous().view(-1, t, config.hidden_dim)                           #rem*beam, t-1, n_hid


        s_t = (dec_h, dec_c)
        enc_out_beam = enc_out[beam_idx].view(n_rem,-1).repeat(1, config.beam_size).view(-1, enc_out.size(1), enc_out.size(2))
        enc_pad_mask_beam = enc_padding_mask[beam_idx].repeat(1, config.beam_size).view(-1, enc_padding_mask.size(1))

        extra_zeros_beam = None
        if extra_zeros is not None:
            extra_zeros_beam = extra_zeros[beam_idx].repeat(1, config.beam_size).view(-1, extra_zeros.size(1))
        enc_extend_vocab_beam = enc_batch_extend_vocab[beam_idx].repeat(1, config.beam_size).view(-1, enc_batch_extend_vocab.size(1))

        final_dist, (dec_h, dec_c), ct_e, sum_temporal_srcs, prev_s = model.decoder(x_t, s_t, enc_out_beam, enc_pad_mask_beam, ct_e, extra_zeros_beam, enc_extend_vocab_beam, sum_temporal_srcs, prev_s)              #final_dist: rem*beam, n_extended_vocab

        final_dist = final_dist.view(n_rem, config.beam_size, -1)                   #final_dist: rem, beam, n_extended_vocab
        dec_h = dec_h.view(n_rem, config.beam_size, -1)                             #rem, beam, n_hid
        dec_c = dec_c.view(n_rem, config.beam_size, -1)                             #rem, beam, n_hid
        ct_e = ct_e.view(n_rem, config.beam_size, -1)                             #rem, beam, 2*n_hid

        if sum_temporal_srcs is not None:
            sum_temporal_srcs = sum_temporal_srcs.view(n_rem, config.beam_size, -1) #rem, beam, n_seq

        if prev_s is not None:
            prev_s = prev_s.view(n_rem, config.beam_size, -1, config.hidden_dim)    #rem, beam, t

        # For all the active beams, perform beam search
        active = []         #indices of active beams after beam search

        for i in range(n_rem):
            b = beam_idx[i].item()
            beam = beams[b]
            if beam.done:
                continue

            sum_temporal_srcs_i = prev_s_i = None
            if sum_temporal_srcs is not None:
                sum_temporal_srcs_i = sum_temporal_srcs[i]                              #beam, n_seq
            if prev_s is not None:
                prev_s_i = prev_s[i]                                                #beam, t, n_hid
            beam.advance(final_dist[i], (dec_h[i], dec_c[i]), ct_e[i], sum_temporal_srcs_i, prev_s_i)
            if beam.done == False:
                active.append(b)

        if len(active) == 0:
            break

        beam_idx = T.LongTensor(active)
        n_rem = len(beam_idx)

    predicted_words = []
    for beam in beams:
        predicted_words.append(beam.get_best())

    return predicted_words













