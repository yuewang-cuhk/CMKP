import torch
import time

EPS = 1e-8


class Beam:
    def __init__(self, size, pad, bos, eos, n_best=1, cuda=False, min_length=0, max_eos_per_output_seq=1):
        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size).fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Minimum prediction length
        self.min_length = min_length

        # Store the number of emitted eos token for each hypothesis sequence
        self.eos_counters = torch.zeros(size, dtype=torch.long).to(self.next_ys[0].device)
        # The max. number of eos token that a hypothesis sequence can have
        self.max_eos_per_output_seq = max_eos_per_output_seq

    def compute_avg_score(self, logprobs):
        seq_len = len(self.next_ys) - 1
        assert seq_len != 0
        return logprobs / seq_len

    def get_current_tokens(self):
        """Get the outputs for the current timestep."""
        return self.next_ys[-1]

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def get_hyp(self, timestep, k):
        """
        walk back to construct the full hypothesis given the finished time step and beam idx
        :param timestep: int
        :param k: int
        :return:
        """
        hyp, attn = [], []
        # iterate from output sequence length (with eos but not bos) - 1 to 0f
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(
                self.next_ys[j + 1][k])  # j+1 so that it will iterate from the <eos> token, and end before the <bos>
            attn.append(
                self.attn[j][k])  # since it does not has attn for bos, it will also iterate from the attn for <eos>
            # attn[j][k] Tensor with size [src_len]
            k = self.prev_ks[j][k]  # find the beam idx of the previous token

        return hyp[::-1], torch.stack(attn)

    def advance(self, word_logits, attn_dist):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_logit`- probs of advancing from the last step [beam_size, vocab_size]
        * `attn_dist`- attention at the last step [beam_size, src_len]

        Returns: True if beam search is complete.
        """
        vocab_size = word_logits.size(1)
        # To be implemented: stepwise penalty

        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_logits)):
                word_logits[k][self._eos] = -1e20
        # Sum the previous scores
        if len(self.prev_ks) > 0:
            beam_scores = word_logits + self.scores.unsqueeze(1).expand_as(word_logits)
            # Don't let EOS have children. If it have reached the max number of eos.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos and self.eos_counters[i] >= self.max_eos_per_output_seq:
                    beam_scores[i] = -1e20

        else:  # This is the first decoding step, every beam are the same
            beam_scores = word_logits[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_idx = flat_beam_scores.topk(self.size, 0, True, True)  # [beam_size]

        self.all_scores.append(self.scores)  # list of tensor with size [beam_size]
        self.scores = best_scores

        # best_scores_idx indicate the idx in the flattened beam * vocab_size array, so need to convert
        # the idx back to which beam and word each score came from.
        # convert it to the beam indices that the top k scores came from, LongTensor, size: [beam_size]
        prev_k = best_scores_idx / vocab_size
        self.prev_ks.append(prev_k)
        # convert it to the vocab indices, LongTensor, size: [beam_size]
        self.next_ys.append((best_scores_idx - prev_k * vocab_size))
        # select the attention dist from the corresponding beam, size: [beam_size, src_len]
        self.attn.append(attn_dist.index_select(0, prev_k))
        self.update_eos_counter()  # update the eos_counter according to prev_ks

        for i in range(self.next_ys[-1].size(0)):  # For each generated token in the current step, check if it is EOS
            if self.next_ys[-1][i] == self._eos:
                self.eos_counters[i] += 1
                # compute the score penalize by length and coverage amd append add it to finished
                if self.eos_counters[i] == self.max_eos_per_output_seq:
                    global_scores = self.compute_avg_score(self.scores)
                    s = global_scores[i]
                    self.finished.append((s, len(self.next_ys) - 1, i))  # score, length of sequence, beam_idx
        # End condition is when top-of-beam is EOS (and its number of EOS tokens reached the max) and no global score.
        if self.next_ys[-1][0] == self._eos and self.eos_counters[0] == self.max_eos_per_output_seq:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs in the finished list
            while len(self.finished) < minimum:
                global_scores = self.compute_avg_score(self.scores)
                s = global_scores[i]
                # score, length of sequence (include eos but not bos), beam_idx
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def update_eos_counter(self):
        # update the eos_counter according to prev_ks
        self.eos_counters = self.eos_counters.index_select(0, self.prev_ks[-1])


class SequenceGenerator(object):
    """Class to generate sequences from an image-to-text model."""

    def __init__(self, model, beam_size, max_sequence_length, copy_attn=False, cuda=True, n_best=None):
        """Initializes the generator.
        Args:
          model: recurrent model, with inputs: (input, dec_hidden) and outputs len(vocab) values
          eos_idx: the idx of the <eos> token
          beam_size: Beam size to use when generating sequences.
          max_sequence_length: The maximum sequence length before stopping the search.
        """
        self.model = model

        self.cur_model = model.cur_model
        self.use_img = model.use_img
        self.use_attr = model.use_attr

        self.beam_size = beam_size
        self.max_sequence_length = max_sequence_length
        self.copy_attn = copy_attn
        self.cuda = cuda
        if n_best is None:
            self.n_best = self.beam_size
        else:
            self.n_best = n_best

    def beam_search(self, src, src_lens, src_oov, src_mask, oov_lists, img=None, attr=None):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param oov_lists: list of oov words (idx2word) for each batch, len=batch
        """
        self.model.eval()
        batch_size = src.size(0)
        beam_size = self.beam_size
        # memory_bank, encoder_final_state, img_feats, attr_feats, combined_feat = \
        #     self.encoder(src, src_lens, src_mask, img, attr)
        # Encoding
        memory_bank, encoder_final_state, img_feats, attr_feats, combined_feat = \
            self.model.encoder(src, src_lens, src_mask, img, attr)

        if self.model.combine_pred:
            classifier_outputs = self.model.linear_classifer_final(combined_feat)
            if self.model.combine_pred_type == 'direct':
                cls_pred, _, cls_pred_mask, cls_pred_oov = self.model.get_cls_pred_logits(classifier_outputs)
                cls_pred = cls_pred.repeat(beam_size, 1)
            else:
                cls_pred, _, cls_pred_mask, cls_pred_oov = self.model.encode_pred(classifier_outputs)
                cls_pred = cls_pred.repeat(beam_size, 1, 1)

            cls_pred_mask = cls_pred_mask.repeat(beam_size, 1)
            cls_pred_oov = cls_pred_oov.repeat(beam_size, 1)

            cls_pred = cls_pred.to(self.model.device)
            cls_pred_mask = cls_pred_mask.to(self.model.device)
            cls_pred_oov = cls_pred_oov.to(self.model.device)
        else:
            cls_pred = None
            cls_pred_mask = None
            cls_pred_oov = None

        decoder_init_state = self.model.init_decoder_state(encoder_final_state)

        max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

        # expand img or attr is not None
        # if self.use_img:
        #     if len(img_feat.shape) == 2:
        #         img_feat = img_feat.repeat(self.beam_size, 1)
        #     if len(img_feat.shape) == 3:
        #         img_feat = img_feat.repeat(self.beam_size, 1, 1)
        #
        # if self.use_attr:
        #     if len(attr_feat.shape) == 2:
        #         attr_feat = attr_feat.repeat(self.beam_size, 1)
        #     if len(attr_feat.shape) == 3:
        #         attr_feat = attr_feat.repeat(self.beam_size, 1, 1)

        combined_feat = combined_feat.repeat(self.beam_size, 1)

        # expand memory_bank, src_mask
        memory_bank = memory_bank.repeat(beam_size, 1, 1)  # [batch * beam_size, max_src_len, memory_bank_size]
        src_mask = src_mask.repeat(beam_size, 1)  # [batch * beam_size, src_seq_len]
        src_oov = src_oov.repeat(self.beam_size, 1)  # [batch * beam_size, src_seq_len]
        decoder_state = decoder_init_state.repeat(1, self.beam_size, 1)  # [dec_layers, bs * beam_size, dec_size]

        beam_list = [
            Beam(beam_size, pad=self.model.pad_idx, bos=self.model.bos_idx, eos=self.model.eos_idx,
                 n_best=self.n_best, cuda=self.cuda) for _ in range(batch_size)]

        # Run beam search.
        for t in range(1, self.max_sequence_length + 1):
            if all((b.done() for b in beam_list)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            # b.get_current_tokens(): [beam_size]
            # torch.stack([ [beam of batch 1], [beam of batch 2], ... ]) -> [batch, beam]
            # after transpose -> [beam, batch]
            # After flatten, it becomes
            # [batch_1_beam_1, batch_2_beam_1,..., batch_N_beam_1, batch_1_beam_2, ..., batch_N_beam_2, ...]
            # this match the dimension of hidden state
            decoder_input = torch.stack([b.get_current_tokens() for b in beam_list]).t().contiguous().view(-1)
            # decoder_input: [batch_size * beam_size]

            # Turn any copied words to UNKS
            if self.copy_attn:
                decoder_input = decoder_input.masked_fill(
                    decoder_input.gt(self.model.vocab_size - 1), self.model.unk_idx)

            # Convert the generated eos token to bos token, only useful in one2many_mode=2 or one2many_mode=3
            # decoder_input = decoder_input.masked_fill(decoder_input == self.model.eos_idx, self.model.bos_idx)

            # run one step of decoding
            # [flattened_batch, vocab_size], [dec_layers, flattened_batch, decoder_size],
            # [flattened_batch, memory_bank_size], [flattened_batch, src_len], [flattened_batch, src_len]
            decoder_dist, decoder_state, context, attn_dist, _ = \
                self.model.decoder(decoder_input, decoder_state, memory_bank, src_mask, max_num_oov, src_oov,
                                   img_feats, attr_feats, combined_feat,
                                   cls_pred, cls_pred_mask, cls_pred_oov)
            log_decoder_dist = torch.log(decoder_dist + EPS)

            # Compute a vector of batch x beam word scores
            log_decoder_dist = log_decoder_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, vocab_size]
            attn_dist = attn_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, src_seq_len]

            # Advance each beam
            for batch_idx, beam in enumerate(beam_list):
                beam.advance(log_decoder_dist[:, batch_idx], attn_dist[:, batch_idx, :src_lens[batch_idx]])
                self.beam_decoder_state_update(batch_idx, beam.get_current_origin(), decoder_state)

        # Extract sentences from beam.
        result_dict = self._from_beam(beam_list)
        result_dict['batch_size'] = batch_size
        return result_dict

    def _from_beam(self, beam_list):
        ret = {"predictions": [], "scores": [], "attention": []}
        for b in beam_list:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            # Collect all the decoded sentences in to hyps (list of list of idx) and attn (list of tensor)
            for i, (times, k) in enumerate(ks[:n_best]):
                # Get the corresponding decoded sentence, and also the attn dist [seq_len, memory_bank_size].
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(
                hyps)  # 3d list of idx (zero dim tensor), with len [batch_size, n_best, output_seq_len]
            ret['scores'].append(scores)  # a 2d list of zero dim tensor, with len [batch_size, n_best]
            ret["attention"].append(
                attn)  # a 2d list of FloatTensor[output sequence length, src_len] , with len [batch_size, n_best]
        return ret

    def beam_decoder_state_update(self, batch_idx, beam_indices, decoder_state):
        """
        :param batch_idx: int
        :param beam_indices: a long tensor of previous beam indices, size: [beam_size]
        :param decoder_state: [dec_layers, flattened_batch_size, decoder_size]
        :return:
        """
        decoder_layers, flattened_batch_size, decoder_size = list(decoder_state.size())
        assert flattened_batch_size % self.beam_size == 0
        original_batch_size = flattened_batch_size // self.beam_size
        # select the hidden states of a particular batch, [dec_layers, batch_size * beam_size, decoder_size] ->
        # [dec_layers, beam_size, decoder_size]
        decoder_state_transformed = decoder_state.view(decoder_layers, self.beam_size, original_batch_size,
                                                       decoder_size)[:, :, batch_idx]
        # select the hidden states of the beams specified by the beam_indices -> [dec_layers, beam_size, decoder_size]
        decoder_state_transformed.data.copy_(decoder_state_transformed.data.index_select(1, beam_indices))


def prediction_to_sentence(prediction, idx2word, vocab_size, oov, eos_idx,
                           unk_idx=None, replace_unk=False, src_word_list=None, attn_dist=None):
    """
    :param prediction: a list of 0 dim tensor
    :param attn_dist: tensor with size [trg_len, src_len]
    :return: a list of words, does not include the final EOS
    """
    sentence = []
    for i, pred in enumerate(prediction):
        _pred = int(pred.item())  # convert zero dim tensor to int
        if i == len(prediction) - 1 and _pred == eos_idx:  # ignore the final EOS token
            break
        if _pred < vocab_size:
            if _pred == unk_idx and replace_unk:
                assert src_word_list is not None and attn_dist is not None, "If you need to replace unk, you must supply src_word_list and attn_dist"
                # _, max_attn_idx = attn_dist[i].max(0)
                _, max_attn_idx = attn_dist[i].topk(2, dim=0)
                if max_attn_idx[0] < len(src_word_list):
                    word = src_word_list[int(max_attn_idx[0].item())]
                else:
                    word = src_word_list[int(max_attn_idx[1].item())]
                    # word = pykp.io.EOS_WORD
            else:
                word = idx2word[_pred]
        else:
            word = oov[_pred - vocab_size]
        sentence.append(word)

    return sentence


def preprocess_beam_search_result(beam_search_result, idx2word, vocab_size, oov_lists,
                                  eos_idx, unk_idx, replace_unk, src_str_list):
    batch_size = beam_search_result['batch_size']
    predictions = beam_search_result['predictions']
    scores = beam_search_result['scores']
    attention = beam_search_result['attention']
    assert len(predictions) == batch_size
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, score_n_best, attn_n_best, oov, src_word_list in zip(predictions, scores, attention, oov_lists,
                                                                          src_str_list):
        # attn_n_best: list of tensor with size [trg_len, src_len], len=n_best
        pred_dict = {}
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk,
                                              src_word_list, attn)
            sentences_n_best.append(sentence)
        pred_dict['sentences'] = sentences_n_best
        # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_dict['scores'] = score_n_best
        # a list of zero dim tensor, with len [n_best]
        pred_dict['attention'] = attn_n_best
        # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
        pred_list.append(pred_dict)
    return pred_list
