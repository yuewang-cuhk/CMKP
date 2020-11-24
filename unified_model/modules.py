import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_att.submodules import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding


def masked_mean(input, mask=None, dim=1):
    # input: [batch_size, seq_len, hidden_size]
    # mask: Float Tensor of size [batch_size, seq_len], where 1.0 for unmask, 0.0 for mask ones
    if mask is None:
        return torch.mean(input, dim=dim)
    else:
        mask = mask.unsqueeze(-1)
        mask_input = input * mask
        sum_mask_input = mask_input.sum(dim=dim)
        for dim in range(mask.size(0)):
            sum_mask_input[dim] = sum_mask_input[dim] / mask[dim].sum()
        return sum_mask_input


def masked_max(input, mask=None, dim=1):
    # input: [batch_size, seq_len, hidden_size]
    # mask: Float Tensor of size [batch_size, seq_len], where 1.0 for unmask, 0.0 for mask ones
    if mask is None:
        max_v, _ = torch.max(input, dim=dim)
        return max_v
    else:
        mask = mask.unsqueeze(-1)
        mask = mask.repeat(1, 1, input.size(-1))
        input = input.masked_fill(mask == 0.0, float('-inf'))
        max_v, _ = torch.max(input, dim=dim)
        return max_v


class MaskedSoftmax(nn.Module):
    def __init__(self, dim):
        super(MaskedSoftmax, self).__init__()
        self.dim = dim

    def forward(self, logit, mask=None):
        if mask is None:
            dist = F.softmax(logit - torch.max(logit, dim=self.dim, keepdim=True)[0], dim=self.dim)
        else:
            dist_ = F.softmax(logit - torch.max(logit, dim=self.dim, keepdim=True)[0], dim=self.dim) * mask
            normalization_factor = dist_.sum(self.dim, keepdim=True)
            dist = dist_ / normalization_factor
        return dist


class Attention(nn.Module):
    def __init__(self, decoder_size, memory_bank_size, need_mask=False):
        super(Attention, self).__init__()
        self.need_mask = need_mask
        self.v = nn.Linear(decoder_size, 1, bias=False)
        self.decode_project = nn.Linear(decoder_size, decoder_size)
        self.memory_project = nn.Linear(memory_bank_size, decoder_size, bias=False)
        self.softmax = MaskedSoftmax(dim=1)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(decoder_size + memory_bank_size, memory_bank_size)

    def score(self, memory_bank, decoder_state):
        """
        :param memory_bank: [batch_size, max_src_len, self.bi_hidden_size]
        :param decoder_state: [batch_size, decoder_size]
        :return: score: [batch_size, max_src_len]
        """
        batch_size, max_src_len, memory_bank_size = list(memory_bank.size())
        decoder_size = decoder_state.size(1)

        # Bahdanau style attention
        # project memory_bank
        memory_bank_ = memory_bank.view(-1, memory_bank_size)  # [batch_size*max_src_len, memory_bank_size]
        encoder_feature = self.memory_project(memory_bank_)  # [batch_size*max_src_len, decoder size]
        # project decoder state
        dec_feature = self.decode_project(decoder_state)  # [batch_size, decoder_size]
        dec_feature_expanded = dec_feature.unsqueeze(1).expand(batch_size, max_src_len, decoder_size).contiguous()
        dec_feature_expanded = dec_feature_expanded.view(-1, decoder_size)  # [batch_size*max_src_len, decoder_size]
        # sum up attention features
        att_features = encoder_feature + dec_feature_expanded  # [batch_size*max_src_len, decoder_size]

        # compute attention score and normalize them
        e = self.tanh(att_features)  # [batch_size*max_src_len, decoder_size]
        scores = self.v(e)  # [batch_size*max_src_len, 1]

        scores = scores.view(-1, max_src_len)  # [batch_size, max_src_len]
        return scores

    def forward(self, decoder_state, memory_bank, src_mask=None, return_attn=False):
        """
        :param decoder_state: [batch_size, decoder_size]
        :param memory_bank: [batch_size, max_input_seq_len, self.bi_hidden_size]
        :param src_mask: [batch_size, max_input_seq_len]
        :return: context, attn_dist
        """
        # init dimension info
        batch_size, max_src_len, memory_bank_size = list(memory_bank.size())

        if self.need_mask:
            assert src_mask is not None

        # if src_mask is None:  # if it does not supply a source mask, create a dummy mask with all ones
        #     src_mask = memory_bank.new_ones(batch_size, max_src_len)

        scores = self.score(memory_bank, decoder_state)
        attn_dist = self.softmax(scores, mask=src_mask)

        # Compute weighted sum of memory bank features
        attn_dist = attn_dist.unsqueeze(1)  # [batch_size, 1, max_input_seq_len]
        memory_bank = memory_bank.view(-1, max_src_len, memory_bank_size)  # batch_size, max_src_len, memory_bank_size]
        context = torch.bmm(attn_dist, memory_bank)  # [batch_size, 1, memory_bank_size]
        context = context.squeeze(1)  # [batch_size, memory_bank_size]
        attn_dist = attn_dist.squeeze(1)  # [batch_size, max_input_seq_len]

        # Update coverage
        assert attn_dist.size() == torch.Size([batch_size, max_src_len])
        assert context.size() == torch.Size([batch_size, memory_bank_size])

        context_output = self.linear(torch.cat([context, decoder_state], dim=1))
        if return_attn:
            return context_output, attn_dist
        return context_output


class CoAttention(nn.Module):
    def __init__(self, text_feat_size, img_feat_size, input_type='text_img'):
        """Initialize model."""
        super(CoAttention, self).__init__()
        self.text_feat_size = text_feat_size
        self.img_feat_size = img_feat_size
        self.input_type = input_type
        assert input_type in ['text_img', 'img_text', 'text_text']
        self.v_text = nn.Linear(text_feat_size, 1, bias=False)
        self.v_img = nn.Linear(img_feat_size, 1, bias=False)
        self.text2img_project = nn.Linear(text_feat_size, img_feat_size, bias=False)
        self.img2text_project = nn.Linear(img_feat_size, text_feat_size, bias=False)
        self.img_project = nn.Linear(img_feat_size, img_feat_size)
        self.text_project = nn.Linear(text_feat_size, text_feat_size)
        self.softmax = MaskedSoftmax(dim=1)
        self.linear = nn.Linear(text_feat_size + img_feat_size, text_feat_size)

    def text_att_scores(self, text_feat, img_feats):
        batch_size, img_num, img_feat_size = list(img_feats.size())
        batch_size, text_feat_size = list(text_feat.size())

        img_feats_ = img_feats.view(-1, img_feat_size)  # [batch_size*img_num, img_feat_size]
        img_feature = self.img2text_project(img_feats_)  # [batch_size*img_num, text_feat_size]

        # Project decoder state: text_feats (in our case)
        text_feature = self.text_project(text_feat)  # [batch_size, text_feat_size]
        text_feature_expanded = text_feature.unsqueeze(1).expand(batch_size, img_num, text_feat_size).contiguous()
        text_feature_expanded = text_feature_expanded.view(-1, text_feat_size)  # [batch_size*img_num, text_feat_size]

        # sum up attention features
        att_features = img_feature + text_feature_expanded  # [batch_size*img_num, text_feat_size]
        e = torch.tanh(att_features)  # [batch_size*img_num, text_feat_size]
        scores = self.v_text(e)  # [batch_size*img_num, 1]
        scores = scores.view(-1, img_num)  # [batch_size, img_num]
        return scores

    def img_att_scores(self, img_feat, text_feats):
        batch_size, max_src_len, text_feat_size = list(text_feats.size())
        batch_size, img_feat_size = list(img_feat.size())

        text_feats_ = text_feats.view(-1, text_feat_size)  # [batch_size*max_src_len, text_feat_size]
        text_feature = self.text2img_project(text_feats_)  # [batch_size*max_src_len, img_feat_size]

        # Project decoder state: text_feats (in our case)
        img_feature = self.img_project(img_feat)  # [batch_size, img_feat_size]
        img_feature_expanded = img_feature.unsqueeze(1).expand(batch_size, max_src_len, img_feat_size).contiguous()
        img_feature_expanded = img_feature_expanded.view(-1, img_feat_size)  # [batch_size*max_src_len, img_feat_size]

        # sum up attention features
        att_features = text_feature + img_feature_expanded  # [batch_size*max_src_len, img_feat_size]
        e = torch.tanh(att_features)  # [batch_size*max_src_len, img_feat_size]
        scores = self.v_img(e)  # [batch_size*max_src_len, 1]
        scores = scores.view(-1, max_src_len)  # [batch_size, max_src_len]
        return scores

    def forward(self, text_feats, img_feats, src_mask):
        # Text
        batch_size, img_num, img_feat_size = list(img_feats.size())
        batch_size, max_src_len, text_feat_size = list(text_feats.size())

        if self.input_type in ['text_img', 'text_text']:
            text_feat = masked_mean(text_feats, src_mask, dim=1)
        elif self.input_type in ['img_text']:
            text_feat = torch.mean(text_feats, dim=1)

        text_scores = self.text_att_scores(text_feat, img_feats)

        if self.input_type in ['text_img']:
            text_att_dist = self.softmax(text_scores)
        elif self.input_type in ['img_text', 'text_text']:
            text_att_dist = self.softmax(text_scores, mask=src_mask)

        text_att_dist = text_att_dist.unsqueeze(1)  # [batch_size, 1, img_num]
        img_feats = img_feats.view(-1, img_num, img_feat_size)  # batch_size, img_num, img_feat_size]
        img_context = torch.bmm(text_att_dist, img_feats)  # [batch_size, 1, img_feat_size]
        img_context = img_context.squeeze(1)  # [batch_size, img_feat_size]

        img_scores = self.img_att_scores(img_context, text_feats)

        if self.input_type in ['text_img', 'text_text']:
            img_att_dist = self.softmax(img_scores, mask=src_mask)
        elif self.input_type in ['img_text']:
            img_att_dist = self.softmax(img_scores)

        img_att_dist = img_att_dist.unsqueeze(1)  # [batch_size, 1, max_src_len]
        text_feats = text_feats.view(-1, max_src_len, text_feat_size)  # [batch_size, max_src_len, text_feat_size]
        text_context = torch.bmm(img_att_dist, text_feats)  # [batch_size, 1, text_feat_size]
        text_context = text_context.squeeze(1)  # [batch_size, text_feat_size]

        combined_features = torch.cat([img_context, text_context], dim=1)
        combined_features = self.linear(combined_features)
        return combined_features


class MyMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_kv, dropout=0.1, need_mask=False, is_regu=False):
        super(MyMultiHeadAttention, self).__init__()
        self.need_mask = need_mask
        self.is_regu = is_regu
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_kv, d_kv, dropout=dropout, is_regu=is_regu)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model, dropout=dropout)

    def forward(self, q, k, v, mask=None):
        # q: [batch_size, d_model] ==>  k: [batch_size, 1, d_model]
        # mask: [batch_size, seq_len] == > [batch_size, 1, seq_len]
        # when there is only one query, we need to expand the dimension
        if len(q.shape) == 2:
            q = q.unsqueeze(1)
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len]
        if self.need_mask:
            assert mask is not None, 'Please pass the attention mask to the multi-head'
        if self.is_regu:
            enc_output, enc_slf_attn, head_diff = self.slf_attn(q, k, v, mask)
        else:
            enc_output, enc_slf_attn = self.slf_attn(q, k, v, mask)
        enc_output = self.pos_ffn(enc_output)

        # enc_output: [batch_size, 1, d_model] ==>  k: [batch_size, d_model]
        enc_output = enc_output.squeeze(1)
        if self.is_regu:
            return enc_output, enc_slf_attn, head_diff
        return enc_output, enc_slf_attn


class MyTransformer(nn.Module):
    def __init__(self, n_layers, n_head, d_model, d_kv, emb_size=200, dropout=0.1):
        super(MyTransformer, self).__init__()

        self.position_enc = PositionalEncoding(emb_size, n_position=200)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            MyMultiHeadAttention(n_head, d_model, d_kv, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq_embed, memory_bank, src_mask, return_attns=False):

        enc_slf_attn_list = []

        enc_output = self.dropout(self.position_enc(src_seq_embed))
        # (enc_output, memory_bank, memory_bank, src_mask)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, memory_bank, memory_bank, src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output
