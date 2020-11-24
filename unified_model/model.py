import torch
import torch.nn as nn
import pickle
import numpy as np
from modules import masked_mean, masked_max, Attention, CoAttention, MaskedSoftmax, MyMultiHeadAttention


def get_multi_head_att_paras(cur_model):
    # default setting
    n_head = 4
    d_kv = 256
    stack_num = 1
    for seg in cur_model.split('_'):
        if seg[0] == 'h' and seg[1:].isdigit():
            n_head = int(seg[1:])
        if seg[0] == 'd' and seg[1:].isdigit():
            d_kv = int(seg[1:])
        if seg[0] == 'x' and seg[1:].isdigit():
            stack_num = int(seg[1:])

    print('\nStacked %d multi-head attention layer with head num: %d, dim: %d' % (stack_num, n_head, d_kv))
    return n_head, d_kv, stack_num


class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, memory_bank_size, copy_attn, pad_idx, dropout, cur_model):
        super(RNNDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.memory_bank_size = memory_bank_size
        self.dropout = nn.Dropout(dropout)
        self.copy_attn = copy_attn
        self.pad_token = pad_idx

        self.cur_model = cur_model
        self.use_img = 'img' in self.cur_model
        self.use_attr = 'attr' in self.cur_model

        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )

        self.input_size = embed_size

        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=hidden_size, num_layers=1,
                          bidirectional=False, batch_first=False)

        self.attention_layer = Attention(decoder_size=hidden_size, memory_bank_size=memory_bank_size, need_mask=True)

        self.combine_pred = 'combine' in cur_model
        self.combine_pred_type = 'direct' if 'direct' in cur_model else 'embed'
        if self.combine_pred:
            if self.combine_pred_type == 'embed':
                self.pred_att = Attention(decoder_size=hidden_size,
                                          memory_bank_size=memory_bank_size,
                                          need_mask=True)
            self.cls_pred_p_gen_linear = nn.Linear(embed_size + hidden_size + memory_bank_size, 1)

        if copy_attn:
            self.p_gen_linear = nn.Linear(embed_size + hidden_size + memory_bank_size, 1)

        self.sigmoid = nn.Sigmoid()

        self.vocab_dist_linear_1 = nn.Linear(hidden_size + memory_bank_size, hidden_size)
        self.vocab_dist_linear_2 = nn.Linear(hidden_size, vocab_size)
        self.softmax = MaskedSoftmax(dim=1)

    def forward(self, y, h, memory_bank, src_mask, max_num_oovs, src_oov,
                img_feats=None, attr_feats=None, combined_feat=None,
                cls_pred=None, cls_pred_mask=None, cls_pred_oov=None):

        """
        :param y: [batch_size]
        :param h: [num_layers, batch_size, decoder_size]
        :param memory_bank: [batch_size, max_src_seq_len, memory_bank_size]
        :param src_mask: [batch_size, max_src_seq_len]
        :param max_num_oovs: int
        :param src_oov: [batch_size, max_src_seq_len]
        :return:
        """
        batch_size, max_src_len = list(src_oov.size())
        assert y.size() == torch.Size([batch_size])
        assert h.size() == torch.Size([1, batch_size, self.hidden_size])

        # init input embedding
        y_emb = self.embedding(y).unsqueeze(0)  # [1, batch_size, embed_size]

        _, h_next = self.rnn(y_emb, h)

        assert h_next.size() == torch.Size([1, batch_size, self.hidden_size])

        last_layer_h_next = h_next[-1, :, :]  # [batch, decoder_size]

        # if 'img_dec_fuse' in self.cur_model:
        #     last_layer_h_next = torch.add(combined_feat, last_layer_h_next)
        context, attn_dist = self.attention_layer(last_layer_h_next, memory_bank, src_mask, return_attn=True)

        if self.combine_pred:
            assert cls_pred is not None, 'cls_pred and cls_pred_mask is not None for combine_pred'

        assert context.size() == torch.Size([batch_size, self.memory_bank_size])
        assert attn_dist.size() == torch.Size([batch_size, max_src_len])

        if combined_feat is not None:
            context = torch.add(context, combined_feat)

        vocab_dist_input = torch.cat((context, last_layer_h_next), dim=1)

        vocab_dist_input_1 = self.dropout(self.vocab_dist_linear_1(vocab_dist_input))
        vocab_dist = self.softmax(self.vocab_dist_linear_2(vocab_dist_input_1))

        p_gen = None
        if self.copy_attn:
            if self.combine_pred:

                # default: self.combine_pred_type == 'direct':
                p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)), dim=1)
                p_gen = self.sigmoid(self.p_gen_linear(p_gen_input))

                vocab_dist_ = p_gen * vocab_dist
                cls_pred_attn_dist_ = (1 - p_gen) * cls_pred
                attn_dist_ = (1 - p_gen) * attn_dist

                if max_num_oovs > 0:
                    extra_zeros = vocab_dist_.new_zeros((batch_size, max_num_oovs))
                    vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=1)

                final_dist = vocab_dist_.scatter_add(1, src_oov, attn_dist_)
                final_dist = final_dist.scatter_add(1, cls_pred_oov, cls_pred_attn_dist_)
                assert final_dist.size() == torch.Size([batch_size, self.vocab_size + max_num_oovs])
            else:
                # [batch_size, memory_bank_size + decoder_size + embed_size]
                p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)), dim=1)
                p_gen = self.sigmoid(self.p_gen_linear(p_gen_input))

                vocab_dist_ = p_gen * vocab_dist
                attn_dist_ = (1 - p_gen) * attn_dist

                if max_num_oovs > 0:
                    extra_zeros = vocab_dist_.new_zeros((batch_size, max_num_oovs))
                    vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=1)

                final_dist = vocab_dist_.scatter_add(1, src_oov, attn_dist_)
                assert final_dist.size() == torch.Size([batch_size, self.vocab_size + max_num_oovs])
        else:
            final_dist = vocab_dist
            assert final_dist.size() == torch.Size([batch_size, self.vocab_size])

        return final_dist, h_next, context, attn_dist, p_gen


class MultimodalEncoder(nn.Module):
    def __init__(self, opt):
        """Initialize model."""
        super(MultimodalEncoder, self).__init__()
        self.data_path = opt.data_path
        self.emb_path = opt.emb_path
        self.bidirectional = opt.bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_size = opt.hidden_size
        self.bi_hidden_size = self.num_directions * opt.hidden_size
        opt.bi_hidden_size = self.bi_hidden_size

        self.cur_model = opt.cur_model
        self.use_text = opt.use_text
        assert self.use_text
        self.use_img = opt.use_img
        self.use_attr = opt.use_attr

        self.img_ext_model = opt.img_ext_model

        self.text_pooling_type = 'avg' if 'avg_text' in opt.cur_model else 'max'  # default is max
        self.img_pooling_type = 'max' if 'max_img' in opt.cur_model else 'avg'  # default is avg
        self.attr_pooling_type = 'avg' if 'avg_attr' in opt.cur_model else 'max'  # default is max

        self.embedding = nn.Embedding(
            opt.vocab_size,
            opt.emb_size,
            opt.pad_idx
        )
        self.init_weights(opt.emb_type, opt.pad_idx)

        self.rnn = nn.GRU(input_size=opt.emb_size, hidden_size=opt.hidden_size, num_layers=opt.num_layers,
                          bidirectional=opt.bidirectional, batch_first=True, dropout=opt.dropout)

        if 'text_self_att' in self.cur_model:
            self.attention = Attention(self.bi_hidden_size, self.bi_hidden_size, need_mask=True)

        if 'text_self_co_att' in self.cur_model:
            self.text_self_co_att = CoAttention(self.bi_hidden_size, self.bi_hidden_size, input_type='text_text')

        if self.use_img:
            # resnet/butd: 2048, vgg: 512
            self.raw_img_feat_size = 2048 if 'resnet' in opt.img_ext_model or 'butd' in opt.img_ext_model else 512
            self.linear_img = nn.Linear(self.raw_img_feat_size, self.bi_hidden_size)
            # single-attention
            if 'text_img_att' in self.cur_model:
                self.text_img_att = Attention(self.bi_hidden_size, self.bi_hidden_size)
            if 'img_text_att' in self.cur_model:
                self.img_text_att = Attention(self.bi_hidden_size, self.bi_hidden_size, need_mask=True)
            if 'text_img_add_text_att' in self.cur_model:
                self.text_img_add_text_att = Attention(2 * self.bi_hidden_size, self.bi_hidden_size, need_mask=True)

            if 'text_img_co_att' in self.cur_model:
                self.text_img_co_att = CoAttention(self.bi_hidden_size, self.bi_hidden_size, input_type='text_img')
            if 'img_text_co_att' in self.cur_model:
                self.img_text_co_att = CoAttention(self.bi_hidden_size, self.bi_hidden_size, input_type='img_text')

        # co-attention
        if 'multi_head_att' in self.cur_model:
            # ['img_text_multi_head_att_h4_d256', 'text_img_multi_head_att_h4_d256',]
            # We hard code the head number and the dimension of the subspace into model name
            # 'img_text_multi_head_att_h1_d128'==> head: 1, dim: 128

            # default setting
            self.is_regu = True if 'regu' in self.cur_model else False
            n_head, d_kv, stack_num = get_multi_head_att_paras(self.cur_model)

            if 'img_text_multi_head_att' in self.cur_model:
                self.img_text_multi_head_att = nn.ModuleList(
                    [MyMultiHeadAttention(n_head, self.bi_hidden_size, d_kv, dropout=opt.dropout, need_mask=True,
                                          is_regu=self.is_regu)
                     for _ in range(stack_num)])
            elif 'text_img_multi_head_att' in self.cur_model:
                self.text_img_multi_head_att = nn.ModuleList(
                    [MyMultiHeadAttention(n_head, self.bi_hidden_size, d_kv, dropout=opt.dropout, need_mask=False)
                     for _ in range(stack_num)])
            elif 'attr_text_multi_head_att' in self.cur_model:
                self.attr_text_multi_head_att = nn.ModuleList(
                    [MyMultiHeadAttention(n_head, self.bi_hidden_size, d_kv, dropout=opt.dropout, need_mask=True)
                     for _ in range(stack_num)])
            elif 'img_attr_add_text_multi_head_att' in self.cur_model:
                self.img_attr_add_text_multi_head_att = nn.ModuleList(
                    [MyMultiHeadAttention(n_head, self.bi_hidden_size, d_kv, dropout=opt.dropout, need_mask=True)
                     for _ in range(stack_num)])
            elif 'img_attr_sep_text_multi_head_att' in self.cur_model:
                self.img_sep_text_multi_head_att = nn.ModuleList(
                    [MyMultiHeadAttention(n_head, self.bi_hidden_size, d_kv, dropout=opt.dropout, need_mask=True)
                     for _ in range(stack_num)])
                self.attr_sep_text_multi_head_att = nn.ModuleList(
                    [MyMultiHeadAttention(n_head, self.bi_hidden_size, d_kv, dropout=opt.dropout, need_mask=True)
                     for _ in range(stack_num)])
            elif 'text_text_multi_head_att' in self.cur_model:
                self.text_text_multi_head_att = nn.ModuleList(
                    [MyMultiHeadAttention(n_head, self.bi_hidden_size, d_kv, dropout=opt.dropout, need_mask=True)
                     for _ in range(stack_num)])
            else:
                raise NotImplementedError

        if self.use_attr:
            self.linear_attr = nn.Linear(opt.emb_size, self.bi_hidden_size)
            if 'text_attr_att' in self.cur_model:
                self.text_attr_att = Attention(self.bi_hidden_size, self.bi_hidden_size)
            if 'attr_text_att' in self.cur_model:
                self.attr_text_att = Attention(self.bi_hidden_size, self.bi_hidden_size, need_mask=True)
            if 'text_attr_add_text_att' in self.cur_model:
                self.text_attr_add_text_att = Attention(2 * self.bi_hidden_size, self.bi_hidden_size, need_mask=True)
            if 'text_attr_real_add_text_att' in self.cur_model:
                self.text_attr_real_add_text_att = Attention(self.bi_hidden_size, self.bi_hidden_size, need_mask=True)
            elif 'text_attr_co_att' in self.cur_model:
                self.text_attr_co_att = CoAttention(self.bi_hidden_size, self.bi_hidden_size, input_type='text_img')
            elif 'attr_text_co_att' in self.cur_model:
                self.attr_text_co_att = CoAttention(self.bi_hidden_size, self.bi_hidden_size, input_type='img_text')

        self.dropout = nn.Dropout(p=opt.dropout)

    def init_weights(self, emb_type, pad_idx):
        """Initialize weights."""
        if emb_type == 'random':
            initrange = 0.1
            self.embedding.weight.data.uniform_(-initrange, initrange)
        else:
            with open(self.emb_path, 'rb') as f:
                weights = pickle.load(f)
            self.embedding.weight.data = torch.Tensor(weights)
            # self.embedding.weight.requires_grad = False
            print('Load glove embedding!')

        self.embedding.weight.data[pad_idx] = 0

    def get_text_memory_bank(self, src, src_lens, return_last_state=True):
        # embed the src post with embedding and Bi-GRU layers
        batch_size, max_src_len = list(src.size())
        src_embed = self.embedding(src)  # [batch, src_len, emb_size]
        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens, batch_first=True)
        memory_bank, enc_final_state = self.rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True)  # unpack (back to padded)
        memory_bank = memory_bank.contiguous()
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.bi_hidden_size])

        if self.bidirectional:
            # [batch, hidden_size*2]
            enc_last_layer_final_state = torch.cat((enc_final_state[-1, :, :], enc_final_state[-2, :, :]), 1)
        else:
            enc_last_layer_final_state = enc_final_state[-1, :, :]  # [batch, hidden_size]

        if return_last_state:
            return memory_bank, enc_last_layer_final_state
        return memory_bank

    def get_img_memory_bank(self, img_feats):
        # read image visual feature and map them to bi_hidden_size
        batch_size = img_feats.shape[0]

        if self.img_ext_model == 'vgg':
            img_feats = img_feats.view(-1, img_feats.shape[2])
            img_feats = self.linear_img(img_feats)
            img_feats = img_feats.view(batch_size, -1, img_feats.shape[-1])  # [batch_size, 49, bi_hidden_size]

        elif self.img_ext_model == 'resnet':
            img_feats = self.linear_img(img_feats)

        elif self.img_ext_model == 'butd':
            img_feats = img_feats.view(-1, img_feats.shape[2])
            img_feats = self.linear_img(img_feats)
            img_feats = img_feats.view(batch_size, -1, img_feats.shape[-1])  # [batch_size, 36, bi_hidden_size]

        elif self.img_ext_model == 'complex_resnet':
            img_feats = img_feats.view(-1, img_feats.shape[2])
            img_feats = self.linear_img(img_feats)
            img_feats = img_feats.view(batch_size, -1, img_feats.shape[-1])  # [batch_size, 49, bi_hidden_size]

        return img_feats

    def get_text_feat(self, memory_bank, mask=None):
        # map memory bank into one feat vector using mask
        assert len(memory_bank.shape) == 3

        if self.text_pooling_type == 'max':
            text_feats = masked_max(memory_bank, mask, dim=1)
        elif self.text_pooling_type == 'avg':
            text_feats = masked_mean(memory_bank, mask, dim=1)
        return text_feats

    def get_img_feat(self, img_feats):
        # img_feats: [batch_size, 49, bi_hidden_size]
        assert len(img_feats.shape) == 3
        if self.img_pooling_type == 'max':
            img_feat, _ = torch.max(img_feats, dim=1)
        elif self.img_pooling_type == 'avg':
            img_feat = torch.mean(img_feats, dim=1)

        return img_feat

    def get_attr_feat(self, attr_feats):
        # map img_feats into one img_feat vector, shape of img_feats: [bs, img_len, img_dim]
        assert len(attr_feats.shape) == 3

        if self.attr_pooling_type == 'max':
            attr_feat, _ = torch.max(attr_feats, dim=1)
        elif self.attr_pooling_type == 'avg':
            attr_feat = torch.mean(attr_feats, dim=1)
        return attr_feat

    def forward(self, src, src_lens, src_mask, img, attr, return_last_state=True):
        # get text features
        memory_bank, encoder_final_state = self.get_text_memory_bank(src, src_lens, return_last_state)
        img_feats = None
        attr_feats = None
        combined_feat = None

        if self.use_img:
            img_feats = self.get_img_memory_bank(img)
            if 'fuse' in self.cur_model:
                img_feat = self.get_img_feat(img_feats)
                text_feat = self.get_text_feat(memory_bank, src_mask)
                combined_feat = img_feat + text_feat

            # single-attention
            if 'text_att' in self.cur_model or 'img_att' in self.cur_model:
                # ['img_text_att', 'text_img_att', 'text_img_add_text_att']
                if 'text_img_att' in self.cur_model:
                    text_feat = self.get_text_feat(memory_bank, src_mask)
                    combined_feat = self.text_img_att(text_feat, img_feats)
                elif 'img_text_att' in self.cur_model:
                    img_feat = self.get_img_feat(img_feats)
                    combined_feat = self.img_text_att(img_feat, memory_bank, src_mask)
                elif 'text_img_add_text_att' in self.cur_model:
                    img_feat = self.get_img_feat(img_feats)
                    text_feat = self.get_text_feat(memory_bank, src_mask)
                    img_text_feat = torch.cat([img_feat, text_feat], dim=1)
                    combined_feat = self.text_img_add_text_att(img_text_feat, memory_bank, src_mask)
                else:
                    raise NotImplementedError("To be implemented")

            # co-attention
            if 'co_att' in self.cur_model:
                # 'text_img_co_att'
                if 'text_img_co_att' in self.cur_model:
                    combined_feat = self.text_img_co_att(memory_bank, img_feats, src_mask)
                elif 'img_text_co_att' in self.cur_model:
                    combined_feat = self.img_text_co_att(img_feats, memory_bank, src_mask)
                else:
                    raise NotImplementedError("To be implemented")

        if self.use_attr:
            attr_feats = self.linear_attr(attr)

            # using texts and attrs
            if 'attr_text_att' in self.cur_model:
                attr_feat = self.get_attr_feat(attr_feats)
                combined_feat = self.attr_text_att(attr_feat, memory_bank, src_mask)

        # multi-head attention for all text, image, attribute
        if 'multi_head_att' in self.cur_model:
            if 'img_text_multi_head_att' in self.cur_model:
                enc_output = self.get_img_feat(img_feats)
                for enc_layer in self.img_text_multi_head_att:
                    enc_output, _ = enc_layer(enc_output, memory_bank, memory_bank, src_mask)
                combined_feat = enc_output

        return memory_bank, encoder_final_state, img_feats, attr_feats, combined_feat


class MultimodalMixture(nn.Module):
    def __init__(self, opt):
        """Initialize model."""
        super(MultimodalMixture, self).__init__()
        self.data_path = opt.data_path
        self.emb_path = opt.emb_path
        self.bidirectional = opt.bidirectional
        self.vocab_size = opt.vocab_size
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_size = opt.hidden_size
        self.bi_hidden_size = self.num_directions * opt.hidden_size

        self.cur_model = opt.cur_model
        self.copy_attn = opt.copy_attn

        self.use_text = opt.use_text
        self.use_img = opt.use_img
        self.use_attr = opt.use_attr

        self.img_ext_model = opt.img_ext_model

        self.bos_idx = opt.bos_idx
        self.eos_idx = opt.eos_idx
        self.unk_idx = opt.eos_idx
        self.pad_idx = opt.pad_idx

        self.encoder = MultimodalEncoder(opt)

        self.decoder = RNNDecoder(opt.vocab_size, opt.emb_size, self.bi_hidden_size, self.bi_hidden_size,
                                  opt.copy_attn, opt.pad_idx, opt.dropout, self.cur_model)

        self.decoder.embedding.weight = self.encoder.embedding.weight
        print('The weights are shared by both encoder and decoder!\n')

        self.combine_pred = opt.combine_pred
        self.combine_pred_type = opt.combine_pred_type
        if self.combine_pred:
            self.device = opt.device
            self.vocab = opt.vocab
            self.trg_class_vocab = opt.trg_class_vocab

            if self.combine_pred_type == 'direct':
                self.mask_softmax = MaskedSoftmax(dim=1)
            else:
                self.linear_pred = nn.Linear(opt.emb_size, opt.bi_hidden_size)
            # self.pred_rnn = nn.GRU(input_size=opt.emb_size, hidden_size=opt.bi_hidden_size, num_layers=1,
            #                        bidirectional=opt.bidirectional, batch_first=True, dropout=opt.dropout)

        self.dropout = nn.Dropout(p=opt.dropout)

        self.linear_classifer_final = nn.Linear(opt.bi_hidden_size, opt.trg_class_vocab_size)

    def init_decoder_state(self, encoder_final_state):
        """
        :param encoder_final_state: [batch_size, self.bi_hidden_size]
        :return: [1, batch_size, decoder_size]
        """
        batch_size = encoder_final_state.size(0)
        decoder_init_state = encoder_final_state.unsqueeze(0).expand((1, batch_size, self.bi_hidden_size))
        # [dec_layers, batch_size, decoder_size]
        return decoder_init_state

    def run_decoder(self, h_t_init, trg, memory_bank, src_mask, max_num_oov, src_oov,
                    img_feats, attr_feats, combined_feat,
                    cls_pred=None, cls_pred_mask=None, cls_pred_oov=None):
        batch_size, max_src_len, memory_bank_size = list(memory_bank.size())

        decoder_dist_all = []
        attention_dist_all = []

        y_t_init = trg.new_ones(batch_size) * self.bos_idx
        max_target_length = trg.size(1)
        for t in range(max_target_length):
            if t == 0:
                h_t = h_t_init
                y_t = y_t_init
            else:
                h_t = h_t_next
                y_t = y_t_next
            decoder_dist, h_t_next, _, attn_dist, p_gen = \
                self.decoder(y_t, h_t, memory_bank, src_mask, max_num_oov, src_oov,
                             img_feats, attr_feats, combined_feat, cls_pred, cls_pred_mask, cls_pred_oov)
            decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
            attention_dist_all.append(attn_dist.unsqueeze(1))  # [batch, 1, src_seq_len]

            y_t_next = trg[:, t]  # [batch]

        decoder_dist_all = torch.cat(decoder_dist_all, dim=1)  # [batch_size, trg_len, vocab_size]
        attention_dist_all = torch.cat(attention_dist_all, dim=1)  # [batch_size, trg_len, src_len]

        if self.copy_attn:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size + max_num_oov))
        else:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size))
        assert attention_dist_all.size() == torch.Size((batch_size, max_target_length, max_src_len))
        return decoder_dist_all, h_t_next, attention_dist_all

    def encode_pred(self, output, n_best=5):
        pred_batch = []
        class_idx2word = self.trg_class_vocab.idx2word
        for b_id in range(output.shape[0]):
            arr = np.array(output[b_id].tolist())
            top_indices = arr.argsort()[-n_best:][::-1]
            preds = [class_idx2word[top_indices[i]] for i in range(n_best)]
            pred_str = ' <seg> '.join(preds)
            pred_idx = [self.vocab(word) for word in pred_str.split()]
            pred_batch.append(pred_idx)

        def pred_pad(input_list):
            input_list_lens = [len(l) for l in input_list]
            max_seq_len = max(input_list_lens)
            padded_batch = self.pad_idx * np.ones((len(input_list), max_seq_len))

            for j in range(len(input_list)):
                current_len = input_list_lens[j]
                padded_batch[j][:current_len] = input_list[j]

            padded_batch = torch.LongTensor(padded_batch)

            input_mask = torch.ne(padded_batch, self.pad_idx)
            input_mask = input_mask.type(torch.FloatTensor)

            return padded_batch, input_list_lens, input_mask

        padded_pred_batch, pred_lens, pred_mask = pred_pad(pred_batch)

        padded_pred_batch = padded_pred_batch.to(self.device)
        pred_mask = pred_mask.to(self.device)

        embed_pred = self.encoder.embedding(padded_pred_batch)
        # encoded_pred, _ = self.pred_rnn(embed_pred)
        encoded_pred = self.linear_pred(embed_pred)
        return encoded_pred, pred_lens, pred_mask, padded_pred_batch

    def get_cls_pred_logits(self, output, n_best=5):
        pred_batch = []
        logit_batch = []
        class_idx2word = self.trg_class_vocab.idx2word
        for b_id in range(output.shape[0]):
            arr = np.array(output[b_id].tolist())
            top_indices = arr.argsort()[-n_best:][::-1]
            preds = [class_idx2word[top_indices[i]] for i in range(n_best)]
            top_logits = [arr[i] for i in top_indices]
            pred_idx = []
            pred_logits = []
            for pred, logit in zip(preds, top_logits):
                for word in pred.split():
                    word_idx = self.vocab(word)
                    pred_idx.append(word_idx)
                    pred_logits.append(logit)

            pred_batch.append(pred_idx)

            logit_batch.append(pred_logits)

        def pred_pad(input_list):
            input_list_lens = [len(l) for l in input_list]
            max_seq_len = max(input_list_lens)
            padded_batch = self.pad_idx * np.ones((len(input_list), max_seq_len))

            for j in range(len(input_list)):
                current_len = input_list_lens[j]
                padded_batch[j][:current_len] = input_list[j]

            padded_batch = torch.LongTensor(padded_batch)

            input_mask = torch.ne(padded_batch, self.pad_idx)
            input_mask = input_mask.type(torch.FloatTensor)

            return padded_batch, input_list_lens, input_mask

        def float_pred_pad(input_list):
            input_list_lens = [len(l) for l in input_list]
            max_seq_len = max(input_list_lens)
            padded_batch = float('-inf') * np.ones((len(input_list), max_seq_len))

            for j in range(len(input_list)):
                current_len = input_list_lens[j]
                padded_batch[j][:current_len] = input_list[j]

            padded_batch = torch.FloatTensor(padded_batch)
            return padded_batch

        padded_pred_batch, pred_lens, pred_mask = pred_pad(pred_batch)
        padded_logit_batch = float_pred_pad(logit_batch)

        padded_pred_batch = padded_pred_batch.to(self.device)
        padded_logit_batch = padded_logit_batch.to(self.device)
        pred_mask = pred_mask.to(self.device)

        normalized_padded_logit_batch = self.mask_softmax(padded_logit_batch)

        return normalized_padded_logit_batch, pred_lens, pred_mask, padded_pred_batch

    def forward(self, src, src_lens, src_mask, src_oov, trg, max_num_oov, img=None, attr=None, only_classifier=False):
        # ============================= Encoding text, image, attribtues =========================
        memory_bank, encoder_final_state, img_feats, attr_feats, combined_feat = \
            self.encoder(src, src_lens, src_mask, img, attr)

        classifier_outputs = self.linear_classifer_final(combined_feat)

        if only_classifier:
            return classifier_outputs

        if self.combine_pred:
            if self.combine_pred_type == 'direct':
                cls_pred, _, cls_pred_mask, cls_pred_oov = self.get_cls_pred_logits(classifier_outputs)
            else:
                cls_pred, _, cls_pred_mask, cls_pred_oov = self.encode_pred(classifier_outputs)
        else:
            cls_pred = None
            cls_pred_mask = None
            cls_pred_oov = None

        # ======================================= Decoding =======================================
        h_t_init = self.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]

        decoder_dist_all, h_t_next, attention_dist_all = \
            self.run_decoder(h_t_init, trg, memory_bank, src_mask, max_num_oov, src_oov,
                             img_feats, attr_feats, combined_feat,
                             cls_pred, cls_pred_mask, cls_pred_oov)
        return decoder_dist_all, h_t_next, attention_dist_all, encoder_final_state, classifier_outputs