import os
from data_util.log import logger
import time
import torch as T
import torch.nn.functional as F
from model import Model
from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab
from train_util import get_enc_data, get_cuda, get_dec_data
from torch.distributions import Categorical
from rouge import Rouge
from numpy import random
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(123)
T.manual_seed(123)
if T.cuda.is_available():
    T.cuda.manual_seed_all(123)


class Train(object):
    def __init__(self, opt):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path,
                               self.vocab,
                               mode='train',
                               batch_size=config.batch_size,
                               single_pass=False)
        self.opt = opt
        self.start_id = self.vocab.word2id(data.START_DECODING)
        self.end_id = self.vocab.word2id(data.STOP_DECODING)
        self.pad_id = self.vocab.word2id(data.PAD_TOKEN)
        self.unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        time.sleep(5)

    def save_model(self, iter):
        save_path = config.save_model_path + "/%07d.tar" % iter
        T.save(
            {
                "iter": iter + 1,
                "model_dict": self.model.state_dict(),
                "trainer_dict": self.trainer.state_dict()
            }, save_path)

    def setup_train(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        self.trainer = T.optim.Adam(self.model.parameters(), lr=config.lr)
        start_iter = 0
        if self.opt.load_model is not None:
            load_model_path = os.path.join(config.save_model_path,
                                           self.opt.load_model)
            checkpoint = T.load(load_model_path)
            start_iter = checkpoint["iter"]
            self.model.load_state_dict(checkpoint["model_dict"])
            self.trainer.load_state_dict(checkpoint["trainer_dict"])
            print("Loaded model at " + load_model_path)
        if self.opt.new_lr is not None:
            self.trainer = T.optim.Adam(self.model.parameters(),
                                        lr=self.opt.new_lr)
        return start_iter

    def train_batch_MLE(self, enc_out, enc_hidden, enc_padding_mask, ct_e,
                        extra_zeros, enc_batch_extend_vocab, batch):
        ''' Calculate Negative Log Likelihood Loss for the given batch. In order to reduce exposure bias,
                pass the previous generated token as input with a probability of 0.25 instead of ground truth label
        Args:
        :param enc_out: Outputs of the encoder for all time steps (batch_size, length_input_sequence, 2*hidden_size)
        :param enc_hidden: Tuple containing final hidden state & cell state of encoder. Shape of h & c: (batch_size, hidden_size)
        :param enc_padding_mask: Mask for encoder input; Tensor of size (batch_size, length_input_sequence) with values of 0 for pad tokens & 1 for others
        :param ct_e: encoder context vector for time_step=0 (eq 5 in https://arxiv.org/pdf/1705.04304.pdf)
        :param extra_zeros: Tensor used to extend vocab distribution for pointer mechanism
        :param enc_batch_extend_vocab: Input batch that stores OOV ids
        :param batch: batch object
        '''
        dec_batch, max_dec_len, dec_lens, target_batch = get_dec_data(
            batch)  #Get input and target batchs for training decoder
        step_losses = []
        s_t = (enc_hidden[0], enc_hidden[1])  #Decoder hidden states
        x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(
            self.start_id))  #Input to the decoder
        prev_s = None  #Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None  #Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        for t in range(min(max_dec_len, config.max_dec_steps)):
            use_gound_truth = get_cuda((T.rand(len(enc_out)) > 0.25)).long(
            )  #Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
            x_t = use_gound_truth * dec_batch[:, t] + (
                1 - use_gound_truth
            ) * x_t  #Select decoder input based on use_ground_truth probabilities
            x_t = self.model.embeds(x_t)
            final_dist, s_t, ct_e, sum_temporal_srcs, prev_s = self.model.decoder(
                x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros,
                enc_batch_extend_vocab, sum_temporal_srcs, prev_s)
            target = target_batch[:, t]
            log_probs = T.log(final_dist + config.eps)
            step_loss = F.nll_loss(log_probs,
                                   target,
                                   reduction="none",
                                   ignore_index=self.pad_id)
            step_losses.append(step_loss)
            x_t = T.multinomial(final_dist, 1).squeeze(
            )  #Sample words from final distribution which can be used as input in next time step
            is_oov = (x_t >= config.vocab_size
                      ).long()  #Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t.detach() + (
                is_oov) * self.unk_id  #Replace OOVs with [UNK] token

        losses = T.sum(
            T.stack(step_losses, 1), 1
        )  #unnormalized losses for each example in the batch; (batch_size)
        batch_avg_loss = losses / dec_lens  #Normalized losses; (batch_size)
        mle_loss = T.mean(batch_avg_loss)  #Average batch loss
        return mle_loss

    def train_batch_RL(self, enc_out, enc_hidden, enc_padding_mask, ct_e,
                       extra_zeros, enc_batch_extend_vocab, article_oovs,
                       greedy):
        '''Generate sentences from decoder entirely using sampled tokens as input. These sentences are used for ROUGE evaluation
        Args
        :param enc_out: Outputs of the encoder for all time steps (batch_size, length_input_sequence, 2*hidden_size)
        :param enc_hidden: Tuple containing final hidden state & cell state of encoder. Shape of h & c: (batch_size, hidden_size)
        :param enc_padding_mask: Mask for encoder input; Tensor of size (batch_size, length_input_sequence) with values of 0 for pad tokens & 1 for others
        :param ct_e: encoder context vector for time_step=0 (eq 5 in https://arxiv.org/pdf/1705.04304.pdf)
        :param extra_zeros: Tensor used to extend vocab distribution for pointer mechanism
        :param enc_batch_extend_vocab: Input batch that stores OOV ids
        :param article_oovs: Batch containing list of OOVs in each example
        :param greedy: If true, performs greedy based sampling, else performs multinomial sampling
        Returns:
        :decoded_strs: List of decoded sentences
        :log_probs: Log probabilities of sampled words
        '''
        s_t = enc_hidden  #Decoder hidden states
        x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(
            self.start_id))  #Input to the decoder
        prev_s = None  #Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None  #Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        inds = []  #Stores sampled indices for each time step
        decoder_padding_mask = []  #Stores padding masks of generated samples
        log_probs = []  #Stores log probabilites of generated samples
        mask = get_cuda(
            T.LongTensor(len(enc_out)).fill_(1)
        )  #Values that indicate whether [STOP] token has already been encountered; 1 => Not encountered, 0 otherwise

        for t in range(config.max_dec_steps):
            x_t = self.model.embeds(x_t)
            probs, s_t, ct_e, sum_temporal_srcs, prev_s = self.model.decoder(
                x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros,
                enc_batch_extend_vocab, sum_temporal_srcs, prev_s)
            if greedy is False:
                multi_dist = Categorical(probs)
                x_t = multi_dist.sample()  #perform multinomial sampling
                log_prob = multi_dist.log_prob(x_t)
                log_probs.append(log_prob)
            else:
                _, x_t = T.max(probs, dim=1)  #perform greedy sampling
            x_t = x_t.detach()
            inds.append(x_t)
            mask_t = get_cuda(T.zeros(
                len(enc_out)))  #Padding mask of batch for current time step
            mask_t[
                mask ==
                1] = 1  #If [STOP] is not encountered till previous time step, mask_t = 1 else mask_t = 0
            mask[
                (mask == 1) + (x_t == self.end_id) ==
                2] = 0  #If [STOP] is not encountered till previous time step and current word is [STOP], make mask = 0
            decoder_padding_mask.append(mask_t)
            is_oov = (x_t >= config.vocab_size
                      ).long()  #Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t + (
                is_oov) * self.unk_id  #Replace OOVs with [UNK] token

        inds = T.stack(inds, dim=1)
        decoder_padding_mask = T.stack(decoder_padding_mask, dim=1)
        if greedy is False:  #If multinomial based sampling, compute log probabilites of sampled words
            log_probs = T.stack(log_probs, dim=1)
            log_probs = log_probs * decoder_padding_mask  #Not considering sampled words with padding mask = 0
            lens = T.sum(decoder_padding_mask,
                         dim=1)  #Length of sampled sentence
            log_probs = T.sum(
                log_probs, dim=1
            ) / lens  # (bs,)                                     #compute normalizied log probability of a sentence
        decoded_strs = []
        for i in range(len(enc_out)):
            id_list = inds[i].cpu().numpy()
            oovs = article_oovs[i]
            S = data.outputids2words(
                id_list, self.vocab,
                oovs)  #Generate sentence corresponding to sampled words
            try:
                end_idx = S.index(data.STOP_DECODING)
                S = S[:end_idx]
            except ValueError:
                S = S
            if len(
                    S
            ) < 2:  #If length of sentence is less than 2 words, replace it with "xxx"; Avoids setences like "." which throws error while calculating ROUGE
                S = ["xxx"]
            S = " ".join(S)
            decoded_strs.append(S)

        return decoded_strs, log_probs

    def reward_function(self, decoded_sents, original_sents):
        rouge = Rouge()
        try:
            scores = rouge.get_scores(decoded_sents, original_sents)
        except Exception:
            print(
                "Rouge failed for multi sentence evaluation.. Finding exact pair"
            )
            scores = []
            for i in range(len(decoded_sents)):
                try:
                    score = rouge.get_scores(decoded_sents[i],
                                             original_sents[i])
                except Exception:
                    print("Error occured at:")
                    print("decoded_sents:", decoded_sents[i])
                    print("original_sents:", original_sents[i])
                    score = [{"rouge-l": {"f": 0.0}}]
                scores.append(score[0])
        rouge_l_f1 = [score["rouge-l"]["f"] for score in scores]
        rouge_l_f1 = get_cuda(T.FloatTensor(rouge_l_f1))
        return rouge_l_f1

    # def write_to_file(self, decoded, max, original, sample_r, baseline_r, iter):
    #     with open("temp.txt", "w") as f:
    #         f.write("iter:"+str(iter)+"\n")
    #         for i in range(len(original)):
    #             f.write("dec: "+decoded[i]+"\n")
    #             f.write("max: "+max[i]+"\n")
    #             f.write("org: "+original[i]+"\n")
    #             f.write("Sample_R: %.4f, Baseline_R: %.4f\n\n"%(sample_r[i].item(), baseline_r[i].item()))

    def train_one_batch(self, batch, iter):
        enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, context = get_enc_data(
            batch)

        enc_batch = self.model.embeds(
            enc_batch)  #Get embeddings for encoder input
        enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)

        # -------------------------------Summarization-----------------------
        if self.opt.train_mle == "yes":  #perform MLE training
            mle_loss = self.train_batch_MLE(enc_out, enc_hidden,
                                            enc_padding_mask, context,
                                            extra_zeros,
                                            enc_batch_extend_vocab, batch)
        else:
            mle_loss = get_cuda(T.FloatTensor([0]))
        # --------------RL training-----------------------------------------------------
        if self.opt.train_rl == "yes":  #perform reinforcement learning training
            # multinomial sampling
            sample_sents, RL_log_probs = self.train_batch_RL(
                enc_out,
                enc_hidden,
                enc_padding_mask,
                context,
                extra_zeros,
                enc_batch_extend_vocab,
                batch.art_oovs,
                greedy=False)
            with T.autograd.no_grad():
                # greedy sampling
                greedy_sents, _ = self.train_batch_RL(enc_out,
                                                      enc_hidden,
                                                      enc_padding_mask,
                                                      context,
                                                      extra_zeros,
                                                      enc_batch_extend_vocab,
                                                      batch.art_oovs,
                                                      greedy=True)

            sample_reward = self.reward_function(sample_sents,
                                                 batch.original_abstracts)
            baseline_reward = self.reward_function(greedy_sents,
                                                   batch.original_abstracts)
            # if iter%200 == 0:
            #     self.write_to_file(sample_sents, greedy_sents, batch.original_abstracts, sample_reward, baseline_reward, iter)
            rl_loss = -(
                sample_reward - baseline_reward
            ) * RL_log_probs  #Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
            rl_loss = T.mean(rl_loss)

            batch_reward = T.mean(sample_reward).item()
        else:
            rl_loss = get_cuda(T.FloatTensor([0]))
            batch_reward = 0

    # ------------------------------------------------------------------------------------
        self.trainer.zero_grad()
        (self.opt.mle_weight * mle_loss +
         self.opt.rl_weight * rl_loss).backward()
        self.trainer.step()

        return mle_loss.item(), batch_reward

    def trainIters(self):
        iter = self.setup_train()
        count = mle_total = r_total = 0
        while iter <= config.max_iterations:
            batch = self.batcher.next_batch()
            try:
                mle_loss, r = self.train_one_batch(batch, iter)
            except KeyboardInterrupt:
                print(
                    "-------------------Keyboard Interrupt------------------")
                exit(0)

            mle_total += mle_loss
            r_total += r
            count += 1
            iter += 1

            if iter % 50 == 0:
                mle_avg = mle_total / count
                r_avg = r_total / count
                logger.info("iter:" + str(iter) + "  mle_loss:" +
                            "%.3f" % mle_avg + "  reward:" + "%.4f" % r_avg)
                count = mle_total = r_total = 0

            if iter % 5000 == 0:
                self.save_model(iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mle', type=str, default="yes")
    parser.add_argument('--train_rl', type=str, default="no")
    parser.add_argument('--mle_weight', type=float, default=1.0)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--new_lr', type=float, default=None)
    opt = parser.parse_args()
    opt.rl_weight = 1 - opt.mle_weight
    print(
        "Training mle: %s, Training rl: %s, mle weight: %.2f, rl weight: %.2f"
        % (opt.train_mle, opt.train_rl, opt.mle_weight, opt.rl_weight))
    print("intra_encoder:", config.intra_encoder, "intra_decoder:",
          config.intra_decoder)

    train_processor = Train(opt)
    train_processor.trainIters()
