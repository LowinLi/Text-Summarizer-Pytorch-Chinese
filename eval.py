import os
import time
from data_util.log import logger
import torch as T
import rouge
from model import Model
from data_util import config, data
from data_util.batcher import Batcher, Example, Batch
from data_util.data import Vocab
from beam_search import beam_search
from train_util import get_enc_data
from rouge import Rouge
import argparse
import jieba
if config.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


class Evaluate(object):
    def __init__(self, data_path, opt, batch_size=config.batch_size):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(data_path,
                               self.vocab,
                               mode='eval',
                               batch_size=batch_size,
                               single_pass=True)
        self.opt = opt
        time.sleep(5)

    def setup_valid(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        if config.cuda:
            checkpoint = T.load(
            os.path.join(config.demo_model_path, self.opt.load_model))
        else:
            checkpoint = T.load(os.path.join(config.demo_model_path, self.opt.load_model), map_location='cpu')
        self.model.load_state_dict(checkpoint["model_dict"])

    def print_original_predicted(self, decoded_sents, ref_sents, article_sents,
                                 loadfile):
        filename = "test_" + loadfile.split(".")[0] + ".txt"

        with open(os.path.join("data", filename), "w") as f:
            for i in range(len(decoded_sents)):
                f.write("article: " + article_sents[i] + "\n")
                f.write("ref: " + ref_sents[i] + "\n")
                f.write("dec: " + decoded_sents[i] + "\n\n")

    def evaluate_batch(self, article):

        self.setup_valid()
        batch = self.batcher.next_batch()
        start_id = self.vocab.word2id(data.START_DECODING)
        end_id = self.vocab.word2id(data.STOP_DECODING)
        unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        decoded_sents = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()
        while batch is not None:
            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(
                batch)
            with T.autograd.no_grad():
                enc_batch = self.model.embeds(enc_batch)
                enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)

            #-----------------------Summarization----------------------------------------------------
            with T.autograd.no_grad():
                pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask,
                                       ct_e, extra_zeros,
                                       enc_batch_extend_vocab, self.model,
                                       start_id, end_id, unk_id)

            for i in range(len(pred_ids)):
                decoded_words = data.outputids2words(pred_ids[i], self.vocab,
                                                     batch.art_oovs[i])
                if len(decoded_words) < 2:
                    decoded_words = "xxx"
                else:
                    decoded_words = " ".join(decoded_words)
                decoded_sents.append(decoded_words)
                abstract = batch.original_abstracts[i]
                article = batch.original_articles[i]
                ref_sents.append(abstract)
                article_sents.append(article)

            batch = self.batcher.next_batch()

        load_file = self.opt.load_model

        if article:
            self.print_original_predicted(decoded_sents, ref_sents,
                                          article_sents, load_file)

        scores = rouge.get_scores(decoded_sents, ref_sents)
        rouge_1 = sum([x["rouge-1"]["f"] for x in scores]) / len(scores)
        rouge_2 = sum([x["rouge-2"]["f"] for x in scores]) / len(scores)
        rouge_l = sum([x["rouge-l"]["f"] for x in scores]) / len(scores)
        logger.info(load_file + " rouge_1:" + "%.4f" % rouge_1 + " rouge_2:" + "%.4f" % rouge_2 + " rouge_l:" + "%.4f" % rouge_l)


class Demo(Evaluate):
    def __init__(self, opt):
        self.vocab = Vocab(config.demo_vocab_path, config.demo_vocab_size)
        self.opt = opt
        self.setup_valid()

    def evaluate(self, article, ref):
        dec = self.abstract(article)
        scores = rouge.get_scores(dec, ref)
        rouge_1 = sum([x["rouge-1"]["f"] for x in scores]) / len(scores)
        rouge_2 = sum([x["rouge-2"]["f"] for x in scores]) / len(scores)
        rouge_l = sum([x["rouge-l"]["f"] for x in scores]) / len(scores)
        return {
            'dec': dec,
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l
        }

    def abstract(self, article):
        start_id = self.vocab.word2id(data.START_DECODING)
        end_id = self.vocab.word2id(data.STOP_DECODING)
        unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        example = Example(' '.join(jieba.cut(article)), '', self.vocab)
        batch = Batch([example], self.vocab, 1)
        enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(
            batch)
        with T.autograd.no_grad():
            enc_batch = self.model.embeds(enc_batch)
            enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)
            pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e,
                                   extra_zeros, enc_batch_extend_vocab,
                                   self.model, start_id, end_id, unk_id)

        for i in range(len(pred_ids)):
            decoded_words = data.outputids2words(pred_ids[i], self.vocab,
                                                 batch.art_oovs[i])
            decoded_words = " ".join(decoded_words)
        return decoded_words


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        type=str,
                        default="validate",
                        choices=["validate", "test", "demo"])
    parser.add_argument("--start_from", type=str, default="0005000.tar")
    parser.add_argument("--load_model", type=str, default='0060000.tar')
    opt = parser.parse_args()

    if opt.task == "validate":
        saved_models = os.listdir(config.save_model_path)
        saved_models.sort()
        file_idx = saved_models.index(opt.start_from)
        saved_models = saved_models[file_idx:]
        for f in saved_models:
            opt.load_model = f
            eval_processor = Evaluate(config.valid_data_path, opt)
            eval_processor.evaluate_batch(False)
    elif opt.task == "test":
        eval_processor = Evaluate(config.test_data_path, opt)
        eval_processor.evaluate_batch(True)
    else:
        demo_processor = Demo(opt)
        logger.info(
            demo_processor.abstract(
                '就在对接货币基金的互联网理财产品诞生一周年的时候余额宝们的收益率破5已悄然成常态而数据显示今年截至6月6日市场上654只债券基金AB类份额分开计算平均收益率达451%且有248只债基产品收益率超过5%占比38%'
            ))
