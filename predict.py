import os
import tokenization
import pickle
from albert_ner_clinical_predict import file_based_convert_examples_to_features
from albert_ner_clinical_predict import NerProcessor,file_based_input_fn_builder,model_fn_builder
import tensorflow as tf
import modeling
from collections import defaultdict
bert_config_file = "albert_base_zh/albert_config_base.json"
init_checkpoint = "albert_base_zh/albert_model.ckpt"


class predictModel:
    def __init__(self,model_dir,do_lower_case=True,max_seq_length=128,
            bert_config_file=bert_config_file,init_checkpoint=init_checkpoint):
        self.model_dir = model_dir
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.bert_config_file = bert_config_file
        self.init_checkpoint = init_checkpoint
        self.processor = NerProcessor()
        self.label_list = self.processor.get_labels()
        self.vocab_file = os.path.join(self.model_dir, "vocab.txt")
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)
        self.predict_batch_size = 8
        self._load_estimator()
        self._load_label2id()

    def _load_estimator(self):
        use_tpu = False
        num_train_steps = None
        num_warmup_steps = None
        learning_rate = 2e-5
        bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(self.label_list) + 1,
            init_checkpoint=self.init_checkpoint,
            learning_rate=learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=use_tpu,
            use_one_hot_embeddings=use_tpu)
        tpu_cluster_resolver = None
        master = None
        save_checkpoints_steps = 1000
        iterations_per_loop = 1000
        num_tpu_cores = 8
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        train_batch_size = 32
        eval_batch_size = 8
        predict_batch_size = 8
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=master,
            model_dir=self.model_dir,
            save_checkpoints_steps=save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=iterations_per_loop,
                num_shards=num_tpu_cores,
                per_host_input_for_training=is_per_host))

        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            predict_batch_size=predict_batch_size)

    def _load_label2id(self):
        with open('albert_base_ner_checkpoints/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            self.id2label = {value: key for key, value in label2id.items()}

    def label2entity(self,tokens, targets):
        suffix2name = {
            'DISEASE': '疾病',
            'SIGNS': '症状',
            'TREATMENT': '治疗方式',
            "BODY":"身体部位",
            "CHECK":"检查方式"
        }
        name2entities = defaultdict(list)
        entity, last_suffix = None, None
        for char, label in zip(tokens, targets):
            if '-' in label:
                prefix, suffix = label.split('-')
                if prefix == 'B':
                    if last_suffix and entity:
                        name2entities[suffix2name[last_suffix]].append(entity)
                    entity, last_suffix = char, suffix
                else:
                    if suffix == last_suffix and entity:
                        entity += char
                    else:
                        entity, last_suffix = None, None
            else:
                if last_suffix and entity:
                    name2entities[suffix2name[last_suffix]].append(entity)
                entity, last_suffix = None, None

        return name2entities

    def predict(self,text):
        predict_drop_remainder = False
        tmp_result_file = "/tmp/tmp_predict.tf_record"
        if os.path.exists(tmp_result_file):
            os.remove(tmp_result_file)

        predict_examples_ = self.processor.get_predict_examples(text)

        tokens = ["[CLS]"] + list(text) + ["[SEP]"]
        file_based_convert_examples_to_features(predict_examples_, self.label_list,
                                                self.max_seq_length, self.tokenizer,
                                                tmp_result_file, mode="test")
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples_))
        tf.logging.info("  Batch size = %d", self.predict_batch_size)
        predict_input_fn = file_based_input_fn_builder(
            input_file=tmp_result_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = self.estimator.predict(input_fn=predict_input_fn)
        target = []
        for prediction in result:
            for id in prediction:
                if id != 0:
                    target.append(self.id2label[id])


        return target,tokens


def predict(text,model_dir,do_lower_case=True,max_seq_length=128,
            bert_config_file=bert_config_file,init_checkpoint=init_checkpoint):
    use_tpu = False
    processor = NerProcessor()
    label_list = processor.get_labels()
    vocab_file = os.path.join(model_dir,"vocab.txt")
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    num_train_steps = None
    num_warmup_steps = None
    learning_rate = 2e-5
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_tpu)
    tpu_cluster_resolver = None
    master = None
    save_checkpoints_steps = 1000
    iterations_per_loop = 1000
    num_tpu_cores = 8
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    train_batch_size = 32
    eval_batch_size = 8
    predict_batch_size  = 8
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=master,
        model_dir=model_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_tpu_cores,
            per_host_input_for_training=is_per_host))

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size = train_batch_size,
        eval_batch_size = eval_batch_size,
        predict_batch_size= predict_batch_size)
    # token_path = os.path.join(output_dir, "token_test.txt")
    with open('albert_base_ner_checkpoints/label2id.pkl', 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}
    predict_drop_remainder = False
    tmp_result_file = "/tmp/tmp_predict.tf_record"
    if os.path.exists(tmp_result_file):
        os.remove(tmp_result_file)

    predict_examples_ = processor.get_predict_examples(text)
    file_based_convert_examples_to_features(predict_examples_, label_list,
                                            max_seq_length, tokenizer,
                                            tmp_result_file, mode="test")
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples_))
    tf.logging.info("  Batch size = %d", predict_batch_size)
    predict_input_fn_ = file_based_input_fn_builder(
        input_file=tmp_result_file,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result_ = estimator.predict(input_fn=predict_input_fn_)
    target = []
    for prediction in result_:
        for id in prediction:
            if id != 0:
                target.append(id2label[id])
    return target

if __name__ == "__main__":
    text = "有可能得了艾滋病"
    model_dir = "albert_base_ner_checkpoints/"
    # print(predict(text,model_dir=model_dir))
    model = predictModel(model_dir=model_dir)
    print(model.predict(text))