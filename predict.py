import os
import tokenization
import pickle
from albert_ner_clinical_predict import file_based_convert_examples_to_features
from albert_ner_clinical_predict import NerProcessor,file_based_input_fn_builder,model_fn_builder
import tensorflow as tf
import modeling
bert_config_file = "albert_base_zh/albert_config_base.json"
init_checkpoint = "albert_base_zh/albert_model.ckpt"


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
    predict(text)