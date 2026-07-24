"""Frozen snapshot of the BERT settings verified in the GoEmotions replication.

These are *copies*, not imports. This folder must run on its own — on a server,
in a container, wherever — without `../bertgo` being present. The values below
are the reference the incremental experiments are read against, so they are
recorded here as literals with their provenance.

Every entry traces to Demszky et al. (2020) §5.3 or to a flag in
google-research/goemotions/bert_classifier.py, and each was verified against
Google's original implementation (tokenizer over all 43,410 training texts,
label construction, the 122 metric keys of calculate_metrics.py, the optimizer,
and the LR schedule — all zero difference).

If you deliberately change one of these, change it here too and say why.
`check_config.py --against ../bertgo` re-checks this snapshot against a live
bertgo checkout when one happens to be available.
"""

# emogo field name -> (value, bertgo field name, provenance)
BERT_REFERENCE = {
    "model_name": (
        "bert-base-cased", "model_name",
        "README: 'we use the cased base model'",
    ),
    "do_lower_case": (
        False, "do_lower_case",
        "flag do_lower_case; cased model",
    ),
    "max_seq_length": (
        50, "max_seq_length",
        "flag max_seq_length",
    ),
    "classifier_dropout": (
        0.1, "classifier_dropout",
        "create_model(): tf.nn.dropout(keep_prob=0.9)",
    ),
    "batch_size": (
        16, "train_batch_size",
        "flag train_batch_size; §5.3 'a small batch size of 16'",
    ),
    "eval_batch_size": (
        64, "eval_batch_size",
        "not in the original (it reuses train_batch_size); metrics are "
        "aggregated over the full split so this is free",
    ),
    "learning_rate": (
        5e-5, "learning_rate",
        "flag learning_rate; §5.3",
    ),
    "warmup_proportion": (
        0.1, "warmup_proportion",
        "flag warmup_proportion",
    ),
    "weight_decay_rate": (
        0.01, "weight_decay_rate",
        "bert/optimization.py AdamWeightDecayOptimizer",
    ),
    "adam_beta1": (0.9, "adam_beta1", "bert/optimization.py"),
    "adam_beta2": (0.999, "adam_beta2", "bert/optimization.py"),
    "adam_epsilon": (
        1e-6, "adam_epsilon",
        "bert/optimization.py; note eps sits outside the sqrt",
    ),
    "max_grad_norm": (
        1.0, "max_grad_norm",
        "create_optimizer(): clip_by_global_norm(1.0)",
    ),
    "exclude_from_weight_decay": (
        ["LayerNorm", "layer_norm", "bias"], "exclude_from_weight_decay",
        "bert/optimization.py exclude_from_weight_decay",
    ),
    "drop_last": (
        True, "drop_last",
        "train input_fn: drop_remainder=True",
    ),
    "fp16": (
        False, "fp16",
        "the verified bertgo run was fp32; keep the incremental runs on the "
        "same footing",
    ),
}

# Settings that differ from bertgo on purpose. Reported, never asserted.
INTENTIONAL_DIFFERENCES = {
    "epochs": "bertgo trains 4.0 epochs once; emogo trains 4 epochs *per task* "
              "(init_epochs/epochs), each with its own warmup+decay cycle.",
    "seed": "EmoGrowth/PyCIL convention is 1993; bertgo used 42. Affects init "
            "and shuffling only — keep it fixed across methods within a table.",
    "threshold": "bertgo binarises at probability > 0.3 (GoEmotions Table 4); "
                 "emogo binarises at logit > 0, i.e. probability > 0.5, the "
                 "EmoGrowth utils/metrics.py convention. Raw logits are saved "
                 "per task so either can be recomputed.",
    "data": "bertgo uses train/dev/test; emogo uses train/test only, since the "
            "incremental protocol has no per-task model selection.",
}
