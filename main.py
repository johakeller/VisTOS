"""
Entry point of the program. The function run_training() starts pretraining
and fine-tuning, as well as evaluation via passed arguments.
"""

import os
import sys

import finetuning
import params
import pretraining
import utils
from model import vistos_att_model, vistos_conv_model


def run_training(
    mode: str,
    dataset=None,
    vis_field_size: int = params.VIS_FIELDS[0],
    model_type: str = "att",
):
    """
    Entry point for pretraining and fine-tuning. Managing training behaviour
    according to passed arguments: model architecture, training mode, dataset,
    and visual field size.
    """

    # check available training modes
    assert (
        mode in params.MODES
    ), f"Mode {mode} not implemented. Available modes are: {params.MODES}"

    # define correct model class
    model_class = vistos_att_model if model_type == "att" else vistos_conv_model
    # path to pre-trained model
    pretrained_model_path = os.path.join(
        params.OUTPUT, f"{model_type}_model_weights_vf{vis_field_size}.pth"
    )

    # pre-training: run x epochs of training, followed by validation, save model to file
    if mode == "pretrain":
        # create model instance (params defined in module params)
        model = model_class.VistosTimeSeriesSeq2Seq.construct(
            vis_field_size=vis_field_size, dropout=params.DROPOUT
        ).to(params.DEVICE)
        # initialize PreTraining object
        pre_training = pretraining.PreTraining(vis_field_size=vis_field_size)
        # run pretraining procedure with model
        pre_training.init_pretraining(
            model, checkpointing=params.CHECKPOINT, model_type=model_type
        )

    # fine-tuning
    elif mode == "finetune" and (dataset is not None):
        if os.path.isfile(pretrained_model_path):
            # take pre-trained Seq2Seq instance to construct fine-tuning model
            fine_tuning = finetuning.FineTuning(
                vis_field_size=vis_field_size, dataset=dataset
            )
            # run fine-tuning procedure with model
            fine_tuning.init_finetuning(
                model_class,
                params.FT_CHECKPOINT,
                False,
                model_type=model_type,
                vis_field_size=vis_field_size,
            )
        else:
            raise FileNotFoundError(f"{pretrained_model_path} not found.")

    # evaluation of fine-tuned model (must be in cache)
    elif mode == "eval" and (dataset is not None):
        if os.path.isfile(pretrained_model_path) and (dataset is not None):
            # take pre-trained Seq2Seq instance to construct fine-tuning model
            fine_tuning = finetuning.FineTuning(
                vis_field_size=vis_field_size, dataset=dataset
            )
            # run evaluation procedure with model
            fine_tuning.init_finetuning(
                model_class,
                params.FT_CHECKPOINT,
                True,
                model_type=model_type,
                vis_field_size=vis_field_size,
            )
        else:
            raise FileNotFoundError(f"{pretrained_model_path} not found.")
    # no dataset indicated or mode unknown
    else:
        raise ValueError(f"Training mode {mode} unkown or .")


def main(args):
    """
    Pretraining and fine-tuning are started from here with two available model
    architectures in several visual field sizes on one pretraining dataset and
    two fine-tunnig datasets. Can start pretrainig, fine-tuning, or an valuation
    of a trained model. The argument 'att' denotes the attention-based spatial
    encoding architecture, 'conv' refers to the convolutional spatial encoding
    architecture. The argument 'pastis' is the PASTIS-R fine-tuning dataset,
    'mtcc' is the multi-temporal-crop-classification (MTCC) dataset.
    Pass arguments in the form:

    model architecture: 'conv'/ 'att', training type 'finetune/eval', and
    dataset: 'pastis'/'mtcc', visual field size: '1' (only for att)/'3', '5', 7
    (only for conv) for a fine-tuning setup

    model architecture: 'conv'/ 'att', 'pretrain', and visual field size: '1'
    (only for att)/'3', '5', 7 (only for conv) for a pretraining setup
    """

    # create output directory (output folder, images and chache withins)
    utils.init_output()
    # define model type -> convolution or attention
    model_type = args[0]
    # pretraining
    if args[1] == "pretrain":
        # define visual field size
        field_size = int(args[2])
        # start pretraining
        run_training(mode="pretrain", vis_field_size=field_size, model_type=model_type)
    # finetuning
    elif args[1] == "finetune":
        dataset = args[2]
        field_size = int(args[3])
        # start finetuning
        run_training(
            mode="finetune",
            dataset=dataset,
            vis_field_size=field_size,
            model_type=model_type,
        )
    # evaluation
    elif args[1] == "eval":
        dataset = args[2]
        field_size = int(args[3])
        # start evaluation
        run_training(
            mode="eval",
            dataset=dataset,
            vis_field_size=field_size,
            model_type=model_type,
        )


if __name__ == "__main__":
    passed_args = sys.argv[1:]
    # first argument: model type
    models = ["att", "conv"]
    # second argument: training mode
    modes = ["finetune", "pretrain", "eval"]
    # third argument: datasets
    datasets = ["pastis", "mtcc"]
    # visual field sizes
    field_sizes = params.VIS_FIELDS
    usage = "Usage: $ python main.py ([att] ([finetune]|[eval] [pastis]|[mtcc] [1|3|5]) | ([pretrain] [1|3|5])) | ([conv] ([finetune]|[eval] [pastis]|[mtcc] [1|3|5|7]) | ([pretrain] [1|3|5|7]))"
    # check user input: models
    if passed_args[0] not in models:
        print(usage)
        sys.exit(1)

    # training mode
    if passed_args[1] not in modes:
        print(usage)
        sys.exit(1)

    # if pretrain then following argument must be field size
    if passed_args[1] == "pretrain":
        if len(passed_args) != 3 or int(passed_args[2]) not in field_sizes:
            print(usage)
            sys.exit(1)

    # if finetune then following argument must be dataset and following argument must be field size
    if passed_args[1] == "finetune":
        if (
            len(passed_args) != 4
            or passed_args[2] not in datasets
            or int(passed_args[3]) not in field_sizes
        ):
            print(usage)
            sys.exit(1)

    # if eval then following argument must be dataset and following argument must be field size
    if passed_args[1] == "eval":
        if (
            len(passed_args) != 4
            or passed_args[2] not in datasets
            or int(passed_args[3]) not in field_sizes
        ):
            print(usage)
            sys.exit(1)

    # start main()
    main(passed_args)
