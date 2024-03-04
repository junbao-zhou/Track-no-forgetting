################################################################################
# training parameters
################################################################################
from tasks.semantic.modules.distill_loss import UnbiasedKnowledgeDistillationLoss, KnowledgeDistillationLoss


class train:
    class optimizer:
        lr = 0.01               # sgd learning rate
        wup_epochs = 1          # warmup during first XX epochs (can be float)
        momentum = 0.9          # sgd momentum
        # learning rate decay per epoch after initial cycle (from min lr)
        lr_decay = 0.99
        w_decay = 0.0001        # weight decay
    max_epochs = 160
    batch_size_per_GPU = 14  # batch size
    report_batch = 10        # every x batches, report loss
    report_epoch = 1        # every x epochs, report validation set
    save_summary = False    # Summary of weight histograms for tensorboard
    save_scans = True       # False doesn't save anything, True saves some
    # sample images (one per batch of the last calculated batch)
    # in log folder
    show_scans = False      # show scans during training
    is_use_base_model = True
    base_model = "/public/home/meijilin/zhoujunbao/SalsaNext/train_novel0/logs/2022-09-03-03:16:13base"
    novel_model = "/public/home/meijilin/zhoujunbao/SalsaNext/train_novel0/logs/2022-09-03-03:16:13base"
    task_name = "16-4"
    task_step = 1
    sample_number = 50

    is_lora = False
    is_freeze_backbone = False

    class loss:
        # nll loss weight = 1 / (class_frequencies + epsilon_w)
        epsilon_w = 0.001
        distill_name = UnbiasedKnowledgeDistillationLoss

        class coefficient:
            NLLLoss = 1
            Lovasz_softmax = 1
            Distill = 1

################################################################################
# postproc parameters
################################################################################


class post:
    class CRF:
        use = False
        train = True
        params = False  # this should be a dict when in use

    class KNN:
        use = True  # This parameter default is false

        params = {
            "knn" : 5,
            "search" : 5,
            "sigma" : 1.0,
            "cutoff" : 1.0,
        }
