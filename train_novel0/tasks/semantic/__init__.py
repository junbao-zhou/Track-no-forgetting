import sys
import os.path as osp
TRAIN_PATH = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
DEPLOY_PATH = "../../../deploy"
sys.path.insert(0, TRAIN_PATH)
