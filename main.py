from src.core.learning.learning import learning
from src.utils.check_device import check_device
from src.core.segment.segment import segment
import sys


def main(mode=None):
    num_classes = 3
    if (mode == "learning"):
        data_dir = "./models"
        learning(data_dir, num_classes)
    elif (mode == "segment"):
        input_ply_path = "./data/opt.ply"
        output_ply_path = "./data/opt_segmented.ply"
        model_path = "./saved_models/pointnet_segmentation.pth"
        segment(input_ply_path, output_ply_path, model_path, num_classes)
    else:
        print("Invalid mode. Use 'learning' or 'segment'.")
    return


if __name__ == "__main__":
    # check_device()
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    main(mode)
