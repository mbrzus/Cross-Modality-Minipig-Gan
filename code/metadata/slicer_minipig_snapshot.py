import slicer
import os
from pathlib import Path
import ScreenCapture


def snapshot(out_path: str, im_id: str, t1w_path: str):
    """
    The function takes a snapshot of slicer view and saves it as a .png file

    :param out_path: path to a directory to save snapshots
    :param im_id: name for the snapshot image (ex. sub-*ses-*run-*)
    in current state the saved snapshots name will be appended with '_RP_snapshot" and saved as a .png image
    """
    slicer.mrmlScene.Clear(False)  # clear slicer scene
    slicer.util.loadVolume(t1w_path)  # load volume image
    cap = ScreenCapture.ScreenCaptureLogic()
    cap.showViewControllers(False)
    cap.captureImageFromView(None, f"{out_path}/{im_id}_snapshot.png")
    cap.showViewControllers(True)


if __name__ == "__main__":

    # path to data
path = "/home/mbrzus/programming/gitlab/MiniPigMALF/BIDS"
out_sanspshot_path = "/home/mbrzus/programming/Cross-Modality-Minipig-Gan/code/metadata/minipig_snapshots/t1w"

# parse through data
for file in Path(path).glob("sub-*/ses-*/anat/*T1w.nii.gz"):
    t1_path = str(file)
    t1_name = t1_path.replace(f"{str(file.parent)}/", '')
    print(t1_name)
    im_name = t1_name[:-22]
    print(im_name)
    snapshot(out_sanspshot_path, im_name, t1_path)
