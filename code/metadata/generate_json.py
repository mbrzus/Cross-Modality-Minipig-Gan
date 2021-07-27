from argparse import ArgumentParser
from pathlib import Path
from random import shuffle
from datetime import datetime
from tqdm import tqdm
import json

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# usage:
# python generate_json.py --image-dir /Shared/sinapse/chdi_bids/DELETEME/PREDICTHD_BIDS/derivatives/physicalACPC/ --splits 0.8 0.1 0.1 --out-dir .


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="BIDS Directory of images")
    parser.add_argument("--t1w-glob", default="*T1w.nii.gz")
    parser.add_argument("--t2w-glob", default="*T2w.nii.gz")
    parser.add_argument(
        "--splits",
        nargs="+",
        type=float,
        help="train, validation, and test percentages as a list in order (train, validation, test) ex: --splits 0.8 0.1 0.1",
    )
    parser.add_argument("--out-dir", default="./splits", help="output directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    def assert_is_dir(d):
        assert Path(str(d)).is_dir(), f"{d} must be a directory"

    assert_is_dir(args.image_dir)

    out_dir = Path(str(args.out_dir))
    if not out_dir.is_dir():
        out_dir.mkdir()

    assert_is_dir(args.out_dir)
    assert len(args.splits) == 3, "splits must be length three"

    if args.verbose:
        print(args)

    return args


def extract_metadata(session_path):
    # TODO: get metadata based on the session path
    return {}


def subject_to_json(subject_path, args):
    subject_json = {}
    sessions = list(subject_path.glob("ses*"))
    for session in sessions:
        subject_json[session.name] = {}
        subject_json[session.name]["t1w"] = sorted(
            [str(path) for path in session.glob(args.t1w_glob)]
        )
        subject_json[session.name]["t2w"] = sorted(
            [str(path) for path in session.glob(args.t2w_glob)]
        )
        subject_json[session.name]["meta"] = extract_metadata(session)

    return subject_json


def generate_json(argv):
    # glob all subject paths
    subjects = list(Path(argv.image_dir).glob("sub*"))
    shuffle(subjects)  # shuffle the subject order

    # split on subjects
    n = len(subjects)
    train_percentage_cutoff = argv.splits[0] * n
    validation_percentage_cutoff = (argv.splits[0] + argv.splits[1]) * n

    main_dict = {"train": {}, "validation": {}, "test": {}}
    for i, subject in enumerate(tqdm(subjects)):
        if i < train_percentage_cutoff:
            main_dict["train"][subject.name] = subject_to_json(subject, argv)
        elif i < validation_percentage_cutoff:
            main_dict["validation"][subject.name] = subject_to_json(subject, argv)
        else:
            main_dict["test"][subject.name] = subject_to_json(subject, argv)

    strfmt_splits = [str(a).replace(".", "") for a in argv.splits]
    # out_file = (
    #     Path(argv.out_dir)
    #     / f"{Path(argv.image_dir).name}_TR{strfmt_splits[0]}_VAL{strfmt_splits[1]}_TE{strfmt_splits[2]}_{TIMESTAMP}.json"
    # )

    out_file = Path(argv.out_dir) / f"structure.json"

    with open(out_file, "w") as f:
        json.dump(main_dict, f, indent=4)

    if argv.verbose:
        print(f"json written to: {out_file}")


if __name__ == "__main__":
    generate_json(parse_args())
