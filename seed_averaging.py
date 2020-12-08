from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import rmsle
from src.utils import get_seed_average_parser, make_submission, save_json

if __name__ == "__main__":

    parser = get_seed_average_parser()
    args = parser.parse_args()

    output = Path(args.output)
    input_dir = Path("input/")
    y = pd.read_feather(input_dir / "train.ftr")["Global_Sales"]

    oof_preds = []
    for oof_pred in np.sort(glob(str(output) + "/*/oof_preds.npy")):
        oof_preds.append(np.load(oof_pred))
    oof_pred = np.mean(oof_preds, axis=0)
    oof_score = rmsle(y, oof_pred)
    print(f"oof_score: {oof_score}")
    output_dict = {"oof_score": oof_score}

    test_preds = []
    for test_pred in np.sort(glob(str(output) + "/*/test_preds.npy")):
        test_preds.append(np.load(test_pred))
    test_pred = np.mean(test_preds, axis=0)

    # ===============================
    # === Make submission
    # ===============================

    sample_submission = pd.read_csv(input_dir / "sample_submission.csv")
    submission_df = make_submission(test_pred, sample_submission)

    # ===============================
    # === Save
    # ===============================

    save_path = output / "output.json"
    save_json(output_dict, save_path)

    np.save(output / "oof_preds.npy", oof_pred)

    np.save(output / "test_preds.npy", test_pred)

    submission_df.to_csv(output / "submission.csv", index=False)
