"""Data drift monitoring with Evidently.

Compares the REFERENCE data (what the model was trained on) against
CURRENT data (newer customers, e.g. last week's signups) and produces
an HTML report flagging which features have drifted.

Run from the project root:
    python -m src.monitoring
"""
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split

from src import config
from src.data import load_data

REPORT_PATH = config.ROOT / "monitoring" / "drift_report.html"


def main():
    df = load_data()

    # We don't have real "new" data yet, so we simulate it: split the
    # dataset into a reference half and a current half. In production,
    # `current` would instead be a fresh export of recent customers.
    reference, current = train_test_split(
        df, test_size=0.5, random_state=config.RANDOM_STATE
    )

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(REPORT_PATH))
    print(f"Drift report saved to {REPORT_PATH}")
    print("Open that file in a browser to see the results.")


if __name__ == "__main__":
    main()