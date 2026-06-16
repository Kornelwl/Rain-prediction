from ingest import run_ingestion
from process import run_processing
from features import run_feature_engineering
from scoring import run_scoring
from evaluate import run_evaluation


def main():
    print("Starting Project 12: Rain Risk Scoring and Short-Term Prediction")
    print("=" * 70)

    run_ingestion()
    run_processing()
    run_feature_engineering()
    run_scoring()
    run_evaluation()

    print("=" * 70)
    print("Project finished successfully.")


if __name__ == "__main__":
    main()