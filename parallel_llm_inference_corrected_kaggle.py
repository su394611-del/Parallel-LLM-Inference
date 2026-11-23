"""Run the batch inference CLI under the legacy Kaggle script name."""

from parallel_llm_inference import main


if __name__ == "__main__":
    raise SystemExit(main())
