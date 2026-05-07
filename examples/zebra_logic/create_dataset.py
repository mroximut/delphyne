import pandas as pd


def create_folio_dataset():
    splits = {
        "train": "folio_v2_train.jsonl",
        "validation": "folio_v2_validation.jsonl",
    }
    df = pd.read_json(
        "hf://datasets/yale-nlp/FOLIO/" + splits["validation"], lines=True
    )
    df.to_csv(
        "examples/zebra_logic/datasets--yale-nlp--FOLIO/folio_v2_validation.csv",
        index=False,
    )


if __name__ == "__main__":
    create_folio_dataset()
    pass
