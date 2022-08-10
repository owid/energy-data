"""Generate OWID energy dataset from most up-to-date sources.

Running this script will generate the full energy dataset in three different formats:
* owid-energy-data.csv
* owid-energy-data.xlsx
* owid-energy-data.json

"""

import argparse
import json
from typing import List

import pandas as pd
from owid import catalog

from scripts.shared import OUTPUT_DIR

# Define paths to output files.
OUTPUT_CSV_FILE = OUTPUT_DIR / "owid-energy-data.csv"
OUTPUT_EXCEL_FILE = OUTPUT_DIR / "owid-energy-data.xlsx"
OUTPUT_JSON_FILE = OUTPUT_DIR / "owid-energy-data.json"
CODEBOOK_FILE = OUTPUT_DIR / "owid-energy-codebook.csv"
# Details of the latest owid-energy dataset from etl.
ENERGY_DATASET_TABLE = "owid_energy"


def df_to_json(complete_dataset: pd.DataFrame, output_path: str, static_columns: List[str]) -> None:
    megajson = {}

    # Round all values to 3 decimals.
    rounded_cols = [col for col in list(complete_dataset) if col not in ("country", "year", "iso_code")]
    complete_dataset[rounded_cols] = complete_dataset[rounded_cols].round(3)

    for _, row in complete_dataset.iterrows():

        row_country = row["country"]
        row_dict_static = row.drop("country")[static_columns].dropna().to_dict()
        row_dict_dynamic = row.drop("country").drop(static_columns).dropna().to_dict()

        if row_country not in megajson:
            megajson[row_country] = row_dict_static
            megajson[row_country]["data"] = [row_dict_dynamic]
        else:
            megajson[row_country]["data"].append(row_dict_dynamic)

    with open(output_path, "w") as file:
        file.write(json.dumps(megajson, indent=4))


def prepare_data(table: catalog.Table) -> pd.DataFrame:
    # Create a dataframe with a dummy index from the table.
    df = pd.DataFrame(table).reset_index()

    # Sort rows and columns conveniently.
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    first_columns = ["country", "year", "iso_code"]
    df = df[first_columns + [column for column in df.columns if column not in first_columns]]

    return df


def prepare_codebook(table: catalog.Table) -> pd.DataFrame:
    table = table.reset_index()

    # Gather column names, descriptions and sources from the variables' metadata.
    metadata = {"column": [], "description": [], "source": []}
    for column in table.columns:
        metadata["column"].append(column)
        metadata["description"].append(table[column].metadata.description)
        metadata["source"].append(table[column].metadata.sources[0].name)

    # Create a dataframe with the gathered metadata and sort conveniently by column name.
    codebook = pd.DataFrame(metadata).sort_values("column").reset_index(drop=True)

    return codebook


def main() -> None:
    #
    # Load data.
    #
    # Load OWID-energy dataset from the catalog.
    table = catalog.find(ENERGY_DATASET_TABLE, namespace="energy", channels=["garden"]).\
        sort_values("version", ascending=False).load()

    #
    # Process data.
    #
    # Minimum processing of the data.
    df = prepare_data(table=table)

    # Prepare codebook.
    codebook = prepare_codebook(table=table)

    #
    # Save outputs.
    #
    # Save dataset to files in different formats.
    df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.3f')
    df.to_excel(OUTPUT_EXCEL_FILE, index=False, float_format='%.3f')
    df_to_json(df, OUTPUT_JSON_FILE, ["iso_code"])

    # Save codebook file.
    codebook.to_csv(CODEBOOK_FILE, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
