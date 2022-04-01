"""Load BP and EIA data on primary energy consumption (and energy per GDP, taken from the Maddison Project Database),
combine data, add variables, and export as a csv file.

"""

import argparse
import os

import numpy as np
import pandas as pd
from owid import catalog

from scripts import GRAPHER_DIR, INPUT_DIR
from utils import add_population_to_dataframe, standardize_countries

# Input data files.
# BP data file.
BP_DATA_FILE = os.path.join(
    INPUT_DIR, "shared", "statistical_review_of_world_energy_bp_2021.csv"
)
BP_COUNTRIES_FILE = os.path.join(
    INPUT_DIR, "shared", "statistical_review_of_world_energy_bp_2021.countries.json"
)
# EIA data file, manually downloaded from
# https://www.eia.gov/international/data/world/total-energy/more-total-energy-data
# after selecting all countries, only "Consumption" in activities, "MTOE" in units, and downloaded as CSV (table).
EIA_DATA_FILE = os.path.join(
    INPUT_DIR, "energy-consumption", "eia_primary_energy_consumption.csv"
)
# EIA countries file, mapping EIA country names to OWID country names, generated using the etl.harmonize tool.
EIA_COUNTRIES_FILE = os.path.join(INPUT_DIR, "energy-consumption", "eia.countries.json")
# Output file of combined BP & EIA data.
OUTPUT_FILE = os.path.join(
    GRAPHER_DIR, "Primary energy consumption BP & EIA (2022).csv"
)

# Conversion factors.
# Million tonnes of oil equivalent to terawatt-hours.
MTOE_TO_TWH = 11.63
# Terawatt-hours to kilowatt-hours.
TWH_TO_KWH = 1e9

# After analysing the resulting time series, we detected several issues with the following countries/regions.
# For the moment, we remove them from the final dataset.
REGIONS_WITH_INCONSISTENT_DATA = [
    "Gibraltar",
    # Remove North America, Central America and South & Central America, since their definitions in BP are different
    # from the definition in OWID population dataset.
    "North America",
    "Central America",
    "South & Central America",
]


def load_maddison_data():
    """Load Maddison Project Database data on GDP.

    Returns
    -------
    gdp : pd.DataFrame
        Data on GDP.

    """
    columns = {
        "country": "Country",
        "year": "Year",
        "gdp": "GDP",
    }
    dtypes = {
        "Country": str,
        "Year": int,
        "GDP": float,
    }
    gdp = catalog.find(
        table="maddison", dataset="ggdc_maddison", namespace="ggdc"
    ).load()
    gdp = gdp.reset_index().rename(columns=columns)[columns.values()]

    gdp = gdp.astype(dtypes)

    return gdp


def load_eia_data():
    """Load EIA data on primary energy consumption.

    Returns
    -------
    eia_data : pd.DataFrame
        EIA data.

    """
    eia_data = (
        pd.read_csv(EIA_DATA_FILE, skiprows=1, na_values="--")
        .drop(columns=["API"])
        .rename(columns={"Unnamed: 1": "Country"})
    )
    assert eia_data.iloc[0, 0] == "total energy consumption (MMTOE)"
    eia_data = eia_data.iloc[1:].reset_index(drop=True)

    # Adjust the format.
    col_name = "Primary energy consumption (TWh)"
    eia_data = eia_data.melt(id_vars="Country", var_name="Year", value_name=col_name)
    eia_data["Year"] = eia_data["Year"].astype(int)
    # Convert units.
    eia_data[col_name] = eia_data[col_name] * MTOE_TO_TWH
    # Remove appended spaces on country names.
    eia_data["Country"] = eia_data["Country"].str.lstrip()

    # Standardize country names.
    eia_data = standardize_countries(
        df=eia_data, countries_file=EIA_COUNTRIES_FILE, country_col="Country"
    )

    # Drop rows with missing values and sort rows conveniently.
    eia_data = (
        eia_data.dropna(subset=[col_name])
        .sort_values(["Country", "Year"])
        .reset_index(drop=True)
    )

    return eia_data


def load_bp_data():
    """Load BP data on primary energy consumption.

    Returns
    -------
    bp_data : pd.DataFrame
        BP data.

    """
    columns = {
        "Entity": "Country",
        "Year": "Year",
        "Primary Energy Consumption - TWh": "Primary energy consumption (TWh)",
    }

    # Import total primary energy consumption data from BP.
    bp_data = pd.read_csv(BP_DATA_FILE, usecols=list(columns)).rename(
        errors="raise", columns=columns
    )

    # Standardize country names.
    bp_data = standardize_countries(
        df=bp_data, countries_file=BP_COUNTRIES_FILE, country_col="Country"
    )

    # Drop rows with missing values and sort rows conveniently.
    bp_data = (
        bp_data.dropna(subset="Primary energy consumption (TWh)")
        .sort_values(["Country", "Year"])
        .reset_index(drop=True)
    )

    return bp_data


def remove_infinity_values(df):
    """Replace any possible spurious infinity values in the data with nans.

    Parameters
    ----------
    df : pd.DataFrame
        Combined data on primary energy consumption.

    Returns
    -------
    combined : pd.DataFrame
        Input dataframe after replacing inf values with nan values.

    """
    combined = df.copy()
    for column in combined.columns:
        issues_mask = combined[column] == np.inf
        issues = combined[issues_mask]
        if len(issues) > 0:
            print(
                f"WARNING: Removing {len(issues)} infinity values found in '{column}'. Affected countries:"
            )
            for country in set(issues["Country"]):
                print(f"* {country}")
            combined.loc[issues_mask, column] = np.nan

    return combined


def main():
    # Load BP data and EIA data on primary energy consumption.
    bp_data = load_bp_data()
    eia_data = load_eia_data()

    # Combine energy consumption from BP and EIA, and prioritise the former.
    combined = (
        pd.concat([bp_data, eia_data], ignore_index=True)
        .drop_duplicates(subset=["Country", "Year"], keep="first")
        .sort_values(["Country", "Year"])
        .reset_index(drop=True)
    )

    # Calculate annual change.
    combined["Annual change primary energy consumption (%)"] = (
        combined.groupby("Country")["Primary energy consumption (TWh)"].pct_change()
        * 100
    )
    combined["Annual change primary energy consumption (TWh)"] = combined.groupby(
        "Country"
    )["Primary energy consumption (TWh)"].diff()

    # Add population to primary energy dataframe
    combined = add_population_to_dataframe(
        df=combined,
        country_col="Country",
        year_col="Year",
        population_col="Population",
        warn_on_missing_countries=True,
        show_full_warning=True,
    )

    combined["Energy per capita (kWh)"] = (
        combined["Primary energy consumption (TWh)"]
        / combined["Population"]
        * TWH_TO_KWH
    )
    combined = combined.drop(errors="raise", columns=["Population"])

    # Load GDP data and calculate energy consumption per unit GDP.
    gdp = load_maddison_data()

    combined = combined.merge(gdp, on=["Country", "Year"], how="left")
    combined["Energy per GDP (kWh per $)"] = (
        combined["Primary energy consumption (TWh)"] / combined["GDP"] * TWH_TO_KWH
    )
    combined = combined.drop(errors="raise", columns=["GDP"])

    # Remove any possible spurious infinity values (which should be further investigated during sanity checks).
    combined = remove_infinity_values(df=combined)

    # Round all values to 3 decimal places
    rounded_cols = [col for col in list(combined) if col not in ("Country", "Year")]
    combined[rounded_cols] = combined[rounded_cols].round(3)

    ####################################################################################################################
    # TODO: Remove this temporary solution once inconsistencies in data have been tackled.
    print(f"WARNING: Removing countries and regions with inconsistent data:")
    for region in REGIONS_WITH_INCONSISTENT_DATA:
        if region in sorted(set(combined["Country"])):
            print(f" * {region}")
    combined = combined[
        ~combined["Country"].isin(REGIONS_WITH_INCONSISTENT_DATA)
    ].reset_index(drop=True)
    ####################################################################################################################

    # Remove rows without any relevant data and sort conveniently.
    combined = (
        combined.dropna(subset=rounded_cols, how="all")
        .sort_values(["Country", "Year"])
        .reset_index(drop=True)
    )

    # Save data to file.
    combined.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
